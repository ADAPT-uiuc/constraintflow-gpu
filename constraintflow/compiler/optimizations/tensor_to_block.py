import json
import os
import torch 

from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses
from constraintflow.gbcsr.sparse_tensor import get_operator_func


counter = -1
def get_var():
    global counter 
    counter += 1
    return 'ttb_var_' + str(counter)

def get_var_while(name, while_iteration):
    return name + '_iter_' + str(while_iteration)


def _collect_ir_ast_nodes(obj, visited_objects, ast_nodes):
    obj_id = id(obj)
    if obj_id in visited_objects:
        return
    visited_objects.add(obj_id)

    if isinstance(obj, IrAst):
        ast_nodes[obj_id] = obj

    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_ir_ast_nodes(key, visited_objects, ast_nodes)
            _collect_ir_ast_nodes(value, visited_objects, ast_nodes)
        return

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_ir_ast_nodes(item, visited_objects, ast_nodes)
        return

    if hasattr(obj, '__dict__'):
        for value in vars(obj).values():
            _collect_ir_ast_nodes(value, visited_objects, ast_nodes)


def deepcopy_cfg_with_fresh_identifiers(cfg):
    cfg_copy = copy.deepcopy(cfg)

    visited_objects = set()
    ast_nodes = {}
    _collect_ir_ast_nodes(cfg_copy, visited_objects, ast_nodes)

    # Preserve relative order by old identifiers while assigning new global ids.
    ordered_nodes = sorted(ast_nodes.values(), key=lambda node: node.identifier)

    next_identifier = IrAst.counter + 1
    for node in ordered_nodes:
        node.identifier = next_identifier
        next_identifier += 1

    IrAst.counter = next_identifier - 1
    return cfg_copy


def get_profiled_branch(layer_index, block_id):
    if block_id is None:
        return None

    filename = f"jit_branch/branch_{layer_index}_{block_id}.json"
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        json_obj = json.load(f)

    taken = json_obj["taken"]
    if taken in ("then", "else"):
        return taken
    return None


# Worklist Algorithm to find live nodes (essentially branches that are taken based on the profiling data that we collect during simulacrum mode)
def get_live_nodes(cfg, layer_index):
    live_nodes = set()
    worklist = [cfg.entry_node]

    while worklist:
        node = worklist.pop()
        if node in live_nodes or node not in cfg.ir:
            continue

        live_nodes.add(node)
        block = cfg.ir[node]
        taken = None
        if block.inner_jump is not None and len(block.inner_jump) == 3:
            taken = get_profiled_branch(layer_index, block.block_id)

        if taken == "then":
            successors = [cfg.get_block_id(block.inner_jump[1])]
        elif taken == "else":
            successors = [cfg.get_block_id(block.inner_jump[2])]
        else:
            successors = cfg.successors[node]

        for successor in successors:
            if successor not in live_nodes:
                worklist.append(successor)

    return live_nodes


def convert_to_ir_ttb(expr, layer_index, while_iteration):
    targets = (IrBinaryOp, IrMult, IrInnerProduct, IrRepeat, IrClamp, IrDot, IrTernary, IrUnaryOp, IrGetDefaultStop, IrGetPriorityLList, IrGetPolyexpStop, IrGetPolyexpNotStop, IrAddDimension, IrRemoveDimension)
    if not isinstance(expr, targets):
        return expr, []

    if isinstance(expr, IrUnaryOp) and expr.op in ('get_shape_1', 'get_shape_0'):
        return expr, []
    binary_instance = expr.ttb_counter
    
    
    
    if isinstance(expr, IrUnaryOp):
        if expr.op == 'any':
            filename = f"jit_any/any_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        elif expr.op == 'all':
            filename = f"jit_all/all_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        elif expr.op == 'get_dims':
            filename = f"jit_get_dims/get_dims_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        else:
            filename = f"jit_unary/unary_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrMult) or isinstance(expr, IrBinaryOp):
        filename = f"jit_binary/binary_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrInnerProduct):
        filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrRepeat):
        filename = f"jit_repeat/repeat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrClamp):
        filename = f"jit_clamp/clamp_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrAddDimension):
        filename = f"jit_unsqueeze/unsqueeze_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrRemoveDimension):
        filename = f"jit_squeeze/squeeze_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrTernary):
        filename = f"jit_where/where_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrDot):
        [lhsIr, rhsIr] = expr.children
        if lhsIr.irMetadata[-1].type != 'Float':
            return expr, []
        filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetDefaultStop):
        filename = f"jit_defaultstop/stop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetPriorityLList):
        filename = f"jit_priority/priority_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    
    elif isinstance(expr, IrGetPolyexpStop):
        filename = f"jit_polyexp_stop/stop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetPolyexpNotStop):
        filename = f"jit_polyexp_not_stop/notstop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    
    with open(filename, 'r') as f:
        json_list = json.load(f)
    if isinstance(expr, IrTernary):
        cond = expr.children[0]
        lhs = expr.children[1]
        rhs = expr.children[2]
    elif isinstance(expr, IrUnaryOp):
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, IrGetDefaultStop):
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrGetPriorityLList):
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrGetPolyexpStop):
        cond = None
        lhs_input = expr.children[0]
        rhs_input = expr.children[1]
        lhs = IrPolyExpMat(lhs_input)
        rhs = IrConvertBoolToFloat(rhs_input)
    elif isinstance(expr, IrGetPolyexpNotStop):
        cond = None
        lhs_input = expr.children[0]
        rhs_input = expr.children[1]
        lhs = IrPolyExpMat(lhs_input)
        rhs = IrPolyExpNotStopFloat(rhs_input)
    elif isinstance(expr, IrAddDimension) or isinstance(expr, IrRemoveDimension):
        cond = None
        lhs = expr.children[0]
        rhs = None
    else:
        cond = None
        lhs = expr.children[0]
        rhs = expr.children[1]
    if isinstance(rhs, IrAst):
        irMetadata = rhs.irMetadata
    elif isinstance(lhs, IrAst):
        irMetadata = lhs.irMetadata
    else:
        irMetadata = None
    new_assignments = []
    output_vars = []

    for json_obj in json_list:
        new_name = get_var()
        new_var = IrVar(new_name, irMetadata)
        if json_obj["method"] == "noop":
            if json_obj["input"] == "cond":
                output = cond
            elif json_obj["input"] == "rhs":
                output = rhs
            elif json_obj["input"] == "lhs":
                output = lhs
            elif 'json_list_' in json_obj["input"]:
                output = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                # print(json_obj["input"])
                raise Exception("NOT IMPLEMENTED")
        
        elif json_obj["method"] == "SparseTensor":
            if "json_list_" in json_obj['blocks']:
                blocksIr = output_vars[int(json_obj['blocks'].split("_")[-1])]
            elif json_obj['blocks'] == []:
                blocksIr = []
            else:
                raise Exception("NOT IMPLEMENTED")

            args = [
                [torch.tensor(json_obj["start_indices"][i], dtype=torch.int64) for i in range(len(json_obj["start_indices"]))],
                blocksIr,
                json_obj["dims"],
                torch.tensor(json_obj["total_size"], dtype=torch.int64),
            ]

            kwargs = {}

            if "end_indices" in json_obj and json_obj["end_indices"] is not None:
                kwargs["end_indices"] = [torch.tensor(json_obj["end_indices"][i], dtype=torch.int64) for i in range(len(json_obj["end_indices"]))]

            if "type" in json_obj and json_obj["type"] is not None:
                kwargs["type"] = getattr(builtins, json_obj["type"])

            if "dense_const" in json_obj and json_obj["dense_const"] is not None:
                kwargs["dense_const"] = json_obj["dense_const"]

            output = IrSparseTensor(*args, **kwargs)
        
        elif json_obj["method"] == "initialise":
            output = json_obj["value"]
            if output == "[]":
                output = IrEmptyList()
            elif isinstance(output, bool):
                output = IrConst(output, 'Bool')
            else:
                raise Exception("NOT IMPLEMENTED")

        elif json_obj["method"] == "bool_value":
            output = IrConst(json_obj["value"], 'Bool')

        elif json_obj["method"] == "get_sub_block_custom_range":
            if "json_list_" in json_obj["lhs"]:
                inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                inputIr = lhs
            elif json_obj["lhs"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSubBlockCustomRange(inputIr, torch.tensor(json_obj["start_index"], dtype=torch.int64), torch.tensor(json_obj["end_index"], dtype=torch.int64), json_obj["block_id"], json_obj["tensor"])

        elif json_obj["method"] == "unary_block":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockUnaryOp(inputIr, json_obj["op"])

        elif json_obj["method"] == "simple_unary":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrSimpleUnary(inputIr, json_obj["op"])

        elif json_obj["method"] == "binary_to_identity_unary":
            if "json_list_" in json_obj["unary_source"]:
                inputIr = output_vars[int(json_obj["unary_source"].split("_")[-1])]
            elif json_obj["unary_source"] == "lhs":
                inputIr = lhs
            elif json_obj["unary_source"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            op = json_obj["op"]
            output = IrBinaryToUnary(inputIr, op)
            

        elif json_obj["method"] == "append_list":
            if 'json_list_' in json_obj["list"]:
                list1 = output_vars[int(json_obj["list"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if 'json_list_' in json_obj["value"]:
                val = output_vars[int(json_obj["value"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrAppendList(list1, val)

        elif json_obj["method"] == "index_lookup":
            if "json_list_" in json_obj["input"]:
                listIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            if isinstance(json_obj["index"], int):
                indexIr = json_obj["index"]
            elif "json_list_" in json_obj["index"]:
                indexIr = output_vars[int(json_obj["index"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            output = IrListExtract(listIr, indexIr)

        elif json_obj["method"] == "extract_block":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            if isinstance(json_obj["index"], int):
                indexIr = json_obj["index"]
            elif "json_list_" in json_obj["index"]:
                indexIr = output_vars[int(json_obj["index"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockExtract(inputIr, indexIr)
            
        elif json_obj["method"] == "binary":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            output = IrBlockBinaryOp(lhsIr, rhsIr, get_operator_func(json_obj["op"]))

        elif json_obj["method"] == "matmul_unequal_dims":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            output = IrBlockInnerProduct(lhsIr, rhsIr, type='unequal_dims')

        elif json_obj["method"] == "matmul_equal_dims":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            output = IrBlockInnerProduct(lhsIr, rhsIr, type='equal_dims')
        
        elif json_obj["method"] == "ConstBlock":
            shape = torch.tensor(json_obj["total_shape"], dtype=torch.int64)
            output = IrConstBlock(json_obj["block"], shape)
        
        elif json_obj["method"] == "repeat":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            repeat_dims = torch.tensor(json_obj["repeat_dims"], dtype=torch.int64)
            output = IrBlockRepeat(inputIr, repeat_dims)

        elif json_obj["method"] == "block_clamp":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockClamp(inputIr, json_obj["const"], json_obj["min_true"])

        elif json_obj["method"] == "block_squeeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockSqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "block_unsqueeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockUnsqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "tensor_ones":
            output = IrTensorOnes(torch.tensor(json_obj["repeat_dims"], dtype=torch.int64))

        elif json_obj["method"] == "tensor_repeat":
            if "json_list_" in json_obj["lhs"]:
                inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                inputIr = lhs
            elif json_obj["lhs"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            repeat_dims = torch.tensor(json_obj["repeat_dims"], dtype=torch.int64)
            output = IrTensorRepeat(inputIr, repeat_dims)

        elif json_obj["method"] == "tensor_clamp":
            if "json_list_" in json_obj["lhs"]:
                inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                inputIr = lhs
            elif json_obj["lhs"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTensorClamp(inputIr, json_obj['const'], json_obj['min_true'])

        elif json_obj["method"] == "simple_binary":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrSimpleBinary(lhsIr, rhsIr, json_obj['op'])
        
        elif json_obj["method"] == "sp_where_block":
            condIr = output_vars[int(json_obj["x"].split("_")[-1])]
            lhsIr  = output_vars[int(json_obj["y"].split("_")[-1])]
            rhsIr  = output_vars[int(json_obj["z"].split("_")[-1])]
            output = IrBlockWhereBlock(condIr, lhsIr, rhsIr)

        elif json_obj["method"] == "scalar_const":
            output = IrConst(json_obj["value"], 'Float')

        elif json_obj["method"] == "create_similar":
            matIr = output_vars[int(json_obj["mat"].split("_")[-1])]
            if "const" in json_obj:
                constIr = output_vars[int(json_obj["const"].split("_")[-1])]
                output = IrBlockPolyexpNotStop(expr.children[0], matIr, constIr)
            else:
                output = IrBlockPolyexpStop(expr.children[0], matIr)

        elif json_obj["method"] == "any":
            output = IrBlockAny(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "any_torch":
            output = IrBlockAny(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "all":
            output = IrBlockAll(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "all_torch":
            output = IrBlockAll(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "get_dims":
            output = IrBlockGetDims(output_vars[int(json_obj["input"].split("_")[-1])])
        
        elif json_obj["method"] == "object_lookup":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrObjectLookup(inputIr, json_obj["object"])

        elif json_obj["method"] == "block_create_similar":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["arg"]:
                argIr = output_vars[int(json_obj["arg"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrBlockCreateSimilar(inputIr, argIr)

        elif json_obj["method"] == "block_set_total_shape_last_dim":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            if isinstance(json_obj["value"], int):
                valueIr = json_obj["value"]
            elif "json_list_" in json_obj["value"]:
                valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            new_assignment = IrAssignment(new_var, inputIr)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            new_assignments.append(IrSetBlockTotalShapeLastDim(new_var, valueIr))

            if len(output_vars) != json_obj["output"] + 1:
                print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
            assert(len(output_vars) == json_obj["output"] + 1)
            continue

        elif json_obj["method"] == "get_sub_block_custom_range_block":
            if "json_list_" in json_obj["lhs"]:
                inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrBlockGetSubBlockCustomRange(
                inputIr,
                torch.tensor(json_obj["start_index"], dtype=torch.int64),
                torch.tensor(json_obj["end_index"], dtype=torch.int64),
                torch.tensor(json_obj["block_start_index"], dtype=torch.int64),
            )
        else:
            raise Exception(f"Unknown method {json_obj['method']} in replay")
        
        new_assignment = IrAssignment(new_var, output)
        new_assignments.append(new_assignment)
        output_vars.append(new_var)

        if(len(output_vars) != json_obj["output"] + 1):
            print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
        assert(len(output_vars) == json_obj["output"] + 1)

    return output_vars[-1], new_assignments



# For the del statements
def collect_ir_var_names(expr, names=None):
    if names is None:
        names = set()
    if isinstance(expr, IrVar):
        names.add(expr.name)
    elif isinstance(expr, int):
        pass
    elif hasattr(expr, "children"):
        for child in expr.children:
            collect_ir_var_names(child, names)
    return names


def make_ttb_del_statement(new_assignments, keep_var_names=None):
    if keep_var_names is None:
        keep_var_names = set()
    var_names = []
    for assignment in new_assignments:
        if not isinstance(assignment, IrAssignment):
            continue
        lhs = assignment.children[0]
        if not isinstance(lhs, IrVar):
            continue
        if lhs.name.startswith("ttb_var_") and lhs.name not in keep_var_names:
            var_names.append(lhs.name)
    if not var_names:
        return None
    return IrDel(var_names)


def tensor_to_block_block(block, layer_index, ir_list = None, while_iteration=None):
    if ir_list is None:
        ir_list = block.children
    length = len(ir_list)
    index = 0
    for i in range(length):
        l = ir_list[index]
        if isinstance(l, IrAssignment):
            new_expr, new_assignments = convert_to_ir_ttb(l.children[1], layer_index, while_iteration)
            new_children = [l.children[0], new_expr]
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
            # Create the IR node for the del statements
            del_stmt = make_ttb_del_statement(new_assignments)
            if del_stmt is not None:
                ir_list.insert(index + 1, del_stmt)
                index += 1
        elif isinstance(l, IrTransRetBasic):
            new_children = []
            new_assignments = []
            for child in l.children:
                new_expr, new_assignments_inner = convert_to_ir_ttb(child, layer_index, while_iteration)
                new_children.append(new_expr)
                new_assignments += new_assignments_inner
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
            
            # We want to collect the var names that are used in the return expression, so that we can 
            # exclude them from the del statements
            keep_var_names = set()
            for child in new_children:
                keep_var_names |= collect_ir_var_names(child)
            del_stmt = make_ttb_del_statement(new_assignments, keep_var_names=keep_var_names)
            if del_stmt is not None:
                ir_list.insert(index, del_stmt)
                index += 1
        index += 1


def replace_all_occurrences_expr(expr, var_map):
    if isinstance(expr, IrVar) and expr.name in var_map.keys(): 
        return var_map[expr.name]
    if isinstance(expr, (int, float, list)):
        return expr
    for i in range(len(expr.children)):
        new_child = replace_all_occurrences_expr(expr.children[i], var_map)
        expr.children[i] = new_child
    return expr

def remove_while(layer_index, num_iterations, cfg, root_node, first_while_node, second_while_node, exit_node, break_node):
    root_block = cfg.ir[root_node]
    first_while_block = cfg.ir[first_while_node]
    second_while_block = cfg.ir[second_while_node]
    exit_block = cfg.ir[exit_node]

    tensor_to_block_block(root_block, layer_index)
    tensor_to_block_block(exit_block, layer_index)


    ir_list = root_block.children
    for i in range(num_iterations):
        combined_list = copy.deepcopy(first_while_block.children + second_while_block.children)
        tensor_to_block_block(None, layer_index=layer_index, ir_list=combined_list, while_iteration=i)
        ir_list += combined_list

    ir_list += exit_block.children

    root_block.update_parent_child(ir_list)

    predecessors_exit = cfg.predecessors[exit_node]
    for pred in predecessors_exit:
        cfg.successors[pred].remove(exit_node)
        if root_node not in cfg.successors[pred]:
            cfg.successors[pred].append(root_node)
    cfg.successors[root_node] = cfg.successors[exit_node]
    root_block.jump = exit_block.jump
    root_block.inner_jump = exit_block.inner_jump


    del cfg.ir[first_while_node]
    del cfg.ir[second_while_node]
    del cfg.ir[break_node]
    del cfg.ir[exit_node]

    del cfg.successors[first_while_node]
    del cfg.successors[second_while_node]
    del cfg.successors[break_node]
    del cfg.successors[exit_node]    

    del cfg.predecessors[first_while_node]
    del cfg.predecessors[second_while_node]
    del cfg.predecessors[break_node]
    del cfg.predecessors[exit_node]

    cfg.nodes.remove(first_while_node)
    cfg.nodes.remove(second_while_node)
    cfg.nodes.remove(break_node)
    cfg.nodes.remove(exit_node)

def unroll_while(cfg, layer_index):
    i = 0
    while True:
        live_nodes = get_live_nodes(cfg, layer_index)
        ordered_live_nodes = [node for node in cfg.nodes if node in live_nodes]
        if i >= len(ordered_live_nodes):
            break

        node = ordered_live_nodes[i]
        
        block = cfg.ir[node]
        if block.inner_jump is None:
            tensor_to_block_block(block, layer_index)
            i+=1
        elif len(block.inner_jump) == 3:
            tensor_to_block_block(block, layer_index)
            i+=1
        elif not isinstance(block.inner_jump[1], IrWhileBlock):
            tensor_to_block_block(block, layer_index)
            i+=1
        else:
            root_node = node
            first_while_node = cfg.successors[node][0]
            first_while_block = cfg.ir[first_while_node]
            second_while_block = first_while_block.jump[1]
            second_while_node = cfg.get_block_id(second_while_block)
            break_node = cfg.get_block_id(first_while_block.inner_jump[1])
            exit_node = cfg.get_block_id(block.jump[1])
            while_number = first_while_block.while_number
            filename = f"jit_while/while_iterations_layer_{layer_index}_while_{while_number}.json"
            with open(filename, 'r') as f:
                json_obj = json.load(f)
                num_iterations = json_obj["num_iterations"]
            remove_while(layer_index, num_iterations, cfg, root_node, first_while_node, second_while_node, exit_node, break_node)


def tensor_to_block_cfg(cfg, layer_index):
    live_nodes = get_live_nodes(cfg, layer_index)
    for node in cfg.nodes:
        if node not in live_nodes:
            continue
        block = cfg.ir[node]
        tensor_to_block_block(block, layer_index)

def tensor_to_block(ir):
    # TODO: DEBUG THE FOLLOWING LINE
    # uses.populate_uses_defs(ir)
    filename = f"jit_layers/layers.json"
    with open(filename, 'r') as f:
        json_obj = json.load(f)
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformerIr = ir.tstore[transformer][i]
            if transformerIr.op == 'Affine':
                layer_indices = json_obj['affine']
            elif transformerIr.op == 'Relu':
                layer_indices = json_obj['relu']
            else:
                print("NOT IMPLEMENTED for op " + transformerIr.op)
                continue
            
            new_cfgs = {}
            for j, layer_index in enumerate(layer_indices):
                cfg = deepcopy_cfg_with_fresh_identifiers(transformerIr.cfg)
                unroll_while(cfg, layer_index)
                new_cfgs[layer_index] = cfg
            
            transformerIr.layerwise_cfgs = new_cfgs
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
    # targets = ()
    # targets = (IrUnaryOp)
    # targets= (IrBinaryOp, IrMult, IrInnerProduct, IrRepeat, IrClamp, IrDot, IrTernary, IrUnaryOp, IrGetDefaultStop)
    
    targets = (
        IrBinaryOp, IrMult, IrInnerProduct, IrRepeat, IrClamp,
        IrDot, IrTernary, IrUnaryOp, IrGetDefaultStop,
        IrGetPriorityLList, IrGetPolyexpStop, IrGetPolyexpNotStop,
        IrAddDimension, IrRemoveDimension, IrAccess,
        IrExtractPolyCoeff, IrExtractSymCoeff, IrMapCoeff,
        IrReduce, IrEpsilon
        # IrGetAbsElemSparseDKey, # IrGetPolyExpSparseConst,
        # IrGetPolyExpSparseMat
    )
    if not isinstance(expr, targets):
        return expr, []

    if isinstance(expr, IrUnaryOp) and expr.op in ('get_shape_1', 'get_shape_0'):
        return expr, []
    binary_instance = expr.ttb_counter

    if while_iteration is None and getattr(expr, "inside_while", False):
        while_iteration = -1
    
    
    
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
        if lhsIr.irMetadata[-1].type == 'Float':
            filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        elif lhsIr.irMetadata[-1].type == 'Neuron' or rhsIr.irMetadata[-1].type == 'Neuron':
            filename = f"jit_Llist_dot/Llist_dot_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        else:
            return expr, []
    elif isinstance(expr, IrGetDefaultStop):
        filename = f"jit_defaultstop/stop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetPriorityLList):
        filename = f"jit_priority/priority_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    
    elif isinstance(expr, IrGetPolyexpStop):
        filename = f"jit_polyexp_stop/stop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetPolyexpNotStop):
        filename = f"jit_polyexp_not_stop/notstop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrReduce):
        filename = f"jit_sum/sum_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrEpsilon):
        filename = f"jit_new_eps/new_eps_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"

    elif isinstance(expr, IrAccess) and (not expr.isMetadata):
        filename = f'jit_Abs_elem_sparse_get_elem/Abs_elem_sparse_get_elem_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrAccess) and expr.isMetadata:
        filename = f'jit_llist_get_metadata/llist_get_metadata_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
        # print(filename)
    elif isinstance(expr, IrExtractPolyCoeff):
        filename = f'jit_poly_exp_sparse_get_mat/poly_exp_sparse_get_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrExtractSymCoeff):
        filename = f'jit_poly_exp_sparse_get_mat/poly_exp_sparse_get_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrMapCoeff):
        filename = f'jit_poly_exp_sparse_get_mat/poly_exp_sparse_get_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    
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
    elif isinstance(expr, IrEpsilon):
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
    elif isinstance(expr, IrAddDimension) or isinstance(expr, IrRemoveDimension) or isinstance(expr, IrReduce):
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, IrAccess) and (not expr.isMetadata):
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, IrAccess) and expr.isMetadata:
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, (IrExtractPolyCoeff, IrExtractSymCoeff, IrMapCoeff)):
        cond = None
        lhs = expr.children[0]
        rhs = None
    else:
        cond = None
        lhs = expr.children[0]
        rhs = expr.children[1]
    if isinstance(expr, IrDot):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrAccess):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrEpsilon):
        irMetadata = expr.irMetadata
    elif isinstance(rhs, IrAst):
        irMetadata = rhs.irMetadata
    elif isinstance(lhs, IrAst):
        irMetadata = lhs.irMetadata
    # elif isinstance(expr, (IrExtractPolyCoeff, IrExtractSymCoeff, IrMapCoeff)):
    #     irMetadata = expr.irMetadata
    else:
        irMetadata = None
    new_assignments = []
    output_vars = []

    for json_index, json_obj in enumerate(json_list):
        new_name = get_var()
        new_var = IrVar(new_name, copy_metadata(irMetadata) if irMetadata is not None else None)
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
                # print(len(output_vars), int(json_obj['blocks'].split("_")[-1]))
                # if 'debug_pos' in json_obj:
                #     print(json_obj['debug_pos'])
                blocksIr = output_vars[int(json_obj['blocks'].split("_")[-1])]
            elif json_obj['blocks'] == []:
                blocksIr = []
            else:
                raise Exception("NOT IMPLEMENTED")
            total_size = json_obj["total_size"]
            if isinstance(total_size, list) and len(total_size) > 0 and total_size[0] == 1:
                total_sizeIr = "torch.tensor([batch_size" + "".join([", " + str(total_size[i]) for i in range(1, len(total_size))]) + "], dtype=torch.int64)"
            else:
                total_sizeIr = torch.tensor(total_size, dtype=torch.int64)

            args = [
                [torch.tensor(json_obj["start_indices"][i], dtype=torch.int64) for i in range(len(json_obj["start_indices"]))],
                blocksIr,
                json_obj["dims"],
                total_sizeIr,
            ]

            kwargs = {}

            if "end_indices" in json_obj and json_obj["end_indices"] is not None:
                end_indices = []
                for end_index in json_obj["end_indices"]:
                    if isinstance(end_index, list) and len(end_index) > 0 and end_index[0] == 1:
                        end_indices.append("torch.tensor([batch_size" + "".join([", " + str(end_index[i]) for i in range(1, len(end_index))]) + "], dtype=torch.int64)")
                    else:
                        end_indices.append(torch.tensor(end_index, dtype=torch.int64))
                kwargs["end_indices"] = end_indices

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
                raise Exception(f"NOT IMPLEMENTED output: {output}")

        elif json_obj["method"] == "bool_value":
            output = IrConst(json_obj["value"], 'Bool')

        # elif json_obj["method"] == "get_sub_block_custom_range":
        #     if "json_list_" in json_obj["lhs"]:
        #         inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
        #     elif json_obj["lhs"] == "lhs":
        #         inputIr = lhs
        #     elif json_obj["lhs"] == "rhs":
        #         inputIr = rhs
        #     else:
        #         raise Exception("NOT IMPLEMENTED")
        #     output = IrGetSubBlockCustomRange(inputIr, torch.tensor(json_obj["start_index"], dtype=torch.int64), torch.tensor(json_obj["end_index"], dtype=torch.int64), json_obj["block_id"], json_obj["tensor"])

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
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            op_ref = json_obj.get("op", json_obj.get("operation"))
            if isinstance(op_ref, str) and "json_list_" in op_ref:
                op_idx = int(op_ref.split("_")[-1])
                # Inline lambdas directly rather than referencing their ttb_var;
                # see the apply_lambda note below for why the reference would be
                # dropped by subexp_inlining.
                if json_list[op_idx]["method"] == "lambda":
                    op_ref = IrLambda(json_list[op_idx]["op"])
                else:
                    op_ref = output_vars[op_idx]
            output = IrSimpleUnary(inputIr, op_ref)

        elif json_obj["method"] == "apply_lambda":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if not (isinstance(json_obj["op"], str) and "json_list_" in json_obj["op"]):
                raise Exception("NOT IMPLEMENTED")
            lambda_obj = json_list[int(json_obj["op"].split("_")[-1])]
            if lambda_obj["method"] != "lambda":
                raise Exception("NOT IMPLEMENTED")
            # Inline the lambda directly as the op instead of referencing the
            # ttb_var it was assigned to. IrSimpleUnary stores op outside of
            # .children, so a variable reference here is invisible to the
            # .children-based def/use analysis in subexp_inlining, which would
            # then delete the lambda's definition while codegen still emits the
            # reference (NameError). An inline IrLambda has no separate def.
            output = IrSimpleUnary(inputIr, IrLambda(lambda_obj["op"]))

        elif json_obj["method"] == "torch_sigmoid":
            input_ref = json_obj.get("block", json_obj.get("input"))
            if "json_list_" in input_ref:
                inputIr = output_vars[int(input_ref.split("_")[-1])]
            elif input_ref == "lhs":
                inputIr = lhs
            elif input_ref == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrSimpleUnary(inputIr, "sigma")

        elif json_obj["method"] == "lambda":
            output = IrLambda(json_obj["op"])

            
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
            if "index" in json_obj:
                if isinstance(json_obj["index"], int):
                    indexIr = json_obj["index"]
                elif "json_list_" in json_obj["index"]:
                    indexIr = output_vars[int(json_obj["index"].split("_")[-1])]
                else:                
                    raise Exception("NOT IMPLEMENTED")
            elif "block_id" in json_obj:
                indexIr = json_obj["block_id"]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockExtract(inputIr, indexIr)

        elif json_obj["method"] == "block_copy":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockCopy(inputIr)
            
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

        elif json_obj["method"] == "torch_binary":
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

            output = IrSimpleBinary(lhsIr, rhsIr, json_obj["op"])

        elif json_obj["method"] == "torch_where":
            condIr = output_vars[int(json_obj["cond"].split("_")[-1])]
            lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            output = IrTorchWhere(condIr, lhsIr, rhsIr)

        elif json_obj["method"] == "DenseBlock":
            ref = json_obj["input"] if "input" in json_obj else json_obj["block"]
            if isinstance(ref, int) and ref < len(output_vars):
                inputIr = output_vars[ref]
            elif isinstance(ref, int):
                inputIr = IrGetKthLayerNetworkParam(ref, "weight")
            elif "json_list_" in str(ref):
                inputIr = output_vars[int(str(ref).split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED DenseBlock ref")
            output = IrDenseBlock(inputIr)

        elif json_obj["method"] == "DiagonalBlock":
            if "json_list_" not in json_obj["block"]:
                raise Exception("NOT IMPLEMENTED DiagonalBlock block ref")
            blockIr = output_vars[int(json_obj["block"].split("_")[-1])]
            if isinstance(json_obj["total_shape"], list) and len(json_obj["total_shape"]) > 0 and json_obj["total_shape"][0] == 1:
                shape = "torch.tensor([batch_size" + "".join([", " + str(json_obj["total_shape"][i]) for i in range(1, len(json_obj["total_shape"]))]) + "], dtype=torch.int64)"
            else:
                shape = torch.tensor(json_obj["total_shape"], dtype=torch.int64)
            output = IrDiagonalBlock(blockIr, shape, json_obj["diag_index"])

        elif json_obj["method"] == "PatchesBlock":
            if "json_list_" not in json_obj["block"]:
                raise Exception("NOT IMPLEMENTED PatchesBlock block ref")
            blockIr = output_vars[int(json_obj["block"].split("_")[-1])]
            if isinstance(json_obj["total_shape"], list) and len(json_obj["total_shape"]) > 0 and json_obj["total_shape"][0] == 1:
                shape = "torch.tensor([batch_size" + "".join([", " + str(json_obj["total_shape"][i]) for i in range(1, len(json_obj["total_shape"]))]) + "], dtype=torch.int64)"
            else:
                shape = torch.tensor(json_obj["total_shape"], dtype=torch.int64)
            output = IrPatchesBlock(
                blockIr,
                shape,
                json_obj["ix"],
                json_obj["iy"],
                json_obj["ox"],
                json_obj["oy"],
                json_obj["sx"],
                json_obj["sy"],
                json_obj["px"],
                json_obj["py"],
                json_obj["kx"],
                json_obj["ky"],
                json_obj["num_channels"],
                json_obj["num_kernels"],
            )

        elif json_obj["method"] == "KernelBlock":
            if isinstance(json_obj["block"], int):
                blockIr = output_vars[json_obj["block"]]
            elif "json_list_" in str(json_obj["block"]):
                blockIr = output_vars[int(str(json_obj["block"]).split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED KernelBlock block ref")
            if isinstance(json_obj["total_shape"], list) and len(json_obj["total_shape"]) > 0 and json_obj["total_shape"][0] == 1:
                shape = "[batch_size" + "".join([", " + str(json_obj["total_shape"][i]) for i in range(1, len(json_obj["total_shape"]))]) + "]"
            else:
                shape = json_obj["total_shape"]
            output = IrKernelBlock(
                blockIr,
                shape,
                json_obj["ix"],
                json_obj["iy"],
                json_obj["ox"],
                json_obj["oy"],
                json_obj["sx"],
                json_obj["sy"],
                json_obj["px"],
                json_obj["py"],
            )

        elif json_obj["method"] == "RepeatBlock":
            if "json_list_" not in json_obj["block"]:
                raise Exception("NOT IMPLEMENTED RepeatBlock block ref")
            blockIr = output_vars[int(json_obj["block"].split("_")[-1])]
            if isinstance(json_obj["total_shape"], list) and len(json_obj["total_shape"]) > 0 and json_obj["total_shape"][0] == 1:
                shape = "torch.tensor([batch_size" + "".join([", " + str(json_obj["total_shape"][i]) for i in range(1, len(json_obj["total_shape"]))]) + "], dtype=torch.int64)"
            else:
                shape = torch.tensor(json_obj["total_shape"], dtype=torch.int64)
            output = IrRepeatBlock(blockIr, shape)

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
            if isinstance(json_obj["total_shape"], list) and len(json_obj["total_shape"]) > 0 and json_obj["total_shape"][0] == 1:
                shape = "torch.tensor([batch_size" + "".join([", " + str(json_obj["total_shape"][i]) for i in range(1, len(json_obj["total_shape"]))]) + "], dtype=torch.int64)"
            else:
                shape = torch.tensor(json_obj["total_shape"], dtype=torch.int64)
            if isinstance(json_obj["block"], str) and "json_list_" in json_obj["block"]:
                blockIr = output_vars[int(json_obj["block"].split("_")[-1])]
                output = IrConstBlock(blockIr, shape)
            else:
                blockIr = json_obj["block"]
                output = IrConstBlock(blockIr, shape)
        
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
            if isinstance(json_obj["repeat_dims"], list) and len(json_obj["repeat_dims"]) > 0 and json_obj["repeat_dims"][0] == 1:
                output = IrTensorOnes("[batch_size" + "".join([", " + str(json_obj["repeat_dims"][i]) for i in range(1, len(json_obj["repeat_dims"]))]) + "]")
            else:
                output = IrTensorOnes(torch.tensor(json_obj["repeat_dims"], dtype=torch.int64))

        # torch_ones: sparse_block DummyBlock.get_dense_const_block (size = total_shape).
        elif json_obj["method"] in ("torch_ones", "torch.ones"):
            size = json_obj.get("size")
            if size is None:
                size = json_obj.get("repeat_dims")
            if size is None:
                size = json_obj.get("input")
                if isinstance(size, list) and len(size) == 1 and isinstance(size[0], list):
                    size = size[0]
            if isinstance(size, list) and len(size) > 0 and size[0] == 1:
                output = IrTensorOnes("[batch_size" + "".join([", " + str(size[i]) for i in range(1, len(size))]) + "]")
            else:
                output = IrTensorOnes(torch.tensor(size, dtype=torch.int64))

        elif json_obj["method"] in ("torch_mul", "torch.mul"):
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

            output = IrSimpleBinary(lhsIr, rhsIr, "mul")

        elif json_obj["method"] == "torch_matmul":
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

            output = IrTorchMatmul(lhsIr, rhsIr)

        elif json_obj["method"] == "torch_unsqueeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchUnsqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "torch_squeeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchSqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "torch_reshape":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            shape = json_obj["shape"]
            if isinstance(shape, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    shape = shape.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchReshape(inputIr, shape)

        elif json_obj["method"] == "torch_view":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            shape = json_obj["shape"]
            if isinstance(shape, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    shape = shape.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchView(inputIr, shape)

        elif json_obj["method"] == "torch_repeat":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchRepeat(inputIr, json_obj["repeats"])

        elif json_obj["method"] == "torch_expand":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            shape = json_obj["shape"]
            if isinstance(shape, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    shape = shape.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchExpand(inputIr, shape)

        elif json_obj["method"] == "torch_sum":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchSum(inputIr, json_obj["dim"])

        elif json_obj["method"] == "torch_zeros":
            size = json_obj["size"]
            if isinstance(size, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    size = size.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            device = json_obj.get("device", None)
            if isinstance(device, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    device = device.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            dtype = json_obj.get("dtype", None)
            if isinstance(dtype, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    dtype = dtype.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchZeros(size, device=device, dtype=dtype)

        elif json_obj["method"] == "torch_eye":
            size = json_obj["size"]
            if isinstance(size, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    size = size.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            device = json_obj.get("device", None)
            if isinstance(device, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    device = device.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            dtype = json_obj.get("dtype", None)
            if isinstance(dtype, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    dtype = dtype.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchEye(size, device=device, dtype=dtype)

        elif json_obj["method"] == "torch_float":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchFloat(inputIr)

        elif json_obj["method"] == "torch_diag_embed":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchDiagEmbed(inputIr)

        elif json_obj["method"] == "torch_stride":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchStride(inputIr)

        elif json_obj["method"] == "torch_as_strided":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            size = json_obj["size"]
            stride = json_obj["stride"]
            output = IrTorchAsStrided(inputIr, size, stride)

        elif json_obj["method"] == "torch_slice":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            index = json_obj["index"]
            for item_idx in range(len(index)):
                if isinstance(index[item_idx], str):
                    for ref_idx in range(len(output_vars) - 1, -1, -1):
                        index[item_idx] = index[item_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
                elif isinstance(index[item_idx], list):
                    for range_idx in range(len(index[item_idx])):
                        if isinstance(index[item_idx][range_idx], str):
                            for ref_idx in range(len(output_vars) - 1, -1, -1):
                                index[item_idx][range_idx] = index[item_idx][range_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchSlice(inputIr, index)

        elif json_obj["method"] == "F.conv2d":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["weight"]:
                weightIr = output_vars[int(json_obj["weight"].split("_")[-1])]
            elif json_obj["weight"] == "lhs":
                weightIr = lhs
            elif json_obj["weight"] == "rhs":
                weightIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFConv2d(inputIr, weightIr, json_obj["stride"], json_obj["padding"])

        elif json_obj["method"] == "F.conv_transpose2d":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["weight"]:
                weightIr = output_vars[int(json_obj["weight"].split("_")[-1])]
            elif json_obj["weight"] == "lhs":
                weightIr = lhs
            elif json_obj["weight"] == "rhs":
                weightIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFConvTranspose2d(
                inputIr,
                weightIr,
                stride=json_obj["stride"],
                padding=json_obj["padding"],
                output_padding=json_obj["output_padding"],
            )

        elif json_obj["method"] == "F.unfold":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFUnfold(inputIr, json_obj["kernel_size"], json_obj["padding"], json_obj["stride"])

        elif json_obj["method"] == "torch_diagonal":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchDiagonal(inputIr, json_obj["dim1"], json_obj["dim2"])

        elif json_obj["method"] == "torch_permute":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            if "permutation" in json_obj:
                permutation = json_obj["permutation"]
            else:
                permutation = json_obj["perm"]
            output = IrTorchPermute(inputIr, permutation)

        elif json_obj["method"] == "torch_transpose":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            dim0, dim1 = json_obj["dims"]
            output = IrTorchTranspose(inputIr, dim0, dim1)

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

        elif json_obj["method"] == "new_eps":
            matIr   = output_vars[int(json_obj["mat"].split("_")[-1])]
            constIr = output_vars[int(json_obj["const"].split("_")[-1])]
            output = IrNewEps(matIr, constIr)

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

        elif json_obj["method"] in ("sparse_block_extract", "extract_sparse_block"):
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrObjectLookup(inputIr, "block")

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

        # elif json_obj["method"] == "get_sub_block_custom_range_block":
        #     if "json_list_" in json_obj["lhs"]:
        #         inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
        #     else:
        #         raise Exception("NOT IMPLEMENTED")

        #     output = IrBlockGetSubBlockCustomRange(
        #         inputIr,
        #         torch.tensor(json_obj["start_index"], dtype=torch.int64),
        #         torch.tensor(json_obj["end_index"], dtype=torch.int64),
        #         torch.tensor(json_obj["block_start_index"], dtype=torch.int64),
        #     )
        elif json_obj["method"] == "torch_any":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrBlockAny(inputIr)

        elif json_obj["method"] == "const_any":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrSimpleBinary(inputIr, IrConst(True, 'Bool'), 'eq')

        elif json_obj["method"] == "torch_clamp":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrTensorClamp(inputIr, json_obj["const"], json_obj["min_true"])

        elif json_obj["method"] == "assign_to_view":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            if "base" in json_obj and "json_list_" in json_obj["base"]:
                baseIr = output_vars[int(json_obj["base"].split("_")[-1])]
            else:
                baseIr = inputIr

            if "json_list_" in json_obj["value"]:
                valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            elif json_obj["value"] == "lhs":
                valueIr = lhs
            elif json_obj["value"] == "rhs":
                valueIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            index = json_obj["index"]
            for item_idx in range(len(index)):
                if isinstance(index[item_idx], str):
                    for ref_idx in range(len(output_vars) - 1, -1, -1):
                        index[item_idx] = index[item_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
                elif isinstance(index[item_idx], list):
                    for range_idx in range(len(index[item_idx])):
                        if isinstance(index[item_idx][range_idx], str):
                            for ref_idx in range(len(output_vars) - 1, -1, -1):
                                index[item_idx][range_idx] = index[item_idx][range_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)

            new_assignment = IrAssignment(new_var, baseIr)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            new_assignments.append(IrAssignToView(inputIr, index, valueIr))

            if len(output_vars) != json_obj["output"] + 1:
                print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
            assert(len(output_vars) == json_obj["output"] + 1)
            continue

        elif json_obj["method"] == "assign_to_block":
            if "json_list_" in json_obj["assign_to"]:
                assignToIr = output_vars[int(json_obj["assign_to"].split("_")[-1])]
            elif json_obj["assign_to"] == "lhs":
                assignToIr = lhs
            elif json_obj["assign_to"] == "rhs":
                assignToIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["value"]:
                valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            elif json_obj["value"] == "lhs":
                valueIr = lhs
            elif json_obj["value"] == "rhs":
                valueIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            new_assignments.append(IrAssignToBlock(assignToIr, valueIr))
            # The writer appends assign_to_block to json_list, so it occupies an
            # index slot that later "output": len(json_list) values account for.
            # Append a placeholder output_var (bound to the mutated block) to keep
            # output_vars positionally aligned, mirroring assign_to_view above.
            new_assignment = IrAssignment(new_var, assignToIr)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            continue
        elif json_obj["method"] == "PolyExpSparse":
            if "json_list_" in json_obj["mat"]:
                mat_ir = output_vars[int(json_obj["mat"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if "json_list_" in json_obj["const"]:
                const_ir = output_vars[int(json_obj["const"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if irMetadata is None:
                raise Exception("PolyExpSparse replay requires PolyExp irMetadata")
            mat_metadata = copy_metadata(irMetadata)
            mat_metadata[-1].type = 'Float'
            mat_metadata[-1].shape.append(IrAst.poly_size)
            mat_metadata[-1].broadcast.append(1)
            mat_ir.irMetadata = mat_metadata
            const_metadata = copy_metadata(irMetadata)
            const_metadata[-1].type = 'Float'
            const_ir.irMetadata = const_metadata
            output = IrCombineToPoly(mat_ir, const_ir)
            new_assignment = IrAssignment(new_var, output)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            continue
        elif json_obj["method"] == "SymExpSparse":
            if "json_list_" in json_obj["mat"]:
                mat_ir = output_vars[int(json_obj["mat"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if "json_list_" in json_obj["const"]:
                const_ir = output_vars[int(json_obj["const"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if irMetadata is None:
                raise Exception("SymExpSparse replay requires SymExp irMetadata")
            mat_metadata = copy_metadata(irMetadata)
            mat_metadata[-1].type = 'Float'
            mat_metadata[-1].shape.append(IrAst.sym_size)
            mat_metadata[-1].broadcast.append(1)
            mat_ir.irMetadata = mat_metadata
            const_metadata = copy_metadata(irMetadata)
            const_metadata[-1].type = 'Float'
            const_ir.irMetadata = const_metadata
            output = IrCombineToSym(mat_ir, const_ir)
            new_assignment = IrAssignment(new_var, output)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            continue
        elif json_obj["method"] == "get_sparse_tensor_blocks":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSparseTensorBlocks(input_ir)
        elif json_obj["method"] == "get_abs_elem_sparse_d_key":
            if "json_list_" in json_obj["input"]:
                # print(len(output_vars), int(json_obj["input"].split("_")[-1]))
                # input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
                pass
            else:
                raise Exception("NOT IMPLEMENTED")
            key = json_obj["key"]
            output = IrGetAbsElemSparseDKey(None, key)
        elif json_obj["method"] == "get_poly_exp_sparse_const":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetPolyExpSparseConst(input_ir)
        elif json_obj["method"] == "get_poly_exp_sparse_mat":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetPolyExpSparseMat(input_ir)
        elif json_obj["method"] == "get_sym_exp_sparse_const":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSymExpSparseConst(input_ir)
        elif json_obj["method"] == "get_sym_exp_sparse_mat":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSymExpSparseMat(input_ir)
        elif json_obj["method"] == "expand_symexp_mat":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrExpandSymExp(input_ir)
        elif json_obj["method"] == "get_kth_layer_bias":
            output = IrGetKthLayerNetworkParam(json_obj["input"], "bias")
        elif json_obj["method"] == "get_kth_layer_weight":
            output = IrGetKthLayerNetworkParam(json_obj["input"], "weight")
        else:
            raise Exception(f"Unknown method {json_obj['method']} in replay at output {json_obj.get('output')}")
        
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
    if expr is None:
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


def collapse_cfg_to_single_block(cfg, layer_index):
    """
    Resolve all control flow in a per-layer reuse CFG into a single
    straight-line block. Starting at the entry block we walk the profiled live
    path -- exactly the order codegen emits -- pruning each profiled branch to
    its taken arm, and concatenate every visited block's instructions into the
    entry block. Afterwards the CFG has one node with no `jump`/`inner_jump`, so
    downstream passes (subexp_inlining, codegen) see a flat instruction stream.

    Reuse requires control flow to be fully resolved by the profile: any block
    whose control flow does not collapse to a single successor -- an unprofiled
    ternary, a plain `if`, or a `while` that was not unrolled -- is a hard error
    rather than a fallback to a runtime branch.
    """
    order = []
    visited = set()

    def visit(block):
        if block is None or id(block) in visited:
            return
        visited.add(id(block))
        order.append(block)
        if block.inner_jump is not None:
            if len(block.inner_jump) == 3:
                taken = get_profiled_branch(layer_index, block.block_id)
                if taken == 'then':
                    visit(block.inner_jump[1])
                elif taken == 'else':
                    visit(block.inner_jump[2])
                else:
                    raise RuntimeError(
                        f'reuse: profiled branch at block {block.block_id} '
                        f'(layer {layer_index}) is unresolved; cannot linearize')
            else:
                raise RuntimeError(
                    f'reuse: unresolvable control flow at layer {layer_index} '
                    f'(a plain `if` or an un-unrolled `while`); reuse requires '
                    f'fully profiled/unrolled control flow')
        if block.jump is not None:
            visit(block.jump[1])

    entry_block = cfg.ir[cfg.entry_node]
    visit(entry_block)

    entry_block.children = [
        instr for block in order for instr in block.children]
    entry_block.inner_jump = None
    entry_block.jump = None

    entry = cfg.entry_node
    cfg.nodes = [entry]
    cfg.ir = {entry: entry_block}
    cfg.successors = {entry: []}
    cfg.predecessors = {entry: []}


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
                collapse_cfg_to_single_block(cfg, layer_index)
                new_cfgs[layer_index] = cfg

            transformerIr.layerwise_cfgs = new_cfgs

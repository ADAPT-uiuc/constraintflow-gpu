import json
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


def convert_to_ir_ttb(expr, layer_index, while_iteration):
    targets = (IrBinaryOp, IrMult, IrInnerProduct, IrRepeat, IrClamp, IrDot)
    if not isinstance(expr, targets):
        return expr, []
    
    binary_instance = expr.ttb_counter

    if isinstance(expr, IrMult) or isinstance(expr, IrBinaryOp):
        filename = f"jit_binary/binary_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrInnerProduct):
        filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrRepeat):
        filename = f"jit_repeat/repeat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrClamp):
        filename = f"jit_clamp/clamp_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrDot):
        [lhsIr, rhsIr] = expr.children
        if lhsIr.irMetadata[-1].type != 'Float':
            return expr, []
        filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    with open(filename, 'r') as f:
        json_list = json.load(f)
    print(binary_instance)
    lhs = expr.children[0]
    rhs = expr.children[1]
    if isinstance(rhs, IrAst):
        irMetadata = rhs.irMetadata
    else:
        irMetadata = None
    new_assignments = []
    output_vars = []

    for json_obj in json_list:
        new_name = get_var()
        new_var = IrVar(new_name, irMetadata)
        if json_obj["method"] == "noop":
            if json_obj["input"] == "rhs":
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
            else:
                raise Exception("NOT IMPLEMENTED")

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
            output = IrTensorClamp(inputIr, json_list['const'], json_list['min_true'])

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
        
        else:
            raise Exception(f"Unknown method {json_obj['method']} in replay")
        
        new_assignment = IrAssignment(new_var, output)
        new_assignments.append(new_assignment)
        output_vars.append(new_var)

        if(len(output_vars) != json_obj["output"] + 1):
            print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
        assert(len(output_vars) == json_obj["output"] + 1)

    return output_vars[-1], new_assignments


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
        index += 1


def replace_all_occurrences_expr(expr, var_map):
    if isinstance(expr, IrVar) and expr.name in var_map.keys(): 
        return var_map[expr.name]
    if isinstance(expr, int):
        return expr
    for i in range(len(expr.children)):
        new_child = replace_all_occurrences_expr(expr.children[i], var_map)
        expr.children[i] = new_child
    return expr

def remove_while(layer_index, num_iterations, cfg, root_node, first_while_node, second_while_node, exit_node, break_node):
    root_block = cfg.ir[root_node]
    first_while_block = cfg.ir[first_while_node]
    second_while_block = cfg.ir[second_while_node]

    tensor_to_block_block(root_block, layer_index)
    # tensor_to_block_block(exit_block, layer_index)

    if exit_node is not None:
        exit_block = cfg.ir[exit_node]


    ir_list = root_block.children

    new_vars = {}
    for i in range(num_iterations-1):
        new_ir_list = []
        combined_list = first_while_block.children + second_while_block.children
        for ir in combined_list:
            if isinstance(ir, IrAssignment):
                new_var_name = get_var_while(ir.children[0].name, i)
                new_var = IrVar(new_var_name, ir.children[0].irMetadata)
                old_rhs = copy.deepcopy(ir.children[1])
                new_rhs = replace_all_occurrences_expr(old_rhs, new_vars)
                new_assignment = IrAssignment(new_var, new_rhs)
                new_ir_list.append(new_assignment)
                new_vars[ir.children[0].name] = new_var
            else:
                raise Exception("NOT IMPLEMENTED for IR type " + str(type(ir)))
        tensor_to_block_block(None, layer_index=layer_index, ir_list=new_ir_list, while_iteration=i)
        ir_list += new_ir_list
    new_ir_list = []
    # new_vars = {}
    combined_list = first_while_block.children 
    for ir in combined_list:
        if isinstance(ir, IrAssignment):
            new_var_name = get_var_while(ir.children[0].name, num_iterations-1)
            new_var = IrVar(new_var_name, ir.children[0].irMetadata)
            old_rhs = copy.deepcopy(ir.children[1])
            new_rhs = replace_all_occurrences_expr(old_rhs, new_vars)
            new_assignment = IrAssignment(new_var, new_rhs)
            new_ir_list.append(new_assignment)
            new_vars[ir.children[0].name] = new_var
        else:
            raise Exception("NOT IMPLEMENTED for IR type " + str(type(ir)))
    tensor_to_block_block(None, layer_index=layer_index, ir_list=new_ir_list, while_iteration=num_iterations-1)
    ir_list += new_ir_list

    new_ir_list = []
    for ir in exit_block.children:
        # ir_list.append(ir)
        if isinstance(ir, IrAssignment):
            old_rhs = copy.deepcopy(ir.children[1])
            new_rhs = replace_all_occurrences_expr(old_rhs, new_vars)
            new_assignment = IrAssignment(ir.children[0], new_rhs)
            new_ir_list.append(new_assignment)
        elif isinstance(ir, IrTransRetBasic):
            new_children = []
            for child in ir.children:
                old_child = copy.deepcopy(child)
                new_child = replace_all_occurrences_expr(old_child, new_vars)
                new_children.append(new_child)
            new_ir_list.append(IrTransRetBasic(new_children))
        else:
            raise Exception("NOT IMPLEMENTED for IR type " + str(type(ir)))
    tensor_to_block_block(None, layer_index=layer_index, ir_list=new_ir_list, while_iteration=None)
    ir_list += new_ir_list

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
    while(i < len(cfg.nodes)):
        node = cfg.nodes[i]
        
        block = cfg.ir[node]
        if block.inner_jump is None:
            tensor_to_block_block(block, layer_index)
            i+=1
            continue
        if len(block.inner_jump) == 3:
            tensor_to_block_block(block, layer_index)
            i+=1
            continue
        if not isinstance(block.inner_jump[1], IrWhileBlock):
            tensor_to_block_block(block, layer_index)
            i+=1
            continue
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
                num_iterations = json_obj["num_iterations"]+1
            
            
            remove_while(layer_index, num_iterations, cfg, root_node, first_while_node, second_while_node, exit_node, break_node)


def tensor_to_block_cfg(cfg, layer_index):
    for node in cfg.nodes:
        block = cfg.ir[node]
        tensor_to_block_block(block, layer_index)

def tensor_to_block(ir):
    # TODO: DEBUG THE FOLLOWING LINE
    # uses.populate_uses_defs(ir)
    for transformer in ir.tstore.keys():
        affine = [1, 3, 5, 7]
        relu = [2, 4, 6]
        for i in range(len(ir.tstore[transformer])):
            transformerIr = ir.tstore[transformer][i]
            if transformerIr.op == 'Affine':
                layer_indices = affine
            elif transformerIr.op == 'Relu':
                layer_indices = relu
            else:
                print("NOT IMPLEMENTED for op " + transformerIr.op)
                continue
            
            new_cfgs = {}
            for j, layer_index in enumerate(layer_indices):
                cfg = deepcopy_cfg_with_fresh_identifiers(transformerIr.cfg)
                unroll_while(cfg, layer_index)
                print(layer_index, 'after unroll while')
                new_cfgs[layer_index] = cfg
            
            transformerIr.layerwise_cfgs = new_cfgs
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


def convert_to_ir(expr, layer_index):
    if not isinstance(expr, IrBinaryOp) and not isinstance(expr, IrMult):
        return expr, []
    
    binary_instance = expr.binary_counter
    with open(f"jit_binary/binary_{layer_index}_{binary_instance}.json", 'r') as f:
        json_list = json.load(f)
    print(binary_instance)
    lhs = expr.children[0]
    rhs = expr.children[1]
    new_assignments = []
    output_vars = []

    for json_obj in json_list:
        new_name = get_var()
        new_var = IrVar(new_name, rhs.irMetadata)
        if json_obj["method"] == "noop":
            if json_obj["input"] == "rhs":
                output = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
        
        elif json_obj["method"] == "SparseTensor":
            if "json_list_" in json_obj['blocks']:
                blocksIr = output_vars[int(json_obj['blocks'].split("_")[-1])]
            else:                        
                raise Exception("NOT IMPLEMENTED")
            output = IrSparseTensor(torch.Tensor(json_obj["start_indices"]), blocksIr, json_obj["dims"], torch.Tensor(json_obj["total_size"]), torch.Tensor(json_obj["end_indices"]), type=getattr(builtins, json_obj["type"]), dense_const=json_obj["dense_const"])
        
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
            output = IrGetSubBlockCustomRange(inputIr, torch.Tensor(json_obj["start_index"]), torch.Tensor(json_obj["end_index"]), json_obj["block_id"], json_obj["tensor"])

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
        
        elif json_obj["method"] == "ConstBlock":
            shape = torch.tensor(json_obj["total_shape"], dtype=torch.int64)
            output = IrConstBlock(json_obj["block"], shape)
        
        
        else:
            raise Exception(f"Unknown method {json_obj['method']} in replay")
        
        new_assignment = IrAssignment(new_var, output)
        new_assignments.append(new_assignment)
        output_vars.append(new_var)

        if(len(output_vars) != json_obj["output"] + 1):
            print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
        assert(len(output_vars) == json_obj["output"] + 1)

    return output_vars[-1], new_assignments

# def tensor_to_block_block(block, layer_index):
#     ir_list = block.children
#     length = len(ir_list)
#     index = 0
#     for i in range(length):
#         l = ir_list[index]
#         if isinstance(l, IrAssignment) and isinstance(ir_list[i].children[1], IrBinaryOp):
#             binary_instance = ir_list[i].children[1].binary_counter
#             print(layer_index, binary_instance, '@@@@@@@@@@@@@@@@@@@@@@@@')
#             filename = f"jit_binary/binary_{layer_index}_{binary_instance}.json"
#             with open(filename, 'r') as f:
#                 json_list = json.load(f)
#             new_expr, new_assignments = convert_to_ir(json_list, l.children[1])
#             new_children = [l.children[0], new_expr]
#             l.update_parent_child(new_children)
#             for j in range(len(new_assignments)):
#                 ir_list.insert(index, new_assignments[j])
#                 index += 1
#         index += 1


def tensor_to_block_block(block, layer_index):
    ir_list = block.children
    length = len(ir_list)
    index = 0
    for i in range(length):
        l = ir_list[index]
        if isinstance(l, IrAssignment):
            new_expr, new_assignments = convert_to_ir(l.children[1], layer_index)
            new_children = [l.children[0], new_expr]
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        elif isinstance(l, IrTransRetBasic):
            new_children = []
            new_assignments = []
            for child in l.children:
                new_expr, new_assignments_inner = convert_to_ir(child, layer_index)
                new_children.append(new_expr)
                new_assignments += new_assignments_inner
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
        index += 1





def tensor_to_block_cfg(cfg, layer_index):
    for node in cfg.nodes:
        block = cfg.ir[node]
        tensor_to_block_block(block, layer_index)

def tensor_to_block(ir):
    uses.populate_uses_defs(ir)
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
                new_cfgs[layer_index] = cfg
            for j, layer_index in enumerate(layer_indices):
                tensor_to_block_cfg(new_cfgs[layer_index], layer_index)
            
            transformerIr.layerwise_cfgs = new_cfgs
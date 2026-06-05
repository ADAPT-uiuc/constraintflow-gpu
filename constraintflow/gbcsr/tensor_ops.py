import torch 
import math
import operator
from constraintflow.gbcsr.sparse_tensor import *
from constraintflow.lib.globals import *

input_size = 784

def check_type_equality(x, y):
    if x == y:
        return True
    if x in [float, int, torch.float, torch.int] and y in [float, int, torch.float, torch.int]:
        return True
    return False

types = {bool: torch.bool, int: torch.int, float: torch.float}
# equivalent_types = {(int, )}

def checkTypes(x, y):
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            if not check_type_equality(x.type, y.type):
            # if x.type != y.type:
                print(x.type, y.type)
                raise Exception('TYPE MISMATCH')
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, bool):
            if not check_type_equality(x.type, type(y)):
            # if type(y) != x.type:
                raise Exception('TYPE MISMATCH')
        if isinstance(y, torch.Tensor):
            if not check_type_equality(types[x.type], y.dtype):
            # if types[x.type] != y.dtype:
                print(x.type, y.dtype)
                raise Exception('TYPE MISMATCH')
    elif isinstance(y, SparseTensor):
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool):
            if not check_type_equality(y.type, type(x)):
            # if type(x) != y.type:
                raise Exception('TYPE MISMATCH')
    elif isinstance(x, SparseTensor):
        if isinstance(y, torch.Tensor):
            if not check_type_equality(x.type, y.dtype):
            # if x.type != y.dtype:
                raise Exception('TYPE MISMATCH')
    elif isinstance(y, SparseTensor):
        if isinstance(x, torch.Tensor):
            if not check_type_equality(x.dtype, y.type):
            # if y.type != x.dtype:
                raise Exception('TYPE MISMATCH')
    elif not check_type_equality(type(x), type(y)):
    # elif type(x) != type(y):
        print(type(x), type(y))
        raise Exception('TYPE MISMATCH')

def checkShapes(x, y):
    if isinstance(x, SparseTensor):
        if isinstance(y, float) or isinstance(y, int):
            return
        elif isinstance(y, torch.Tensor):
            if not (x.total_size == torch.tensor(y.shape)).all():
                print(x.total_size, y.shape)
                raise Exception('SHAPE MISMATCH')
        elif isinstance(y, SparseTensor):
            if not (x.total_size == y.total_size).all():
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
    elif isinstance(y, SparseTensor):
        if isinstance(x, float) or isinstance(x, int):
            return True
        elif isinstance(x, torch.Tensor):
            if not (y.total_size == torch.tensor(x.shape)).all():
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
    
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if x.shape != y.shape:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')

def sanityCheck(x, y):
    # return
    start_time = time.perf_counter()
    checkTypes(x, y)
    checkShapes(x, y)
    end_time = time.perf_counter()
    sanity_time.update_total_time(end_time - start_time)

def unary(x, op, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    if isinstance(x, torch.Tensor):
        json_list = []
        json_list.append({"method": "noop", "input": "lhs", "output": 0})
        op_name = '-' if op == operator.neg else 'not' if op == operator.not_ else 'sigma'
        json_list.append({"method": "simple_unary", "input": "json_list_0", "op": op_name, "output": 1})
        res = op(x)
        if dummy_mode:
            if layer_index is not None and counter is not None:
                os.makedirs("jit_unary", exist_ok=True)
                capture_path = f"jit_unary/unary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                with open(capture_path, 'w') as f:
                    json.dump(json_list, f, indent=4)
    elif isinstance(x, SparseTensor):
        json_list = []
        json_list.append({"method": "noop", "input": "lhs", "output": 0})
        res = x.unary(op, json_list=json_list, lhs_index=0)
        if dummy_mode:
            if layer_index is not None and counter is not None:
                os.makedirs("jit_unary", exist_ok=True)
                capture_path = f"jit_unary/unary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                with open(capture_path, 'w') as f:
                    json.dump(json_list, f, indent=4)
    else:
        res = op(x)
    unary_time.update_total_time(time.perf_counter() - start_time)
    return res

def any(x, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    if type(x)!=torch.Tensor and type(x)!=SparseTensor:
        raise Exception('TYPE MISMATCH')
    json_list = [
        {"method": "noop", "input": "lhs", "output": 0},
    ]
    if type(x) == torch.Tensor:
        json_list.append({"method": "any", "input": "json_list_0", "output": len(json_list)})
        res = x.any()
    else:
        res = x.any(json_list=json_list, lhs_index=0)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_any", exist_ok=True)
            with open(f"jit_any/any_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", 'w') as f:
                json.dump(json_list, f, indent=4)
    any_time.update_total_time(time.perf_counter() - start_time)
    return res

def all(x, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    if type(x)!=torch.Tensor and type(x)!=SparseTensor:
        raise Exception('TYPE MISMATCH')
    
    json_list = [{"method": "noop", "input": "lhs", "output": 0}]
    if type(x) == torch.Tensor:
        json_list.append({"method": "all", "input": "json_list_0", "output": len(json_list)})
        res = x.all()
    else:
        #It never takes this branch
        nknkkjk
        res = x.all()

    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_all", exist_ok=True)
            with open(f"jit_all/all_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", 'w') as f:
                json.dump(json_list, f, indent=4)
    all_time.update_total_time(time.perf_counter() - start_time)
    return res

# def all(x):
#     if type(x)!=torch.Tensor:
#         raise Exception('TYPE MISMATCH')
#     return x.all()

def binary(x, y, op, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None, parent_json_list = None):
    total_start_time = time.perf_counter()
    start_time = time.perf_counter()
    sanityCheck(x, y)
    binary_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
    if parent_json_list is None:
        json_list = []
    else:
        json_list = parent_json_list
    if isinstance(x, SparseTensor):
        start_time = time.perf_counter()
        lhs_idx = len(json_list)
        json_obj = {"method": "noop", "input": "lhs", "output": len(json_list)}
        json_list.append(json_obj)
        rhs_idx = len(json_list)
        json_obj = {"method": "noop", "input": "rhs", "output": len(json_list)}
        json_list.append(json_obj)

        res = x.binary(y, op, json_list = json_list, lhs_index=lhs_idx, rhs_index=rhs_idx)
        binary_tensor_ops_x_sparsity.update_total_time(time.perf_counter() - start_time)

        
        
    elif isinstance(y, SparseTensor):
        lhs_idx = len(json_list)
        json_obj = {"method": "noop", "input": "lhs", "output": len(json_list)}
        json_list.append(json_obj)
        rhs_idx = len(json_list)
        json_obj = {"method": "noop", "input": "rhs", "output": len(json_list)}
        json_list.append(json_obj)

        start_time = time.perf_counter()
        temp = convert_dense_to_sparse(x, y.total_size, json_list=json_list, x_index=lhs_idx)
        binary_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
        res = temp.binary(y, op, json_list = json_list, lhs_index=lhs_idx, rhs_index=rhs_idx)
        binary_tensor_ops_y_sparsity.update_total_time(time.perf_counter() - start_time)
        
    else:
        start_time = time.perf_counter()
        json_obj = {
            "method": "simple_binary",
            "lhs": "lhs",
            "rhs": "rhs",
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = op(x, y)
        binary_tensor_ops_no_sparse.update_total_time(time.perf_counter() - start_time)
    total_binary_tensor_ops.update_total_time(time.perf_counter() - total_start_time)
    # if inside_while:
    #     print(while_iteration)
    if dummy_mode and parent_json_list is None:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_binary", exist_ok=True)
            capture_path = f"jit_binary/binary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            # if inside_while:
            #     print(while_iteration)
            #     print(capture_path)
            
            with open(capture_path, 'w') as f:
                json.dump(json_list, f, indent=4)
    return res

def cf_max(x, y, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            json_list = []
            json_list.append({"method": "noop", "input": "lhs", "output": 0})
            json_list.append({"method": "noop", "input": "rhs", "output": 1})
            res = sparse_max(x, y, json_list=json_list, x_json_index=0, y_json_index=1)
            if dummy_mode:
                if layer_index is not None and counter is not None:
                    os.makedirs("jit_binary", exist_ok=True)
                    capture_path = f"jit_binary/binary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                    with open(capture_path, 'w') as f:
                        json.dump(json_list, f, indent=4)
            where_time.update_total_time(time.perf_counter() - start_time)
            return res
    res = torch.max(x, y)
    where_time.update_total_time(time.perf_counter() - start_time)
    return res

def cf_min(x, y, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            json_list = []
            json_list.append({"method": "noop", "input": "lhs", "output": 0})
            json_list.append({"method": "noop", "input": "rhs", "output": 1})
            res = sparse_min(x, y, json_list=json_list, x_json_index=0, y_json_index=1)
            if dummy_mode:
                if layer_index is not None and counter is not None:
                    os.makedirs("jit_binary", exist_ok=True)
                    capture_path = f"jit_binary/binary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                    with open(capture_path, 'w') as f:
                        json.dump(json_list, f, indent=4)
            where_time.update_total_time(time.perf_counter() - start_time)
            return res
    res = torch.min(x, y)
    where_time.update_total_time(time.perf_counter() - start_time)
    return res

def lcm(a, b):
    if isinstance(a, float) or isinstance(a, int) or isinstance(a, bool):
        return b
    if isinstance(b, float) or isinstance(b, int) or isinstance(b, bool):
        return a
    assert(a.shape[0] == b.shape[0])
    total_size = []
    for j in range(len(a)):
        total_size.append(math.lcm(int(a[j].item()), int(b[j].item())))
    return torch.tensor(total_size)

def const_to_sparse(c, total_size):
    return SparseTensor([], [], total_size.shape[0], total_size, type=type(c), dense_const=c)

def where(x, y, z, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    start_time = time.perf_counter()
    json_list = []
    json_list.append({"method": "noop", "input": "cond", "output": 0})   # index 0
    json_list.append({"method": "noop", "input": "lhs",  "output": 1})   # index 1
    json_list.append({"method": "noop", "input": "rhs",  "output": 2})   # index 2
    
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(z, torch.Tensor):
        checkShapes(x, y)
        sanityCheck(y, z)
        res = torch.where(x, y, z)
    if isinstance(x, bool) and isinstance(y, float) and isinstance(z, float):
        if x:
            res = y
        else:
            res = z
    
    if isinstance(x, SparseTensor):
        x_size = x.total_size
    elif isinstance(x, torch.Tensor):
        x_size = torch.tensor(x.shape)
    else:
        x_size = 0

    if isinstance(y, SparseTensor):
        y_size = y.total_size
    elif isinstance(y, torch.Tensor):
        y_size = torch.tensor(y.shape)
    else:
        y_size = 0

    if isinstance(z, SparseTensor):
        z_size = z.total_size
    elif isinstance(z, torch.Tensor):
        z_size = torch.tensor(z.shape)
    else:
        z_size = 0

    total_size = lcm(x_size, lcm(y_size, z_size))
    if isinstance(x, torch.Tensor):
        x1 = convert_dense_to_sparse(x, json_list=json_list, x_index=0)
        x_json_index = len(json_list)-1

    elif isinstance(x, bool):
        x1 = const_to_sparse(x, total_size)
        json_list.append({"method": "SparseTensor", "start_indices": [], "blocks": [],
                      "dims": len(total_size), "total_size": total_size.tolist(),
                      "type": "bool", "dense_const": x, "output": len(json_list)})
        x_json_index = len(json_list)-1

    else:
        x1 = x
        x_json_index = 0

    if isinstance(y, torch.Tensor):
        y1 = convert_dense_to_sparse(y, json_list=json_list, x_index=1)
        y_json_index = len(json_list)-1
    elif isinstance(y, float):
        y1 = const_to_sparse(y, total_size)
        json_list.append({"method": "SparseTensor", "start_indices": [], "blocks": [],
                      "dims": len(total_size), "total_size": total_size.tolist(),
                      "type": "float", "dense_const": y, "output": len(json_list)})
        y_json_index = len(json_list)-1
    else:
        y1 = y
        y_json_index = 1


    if isinstance(z, torch.Tensor):
        z1 = convert_dense_to_sparse(z, json_list=json_list, x_index=2)
        z_json_index = len(json_list)-1
    elif isinstance(z, float):
        z1 = const_to_sparse(z, total_size)
        json_list.append({"method": "SparseTensor", "start_indices": [], "blocks": [],
                      "dims": len(total_size), "total_size": total_size.tolist(),
                      "type": "float", "dense_const": z, "output": len(json_list)})
        z_json_index = len(json_list)-1
    else:
        z1 = z
        z_json_index = 2
    checkShapes(x1, y1)
    sanityCheck(y1, z1)

    res = sp_where(x1, y1, z1, json_list = json_list, x_json_index = x_json_index, y_json_index = y_json_index, z_json_index = z_json_index)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_where", exist_ok=True)
            capture_path = f"jit_where/where_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            with open(capture_path, 'w') as f:
                json.dump(json_list, f, indent=4)
    where_time.update_total_time(time.perf_counter() - start_time)
    return res

def inner_prod(x, y, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    total_start_time = time.perf_counter()
    # time.sleep(0.00625)
    checkTypes(x, y)
    matmul_tensor_ops_expenses.just_update_total_time(time.perf_counter() - total_start_time)
    start_time = time.perf_counter()
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            if x.total_size.shape[0] == y.total_size.shape[0]:
                if x.total_size[-1] != y.total_size[-2]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
                if (x.total_size[:-2] != y.total_size[:-2]).all():
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            elif x.total_size.shape[0] > y.total_size.shape[0]:
                if x.total_size[-1] != y.total_size[-1]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.total_size[:-1]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
            matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
            json_list = []
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "noop",
                "input": "rhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = x.matmul(y, json_list = json_list, lhs_index=0, rhs_index=1)

            if dummy_mode:
                if layer_index is not None and counter is not None:
                    os.makedirs("jit_matmul", exist_ok=True)
                    # capture_occurrence = _next_jit_occurrence(_jit_save_occurrence, layer_index, counter)
                    capture_path = f"jit_matmul/matmul_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                    
                    with open(capture_path, 'w') as f:
                        json.dump(json_list, f, indent=4)
        else:
            if x.total_size.shape[0] == y.shape.shape[0]:
                if x.total_size[-1] != y.shape[-2]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.shape[:-2]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
            elif x.total_size.shape[0] > y.shape.shape[0]:
                if x.total_size[-1] != y.shape[-1]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.shape[:-1]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.shape)
                raise Exception('SHAPE MISMATCH')
            matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
            res = x.matmul(y)
    elif isinstance(y, SparseTensor):
        if x.shape.shape[0] == y.total_size.shape[0]:
            if x.shape[-1] != y.total_size[-2]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.total_size[:-2]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
        elif x.shape.shape[0] > y.total_size.shape[0]:
            if x.shape[-1] != y.total_size[-1]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.total_size[:-1]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
        else:
            print(x.shape, y.total_size)
            raise Exception('SHAPE MISMATCH')
        x = convert_dense_to_sparse(x)
        matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
        res = x.matmul(y)
    else:
        if x.shape.shape[0] == y.shape.shape[0]:
            if x.shape[-1] != y.shape[-2]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.shape[:-2]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
        elif x.shape.shape[0] > y.shape.shape[0]:
            if x.shape[-1] != y.shape[-1]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.shape[:-1]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
        else:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')
        matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
        res = x@y

    matmul_tensor_ops.update_total_time(time.perf_counter() - total_start_time)
    return res



def convert_to_float(x):
    if isinstance(x, torch.Tensor):
        res = x.float()
    elif isinstance(x, SparseTensor):
        res = x.float()
    else:
        return x
    return res

# This does not seem to be used anywhere
# def get_default_stop1(shape):
#     return SparseTensor([], [], len(shape), torch.tensor(shape), type=bool, dense_const=False)

def get_default_stop(shape, abs_elem, batch_size, curr_size, poly_size, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    res = []
    res_start_indices = []
    res_end_indices = []
    json_list = []
    live_layers = list((abs_elem.d['llist']))
    json_list.append({"method": "initialise", "value": "[]", "output": 0})
    current_list_index = 0

    for i in range(len(abs_elem.network)):
        if live_layers[i]:
            res_start_indices.append(torch.tensor([0, 0, abs_elem.network[i].start]))
            res_end_indices.append(torch.tensor([batch_size, curr_size, abs_elem.network[i].end]))
            json_obj = {
                "method": "ConstBlock",
                "block": False,
                "total_shape": [batch_size, curr_size, abs_elem.network[i].size],
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res.append(ConstBlock(False, torch.tensor([batch_size, curr_size, abs_elem.network[i].size])))

            block_index = len(json_list) - 1
            json_obj = {
                "method": "append_list",
                "list": f"json_list_{current_list_index}",
                "value": f"json_list_{block_index}",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            current_list_index = len(json_list) - 1
    
    json_list.append({
        "method": "SparseTensor",
        "start_indices": [si.tolist() for si in res_start_indices],
        "blocks": f"json_list_{current_list_index}",
        "end_indices": [ei.tolist() for ei in res_end_indices],
        "dims": len(shape),
        "total_size": list(shape),
        "type": "bool",
        "dense_const": True,
        "output": len(json_list)
    })
    
    temp = SparseTensor(res_start_indices, res, len(shape), torch.tensor(shape), res_end_indices, type=bool, dense_const=True)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_defaultstop", exist_ok=True)
            capture_path = f"jit_defaultstop/stop_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            with open(capture_path, 'w') as f:
                json.dump(json_list, f, indent=4)
    return temp

# This does not seem to be used anywhere
# def get_default_stop2(shape):
#     global input_size
#     vertices_stop_default = torch.zeros(shape)
#     vertices_stop_default[:, 0:834] = 1
#     vertices_stop_default = vertices_stop_default.bool()
#     return vertices_stop_default

def get_max_priority(sp_tensor, active_vertices: SparseTensor, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    priorities = []
    for i in range(sp_tensor.num_blocks):
        # print(f'sp_tensor.blocks[i] type: {type(sp_tensor.blocks[i])}')
        assert(isinstance(sp_tensor.blocks[i], ConstBlock))
        if active_vertices.exists_sub_block(sp_tensor.start_indices[i], sp_tensor.end_indices[i]):
            priorities.append(sp_tensor.blocks[i].block)
        else:
            priorities.append(float('-inf'))
    if len(priorities) == 0:
        max_priority = float('-inf')
    else:
        max_priority = max(priorities)
    res_blocks = []
    res_start_indices = []
    res_end_indices = []
    
    for i in range(sp_tensor.num_blocks):
        if priorities[i] == max_priority:
            # if active_vertices.get_sparse_custom_range(sp_tensor.start_indices[i], sp_tensor.end_indices[i]).any():
            res_blocks.append(ConstBlock(True, sp_tensor.blocks[i].total_shape))
            res_start_indices.append(sp_tensor.start_indices[i])
            res_end_indices.append(sp_tensor.end_indices[i])
            # continue
    json_list = []
    json_list.append({"method": "initialise", "value": "[]", "output": 0})
    current_list_index = 0
    if dummy_mode:
        for i in range(len(res_blocks)):
            json_list.append({
                "method": "ConstBlock",
                "block": True,
                "total_shape": res_blocks[i].total_shape.tolist(),
                "output": len(json_list),
            })
            block_index = len(json_list) - 1
            json_list.append({
                "method": "append_list",
                "list": f"json_list_{current_list_index}",
                "value": f"json_list_{block_index}",
                "output": len(json_list),
            })
            current_list_index = len(json_list) - 1
        json_list.append({
            "method": "SparseTensor",
            "start_indices": [si.tolist() for si in res_start_indices],
            "blocks": f"json_list_{current_list_index}",
            "end_indices": [ei.tolist() for ei in res_end_indices],
            "dims": sp_tensor.dims,
            "total_size": sp_tensor.total_size.tolist(),
            "type": "bool",
            "dense_const": False,
            "output": len(json_list),
        })
        if dummy_mode:
            os.makedirs("jit_priority", exist_ok=True)
            capture_path = f"jit_priority/priority_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            with open(capture_path, 'w') as f:
                json.dump(json_list, f, indent=4)
    return SparseTensor(res_start_indices, res_blocks, sp_tensor.dims, sp_tensor.total_size, end_indices=res_end_indices, type=bool, dense_const=False)

def filter_trav_exp_stop(trav_exp, stop, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    stop_float = convert_to_float(stop)
    json_list = []
    json_obj = {
        "method": "noop",
        "input": "lhs",
        "output": len(json_list),
    }    
    json_list.append(json_obj)
    lhs_index = len(json_list)-1

    polyexp_stop_mat = binary(trav_exp.mat, stop_float, operator.mul, layer_index=layer_index, counter=counter, inside_while=inside_while, while_number=while_number, while_iteration=while_iteration, parent_json_list=json_list)
    json_obj = {
        "method": "create_similar",
        "input": f"json_list_{lhs_index}",
        "mat": f"json_list_{len(json_list)-1}",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    polyexp_stop = trav_exp.create_similar(mat = polyexp_stop_mat)
    if dummy_mode:
        os.makedirs("jit_polyexp_stop", exist_ok=True)
        capture_path = f"jit_polyexp_stop/stop_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        with open(capture_path, 'w') as f:
            json.dump(json_list, f, indent=4)

    return polyexp_stop

def filter_trav_exp_not_stop(trav_exp, stop, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    json_list = []
    json_obj = {
        "method": "noop",
        "input": "lhs",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    lhs_index = len(json_list)-1

    if isinstance(trav_exp.const, SparseTensor):
        polyexp_not_stop_const = SparseTensor([], [], trav_exp.const.dims, trav_exp.const.total_size, type=float, dense_const=0)
        json_obj = {
            "method": "SparseTensor",
            "start_indices": [],
            "blocks": [],
            "dims": trav_exp.const.dims,
            "total_size": trav_exp.const.total_size.tolist(),
            "type": "float",
            "dense_const": 0,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        polyexp_not_stop_const_index = len(json_list)-1
    else:
        polyexp_not_stop_const = 0
        json_obj = {"method":"scalar_cost", "value": 0, "output": len(json_list)}
        json_list.append(json_obj)
        polyexp_not_stop_const_index = len(json_list)-1
    stop_float = convert_to_float(stop.unary(operator.not_))
    polyexp_not_stop_mat = binary(trav_exp.mat, stop_float, operator.mul, layer_index=layer_index, counter=counter, inside_while=inside_while, while_number=while_number, while_iteration=while_iteration, parent_json_list=json_list)
    polyexp_not_stop = trav_exp.create_similar(mat = polyexp_not_stop_mat, const = polyexp_not_stop_const)
    json_obj = {
        "method": "create_similar",
        "input": f"json_list_{lhs_index}",
        "mat": f"json_list_{len(json_list)-1}",
        "const": f"json_list_{polyexp_not_stop_const_index}",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    if dummy_mode:
        os.makedirs("jit_polyexp_not_stop", exist_ok=True)
        capture_path = f"jit_polyexp_not_stop/notstop_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        with open(capture_path, 'w') as f:
            json.dump(json_list, f, indent=4)
    return polyexp_not_stop



def get_dims(x, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    if isinstance(x, SparseTensor):
        res = x.dims
    elif isinstance(x, torch.Tensor):
        res = x.dim()
    else:
        assert(False), f'get_dims type of x: {type(x)}'
        res = 1
    json_list = [
        {"method": "noop", "input": "lhs", "output": 0},
        {"method": "get_dims", "input": "json_list_0", "output": 1},
    ]
    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_get_dims", exist_ok=True)
            with open(f"jit_get_dims/get_dims_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", 'w') as f:
                json.dump(json_list, f, indent=4)
    return res

def get_shape_1(x):
    if isinstance(x, SparseTensor):
        return x.total_size[1]
    if not isinstance(x, torch.Tensor):
        raise Exception('TYPE MISMATCH')
    return x.shape[1]

def get_shape_0(x):
    if (not isinstance(x, torch.Tensor)) or (not isinstance(x, SparseTensor)):
        raise Exception('TYPE MISMATCH')
    if isinstance(x, SparseTensor):
        return x.total_size[0]
    return x.shape[0]

def repeat(mat, repeat_dims, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    json_list = []

    start_time = time.perf_counter()
    if isinstance(mat, float):
        json_obj = {
            "method": "tensor_ones",
            "repeat_dims": repeat_dims.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "multiplication",
            "lhs": "lhs",
            "rhs": "json_list_" + str(len(json_list)-1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat*torch.ones(*(repeat_dims.tolist()))
    elif isinstance(mat, torch.Tensor):
        json_obj = {
            "method": "tensor_repeat",
            "lhs": "lhs",
            "repeat_dims": repeat_dims.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat.repeat(*(repeat_dims.tolist()))
    else:
        json_obj = {
            "method": "noop",
            "input": "lhs",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat.repeat(repeat_dims, json_list=json_list, lhs_index=0)
    repeat_time.update_total_time(time.perf_counter() - start_time)

    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_repeat", exist_ok=True)
            capture_path = f"jit_repeat/repeat_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            with open(capture_path, 'w') as f:
                json.dump(json_list, f, indent=4)
    return res

def clamp(mat, const, min_true, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    start_time = time.perf_counter()
    json_list = []
    if isinstance(mat, float):
        if min_true:
            if mat>const:
                res = mat
                json_obj = {
                    "method": "noop",
                    "input": "lhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                
            else:
                res = const
                json_obj = {
                    "method": "noop",
                    "input": "rhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
        else:
            if mat<const:
                res = mat 
                json_obj = {
                    "method": "noop",
                    "input": "lhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
            else:
                res = const
                json_obj = {
                    "method": "noop",
                    "input": "rhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
        clamp_op_expense.update_total_time(time.perf_counter() - start_time)
    elif isinstance(mat, torch.Tensor):
        if min_true:
            clamp_op_expense.update_total_time(time.perf_counter() - start_time)
            res = mat.clamp(min=const)
            
        else:
            clamp_op_expense.update_total_time(time.perf_counter() - start_time)
            res = mat.clamp(max=const)
        json_obj = {
            "method": "tensor_clamp",
            "lhs": "lhs",
            "const": const,
            "min_true": min_true,
            "output": len(json_list),
        }
        json_list.append(json_obj)
    else:
        json_obj = {
            "method": "noop",
            "input": "lhs",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat.clamp(const, min_true, json_list=json_list, lhs_index=0)
    clamp_total_time.update_total_time(time.perf_counter() - start_time)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            os.makedirs("jit_clamp", exist_ok=True)
            capture_path = f"jit_clamp/clamp_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            with open(capture_path, 'w') as f:
                json.dump(json_list, f, indent=4)
    # clamp_time.update_total_time(time.perf_counter() - start_time)
    return res
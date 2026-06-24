from typing import Any
from constraintflow.gbcsr.sparse_tensor import *
from constraintflow.gbcsr.sparse_block import *

def get_num_eps(mat):
    if mat==None:
        return 0
    num = 0
    for i in range(mat.num_blocks):
        num += mat.blocks[i].total_shape[-1]
    return num

def get_new_eps(network, initial_shape, json_list=None, layer_index=None,
                counter=None, inside_while=False, while_number=None,
                while_iteration=None):
    owns_capture = (json_list is None) and dummy_mode
    if owns_capture:
        json_list = []
    trace = json_list is not None
    if not trace:
        json_list = []

    num = initial_shape[-1].item()
    const = SparseTensor([], [], len(initial_shape), initial_shape)
    if trace:
        const_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'SparseTensor',
            'start_indices': [],
            'blocks': [],
            'dims': len(initial_shape),
            'total_size': initial_shape.tolist(),
            'output': const_idx,
        }
        json_list.append(json_obj)
    start_index = torch.concat([torch.zeros(len(initial_shape), dtype=int), torch.tensor([SymExpSparse.count])])

    mat_tensor = torch.ones(num, dtype=int)
    if trace:
        mat_tensor_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'torch_ones',
            'size': [num],
            'output': mat_tensor_idx,
        }
        json_list.append(json_obj)
    for i in range(len(initial_shape)-1):
        mat_tensor = mat_tensor.unsqueeze(0)
        if trace:
            unsqueeze_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'torch_unsqueeze',
                'input': 'json_list_' + str(mat_tensor_idx),
                'index': 0,
                'output': unsqueeze_idx,
            }
            json_list.append(json_obj)
            mat_tensor_idx = unsqueeze_idx
    mat_tensor = mat_tensor.repeat(*(list(initial_shape[:-1]) + [1]))
    if trace:
        repeat_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'torch_repeat',
            'input': 'json_list_' + str(mat_tensor_idx),
            'repeats': [int(d) for d in initial_shape[:-1]] + [1],
            'output': repeat_idx,
        }
        json_list.append(json_obj)
        mat_tensor_idx = repeat_idx

    mat_total_shape = torch.tensor(list(initial_shape) + [num])
    mat = DiagonalBlock(mat_tensor, mat_total_shape, diag_index=len(initial_shape))
    if trace:
        diag_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'DiagonalBlock',
            'block': 'json_list_' + str(mat_tensor_idx),
            'total_shape': mat_total_shape.tolist(),
            'diag_index': len(initial_shape),
            'output': diag_idx,
        }
        json_list.append(json_obj)
        block_list_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'initialise',
            'name': 'blocks',
            'value': '[]',
            'output': block_list_idx,
        }
        json_list.append(json_obj)
        json_obj: dict[str, Any] = {
            'method': 'append_list',
            'list': 'json_list_' + str(block_list_idx),
            'value': 'json_list_' + str(diag_idx),
            'output': len(json_list),
        }
        json_list.append(json_obj)
        block_list_idx = len(json_list) - 1

    mat_total_size = torch.tensor(list(initial_shape) + [num+SymExpSparse.count])
    mat = SparseTensor([start_index], [mat], len(initial_shape)+1, mat_total_size)
    if trace:
        mat_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'SparseTensor',
            'start_indices': [start_index.tolist()],
            'blocks': 'json_list_' + str(block_list_idx),
            'dims': len(initial_shape)+1,
            'total_size': mat_total_size.tolist(),
            'end_indices': [mat.end_indices[0].tolist()],
            'type': mat.type.__name__,
            'dense_const': mat.dense_const,
            'output': mat_idx,
        }
        json_list.append(json_obj)

    if network.no_sparsity:
        if trace:
            dense_mat, dense_mat_idx = mat.blocks[0].get_dense(json_list=json_list, template_index=diag_idx, simulacrum=True)
        else:
            dense_mat = mat.blocks[0].get_dense()
        mat.blocks[0] = DenseBlock(dense_mat)
        mat.end_indices[0] = start_index + torch.tensor(dense_mat.shape)
        if trace:
            dense_block_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'DenseBlock',
                'block': 'json_list_' + str(dense_mat_idx),
                'output': dense_block_idx,
            }
            json_list.append(json_obj)
            block_list_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'initialise',
                'name': 'blocks',
                'value': '[]',
                'output': block_list_idx,
            }
            json_list.append(json_obj)
            json_obj: dict[str, Any] = {
                'method': 'append_list',
                'list': 'json_list_' + str(block_list_idx),
                'value': 'json_list_' + str(dense_block_idx),
                'output': len(json_list),
            }
            json_list.append(json_obj)
            block_list_idx = len(json_list) - 1
            mat_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'SparseTensor',
                'start_indices': [start_index.tolist()],
                'blocks': 'json_list_' + str(block_list_idx),
                'dims': len(initial_shape)+1,
                'total_size': mat_total_size.tolist(),
                'end_indices': [mat.end_indices[0].tolist()],
                'type': mat.type.__name__,
                'dense_const': mat.dense_const,
                'output': mat_idx,
            }
            json_list.append(json_obj)

    if trace:
        new_eps_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'new_eps',
            'mat': 'json_list_' + str(mat_idx),
            'const': 'json_list_' + str(const_idx),
            'output': new_eps_idx,
        }
        json_list.append(json_obj)

    SymExpSparse.count += num
    if owns_capture:
        write_jit_capture_file(
            'jit_new_eps',
            'new_eps',
            layer_index,
            counter,
            inside_while,
            while_number,
            while_iteration,
            json_list
        )
    return SymExpSparse(network, mat, const)

class SymExpSparse:
    count = 0
    def __init__(self, network, mat = None, const = 0.0):
        if SymExpSparse.count < get_num_eps(mat) :
            SymExpSparse.count = get_num_eps(mat)
        self.mat = mat
        self.const = const
        self.network = network
        if mat==None:
            if isinstance(const, SparseTensor):
                self.mat = SparseTensor([], [], const.dims+1, torch.tensor(list(const.total_size) + [SymExpSparse.count]))
            

    def expand_mat(self):
        assert(self.mat.dense_const==0)
        self.mat.total_size[-1] = SymExpSparse.count

    def get_mat(self, sym_size, json_list=None, layer_index=None,
                counter=None, inside_while=False, while_number=None,
                while_iteration=None, lhs_index=-1):
        owns_capture = (json_list is None) and dummy_mode
        if owns_capture:
            json_list = [{"method": "noop", "input": "lhs", "output": 0}]
            lhs_index = 0
        trace = json_list is not None
        if not trace:
            json_list = []
        if trace and not owns_capture:
            assert lhs_index != -1

        if self.mat == None:
            if trace:
                st_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'SparseTensor',
                    'start_indices': [],
                    'blocks': [],
                    'dims': 0,
                    'total_size': [],
                    'output': st_idx,
                }
                json_list.append(json_obj)
            if owns_capture:
                write_jit_capture_file(
                    'jit_poly_exp_sparse_get_mat',
                    'poly_exp_sparse_get_mat',
                    layer_index,
                    counter,
                    inside_while,
                    while_number,
                    while_iteration,
                    json_list
                )
            return SparseTensor([], [], 0, torch.tensor([]))

        self.expand_mat()
        if trace:
            mat_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'get_sym_exp_sparse_mat',
                'input': 'json_list_' + str(lhs_index),
                'output': mat_idx,
            }
            json_list.append(json_obj)
            expand_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'expand_symexp_mat',
                'input': 'json_list_' + str(mat_idx),
                'output': expand_idx,
            }
            json_list.append(json_obj)
        if owns_capture:
            write_jit_capture_file(
                'jit_poly_exp_sparse_get_mat',
                'poly_exp_sparse_get_mat',
                layer_index,
                counter,
                inside_while,
                while_number,
                while_iteration,
                json_list
            )
        return self.mat

    def get_const(self):
        return self.const
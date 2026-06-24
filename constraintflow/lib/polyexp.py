import torch 
from constraintflow.gbcsr.sparse_tensor import *
from typing import Any
 

class PolyExpSparse:
    def __init__(self, network, mat, const):
        self.network = network
        self.mat = mat 
        self.const = const
        if not isinstance(self.const, SparseTensor):
            if isinstance(self.const, torch.Tensor):
                self.const = SparseTensor([torch.tensor([0]*self.const.dim())], [SparseBlock(self.const)], self.const.dim(), torch.tensor(self.const.shape))

    def copy(self):
        if isinstance(self.mat, SparseTensor):
            new_mat = self.mat.copy()
        elif isinstance(self.mat, torch.Tensor):
            new_mat = self.mat.clone()
        else:
            new_mat = self.mat

        if isinstance(self.const, SparseTensor):
            new_const = self.const.copy()
        elif isinstance(self.const, torch.Tensor):
            new_const = self.const.clone()
        else:
            new_const = self.const

        return PolyExpSparse(self.network, new_mat, new_const)

    def get_mat(self, abs_elem, dense=False, json_list=None, layer_index=None,
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
        if isinstance(self.mat, float):
            if trace:
                mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'get_poly_exp_sparse_mat',
                    'input': 'json_list_' + str(lhs_index),
                    'output': mat_idx,
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
        if dense:
            block = self.mat.get_dense()
            sp_mat_idx = -1
            if trace:
                mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'get_poly_exp_sparse_mat',
                    'input': 'json_list_' + str(lhs_index),
                    'output': mat_idx,
                }
                json_list.append(json_obj)
                sb_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'SparseBlock',
                    'block': block.tolist(),
                    'output': sb_idx,
                }
                json_list.append(json_obj)
                block_list_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'initialise',
                    'name': 'get_dense_sb',
                    'value': '[]',
                    'output': block_list_idx,
                }
                json_list.append(json_obj)
                appended_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'append_list',
                    'list': 'json_list_' + str(block_list_idx),
                    'value': 'json_list_' + str(sb_idx),
                    'output': appended_idx,
                }
                json_list.append(json_obj)
                sp_mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'SparseTensor',
                    'start_indices': [torch.tensor([0]*block.dim()).tolist()],
                    'blocks': 'json_list_' + str(appended_idx),
                    'dims': block.dim(),
                    'total_size': block.shape,
                    'output': sp_mat_idx,
                }
                json_list.append(json_obj)
            sp_mat = SparseTensor([torch.tensor([0]*block.dim())], [SparseBlock(block)], block.dim(), torch.tensor(block.shape))
        else:
            sp_mat_idx = -1
            if trace:
                sp_mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'get_poly_exp_sparse_mat',
                    'input': 'json_list_' + str(lhs_index),
                    'output': sp_mat_idx,
                }
                json_list.append(json_obj)
            sp_mat = self.mat
        start, end = torch.nonzero(abs_elem.d['llist']).flatten().tolist()[0], torch.nonzero(abs_elem.d['llist']).flatten().tolist()[-1]
        start, end = self.network[start].start, self.network[end].end
        start_index = torch.zeros(sp_mat.dims, dtype=torch.int64)
        end_index = sp_mat.total_size
        start_index[-1] = start
        end_index[-1] = end
        sp_mat_gscr = sp_mat.get_sparse_custom_range(start_index, end_index, json_list=json_list if trace else None, layer_index=layer_index, counter=counter, inside_while=inside_while, while_number=while_number, while_iteration=while_iteration, lhs_index=sp_mat_idx)
        if dummy_mode:
            sp_mat_gscr, _ = sp_mat_gscr
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
        return sp_mat_gscr
        
    def get_const(self):
        return self.const
    
    def get_dense_layers(self):
        layer = 0
        dense_layers = set()
        for j, i in enumerate(self.mat.start_indices):
            while(True):
                if self.network[layer].start<=i[-1]:
                    break
                layer+=1
            
            while(True):
                dense_layers.add(layer)
                if self.network[layer].start<=self.mat.end_indices[j][-1]:
                    break
                layer+=1
        return list(dense_layers)
    
    def create_similar(self, network=None, mat=None, const=None):
        if network == None:
            network = self.network
        if mat == None:
            mat = self.mat
        if const == None:
            const = self.const
        return PolyExpSparse(network, mat, const)

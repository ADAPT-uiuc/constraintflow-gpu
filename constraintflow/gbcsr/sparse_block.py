from numpy import block
import copy
import torch 
import torch.nn.functional as F
import time
import operator
from constraintflow.gbcsr.op_helper import *
from constraintflow.lib.globals import *

def meta_tensor(block_shape, dtype=None):
    return torch.empty(block_shape, device='meta', dtype=dtype)

def _sync():
    device_mode.sync()

def get_slice(start_index, end_index):
    dims = start_index.shape[0]
    s = []
    for j in range(dims):
        s.append(slice(int(start_index[j]), int(end_index[j])))
    return s

def get_diagonal(block, diag_index, json_list = None, template_index = None):
    if json_list is not None:
        json_obj = {
            "method": "torch_diagonal",
            "input": "json_list_" + str(template_index),
            "dim1": diag_index-1,
            "dim2": diag_index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
    res = block.diagonal(dim1=diag_index-1, dim2=diag_index)
    permutation = list(range(len(res.shape)))
    last_index = permutation[-1]
    permutation = permutation[:-1]
    permutation.insert(diag_index-1, last_index)
    if json_list is not None:
        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(len(json_list) - 1),
            "permutation": permutation,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return res.permute(permutation), len(json_list) - 1
    return res.permute(permutation)

def operation(x, y, op):
    start_time = time.perf_counter()
    # if baseline_gpu_mode:
    #    torch.cuda.synchronize()
    # print(f'{type(x)} {type(y)} {x.device} {y.device}')
    z = op(x, y)
    # if baseline_gpu_mode:
        # torch.cuda.synchronize()
    binary_profilier.update_total_time(time.perf_counter() - start_time)
    return z

def unary_operation(x, op, json_list = None, block_index = -1, unary_index = -1):
    start_time = time.perf_counter()
    #_sync()
    start_op_time = time.perf_counter()
    if op == 'sigma':
        if json_list is not None:
            json_list.append({
                "method": "torch_sigmoid",
                "block": "json_list_" + str(block_index),
                "output" : len(json_list)
            })
        z = torch.sigmoid(x)
    else:
        if json_list is not None:
            json_list.append({
                "method": "simple_unary",
                "input": "json_list_" + str(block_index),
                "operation": "json_list_" + str(unary_index),
                "output" : len(json_list)
            })
        z = op(x)
    #_sync()
    unary_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
    unary_profilier.update_total_time(time.perf_counter() - start_time)
    if json_list is not None:
        return z, len(json_list) - 1
    return z

def where_block(x, y, z):
    start_time = time.time()
    w = torch.where(x, y, z)
    where_time.update_op_time(time.time() - start_time)
    return w

class SparseBlock:
    repeat_dims = []
    def __init__(self, block, total_shape, block_type='D'):
        self.total_shape = total_shape
        self.block_type = block_type
        if block_type != 'C' and dummy_mode:
            block_shape = block.shape if isinstance(block, torch.Tensor) else self.total_shape
            if isinstance(block, bool) or (isinstance(block, torch.Tensor) and block.dtype == torch.bool):
                block_dtype = torch.bool
            else:
                block_dtype = torch.float
            self.block = meta_tensor(block_shape, dtype=block_dtype)
            return

        if isinstance(block, torch.Tensor) and not block.is_meta:
            start_transfer = time.perf_counter()

            block = block.to(device_mode.get_device())
            binary_profilier.update_data_transfer_time(time.perf_counter() - start_transfer)
        if isinstance(block, bool) or (isinstance(block, torch.Tensor) and block.dtype == torch.bool):
            self.block = block 
        else:
            if isinstance(block, torch.Tensor):
                self.block = block.type(torch.float)
            else:
                self.block = float(block)
    def copy(self):
        new_block = self.block 
        if isinstance(new_block, torch.Tensor):
            new_block = new_block.clone()
        return self.create_similar(new_block)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        copied = self.copy()
        if memo is not None:
            memo[id(self)] = copied
        return copied

    def get_dense(self):
        raise Exception(f'Not implemented for {type(self)}')
    
    def repeat(self):
        raise Exception(f'Not implemented for {type(self)}')
    
    def unsqueeze(self, index):
        raise Exception(f'Not implemented for {type(self)}')
    
    def squeeze(self, index):
        raise Exception(f'Not implemented for {type(self)}')
    
    def matmul_equal_dims(self, sp_block):
        raise Exception(f'Not implemented for {type(self)}')
    
    def matmul_unequal_dims(self, sp_block):
        raise Exception(f'Not implemented for {type(self)}')
    
    def binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        # assert((self.total_shape == sp_block.total_shape).all())
        start_time = time.perf_counter()
        if isinstance(sp_block, ConstBlock):
            if sp_block.block == identity_element(op):
                json_obj = {
                    "method": "noop",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return self, len(json_list) - 1
            elif sp_block.block == annihilator_element(op):
                json_obj = {
                    "method": "noop",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return sp_block, len(json_list) - 1
            if op(0, sp_block.block) in [0, False]:
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1

                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1

                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                new_block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=new_block, json_list=json_list, template_index=binary_index, simulacrum=True)
                return res, res_index 
        if op in disjunction_ops:
            res, res_index = self.disjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index 
        elif op in conjunction_ops:
            res, res_index = self.conjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index 
        else:
            # assert(False)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            pass
    
    def disjunctive_binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        if isinstance(self, type(sp_block)):
            if not isinstance(self, (KernelBlock, PatchesBlock)) or self.parameters() == sp_block.parameters():
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=block, json_list=json_list, template_index=binary_index, simulacrum=True)
                return res, res_index 
                
            block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = operation(block_1, block_2, op)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(len(json_list) - 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            return DenseBlock(block), len(json_list) - 1
        elif not isinstance(self, type(sp_block)):
            block_1 = self 
            block_2 = sp_block
            if isinstance(self, KernelBlock):
                block_1, lhs_index = self.convert_to_patches(json_list=json_list, index=lhs_index,  simulacrum=True)
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
            if isinstance(self, ConstBlock):
                self_block, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
                block_1 = DenseBlock(self_block)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
            if isinstance(sp_block, KernelBlock):
                block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, template_index=rhs_index)
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
            if isinstance(sp_block, ConstBlock):
                block_temp, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
                block_2 = DenseBlock(block_temp)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
        block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
        block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
        json_obj = {
            "method": "torch_binary",
            "lhs": "json_list_" + str(lhs_index),
            "rhs": "json_list_" + str(rhs_index),
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        block = operation(block_1, block_2, op)
        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return DenseBlock(block), len(json_list) - 1
            
    def conjunctive_binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        if isinstance(self, type(sp_block)):
            if not isinstance(self, (KernelBlock, PatchesBlock)) or self.parameters() == sp_block.parameters():
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=block, json_list=json_list, template_index=binary_index,  simulacrum=True)
                return res, res_index 
            block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = operation(block_1, block_2, op)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(len(json_list) - 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            return DenseBlock(block), len(json_list) - 1 
        elif not isinstance(self, type(sp_block)):
            block_1 = self 
            block_2 = sp_block
            if isinstance(self, KernelBlock):
                block_1, lhs_index = self.convert_to_patches(json_list=json_list, template_index=lhs_index)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            if isinstance(self, ConstBlock):
                self_block, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)

                block_1 = DenseBlock(self_block)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=len(json_list)-1, rhs_index=rhs_index)
            if isinstance(sp_block, KernelBlock):
                block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, index=rhs_index)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            if isinstance(sp_block, ConstBlock):
                sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=lhs_index, rhs_index=len(json_list)-1)
        if isinstance(self, DiagonalBlock) and isinstance(sp_block, DenseBlock):
            block_1 = self.block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1

            block_2 = sp_block.block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1

            block_2, rhs_index = get_diagonal(block_2, self.diag_index, json_list=json_list, template_index=rhs_index)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = operation(block_1, block_2, op)
            res, res_index = self.create_similar(block=block, json_list=json_list, template_index=len(json_list)-1,  simulacrum=True)
            return res, res_index 
        elif isinstance(sp_block, DiagonalBlock) and isinstance(self, DenseBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_1, lhs_index = get_diagonal(self.block, sp_block.diag_index, json_list=json_list, template_index=len(json_list)-1)
            block_2 = sp_block.block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)

            block = operation(block_1, block_2, op)
            res, res_index =  sp_block.create_similar(block=block, json_list=json_list, template_index=len(json_list)-1,  simulacrum=True)
            return res, res_index 
        block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
        block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
        block = operation(block_1, block_2, op)
        json_obj = {
            "method": "torch_binary",
            "lhs": "json_list_" + str(lhs_index),
            "rhs": "json_list_" + str(rhs_index),
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return DenseBlock(block), len(json_list) - 1 



    def unary(self, op):
        if isinstance(self.block, torch.Tensor) and self.block.dtype == bool or isinstance(self.block, bool):
            # assert(op == operator.not_)
            pass
        else:
            # assert(op == operator.neg)
            pass
        return self.create_similar(unary_operation(self.block, op))
    
    def float(self):
        return self.create_similar((self.block).float())
    
    def sum(self, dim):
        raise Exception(f'Not implemented for {type(self)}')
    
    def size(self):
        raise Exception(f'Not implemented for {type(self)}')
    
    def clamp(self, const, min_true):
        start_time_total = time.perf_counter()
        # _sync()
        if min_true:
            clamp_sparse_block_expense.update_total_time(time.perf_counter() - start_time_total)
            start_op_time = time.perf_counter()
            new_block = self.block.clamp(min=const)
            clamp_sparse_block_op_time.update_total_time(time.perf_counter() - start_op_time)
        else:
            clamp_sparse_block_expense.update_total_time(time.perf_counter() - start_time_total)
            start_op_time = time.perf_counter()
            new_block = self.block.clamp(max=const)
            clamp_sparse_block_op_time.update_total_time(time.perf_counter() - start_op_time)
        start_time = time.perf_counter()
        
        # _sync()
        res = self.create_similar(block=new_block)
        clamp_sparse_block_expense.update_total_time(time.perf_counter() - start_time)
        return res

    def type(self):
        if isinstance(self, DiagonalBlock):
            return 'Diagonal'
        if isinstance(self, KernelBlock):
            return 'Kernel'
        if isinstance(self, DenseBlock):
            return 'Dense'
        
    def overwrite_dense_block(self, sp_block, start_index, s):
        raise Exception(f'Not implemented')
    
    def any(self):
        start_time = time.time()
        if dummy_mode:
            return True
        res = self.block.any()
        any_time.update_op_time(time.time() - start_time)
        return res


class DenseBlock(SparseBlock):
    def __init__(self, block):
        total_shape = torch.tensor(block.shape)
        super().__init__(block, total_shape, 'D')
        self.batch_size = total_shape[0]
        # if isinstance(self.batch_size, torch.Tensor):
        #     self.batch_size = self.batch_size.item()

    def get_dense(self, json_list=[], template_index=-1, simulacrum=False):
        json_list.append({"method": "extract_sparse_block", 
            "input": "json_list_" + str(template_index), 
            "output": len(json_list)
        })
        if simulacrum:
            return self.block, len(json_list) - 1
        return self.block
    
    def repeat(self, repeat_dims):
        return RepeatBlock(self.block, self.total_shape*repeat_dims)
        start_time = time.time()
        expand_dims = torch.tensor(self.block.shape) * repeat_dims
        new_block = self.block.expand(*expand_dims)
        repeat_time.update_op_time(time.time() - start_time)
        res = DenseBlock(new_block)
        # res = DenseBlock(self.block.repeat(*repeat_dims))
        return res
    
    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        start_time = time.time()
        json_obj = {
            "method": "extract_sparse_block",
            "input": "json_list_" + str(template_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1

        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(template_index),
            "index": index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        res = self.block.unsqueeze(index)

        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(res_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1

        if simulacrum:
            return DenseBlock(res), res_index
        end_time = time.time()
        unsqueeze_time.update_op_time(end_time-start_time)
        return DenseBlock(res)
    
    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        start_time = time.time()


        json_obj = {
            "method": "extract_sparse_block",
            "input": "json_list_" + str(template_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_squeeze",
            "input": "json_list_" + str(template_index),
            "index": index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        new_block = self.block.squeeze(index)

        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(res_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        res = DenseBlock(new_block)
        if simulacrum:
            return res, res_index
        end_time = time.time()
        squeeze_time.update_op_time(end_time - start_time)
        return res
    
    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block
            
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = sp_block.block

            
            json_obj = {
                    "method": "torch_matmul",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "output": len(json_list),
            }
            json_list.append(json_obj)
            c = a @ b


            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(len(json_list) - 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = DenseBlock(c)
        elif isinstance(sp_block, DiagonalBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            lhs = self.block

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            rhs = sp_block.block
            
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(rhs_index),
                "index": sp_block.diag_index-1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            rhs = rhs.unsqueeze(sp_block.diag_index-1)
            
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = lhs * rhs

            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(len(json_list) - 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = DenseBlock(res)

        elif isinstance(sp_block, KernelBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            kernel = sp_block.block

            kx = sp_block.kx
            ky = sp_block.ky
            sx = sp_block.sx
            sy = sp_block.sy
            px = sp_block.px
            py = sp_block.py
            ox = sp_block.ox
            oy = sp_block.oy
            ix = sp_block.ix
            iy = sp_block.iy
            num_kernels = sp_block.num_kernels
            batch_size = self.batch_size
            curr_size = self.block.shape[-2]
            new_px = (ix + 2*px - kx) % sx
            new_py = (iy + 2*py - ky) % sy

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(lhs_index),
                "shape": (batch_size*curr_size, num_kernels, ox, oy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            input_tensor = self.block.reshape(batch_size*curr_size, num_kernels, ox, oy)

            json_obj = {
                "method": "F.conv_transpose2d",
                "input": "json_list_" + str(lhs_index),
                "kernel": "json_list_" + str(rhs_index),
                "stride": (sx, sy),
                "padding": (px, py),
                "output_padding": (new_px, new_py),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            output_tensor = F.conv_transpose2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py))

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (batch_size, curr_size, -1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(res_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = DenseBlock(output_tensor.reshape(batch_size, curr_size, -1))
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]

                json_obj = {
                    "method": "ConstBlock",
                    "block": 0,
                    "total_shape": new_total_shape.tolist(),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res_index = len(json_list) - 1
                res = ConstBlock(0, new_total_shape)
                
            else:
                raise NotImplementedError
        else:
            start_op_time = time.perf_counter()
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)


            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            lhs = self.block


            json_obj = {
                "method": "torch_matmul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c = lhs @ block_2

            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(len(json_list) - 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = DenseBlock(c)
        return res, res_index
        
    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        if isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block 

    


            rhs_block, rhs_index = sp_block.unsqueeze(-1, json_list=json_list, template_index=rhs_index, simulacrum=True)

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = rhs_block.block
            

            json_obj = {
                "method": "torch_matmul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c = a @ b

            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(len(json_list) - 1),
                "index": -1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = (c).squeeze(-1)


            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(res_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = DenseBlock(res)
        elif isinstance(sp_block, DiagonalBlock):
            raise NotImplementedError
        elif isinstance(sp_block, KernelBlock):
            raise NotImplementedError
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]
                json_obj = {
                    "method": "ConstBlock",
                    "block": 0,
                    "total_shape": new_total_shape.tolist(),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res_index = len(json_list) - 1
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        elif isinstance(sp_block, RepeatBlock):
            # warnings.warn(f'Matmul with unequal dims inefficient for {type(self)} and {type(sp_block)}')

            sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)

            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            sp_block = DenseBlock(sp_block)

            res, res_index = self.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            return res, res_index
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        return res, res_index
        
    
    def convert_to_patches(self, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        raise Exception(f'Not an efficient implementation')
        flag = self.block.dtype == torch.bool 
        if flag:
            block = self.block.float() # batch_size, curr_size, prev_size
        else:
            block = self.block # batch_size, curr_size, prev_size
        batch_size, curr_size, prev_size = block.shape
        block = block.reshape(block.size(0) * block.size(1), num_channels, ix, iy) # batch_size*curr_size, num_channels, ix, iy
        block = F.unfold(block, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy)) # batch_size*curr_size, num_channels * kx * ky, ox * oy
        block = block.reshape(batch_size, curr_size, *block.shape[-2:]) # batch_size, curr_size, num_channels * kx * ky, ox * oy
        block = block.repeat(1, 1, 1, num_kernels) # batch_size, curr_size, num_channels * kx * ky, num_kernels * ox * oy
        block = block.permute(0, 1, 3, 2) # batch_size, curr_size, curr_size, num_channels * kx * ky
        block = get_diagonal(block, 2) # batch_size, curr_size, num_channels * kx * ky   
        if flag:
            block = block.bool()
        end_time = time.time()
        mat_to_patches_time.update_total_time(end_time - start_time)
        return PatchesBlock(block, self.total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)
        
    
        
    def sum(self, dim):
        return DenseBlock(self.block.sum(dim))
    
    def size(self):
        return self.block.size()
        
    def overwrite_dense_block(self, sp_block, start_index, s):
        new_block = self.block 
        new_block[s] = sp_block.block
        return [DenseBlock(new_block)], [torch.tensor([0]*len(self.total_shape))]
        
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index):
        s = get_slice(start_index - block_start_index, end_index - block_start_index)
        return DenseBlock(self.block[tuple(s)])
    
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(template_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        if simulacrum:
            return DenseBlock(block), len(json_list) - 1
        return DenseBlock(block)

class KernelBlock(SparseBlock):
    def __new__(cls, block, total_shape, ix, iy, ox, oy, sx, sy, px, py):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, KernelBlock.__init__, block, total_shape, ix, iy, ox, oy, sx, sy, px, py)
        return super().__new__(cls)

    def __init__(self, block, total_shape, ix, iy, ox, oy, sx, sy, px, py):
        super().__init__(block, total_shape, 'K')
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py
        self.kx = block.shape[-2]
        self.ky = block.shape[-1]
        self.num_channels = block.shape[1]
        self.num_kernels = block.shape[0]

    def __str__(self):
        res = f'KernelBlock: \n \
            Total Shape: {self.total_shape} \n \
            Block Shape: {self.block.shape} \n \
            Num Kernels: {self.num_kernels} \n \
            Num Channels: {self.num_channels} \n \
            Kernel Shape: {self.block.shape[-2:]} \n \
            Stride: {self.sx, self.sy} \n \
            Padding: {self.px, self.py} \n \
            Input Size: {self.ix, self.iy} \n \
            Output Size: {self.ox, self.oy} \n'
        return res
    
    def parameters(self):
        return [self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels]

    def get_dense(self):
        kernel = self.block.float()
        px = self.px
        py = self.py
        sx = self.sx
        sy = self.sy
        ix = self.ix
        iy = self.iy
        kx = self.kx
        ky = self.ky
        ox = self.ox
        oy = self.oy
        num_kernels = self.num_kernels
        new_px = (ix + 2*px - kx) % sx
        new_py = (iy + 2*py - ky) % sy
        curr_size = num_kernels*ox*oy
        eye = torch.eye(num_kernels*ox*oy).unsqueeze(0).reshape(curr_size, num_kernels, ox, oy)
        res = F.conv_transpose2d(eye, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py)).reshape(1, curr_size, -1)
        return res
    
    def repeat(self, repeat_dims):
        res = KernelBlock(self.block, self.total_shape*repeat_dims, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py)
        return res

    def unsqueeze(self, index):
        return KernelBlock(self.block, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py)
    
    def squeeze(self, index):
        return KernelBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py)
    

    
    def matmul_equal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DiagonalBlock):
            # AUTO-LIRPA
            kernel = self.block
            px = self.px
            py = self.py
            sx = self.sx
            sy = self.sy
            ix = self.ix
            iy = self.iy
            ox = self.ox
            oy = self.oy
            kx = self.kx
            ky = self.ky
            num_kernels = self.num_kernels
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)
            
            start_op_time = time.perf_counter()
            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy)) # batch_size, num_channels * kx * ky, ox * oy
            x_unf = x_unf.permute(0,2,1).repeat(1, num_kernels, 1)
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            
            
            k_new = self.convert_to_patches().block
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            start_op_time = time.perf_counter()
            patches = x_unf * k_new
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        elif isinstance(sp_block, DenseBlock):
            b = sp_block.block # batch_size, prev_size, sym_size
            batch_size = b.shape[0]
            prev_size = b.shape[1]
            sym_size = b.shape[2]
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            start_op_time = time.perf_counter()
            b = b.transpose(1,2) # batch_size, sym_size, prev_size
            b = b.reshape(b.shape[0]*b.shape[1], self.num_channels, self.ix, self.iy)
            res = F.conv2d(b, self.block, stride=(self.sx, self.sy), padding=(self.px, self.py)) # batch_size*sym_size, num_kernels, ox, oy
            res = res.reshape(batch_size, sym_size, -1)
            res = res.transpose(1,2) # batch_size, curr_size, sym_size
            
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(res)
        elif isinstance(sp_block, PatchesBlock):
            d_block = DenseBlock(sp_block.get_dense())
            res = self.matmul_equal_dims(d_block)
            return res
        else:
            temp = self.convert_to_patches()
            res = temp.matmul_equal_dims(sp_block)
            return res
        equal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res


    def compute_patches_stride_padding(self, patches_padding, patches_stride, op_padding, op_stride):
        """
        Compute stride and padding after a conv layer with patches mode.
        """
        for p in (patches_padding, patches_stride, op_padding, op_stride):
            # assert isinstance(p, int) or (isinstance(p, (list, tuple)) and (len(p) == 2 or len(p) == 4))
            pass
        # If p is int, then same padding on all 4 sides.
        # If p is 2-tuple, then it is padding p[0] on both sides of H, p[1] on both sides of W
        # If p is 4-tuple, then it is padding p[2], p[3] on top and bottom sides of H, p[0] and p[1] on left and right sides of W

        # If any of the inputs are not tuple/list, we convert them to tuple.
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p, p) if isinstance(p, int) else p for p in [patches_padding, op_padding, patches_stride, op_stride]]
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p[1], p[1], p[0], p[0]) if len(p) == 2 else p for p in [full_patch_padding, full_op_padding, full_patch_stride, full_op_stride]]
        # Compute the new padding and stride after this layer.
        new_padding = tuple(pp * os + op  for pp, op, os in zip(full_patch_padding, full_op_padding, full_op_stride))
        new_stride = tuple(ps * os for ps, os in zip(full_patch_stride, full_op_stride))

        return new_padding, new_stride
    
    def matmul_unequal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            kernel = self.block
            sx = self.sx
            sy = self.sy
            px = self.px
            py = self.py
            ix = self.ix
            iy = self.iy
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)
            
            start_op_time = time.perf_counter()
            input_tensor = sp_block.block.reshape(batch_size, num_channels, ix, iy)
            
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            block = F.conv2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py))
            

            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            block = block.reshape(batch_size, -1)
            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(block)
        elif isinstance(sp_block, DiagonalBlock):
            raise NotImplementedError
        elif isinstance(sp_block, KernelBlock):
            raise NotImplementedError
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res
    
    def convert_to_patches(self):
        num_kernels = self.num_kernels
        ox = self.ox
        oy = self.oy
        patches = self.block.view(1, num_kernels, -1).unsqueeze(-1).repeat(1, 1, 1, ox*oy)
        patches = patches.permute(0,1,3,2).reshape(patches.size(0),num_kernels*ox*oy, -1)
        return PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)

        # The following doesnt work for some reason
        patches = self.block.view(1, num_kernels, -1).repeat(1, ox*oy, 1)
        return PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)


    
    def create_similar(self, block):
        return KernelBlock(block, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py)

class DiagonalBlock(SparseBlock):
    def __new__(cls, block, total_shape, diag_index):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, DiagonalBlock.__init__, block, total_shape, diag_index)
        return super().__new__(cls)

    def __init__(self, block, total_shape, diag_index):
        super().__init__(block, total_shape, 'Diag')
        self.diag_index = diag_index
        self.batch_size = self.total_shape[0]
        # assert((torch.tensor(block.shape) == torch.concat([total_shape[:diag_index], total_shape[diag_index+1:]])).all())

    def get_dense(self):
        if self.diag_index == len(self.total_shape):
            return torch.diag_embed(self.block)
        
        # Apply diag_embed at the correct dimension
        # We'll permute, apply diag_embed, and permute back
        shape = list(self.block.shape)
        diag_index = self.diag_index-1

        # Step 1: move the diagonal dim to the last
        perm = list(range(len(shape)))
        perm.pop(diag_index)
        perm.append(diag_index)

        # DUSH: On block
        diag_moved = self.block.permute(perm)

        # DUSH : On block
        # Step 2: embed to diagonal matrices (adds 1 dimension at the end)
        diag_expanded = torch.diag_embed(diag_moved)  # last dim becomes (c, c)

        # Step 3: move dims back to restore original ordering + new dimension
        # Insert the extra dim right after diag_index
        new_perm = list(range(len(shape)-1))
        new_perm.insert(diag_index, len(new_perm))
        new_perm.insert(diag_index+1, len(new_perm))
        decompressed = diag_expanded.permute(new_perm)

        return decompressed
    
        return super().get_dense()
    
    def repeat(self, repeat_dims):
        # # assert(repeat_dims[self.diag_index]==1 and repeat_dims[self.diag_index-1]==1)
        start_time = time.time()
        new_repeat_dims = torch.concat([repeat_dims[:self.diag_index], repeat_dims[self.diag_index+1:]])
        expand_dims = torch.tensor(self.block.shape) * new_repeat_dims
        new_block = self.block.expand(*expand_dims)
        repeat_time.update_op_time(time.time() - start_time)
        res = DiagonalBlock(new_block, self.total_shape*repeat_dims, self.diag_index)
        # res = DiagonalBlock(self.block.repeat(*new_repeat_dims), self.total_shape*repeat_dims, self.diag_index)
        return res


    def unsqueeze(self, index):
        # assert(index!=self.diag_index)
        start_time = time.time()
        if index<self.diag_index:
            res = self.block.unsqueeze(index)
            end_time = time.time()
            unsqueeze_time.update_op_time(end_time-start_time)
            return DiagonalBlock(res, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.diag_index+1)
        res = self.block.unsqueeze(index-1)
        end_time = time.time()
        unsqueeze_time.update_op_time(end_time-start_time)
        return DiagonalBlock(res, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.diag_index)
    
    def squeeze(self, index):
        if index==self.diag_index:
            return DenseBlock(self.block)
        elif index<self.diag_index:
            start_time = time.time()
            new_block = self.block.squeeze(index)
            end_time = time.time()
            squeeze_time.update_op_time(end_time - start_time)
            return DiagonalBlock(new_block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.diag_index-1)
        else:
            return DiagonalBlock(self.block.squeeze(index-1), torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.diag_index)
        
    def sum(self, dim):
        if dim<self.diag_index-1:
            return DiagonalBlock(self.block.sum(dim), torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]]), self.diag_index)
        elif dim>self.diag_index:
            return DiagonalBlock(self.block.sum(dim-1), torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]]), self.diag_index)
        else:
            return DenseBlock(self.block)

    def matmul_equal_dims(self, sp_block):
        total_start_time = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            start_op_time = time.perf_counter()
            a = self.block.unsqueeze(self.diag_index)
            b = sp_block.block
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            c = a * b
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(c)
        elif isinstance(sp_block, DiagonalBlock):
            raise NotImplementedError
        elif isinstance(sp_block, KernelBlock):
            block_2 = sp_block.convert_to_patches().block # batch_size, curr_size, num_channels * kx * ky
            start_op_time = time.perf_counter()
            block_1 = self.block.unsqueeze(self.diag_index)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            block = block_1 * block_2
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = PatchesBlock(block, sp_block.total_shape, sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy, sp_block.sx, sp_block.sy, sp_block.px, sp_block.py, sp_block.kx, sp_block.ky, sp_block.num_channels, sp_block.num_kernels)
        elif isinstance(sp_block, RepeatBlock) and (sp_block.repeat_dims==1).all():
            sp_block = DenseBlock(sp_block.get_dense())
            res = self.matmul_equal_dims(sp_block)
            return res
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        equal_matmul_profilier.update_total_time(time.perf_counter() - total_start_time)

        return res
        

    def matmul_unequal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            start_op_time = time.perf_counter()
            a = self.block.unsqueeze(self.diag_index)
            b = sp_block.block.unsqueeze(-1)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            res = a * b
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            res = res.squeeze(-1)
            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(res)
        elif isinstance(sp_block, RepeatBlock) and (sp_block.repeat_dims==1).all():
            sp_block = DenseBlock(sp_block.get_dense())
            res = self.matmul_unequal_dims(sp_block)
            return res
        else:
            res = super().matmul_unequal_dims(sp_block)
        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res
    
    
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index):
        start_index = torch.concat([start_index[:self.diag_index], start_index[self.diag_index+1:]])
        end_index = torch.concat([end_index[:self.diag_index], end_index[self.diag_index+1:]])
        block_start_index = torch.concat([block_start_index[:self.diag_index], block_start_index[self.diag_index+1:]])
        s = get_slice(start_index - block_start_index, end_index - block_start_index)
        block = self.block[tuple(s)]
        return DiagonalBlock(block, torch.concat([torch.tensor(block.shape[:self.diag_index]), torch.tensor(block.shape[self.diag_index-1:])]),  self.diag_index)

    def create_similar(self, block):
        return DiagonalBlock(block, self.total_shape, self.diag_index)    
    
    def binary(self, sp_block, op, *args, **kwargs):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, RepeatBlock):
            if (sp_block.repeat_dims[self.diag_index] > 1) and sp_block.only_one_repeat:
                block_1 = self.block 
                block_2 = sp_block.block.squeeze(self.diag_index)
                block = operation(block_1, block_2, op)
                res = self.create_similar(block)
                binary_block_expenses.update_total_time(time.perf_counter() - start_time_total)
                return res
            elif (sp_block.repeat_dims[self.diag_index-1] > 1) and sp_block.only_one_repeat:
                block_1 = self.block 
                block_2 = sp_block.block.squeeze(self.diag_index-1)
                block = operation(block_1, block_2, op)
                res = self.create_similar(block)
                binary_block_expenses.update_total_time(time.perf_counter() - start_time_total)
                return res
        return super().binary(sp_block, op)
        
class PatchesBlock(SparseBlock):
    def __new__(cls, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, PatchesBlock.__init__, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)
        return super().__new__(cls)

    def __init__(self, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        if block.dtype == torch.bool:
            block = block 
        else:
            block = block.type(torch.float)
        # block.shape = batch, num_kernels*ox*oy, num_channels*kx*ky 
        # self.total_shape = total_shape
        super().__init__(block, total_shape, 'P')
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py
        self.kx = kx
        self.ky = ky
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.batch_size = block.shape[0]

    def parameters(self):
        return [self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels]
        
    def repeat(self, repeat_dims):
        res =  PatchesBlock(self.block, self.total_shape*repeat_dims, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
        return res


    def unsqueeze(self, index):
        return PatchesBlock(self.block, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
    
    def squeeze(self, index):
        return PatchesBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
    

    # AVAL: understand this function
    def compute_patches_stride_padding(self, patches_padding, patches_stride, op_padding, op_stride):
        """
        Compute stride and padding after a conv layer with patches mode.
        """
        for p in (patches_padding, patches_stride, op_padding, op_stride):
            # assert isinstance(p, int) or (isinstance(p, (list, tuple)) and (len(p) == 2 or len(p) == 4))
            pass
        # If p is int, then same padding on all 4 sides.
        # If p is 2-tuple, then it is padding p[0] on both sides of H, p[1] on both sides of W
        # If p is 4-tuple, then it is padding p[2], p[3] on top and bottom sides of H, p[0] and p[1] on left and right sides of W

        # If any of the inputs are not tuple/list, we convert them to tuple.
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p, p) if isinstance(p, int) else p for p in [patches_padding, op_padding, patches_stride, op_stride]]
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p[1], p[1], p[0], p[0]) if len(p) == 2 else p for p in [full_patch_padding, full_op_padding, full_patch_stride, full_op_stride]]
        # Compute the new padding and stride after this layer.
        new_padding = tuple(pp * os + op  for pp, op, os in zip(full_patch_padding, full_op_padding, full_op_stride))
        new_stride = tuple(ps * os for ps, os in zip(full_patch_stride, full_op_stride))

        return new_padding, new_stride




    def matmul_equal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        # AVAL: understand the if case
        if isinstance(sp_block, KernelBlock):
            start_op = time.perf_counter()  
            flattened_patches = self.block.reshape(self.batch_size*self.num_kernels*self.ox*self.oy, self.num_channels, self.kx, self.ky)
            patches = F.conv_transpose2d(flattened_patches, sp_block.block, stride=(sp_block.sx, sp_block.sy))
            kx = patches.shape[-2]
            ky = patches.shape[-1]
            patches = patches.reshape(self.batch_size, self.num_kernels*self.ox*self.oy, -1)
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op)

            # assert(kx == (self.kx-1)*sp_block.sx + sp_block.kx)
            # assert(ky == (self.ky-1)*sp_block.sy + sp_block.ky)
            new_padding, new_stride = self.compute_patches_stride_padding(patches_padding=(self.px, self.py), patches_stride=(self.sx, self.sy), op_padding=(sp_block.px, sp_block.py), op_stride=(sp_block.sx, sp_block.sy))
            new_total_shape = torch.concat([torch.max(self.total_shape[:-2], sp_block.total_shape[:-2]), self.total_shape[-2:-1], sp_block.total_shape[-1:]])
            res = PatchesBlock(patches, new_total_shape, sp_block.ix, sp_block.iy, self.ox, self.oy, new_stride[0], new_stride[1], new_padding[0], new_padding[1], kx, ky, sp_block.num_channels, self.num_kernels)

        elif isinstance(sp_block, DiagonalBlock):
            patches = self.block
            px = self.px
            py = self.py
            sx = self.sx
            sy = self.sy
            ix = self.ix
            iy = self.iy
            kx = self.kx
            ky = self.ky
            num_kernels = self.num_kernels
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)
            start_op_time = time.perf_counter()
            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))
            x_unf = x_unf.transpose(1,2).repeat(1, num_kernels, 1)
            # x_unf = x_unf.unsqueeze(1).repeat(1, num_kernels, 1, 1)
            if patches.shape[0] != batch_size:
                # patches = patches.repeat(batch_size, 1, 1, 1)
                patches = patches.expand(batch_size, patches.size(1), patches.size(2))
            # patches = patches.repeat(batch_size, 1, 1, 1)
            patches = x_unf * patches
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
        else:
            print(type(sp_block))
            raise NotImplementedError

        equal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res

    def matmul_unequal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            patches = self.block
            sx = self.sx
            sy = self.sy
            px = self.px
            py = self.py
            ix = self.ix
            iy = self.iy
            kx = self.kx
            ky = self.ky
            num_channels = self.num_channels
            num_kernels = self.num_kernels
            batch_size = int(sp_block.batch_size)

            start_op_time = time.perf_counter()
            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))
            x_unf = x_unf.transpose(1,2).repeat(1, num_kernels, 1)
            if patches.shape[0] != batch_size:
                patches = patches.expand(batch_size, patches.size(1), patches.size(2))
            
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            patches = x_unf * patches
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            
            
            ret = patches.sum(dim=-1)
            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(ret)
        elif isinstance(sp_block, ConstBlock) and not sp_block.block == 0:
            res = ConstBlock(0, self.total_shape[:-1])
        elif isinstance(sp_block, RepeatBlock):
            # warnings.warn(f'Matmul with unequal dims inefficient for {type(self)} and {type(sp_block)}')
            sp_block = DenseBlock(sp_block.get_dense())
            res = self.matmul_unequal_dims(sp_block)
            return res
        else:
            print(type(sp_block), sp_block.block)
            raise NotImplementedError

        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res
        
    # AVAL: Understand this function
    def get_dense(self):
        """Converting a Patches piece into a full dense matrix."""
        start_time = time.time()
        batch_size = self.total_shape[0]
        output_channel, output_x, output_y = self.num_kernels, self.ox, self.oy
        input_channel, kernel_x, kernel_y = self.num_channels, self.kx, self.ky
        input_x, input_y = self.ix, self.iy
        padding = (self.py, self.py, self.px, self.px)
        stride = self.sx

        # DUSH: On block
        pieces = self.block
        # pieces = self.block.permute(1,0,2,3)
        # pieces = pieces.view(batch_size, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y).transpose(0, 1)
        pieces = pieces.view(-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y)
        if pieces.shape[0] < batch_size:
            # If the batch size of pieces is not equal to the batch size of the total shape, we need to expand it.
            pieces = pieces.expand(batch_size, *pieces.shape[1:])
        # Fix all patches in a full A matrix.
        A_matrix = torch.zeros(batch_size, output_channel, output_x, output_y, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device, dtype=pieces.dtype)
        # Save its orignal stride.
        orig_stride = A_matrix.stride()
        # This is the main trick - we create a *view* of the original matrix, and it contains all sliding windows for the convolution.
        # Since we only created a view (in fact, only metadata of the matrix changed), it should be very efficient.
        matrix_strided = torch.as_strided(A_matrix, [batch_size, output_channel, output_x, output_y, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[4], input_y + padding[0] + padding[1], 1])
        # Now we need to fill the conv kernel parameters into the last three dimensions of matrix_strided.
        first_indices = torch.arange(output_x * output_y, device=pieces.device)
        second_indices = torch.div(first_indices, output_y, rounding_mode="trunc")
        third_indices = torch.fmod(first_indices, output_y)
        # pieces have shape (out_c, batch, out_h, out_w, c, h, w).
        # pieces = pieces.transpose(0, 1)   # pieces has the out_c dimension at the front, need to move it to the second.
        # DUSH: On block
        matrix_strided[:,:,second_indices,third_indices,second_indices,third_indices,:,:,:] = pieces.reshape(*pieces.shape[:2], -1, *pieces.shape[4:])
        A_matrix = A_matrix.view(batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
        A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]
        A_matrix = A_matrix.reshape(A_matrix.shape[0], A_matrix.shape[1], -1)
        # A_matrix = A_matrix.view(batch_size, output_channel*output_x*output_y, input_channel*input_x*input_y)
        end_time = time.time()
        patches_to_mat_time.update_total_time(end_time - start_time)
        if len(A_matrix.shape)!=len(self.total_shape):
            # AVAL: This is a hack to make sure the shape is correct.
            # We need to reshape it to the total shape.
            if (torch.tensor(A_matrix.shape) == self.total_shape[:-1]).all():
                A_matrix = A_matrix.unsqueeze(-1).expand(*self.total_shape)
            else:
                diffdim = -1
                for i in range(len(A_matrix.shape)):
                    if(diffdim == -1 and self.total_shape[i] != A_matrix.shape[i]):
                            diffdim = i
                    if diffdim != -1 and self.total_shape[i+1] != A_matrix.shape[i]:
                        raise NotImplementedError(f'PatchesBlock get_dense: {A_matrix.shape} != {self.total_shape[:-1]}')
                A_matrix = A_matrix.unsqueeze(diffdim).expand(*self.total_shape)
                
        return A_matrix
    
    
    
    def create_similar(self, block):
        return PatchesBlock(block, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)

    
    def sum(self, dim):
        if dim == len(self.total_shape)-1:
            return DenseBlock(self.block.sum(dim))
        else:
            denseblock = self.get_dense()
            return DenseBlock(denseblock.sum(dim))
            raise NotImplementedError
        
    def binary(self, sp_block, op, *args, **kwargs):
        start_time = time.perf_counter()
        if isinstance(sp_block, RepeatBlock):
            if sp_block.only_one_repeat and sp_block.repeat_dims[-1] > 1:
                block_1 = self.block 
                block_2 = sp_block.block.expand(self.block.shape)
                block = operation(block_1, block_2, op)
                res = self.create_similar(block)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res
        return super().binary(sp_block, op)
    
    def total_expand(self):
        new_dims = list(self.total_shape[:-1]) + [(self.block.shape[-1])]
        self.block = self.block.expand(*new_dims)
        return self


class ConstBlock(SparseBlock):
    def __new__(cls, block, total_shape):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, ConstBlock.__init__, block, total_shape)
        return super().__new__(cls)

    def __init__(self, block, total_shape):
        super().__init__(block, total_shape, 'C')
        # assert total_shape.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}

    # Done
    def get_dense(self, json_list=[], template_index=-1, simulacrum = False):
        json_list.append({
            "method": "extract_sparse_block", 
            "input": "json_list_" + str(template_index), 
            "output": len(json_list)
        })
        json_list.append({
            "method": "torch_ones",
            "size": self.total_shape.tolist(),
            "output": len(json_list)
        })
        json_list.append({
            "method": "torch_mul",
            "lhs": "json_list_" + str(len(json_list)-1),
            "rhs": "json_list_" + str(len(json_list)-2),
            "output": len(json_list)
        })
        ret = torch.ones(*self.total_shape.tolist()) * self.block
        if simulacrum:
            return ret, len(json_list) - 1
        return ret
    
    # Done
    def repeat(self, repeat_dims):
        res =  ConstBlock(self.block, self.total_shape*repeat_dims)
        return res

    # Done
    def unsqueeze(self, index):
        return ConstBlock(self.block, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]))
    
    # Done
    def squeeze(self, index):
        return ConstBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]))

    # Done
    def overwrite_dense_block(self, sp_block, start_index, s):
        block_shape = sp_block.total_shape
        if (block_shape == self.total_shape).all():
            return [sp_block], [torch.tensor([0]*len(self.total_shape))]
        elif (block_shape != self.total_shape).float().sum() == 1:
            unequal_dim = torch.nonzero(block_shape - self.total_shape).item()
            if (start_index == 0).all():
                block_1 = sp_block
                start_index_1 = torch.tensor([0]*len(self.total_shape))
                end_index_2 = self.total_shape
                total_shape_2 = self.total_shape.clone()
                total_shape_2[unequal_dim] -= block_shape[unequal_dim]
                start_index_2 = end_index_2 - total_shape_2
                block_2 = ConstBlock(self.block, total_shape_2)
                return [block_1, block_2], [start_index_1, start_index_2]
            elif (start_index + block_shape == self.total_shape).all():
                start_index_1 = torch.tensor([0]*len(self.total_shape))
                end_index_1 = self.total_shape.clone()
                end_index_1[unequal_dim] -= block_shape[unequal_dim]
                block_1 = ConstBlock(self.block, end_index_1)
                end_index_2 = self.total_shape
                start_index_2 = end_index_2 - block_shape
                block_2 = sp_block
                return [block_1, block_2], [start_index_1, start_index_2]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError

    # Done
    def binary(self, sp_block, op, json_list=[], lhs_index = -1, rhs_index=-1):
        start_time = time.perf_counter()
        if self.block == identity_element(op):
            start = time.perf_counter()
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
                "meta": "We are fucked"
            })
            extract_index = len(json_list) - 1
            unary_op, lambda_index = binary_to_identity_unary(op, json_list)

            block, block_index = unary_operation(sp_block.block, unary_op, json_list = json_list, block_index = extract_index, unary_index = lambda_index)
            binary_profilier.update_total_time(time.perf_counter() - start)
            res, res_index = sp_block.create_similar(block, json_list=json_list, template_index=block_index, simulacrum=True)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        elif self.block == annihilator_element(op):
            json_list.append({
                "method": "noop",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            })
            return self, len(json_list) - 1

        elif self.block == 0 and op == operator.truediv:
            json_list.append({
                "method": "noop",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            })
            return self, len(json_list) - 1
        elif op!=operator.truediv and op(self.block, 0) in [0, False]:
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            })
            rhs_index = len(json_list) - 1
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            })
            lhs_index = len(json_list) - 1

            json_list.append({
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            })
            binary_index = len(json_list) - 1
            block = operation(self.block, sp_block.block, op)
            # res = sp_block.create_similar(block)
            res, res_index = sp_block.create_similar(
                            block=block,
                            json_list=json_list,
                            template_index=binary_index, simulacrum = True)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        elif op in disjunction_ops:
            res, res_index = self.disjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        elif op in conjunction_ops:
            res, res_index = self.conjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        else:
            raise NotImplementedError

    # Done
    def any(self):
        return self.block == True

    
    # Done
    def float(self):
        if self.block == False:
            return self.create_similar(0.0)
        elif self.block == True:
            return self.create_similar(1.0)
        else:
            # assert False
            pass
    
    # Done
    def clamp(self, const, min_true):
        start_time = time.perf_counter()
        if min_true:
            if self.block >= const:
                res = self
            else:
                res = ConstBlock(const, self.total_shape)
        else:
            if self.block <= const:
                res = self
            else:
                res = ConstBlock(const, self.total_shape)
        clamp_const_block_expense.update_total_time(time.perf_counter() - start_time)
        return res


    # Done
    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        # start_time = time.perf_counter()
        if self.block == 0:
            new_total_shape = self.total_shape.clone()
            new_total_shape[-1] = sp_block.total_shape[-1]

            json_obj = {
                "method": "ConstBlock",
                "block": 0,
                "total_shape": new_total_shape.tolist(),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = ConstBlock(0, new_total_shape)
        else:
            raise NotImplementedError
        # matmul_time.update_op_time(time.time() - start_time)
        return res, res_index

    # Done
    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        # start_time = time.perf_counter()
        if self.block == 0:
            new_total_shape = self.total_shape.clone()[:-1]

            json_obj = {
                "method": "ConstBlock",
                "block": 0,
                "total_shape": new_total_shape.tolist(),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = ConstBlock(0, new_total_shape)
        else:
            raise NotImplementedError
        # matmul_time.update_op_time(time.time() - start_time) 
        return res, res_index


    # Done        
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index):
        return ConstBlock(self.block, end_index - start_index)
    
    # Done
    def get_patches(self, batch_size, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        block = torch.ones(batch_size, num_kernels*ox*oy, num_channels*kx*ky)*self.block
        return PatchesBlock(block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)

    # Done     
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum = False):
        json_list.append({
            "method": "ConstBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "output": len(json_list)
        })
        if simulacrum:
            return ConstBlock(block, self.total_shape), len(json_list) - 1
        return ConstBlock(block, self.total_shape)
    
    # Done
    def sum(self, dim):
        new_const = self.block * self.total_shape[dim]
        new_total_shape = torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]])
        return ConstBlock(new_const, new_total_shape)
    


class RepeatBlock(SparseBlock):
    def __new__(cls, block, total_shape):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, RepeatBlock.__init__, block, total_shape)
        return super().__new__(cls)

    def __init__(self, block, total_shape):
        super().__init__(block, total_shape, block_type='R')
        # assert total_shape.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
        self.repeat_dims = self.total_shape / torch.tensor(self.block.shape)
        self.only_one_repeat = (self.repeat_dims != 1).sum() == 1

    def __str__(self):
        ret = f'RepeatBlock: {self.block.shape} with total shape {self.total_shape} and repeat dims {self.repeat_dims}'
        return ret

    def get_dense(self):
        # DUSH: On block
        return self.block.expand(*self.total_shape)

    def repeat(self, repeat_dims):
        return RepeatBlock(self.block, self.total_shape*repeat_dims)
    
    def unsqueeze(self, index):
        return RepeatBlock(self.block.unsqueeze(index), torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]))

    def squeeze(self, index):
        start_time = time.time()
        new_block = self.block.squeeze(index)
        end_time = time.time()
        squeeze_time.update_op_time(end_time - start_time)
        return RepeatBlock(new_block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]))
    
    def create_similar(self, block):
        return RepeatBlock(block, self.total_shape) 
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index):
        slice_start = start_index - block_start_index
        slice_end = end_index - block_start_index
        new_total_shape = slice_end - slice_start
        slice_start = torch.where(self.repeat_dims > 1, 0, slice_start)
        slice_end = torch.where(self.repeat_dims > 1, 1, slice_end)
        s = get_slice(slice_start, slice_end)
        b = self.block[tuple(s)]
        res = RepeatBlock(b, new_total_shape)
        return res
    
    def matmul_equal_dims(self, sp_block):
        return DenseBlock(self.get_dense()).matmul_equal_dims(sp_block)
    
    def matmul_unequal_dims(self, sp_block):
        return DenseBlock(self.get_dense()).matmul_unequal_dims(sp_block)
    
    def binary(self, sp_block, op, *args, **kwargs):
        start_time = time.perf_counter()
        if isinstance(sp_block, DiagonalBlock):
            if (self.repeat_dims[sp_block.diag_index] > 1) and self.only_one_repeat:
                block_1 = self.block.squeeze(sp_block.diag_index)
                block_2 = sp_block.block 
                block = operation(block_1, block_2, op)
                res = sp_block.create_similar(block)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res
            elif (self.repeat_dims[sp_block.diag_index-1] > 1) and self.only_one_repeat:
                block_1 = self.block.squeeze(sp_block.diag_index-1)
                block_2 = sp_block.block 
                block = operation(block_1, block_2, op)
                res = sp_block.create_similar(block)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res
        if isinstance(sp_block, PatchesBlock):
            if self.only_one_repeat and self.repeat_dims[-1] > 1 and len(sp_block.total_shape) == 3: #last condition means patches was not unsqueezed
                sp_block = sp_block.total_expand()
                block_1 = self.block.expand(sp_block.block.shape)
                block_2 = sp_block.block 
                block = operation(block_1, block_2, op)
                res = sp_block.create_similar(block)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res
        if isinstance(sp_block, KernelBlock):
            if self.only_one_repeat and self.repeat_dims[-1] > 1:
                block = sp_block.convert_to_patches()
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return self.binary(block, op)
        
        block_1 = DenseBlock(self.get_dense())
        binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
        return block_1.binary(sp_block, op)
    
    def unary(self, op):
        return self.create_similar(unary_operation(self.block, op))
    
    def any(self):
        if isinstance(self.block, torch.Tensor) and self.block.is_meta:
            return True
        return self.block.any()
    
    def clamp(self, const, min_true):
        start_total_time = time.perf_counter()
        # _sync()
        clamp_repeat_block_expense.update_total_time(time.perf_counter() - start_total_time)
        if min_true:
            start_time = time.perf_counter()
            new_block = self.block.clamp(min=const)
            clamp_repeat_block_op_time.update_op_time(time.perf_counter() - start_time)
        else:
            start_time = time.perf_counter()
            new_block = self.block.clamp(max=const)
            clamp_repeat_block_op_time.update_op_time(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        # _sync()
        clamp_repeat_block_expense.update_total_time(time.perf_counter() - start_time)
        return RepeatBlock(new_block, self.total_shape)


DENSE_DEFAULT_COERCED_BLOCK_TYPES = (
    KernelBlock,
    DiagonalBlock,
    PatchesBlock,
    RepeatBlock,
)

def _should_coerce_to_dense(cls):
    return dense_default_mode and cls in DENSE_DEFAULT_COERCED_BLOCK_TYPES

def _coerce_to_dense_block(cls, init_fn, *args, **kwargs):
    tmp = object.__new__(cls)
    init_fn(tmp, *args, **kwargs)
    return DenseBlock(tmp.get_dense())



def sp_where_block(x: SparseBlock, y: SparseBlock, z: SparseBlock, dummy: bool=False):
    # if dummy_mode or dummy:
    #     res = copy.copy(y)
    #     if isinstance(y.total_shape, torch.Tensor):
    #         res.total_shape = y.total_shape.clone()
    #     else:
    #         res.total_shape = copy.copy(y.total_shape)
    #     if not isinstance(res, ConstBlock):
    #         if isinstance(res.block, torch.Tensor):
    #             res.block = meta_tensor(res.block.shape)
    #         else:
    #             res.block = meta_tensor(res.total_shape)
    #     return res

    if isinstance(x, ConstBlock):
        if x.block:
            return y
        else:
            return z
    elif isinstance(x, DenseBlock):
        if isinstance(y, type(z)):
            if not isinstance(y, (KernelBlock, PatchesBlock)):
                if isinstance(y, DenseBlock):
                    y_block = y.block
                    z_block = z.block
                    res = y.create_similar(block=where_block(x.block, y_block, z_block))
                elif isinstance(y, DiagonalBlock):
                    y_block = y.block
                    z_block = z.block
                    res = y.create_similar(block=where_block(get_diagonal(x.block, y.diag_index), y_block, z_block))
                elif isinstance(y, ConstBlock):
                    y_block = y.get_dense()
                    z_block = z.get_dense()
                    res = x.create_similar(block=where_block(x.block, y_block, z_block))
                    raise Exception('Check this case')
                else:
                    raise NotImplementedError
                return res
            if y.parameters() == z.parameters():
                if isinstance(y, KernelBlock):
                    y = y.convert_to_patches()
                    z = z.convert_to_patches()
                res = DenseBlock(where_block(x.block, y.get_dense(), z.get_dense()))
                # res = y.create_similar(block=where_block(x.convert_to_patches(*y.parameters()).block, y.block, z.block))
                return res
            block_1 = y.get_dense()
            block_2 = z.get_dense()
            return DenseBlock(where_block(x.block, block_1, block_2))
        elif not isinstance(y, type(z)):
            block_1 = y 
            block_2 = z
            flag = False
            if isinstance(y, KernelBlock):
                block_1 = y.convert_to_patches()
                flag = True
            if isinstance(y, ConstBlock):
                if y.block == 0 and isinstance(z, (DiagonalBlock, PatchesBlock)):
                    block_1 = z.create_similar(torch.zeros(z.block.shape))
                else:
                    block_1 = DenseBlock(y.get_dense())
                flag = True
            if isinstance(z, KernelBlock):
                block_2 = z.convert_to_patches()
                flag = True
            if isinstance(z, ConstBlock):
                if z.block == 0 and isinstance(y, (DiagonalBlock, PatchesBlock)):
                    block_2 = y.create_similar(torch.zeros(y.block.shape))
                else:
                    block_2 = DenseBlock(z.get_dense())
                flag = True
            if flag:
                return sp_where_block(x, block_1, block_2)
        block_1 = y.get_dense()
        block_2 = z.get_dense()
        return DenseBlock(where_block(x.block, block_1, block_2))
    
    elif isinstance(x, RepeatBlock):
        if isinstance(y, type(z)):
            if not isinstance(y, (KernelBlock, PatchesBlock)):
                if isinstance(y, DenseBlock):
                    x_block = x.get_dense()
                    y_block = y.block
                    z_block = z.block
                    res = y.create_similar(block=where_block(x.block, y_block, z_block))
                elif isinstance(y, DiagonalBlock):
                    if x.repeat_dims[y.diag_index] > 1 and x.only_one_repeat:
                        x_block = x.squeeze(y.diag_index).block
                        y_block = y.block
                        z_block = z.block
                        res = y.create_similar(block=where_block(x_block, y_block, z_block))
                    elif x.repeat_dims[y.diag_index-1] > 1 and x.only_one_repeat:
                        x_block = x.squeeze(y.diag_index-1).block
                        y_block = y.block
                        z_block = z.block
                        res = y.create_similar(block=where_block(x_block, y_block, z_block))
                elif isinstance(y, ConstBlock):
                    y_block = torch.ones(x.block.shape)*y.block 
                    z_block = torch.ones(x.block.shape)*z.block 
                    res = x.create_similar(block=where_block(x.block, y_block, z_block))
                    # raise Exception('Check this case')
                else:
                    raise NotImplementedError
                return res
            if y.parameters() == z.parameters():
                if isinstance(y, KernelBlock):
                    y = y.convert_to_patches()
                    z = z.convert_to_patches()
                y = y.total_expand()
                z = z.total_expand()
                if x.only_one_repeat and x.repeat_dims[-1] > 1:
                    x_block = x.block.expand(y.block.shape)
                    y_block = y.block
                    z_block = z.block
                    res = y.create_similar(block=where_block(x_block, y_block, z_block))
                else:
                    res = DenseBlock(where_block(x.block, y.get_dense(), z.get_dense()))
                return res
            block_1 = y.get_dense()
            block_2 = z.get_dense()
            return DenseBlock(where_block(x.block, block_1, block_2))
        elif not isinstance(y, type(z)):
            block_1 = y 
            block_2 = z
            flag = False
            if isinstance(y, KernelBlock):
                block_1 = y.convert_to_patches()
                flag = True
            if isinstance(y, ConstBlock):
                if y.block == 0 and isinstance(z, (DiagonalBlock, PatchesBlock)):
                    block_1 = z.create_similar(torch.zeros(z.block.shape))
                else:
                    block_1 = DenseBlock(y.get_dense())
                flag = True
            if isinstance(z, KernelBlock):
                block_2 = z.convert_to_patches()
                flag = True
            if isinstance(z, ConstBlock):
                if z.block == 0 and isinstance(y, (DiagonalBlock, PatchesBlock)):
                    block_2 = y.create_similar(torch.zeros(y.block.shape))
                else:
                    block_2 = DenseBlock(z.get_dense())
                flag = True
            if flag:
                return sp_where_block(x, block_1, block_2)
        block_1 = y.get_dense()
        block_2 = z.get_dense()
        x_block = x.get_dense()
        return DenseBlock(where_block(x_block, block_1, block_2))
    
    
        
    elif isinstance(x, DiagonalBlock):
        if isinstance(y, ConstBlock):
            if isinstance(z, ConstBlock):
                y_dense = y.get_dense()
                z_dense = z.get_dense()
                y_diag = get_diagonal(y_dense, x.diag_index)
                z_diag = get_diagonal(z_dense, x.diag_index)
                return DiagonalBlock(where_block(x.block, y_diag, z_diag), x.total_shape, x.diag_index)
            elif isinstance(z, DiagonalBlock):
                y_dense = y.get_dense()
                y_diag = get_diagonal(y_dense, x.diag_index)
                return DiagonalBlock(where_block(x.block, y_diag, z.block), x.total_shape, x.diag_index)
            elif isinstance(z, DenseBlock):
                x_dense = x.get_dense()
                y_dense = y.get_dense()
                return DenseBlock(where_block(x_dense, y_dense, z.block))
            else:
                raise NotImplementedError
        elif isinstance(y, DiagonalBlock):
            if isinstance(z, ConstBlock):
                z_dense = z.get_dense()
                z_diag = get_diagonal(z_dense, y.diag_index)
                return DiagonalBlock(where_block(x.block, y.block, z_diag), y.total_shape, y.diag_index)
            elif isinstance(z, DiagonalBlock):
                return DiagonalBlock(where_block(x.block, y.block, z.block), y.total_shape, y.diag_index)
            elif isinstance(z, DenseBlock):
                x_dense = x.get_dense()
                y_dense = y.get_dense()
                return DenseBlock(where_block(x_dense, y_dense, z.block))
            else:
                raise NotImplementedError
        elif isinstance(y, DenseBlock):
            z_dense = z.get_dense()
            x_dense = x.get_dense()
            return DenseBlock(where_block(x_dense, y.block, z_dense))
        else:
            raise NotImplementedError
    x_dense = x.get_dense()
    y_dense = y.get_dense()
    z_dense = z.get_dense()
    return DenseBlock(where_block(x_dense, y_dense, z_dense))

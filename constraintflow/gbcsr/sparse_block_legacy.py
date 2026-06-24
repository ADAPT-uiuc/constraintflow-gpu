from numpy import block
import copy
import torch 
import torch.nn.functional as F
import time
import operator
from constraintflow.gbcsr.op_helper import *
from constraintflow.lib.globals import *

# dummy_mode = dummy_mode.get_flag()

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
    if isinstance(x, DummyBlock):
        assert isinstance(y, DummyBlock)
        return x.copy()
    # print(f'{type(x)} {type(y)} {x.device} {y.device}')
    z = op(x, y)
    # if baseline_gpu_mode:
        # torch.cuda.synchronize()
    binary_profilier.update_total_time(time.perf_counter() - start_time)
    return z

def unary_operation(x, op):
    start_time = time.perf_counter()
    #_sync()
    start_op_time = time.perf_counter()
    if op == 'sigma':
        z = torch.sigmoid(x)
    else:
        z = op(x)
    #_sync()
    unary_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
    unary_profilier.update_total_time(time.perf_counter() - start_time)
    return z

def where_block(x, y, z):
    start_time = time.time()
    w = torch.where(x, y, z)
    where_time.update_op_time(time.time() - start_time)
    return w

class SparseBlock:
    repeat_dims = []
    if dummy_mode:
        def __new__(cls, *args, **kwargs):
            return DummyBlock(*args, **kwargs)
    def __init__(self, block, total_shape, block_type='D'):
        if dummy_mode:
            return
        if isinstance(block, torch.Tensor):
            start_transfer = time.perf_counter()
            block = block.to(device_mode.get_device())
            #_sync()
            binary_profilier.update_data_transfer_time(time.perf_counter() - start_transfer)
        if isinstance(block, bool) or (isinstance(block, torch.Tensor) and block.dtype == torch.bool):
            self.block = block 
        else:
            if isinstance(block, torch.Tensor):
                self.block = block.type(torch.float)
            else:
                self.block = float(block)
        self.total_shape = total_shape
        self.block_type = block_type

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
    
    def binary(self, sp_block, op, *args, **kwargs):
        # assert((self.total_shape == sp_block.total_shape).all())
        start_time = time.perf_counter()
        if isinstance(sp_block, ConstBlock):
            if sp_block.block == identity_element(op):
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return self
            elif sp_block.block == annihilator_element(op):
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return sp_block
            if op(0, sp_block.block) in [0, False]:
                res = self.create_similar(block=operation(self.block, sp_block.block, op)) 
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res
        if op in disjunction_ops:
            res = self.disjunctive_binary(sp_block, op)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res
        elif op in conjunction_ops:
            res = self.conjunctive_binary(sp_block, op)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res
        else:
            # assert(False)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            pass
    
    def disjunctive_binary(self, sp_block, op):
        if isinstance(self, type(sp_block)):
            if not isinstance(self, (KernelBlock, PatchesBlock)) or self.parameters() == sp_block.parameters():
                block = operation(self.block, sp_block.block, op)
                res = self.create_similar(block=block)
                return res
            block_1 = self.get_dense()
            block_2 = sp_block.get_dense()
            block = operation(block_1, block_2, op)
            return DenseBlock(block)
        elif not isinstance(self, type(sp_block)):
            block_1 = self 
            block_2 = sp_block
            if isinstance(self, KernelBlock):
                block_1 = self.convert_to_patches()
                return block_1.disjunctive_binary(block_2, op)
            if isinstance(self, ConstBlock):
                block_1 = DenseBlock(self.get_dense())
                return block_1.disjunctive_binary(block_2, op)
            if isinstance(sp_block, KernelBlock):
                block_2 = sp_block.convert_to_patches()
                return block_1.disjunctive_binary(block_2, op)
            if isinstance(sp_block, ConstBlock):
                block_2 = DenseBlock(sp_block.get_dense())
                return block_1.disjunctive_binary(block_2, op)
        block_1 = self.get_dense()
        block_2 = sp_block.get_dense()
        block = operation(block_1, block_2, op)
        return DenseBlock(block)
    
    def conjunctive_binary(self, sp_block, op):
        if isinstance(self, type(sp_block)):
            if not isinstance(self, (KernelBlock, PatchesBlock)) or self.parameters() == sp_block.parameters():
                res = self.create_similar(block=operation(self.block, sp_block.block, op))
                return res
            block_1 = self.get_dense()
            block_2 = sp_block.get_dense()
            block = operation(block_1, block_2, op)
            return DenseBlock(block)
        elif not isinstance(self, type(sp_block)):
            block_1 = self 
            block_2 = sp_block
            if isinstance(self, KernelBlock):
                block_1 = self.convert_to_patches()
                return block_1.conjunctive_binary(block_2, op)
            if isinstance(self, ConstBlock):
                block_1 = DenseBlock(self.get_dense())
                return block_1.conjunctive_binary(block_2, op)
            if isinstance(sp_block, KernelBlock):
                block_2 = sp_block.convert_to_patches()
                return block_1.conjunctive_binary(block_2, op)
            if isinstance(sp_block, ConstBlock):
                block_2 = DenseBlock(sp_block.get_dense())
                return block_1.conjunctive_binary(block_2, op)
        
        if isinstance(self, DiagonalBlock) and isinstance(sp_block, DenseBlock):
            block_1 = self.block
            block_2 = get_diagonal(sp_block.block, self.diag_index)
            block = operation(block_1, block_2, op)
            return self.create_similar(block=block)
        elif isinstance(sp_block, DiagonalBlock) and isinstance(self, DenseBlock):
            block_1 = get_diagonal(self.block, sp_block.diag_index)
            block_2 = sp_block.block
            block = operation(block_1, block_2, op)
            return sp_block.create_similar(block=block)
        block_1 = self.get_dense()
        block_2 = sp_block.get_dense()
        block = operation(block_1, block_2, op)
        return DenseBlock(block)
        if isinstance(self, DiagonalBlock):
            block = get_diagonal(block, self.diag_index)
            return self.create_similar(block=block)
        elif isinstance(sp_block, DiagonalBlock):
            block = get_diagonal(block, sp_block.diag_index)
            return sp_block.create_similar(block=block)
        else:
            return DenseBlock(block)



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
        res = self.block.any()
        any_time.update_op_time(time.time() - start_time)
        return res


class DenseBlock(SparseBlock):
    if dummy_mode:
        def __new__(cls, block):
            total_shape = torch.tensor(block.shape)
            return super().__new__(cls, block=block, total_shape=total_shape, block_type='D')
    def __init__(self, block):
        if dummy_mode:
            return
        total_shape = torch.tensor(block.shape)
        super().__init__(block, total_shape, 'D')
        self.batch_size = total_shape[0]
        # if isinstance(self.batch_size, torch.Tensor):
        #     self.batch_size = self.batch_size.item()

    def get_dense(self):
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
    
    def unsqueeze(self, index):
        start_time = time.time()
        res = self.block.unsqueeze(index)
        end_time = time.time()
        unsqueeze_time.update_op_time(end_time-start_time)
        return DenseBlock(res)
    
    def squeeze(self, index):
        start_time = time.time()
        new_block = self.block.squeeze(index)
        res = DenseBlock(new_block)
        end_time = time.time()
        squeeze_time.update_op_time(end_time - start_time)
        return res
    
    def matmul_equal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            a = self.block
            b = sp_block.block
            start_op_time = time.perf_counter()
            c = a @ b
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(c)
        elif isinstance(sp_block, DiagonalBlock):
            start_op_time = time.perf_counter()
            res = DenseBlock(self.block * sp_block.block.unsqueeze(sp_block.diag_index-1))
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
        elif isinstance(sp_block, KernelBlock):
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
            start_op_time = time.perf_counter()
            input_tensor = self.block.reshape(batch_size*curr_size, num_kernels, ox, oy)
            output_tensor = F.conv_transpose2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py))
            res = DenseBlock(output_tensor.reshape(batch_size, curr_size, -1))
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        else:
            # if sp_block.repeat_dims[0]!=1 and sp_block.only_one_repeat:
            #     block_2 = sp_block.block 
            # else:
            block_2 = sp_block.get_dense()
            start_op_time = time.perf_counter()
            c = self.block @ block_2
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(c)
        equal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res
        
    def matmul_unequal_dims(self, sp_block):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            a = self.block 
            b = sp_block.unsqueeze(-1).block
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            start_op_time = time.perf_counter()
            c = a @ b
            res = (c).squeeze(-1)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(res)
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
        elif isinstance(sp_block, RepeatBlock):
            # warnings.warn(f'Matmul with unequal dims inefficient for {type(self)} and {type(sp_block)}')
            sp_block = DenseBlock(sp_block.get_dense())
            res = self.matmul_unequal_dims(sp_block)
            return res
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res
        
    
    def convert_to_patches(self, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        raise Exception(f'Not an efficient implementation')
        start_time = time.time()
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
    
    def create_similar(self, block):
        return DenseBlock(block)

class KernelBlock(SparseBlock):
    if dummy_mode:
        def __new__(cls, block, total_shape, ix, iy, ox, oy, sx, sy, px, py):
            return super().__new__(
                cls,
                block=block,
                total_shape=total_shape,
                block_type='K',
                ix=ix, iy=iy, ox=ox, oy=oy, sx=sx, sy=sy, px=px, py=py,
            )
    def __init__(self, block, total_shape, ix, iy, ox, oy, sx, sy, px, py):
        if dummy_mode:
            return
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
    if dummy_mode:
        def __new__(cls, block, total_shape, diag_index):
            return super().__new__(
                cls,
                block=block,
                total_shape=total_shape,
                block_type='Diag',
                diag_index=diag_index,
            )
    
    def __init__(self, block, total_shape, diag_index):
        if dummy_mode:
            return
        super().__init__(block, total_shape, 'Diag')
        self.diag_index = diag_index
        self.batch_size = total_shape[0]
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
            return DenseBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]))
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
    if dummy_mode:
        def __new__(cls, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
            return super().__new__(
                cls,
                block=block,
                total_shape=total_shape,
                block_type='P',
                ix=ix, iy=iy, ox=ox, oy=oy, sx=sx, sy=sy, px=px, py=py,
                kx=kx, ky=ky, num_channels=num_channels, num_kernels=num_kernels,
            )
    def __init__(self, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        if dummy_mode:
            return
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
        elif isinstance(sp_block, ConstBlock) and sp_block.block == 0:
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

class _ConstBlockMeta(type):
    """Custom metaclass so that `isinstance(x, ConstBlock)` returns True for
    DummyBlock instances with `block_type == 'C'` (the dummy-mode
    representation of a ConstBlock). The codebase has dozens of
    `isinstance(..., ConstBlock)` checks (in sparse_block.py, sparse_tensor.py,
    tensor_ops.py); rather than rewriting them all to also accept the dummy
    form, we make the standard isinstance protocol do the right thing.

    Only `__instancecheck__` is overridden -- regular ConstBlock construction,
    inheritance, type() identity, and issubclass() are unaffected.
    """
    def __instancecheck__(cls, instance):
        if super().__instancecheck__(instance):
            return True
        return getattr(instance, 'block_type', None) == 'C'


class ConstBlock(SparseBlock, metaclass=_ConstBlockMeta):
    if dummy_mode:
        def __new__(cls, block, total_shape):
            return DummyBlock(block=block, total_shape=total_shape, block_type='C')
    def __init__(self, block, total_shape):
        if dummy_mode:
            return
        super().__init__(block, total_shape, 'C')
        # assert total_shape.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}

    # Done
    def get_dense(self):
        ret = torch.ones(*self.total_shape.tolist()) * self.block
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
    def binary(self, sp_block, op, *args, **kwargs):
        start_time = time.perf_counter()
        if self.block == identity_element(op):
            start = time.perf_counter()
            block = unary_operation(sp_block.block, binary_to_identity_unary(op))
            binary_profilier.update_total_time(time.perf_counter() - start)
            res =sp_block.create_similar(block)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res
        elif self.block == annihilator_element(op):
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return self
        elif self.block == 0 and op == operator.truediv:
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return self
        elif op!=operator.truediv and op(self.block, 0) in [0, False]:
            block = operation(self.block, sp_block.block, op)
            res = sp_block.create_similar(block)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res
        elif op in disjunction_ops:
            res = self.disjunctive_binary(sp_block, op)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res
        elif op in conjunction_ops:
            res = self.conjunctive_binary(sp_block, op)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res
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
    def matmul_equal_dims(self, sp_block):
        # start_time = time.perf_counter()
        if self.block==0:
            new_total_shape = self.total_shape.clone()
            new_total_shape[-1] = sp_block.total_shape[-1]
            res = ConstBlock(0, new_total_shape)
        else:
            raise NotImplementedError
        # matmul_time.update_op_time(time.time() - start_time)
        return res

    # Done        
    def matmul_unequal_dims(self, sp_block):
        # start_time = time.perf_counter()
        if self.block == 0:
            new_total_shape = self.total_shape.clone()[:-1]
            res = ConstBlock(0, new_total_shape)
        else:
            raise NotImplementedError
        # matmul_time.update_op_time(time.time() - start_time)
        return res


    # Done        
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index):
        return ConstBlock(self.block, end_index - start_index)
    
    # Done
    def get_patches(self, batch_size, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        block = torch.ones(batch_size, num_kernels*ox*oy, num_channels*kx*ky)*self.block
        return PatchesBlock(block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)

    # Done     
    def create_similar(self, block):
        return ConstBlock(block, self.total_shape)
    
    # Done
    def sum(self, dim):
        new_const = self.block * self.total_shape[dim]
        new_total_shape = torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]])
        return ConstBlock(new_const, new_total_shape)
    


class RepeatBlock(SparseBlock):
    if dummy_mode:
        def __new__(cls, block, total_shape):
            inner_shape = torch.tensor(block.shape)
            return super().__new__(
                cls,
                block=block,
                total_shape=total_shape,
                block_type='R',
                inner_shape=inner_shape,
            )
    def __init__(self, block, total_shape):
        if dummy_mode:
            return
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
            return False
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


class DummyBlock():
    def __init__(self, block=None, total_shape=None, block_type='D', **kwargs):
        if total_shape is None:
            raise ValueError('DummyBlock requires total_shape')
        if not isinstance(total_shape, torch.Tensor):
            total_shape = torch.as_tensor(total_shape, dtype=torch.int64)
        self.total_shape = total_shape
        self.block_type = block_type

        if block_type == 'C':
            self.block = block
        elif isinstance(block, torch.Tensor):
            self.block = torch.empty(tuple(block.shape), device='meta')
        else:
            self.block = torch.empty(tuple(self.total_shape.tolist()), device='meta')

        if block_type == 'D':
            self.batch_size = total_shape[0] 
        elif block_type == 'C':
            pass
        elif block_type == 'K':
            self.ix = kwargs['ix']
            self.iy = kwargs['iy']
            self.ox = kwargs['ox']
            self.oy = kwargs['oy']
            self.sx = kwargs['sx']
            self.sy = kwargs['sy']
            self.px = kwargs['px']
            self.py = kwargs['py']
            self.kx = block.shape[-2]
            self.ky = block.shape[-1]
            self.num_channels = block.shape[1]
            self.num_kernels = block.shape[0]
        elif block_type == 'P':
            self.ix = kwargs['ix']
            self.iy = kwargs['iy']
            self.ox = kwargs['ox']
            self.oy = kwargs['oy']
            self.sx = kwargs['sx']
            self.sy = kwargs['sy']
            self.px = kwargs['px']
            self.py = kwargs['py']
            self.kx = kwargs['kx']
            self.ky = kwargs['ky']
            self.num_channels = kwargs['num_channels']
            self.num_kernels = kwargs['num_kernels']
            self.batch_size = block.shape[0]
        elif block_type == 'Diag':
            self.diag_index = kwargs['diag_index']
            self.batch_size = total_shape[0]
            if self.diag_index >= len(self.total_shape):
                block_shape = self.total_shape[:-1]
            else:
                block_shape = torch.concat([self.total_shape[:self.diag_index], self.total_shape[self.diag_index + 1:]])
            self.block = torch.empty(tuple(block_shape.tolist()), device='meta')
        elif block_type == 'R':
            self.repeat_dims = total_shape / torch.tensor(self.block.shape)
            self.only_one_repeat = (self.repeat_dims != 1).sum() == 1
        else:
            raise ValueError(f'Unknown DummyBlock block_type {block_type!r}')

    def _is_const_operand(self, sp_block):
        return sp_block.block_type == 'C'

    def binary_sparse_block(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        if self._is_const_operand(sp_block):
            if sp_block.block == identity_element(op):
                if json_list is not None and lhs_index is not None:
                    json_obj = {
                        "method": "noop",
                        "input": "json_list_" + str(lhs_index),
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                return self
            elif sp_block.block == annihilator_element(op):
                if json_list is not None and rhs_index is not None:
                    json_obj = {
                        "method": "noop",
                        "input": "json_list_" + str(rhs_index),
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                return sp_block
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
                res = self.create_similar(block=new_block, json_list=json_list, template_index=binary_index)
                return res
        if op in disjunction_ops:
            return self.disjunctive_binary_sparse_block(sp_block, op, json_list, lhs_index, rhs_index)
        elif op in conjunction_ops:
            return self.conjunctive_binary_sparse_block(sp_block, op, json_list, lhs_index, rhs_index)
    
    def disjunctive_binary_sparse_block(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        other_type = getattr(sp_block, 'block_type')
        if self.block_type == other_type:
            if self.block_type not in ('K', 'P') or (
                [self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels]
                == [sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy, sp_block.sx, sp_block.sy, sp_block.px, sp_block.py, sp_block.kx, sp_block.ky, sp_block.num_channels, sp_block.num_kernels]
            ):
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
                return self.create_similar(block=block, json_list=json_list, template_index=binary_index)
            block_1, lhs_index = self.get_dense(json_list=json_list, index=lhs_index)
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
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
            return DenseBlock(block)
        block_1 = self
        block_2 = sp_block
        if self.block_type == 'K':
            block_1, lhs_index = self.convert_to_patches(json_list=json_list, index=lhs_index)
            return block_1.disjunctive_binary_sparse_block(block_2, op, json_list, lhs_index, rhs_index)
        if getattr(sp_block, 'block_type') == 'K':
            block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, index=rhs_index)
            return block_1.disjunctive_binary_sparse_block(block_2, op, json_list, lhs_index, rhs_index)
        if self._is_const_operand(sp_block):
            block_temp, rhs_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            block_2 = DenseBlock(block_temp)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            return block_1.disjunctive_binary_sparse_block(block_2, op, json_list, lhs_index, rhs_index)
        block_1, lhs_index = self.get_dense(json_list=json_list, index=lhs_index)
        block_2, rhs_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
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
        return DenseBlock(block)
    
    def conjunctive_binary_sparse_block(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        other_type = getattr(sp_block, 'block_type')
        if self.block_type == other_type:
            if self.block_type not in ('K', 'P') or (
                [self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels]
                == [sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy, sp_block.sx, sp_block.sy, sp_block.px, sp_block.py, sp_block.kx, sp_block.ky, sp_block.num_channels, sp_block.num_kernels]
            ):
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
                return self.create_similar(block=block, json_list=json_list, template_index=binary_index)
            block_1, lhs_index = self.get_dense(json_list=json_list, index=lhs_index)
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
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
            return DenseBlock(block)


        block_1 = self
        block_2 = sp_block
        if self.block_type == 'K':
            block_1, lhs_index = self.convert_to_patches(json_list=json_list, index=lhs_index)
            return block_1.conjunctive_binary_sparse_block(block_2, op, json_list, lhs_index, rhs_index)
        if getattr(sp_block, 'block_type') == 'K':
            block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, index=rhs_index)
            return block_1.conjunctive_binary_sparse_block(block_2, op, json_list, lhs_index, rhs_index)
        if self._is_const_operand(sp_block):
            block_temp, index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            block_2 = DenseBlock(block_temp)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            return block_1.conjunctive_binary_sparse_block(block_2, op, json_list, lhs_index, rhs_index)

        if self.block_type == 'Diag' and getattr(sp_block, 'block_type') == 'D':
            block_1 = self.block
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            block_2, rhs_index = get_diagonal(sp_block.block, self.diag_index, json_list=json_list, template_index=rhs_index)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            binary_index = len(json_list) - 1
            block = operation(block_1, block_2, op)
            return self.create_similar(block=block, json_list=json_list, template_index=binary_index)
        elif getattr(sp_block, 'block_type') == 'Diag' and self.block_type == 'D':
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            
            block_1, lhs_index = get_diagonal(self.block, sp_block.diag_index, json_list=json_list, template_index=lhs_index)
            block_2 = sp_block.block
            
            json_obj = {
                "method": "sparse_block_extract",
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
            binary_index = len(json_list) - 1
            block = operation(block_1, block_2, op)
            return sp_block.create_similar(block=block, json_list=json_list, template_index=binary_index)


        block_1, lhs_index = self.get_dense(json_list=json_list, index=lhs_index)
        block_2, rhs_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
        json_obj = {
            "method": "torch_binary",
            "lhs": "json_list_" + str(lhs_index),
            "rhs": "json_list_" + str(rhs_index),
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        block = operation(block_1, block_2, op)
        if self.block_type == 'Diag':
            block, index = get_diagonal(block, self.diag_index, json_list=json_list, template_index=len(json_list) - 1)
            return self.create_similar(block=block, json_list=json_list, template_index=index)
        elif getattr(sp_block, 'block_type') == 'Diag':
            block, index = get_diagonal(block, sp_block.diag_index, json_list=json_list, template_index=len(json_list) - 1)
            return sp_block.create_similar(block=block, json_list=json_list, template_index=index)
        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return DenseBlock(block)
    
    def binary_diagonal_block(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        if getattr(sp_block, 'block_type') == 'R':
            if (sp_block.repeat_dims[self.diag_index] > 1) and sp_block.only_one_repeat:

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1

                block_1 = self.block 

                json_obj = {
                    "method": "block_squeeze",
                    "input": "json_list_" + str(rhs_index),
                    "index": self.diag_index,
                    "output": len(json_list),
                }
                json_list.append(json_obj)

                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(len(json_list) - 1),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block_2 = sp_block.block.squeeze(self.diag_index)
                block = operation(block_1, block_2, op)
                res = self.create_similar(block, json_list=json_list, template_index=binary_index)
                return res
            elif (sp_block.repeat_dims[self.diag_index-1] > 1) and sp_block.only_one_repeat:
                block_1 = self.block 
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1

                block_1 = self.block 

                json_obj = {
                    "method": "block_squeeze",
                    "input": "json_list_" + str(rhs_index),
                    "index": self.diag_index-1,
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
                binary_index = len(json_list) - 1
                block_2 = sp_block.block.squeeze(self.diag_index-1)
                block = operation(block_1, block_2, op)
                res = self.create_similar(block, json_list=json_list, template_index=binary_index)
                return res
        return self.binary_sparse_block(sp_block, op, json_list, lhs_index, rhs_index)
    
    def binary_patches_block(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        if getattr(sp_block, 'block_type') == 'R':
            if sp_block.only_one_repeat and sp_block.repeat_dims[-1] > 1:
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
                    "method": "torch_expand",
                    "input": "json_list_" + str(rhs_index),
                    "shape": list(self.block.shape),
                    "output": len(json_list),
                })
                rhs_index = len(json_list) - 1
                json_list.append({
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                })
                binary_index = len(json_list) - 1
                block_1 = self.block
                block_2 = sp_block.block.expand(self.block.shape)
                block = operation(block_1, block_2, op)
                return self.create_similar(
                    block, json_list=json_list, template_index=binary_index
                )
        return self.binary_sparse_block(sp_block, op, json_list, lhs_index, rhs_index)

    def binary_const_block(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        if self.block == identity_element(op):
            if op == operator.sub:
                json_list.append({
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                })
                block = sp_block.block
                json_list.append({
                    "method": "torch_unary",
                    "input": "json_list_" + str(len(json_list) - 1),
                    "op": binary_to_identity_unary(op).__name__,
                    "output": len(json_list),
                })
                block = unary_operation(sp_block.block, binary_to_identity_unary(op))
                res = sp_block.create_similar(block, json_list=json_list, template_index=len(json_list) - 1)
                return res
            if json_list is not None and rhs_index is not None:
                json_list.append({
                    "method": "noop",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                })
            return sp_block
        if self.block == annihilator_element(op):
            if json_list is not None and lhs_index is not None:
                json_list.append({
                    "method": "noop",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                })
            return self
        if self.block == 0 and op == operator.truediv:
            if json_list is not None and lhs_index is not None:
                json_list.append({
                    "method": "noop",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                })
            return self
        if op != operator.truediv and op(self.block, 0) in [0, False]:

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
            new_block = operation(self.block, sp_block.block, op)
            return sp_block.create_similar(
                block=new_block,
                json_list=json_list,
                template_index=binary_index,
            )
        if op in disjunction_ops:
            return self.disjunctive_binary_sparse_block(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
        elif op in conjunction_ops:
            return self.conjunctive_binary_sparse_block(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
        else:
            raise NotImplementedError
    def get_dense_dense_block(self, json_list=None, index=None):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return self.block, len(json_list) - 1



    def get_dense(self):
        # DUSH: On block
        return self.block.expand(*self.total_shape)
    def get_dense_repeat_block(self, json_list=None, index=None):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_expand",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": self.total_shape.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return self.block.expand(*self.total_shape), len(json_list) - 1

    def get_dense_const_block(self, json_list=None, index=None):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        index = len(json_list) - 1
        json_obj = {
            "method": "torch_ones",
            "size": self.total_shape.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_mul",
            "lhs": "json_list_" + str(len(json_list) - 1),
            "rhs": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        ret = torch.empty(tuple(self.total_shape.tolist()), device='meta')
        return ret, len(json_list) - 1

    def get_dense_diagonal_block(self, json_list=None, index=None):
        if self.diag_index == len(self.total_shape):
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            index = len(json_list) - 1
            json_obj = {
                "method": "torch_diag_embed",
                "input": "json_list_" + str(index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            return torch.diag_embed(self.block), len(json_list) - 1
        
        shape = list(self.block.shape)
        diag_index = self.diag_index-1

        perm = list(range(len(shape)))
        perm.pop(diag_index)
        perm.append(diag_index)

        # DUSH: On block
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        index = len(json_list) - 1
        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(index),
            "perm": perm,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        diag_moved = self.block.permute(perm)

        json_obj = {
            "method": "torch_diag_embed",
            "input": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        diag_expanded = torch.diag_embed(diag_moved)  # last dim becomes (c, c)

        # Step 3: move dims back to restore original ordering + new dimension
        # Insert the extra dim right after diag_index
        new_perm = list(range(len(shape)-1))
        new_perm.insert(diag_index, len(new_perm))
        new_perm.insert(diag_index+1, len(new_perm))
        json_obj = {
            "method" : "torch_permute",
            "input": "json_list_" + str(len(json_list) - 1),
            "perm": new_perm,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        decompressed = diag_expanded.permute(new_perm)

        return decompressed, len(json_list) - 1
    
    
    def get_dense_kernel_block(self, json_list=None, index=None):
        kernel = self.block.float()
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_float",
            "input": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
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
        json_obj = {
            "method": "torch_eye",
            "size": num_kernels*ox*oy,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(len(json_list) - 1),
            "index": 0,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (curr_size, num_kernels, ox, oy),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "F.conv_transpose2d",
            "input": "json_list_" + str(len(json_list) - 1),
            "weight": "json_list_" + str(len(json_list) - 4),
            "stride": (sx, sy),
            "padding": (px, py),
            "output_padding": (new_px, new_py),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (1, curr_size, -1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = F.conv_transpose2d(eye, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py)).reshape(1, curr_size, -1)
        return res, len(json_list) - 1
    
    def get_dense_patches_block(self, json_list=None, index=None):
        batch_size = int(self.total_shape[0])
        output_channel, output_x, output_y = self.num_kernels, self.ox, self.oy
        input_channel, kernel_x, kernel_y = self.num_channels, self.kx, self.ky
        input_x, input_y = self.ix, self.iy
        padding = (self.py, self.py, self.px, self.px)
        stride = self.sx

        # DUSH: On block
        pieces = self.block
        json_obj = {
            "method": "extract_sparse_block",
            "input": "json_list_" + str(index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        pieces_index = len(json_list) - 1

        json_obj = {
            "method": "torch_view",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y),
            "output": len(json_list),
        }
        json_list.append(json_obj)

        pieces = pieces.view(-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y)
        if pieces.shape[0] < batch_size:
            json_obj = {
                "method": "torch_expand",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (batch_size, *pieces.shape[1:]),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pieces = pieces.expand(batch_size, *pieces.shape[1:])
        
        pieces_current_index = len(json_list) - 1
        json_obj = {
            "method": "torch_zeros",
            "size": (batch_size, output_channel, output_x, output_y, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1])),
            "device": "json_list_" + str(pieces_current_index) + ".device",
            "dtype": "json_list_" + str(pieces_current_index) + ".dtype",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        A_matrix_base_index = len(json_list) - 1
        A_matrix = torch.zeros(batch_size, output_channel, output_x, output_y, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device, dtype=pieces.dtype)
        json_obj = {
            "method": "torch_stride",
            "input": "json_list_" + str(A_matrix_base_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)

        
        orig_stride = A_matrix.stride()
        json_obj = {
            "method": "torch_as_strided",
            "input": "json_list_" + str(A_matrix_base_index),
            "size": [batch_size, output_channel, output_x, output_y, output_x, output_y, input_channel, kernel_x, kernel_y],
            "stride": [orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[4], input_y + padding[0] + padding[1], 1],
            "output": len(json_list),
        }
        json_list.append(json_obj)
        matrix_strided_index = len(json_list) - 1
        matrix_strided = torch.as_strided(A_matrix, [batch_size, output_channel, output_x, output_y, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[4], input_y + padding[0] + padding[1], 1])
        first_indices = torch.arange(output_x * output_y, device=pieces.device)
        second_indices = torch.div(first_indices, output_y, rounding_mode="trunc")
        third_indices = torch.fmod(first_indices, output_y)
        # DUSH: On block
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(pieces_index),
            "shape": "(*json_list_" + str(pieces_index) + ".shape[:2], -1, *json_list_" + str(pieces_index) + ".shape[4:])",
            "output": len(json_list),
        }
        json_list.append(json_obj)

        matrix_strided[:,:,second_indices,third_indices,second_indices,third_indices,:,:,:] = pieces.reshape(*pieces.shape[:2], -1, *pieces.shape[4:])
        second_indices_expr = "torch.div(torch.arange(" + str(output_x * output_y) + ", device=json_list_" + str(pieces_current_index) + ".device), " + str(output_y) + ', rounding_mode="trunc")'
        third_indices_expr = "torch.fmod(torch.arange(" + str(output_x * output_y) + ", device=json_list_" + str(pieces_current_index) + ".device), " + str(output_y) + ")"
        json_obj = {
            "method": "assign_to_view",
            "input": "json_list_" + str(matrix_strided_index),
            "base": "json_list_" + str(A_matrix_base_index),
            "index": [0, 0, second_indices_expr, third_indices_expr, second_indices_expr, third_indices_expr, 0, 0, 0],
            "value": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_view",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1]),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        A_matrix = A_matrix.view(batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
        json_obj = {
            "method": "torch_slice",
            "input": "json_list_" + str(len(json_list) - 1),
            "index": [0, 0, 0, [padding[2], input_x + padding[2]], [padding[0], input_y + padding[0]]],
            "output": len(json_list),
        }
        json_list.append(json_obj)
        A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]
        A_matrix = A_matrix.reshape(A_matrix.shape[0], A_matrix.shape[1], -1)
        prev_index = len(json_list) - 1
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(prev_index),
            "shape": "(json_list_" + str(prev_index) + ".shape[0], json_list_" + str(prev_index) + ".shape[1], -1)",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        if len(A_matrix.shape)!=len(self.total_shape):
            # AVAL: This is a hack to make sure the shape is correct.
            # We need to reshape it to the total shape.
            if (torch.tensor(A_matrix.shape) == self.total_shape[:-1]).all():
                diffdim = -1
                A_matrix = A_matrix.unsqueeze(diffdim).expand(*self.total_shape)
            else:
                diffdim = -1
                for i in range(len(A_matrix.shape)):
                    if(diffdim == -1 and self.total_shape[i] != A_matrix.shape[i]):
                            diffdim = i
                    if diffdim != -1 and self.total_shape[i+1] != A_matrix.shape[i]:
                        raise NotImplementedError(f'PatchesBlock get_dense: {A_matrix.shape} != {self.total_shape[:-1]}')
                A_matrix = A_matrix.unsqueeze(diffdim).expand(*self.total_shape)

            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(len(json_list) - 1),
                "index": diffdim,
                "output": len(json_list),
            }
            json_list.append(json_obj)

            json_obj = {
                "method": "torch_expand",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": self.total_shape.tolist(),
                "output": len(json_list),
            }
            json_list.append(json_obj)
        return A_matrix
    
    def get_dense(self, json_list=None, index=None):
        # Public dispatcher: returns (tensor, last_entry_index) just like the
        # per-type helpers. Adding a new block_type means adding both a
        # `get_dense_<type>_block` and a branch here.
        if self.block_type == 'D':
            return self.get_dense_dense_block(json_list=json_list, index=index)
        elif self.block_type == 'Diag':
            return self.get_dense_diagonal_block(json_list=json_list, index=index)
        elif self.block_type == 'K':
            return self.get_dense_kernel_block(json_list=json_list, index=index)
        elif self.block_type == 'P':
            return self.get_dense_patches_block(json_list=json_list, index=index)
        elif self.block_type == 'R':
            return self.get_dense_repeat_block(json_list=json_list, index=index)
        elif self.block_type == 'C':
            return self.get_dense_const_block(json_list=json_list, index=index)
        else:
            raise ValueError(f'Unknown SparseBlock block_type {self.block_type!r}')

    def binary(self, sp_block, op, json_list=None, lhs_index=None, rhs_index=None):
        if self.block_type in ('D', 'K', 'R'):
            return self.binary_sparse_block(sp_block, op, json_list, lhs_index, rhs_index)
        elif self.block_type == 'Diag':
            return self.binary_diagonal_block(sp_block, op, json_list, lhs_index, rhs_index)
        elif self.block_type == 'P':
            return self.binary_patches_block(sp_block, op, json_list, lhs_index, rhs_index)
        elif self.block_type == 'C':
            return self.binary_const_block(sp_block, op, json_list, lhs_index, rhs_index)
        else:
            raise ValueError(f'Unknown SparseBlock block_type {self.block_type!r}')
    


    def create_similar(self, block, json_list = None, template_index = None):
        if self.block_type == 'P':
            return self.create_similar_patches(block, json_list, template_index)
        elif self.block_type == 'R':
            return self.create_similar_repeat(block, json_list, template_index)
        elif self.block_type == 'D':
            return self.create_similar_dense(block, json_list, template_index)
        elif self.block_type == 'K':
            return self.create_similar_kernel(block, json_list, template_index)
        elif self.block_type == 'Diag':
            return self.create_similar_diagonal(block, json_list, template_index)
        elif self.block_type == 'C':
            return self.create_similar_const(block, json_list, template_index)
        else:
            raise ValueError(f'Unknown SparseBlock block_type {self.block_type!r}')

    
    def create_similar_patches(self, block, json_list = None, template_index = None):
        json_obj = {
            "method": "PatchesBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "ix": self.ix,
            "iy": self.iy,
            "ox": self.ox,
            "oy": self.oy,
            "sx": self.sx,
            "sy": self.sy,
            "px": self.px,
            "py": self.py,
            "kx": self.kx,
            "ky": self.ky,
            "num_channels": self.num_channels,
            "num_kernels": self.num_kernels,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return PatchesBlock(block, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)

    def create_similar_repeat(self, block, json_list = None, template_index = None):
        json_obj = {
            "method": "RepeatBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return RepeatBlock(block, self.total_shape) 

    def create_similar_diagonal(self, block, json_list = None, template_index = None):
        json_obj = {
            "method": "DiagonalBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "diag_index": self.diag_index,
            "output": len(json_list),
        }
        json_list.append(json_obj)

        return DiagonalBlock(block, self.total_shape, self.diag_index)    
    
    def create_similar_dense(self, block, json_list = None, template_index = None):
        json_obj = {
            "method": "DenseBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return DenseBlock(block)

    def create_similar_kernel(self, block, json_list = None, template_index = None):
        json_obj = {
            "method": "KernelBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "ix": self.ix,
            "iy": self.iy,
            "ox": self.ox,
            "oy": self.oy,
            "sx": self.sx,
            "sy": self.sy,
            "px": self.px,
            "py": self.py,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return KernelBlock(block, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py)

    def create_similar_const(self, block, json_list = None, template_index = None):
        json_obj = {
            "method": "ConstBlock",
            "block": "json_list_" + str(template_index),
            "total_shape": self.total_shape.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return ConstBlock(block, self.total_shape)


    def unary(self, op):
        if self.block_type == 'C':
            return ConstBlock(unary_operation(self.block, op), self.total_shape)
        return copy.copy(self)
    

    def convert_to_patches(self, json_list = None, index = None):
        if self.block_type == 'K':
            return self.convert_to_patches_kernel_block(json_list, index)
        else:
            raise ValueError(f'Unknown SparseBlock block_type {self.block_type!r}')

    def convert_to_patches_kernel_block(self, json_list = None, index = None):
        num_kernels = self.num_kernels
        ox = self.ox
        oy = self.oy
        json_obj = {
            "method": "torch_view",
            "input": "json_list_" + str(index),
            "shape": (-1, num_kernels, -1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(len(json_list) - 1),
            "index": -1,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_repeat",
            "input": "json_list_" + str(len(json_list) - 1),
            "repeats": (1, 1, 1, ox*oy),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(len(json_list) - 1),
            "permutation": (0,1,3,2),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (-1, num_kernels*ox*oy, -1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "PatchesBlock",
            "block": "json_list_" + str(len(json_list) - 1),
            "total_shape": self.total_shape.tolist(),
            "ix": self.ix,
            "iy": self.iy,
            "ox": self.ox,
            "oy": self.oy,
            "sx": self.sx,
            "sy": self.sy,
            "px": self.px,
            "py": self.py,
            "kx": self.kx,
            "ky": self.ky,
            "num_channels": self.num_channels,
            "num_kernels": self.num_kernels,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        patches = self.block.view(1, num_kernels, -1).unsqueeze(-1).repeat(1, 1, 1, ox*oy)
        patches = patches.permute(0,1,3,2).reshape(patches.size(0),num_kernels*ox*oy, -1)
        return PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels), len(json_list) - 1

    def _replace_block_with_meta(self, res):
        if res.block_type != 'C':
            if res.block_type == 'Diag':
                if res.diag_index >= len(res.total_shape):
                    shape = res.total_shape[:-1]
                else:
                    shape = torch.concat([res.total_shape[:res.diag_index], res.total_shape[res.diag_index + 1:]])
            else:
                shape = res.total_shape
            res.block = torch.empty(tuple(shape.tolist()), device='meta')

    def matmul_equal_dims_dense_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        trace = True
        if sp_block.block_type == 'D':
            if trace:
                lhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "output": lhs_block_index,
                }
                json_list.append(json_obj)
                rhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": rhs_block_index,
                }
                json_list.append(json_obj)
                matmul_index = len(json_list)
                json_obj = {
                    "method": "torch_matmul",
                    "lhs": "json_list_" + str(lhs_block_index),
                    "rhs": "json_list_" + str(rhs_block_index),
                    "output": matmul_index,
                }
                json_list.append(json_obj)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(matmul_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
            a = self.block
            b = sp_block.block
            c = a @ b
            res = DenseBlock(c)
        elif sp_block.block_type == 'Diag':
            if trace:
                rhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": rhs_block_index,
                }
                json_list.append(json_obj)
                unsqueeze_index = len(json_list)
                json_obj = {
                    "method": "torch_unsqueeze",
                    "input": "json_list_" + str(rhs_block_index),
                    "output": unsqueeze_index,
                    "index": sp_block.diag_index - 1,
                }
                json_list.append(json_obj)
                lhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "output": lhs_block_index,
                }
                json_list.append(json_obj)
                mul_index = len(json_list)
                json_obj = {
                    "method": "torch_mul",
                    "lhs": "json_list_" + str(lhs_block_index),
                    "rhs": "json_list_" + str(unsqueeze_index),
                    "output": mul_index,
                }
                json_list.append(json_obj)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(mul_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
            block_2 = sp_block.block.unsqueeze(sp_block.diag_index-1)
            res = DenseBlock(self.block * block_2)
        elif sp_block.block_type == 'K':
            if trace:
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                kernel_index = len(json_list) - 1
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
            batch_size = int(self.batch_size)
            curr_size = int(self.block.shape[-2])
            new_px = (ix + 2*px - kx) % sx
            new_py = (iy + 2*py - ky) % sy
            if trace:
                lhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "output": lhs_block_index,
                }
                json_list.append(json_obj)
                input_tensor_index = len(json_list)
                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(lhs_block_index),
                    "shape": (batch_size*curr_size, num_kernels, ox, oy),
                    "output": input_tensor_index,
                }
                json_list.append(json_obj)
                conv_index = len(json_list)
                json_obj = {
                    "method": "F.conv_transpose2d",
                    "input": "json_list_" + str(input_tensor_index),
                    "weight": "json_list_" + str(kernel_index),
                    "stride": (sx, sy),
                    "padding": (px, py),
                    "output_padding": (new_px, new_py),
                    "output": conv_index,
                }
                json_list.append(json_obj)
                reshape_index = len(json_list)
                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(conv_index),
                    "shape": (batch_size, curr_size, -1),
                    "output": reshape_index,
                }
                json_list.append(json_obj)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(reshape_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
            input_tensor = self.block.reshape(batch_size*curr_size, num_kernels, ox, oy)
            output_tensor = F.conv_transpose2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py))
            res = DenseBlock(output_tensor.reshape(batch_size, curr_size, -1))
        elif sp_block.block_type == 'C':
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]
                if trace:
                    json_obj = {
                        "method": "ConstBlock",
                        "block": 0,
                        "total_shape": new_total_shape.tolist(),
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        else:
            if trace:
                block_2, rhs_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            else:
                block_2 = torch.empty(tuple(sp_block.total_shape.tolist()), device='meta')
            if trace:
                lhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "output": lhs_block_index,
                }
                json_list.append(json_obj)
                matmul_index = len(json_list)
                json_obj = {
                    "method": "torch_matmul",
                    "lhs": "json_list_" + str(lhs_block_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "output": matmul_index,
                }
                json_list.append(json_obj)
                json_obj = {
                    "method": "DenseBlock",
                    "block": "json_list_" + str(matmul_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
            c = self.block @ block_2
            res = DenseBlock(c)
        return res

    def matmul_unequal_dims_dense_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type in ('D', 'R'):
            if sp_block.block_type == 'R':
                block_2, rhs_block_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            else:
                block_2 = sp_block.block
                rhs_block_index = len(json_list)
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "output": rhs_block_index,
                }
                json_list.append(json_obj)
            rhs_unsqueeze_index = len(json_list)
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(rhs_block_index),
                "index": -1,
                "output": rhs_unsqueeze_index,
            }
            json_list.append(json_obj)
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            matmul_index = len(json_list)
            json_obj = {
                "method": "torch_matmul",
                "lhs": "json_list_" + str(lhs_block_index),
                "rhs": "json_list_" + str(rhs_unsqueeze_index),
                "output": matmul_index,
            }
            json_list.append(json_obj)
            squeeze_index = len(json_list)
            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(matmul_index),
                "index": -1,
                "output": squeeze_index,
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(squeeze_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            a = self.block
            b = block_2.unsqueeze(-1)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            start_op_time = time.perf_counter()
            c = a @ b
            res = (c).squeeze(-1)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            res = DenseBlock(res)
        elif sp_block.block_type == 'Diag':
            raise NotImplementedError
        elif sp_block.block_type == 'K':
            raise NotImplementedError
        elif sp_block.block_type == 'C':
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]
                json_obj = {
                    "method": "ConstBlock",
                    "block": 0,
                    "total_shape": new_total_shape.tolist(),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        return res

    def matmul_equal_dims_kernel_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type == 'Diag':
            kernel = self.block
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

            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            view_index = len(json_list)
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(rhs_block_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": view_index,
            }
            json_list.append(json_obj)
            unfold_index = len(json_list)
            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(view_index),
                "kernel_size": (kx, ky),
                "padding": (px, py),
                "stride": (sx, sy),
                "output": unfold_index,
            }
            json_list.append(json_obj)
            permute_index = len(json_list)
            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(unfold_index),
                "permutation": (0, 2, 1),
                "output": permute_index,
            }
            json_list.append(json_obj)
            repeat_index = len(json_list)
            json_obj = {
                "method": "torch_repeat",
                "input": "json_list_" + str(permute_index),
                "repeats": (1, num_kernels, 1),
                "output": repeat_index,
            }
            json_list.append(json_obj)
            k_new, lhs_patches_index = self.convert_to_patches(json_list=json_list, index=lhs_index)
            lhs_patches_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_patches_index),
                "output": lhs_patches_block_index,
            }
            json_list.append(json_obj)
            mul_index = len(json_list)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(repeat_index),
                "rhs": "json_list_" + str(lhs_patches_block_index),
                "output": mul_index,
            }
            json_list.append(json_obj)

            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))
            x_unf = x_unf.permute(0,2,1).repeat(1, num_kernels, 1)
            patches = x_unf * k_new.block
            json_obj = {
                "method": "PatchesBlock",
                "block": "json_list_" + str(mul_index),
                "total_shape": self.total_shape.tolist(),
                "ix": self.ix,
                "iy": self.iy,
                "ox": self.ox,
                "oy": self.oy,
                "sx": self.sx,
                "sy": self.sy,
                "px": self.px,
                "py": self.py,
                "kx": self.kx,
                "ky": self.ky,
                "num_channels": self.num_channels,
                "num_kernels": self.num_kernels,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
        elif sp_block.block_type == 'C':
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
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        elif sp_block.block_type == 'D':
            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            b = sp_block.block
            batch_size = b.shape[0]
            sym_size = b.shape[2]
            permute_index = len(json_list)
            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(rhs_block_index),
                "permutation": (0, 2, 1),
                "output": permute_index,
            }
            json_list.append(json_obj)
            reshape_input_index = len(json_list)
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(permute_index),
                "shape": (batch_size*sym_size, self.num_channels, self.ix, self.iy),
                "output": reshape_input_index,
            }
            json_list.append(json_obj)
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            conv_index = len(json_list)
            json_obj = {
                "method": "F.conv2d",
                "input": "json_list_" + str(reshape_input_index),
                "weight": "json_list_" + str(lhs_block_index),
                "stride": (self.sx, self.sy),
                "padding": (self.px, self.py),
                "output": conv_index,
            }
            json_list.append(json_obj)
            reshape_output_index = len(json_list)
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(conv_index),
                "shape": (batch_size, sym_size, -1),
                "output": reshape_output_index,
            }
            json_list.append(json_obj)
            final_permute_index = len(json_list)
            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(reshape_output_index),
                "permutation": (0, 2, 1),
                "output": final_permute_index,
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(final_permute_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)

            b = b.transpose(1,2)
            b = b.reshape(b.shape[0]*b.shape[1], self.num_channels, self.ix, self.iy)
            block = F.conv2d(b, self.block, stride=(self.sx, self.sy), padding=(self.px, self.py))
            block = block.reshape(batch_size, sym_size, -1)
            block = block.transpose(1,2)
            res = DenseBlock(block)
        elif sp_block.block_type == 'P':
            block_2, rhs_block_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            dense_block_index = len(json_list)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(rhs_block_index),
                "output": dense_block_index,
            }
            json_list.append(json_obj)
            sp_block = DenseBlock(block_2)
            res = self.matmul_equal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_block_index)
            return res
        else:
            block_1, lhs_patches_index = self.convert_to_patches(json_list=json_list, index=lhs_index)
            res = block_1.matmul_equal_dims(sp_block, json_list=json_list, lhs_index=lhs_patches_index, rhs_index=rhs_index)
            return res
        return res

    def matmul_unequal_dims_kernel_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type == 'D':
            kernel = self.block
            sx = self.sx
            sy = self.sy
            px = self.px
            py = self.py
            ix = self.ix
            iy = self.iy
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)

            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            reshape_input_index = len(json_list)
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(rhs_block_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": reshape_input_index,
            }
            json_list.append(json_obj)
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            conv_index = len(json_list)
            json_obj = {
                "method": "F.conv2d",
                "input": "json_list_" + str(reshape_input_index),
                "weight": "json_list_" + str(lhs_block_index),
                "stride": (sx, sy),
                "padding": (px, py),
                "output": conv_index,
            }
            json_list.append(json_obj)
            reshape_output_index = len(json_list)
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(conv_index),
                "shape": (batch_size, -1),
                "output": reshape_output_index,
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(reshape_output_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)

            input_tensor = sp_block.block.reshape(batch_size, num_channels, ix, iy)
            block = F.conv2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py))
            block = block.reshape(batch_size, -1)
            res = DenseBlock(block)
        elif sp_block.block_type == 'Diag':
            raise NotImplementedError
        elif sp_block.block_type == 'K':
            raise NotImplementedError
        elif sp_block.block_type == 'C':
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]
                json_obj = {
                    "method": "ConstBlock",
                    "block": 0,
                    "total_shape": new_total_shape.tolist(),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res = ConstBlock(0, new_total_shape)
            else:
                raise NotImplementedError
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        return res

    def matmul_equal_dims_diagonal_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type == 'D':
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            lhs_unsqueeze_index = len(json_list)
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(lhs_block_index),
                "index": self.diag_index,
                "output": lhs_unsqueeze_index,
            }
            json_list.append(json_obj)
            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            mul_index = len(json_list)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_unsqueeze_index),
                "rhs": "json_list_" + str(rhs_block_index),
                "output": mul_index,
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(mul_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            a = self.block.unsqueeze(self.diag_index)
            b = sp_block.block
            c = a * b
            res = DenseBlock(c)
        elif sp_block.block_type == 'Diag':
            raise NotImplementedError
        elif sp_block.block_type == 'K':
            block_2, rhs_patches_index = sp_block.convert_to_patches(json_list=json_list, index=rhs_index)
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            lhs_unsqueeze_index = len(json_list)
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(lhs_block_index),
                "index": self.diag_index,
                "output": lhs_unsqueeze_index,
            }
            json_list.append(json_obj)
            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_patches_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            mul_index = len(json_list)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_unsqueeze_index),
                "rhs": "json_list_" + str(rhs_block_index),
                "output": mul_index,
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "PatchesBlock",
                "block": "json_list_" + str(mul_index),
                "total_shape": sp_block.total_shape.tolist(),
                "ix": sp_block.ix,
                "iy": sp_block.iy,
                "ox": sp_block.ox,
                "oy": sp_block.oy,
                "sx": sp_block.sx,
                "sy": sp_block.sy,
                "px": sp_block.px,
                "py": sp_block.py,
                "kx": sp_block.kx,
                "ky": sp_block.ky,
                "num_channels": sp_block.num_channels,
                "num_kernels": sp_block.num_kernels,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_1 = self.block.unsqueeze(self.diag_index)
            block = block_1 * block_2.block
            res = PatchesBlock(block, sp_block.total_shape, sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy, sp_block.sx, sp_block.sy, sp_block.px, sp_block.py, sp_block.kx, sp_block.ky, sp_block.num_channels, sp_block.num_kernels)
        elif sp_block.block_type == 'R' and (sp_block.repeat_dims == 1).all():
            block_2, rhs_block_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            dense_block_index = len(json_list)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(rhs_block_index),
                "output": dense_block_index,
            }
            json_list.append(json_obj)
            sp_block = DenseBlock(block_2)
            res = self.matmul_equal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_block_index)
            return res
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        return res

    def matmul_unequal_dims_diagonal_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type == 'D':
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            lhs_unsqueeze_index = len(json_list)
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(lhs_block_index),
                "index": self.diag_index,
                "output": lhs_unsqueeze_index,
            }
            json_list.append(json_obj)
            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            rhs_unsqueeze_index = len(json_list)
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(rhs_block_index),
                "index": -1,
                "output": rhs_unsqueeze_index,
            }
            json_list.append(json_obj)
            mul_index = len(json_list)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_unsqueeze_index),
                "rhs": "json_list_" + str(rhs_unsqueeze_index),
                "output": mul_index,
            }
            json_list.append(json_obj)
            squeeze_index = len(json_list)
            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(mul_index),
                "index": -1,
                "output": squeeze_index,
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(squeeze_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            a = self.block.unsqueeze(self.diag_index)
            b = sp_block.block.unsqueeze(-1)
            res = a * b
            res = res.squeeze(-1)
            res = DenseBlock(res)
        elif sp_block.block_type == 'R' and (sp_block.repeat_dims == 1).all():
            block_2, rhs_block_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            dense_block_index = len(json_list)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(rhs_block_index),
                "output": dense_block_index,
            }
            json_list.append(json_obj)
            sp_block = DenseBlock(block_2)
            res = self.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_block_index)
            return res
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        return res

    def matmul_equal_dims_patches_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type == 'K':
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)
            flattened_index = len(json_list)
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(lhs_block_index),
                "shape": (self.batch_size*self.num_kernels*self.ox*self.oy, self.num_channels, self.kx, self.ky),
                "output": flattened_index,
            }
            json_list.append(json_obj)
            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            conv_index = len(json_list)
            json_obj = {
                "method": "F.conv_transpose2d",
                "input": "json_list_" + str(flattened_index),
                "weight": "json_list_" + str(rhs_block_index),
                "stride": (sp_block.sx, sp_block.sy),
                "output": conv_index,
            }
            json_list.append(json_obj)

            flattened_patches = self.block.reshape(self.batch_size*self.num_kernels*self.ox*self.oy, self.num_channels, self.kx, self.ky)
            patches = F.conv_transpose2d(flattened_patches, sp_block.block, stride=(sp_block.sx, sp_block.sy))
            kx = patches.shape[-2]
            ky = patches.shape[-1]
            reshape_index = len(json_list)
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(conv_index),
                "shape": (self.batch_size, self.num_kernels*self.ox*self.oy, -1),
                "output": reshape_index,
            }
            json_list.append(json_obj)
            patches = patches.reshape(self.batch_size, self.num_kernels*self.ox*self.oy, -1)

            full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p, p) if isinstance(p, int) else p
                for p in [(self.px, self.py), (sp_block.px, sp_block.py), (self.sx, self.sy), (sp_block.sx, sp_block.sy)]
            ]
            full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p[1], p[1], p[0], p[0]) if len(p) == 2 else p
                for p in [full_patch_padding, full_op_padding, full_patch_stride, full_op_stride]
            ]
            new_padding = tuple(pp * os + op for pp, op, os in zip(full_patch_padding, full_op_padding, full_op_stride))
            new_stride = tuple(ps * os for ps, os in zip(full_patch_stride, full_op_stride))
            new_total_shape = torch.concat([torch.max(self.total_shape[:-2], sp_block.total_shape[:-2]), self.total_shape[-2:-1], sp_block.total_shape[-1:]])

            json_obj = {
                "method": "PatchesBlock",
                "block": "json_list_" + str(reshape_index),
                "total_shape": new_total_shape.tolist(),
                "ix": sp_block.ix,
                "iy": sp_block.iy,
                "ox": self.ox,
                "oy": self.oy,
                "sx": new_stride[0],
                "sy": new_stride[1],
                "px": new_padding[0],
                "py": new_padding[1],
                "kx": kx,
                "ky": ky,
                "num_channels": sp_block.num_channels,
                "num_kernels": self.num_kernels,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = PatchesBlock(patches, new_total_shape, sp_block.ix, sp_block.iy, self.ox, self.oy, new_stride[0], new_stride[1], new_padding[0], new_padding[1], kx, ky, sp_block.num_channels, self.num_kernels)
        elif sp_block.block_type == 'Diag':
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

            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            view_index = len(json_list)
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(rhs_block_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": view_index,
            }
            json_list.append(json_obj)
            unfold_index = len(json_list)
            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(view_index),
                "kernel_size": (kx, ky),
                "padding": (px, py),
                "stride": (sx, sy),
                "output": unfold_index,
            }
            json_list.append(json_obj)
            transpose_index = len(json_list)
            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(unfold_index),
                "permutation": (0, 2, 1),
                "output": transpose_index,
            }
            json_list.append(json_obj)
            repeat_index = len(json_list)
            json_obj = {
                "method": "torch_repeat",
                "input": "json_list_" + str(transpose_index),
                "repeats": (1, num_kernels, 1),
                "output": repeat_index,
            }
            json_list.append(json_obj)
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)

            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))
            x_unf = x_unf.transpose(1,2).repeat(1, num_kernels, 1)
            if patches.shape[0] != batch_size:
                expand_index = len(json_list)
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(lhs_block_index),
                    "shape": (batch_size, patches.size(1), patches.size(2)),
                    "output": expand_index,
                }
                json_list.append(json_obj)
                lhs_block_index = expand_index
                patches = patches.expand(batch_size, patches.size(1), patches.size(2))

            mul_index = len(json_list)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(repeat_index),
                "rhs": "json_list_" + str(lhs_block_index),
                "output": mul_index,
            }
            json_list.append(json_obj)
            patches = x_unf * patches
            json_obj = {
                "method": "PatchesBlock",
                "block": "json_list_" + str(mul_index),
                "total_shape": self.total_shape.tolist(),
                "ix": self.ix,
                "iy": self.iy,
                "ox": self.ox,
                "oy": self.oy,
                "sx": self.sx,
                "sy": self.sy,
                "px": self.px,
                "py": self.py,
                "kx": self.kx,
                "ky": self.ky,
                "num_channels": self.num_channels,
                "num_kernels": self.num_kernels,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)
        else:
            raise NotImplementedError
        return res

    def matmul_unequal_dims_patches_block(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if sp_block.block_type == 'D':
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

            rhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "output": rhs_block_index,
            }
            json_list.append(json_obj)
            view_index = len(json_list)
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(rhs_block_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": view_index,
            }
            json_list.append(json_obj)
            unfold_index = len(json_list)
            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(view_index),
                "kernel_size": (kx, ky),
                "padding": (px, py),
                "stride": (sx, sy),
                "output": unfold_index,
            }
            json_list.append(json_obj)
            transpose_index = len(json_list)
            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(unfold_index),
                "permutation": (0, 2, 1),
                "output": transpose_index,
            }
            json_list.append(json_obj)
            repeat_index = len(json_list)
            json_obj = {
                "method": "torch_repeat",
                "input": "json_list_" + str(transpose_index),
                "repeats": (1, num_kernels, 1),
                "output": repeat_index,
            }
            json_list.append(json_obj)
            lhs_block_index = len(json_list)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "output": lhs_block_index,
            }
            json_list.append(json_obj)

            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))
            x_unf = x_unf.transpose(1,2).repeat(1, num_kernels, 1)
            if patches.shape[0] != batch_size:
                expand_index = len(json_list)
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(lhs_block_index),
                    "shape": (batch_size, patches.size(1), patches.size(2)),
                    "output": expand_index,
                }
                json_list.append(json_obj)
                lhs_block_index = expand_index
                patches = patches.expand(batch_size, patches.size(1), patches.size(2))

            mul_index = len(json_list)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(repeat_index),
                "rhs": "json_list_" + str(lhs_block_index),
                "output": mul_index,
            }
            json_list.append(json_obj)
            patches = x_unf * patches
            sum_index = len(json_list)
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(mul_index),
                "dim": -1,
                "output": sum_index,
            }
            json_list.append(json_obj)
            ret = patches.sum(dim=-1)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(sum_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = DenseBlock(ret)
        elif sp_block.block_type == 'C' and sp_block.block == 0:
            json_obj = {
                "method": "ConstBlock",
                "block": 0,
                "total_shape": self.total_shape[:-1].tolist(),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = ConstBlock(0, self.total_shape[:-1])
        elif sp_block.block_type == 'R':
            block_2, rhs_block_index = sp_block.get_dense(json_list=json_list, index=rhs_index)
            dense_block_index = len(json_list)
            json_obj = {
                "method": "DenseBlock",
                "block": "json_list_" + str(rhs_block_index),
                "output": dense_block_index,
            }
            json_list.append(json_obj)
            sp_block = DenseBlock(block_2)
            res = self.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_block_index)
            return res
        else:
            raise NotImplementedError
        return res

    def matmul_equal_dims(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if self.block_type == 'D':
            return self.matmul_equal_dims_dense_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'K':
            return self.matmul_equal_dims_kernel_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'Diag':
            return self.matmul_equal_dims_diagonal_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'P':
            return self.matmul_equal_dims_patches_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'C':
            if self.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]
                return ConstBlock(0, new_total_shape)
            raise NotImplementedError(
                f"matmul_equal_dims on a ConstBlock with non-zero scalar "
                f"{self.block!r} is not supported (matches legacy ConstBlock.matmul_equal_dims)"
            )
        res = copy.copy(self)
        res.total_shape = self.total_shape.clone()
        res.total_shape[-1] = sp_block.total_shape[-1]
        self._replace_block_with_meta(res)
        return res

    def matmul_unequal_dims(self, sp_block, json_list=None, lhs_index=None, rhs_index=None):
        if self.block_type == 'D':
            return self.matmul_unequal_dims_dense_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'K':
            return self.matmul_unequal_dims_kernel_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'Diag':
            return self.matmul_unequal_dims_diagonal_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'P':
            return self.matmul_unequal_dims_patches_block(
                sp_block,
                json_list=json_list,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
            )
        if self.block_type == 'C':
            # Mirror ConstBlock.matmul_unequal_dims: only block==0 is supported.
            if self.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]
                return ConstBlock(0, new_total_shape)
            raise NotImplementedError(
                f"matmul_unequal_dims on a ConstBlock with non-zero scalar "
                f"{self.block!r} is not supported (matches legacy ConstBlock.matmul_unequal_dims)"
            )
        res = copy.copy(self)
        res.total_shape = self.total_shape.clone()[:-1]
        self._replace_block_with_meta(res)
        return res
    
    def sum(self, dim):
        if self.block_type == 'C':
            new_scalar = self.block * self.total_shape[dim]
            new_total_shape = torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]])
            return ConstBlock(new_scalar, new_total_shape)
        res = copy.copy(self)
        res.total_shape = torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]])
        self._replace_block_with_meta(res)
        return res

    def squeeze(self, index):
        if self.block_type == 'C':
            return ConstBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]))
        res = copy.copy(self)
        res.total_shape = torch.concat([self.total_shape[:index], self.total_shape[index+1:]])
        self._replace_block_with_meta(res)
        return res
    
    def unsqueeze(self, index):
        if self.block_type == 'C':
            return ConstBlock(self.block, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]))
        res = copy.copy(self)
        res.total_shape = torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]])
        self._replace_block_with_meta(res)
        return res

    def repeat(self, repeat_dims):
        if self.block_type == 'C':
            res = ConstBlock(self.block, self.total_shape*repeat_dims)
            return res
        res = copy.copy(self)
        res.total_shape = self.total_shape * repeat_dims
        self._replace_block_with_meta(res)
        return res
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index):
        if self.block_type == 'C':
            return ConstBlock(self.block, end_index - start_index)
        res = copy.copy(self)
        res.total_shape = end_index - start_index
        self._replace_block_with_meta(res)
        return res

    def copy(self):
        return copy.copy(self)
    
    def any(self):
        if self.block_type == 'C':
            return self.block == True
        return True
    
    def clamp(self, const, min_true):
        if self.block_type == 'C':
            if min_true:
                if self.block >= const:
                    return self
                return ConstBlock(const, self.total_shape)
            else:
                if self.block <= const:
                    return self
                return ConstBlock(const, self.total_shape)
        return self

    def float(self):
        if self.block_type == 'C':
            if self.block == False:
                return ConstBlock(0.0, self.total_shape)
            elif self.block == True:
                return ConstBlock(1.0, self.total_shape)
            return self
        return self

    def overwrite_dense_block(self, sp_block, start_index, s):
        if self.block_type == 'C':
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
        raise NotImplementedError(
            f'overwrite_dense_block not implemented for DummyBlock block_type {self.block_type!r}'
        )

    def get_patches(self, batch_size, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        if self.block_type == 'C':
            block = torch.ones(batch_size, num_kernels*ox*oy, num_channels*kx*ky)*self.block
            return PatchesBlock(block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)
        raise NotImplementedError(
            f'get_patches not implemented for DummyBlock block_type {self.block_type!r}'
        )


def sp_where_block(x: SparseBlock, y: SparseBlock, z: SparseBlock, dummy: bool=False):
    if dummy_mode or dummy:
        if isinstance(y, DummyBlock):
            res = copy.copy(y)
            res.total_shape = y.total_shape.clone()
            if res.block_type != 'C':
                res.block = torch.empty(tuple(res.total_shape.tolist()), device='meta')
            return res
        return DummyBlock(block=None, total_shape=y.total_shape.clone(), block_type=getattr(y, 'block_type', 'D'))
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

from constraintflow.lib.abs_elem import Abs_elem_sparse
from constraintflow.lib.polyexp import *
from constraintflow.lib.symexp import *
from constraintflow.lib.llist import *
from constraintflow.lib.network import Network, LayerType
from constraintflow.lib.globals import *
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.gbcsr.sparse_block import DenseBlock, DiagonalBlock

import torch
import torch.nn.functional as F
import time


def get_dense_inlined(t):
    # Inlined get_dense; the simulacrum needs the real meta-tensor path.
    if dummy_mode:
        return t.get_dense()
    res = torch.ones(list(t.total_size), dtype=t.type) * t.dense_const
    for i in range(t.num_blocks):
        s = [slice(int(t.start_indices[i][j]), int(t.end_indices[i][j])) for j in range(t.start_indices[i].shape[0])]
        b = t.blocks[i]
        if b.block_type == 'D':
            res[tuple(s)] = b.block
        elif b.block_type == 'C':
            res[tuple(s)] = torch.ones(*b.total_shape.tolist()) * b.block
        elif b.block_type == 'R':
            res[tuple(s)] = b.block.expand(*b.total_shape)
        elif b.block_type == 'Diag':
            if b.diag_index == len(b.total_shape):
                res[tuple(s)] = torch.diag_embed(b.block)
            else:
                shape = list(b.block.shape)
                d = b.diag_index - 1
                perm = list(range(len(shape)))
                perm.pop(d)
                perm.append(d)
                new_perm = list(range(len(shape)-1))
                new_perm.insert(d, len(new_perm))
                new_perm.insert(d+1, len(new_perm))
                res[tuple(s)] = torch.diag_embed(b.block.permute(perm)).permute(new_perm)
        elif b.block_type == 'K':
            new_px = (b.ix + 2*b.px - b.kx) % b.sx
            new_py = (b.iy + 2*b.py - b.ky) % b.sy
            curr_size = b.num_kernels*b.ox*b.oy
            eye = torch.eye(curr_size).unsqueeze(0).reshape(curr_size, b.num_kernels, b.ox, b.oy)
            res[tuple(s)] = F.conv_transpose2d(eye, b.block.float(), stride=(b.sx, b.sy), padding=(b.px, b.py), output_padding=(new_px, new_py)).reshape(1, curr_size, -1)
        elif b.block_type == 'P':
            batch_size = b.total_shape[0]
            output_channel, output_x, output_y = b.num_kernels, b.ox, b.oy
            input_channel, kernel_x, kernel_y = b.num_channels, b.kx, b.ky
            input_x, input_y = b.ix, b.iy
            padding = (b.py, b.py, b.px, b.px)
            stride = b.sx
            pieces = b.block.view(-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y)
            if pieces.shape[0] < batch_size:
                pieces = pieces.expand(batch_size, *pieces.shape[1:])
            A_matrix = torch.zeros(batch_size, output_channel, output_x, output_y, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device, dtype=pieces.dtype)
            orig_stride = A_matrix.stride()
            matrix_strided = torch.as_strided(A_matrix, [batch_size, output_channel, output_x, output_y, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[4], input_y + padding[0] + padding[1], 1])
            first_indices = torch.arange(output_x * output_y, device=pieces.device)
            second_indices = torch.div(first_indices, output_y, rounding_mode="trunc")
            third_indices = torch.fmod(first_indices, output_y)
            matrix_strided[:,:,second_indices,third_indices,second_indices,third_indices,:,:,:] = pieces.reshape(*pieces.shape[:2], -1, *pieces.shape[4:])
            A_matrix = A_matrix.view(batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
            A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]
            A_matrix = A_matrix.reshape(A_matrix.shape[0], A_matrix.shape[1], -1)
            if len(A_matrix.shape)!=len(b.total_shape):
                if (torch.tensor(A_matrix.shape) == b.total_shape[:-1]).all():
                    A_matrix = A_matrix.unsqueeze(-1).expand(*b.total_shape)
                else:
                    diffdim = -1
                    for i in range(len(A_matrix.shape)):
                        if(diffdim == -1 and b.total_shape[i] != A_matrix.shape[i]):
                                diffdim = i
                        if diffdim != -1 and b.total_shape[i+1] != A_matrix.shape[i]:
                            raise NotImplementedError(f'PatchesBlock get_dense: {A_matrix.shape} != {b.total_shape[:-1]}')
                    A_matrix = A_matrix.unsqueeze(diffdim).expand(*b.total_shape)
            res[tuple(s)] = A_matrix
        else:
            raise NotImplementedError(f'get_dense: unknown block_type {b.block_type}')
    return res


class Flow:
    def __init__(self, abs_elem: Abs_elem_sparse, transformer, model: Network, print_intermediate_results=False, no_sparsity=False):
        self.abs_elem = abs_elem 
        # type of self.transformer: generated abstract transformer.
        self.transformer = transformer 
        self.model = model
        self.input_size = model.input_size
        self.batch_size = abs_elem.batch_size
        self.print_intermediate_results = print_intermediate_results
        self.no_sparsity = no_sparsity
        self.logfile = open("flow_log_1.txt", "w")

    def flow(self):
        begin_time = time.time()
        prev_size = self.model.input_size
        size = self.model.input_size

        layer_types = ['affine', 'relu', 'sigmoid']
        json_obj = {i: [] for i in layer_types}

        

        for tmp, layer in enumerate(self.model):
            t_time = time.time()
            poly_size = self.model[self.abs_elem.live_layers[-1]].end
            curr_size = self.model[tmp].end-size

            if layer.type == LayerType.ReLU:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
                json_obj['relu'].append(tmp)

            elif layer.type == LayerType.Sigmoid:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Sigmoid(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
                json_obj['sigmoid'].append(tmp)
            elif layer.type == LayerType.Linear:
                prev = Llist(self.model, [1, 1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
                json_obj['affine'].append(tmp)
                
            elif layer.type == LayerType.Conv2D:
                prev = Llist(self.model, [1, 1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
                json_obj['affine'].append(tmp)

            elif layer.type == LayerType.Input:
                continue
            elif layer.type == LayerType.Add:
                prev1 = Llist(self.model, [1], None, None, [layer.parents[0]])
                prev2 = Llist(self.model, [1], None, None, [layer.parents[1]])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = []
                for key in self.abs_elem.d.keys():
                    if key == 'llist':
                        continue
                    elif isinstance(self.abs_elem.d[key], SparseTensor):
                        res = self.abs_elem.get_elem(key, prev1).binary(self.abs_elem.get_elem(key, prev2), operator.add)
                        abs_shape.append(res)
                    elif isinstance(self.abs_elem.d[key], PolyExpSparse) or isinstance(self.abs_elem.d[key], SymExpSparse):
                        exp1 = self.abs_elem.get_elem(key, prev1)
                        exp2 = self.abs_elem.get_elem(key, prev2)
                        const1 = exp1.get_const()
                        const2 = exp2.get_const()
                        const = const1.binary(const2, operator.add)
                        mat1 = exp1.get_mat(self.abs_elem)
                        mat2 = exp2.get_mat(self.abs_elem)
                        mat = mat1.binary(mat2, operator.add)
                        if isinstance(self.abs_elem.d[key], PolyExpSparse):
                            abs_shape.append(PolyExpSparse(self.model, mat, const))
                        elif isinstance(self.abs_elem.d[key], SymExpSparse):
                            abs_shape.append(SymExpSparse(self.model, mat, const))
            elif layer.type == LayerType.Concat:
                debug_flag.set_flag()
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = []
                for key in self.abs_elem.d.keys():
                    if key == 'llist':
                        continue
                    if isinstance(self.abs_elem.d[key], SparseTensor):
                        start_indices = []
                        end_indices = []
                        blocks = []
                        new_start_index = 0
                        for par in layer.parents:
                            start_index = torch.tensor([0, self.model[par].start])
                            end_index = torch.tensor([self.batch_size, self.model[par].end])
                            block_id = self.abs_elem.d[key].get_block_id(start_index, end_index)[0][0]
                            block = self.abs_elem.d[key].blocks[block_id]
                            start_indices.append(torch.tensor([0, new_start_index]))
                            end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size]))
                            new_start_index += self.model[par].size
                            blocks.append(block)
                        abs_shape.append(SparseTensor(start_indices, blocks, 2, torch.tensor([self.batch_size, self.model[tmp].size])))
                    elif isinstance(self.abs_elem.d[key], PolyExpSparse) or isinstance(self.abs_elem.d[key], SymExpSparse):
                        debug_flag.set_flag()
                        start_indices = []
                        end_indices = []
                        blocks = []
                        new_start_index = 0
                        for par in layer.parents:
                            start_index = torch.tensor([0, self.model[par].start])
                            end_index = torch.tensor([self.batch_size, self.model[par].end])
                            block_id = self.abs_elem.d[key].const.get_block_id(start_index, end_index)[0][0]
                            block = self.abs_elem.d[key].const.blocks[block_id]
                            start_indices.append(torch.tensor([0, new_start_index]))
                            end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size]))
                            new_start_index += self.model[par].size
                            blocks.append(block)
                        const = SparseTensor(start_indices, blocks, 2, torch.tensor([self.batch_size, self.model[tmp].size]))

                        if isinstance(self.abs_elem.d[key], PolyExpSparse):
                            start_indices = []
                            end_indices = []
                            blocks = []
                            new_start_index = 0
                            for par in layer.parents:
                                start_index = torch.tensor([0, self.model[par].start, 0])
                                end_index = torch.tensor([self.batch_size, self.model[par].end, self.abs_elem.d[key].mat.total_size[-1]])
                                block_ids, block_start_indices, block_end_indices = self.abs_elem.d[key].mat.get_block_id(start_index, end_index)
                                for i in range(len(block_ids)):
                                    block_id = block_ids[i]
                                    block = self.abs_elem.d[key].mat.blocks[block_id]
                                    start_indices.append(torch.tensor([0, new_start_index, block_start_indices[i][2]]))
                                    end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size, block_end_indices[i][2]]))
                                    blocks.append(block)
                                new_start_index += self.model[par].size
                            total_size = torch.tensor([self.batch_size, self.model[tmp].size, self.abs_elem.d[key].mat.total_size[-1]])
                            mat = SparseTensor(start_indices, blocks, 3, total_size)
                            abs_shape.append(PolyExpSparse(self.model, mat, const))
                        elif isinstance(self.abs_elem.d[key], SymExpSparse):
                            start_indices = []
                            end_indices = []
                            blocks = []
                            new_start_index = 0
                            for p, par in enumerate(layer.parents):
                                start_index = torch.tensor([0, self.model[par].start, 0])
                                end_index = torch.tensor([self.batch_size, self.model[par].end, SymExpSparse.count])
                                block_ids, block_start_indices, block_end_indices = self.abs_elem.d[key].mat.get_block_id(start_index, end_index)
                                for i in range(len(block_ids)):
                                    block_id = block_ids[i]
                                    block = self.abs_elem.d[key].mat.blocks[block_id]
                                    if p==0:
                                        start_indices.append(torch.tensor([0, new_start_index, block_start_indices[i][2]]))
                                        end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size, block_end_indices[i][2]]))
                                        blocks.append(block)
                                    if p==1:
                                        end_indices[i] = torch.tensor([0, end_indices[i][1]+new_start_index, block_start_indices[i][2]])
                                        if isinstance(block, DenseBlock):
                                            blocks[i] = DenseBlock(torch.concat([blocks[i].block, block.block], dim=1))
                                        elif isinstance(block, DiagonalBlock):
                                            new_total_shape = torch.tensor([block.total_shape[0].item(), block.total_shape[1].item() + blocks[i].total_shape[1].item(), max(block.total_shape[2].item(), blocks[i].total_shape[2].item())])
                                            blocks[i] = DiagonalBlock(torch.concat([blocks[i].block, block.block], dim=1), new_total_shape, diag_index=block.diag_index)
                                new_start_index += self.model[par].size
                            total_size = torch.tensor([self.batch_size, new_start_index+self.model[tmp].size, SymExpSparse.count])
                            mat = SparseTensor(start_indices, blocks, 3, total_size, end_indices=end_indices)
                            abs_shape.append(SymExpSparse(self.model, mat, const))
                    else:
                        assert(False)
            else:
                print(layer.type)
                assert(False)
            size += curr_size
            prev_size = self.model[tmp].size
            print(layer.type)
            self.abs_elem.update(curr, abs_shape)
            # print(f"abs_elem: {tmp}")
            # print(f"LList")
            # print(self.abs_elem.d['llist'])
            # print("\n")
            # print(f"l")
            # print(self.abs_elem.d['l'])
            # print("\n")
            # print(f"u")
            # print(self.abs_elem.d['u'])
            # print("\n")
            # print(f"L")
            # print(self.abs_elem.d['L'])
            # print("\n")
            # print(f"U")
            # print(self.abs_elem.d['U'])


            if self.print_intermediate_results:
                self.logfile.write(f"Layer {tmp+1}: {layer.type}\n")
                print(tmp+1, layer.type, layer.shape)
                print(time.time()-t_time)
                print('---------------------------')
                lb = get_dense_inlined(abs_shape[0])
                ub = get_dense_inlined(abs_shape[1])
                self.logfile.write(f'l: {lb}\n')
                self.logfile.write(f'u: {ub}\n')
                print(f'l: {lb}')
                print(f'u: {ub}')
                # if len(abs_shape) > 3:
                #     L = (abs_shape[2].mat)
                #     U = (abs_shape[3].mat)
                #     print(f'L: {L}')
                #     print(f'U: {U}')
                # elif len(abs_shape) > 2 and hasattr(abs_shape[2], 'mat'):
                #     print(f'Z: {abs_shape[2].mat}')
        lb = get_dense_inlined(abs_shape[0])
        ub = get_dense_inlined(abs_shape[1])

        if dummy_mode:
            save_capture("jit_layers/layers.json", json_obj)


        return lb, ub

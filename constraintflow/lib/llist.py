import torch 
import math

from constraintflow.lib.polyexp import *
from constraintflow.lib.network import LayerType
from constraintflow.gbcsr.sparse_block import DenseBlock, KernelBlock, ConstBlock, DiagonalBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.lib.globals import dummy_mode

class Llist:
    """List of layers."""
    def __init__(self, network, initial_shape, start=None, end=None, llist=None):
        # network type: lib.network.Network
        self.network = network
        # Is this understanding correct:
        # initial_shape is the prefixing dimensions for structural/shape
        # convenience, like batching dimensions?
        self.initial_shape = initial_shape
        self.start = start
        self.end = end
        # llist type: list[int]
        # list of layers by their layer No. (index in `Network`, which is a
        # list of `Layer`s)
        self.llist = llist
        self.llist_flag = True
        if llist==None:
            self.llist_flag = False

    def get_metadata(self, elem, batch_size):
        """
        Metadata is neural network-specific information.
        Not certifier-specific information.
        """
        if dummy_mode:
            return self.get_metadata_dummy(elem, batch_size)
        # ---- Is this true, and why? ----
        # Currently, get_metadata only supports consecutive intervals of
        # layers. Must coalesce successfully first.
        self.coalesce()
        if not self.llist_flag:
            # type of ret: list[gbscr.sparse_block.SparseBlock]
            ret = []
            start_indices = []
            temp = 0
            for k in range(self.start, self.end):
                if elem == 'weight' or elem == 'w':
                    if self.network[k].type == LayerType.Linear:
                        block = DenseBlock(self.network[k].weight)
                        if not self.network[k].last_layer:
                            for i in range(len(self.initial_shape)):
                                block = block.unsqueeze(0)
                            repeat_dims = [batch_size]
                            for i in range(len(block.total_shape)-1):
                                repeat_dims.append(1)
                            repeat_dims = torch.tensor(repeat_dims)
                            block = block.repeat(repeat_dims)
                        ret.append(block)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].weight.shape[0]
                    elif self.network[k].type == LayerType.Conv2D:
                        ix, iy = self.network[self.network[k].parents[0]].shape[-2:]
                        ox, oy = self.network[k].shape[-2:]
                        sx, sy = self.network[k].stride
                        px, py = self.network[k].padding
                        block = KernelBlock(self.network[k].weight, torch.tensor([self.network[k].size, self.network[self.network[k].parents[0]].size]), ix, iy, ox, oy, sx, sy, px, py)
                        if self.network.no_sparsity:
                            block = DenseBlock(block.get_dense().squeeze(0))
                        for i in range(len(self.initial_shape)):
                            block = block.unsqueeze(0)
                        repeat_dims = [batch_size]
                        for i in range(len(block.total_shape)-1):
                            repeat_dims.append(1)
                        repeat_dims = torch.tensor(repeat_dims)
                        block = block.repeat(repeat_dims)
                        ret.append(block)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].size
                    else:
                        raise NotImplementedError
                elif elem == 'bias' or elem == 'b':
                    block = DenseBlock(self.network[k].bias)
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'layer':
                    # When elem == 'layer', give the layer number of the layer
                    #   which the neuron belongs to.
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    block = ConstBlock(k, torch.tensor([self.network[k].size]))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'last_layer':
                    # When elem == 'last_layer', give whether the neuron
                    #   belongs to the last layer or not.
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    mat = (k == len(self.network)-1)
                    block = ConstBlock(int(mat), torch.tensor([self.network[k].size]))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                else:
                    raise NotImplementedError
            total_shape = start_indices[-1] + ret[-1].total_shape
            dim = len(total_shape)
            return SparseTensor(start_indices, ret, dim, total_shape)
        else:
            raise NotImplementedError
    
    def get_metadata_dummy(self, elem, batch_size):
        """
        Compile-time counterpart of `get_metadata`.
        All sparse blocks replaced with dummy blocks.
        Shape information
        """
        self.coalesce()
        # print(self.start, self.end)
        if not self.llist_flag:
            # type of ret: list[gbscr.sparse_block.SparseBlock]
            ret = []
            start_indices = []
            temp = 0
            for k in range(self.start, self.end):
                if elem == 'weight' or elem == 'w':
                    if self.network[k].type == LayerType.Linear:
                        block = DummyBlock(None, torch.tensor(self.network[k].weight.shape))
                        if not self.network[k].last_layer:
                            for i in range(len(self.initial_shape)):
                                block = block.unsqueeze(0)
                            repeat_dims = [batch_size]
                            for i in range(len(block.total_shape)-1):
                                repeat_dims.append(1)
                            repeat_dims = torch.tensor(repeat_dims)
                            block = block.repeat(repeat_dims)
                        ret.append(block)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].weight.shape[0]
                    elif self.network[k].type == LayerType.Conv2D:
                        ix, iy = self.network[self.network[k].parents[0]].shape[-2:]
                        ox, oy = self.network[k].shape[-2:]
                        sx, sy = self.network[k].stride
                        px, py = self.network[k].padding
                        block = DummyBlock(None, torch.tensor([self.network[k].size, self.network[self.network[k].parents[0]].size]))
                        if self.network.no_sparsity:
                            block = DummyBlock(None, torch.tensor(block.get_dense().squeeze(0).shape))
                        for i in range(len(self.initial_shape)):
                            block = block.unsqueeze(0)
                        repeat_dims = [batch_size]
                        for i in range(len(block.total_shape)-1):
                            repeat_dims.append(1)
                        repeat_dims = torch.tensor(repeat_dims)
                        block = block.repeat(repeat_dims)
                        ret.append(block)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].size
                    else:
                        raise NotImplementedError
                elif elem == 'bias' or elem == 'b':
                    block = DummyBlock(None, torch.tensor(self.network[k].bias.shape))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'layer':
                    # When elem == 'layer', give the layer number of the layer
                    #   which the neuron belongs to.
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    block = DummyBlock(None, torch.tensor([self.network[k].size]))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'last_layer':
                    # When elem == 'last_layer', give whether the neuron
                    #   belongs to the last layer or not.
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    mat = (k == len(self.network)-1)
                    block = DummyBlock(None, torch.tensor([self.network[k].size]))
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    ret.append(block)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                else:
                    raise NotImplementedError
            total_shape = start_indices[-1] + ret[-1].total_shape
            dim = len(total_shape)
            return SparseTensor(start_indices, ret, dim, total_shape)
        else:
            raise NotImplementedError
        
    def coalesce(self):
        if not self.llist_flag:
            return True
        # Can coalesce only if currently operated layers (llist) are
        # consecutive.
        for i in range(len(self.llist)-1):
            if self.llist[i]!=self.llist[i+1]-1:
                return False
        self.start = self.llist[0]
        self.end = self.llist[-1]+1
        self.llist_flag = False
        return True
    
    def decoalesce(self):
        if self.llist_flag:
            return True
        self.llist = []
        for i in range(self.start, self.end):
            self.llist.append(i)
        self.llist_flag = True
        return True
                
    def dot(self, mats, total_size):
        if not isinstance(mats, list):
            mats = [mats]
        else:
            assert(False)
        initial_shape = self.initial_shape
        polyexp_const = SparseTensor([], [], 1, torch.tensor([1]))
        polyexp_const = 0.0
        if self.llist_flag:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in self.llist]
        else:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in range(self.start, self.end)]
        cols = 0
        for j, mat in enumerate(self.llist):
            if self.llist_flag:
                cols += self.network[self.llist[j]].size
            else:
                cols += self.network[self.start+j].size
        assert(mats[0].total_size[-1] == cols)
        
        initial_shape = []
        for j in range(len(self.initial_shape)):
            initial_shape.append(math.lcm(self.initial_shape[j], mats[0].total_size[j].item()))
        
        new_total_size = torch.tensor(initial_shape+[total_size])
        res_blocks = [i.copy() for i in mats[0].blocks]
        return PolyExpSparse(self.network, SparseTensor(start_indices, res_blocks, len(self.initial_shape)+1, new_total_size), polyexp_const)
    
    def convert_to_poly(self, abs_elem):
        mats = []
        start_indices = []
        index = 0
        if self.llist:
            for i in self.llist:
                mat = torch.ones(self.network[i].size).reshape(*self.initial_shape, self.network[i].size)
                mats.append(DiagonalBlock(mat, total_shape=torch.tensor([*self.initial_shape, self.network[i].size, self.network[i].size]), diag_index=len(self.initial_shape) + 1))
                start_indices.append(torch.tensor([0]*len(self.initial_shape) + [index, self.network[i].start]))
                index += self.network[i].size
        else:
            raise NotImplementedError
    
        polyexp_const = SparseTensor([], [], len(self.initial_shape)+1, torch.tensor(self.initial_shape+[index]))
        return PolyExpSparse(self.network, SparseTensor(start_indices, mats, len(self.initial_shape)+2, torch.tensor(self.initial_shape+[index, abs_elem.get_poly_size()])), polyexp_const)
        
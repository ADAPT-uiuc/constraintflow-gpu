import json
import os
import torch
from constraintflow.lib.polyexp import PolyExpSparse
from constraintflow.lib.symexp import *
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.lib.llist import Llist
from constraintflow.gbcsr.tensor_ops import *
class ibp:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):
		rewrite_new_0 = prev.dot(curr.get_metadata('weight', batch_size), abs_elem.get_poly_size())
		rewrite_new_1 = repeat(curr.get_metadata('bias', batch_size), torch.tensor([batch_size, 1]), layer_index = layer_index, counter = 2, inside_while = False, while_number = -1) #2
		rewrite_new_2 = binary(rewrite_new_0.get_const(), rewrite_new_1, operator.add, layer_index = layer_index, counter = 3, inside_while = False, while_number = -1) #3
		cse_var_11 = PolyExpSparse(abs_elem.network, rewrite_new_0.get_mat(abs_elem) , rewrite_new_2)
		cse_var_13 = Llist(abs_elem.network, [1]*(cse_var_11.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_12 = cse_var_11.get_mat(abs_elem)
		cse_var_16 = clamp(cse_var_12, 0, True, layer_index = layer_index, counter = 12, inside_while = False, while_number = -1) #12
		cse_var_18 = (abs_elem.get_elem('l', cse_var_13)).squeeze(1, layer_index = layer_index, counter = 16, inside_while = False, while_number = -1)
		rewrite_new_5 = inner_prod(cse_var_16, cse_var_18, layer_index = layer_index, counter = 6, inside_while = False, while_number = -1) #6
		cse_var_15 = clamp(cse_var_12, 0, False, layer_index = layer_index, counter = 15, inside_while = False, while_number = -1) #15
		cse_var_19 = (abs_elem.get_elem('u', cse_var_13)).squeeze(1, layer_index = layer_index, counter = 13, inside_while = False, while_number = -1)
		rewrite_new_8 = inner_prod(cse_var_15, cse_var_19, layer_index = layer_index, counter = 9, inside_while = False, while_number = -1) #9
		rewrite_new_9 = binary(rewrite_new_5, rewrite_new_8, operator.add, layer_index = layer_index, counter = 10, inside_while = False, while_number = -1) #10
		rewrite_new_10 = binary(rewrite_new_9, rewrite_new_2, operator.add, layer_index = layer_index, counter = 11, inside_while = False, while_number = -1) #11
		rewrite_new_13 = inner_prod(cse_var_16, cse_var_19, layer_index = layer_index, counter = 14, inside_while = False, while_number = -1) #14
		rewrite_new_16 = inner_prod(cse_var_15, cse_var_18, layer_index = layer_index, counter = 17, inside_while = False, while_number = -1) #17
		rewrite_new_17 = binary(rewrite_new_13, rewrite_new_16, operator.add, layer_index = layer_index, counter = 18, inside_while = False, while_number = -1) #18
		rewrite_new_18 = binary(rewrite_new_17, rewrite_new_2, operator.add, layer_index = layer_index, counter = 19, inside_while = False, while_number = -1) #19
		return rewrite_new_10, rewrite_new_18, 
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):
		cse_var_9 = SparseTensor([], [], 0, torch.tensor([]), dense_const=0, type= type(0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_10 = abs_elem.get_elem('l', prev)
		rewrite_new_21 = binary(cse_var_10, cse_var_9, operator.ge, layer_index = layer_index, counter = 20, inside_while = False, while_number = -1) #20
		rewrite_new_19 = convert_to_float(rewrite_new_21)
		rewrite_new_22 = binary(rewrite_new_19, cse_var_10, operator.mul, layer_index = layer_index, counter = 21, inside_while = False, while_number = -1) #21
		cse_var_17 = SparseTensor([], [], 0, torch.tensor([]), dense_const=1.0, type= type(1.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		rewrite_new_23 = binary(cse_var_17, rewrite_new_19, operator.sub, layer_index = layer_index, counter = 22, inside_while = False, while_number = -1) #22
		rewrite_new_24 = binary(rewrite_new_23, cse_var_9, operator.mul, layer_index = layer_index, counter = 23, inside_while = False, while_number = -1) #23
		rewrite_new_25 = binary(rewrite_new_22, rewrite_new_24, operator.add, layer_index = layer_index, counter = 24, inside_while = False, while_number = -1) #24
		cse_var_8 = abs_elem.get_elem('u', prev)
		rewrite_new_26 = binary(cse_var_8, cse_var_9, operator.ge, layer_index = layer_index, counter = 25, inside_while = False, while_number = -1) #25
		rewrite_new_20 = convert_to_float(rewrite_new_26)
		rewrite_new_27 = binary(rewrite_new_20, cse_var_8, operator.mul, layer_index = layer_index, counter = 26, inside_while = False, while_number = -1) #26
		rewrite_new_28 = binary(cse_var_17, rewrite_new_20, operator.sub, layer_index = layer_index, counter = 27, inside_while = False, while_number = -1) #27
		rewrite_new_29 = binary(rewrite_new_28, cse_var_9, operator.mul, layer_index = layer_index, counter = 28, inside_while = False, while_number = -1) #28
		rewrite_new_30 = binary(rewrite_new_27, rewrite_new_29, operator.add, layer_index = layer_index, counter = 29, inside_while = False, while_number = -1) #29
		return rewrite_new_25, rewrite_new_30, 
	
	def Sigmoid(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):
		rewrite_new_31 = unary(abs_elem.get_elem('l', prev), 'sigma', layer_index=layer_index, counter=30, inside_while=False, while_number=-1)
		rewrite_new_32 = unary(abs_elem.get_elem('u', prev), 'sigma', layer_index=layer_index, counter=31, inside_while=False, while_number=-1)
		return rewrite_new_31, rewrite_new_32, 
	

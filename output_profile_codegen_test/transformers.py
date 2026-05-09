import json
import os
import torch
from constraintflow.lib.polyexp import PolyExpSparse
from constraintflow.lib.symexp import *
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.lib.llist import Llist
from constraintflow.gbcsr.tensor_ops import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):
		if layer_index == 7:
			return self.Affine_7(abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = layer_index)
	
	def Affine_7(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):
		while_iteration = -1
		rewrite_new_0 = binary(curr.get_metadata('last_layer', batch_size), SparseTensor([], [], 0, torch.tensor([]), dense_const=1, type= type(1)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([1, curr_size])), operator.eq, layer_index = layer_index, counter = 1, inside_while = False, while_number = -1) #1
		rewrite_new_1 = any(rewrite_new_0)
		rewrite_new_2 = repeat(curr.get_metadata('bias', batch_size), torch.tensor([batch_size, 1]), layer_index = layer_index, counter = 2, inside_while = False, while_number = -1) #2
		rewrite_new_3 = prev.dot(curr.get_metadata('weight', batch_size), abs_elem.get_poly_size())
		rewrite_new_4 = binary(rewrite_new_3.get_const(), rewrite_new_2, operator.add, layer_index = layer_index, counter = 4, inside_while = False, while_number = -1) #4
		cse_var_71 = rewrite_new_3.get_mat(abs_elem)
		cse_var_45 = PolyExpSparse(abs_elem.network, cse_var_71 , rewrite_new_4)
		cse_var_113 = get_dims(cse_var_71)
		cse_var_115 = binary(cse_var_113, 1, operator.sub, layer_index = layer_index, counter = 15, inside_while = False, while_number = -1) #15
		vertices_stop1 = False
		cse_var_14 = SparseTensor([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size]))
		cse_var_117 = Llist(abs_elem.network, [1]*(cse_var_115), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist()).get_metadata('layer', batch_size)
		cse_var_116 = Llist(abs_elem.network, [1]*(cse_var_115), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_118 = abs_elem.get_elem('L', cse_var_116)
		cse_var_119 = abs_elem.get_elem('U', cse_var_116)
		cse_var_98 = cse_var_119.get_const()
		cse_var_94 = cse_var_119.get_mat(abs_elem)
		cse_var_100 = cse_var_118.get_const()
		cse_var_96 = cse_var_118.get_mat(abs_elem)
		phi_trav_exp1_2_1 = cse_var_45
		while_iteration = -1
		while(True):
			while_iteration += 1
			print('while_iteration', while_iteration)
			cse_var_28 = phi_trav_exp1_2_1.get_mat(abs_elem)
			rewrite_new_8 = binary(cse_var_28, cse_var_14, operator.ne, layer_index = layer_index, counter = 6, inside_while = True, while_number = 0, while_iteration=while_iteration) #6
			vertices_stop_default1 = get_default_stop([batch_size, curr_size, poly_size], abs_elem, batch_size, curr_size, poly_size)
			rewrite_new_9 = binary(SparseTensor([], [], 0, torch.tensor([]), dense_const=vertices_stop1, type= type(vertices_stop1)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size])), vertices_stop_default1, operator.or_, layer_index = layer_index, counter = 7, inside_while = True, while_number = 0, while_iteration=while_iteration) #7
			rewrite_new_10 = unary(rewrite_new_9, operator.not_)
			rewrite_new_11 = binary(rewrite_new_8, rewrite_new_10, operator.and_, layer_index = layer_index, counter = 8, inside_while = True, while_number = 0, while_iteration=while_iteration) #8
			if(unary(any(rewrite_new_11), operator.not_)):
				break
			priority_vertices1 = get_max_priority(cse_var_117, rewrite_new_11)
			rewrite_new_21 = unary(priority_vertices1, operator.not_)
			rewrite_new_22 = repeat(rewrite_new_21, torch.tensor([batch_size, curr_size, 1]), layer_index = layer_index, counter = 16, inside_while = True, while_number = 0, while_iteration=while_iteration) #16
			rewrite_new_23 = binary(rewrite_new_9, rewrite_new_22, operator.or_, layer_index = layer_index, counter = 17, inside_while = True, while_number = 0, while_iteration=while_iteration) #17
			polyexp_stop1 = filter_trav_exp_stop(phi_trav_exp1_2_1, rewrite_new_23)
			polyexp_not_stop1 = filter_trav_exp_not_stop(phi_trav_exp1_2_1, rewrite_new_23)
			cse_var_107 = polyexp_not_stop1.get_mat(abs_elem)
			cse_var_104 = cse_var_107.unsqueeze(3)
			cse_var_101 = clamp(cse_var_107, 0, True, layer_index = layer_index, counter = 44, inside_while = True, while_number = 0, while_iteration=while_iteration) #44
			cse_var_99 = clamp(cse_var_107, 0, False, layer_index = layer_index, counter = 46, inside_while = True, while_number = 0, while_iteration=while_iteration) #46
			cse_var_97 = clamp(cse_var_104, 0, True, layer_index = layer_index, counter = 49, inside_while = True, while_number = 0, while_iteration=while_iteration) #49
			cse_var_95 = clamp(cse_var_104, 0, False, layer_index = layer_index, counter = 51, inside_while = True, while_number = 0, while_iteration=while_iteration) #51
			rewrite_new_51 = inner_prod(cse_var_101, cse_var_100.squeeze(1), layer_index = layer_index, counter = 45, inside_while = True, while_number = 0, while_iteration=while_iteration) #45
			rewrite_new_53 = inner_prod(cse_var_99, cse_var_98.squeeze(1), layer_index = layer_index, counter = 47, inside_while = True, while_number = 0, while_iteration=while_iteration) #47
			rewrite_new_54 = binary(rewrite_new_51, rewrite_new_53, operator.add, layer_index = layer_index, counter = 48, inside_while = True, while_number = 0, while_iteration=while_iteration) #48
			rewrite_new_56 = inner_prod(cse_var_97.squeeze(3), cse_var_96.squeeze(1), layer_index = layer_index, counter = 50, inside_while = True, while_number = 0, while_iteration=while_iteration) #50
			rewrite_new_58 = inner_prod(cse_var_95.squeeze(3), cse_var_94.squeeze(1), layer_index = layer_index, counter = 52, inside_while = True, while_number = 0, while_iteration=while_iteration) #52
			rewrite_new_59 = binary(rewrite_new_56, rewrite_new_58, operator.add, layer_index = layer_index, counter = 53, inside_while = True, while_number = 0, while_iteration=while_iteration) #53
			rewrite_new_60 = binary(rewrite_new_54, polyexp_not_stop1.get_const(), operator.add, layer_index = layer_index, counter = 54, inside_while = True, while_number = 0, while_iteration=while_iteration) #54
			rewrite_new_61 = binary(rewrite_new_59, polyexp_stop1.get_mat(abs_elem), operator.add, layer_index = layer_index, counter = 55, inside_while = True, while_number = 0, while_iteration=while_iteration) #55
			rewrite_new_62 = binary(rewrite_new_60, polyexp_stop1.get_const(), operator.add, layer_index = layer_index, counter = 56, inside_while = True, while_number = 0, while_iteration=while_iteration) #56
			trav_exp1_5_3 = PolyExpSparse(abs_elem.network, rewrite_new_61 , rewrite_new_62)
			phi_trav_exp1_2_1 = trav_exp1_5_3
		json_obj = {"num_iterations": while_iteration}
		os.makedirs("jit_while", exist_ok=True)
		capture_path = "jit_while/while_iterations_layer_" + str(layer_index) + "_while_" + str(0) + ".json"
		with open(capture_path, "w") as json_file:
			print(f"Capturing while iteration count to {capture_path}")
			json.dump(json_obj, json_file)
		cse_var_26 = Llist(abs_elem.network, [1]*(phi_trav_exp1_2_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_108 = phi_trav_exp1_2_1.get_mat(abs_elem)
		rewrite_new_12 = clamp(cse_var_108, 0, True, layer_index = layer_index, counter = 9, inside_while = False, while_number = -1) #9
		rewrite_new_13 = inner_prod(rewrite_new_12, abs_elem.get_elem('l', cse_var_26).squeeze(1), layer_index = layer_index, counter = 10, inside_while = False, while_number = -1) #10
		rewrite_new_14 = clamp(cse_var_108, 0, False, layer_index = layer_index, counter = 11, inside_while = False, while_number = -1) #11
		rewrite_new_15 = inner_prod(rewrite_new_14, abs_elem.get_elem('u', cse_var_26).squeeze(1), layer_index = layer_index, counter = 12, inside_while = False, while_number = -1) #12
		rewrite_new_16 = binary(rewrite_new_13, rewrite_new_15, operator.add, layer_index = layer_index, counter = 13, inside_while = False, while_number = -1) #13
		rewrite_new_17 = binary(rewrite_new_16, phi_trav_exp1_2_1.get_const(), operator.add, layer_index = layer_index, counter = 14, inside_while = False, while_number = -1) #14
		vertices_stop2 = False
		cse_var_83 = cse_var_118.get_const()
		cse_var_79 = cse_var_118.get_mat(abs_elem)
		cse_var_85 = cse_var_119.get_const()
		cse_var_81 = cse_var_119.get_mat(abs_elem)
		phi_trav_exp2_6_1 = cse_var_45
		while_iteration = -1
		while(True):
			while_iteration += 1
			print('while_iteration', while_iteration)
			cse_var_15 = phi_trav_exp2_6_1.get_mat(abs_elem)
			rewrite_new_64 = binary(cse_var_15, cse_var_14, operator.ne, layer_index = layer_index, counter = 57, inside_while = True, while_number = 1, while_iteration=while_iteration) #57
			vertices_stop_default2 = get_default_stop([batch_size, curr_size, poly_size], abs_elem, batch_size, curr_size, poly_size)
			rewrite_new_65 = binary(SparseTensor([], [], 0, torch.tensor([]), dense_const=vertices_stop2, type= type(vertices_stop2)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size])), vertices_stop_default2, operator.or_, layer_index = layer_index, counter = 58, inside_while = True, while_number = 1, while_iteration=while_iteration) #58
			rewrite_new_66 = unary(rewrite_new_65, operator.not_)
			rewrite_new_67 = binary(rewrite_new_64, rewrite_new_66, operator.and_, layer_index = layer_index, counter = 59, inside_while = True, while_number = 1, while_iteration=while_iteration) #59
			if(unary(any(rewrite_new_67), operator.not_)):
				break
			priority_vertices2 = get_max_priority(cse_var_117, rewrite_new_67)
			rewrite_new_74 = unary(priority_vertices2, operator.not_)
			rewrite_new_75 = repeat(rewrite_new_74, torch.tensor([batch_size, curr_size, 1]), layer_index = layer_index, counter = 66, inside_while = True, while_number = 1, while_iteration=while_iteration) #66
			rewrite_new_76 = binary(rewrite_new_65, rewrite_new_75, operator.or_, layer_index = layer_index, counter = 67, inside_while = True, while_number = 1, while_iteration=while_iteration) #67
			polyexp_stop2 = filter_trav_exp_stop(phi_trav_exp2_6_1, rewrite_new_76)
			polyexp_not_stop2 = filter_trav_exp_not_stop(phi_trav_exp2_6_1, rewrite_new_76)
			cse_var_92 = polyexp_not_stop2.get_mat(abs_elem)
			cse_var_89 = cse_var_92.unsqueeze(3)
			cse_var_86 = clamp(cse_var_92, 0, True, layer_index = layer_index, counter = 94, inside_while = True, while_number = 1, while_iteration=while_iteration) #94
			cse_var_84 = clamp(cse_var_92, 0, False, layer_index = layer_index, counter = 96, inside_while = True, while_number = 1, while_iteration=while_iteration) #96
			cse_var_82 = clamp(cse_var_89, 0, True, layer_index = layer_index, counter = 99, inside_while = True, while_number = 1, while_iteration=while_iteration) #99
			cse_var_80 = clamp(cse_var_89, 0, False, layer_index = layer_index, counter = 101, inside_while = True, while_number = 1, while_iteration=while_iteration) #101
			rewrite_new_104 = inner_prod(cse_var_86, cse_var_85.squeeze(1), layer_index = layer_index, counter = 95, inside_while = True, while_number = 1, while_iteration=while_iteration) #95
			rewrite_new_106 = inner_prod(cse_var_84, cse_var_83.squeeze(1), layer_index = layer_index, counter = 97, inside_while = True, while_number = 1, while_iteration=while_iteration) #97
			rewrite_new_107 = binary(rewrite_new_104, rewrite_new_106, operator.add, layer_index = layer_index, counter = 98, inside_while = True, while_number = 1, while_iteration=while_iteration) #98
			rewrite_new_109 = inner_prod(cse_var_82.squeeze(3), cse_var_81.squeeze(1), layer_index = layer_index, counter = 100, inside_while = True, while_number = 1, while_iteration=while_iteration) #100
			rewrite_new_111 = inner_prod(cse_var_80.squeeze(3), cse_var_79.squeeze(1), layer_index = layer_index, counter = 102, inside_while = True, while_number = 1, while_iteration=while_iteration) #102
			rewrite_new_112 = binary(rewrite_new_109, rewrite_new_111, operator.add, layer_index = layer_index, counter = 103, inside_while = True, while_number = 1, while_iteration=while_iteration) #103
			rewrite_new_113 = binary(rewrite_new_107, polyexp_not_stop2.get_const(), operator.add, layer_index = layer_index, counter = 104, inside_while = True, while_number = 1, while_iteration=while_iteration) #104
			rewrite_new_114 = binary(rewrite_new_112, polyexp_stop2.get_mat(abs_elem), operator.add, layer_index = layer_index, counter = 105, inside_while = True, while_number = 1, while_iteration=while_iteration) #105
			rewrite_new_115 = binary(rewrite_new_113, polyexp_stop2.get_const(), operator.add, layer_index = layer_index, counter = 106, inside_while = True, while_number = 1, while_iteration=while_iteration) #106
			trav_exp2_9_3 = PolyExpSparse(abs_elem.network, rewrite_new_114 , rewrite_new_115)
			phi_trav_exp2_6_1 = trav_exp2_9_3
		json_obj = {"num_iterations": while_iteration}
		os.makedirs("jit_while", exist_ok=True)
		capture_path = "jit_while/while_iterations_layer_" + str(layer_index) + "_while_" + str(1) + ".json"
		with open(capture_path, "w") as json_file:
			print(f"Capturing while iteration count to {capture_path}")
			json.dump(json_obj, json_file)
		cse_var_12 = Llist(abs_elem.network, [1]*(phi_trav_exp2_6_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_93 = phi_trav_exp2_6_1.get_mat(abs_elem)
		rewrite_new_68 = clamp(cse_var_93, 0, True, layer_index = layer_index, counter = 60, inside_while = False, while_number = -1) #60
		rewrite_new_69 = inner_prod(rewrite_new_68, abs_elem.get_elem('u', cse_var_12).squeeze(1), layer_index = layer_index, counter = 61, inside_while = False, while_number = -1) #61
		rewrite_new_70 = clamp(cse_var_93, 0, False, layer_index = layer_index, counter = 62, inside_while = False, while_number = -1) #62
		rewrite_new_71 = inner_prod(rewrite_new_70, abs_elem.get_elem('l', cse_var_12).squeeze(1), layer_index = layer_index, counter = 63, inside_while = False, while_number = -1) #63
		rewrite_new_72 = binary(rewrite_new_69, rewrite_new_71, operator.add, layer_index = layer_index, counter = 64, inside_while = False, while_number = -1) #64
		rewrite_new_73 = binary(rewrite_new_72, phi_trav_exp2_6_1.get_const(), operator.add, layer_index = layer_index, counter = 65, inside_while = False, while_number = -1) #65
		return rewrite_new_17, rewrite_new_73, cse_var_45, cse_var_45
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):
		cse_var_42 = abs_elem.get_elem('l', prev)
		cse_var_43 = SparseTensor([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		rewrite_new_132 = binary(cse_var_42, cse_var_43, operator.ge, layer_index = layer_index, counter = 119, inside_while = False, while_number = -1) #119
		rewrite_new_128 = convert_to_float(rewrite_new_132)
		rewrite_new_133 = binary(rewrite_new_128, cse_var_42, operator.mul, layer_index = layer_index, counter = 120, inside_while = False, while_number = -1) #120
		cse_var_110 = SparseTensor([], [], 0, torch.tensor([]), dense_const=1.0, type= type(1.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		rewrite_new_134 = binary(cse_var_110, rewrite_new_128, operator.sub, layer_index = layer_index, counter = 121, inside_while = False, while_number = -1) #121
		rewrite_new_135 = binary(rewrite_new_134, cse_var_43, operator.mul, layer_index = layer_index, counter = 122, inside_while = False, while_number = -1) #122
		rewrite_new_136 = binary(rewrite_new_133, rewrite_new_135, operator.add, layer_index = layer_index, counter = 123, inside_while = False, while_number = -1) #123
		cse_var_41 = abs_elem.get_elem('u', prev)
		rewrite_new_137 = binary(cse_var_41, cse_var_43, operator.ge, layer_index = layer_index, counter = 124, inside_while = False, while_number = -1) #124
		rewrite_new_129 = convert_to_float(rewrite_new_137)
		rewrite_new_138 = binary(rewrite_new_129, cse_var_41, operator.mul, layer_index = layer_index, counter = 125, inside_while = False, while_number = -1) #125
		rewrite_new_139 = binary(cse_var_110, rewrite_new_129, operator.sub, layer_index = layer_index, counter = 126, inside_while = False, while_number = -1) #126
		rewrite_new_140 = binary(rewrite_new_139, cse_var_43, operator.mul, layer_index = layer_index, counter = 127, inside_while = False, while_number = -1) #127
		rewrite_new_141 = binary(rewrite_new_138, rewrite_new_140, operator.add, layer_index = layer_index, counter = 128, inside_while = False, while_number = -1) #128
		cse_var_39 = prev.convert_to_poly(abs_elem)
		rewrite_new_142 = repeat(cse_var_39.get_const(), torch.tensor([batch_size, 1]), layer_index = layer_index, counter = 129, inside_while = False, while_number = -1) #129
		rewrite_new_143 = repeat(cse_var_39.get_mat(abs_elem), torch.tensor([batch_size, 1, 1]), layer_index = layer_index, counter = 130, inside_while = False, while_number = -1) #130
		cse_var_38 = PolyExpSparse(abs_elem.network, 0.0, 0.0)
		rewrite_new_144 = binary(cse_var_42, cse_var_41, operator.add, layer_index = layer_index, counter = 131, inside_while = False, while_number = -1) #131
		rewrite_new_145 = binary(rewrite_new_144, cse_var_43, operator.ge, layer_index = layer_index, counter = 132, inside_while = False, while_number = -1) #132
		cse_var_109 = convert_to_float(rewrite_new_145)
		rewrite_new_146 = repeat(cse_var_109.unsqueeze(2), torch.tensor([1, 1, poly_size]), layer_index = layer_index, counter = 133, inside_while = False, while_number = -1) #133
		rewrite_new_147 = binary(rewrite_new_146, rewrite_new_143, operator.mul, layer_index = layer_index, counter = 134, inside_while = False, while_number = -1) #134
		cse_var_114 = binary(cse_var_110, cse_var_109, operator.sub, layer_index = layer_index, counter = 140, inside_while = False, while_number = -1) #140
		rewrite_new_149 = binary(cse_var_114, SparseTensor([], [], 0, torch.tensor([]), dense_const=cse_var_38.get_mat(abs_elem), type= type(cse_var_38.get_mat(abs_elem))).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size])), operator.mul, layer_index = layer_index, counter = 136, inside_while = False, while_number = -1) #136
		rewrite_new_150 = repeat(rewrite_new_149.unsqueeze(2), torch.tensor([1, 1, poly_size]), layer_index = layer_index, counter = 137, inside_while = False, while_number = -1) #137
		rewrite_new_151 = binary(rewrite_new_147, rewrite_new_150, operator.add, layer_index = layer_index, counter = 138, inside_while = False, while_number = -1) #138
		rewrite_new_152 = binary(cse_var_109, rewrite_new_142, operator.mul, layer_index = layer_index, counter = 139, inside_while = False, while_number = -1) #139
		rewrite_new_154 = binary(cse_var_114, SparseTensor([], [], 0, torch.tensor([]), dense_const=cse_var_38.get_const(), type= type(cse_var_38.get_const())).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size])), operator.mul, layer_index = layer_index, counter = 141, inside_while = False, while_number = -1) #141
		rewrite_new_155 = binary(rewrite_new_152, rewrite_new_154, operator.add, layer_index = layer_index, counter = 142, inside_while = False, while_number = -1) #142
		L_new = PolyExpSparse(abs_elem.network, rewrite_new_151 , rewrite_new_155)
		rewrite_new_156 = binary(cse_var_41, cse_var_42, operator.sub, layer_index = layer_index, counter = 143, inside_while = False, while_number = -1) #143
		rewrite_new_157 = binary(cse_var_41, rewrite_new_156, operator.truediv, layer_index = layer_index, counter = 144, inside_while = False, while_number = -1) #144
		rewrite_new_158 = binary(rewrite_new_157, rewrite_new_142, operator.mul, layer_index = layer_index, counter = 145, inside_while = False, while_number = -1) #145
		rewrite_new_159 = repeat(rewrite_new_157.unsqueeze(2), torch.tensor([1, 1, poly_size]), layer_index = layer_index, counter = 146, inside_while = False, while_number = -1) #146
		rewrite_new_160 = binary(rewrite_new_159, rewrite_new_143, operator.mul, layer_index = layer_index, counter = 147, inside_while = False, while_number = -1) #147
		rewrite_new_161 = binary(cse_var_41, cse_var_42, operator.mul, layer_index = layer_index, counter = 148, inside_while = False, while_number = -1) #148
		rewrite_new_162 = binary(rewrite_new_161, rewrite_new_156, operator.truediv, layer_index = layer_index, counter = 149, inside_while = False, while_number = -1) #149
		rewrite_new_163 = binary(rewrite_new_158, rewrite_new_162, operator.sub, layer_index = layer_index, counter = 150, inside_while = False, while_number = -1) #150
		cse_var_33 = PolyExpSparse(abs_elem.network, 0.0, cse_var_43)
		rewrite_new_164 = binary(cse_var_41, cse_var_43, operator.le, layer_index = layer_index, counter = 151, inside_while = False, while_number = -1) #151
		cse_var_72 = where(rewrite_new_164, cse_var_33.get_const(), rewrite_new_163)
		rewrite_new_165 = repeat(rewrite_new_164.unsqueeze(2), torch.tensor([1, 1, poly_size]), layer_index = layer_index, counter = 152, inside_while = False, while_number = -1) #152
		cse_var_73 = where(rewrite_new_165, cse_var_33.get_mat(abs_elem), rewrite_new_160)
		rewrite_new_166 = repeat(rewrite_new_132.unsqueeze(2), torch.tensor([1, 1, poly_size]), layer_index = layer_index, counter = 153, inside_while = False, while_number = -1) #153
		U_new = PolyExpSparse(abs_elem.network, where(rewrite_new_166, rewrite_new_143, cse_var_73) , where(rewrite_new_132, rewrite_new_142, cse_var_72))
		return rewrite_new_136, rewrite_new_141, L_new, U_new
	

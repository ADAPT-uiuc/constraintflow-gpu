import sys
import os
from constraintflow.lib.spec import *
from constraintflow.lib.flow_sparse import Flow
from constraintflow.lib.abs_elem import Abs_elem_sparse
from constraintflow.lib.symexp import *
from transformers import *


def run(network_file, batch_size, eps, dataset_X, dataset_y, dataset, train, print_intermediate_results, no_sparsity):
	network, l, u, L, U, Z, llist = get_network_and_input_spec(network_file, batch_size, dataset_X, dataset_y, dataset, eps=eps, train=train, no_sparsity=no_sparsity)
	abs_elem = Abs_elem_sparse({'llist' : llist, 'l' : l, 'u' : u, 'L' : L, 'U' : U}, {'l': 'Float', 'u': 'Float', 'L': 'PolyExp', 'U': 'PolyExp', 'llist': 'bool'}, network, batch_size=batch_size, no_sparsity=no_sparsity)
	flow = Flow(abs_elem, deeppoly(), network, print_intermediate_results, no_sparsity)
	return flow.flow()

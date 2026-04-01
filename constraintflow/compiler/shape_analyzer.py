from constraintflow.lib.network import Layer, LayerType, Network
from constraintflow.lib.flow_sparse import Flow
from constraintflow.gbcsr.sparse_block import DummyBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.lib.abs_elem import Abs_elem_sparse
import importlib

import torch


class NetworkAnalyzer:
    def __init__(self, network: Network, transformer_name: str) -> None:
        self.network = network
        self.transformer_module_name = 'transformers'
        self.transformer_module = importlib.import_module(
            self.transformer_module_name)
        self.transformer_class = getattr(self.transformer_module, transformer_name)
        self.transformer = self.transformer_class()

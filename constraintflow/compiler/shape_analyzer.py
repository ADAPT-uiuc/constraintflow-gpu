from constraintflow.lib.network import Layer, LayerType, Network
from constraintflow.gbcsr.sparse_block import DummyBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor
import torch


class NetworkAnalyzer:
    def __init__(self, network: Network) -> None:
        self.network = network

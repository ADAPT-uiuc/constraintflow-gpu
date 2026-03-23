from constraintflow.lib.network import Layer, LayerType, Network
from constraintflow.gbcsr.sparse_block import DummyBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor


class NetworkAnalyzer:
    def __init__(self, network: Network, initial_shape: list[int]) -> None:
        self.network: Network = network
        self.initial_shape: list[int] = initial_shape
        self.start_layer: int = 0
        self.end_layer: int = network.size

    def analyze_shapes_of_blocks(self, elem: str, batch_size: int) \
            -> list[SparseTensor]:
        ret: list[SparseTensor] = []
        for k in range(self.start_layer, self.end_layer):
            if elem == 'weight' or elem == 'w':
                pass
            elif elem == 'bias' or elem == 'b':
                pass
        assert False, 'TODO'

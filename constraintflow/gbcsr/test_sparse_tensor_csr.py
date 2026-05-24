"""Step 1: CSR init round-trips against sparse_tensor_original."""

import copy

import torch

from constraintflow.gbcsr.sparse_block import ConstBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor, convert_dense_to_sparse
from constraintflow.gbcsr.sparse_tensor_original import (
    SparseTensor as OriginalSparseTensor,
    convert_dense_to_sparse as original_convert,
)


def test_get_dense_matches_original_const_blocks():
    total_size = torch.tensor([2, 3])
    blocks = [
        ConstBlock(1.0, torch.tensor([1, 2])),
        ConstBlock(2.0, torch.tensor([1, 1])),
    ]
    start_indices = [torch.tensor([0, 0]), torch.tensor([1, 2])]
    end_indices = [torch.tensor([1, 2]), torch.tensor([2, 3])]
    csr = SparseTensor(
        start_indices, blocks, 2, total_size, end_indices, type=float, dense_const=0.0
    )
    orig = OriginalSparseTensor(
        start_indices, blocks, 2, total_size, end_indices, type=float, dense_const=0.0
    )
    assert torch.allclose(csr.get_dense(), orig.get_dense())
    assert csr.num_blocks == 2
    assert not csr._csr_native


def test_deepcopy_preserves_csr_and_dense():
    import torch

    from constraintflow.lib.spec import create_sparse_init

    x = torch.randn(1, 784)
    l = create_sparse_init(x, float("-inf"), 1, 784, False)
    l2 = copy.deepcopy(l)
    assert torch.allclose(l.get_dense(), l2.get_dense())
    assert l2._csr is not None
    assert l._csr.values().numel() == l2._csr.values().numel()


def test_convert_dense_to_sparse_roundtrip():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    csr = convert_dense_to_sparse(x)
    orig = original_convert(x)
    assert torch.allclose(csr.get_dense(), orig.get_dense())
    assert torch.allclose(csr.get_dense(), x.to(dtype=csr.dtype))

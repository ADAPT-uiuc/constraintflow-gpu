"""
CSR-backed SparseTensor — built incrementally.

Step 1: __init__ accepts the original block-list format and stores CSR.
        get_dense() decodes CSR for verification; other ops come later.
"""

import copy
import operator
import time

import torch

from constraintflow.gbcsr.sparse_block import *  # noqa: F403 — re-export for `import *` callers
from constraintflow.gbcsr.sparse_block import DenseBlock, get_slice
from constraintflow.lib.globals import sparse_tensor_init_time

_TYPE_TO_DTYPE = {
    float: torch.float64,  # match sparse_tensor_original (dtype=self.type)
    int: torch.int64,
    bool: torch.bool,
}


def _dtype_for_type(t):
    return _TYPE_TO_DTYPE.get(t, torch.float32)


def _csr_rows_cols(total_size):
    total_size = torch.as_tensor(total_size)
    if total_size.numel() == 0:
        return 1, 1
    if len(total_size) == 1:
        return int(total_size[0]), 1
    rows = int(torch.prod(total_size[:-1]).item())
    cols = int(total_size[-1].item())
    return rows, cols


def _reshape2d(dense, total_size):
    rows, cols = _csr_rows_cols(total_size)
    return dense.reshape(rows, cols), rows, cols


def _unreshape2d(flat2d, total_size):
    return flat2d.reshape(list(map(int, torch.as_tensor(total_size).tolist())))


def _dense_from_blocks(start_indices, blocks, end_indices, total_size, type_, dense_const):
    """Same materialization as sparse_tensor_original.SparseTensor.get_dense()."""
    shape = list(map(int, torch.as_tensor(total_size).tolist()))
    if len(shape) == 0:
        shape = [1]
    res = torch.ones(shape, dtype=type_) * dense_const
    for i in range(len(blocks)):
        s = get_slice(start_indices[i], end_indices[i])
        res[tuple(s)] = blocks[i].get_dense()
    return res


def _csr_from_dense2d(dense2d, dense_const):
    mask = dense2d != dense_const
    if not mask.any():
        return None
    indices = mask.nonzero(as_tuple=False).t()
    values = dense2d[mask]
    coo = torch.sparse_coo_tensor(indices, values, size=dense2d.shape).coalesce()
    return coo.to_sparse_csr()


def _clone_csr(csr):
    if csr is None:
        return None
    coo = csr.to_sparse_coo()
    cloned = torch.sparse_coo_tensor(
        coo.indices().clone(),
        coo.values().clone(),
        size=coo.shape,
        dtype=coo.dtype,
        device=coo.device,
    ).coalesce()
    return cloned.to_sparse_csr()


def _dense2d_from_csr(csr, dense_const, dtype, rows, cols):
    dense2d = torch.full((rows, cols), fill_value=dense_const, dtype=dtype)
    if csr is None:
        return dense2d
    coo = csr.to_sparse_coo()
    dense2d[coo.indices()[0], coo.indices()[1]] = coo.values().to(dtype)
    return dense2d


def _default_end_indices(start_indices, blocks, total_size):
    end_indices = []
    for i, block in enumerate(blocks):
        end = start_indices[i] + torch.as_tensor(block.total_shape)
        end_indices.append(end)
        assert (end <= torch.as_tensor(total_size)).all()
    return end_indices


def convert_dense_to_sparse(x, total_shape=None, json_list=None, x_index=-1, **_kwargs):
    if isinstance(x, torch.Tensor):
        t = float
        dense_const = 0.0
        if x.dtype == torch.bool:
            t = bool
            dense_const = False
        return SparseTensor(
            [torch.zeros(x.dim(), dtype=torch.int64)],
            [DenseBlock(x)],
            x.dim(),
            torch.as_tensor(x.shape),
            type=t,
            dense_const=dense_const,
        )
    if isinstance(x, float):
        return SparseTensor([], [], len(total_shape), total_shape, type=float, dense_const=x)
    if isinstance(x, int):
        return SparseTensor([], [], len(total_shape), total_shape, type=int, dense_const=x)
    if isinstance(x, bool):
        return SparseTensor([], [], len(total_shape), total_shape, type=bool, dense_const=x)
    raise Exception("TYPE MISMATCH")


_OP_MAP = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "or_": operator.or_,
    "and_": operator.and_,
}


def get_operator_func(name: str):
    try:
        return _OP_MAP[name]
    except KeyError:
        raise ValueError(f"Unsupported operator: {name}")


class SparseTensor:
    """Ingress: original block-list layout. Internal storage: PyTorch CSR."""

    def __init__(
        self,
        start_indices,
        blocks,
        dims,
        total_size,
        end_indices=None,
        type=float,
        dense_const=0.0,
    ):
        t1 = time.time()
        self.start_indices = start_indices
        self.blocks = blocks
        self.total_size = torch.as_tensor(total_size, dtype=torch.int64)
        self.dims = dims
        self.num_blocks = len(start_indices)
        self.type = type
        self.dense_const = dense_const
        self.dtype = _dtype_for_type(type)
        self._csr_native = False

        if end_indices is None:
            end_indices = _default_end_indices(start_indices, blocks, self.total_size)
        self.end_indices = end_indices

        if self.num_blocks > 0:
            if (self.end_indices[0] - self.start_indices[0] == self.total_size).all():
                if self.type == float:
                    self.dense_const = 0.0
                elif self.type == bool:
                    self.dense_const = False

        dense = _dense_from_blocks(
            self.start_indices,
            self.blocks,
            self.end_indices,
            self.total_size,
            self.type,
            self.dense_const,
        )
        dense2d, self._csr_rows, self._csr_cols = _reshape2d(dense, self.total_size)
        self._csr = _csr_from_dense2d(dense2d, self.dense_const)

        sparse_tensor_init_time.update_total_time(time.time() - t1)

    def get_dense(self):
        dense2d = _dense2d_from_csr(
            self._csr, self.dense_const, self.dtype, self._csr_rows, self._csr_cols
        )
        return _unreshape2d(dense2d, self.total_size)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        out = object.__new__(SparseTensor)
        memo[id(self)] = out
        out.start_indices = copy.deepcopy(self.start_indices, memo)
        out.blocks = copy.deepcopy(self.blocks, memo)
        out.end_indices = copy.deepcopy(self.end_indices, memo)
        out.total_size = self.total_size.clone()
        out.dims = self.dims
        out.num_blocks = self.num_blocks
        out.type = self.type
        out.dense_const = self.dense_const
        out.dtype = self.dtype
        out._csr_native = self._csr_native
        out._csr_rows = self._csr_rows
        out._csr_cols = self._csr_cols
        out._csr = _clone_csr(self._csr)
        return out

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        nnz = 0 if self._csr is None else self._csr.values().numel()
        return (
            f"SparseTensor(csr) blocks={self.num_blocks} "
            f"shape={list(self.total_size.tolist())} nnz={nnz} "
            f"dense_const={self.dense_const}\n"
        )

"""
Real byte sizes for the scalarized flow block, by fake-executing it.

The block is straight-line tensor code with no data-dependent ops, so running each
rendered statement under FakeTensorMode gives exact shapes with no allocation --
cheaper and less code than shape rules for every IrTorch* node.
"""

import weakref
import operator

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from constraintflow.compiler.optimizations import flow_split
from constraintflow.lib.globals import device_mode
from constraintflow.gbcsr.sparse_block import patches_to_dense

_DTYPE = {'Bool': torch.bool, 'Int': torch.int64}


class ProbeSizes(dict):
    """Statement records plus storage metadata for parameters and constants."""
    def __init__(self):
        super().__init__()
        self.inputs = {}


class _Recorder(TorchDispatchMode):
    """Bytes allocated by every op, including unnamed intra-statement temporaries."""

    def __init__(self):
        self.bytes = []
        self.known_roots = set()
        self.live = {}
        self.temporary_peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t._base is None:
                self.bytes.append(t.numel() * t.element_size())
            if isinstance(t, torch.Tensor):
                storage = t.untyped_storage()
                root = storage._cdata
                if root not in self.known_roots:
                    refs, size = self.live.setdefault(root, ([], storage.nbytes()))
                    refs.append(weakref.ref(t))
        self.live = {root: ([r for r in refs if r() is not None], size)
                     for root, (refs, size) in self.live.items()
                     if any(r() is not None for r in refs)}
        self.temporary_peak = max(self.temporary_peak,
                                  sum(size for refs, size in self.live.values()))
        return out


def probe(stmts, sources, param_meta, batch_size, consts, prelude=()):
    """
    public
    Per-statement {'name', 'root', 'nbytes', 'alloc'}, keyed by statement index.
    `sources` are the rendered statements. Returns None if fake execution fails.
    """
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        return None
    device = device_mode.get_device()
    out = ProbeSizes()
    i = 0
    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            env = {'torch': torch, 'F': F, 'operator': operator,
                   'device_mode': device_mode, 'batch_size': batch_size,
                   'patches_to_dense': patches_to_dense}
            for name, (shape, type_) in param_meta.items():
                if shape is None:                 # scalar payload, not a tensor
                    env[name] = (type_ == 'Bool') and False or 0.0
                else:
                    env[name] = torch.empty(
                        shape, dtype=_DTYPE.get(type_, torch.float32), device=device)
            for text, name in consts.items():
                env[name] = eval(text, {'torch': torch})
            for text in prelude:               # view-write helpers
                exec(text, env)
            seen = {}
            inputs = set()
            for name, value in env.items():
                if isinstance(value, torch.Tensor):
                    storage = value.untyped_storage()
                    key = storage._cdata
                    seen.setdefault(key, (storage, 'input:' + name))
                    inputs.add(key)
                    out.inputs[name] = {'root': seen[key][1],
                                        'storage_bytes': storage.nbytes(),
                                        'shape': list(value.shape), 'dtype': str(value.dtype),
                                        'stride': list(value.stride()),
                                        'device': str(value.device)}
            recorder = _Recorder()
            for i, (stmt, src) in enumerate(zip(stmts, sources)):
                if not src or src.startswith('return'):
                    break
                recorder.bytes = []
                recorder.known_roots = set(seen)
                recorder.live = {}
                recorder.temporary_peak = 0
                with recorder:
                    exec(src, env)
                transient = sum(recorder.bytes)
                peak = max(recorder.bytes, default=0)
                name = flow_split._def_name(stmt)
                value = env.get(name) if name else None
                if not isinstance(value, torch.Tensor):
                    continue                  # in-place write allocates nothing
                storage = value.untyped_storage()
                key = storage._cdata
                fresh = key not in seen
                seen.setdefault(key, (storage, name + '@' + str(i)))
                root = seen[key][1]
                nbytes = value.numel() * value.element_size()
                out[i] = {'name': name, 'root': root, 'nbytes': nbytes,
                          'storage_bytes': storage.nbytes(), 'input': key in inputs,
                          'alloc': storage.nbytes() if fresh else 0,
                          'transient': transient, 'peak': peak,
                          'temporary_peak': recorder.temporary_peak,
                          'shape': list(value.shape), 'dtype': str(value.dtype),
                                        'stride': list(value.stride()),
                                        'device': str(value.device)}
    except Exception as e:
        print('[flow-sizes] fake execution failed at statement ' + str(i)
              + ': ' + type(e).__name__ + ': ' + str(e))
        return None
    return out


def _mb(n):
    return '{:.1f} MB'.format(n / 1024 ** 2)


def summary(stmts, sized):
    """One-line compile-log summary of a probe result."""
    views = sum(1 for v in sized.values() if not v['alloc'])
    total = sum(v['alloc'] for v in sized.values())
    peak = max(flow_split.live_bytes(
        stmts, *flow_split.root_liveness(stmts, sized)) or [0])
    transient = max((v['transient'] for v in sized.values()), default=0)
    op = max((v['peak'] for v in sized.values()), default=0)
    return ('{} statements, {} allocating / {} views, {} named total, {} named peak '
            'live, worst statement {} (largest op {})'.format(
                len(stmts), len(sized) - views, views,
                _mb(total), _mb(peak), _mb(transient), _mb(op)))


def peak_live_bytes(stmts, sized):
    """Named live storage plus simultaneously live intra-expression storage.

    Backend workspaces are deliberately separate: FakeTensor cannot measure
    cuDNN's algorithm-dependent allocations.
    """
    sizes, definitions, last = flow_split.root_liveness(stmts, sized)
    live = flow_split.live_bytes(stmts, sizes, definitions, last)
    return max((live[i] + sized.get(i, {}).get('temporary_peak', 0)
                for i in range(len(stmts))), default=0)

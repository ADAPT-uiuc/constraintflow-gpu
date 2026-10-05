"""Conversion at the Torch preparation / JAX kernel boundary only."""
import jax
import jax.numpy as jnp
import torch


def to_jax(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().contiguous()
        if value.dtype in (torch.float64, torch.int64, torch.complex128) and not jax.config.x64_enabled:
            raise ValueError('64-bit inputs require JAX_ENABLE_X64=1; refusing to silently narrow them')
        # CPU Torch tensors must stay on CPU even when JAX defaults to GPU.
        return jax.dlpack.from_dlpack(value)
    return jnp.asarray(value)


def run_flow(kernel, *args):
    converted = [a if isinstance(a, (int, float, bool)) else to_jax(a) for a in args]
    outputs = jax.block_until_ready(kernel(*converted))
    return tuple(torch.from_dlpack(x) for x in outputs)

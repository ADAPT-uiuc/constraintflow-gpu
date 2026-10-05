"""Pure JAX implementations of the tensor IR's layout-sensitive operations.

Shape arguments are compile-time Python values. This module deliberately has no
Torch dependency; the optional Torch bridge lives in jax_bridge.py.
"""
import jax.numpy as jnp
from jax import lax


def pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def squeeze(x, axis):
    # Torch leaves non-singleton dimensions alone, whereas jnp.squeeze raises.
    return jnp.squeeze(x, axis) if x.shape[axis] == 1 else x


def expand(x, shape):
    leading = len(shape) - x.ndim
    if leading < 0 or any(n == -1 for n in shape[:leading]):
        raise ValueError('expand cannot remove dimensions or infer new leading dimensions')
    return jnp.broadcast_to(x, tuple(x.shape[i - leading] if n == -1 else n
                                     for i, n in enumerate(shape)))


def pad(x, pads):
    widths = [(0, 0)] * x.ndim
    for i in range(len(pads) // 2):
        widths[-1 - i] = (pads[2 * i], pads[2 * i + 1])
    # Torch permits negative padding (crop before adding positive padding).
    slices = tuple(slice(max(-lo, 0), n - max(-hi, 0))
                   for n, (lo, hi) in zip(x.shape, widths))
    return jnp.pad(x[slices], [(max(lo, 0), max(hi, 0)) for lo, hi in widths])


def diag_embed(x):
    return jnp.where(jnp.eye(x.shape[-1], dtype=jnp.bool_), x[..., :, None],
                     jnp.zeros((), dtype=x.dtype))


def conv2d(x, weight, stride=1, padding=0):
    return lax.conv_general_dilated(
        x, weight, pair(stride), tuple((p, p) for p in pair(padding)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'), precision=lax.Precision.HIGHEST)


def conv_transpose2d(x, weight, stride=1, padding=0, output_padding=0):
    pads = tuple((k - 1 - p, k - 1 - p + o)
                 for k, p, o in zip(weight.shape[2:], pair(padding), pair(output_padding)))
    kernel = jnp.flip(jnp.swapaxes(weight, 0, 1), axis=(2, 3))
    return lax.conv_general_dilated(
        x, kernel, (1, 1), pads, lhs_dilation=pair(stride),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'), precision=lax.Precision.HIGHEST)


def unfold(x, kernel_size, padding=0, stride=1):
    patches = lax.conv_general_dilated_patches(
        x, pair(kernel_size), pair(stride), tuple((p, p) for p in pair(padding)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'), precision=lax.Precision.HIGHEST)
    return patches.reshape(x.shape[0], patches.shape[1], -1)


def fold(x, output_size, kernel_size, stride=1, padding=0):
    """Overlap-add the columns in Torch's C, kH, kW ordering."""
    h, w = pair(output_size)
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    channels = x.shape[1] // (kh * kw)
    y = jnp.arange(kh)[:, None, None, None] + sh * jnp.arange(oh)[None, None, :, None] - ph
    z = jnp.arange(kw)[None, :, None, None] + sw * jnp.arange(ow)[None, None, None, :] - pw
    valid = (y >= 0) & (y < h) & (z >= 0) & (z < w)
    indices = jnp.where(valid, y * w + z, h * w).reshape(-1)
    values = x.reshape(x.shape[0], channels, kh * kw * oh * ow)
    result = jnp.zeros((x.shape[0], channels, h * w), dtype=x.dtype)
    return result.at[:, :, indices].add(values, mode='drop').reshape(x.shape[0], channels, h, w)


def scatter(x, dim, index, source):
    dim %= x.ndim
    coords = list(jnp.indices(index.shape, sparse=True))
    coords[dim] = index
    return x.at[tuple(coords)].set(source)


def patches_to_dense(pieces, batch, oc, ox, oy, ic, kx, ky, ix, iy, px, py, sx, sy):
    rows = ox * oy
    x = jnp.arange(ix)[None, :] + px - jnp.arange(ox)[:, None] * sx
    y = jnp.arange(iy)[None, :] + py - jnp.arange(oy)[:, None] * sy
    valid = ((x >= 0) & (x < kx))[:, None, :, None] & ((y >= 0) & (y < ky))[None, :, None, :]
    index = jnp.clip(x, 0, kx - 1)[:, None, :, None] * ky + jnp.clip(y, 0, ky - 1)[None, :, None, :]
    source = jnp.broadcast_to(pieces.reshape(-1, oc, rows, ic, kx * ky), (batch, oc, rows, ic, kx * ky))
    index = jnp.broadcast_to(index.reshape(1, 1, rows, 1, ix * iy), (batch, oc, rows, ic, ix * iy))
    result = jnp.take_along_axis(source, index, axis=-1)
    return jnp.where(valid.reshape(1, 1, rows, 1, ix * iy), result, 0).reshape(batch, oc * rows, ic * ix * iy)


def col2im_columns(out_x, out_y, ker_x, ker_y, padded_rows, padded_cols, stride):
    flat = jnp.arange(out_x * out_y)
    row = flat // out_y * padded_rows * stride
    col = flat % out_y * stride
    return ((row + col).reshape(-1, 1, 1)
            + (jnp.arange(ker_x) * padded_cols).reshape(1, -1, 1)
            + jnp.arange(ker_y).reshape(1, 1, -1)).reshape(out_x * out_y, ker_x * ker_y)

"""Environment toggles for the DeepPoly-efficient optimization passes.

Every flag defaults to off, which reproduces the compiler before these passes.
CF_<NAME>=1 enables one change; CF_DPE_OPTS=1 enables all of them, and
CF_<NAME>=0 then disables an individual one. Flags are read on each call.
"""
import os

FLAGS = (
    'CF_SROA_FUNCTIONAL',       # functional = no view writes after SROA, also for torch
    'CF_BOX_PATCH_REDUCTIONS',  # center/radius fusion for patch sums and affine constants
    'CF_PATCH_FACTOR',          # A @ (diag(s) @ W) -> (A @ diag(s)) @ W
    'CF_CONV_CHUNKS',           # recombine output-channel chunks after patch factoring
    'CF_CONV_SUM',              # C(x,W) + C(y,W) -> C(x+y,W) after patch factoring
    'CF_TRANSPOSE_GRIDS',       # product grids over conv_transpose2d, meta-device shape eval
    'CF_GRID_PACKED_BYTES',     # count packed grid inputs against the grid memory limit
    'CF_LIGHT_TRAVERSAL_CLONE', # clone traversal bodies without use/parent edges
    'CF_VIEW_ALIASES',          # inline storage-sharing views before memory planning
    'CF_EXPRESSION_LIMITS',     # split expressions deeper than 64 into temporaries
    'CF_LAZY_TEMPORARIES',      # profile unnamed temporaries only for memory planning
    'CF_SIGN_REDUCTION_MEMO',   # keep sign_reductions structural keys across statements
)


def enabled(name):
    value = os.environ.get(name)
    if value is not None:
        return value == '1'
    return os.environ.get('CF_DPE_OPTS') == '1'

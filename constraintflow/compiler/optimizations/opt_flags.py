"""Switches for the DeepPoly-efficient optimization passes, all on by default.

`jit` sets them from --<name>/--no-<name> (and --no-dpe-opts for all of them).
"""

FLAGS = (
    'sroa_functional',       # functional = no view writes after SROA, also for torch
    'box_patch_reductions',  # center/radius fusion for patch sums and affine constants
    'patch_factor',          # A @ (diag(s) @ W) -> (A @ diag(s)) @ W
    'conv_chunks',           # recombine output-channel chunks after patch factoring
    'conv_sum',              # C(x,W) + C(y,W) -> C(x+y,W) after patch factoring
    'transpose_grids',       # product grids over conv_transpose2d, meta-device shape eval
    'grid_packed_bytes',     # count packed grid inputs against the grid memory limit
    'view_aliases',          # inline storage-sharing views before memory planning
    'expression_limits',     # split expressions deeper than 64 into temporaries
    'fused_matvec',          # lower pointwise-produced matvecs to row reductions
    'shared_lhs_mm',         # Inductor pass: mm(x, W), mm(x, b) -> one mm(x, [W | b])
)

# Analysis back-edges (def-use chains, parent links) can lead from one statement
# to the rest of the program. Tree walks that skip them visit only the code
# actually reachable through statements, children and operator attributes.
ANALYSIS_EDGES = frozenset(('uses', 'parents', 'defs'))

_enabled = set(FLAGS)


def enabled(name):
    return name in _enabled


def set_enabled(names):
    unknown = set(names) - set(FLAGS)
    if unknown:
        raise ValueError('unknown optimization flags: ' + ', '.join(sorted(unknown)))
    _enabled.clear()
    _enabled.update(names)


def reset():
    set_enabled(FLAGS)

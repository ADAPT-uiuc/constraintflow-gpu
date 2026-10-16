# Bounds-preserving DeepPoly-efficient optimization

This worktree preserves `examples/compiler_examples/deeppoly_efficient.cf`,
including its negative-layer traversal priority and ReLU relaxation rules.
The user explicitly requires the original bounds. A different traversal schedule
was evaluated as a diagnostic and rejected: it changes N34 bounds by about 0.028.
It is not part of this implementation or the results below.

## Flags

Every change below is off by default; with no flags set the compiler emits the
same code as before this work. Toggles live in
`constraintflow/compiler/optimizations/opt_flags.py` and are read at build time.

- `CF_DPE_OPTS=1` enables all of the flags in the table; `CF_<NAME>=0` then
  disables an individual one. `CF_<NAME>=1` enables one on its own.
- The three opt-in passes at the bottom are not part of `CF_DPE_OPTS`; set them
  explicitly, as before.

| Flag | Change | Affects bounds? |
|---|---|---|
| `CF_SROA_FUNCTIONAL` | Purity is decided after SROA (no view writes) for torch too, not only JAX. Enables every functional-only pass on builds that were previously `functional=False` (zono, polyzono, reuse, zid conv nets). | Indirectly, through the passes it enables |
| `CF_BOX_PATCH_REDUCTIONS` | Center/radius fusion for patch sums and affine-constant vectors; reruns box fusion in codegen. | fp32 rounding |
| `CF_PATCH_FACTOR` | `A @ (diag(s) @ W)` becomes `(A @ diag(s)) @ W` via `conv_transpose2d`. | fp32 rounding |
| `CF_CONV_CHUNKS` | Recombines output-channel chunks (only after patch factoring). | fp32 rounding |
| `CF_CONV_SUM` | `C(x,W) + C(y,W)` becomes `C(x+y,W)` (only after patch factoring). | fp32 rounding |
| `CF_TRANSPOSE_GRIDS` | Product grids over `conv_transpose2d`; meta-device operand shape evaluation. | fp32 rounding |
| `CF_GRID_PACKED_BYTES` | Grid memory limit counts packed inputs (fixes the N35/N36 16 GiB OOM). | No |
| `CF_LIGHT_TRAVERSAL_CLONE` | Traversal unrolling skips copying use/parent edges. | No (compile time) |
| `CF_VIEW_ALIASES` | Inlines storage-sharing views before memory planning. | No |
| `CF_EXPRESSION_LIMITS` | Splits expressions deeper than 64 into temporaries. | No |
| `CF_LAZY_TEMPORARIES` | Profiles unnamed temporaries only for memory planning. | No (compile time) |
| `CF_SIGN_REDUCTION_MEMO` | Keeps sign-reduction structural keys across statements. | No (compile time) |
| `CF_TREE_RENAME` | `splice_flow` renames walk the IR tree, not `uses`/`parents`/`defs` edges. | No (compile time) |
| `CF_TREE_PURITY_SCAN` | SROA's purity scan walks the IR tree, not analysis edges. | No (compile time) |
| `CF_SHARED_GATHERS=1` | Shares small image unfolds (about 31 GB peak on N33). | fp32 rounding |
| `CF_FUSED_MATVEC=1` | Lowers pointwise-produced matvecs to row reductions. | fp32 rounding |
| `CF_FAST_BLOCK_CSE=1` | One-walk hash-bucket CSE for functional SSA. | No (compile time) |

All measurements below were taken with every `CF_DPE_OPTS` change enabled.

Compiler changes:

- Classify post-SROA functional tensor code after copy/alias checks.
- Factor row-scaled convolution patches, including aligned input-channel chunks:
  `A @ (diag(s) @ W)` becomes `(A @ diag(s)) @ W` with exact layout/shape guards.
- Combine compatible single-use transpose convolutions, including corresponding
  concatenated channel chunks.
- Fuse complementary sign reductions for patch and affine concretizations.
- Extend product grids to guarded transpose-convolution groups.
- Avoid copying def-use analysis backedges during traversal-body cloning.
- Optional `CF_SHARED_GATHERS=1` shares small image unfolds with explicit region
  boundaries. An opaque-op experiment had no meaningful speed benefit and was
  discarded.

The rewrites preserve real-arithmetic expressions. Floating-point rounding can
change; they do not promise bitwise equality. Numerical tests compare original
and rewritten operations in float64, including non-square kernels, padding,
strides, position-dependent coefficients, piecewise slopes, channel chunks,
and rejected layouts. N33/N34 full-network comparisons use original DPE bounds.

## Measured results

A100 80 GB, FP32, TF32 disabled, batch size 1, ten CIFAR images after five warmups.
Timings exclude compilation. Fresh auto_LiRPA runs use the same models and inputs.

| Configuration | Original DPE | Optimized DPE | Fresh auto_LiRPA | DPE improvement | Max bound change |
|---|---:|---:|---:|---:|---:|
| N33, shared gathers | 18.7843 s | 1.7542 s | 1.5840 s | 10.71x | 0.00000334 |
| N34, shared gathers | 18.8212 s | 1.7684 s | 1.6114 s | 10.64x | 0.00002098 |

All ten verification decisions agree with original DPE in both configurations.
Shared gathers use about 31.16 GB peak GPU allocation. The default configuration
without shared gathers used 7.49 GB and ran N33 in 2.0112 s (bound difference
0.00000191). Shared gathers are therefore optional.

**The requested speedup over auto_LiRPA has not been achieved.** Fresh
`deeppoly_add` runs N33 in 0.5953 s, about 2.66x faster than fresh auto_LiRPA.
Original-schedule optimized DPE still takes about three times that CF runtime.
N35 and N36 initially failed during warmup: a product-grid concretization
attempted to allocate a 16 GiB packed matrix with 76.08 GiB already allocated.
The grid pass checked result size but overlooked the cost of packing its inputs.

The memory fix now counts packed inputs and outputs against the existing
512 MiB grid limit. Oversized candidates keep their separate products, which
also lets early reductions release their inputs earlier. Storage-sharing view
assignments are inlined before planning so a base and its views do not prevent
offloading their shared storage. Finally, the memory planner receives a full
profile of unnamed temporaries before it schedules the code. The original DPE
program, traversal, and relaxation rules remain unchanged.

| Configuration | Fixed DPE median | Peak allocation (GB) | Fresh auto_LiRPA |
|---|---:|---:|---:|
| N35 | 103.2486 s | 30.36 | 15.0973 s |
| N36 | 106.6650 s | 30.36 | 15.2751 s |

These follow-up measurements use three images after one warmup, FP32 with TF32
disabled, batch size 1. N35 completes with finite bounds and upper >= lower;
its three times are 102.4323, 103.2486, and 103.4591 s. The 8–10 s target has
not been achieved. There are no saved original-DPE N35/N36 bounds for a direct
full-network numerical comparison; comparisons to auto_LiRPA or deeppoly_add
are comparisons to different algorithms, not an original-DPE equivalence check.

The view pass removes all 9,820 N35 view aliases, with identical full output
operator-DAG hashes before and after. The oversized-input regression test uses
the N35 failing shapes without allocating real multi-GiB tensors. All 35 targeted
tests pass. A full N33 regression (ten images, five warmups) takes 2.0183 s,
peaks at 7.06 GB, and has the same 1.90735e-6 maximum difference from original DPE
as before; all ten verification decisions agree.

Compilation remains expensive. N35's first warmup, including PyTorch compilation,
takes 4856.4 s (80.9 min), excluded from inference timing. The fixed generated
file is 13,672,664 bytes with 42,828 tensor statements and 683 unique compiled
functions across 2,260 region calls. The saved deeppoly_add N35 kernel is
678,323 bytes with 3,506 statements and 168 unique compiled functions. These
file statistics describe the saved baseline and current worktree, rather than
a controlled comparison of compiler versions.

Follow-up logs and reports are under `optimization-results/N35-memory-fix-pilot/`
and `N36-memory-fix-pilot/`. N33 validation is in `N33-memory-fix-timing/`;
`N35-memory-fix-view-proof.json` records the output-DAG check.
N37's historical auto_LiRPA run exhausted GPU memory, so there is no finite
historical auto_LiRPA speedup target for that configuration.

Raw reports: `optimization-results/N33-shared-gathers-timing/summary.json` and
`optimization-results/N34-shared-gathers-timing/summary.json`. All generated
kernels, input/output bounds, logs, and diagnostic variants are in ignored
`optimization-results/`; none changes the source DPE program.

## Reproduction

Use the existing CF Python environment. Put **this worktree first** in
`PYTHONPATH`, followed by the harness root. The dataset symlink reads CIFAR from
the main checkout. All generated code stays under `optimization-results/`.

```sh
CF_DPE_OPTS=1 PYTHONPATH="$PWD:/home/db50/cf-harness" CF_DISABLE_TF32=1 \
  TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0 \
  /home/db50/cf-harness/constraintflow-gpu/cfg/bin/python3 \
  constraintflow/cli.py jit examples/compiler_examples/deeppoly_efficient.cf \
  --network /home/db50/cf-harness/nets/cifar/ResNetTiny_DiffAI.onnx \
  --dataset cifar --eps 0.001 --batch-size 1 --device gpu --inductor \
  --center-radius --early-reductions --flow-segment-mb 8192 \
  --compact-patches --fuse-sign-convs --in-memory --fuse-affine-subst \
  --output-path optimization-results/N33-optimized
```

Set `CF_SHARED_GATHERS=1` on the build command to evaluate shared gathers.
Measure with the same Python:

```sh
python benchmarks/dpe/measure.py --kernel optimization-results/N33-optimized \
  --config N33 --images 10 --warmup 5 --tag N33-optimized-timing
```

The runner records commands, kernel hash, raw bounds, timings, memory, and saved
and fresh reference comparisons. Historical `deeppoly_add` CF targets are N33
0.59305 s, N34 0.59776 s, N35 8.34962 s, and N36 8.67903 s per image.

Run targeted regression tests using the CF Python:

```sh
PYTHONPATH="$PWD/tests:$PWD:/home/db50/cf-harness" python -m unittest \
  test_patch_factor test_conv_sum test_sroa_purity test_transpose_grids \
  test_box_patch_reductions test_shared_gathers test_traversal_clone \
  test_ssa_cse test_conv_chunks test_sign_reduction_cache \
  test_expression_limits test_flow_shape_modes test_view_aliases \
  test_fused_matvec test_opt_flags
```

Tests pin the flag they exercise, so the suite passes with flags on or off.

## N35 inference follow-up

With `CF_FUSED_MATVEC=1`, large pointwise-produced matrix-vector products
are lowered to fusible row reductions. N35 takes 90.6453 s median over three
images after one warmup, down from 103.2486 s, with 25.73 GB peak allocation.
The maximum bound change from the memory-fixed DPE reference is 4.77e-7; all
three verification decisions agree. Traversal and FP32/TF32 settings are unchanged.

A subsequent late gather-sharing experiment measured 89.8865 s and 28.22 GB.
Its 0.8% timing difference does not justify retaining the new pass; the late
gather integration was removed and its N36 follow-up canceled. The earlier
optional shared-gather pass used in the N33/N34 table is separate and unchanged.
N37 is being run with fused matvecs and without gather sharing.

# Bounds-preserving DeepPoly-efficient optimization

This worktree preserves `examples/compiler_examples/deeppoly_efficient.cf`,
including its negative-layer traversal priority and ReLU relaxation rules.
The user explicitly requires the original bounds. A different traversal schedule
was evaluated as a diagnostic and rejected: it changes N34 bounds by about 0.028.
It is not part of this implementation or the results below.

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
and rejected layouts. Full-network comparisons use original DPE bounds.

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
The N35 original-schedule build was stopped before completion when the user
accepted the existing gains. Channel-chunk rewrites have numerical tests and
static wide-network coverage, but no completed N35/N36 performance result.
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
PYTHONPATH="$PWD:/home/db50/cf-harness" CF_DISABLE_TF32=1 \
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
  test_box_patch_reductions test_shared_gathers test_traversal_clone
```

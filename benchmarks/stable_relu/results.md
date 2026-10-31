# Stable-ReLU results

Measured on NVIDIA A100 80GB PCIe, PyTorch 2.10.0+cu128, FP32, TF32 disabled,
batch size 1. Image 10 is the warmup; medians below cover images 0–9.
Baselines were rerun with auto_LiRPA `--sparse` and GPUPoly's default skipping.
Runs were sequential, without GPU contention.

| Configuration | cf-gpu with skipping, including setup | Sparse auto_LiRPA | Speedup | cf-gpu bounding only | GPUPoly bounding only | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| N42, TinyImageNet ResNet medium, eps 1/255 | 0.0930 s | 0.9489 s | 10.20× | 0.0694 s | 0.3037 s | 4.38× |
| N36, ResNet18 PGD, eps 0.00004 | 6.0577 s | 9.5588 s | 1.58× | 5.9702 s | 31.1332 s | 5.21× |

cf-gpu's setup-inclusive time includes input transfer and ONNX/network loading.
GPUPoly reports bounding separately from loading, so its speedup uses `bound_s`.
Compilation is excluded from these steady-state measurements. Cold warmups took
84.77 seconds for N42 and 369.99 seconds for N36; this is not a cold-start speedup.

The saved dense cf-gpu medians were 2.0006 s (N42) and 8.3691 s (N36), giving
21.52× and 1.38× improvements respectively. These dense timings are historical,
from the report's existing runs, rather than fresh timing measurements. Dense
correctness comparisons were performed afresh on every tested image.

## Correctness

All 20 timed images and both warmup images passed a comparison against dense
cf-gpu using `torch.testing.assert_close(atol=0.001, rtol=0.001)`, with identical
verification decisions. Maximum absolute final-bound changes among timed images
were 0.0017843 on N42 and 0.000002862 on N36. The former passes because the check
includes relative tolerance; this is not a claim of absolute error below 0.001.
Stable intermediate bounds are intentionally retained as intervals, and different
convolution batch sizes can change floating-point rounding.

| Configuration | Selected cf-gpu verified | Dense cf-gpu verified | auto_LiRPA verified | GPUPoly verified |
|---|---:|---:|---:|---:|
| N42 | 2/10 | 2/10 | 2/10 | 2/10 |
| N36 | 9/10 | 9/10 | 9/10 | 7/10 |

GPUPoly's N36 verification count differs from cf-gpu and auto_LiRPA; these timings
compare their respective published bounding procedures, not numerically identical
bounds across tools. The optimization preserves cf-gpu's tested decisions.

Selected-path peak CUDA allocation was approximately 0.86 GiB on N42 and 6.30 GiB
on N36. Peak allocation excludes the subsequent dense-reference check.

Seven new regression tests cover changing masks, all-stable/all-unstable inputs,
zero boundaries, invalid intervals, final-chunk padding, spatial patch gathering,
reused compiler regions, fresh JIT generation, and fixed-shape fullgraph reuse.
Four adjacent compiler tests also passed.

## Artifacts and scope

Raw per-image times and bounds are in `results/N42*.jsonl` and `results/N36*.jsonl`.
Metadata records input/model/source hashes. Baseline logs and reproduction scripts
are alongside them. See [README.md](README.md) for invocation commands.

This option supports batch one and the standard DeepPoly-add specifications.
The coordinator is eager; numerical kernels are compiled with fullgraph=True.
Whole-flow CUDA graph capture and training are not supported. Other configurations
in the original impact report have not been benchmarked with this change.

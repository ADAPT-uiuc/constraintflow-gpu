# Stable-ReLU results

Measured on NVIDIA A100 80GB PCIe, PyTorch 2.10.0+cu128, FP32, TF32 disabled,
batch size 1. Image 10 is the warmup; medians below cover images 0–9.
Baselines were rerun with auto_LiRPA `--sparse` and GPUPoly's default skipping.
Runs were sequential, without GPU contention.

The table below records the original measurements; the N42 warmup defect and
its corrected cold-cache validation are documented below.

| Configuration | cf-gpu with skipping, including setup | Sparse auto_LiRPA | Speedup | cf-gpu bounding only | GPUPoly bounding only | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| N42, TinyImageNet ResNet medium, eps 1/255 | 0.0930 s | 0.9489 s | 10.20× | 0.0694 s | 0.3037 s | 4.38× |
| N36, ResNet18 PGD, eps 0.00004 | 6.0577 s | 9.5588 s | 1.58× | 5.9702 s | 31.1332 s | 5.21× |

cf-gpu's setup-inclusive time includes input transfer and ONNX/network loading.
GPUPoly reports bounding separately from loading, so its speedup uses `bound_s`.
**Warmup correction:** the original N42 run below did not fully warm all kernels.
Image 2 took 11.8894 seconds because its row kernel first ran on that image. Its
mean was 1.2730 seconds; the 0.0930-second median hid this event. The original raw
records are retained. These original N42 figures must not be read as evidence
that compilation was excluded from every timed image. See the cold-cache
warmup-fix validation below. Cold warmups took
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

These original measurements used batch one and the standard DeepPoly-add specifications.
Batched support and its validation are documented below.
The coordinator is eager; numerical kernels are compiled with fullgraph=True.
Whole-flow CUDA graph capture and training are not supported. Other configurations
in the original impact report have not been benchmarked with this change.

## Cold-cache validation of the warmup fix

The original `refine_stable` skipped calling the compiled row kernel when its
selection was empty. Fixed shapes prevented shape recompilation but did not
prevent a previously unused function's first compilation. N42 warmup image 10
left several kernels untouched; image 2 first needed one of them.

`warmup_stable_relu` now executes a discarded fixed-size dummy chunk for empty
selections. It uses valid indices, the normal kernel arguments and device, and
does not replace any bounds. Both CLI warmup protocols and the benchmark use it.
Normal timed calls continue to skip all-stable layers.

A new N42 run used fresh empty Inductor and Triton cache directories, one warmup
on image 10, and images 0–9 for timing. Warmup compiled 20 graphs (ten pre-regions,
nine row kernels, and the final region), including row kernels whose selections
remain empty on all tested images. No timed image compiled a new Dynamo graph.

| Metric | Original N42 | With complete warmup |
|---|---:|---:|
| Image 2, including setup | 11.8894 s | 0.0925 s |
| Mean, including setup | 1.2730 s | 0.0928 s |
| Median, including setup | 0.0930 s | 0.0924 s |
| Maximum timed image | 11.8894 s | 0.1026 s |

The complete cold warmup took 152.06 seconds. This moves compilation into warmup;
it does not eliminate compilation cost. All ten timed images plus the warmup
passed the same dense-bound checks and preserved verification decisions. Maximum
absolute timed-image bound difference was 0.0017910 (relative tolerance applies).
Raw results and fresh-cache paths are recorded in
`results/N42_warmup_fixed.jsonl` and its `.metadata.json` companion.

Nine regression tests pass, including both CLI warmup protocols, unchanged bounds
from discarded dummy chunks, cleanup after warmup exceptions, and a compiled
kernel warmed on an all-stable input then reused across changing masks without
additional graph compilation. Custom API callers must use the explicit warmup
helper; the CLI requires `--warmup` of at least one for this preparation.

## VGG16, VNN-COMP spec0 (Kerry blue terrier)

Single timed run per tool on the same A100, with FP32 and TF32 disabled.
Both environments used PyTorch 2.10.0+cu128. The parsed input bounds and output
specification arrays match exactly. Both tools expand MaxPool with the shared
parser and compute the same lower-only DeepPoly/CROWN margin.

| Tool | Bounding | Loading | Total | Additional peak allocation | Margin lower bound |
|---|---:|---:|---:|---:|---:|
| cf-gpu, stable ReLU, chunk 128 | 76.13 s | 4.39 s | 80.52 s | 8.16 GiB | 7.179458 |
| auto_LiRPA, sparse, CROWN batch 512 | 110.03 s | 17.82 s | 127.84 s | 49.36 GiB | 7.179489 |

cf-gpu is **1.45× faster for bounding** and
**1.59× faster including loading** in this run. Both verify.
The absolute margin difference is 0.00003099,
within the harness tolerance (atol=rtol=0.001). The cf-gpu result also agrees with
the historical dense cf-gpu margin, 7.17949343; dense VGG16 was
not rerun. The timed cf-gpu pass compiled **zero new Dynamo graphs**.

Peak allocation follows the harness convention: peak allocated bytes minus
allocation resident at call entry. cf-gpu's absolute timed peak was
8.82 GiB. This does not measure CUDA reserved memory.

### Preparation cost and fixes

The first attempt ran out of memory. VGG's generated convolution weights use
`expand` rather than `repeat`; the pass now gathers their channels before spatial
expansion. The eager coordinator also releases tuple-unpacked outputs after their
last use instead of retaining all layer temporaries. Eleven regression tests pass,
including exact expanded/repeated-weight gathering and tensor lifetime checks.

Source transformation took 43.35 seconds. The first corrected warmup was
interrupted after roughly 18 minutes to remove its alarm at the user's request.
The restarted process reused the partially populated Inductor cache and completed
warmup in 196.78 seconds, registering 50 Dynamo graphs. **This is not
a cold compilation measurement**; preparation also includes the interrupted
attempt. Auto_LiRPA's warmup took 108.56 seconds. Warmup now has no timeout.
None of these preparation costs is included in the timed table.

The VGG runner uses cached lower-only generated source through `SpecBounds` for
verification. This does not extend the optimization to gradient-based training
or arbitrary certifiers. Raw results, metadata, comparison checks, initial OOM,
interrupted warmup, and successful run logs are saved as `results/VGG16_*`.
See [README.md](README.md#vgg16-spec0) for reproduction commands.

## Batched support: N32 batch ten and N42 batch one

The pass now accepts fixed positive batch sizes. It selects the union of rows
unstable in any example, evaluates fixed padded chunks across the batch, and
preserves each example's own stable interval bounds. Convolution reshapes handle
both `[batch, rows, ...]` and folded `[batch * rows, ...]` layouts. The final
specification remains per-example. CLI setup uses the requested batch size.
The numerical kernels remain fullgraph compiled; the coordinator remains eager.

### N32 vs sparse auto-LiRPA

N32 is CIFAR `convBigRELU__DiffAI`, epsilon 0.5/255, batch size ten.
Images 30–39 warm up both tools; images 0–29 form three timed batches.
Runs were sequential on the A100, FP32, TF32 off. Dataset files, normalized boxes,
and labels were checked for exact equality. Times include model loading, input
transfer and bounding. cf-gpu consumes precomputed normalized boxes; auto-LiRPA
also builds those boxes inside its timer, as in the existing harness protocol.

| Tool | Median per batch of ten | Median amortized per image | Verified |
|---|---:|---:|---:|
| cf-gpu stable ReLU | 0.058253 s | 0.005825 s | 16/30 |
| auto-LiRPA sparse | 0.374239 s | 0.037424 s | 16/30 |

The setup-inclusive speedup is **6.42×**. Individual cf-gpu batches
were 0.058890, 0.058253, 0.041034 seconds; auto-LiRPA batches were
0.402523, 0.374239, 0.357442 seconds. This is a three-batch datapoint,
not a broad batch-size scaling study.

All lower and upper output bounds across 30 images agree between tools within
atol=rtol=0.001; the maximum absolute difference was **0.00000286**.
All four cf-gpu batches, including warmup, also passed dense cf-gpu comparisons
and preserved per-image decisions; maximum absolute dense difference was
0.00059271. Every timed batch compiled zero new
Dynamo graphs. N32 warmup registered 12 graphs and took
37.55 seconds, with caches partially populated by development
attempts; it is not a cold compilation measurement. Warmup has no timeout.

### N42 batch-one regression

N42 retains the same ten images and image-10 warmup. This run used fresh Inductor
and Triton cache directories. Warmup took 153.82 seconds and
registered 20 graphs, including kernels skipped by normal
all-stable calls. All ten timed images compiled zero new graphs.

| Metric | Previous complete-warmup run | After batch support |
|---|---:|---:|
| Median including loading | 0.092444 s | 0.093951 s |
| Verified | 2/10 | 2/10 |

The median changed by 1.63% in this run. Image 2 took
0.092606 seconds with no late compilation. All ten timed images plus
warmup passed dense-bound checks and retained verification decisions. Maximum
absolute difference from the previous optimized run was 0.00001526; maximum
absolute difference from freshly evaluated dense bounds was
0.00179577 (the relative tolerance applies).

Regression coverage includes batch-one and batched CLI generation, batched
convolution geometry, dense equivalence with all rows selected, per-example
invariance when other batch members change, union selection, retained stability
proofs, and compiled graph reuse across changing batch masks. An interval proof
can be tighter than a dense DeepPoly bound for a stable neuron, so universal
bitwise equivalence with dense intermediate bounds is neither required nor
claimed. Raw results, metadata, logs and summary are in `results/batched_*`.

Final validation passed all 13 stable-ReLU tests and all three adjacent compiler-option tests.

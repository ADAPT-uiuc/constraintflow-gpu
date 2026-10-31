# Stable-ReLU refinement

`jit --stable-relu` computes inexpensive interval bounds from previously refined
preactivations, then back-substitutes only neurons whose intervals cross zero.
Stable neurons retain their interval bounds: their ReLU relaxation is already
exact. The final specification bounds are always computed in full.

The pass supports the standard `deeppoly_priority_add.cf` and
`deeppoly_priority_add_lower.cf` programs, fixed positive batch sizes, and affine/ReLU/residual
networks. It is opt-in. Unsupported layouts fail during transformation, before
replacing the generated source. `transformers_dense.py` retains the original.
Training, JAX, and whole-flow CUDA graph capture are not supported by this option.

## Batched inputs

Use `--batch-size B` when compiling and running. Each layer selects the union of
neurons unstable in any example, then computes those rows for all examples in
fixed-size chunks. Each example retains its own stable interval bounds even when
another example needs the same row refined. Final output specifications remain
per-example, including batches with different class labels.

Batch size is fixed for a generated program. Changing masks between batches does
not change compiled row-kernel shapes. Larger batches can have a larger union of
unstable neurons, so skipping can save less work than at batch size one.

## Compilation

The coordinator uses an eager `nonzero` to select uncertain rows (one device
synchronization per refined layer). Numerical regions use
`torch.compile(fullgraph=True, dynamic=False)`. The last row chunk is padded,
so changing stability masks does not create new kernel shapes. This does not
claim that the complete coordinator can be captured as one fullgraph.

Both CLI warmup protocols (`run --warmup`, with either `--repeat` or
`--num-images`) now exercise every row kernel. If a layer has no unstable neurons,
warmup calls its kernel once with valid dummy indices and discards the results.
Ordinary calls still skip all-stable layers. At least one warmup is required to
move this first-use compilation out of timed runs; `--warmup 0` remains a cold run.

Custom callers can use the same explicit warmup API, with the same grad mode,
dtype and device as their subsequent calls:

```python
from constraintflow.compiler.optimizations.stable_relu import warmup_stable_relu

with torch.no_grad():
    warmup_bounds = warmup_stable_relu(flow, *inputs)
    torch.cuda.synchronize()
    bounds = flow(*next_inputs)
```

The benchmark script uses this warmup and records Dynamo's `new_graphs` count.
It fails if any timed image compiles a new graph, even if the median looks normal.

`--stable-relu-chunk 128` controls the fixed chunk size. Selection and patch-to-dense
conversion retain the original channel/spatial coordinates, including padding,
rectangular geometry, residual branches, and reused compiler regions. Masks are
recomputed on every call; no stability decision is baked into generated code.

From the worktree, using an interpreter with the project's dependencies:

```bash
python -m constraintflow.cli jit examples/compiler_examples/deeppoly_priority_add.cf \
  --network /path/to/model.onnx --dataset cifar --eps 0.00004 \
  --device gpu --batch-size 1 --inductor --fuse-affine-subst \
  --stable-relu --output-path /tmp/stable-program
```

## Reproduce measurements

Set `CF_HARNESS` to the surrounding benchmark repository and use its existing
Python environment. No models or datasets are downloaded by these scripts.

```bash
export PYTHONPATH="$PWD"
PY="$CF_HARNESS/constraintflow-gpu/cfg/bin/python"
"$PY" benchmarks/stable_relu/benchmark.py \
  --kernel "$CF_HARNESS/constraintflow-gpu/local_cache/N42_deeppoly_priority_add_sign" \
  --network "$CF_HARNESS/nets/TinyImageNet_resnet_medium.onnx" \
  --boxes "$CF_HARNESS/experiments/results/sparse_ab_n42/boxes.npz" \
  --dataset tinyimagenet --source-output /tmp/N42-selected.py \
  --output /tmp/N42-selected.jsonl --compare

"$PY" benchmarks/stable_relu/prepare_cifar_boxes.py \
  --data-root "$CF_HARNESS/constraintflow-gpu" --output /tmp/N36-boxes.npz
"$PY" benchmarks/stable_relu/benchmark.py \
  --kernel "$CF_HARNESS/constraintflow-gpu/local_cache/N36_deeppoly_priority_add_sign" \
  --network "$CF_HARNESS/nets/cifar/ResNet18_PGD.onnx" --boxes /tmp/N36-boxes.npz \
  --dataset cifar --source-output /tmp/N36-selected.py \
  --output /tmp/N36-selected.jsonl --compare
```

Each run uses image 10 for warmup and images 0–9 for timing, FP32, TF32 disabled,
with synchronization around timing. `bound_s` measures bounding alone; `time_s`
also includes input transfer and ONNX/network loading. Cold compilation is in
the warmup record. `--compare` checks every image against dense cf-gpu using
`atol=rtol=0.001` and requires identical verification decisions. Floating-point
rounding may differ; intermediate stable-neuron intervals are intentionally
less tight than fully refined intervals.

Fresh baseline measurements (run after cf-gpu finishes to avoid contention):

```bash
CF_N36_BOXES=/tmp/N36-boxes.npz bash benchmarks/stable_relu/run_baselines.sh
```

Tests:

```bash
python -m unittest discover -s tests -p test_stable_relu.py -v
```

See [results.md](results.md) for measurements and limitations.

## VGG16 spec0

`vnncomp_benchmark.py` applies the pass to the harness's cached, unchunked
lower-bound DeepPoly program. It uses the original VNN-COMP box and output margin,
with MaxPool already expanded by the shared parser. This is verification through
`SpecBounds`; it does not enable gradient-based training.

From the harness root:

```bash
export PYTHONPATH="$PWD/wt/stable-relu"
export TORCHINDUCTOR_COMPILE_THREADS=1 CF_DISABLE_TF32=1 NVIDIA_TF32_OVERRIDE=0
constraintflow-gpu/cfg/bin/python -u wt/stable-relu/benchmarks/stable_relu/vnncomp_benchmark.py \
  --harness "$PWD" \
  --kernel experiments/results/vnncomp/agent_full/vggnet16_2026/compile/cf_d151d6abb8c2286e \
  --network nets/vggnet16_2026/onnx/vgg16-7.onnx \
  --spec experiments/results/vnncomp/agent_full/vggnet16_2026/specs/spec0_Kerry_blue_terrier.npz \
  --output "$PWD/experiments/results/vnncomp/stable_relu_spec0/cf"

constraintflow-gpu/cfg/bin/python experiments/compare_vnncomp.py \
  --prefix stable_relu_spec0_lirpa --tools lirpa --specs spec0 \
  --lirpa-sparse --lirpa-crown-batch 512
```

Run the tools sequentially. `--prepare-only` emits transformed source without
running bounds; `--reuse-source` runs that source without repeating transformation.
The runner records warmup separately, asserts zero new Dynamo graphs during the
timed call, and charges model loading plus verification to the reported total.
Peak memory uses the harness's additional-allocation convention; absolute peak
allocation is also recorded. Warmup has no timeout. Timed verification has a 1200-second limit,
subtracting model loading from that allowance.

## Batched N32 comparison and batch-one regression

From the harness root, using the worktree on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/wt/stable-relu"
export CF_DISABLE_TF32=1 NVIDIA_TF32_OVERRIDE=0 TORCHINDUCTOR_COMPILE_THREADS=1
constraintflow-gpu/cfg/bin/python wt/stable-relu/benchmarks/stable_relu/prepare_cifar_boxes.py \
  --data-root constraintflow-gpu --eps 0.00196078431372549 --images 40 \
  --output experiments/results/stable_relu_batched/N32_boxes.npz
```

Generate the dense batch-ten N32 source with `constraintflow.cli jit`,
`--batch-size 10 --dataset cifar --fuse-affine-subst`, running from
`constraintflow-gpu` so it finds the existing CIFAR dataset. Then:

```bash
constraintflow-gpu/cfg/bin/python wt/stable-relu/benchmarks/stable_relu/benchmark.py \
  --kernel experiments/results/stable_relu_batched/N32_dense \
  --network nets/cifar/convBigRELU__DiffAI.onnx \
  --boxes experiments/results/stable_relu_batched/N32_boxes.npz --dataset cifar \
  --batch-size 10 --num-images 30 --compare \
  --source-output /tmp/N32_selected.py --output /tmp/N32_cf.jsonl
```

Run auto-LiRPA sequentially from `auto_LiRPA/examples/vision`, using its
`lrpa311/bin/python3 simple_verification.py`, with
`--network convBigRELU__DiffAI --network_format onnx --dataset cifar
--eps 0.00196078431372549 --batch_size 10 --num_images 30 --warmup 1
--program deeppoly --sparse --single_experiment --logs stdout --device gpu`,
plus absolute `--nets_dir` and `--results_file` paths.
Both tools warm on images 30–39 and time images 0–29 in three batches of ten.
The cf-gpu benchmark checks all four batches against dense bounds and records
per-image verification decisions. Warmup has no timeout.

The N42 regression uses the original reproduction command above with
`--batch-size 1`, ten images, and fresh `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` directories. Changing the batch size requires generating a
program for that batch size; `--batch-size` does not retarget an existing program.

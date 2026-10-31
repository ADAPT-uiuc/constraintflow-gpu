# Stable-ReLU refinement

`jit --stable-relu` computes inexpensive interval bounds from previously refined
preactivations, then back-substitutes only neurons whose intervals cross zero.
Stable neurons retain their interval bounds: their ReLU relaxation is already
exact. The final specification bounds are always computed in full.

The pass supports the standard `deeppoly_priority_add.cf` and
`deeppoly_priority_add_lower.cf` programs, batch size one, and affine/ReLU/residual
networks. It is opt-in. Unsupported layouts fail during transformation, before
replacing the generated source. `transformers_dense.py` retains the original.
Training, JAX, and whole-flow CUDA graph capture are not supported by this option.

## Compilation

The coordinator uses an eager `nonzero` to select uncertain rows (one device
synchronization per refined layer). Numerical regions use
`torch.compile(fullgraph=True, dynamic=False)`. The last row chunk is padded,
so changing stability masks does not create new kernel shapes. This does not
claim that the complete coordinator can be captured as one fullgraph.

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

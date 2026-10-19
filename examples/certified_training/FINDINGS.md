# Certified training: cf-gpu vs auto_LiRPA — findings

October 2026. Setup: `compare_driver.py` runs auto_LiRPA's own `examples/vision/simple_training.py`
(unmodified) and cf-gpu's copy of it (`simple_training.py`, which swaps `BoundedModule` for
`TrainingBounds`), with the same converted ONNX model, data, seed and ε. TF32 off in both.
Timed: setup and training epochs. Not timed (reported separately): cf-gpu code generation,
warmup, per-epoch test evaluation.

## 1. Correctness: the tools compute the same bounds

On identical weights, IBP bounds are **bit-identical**, and CROWN-IBP / CROWN (DeepPoly) bounds
agree to ≤1e-6 relative, on all six networks below and on the trained weights of both tools.
Gradients of the bounds agree to ~1e-7–1e-5; a few single parameters differ by ~1e-3 on
pretrained weights, consistent with ReLU ties (the gap disappears on perturbed weights).

Two setup fixes were needed for this:
- The ONNX loader ends in a no-op `Flatten` after the last Linear, which stopped auto_LiRPA from
  folding the margin spec into that layer (its IBP was ~30% looser). The driver replaces it with
  `Identity`; both tools now fold.
- The ERAN MNIST nets expect normalized input, which the loader strips; `--normalize` applies the
  verification sweeps' normalization in both tools.

Precision (`cli.get_precision`: all margin lower bounds ≥ 0) reproduces the verification sweeps'
verified counts exactly at the sweeps' ε (e.g. N14: CROWN-IBP 921, CROWN 923, zid 918 of 1000).

## 2. IBP training grid (100 epochs, ε = 0.1 MNIST / 2/255 CIFAR, full 10k test set)

`run_ibp_grid.sh`, results in `experiments/results/ibp-training-grid/report.txt`.
Schedule: auto_LiRPA's default (`SmoothedScheduler start=3,length=60`), batch 256, seed 100.

| Net | IBP precision before → after (auto_LiRPA / cf-gpu) | Setup + training speedup |
|---|---|---|
| N10 ffnnRELU__PGDK_w_0.3_6_500 | 0 → 0.922 / 0.927 | 2.36× |
| N11 mnist_relu_4_1024 | 0 → 0.877 / 0.878 | 1.52× |
| N14 convSmallRELU__DiffAI | 0.868 → 0.940 / 0.940 | 1.69× |
| N22 ffnnRELU__Point_6_500 (CIFAR) | 0 → 0.289 / 0.282 | 1.08× |
| N30 convMedGRELU__PGDK_w_0.0078 (CIFAR) | 0 → 0.406 / 0.404 | 1.03× |
| N32 convBigRELU__DiffAI (CIFAR) | 0.014 → 0.382 / 0.378 | 0.99× |

Cross-evaluation (`cross_eval_ibp_grid.sh`): each tool's trained weights get **identical**
precision from both tools. The small after-training gaps are fp32 drift over ~23k optimizer
steps (gradients differ by ~1e-7), not a bounding difference.

## 3. Why training speedups are far smaller than verification speedups

Verification sweeps show cf-gpu 15–23× faster for IBP / CROWN-IBP. Training shows ~1–2.4×.

1. **Shared costs (Amdahl).** IBP bounding is cheap (0.6–3 s per epoch); CIFAR epochs are
   dominated by the CPU `DataLoader` with augmentation, paid equally by both tools. MNIST uses
   GPU data (`--gpu_data`), hence its larger speedups.
2. **The backward pass** — the key finding, below.

## 4. Key finding: cf-gpu's bounds are 2.2× faster, its backward is 2× slower

One CROWN-IBP training step on N32, batch 256, averaged over 20 steps after 5 warmup steps,
CUDA-synchronized per phase (`profile_step.py`; GPU shared with another job, equally for both):

| Phase (ms / step) | auto_LiRPA | cf-gpu | cf-gpu vs auto_LiRPA |
|---|---|---|---|
| Network forward (logits) | 4.0 | 4.1 | same |
| **Bounds forward** (IBP + CROWN pass) | 115.8 | **51.6** | **2.2× faster** |
| **Backward** (bounds + network) | **79.3** | 156.6 | **2.0× slower** |
| Optimizer step | 2.5 | 2.7 | same |
| Total | 201.7 | 215.0 | 1.07× slower |

A full N32 CROWN-IBP epoch agrees: auto_LiRPA 19.5 s, cf-gpu 25.9 s
(`experiments/results/n32-one-epoch/`).

## Other notes

- `--fuse-affine-subst` is now passed for DeepPoly (`CROWN`) training programs, as in the
  verification harness; bounds/gradients unchanged vs auto_LiRPA (N10 grad 2.1e-5 → 1.5e-6).
- CROWN training is impractical on the largest nets in both tools (auto_LiRPA N32: ~41 GB at
  batch 16, ~1.6 h of bounding per epoch). Lowering the batch size saves memory, not time: per-epoch
  work is the same.
- zid (cf-gpu only) on N32, batch 64: 277 s per epoch (on a shared GPU).

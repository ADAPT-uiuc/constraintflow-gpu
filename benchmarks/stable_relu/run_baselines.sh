#!/usr/bin/env bash
set -euo pipefail
: "${CF_HARNESS:?Set CF_HARNESS to the benchmark repository}"
SR_RESULTS="$(cd "$(dirname "$0")" && pwd)/results"
SR_N36_BOXES="${CF_N36_BOXES:-/tmp/N36_boxes.npz}"
export CF_DISABLE_TF32=1 NVIDIA_TF32_OVERRIDE=0
export GPUPOLY_LIB="$CF_HARNESS/ELINA/gpupoly/build_sm80/libgpupoly.so.0.13"
mkdir -p "$SR_RESULTS"
for SR_ID in N42 N36; do
  if [[ "$SR_ID" == N42 ]]; then
    SR_NAME=TinyImageNet_resnet_medium
    SR_DATASET=tinyimagenet
    SR_EPS=0.00392156862745098
    SR_NETWORK="$CF_HARNESS/nets/TinyImageNet_resnet_medium.onnx"
    SR_BOXES="$CF_HARNESS/experiments/results/sparse_ab_n42/boxes.npz"
  else
    SR_NAME=ResNet18_PGD
    SR_DATASET=cifar
    SR_EPS=0.00004
    SR_NETWORK="$CF_HARNESS/nets/cifar/ResNet18_PGD.onnx"
    SR_BOXES="$SR_N36_BOXES"
  fi
  (
    cd "$CF_HARNESS/auto_LiRPA/examples/vision"
    "$CF_HARNESS/auto_LiRPA/lrpa311/bin/python3" simple_verification.py \
      --network "$SR_NAME" --network_format onnx --dataset "$SR_DATASET" \
      --eps "$SR_EPS" --batch_size 1 --warmup 1 --num_images 10 \
      --results_file "$SR_RESULTS/${SR_ID}_lirpa.jsonl" --program deeppoly \
      --single_experiment --logs stdout --nets_dir "$CF_HARNESS/nets" --device gpu --sparse \
      > "$SR_RESULTS/${SR_ID}_lirpa.log" 2>&1
  )
  "$CF_HARNESS/ELINA/gpupoly/env/bin/python" -I "$CF_HARNESS/experiments/gpupoly/run_gpupoly.py" \
    --network "$SR_NETWORK" --boxes "$SR_BOXES" --batch-size 1 --num-images 10 --warmup 1 \
    --results-file "$SR_RESULTS/${SR_ID}_gpupoly.jsonl" \
    > "$SR_RESULTS/${SR_ID}_gpupoly.log" 2>&1
  echo "$SR_ID baselines finished"
done

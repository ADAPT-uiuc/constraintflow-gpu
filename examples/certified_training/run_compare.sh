#!/usr/bin/env bash
# auto_LiRPA vs cf-gpu certified training (simple_training.py) on the small MNIST FC nets.
#   [NETS="net ..."] [DRIVER_ARGS=--gpu_data] run_compare.sh OUT_DIR [simple_training.py args...]
# For each net: train with each tool, then compute both tools' bounds on each tool's final checkpoint.
set -u
HARNESS=/home/db50/cf-harness
WT=$(cd "$(dirname "$0")/../.." && pwd)
DRIVER=$WT/examples/certified_training/compare_driver.py
LIRPA_PY=$HARNESS/auto_LiRPA/lrpa311/bin/python
CF_PY=$HARNESS/constraintflow-gpu/cfg/bin/python3
OUT=$1; shift
ARGS=("$@")
NETS=(${NETS:-mnist_relu_3_50 mnist_relu_3_100 mnist_relu_5_100 mnist_relu_6_100 mnist_relu_9_100})

mkdir -p "$OUT/data" && ln -sfn "$HARNESS/constraintflow-gpu/MNIST" "$OUT/data/MNIST"
cd "$OUT"
run_tool() {  # tool net extra-driver-args...
    local tool=$1 net=$2; shift 2
    if [ "$tool" = lirpa ]; then
        "$LIRPA_PY" "$DRIVER" --tool lirpa --onnx "$HARNESS/nets/mnist/$net.onnx" ${DRIVER_ARGS:-} "$@" -- "${ARGS[@]}"
    else
        PYTHONPATH=$WT "$CF_PY" "$DRIVER" --tool cf --onnx "$HARNESS/nets/mnist/$net.onnx" ${DRIVER_ARGS:-} "$@" -- "${ARGS[@]}"
    fi
}
for net in "${NETS[@]}"; do
    for tool in lirpa cf; do
        echo "$(date +%T) train $net $tool"
        run_tool $tool $net --out "$net/$tool" > "$net.$tool.train.log" 2>&1 || echo "FAILED train $net $tool"
    done
    for ckpt_tool in lirpa cf; do
        for tool in lirpa cf; do
            echo "$(date +%T) bounds $net ckpt=$ckpt_tool tool=$tool"
            run_tool $tool $net --out "$net/bounds_$ckpt_tool" --bounds_ckpt "$net/$ckpt_tool/ckpt" \
                > "$net.bounds.$ckpt_tool.$tool.log" 2>&1 || echo "FAILED bounds $net $ckpt_tool $tool"
        done
    done
done
PYTHONPATH=$WT "$CF_PY" "$WT/examples/certified_training/compare_report.py" "$OUT" "${NETS[@]}" | tee summary.md
echo "$(date +%T) done"

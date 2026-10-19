#!/bin/bash
# IBP certified training, auto_LiRPA vs cf-gpu, on six ONNX nets at the literature eps
# (MNIST 0.1, CIFAR 2/255; trained and evaluated at the same eps).
# Per net and eps: IBP precision before, bounds+gradients compared on the ONNX weights, training in
# both tools, IBP precision after, and bounds+gradients on each tool's trained weights in both tools.
#
#   bash run_ibp_grid.sh OUT_DIR          (summarize with: python report_ibp_grid.py OUT_DIR)
#   EPOCHS=, SCHEDULER=, SCHEDULER_OPTS=, BATCH=, SEED= override the settings; ONLY=<net name> runs one net.
set -u
HARNESS=/home/db50/cf-harness
HERE=$(cd "$(dirname "$0")" && pwd)
DRIVER=$HERE/compare_driver.py
LIRPA_PY=$HARNESS/auto_LiRPA/lrpa311/bin/python
CF_PY=$HARNESS/constraintflow-gpu/cfg/bin/python3

EPOCHS=${EPOCHS:-100}
SCHEDULER=${SCHEDULER:-SmoothedScheduler}
SCHEDULER_OPTS=${SCHEDULER_OPTS:-start=3,length=60}  # auto_LiRPA simple_training.py default: eps ramps over epochs 3-62
BATCH=${BATCH:-256}
SEED=${SEED:-100}
PRECISION="--num_images 10000 --eval_batch_size 100"  # the full test set
# net, dataset, eps values (one run per value)
NETS=("mnist/ffnnRELU__PGDK_w_0.3_6_500 MNIST 0.1"
      "mnist/mnist_relu_4_1024 MNIST 0.1"
      "mnist/convSmallRELU__DiffAI MNIST 0.1"
      "cifar/ffnnRELU__Point_6_500 CIFAR 0.00784313725490196"
      "cifar/convMedGRELU__PGDK_w_0.0078 CIFAR 0.00784313725490196"
      "cifar/convBigRELU__DiffAI CIFAR 0.00784313725490196")

OUT=$(mkdir -p "$1" && cd "$1" && pwd)
mkdir -p "$OUT/data"
ln -sfn "$HARNESS/constraintflow-gpu/MNIST" "$OUT/data/MNIST"
ln -sfn "$HARNESS/constraintflow-gpu/cifar-10-batches-py" "$OUT/data/cifar-10-batches-py"
cd "$OUT"
export CF_DISABLE_TF32=1

drive() {  # tool onnx out log driver-args... -- script-args...
    local tool=$1 onnx=$2 out=$3 log=$4; shift 4
    local cache=()
    [ "$tool" = cf ] && cache=(--cf_cache_dir "$OUT/cf_cache")
    if [ "$tool" = lirpa ]; then
        "$LIRPA_PY" "$DRIVER" --tool lirpa --onnx "$onnx" --out "$out" "$@"
    else
        PYTHONPATH=$HARNESS/constraintflow-gpu "$CF_PY" "$DRIVER" --tool cf --onnx "$onnx" --out "$out" "$@" "${cache[@]}"
    fi > "$log" 2>&1 || echo "$(date +%T) FAILED: $log"
}

for spec in "${NETS[@]}"; do
    read -r net data eps_values <<< "$spec"
    onnx=$HARNESS/nets/$net.onnx
    name=$(basename "$net")
    [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ] && continue
    data_args=()
    [ "$data" = MNIST ] && data_args=(--normalize --gpu_data)
    for eps in $eps_values; do
        dir=$OUT/$name/eps_$eps
        mkdir -p "$dir"
        common=(--data "$data" --eps "$eps" --seed "$SEED" --batch_size "$BATCH")
        for tool in lirpa cf; do
            echo "$(date +%T) $name eps=$eps $tool: precision and bounds before"
            drive $tool "$onnx" "$dir/before" "$dir/before.prec.$tool.log" --methods IBP --precision_ckpt onnx $PRECISION \
                "${data_args[@]}" -- "${common[@]}"
            drive $tool "$onnx" "$dir/before" "$dir/before.bounds.$tool.log" --methods IBP --bounds_ckpt onnx \
                "${data_args[@]}" -- "${common[@]}"
            echo "$(date +%T) $name eps=$eps $tool: training"
            drive $tool "$onnx" "$dir/$tool" "$dir/train.$tool.log" "${data_args[@]}" -- "${common[@]}" \
                --bound_type IBP --num_epochs "$EPOCHS" --scheduler_name "$SCHEDULER" --scheduler_opts "$SCHEDULER_OPTS"
            echo "$(date +%T) $name eps=$eps $tool: precision after"
            drive $tool "$onnx" "$dir/$tool" "$dir/after.prec.$tool.log" --methods IBP --precision_ckpt "$dir/$tool/ckpt" $PRECISION \
                "${data_args[@]}" -- "${common[@]}"
        done
        # bounds and gradients on each tool's trained weights, computed by both tools
        for trained in lirpa cf; do
            for tool in lirpa cf; do
                drive $tool "$onnx" "$dir/after_$trained" "$dir/after_$trained.bounds.$tool.log" --methods IBP \
                    --bounds_ckpt "$dir/$trained/ckpt" "${data_args[@]}" -- "${common[@]}"
            done
        done
        echo "$(date +%T) $name eps=$eps done"
    done
done

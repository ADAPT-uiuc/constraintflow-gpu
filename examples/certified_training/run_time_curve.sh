#!/bin/bash
# Accuracy over training time, auto_LiRPA vs cf-gpu: both tools train the same recipe (simple_training.py defaults:
# IBP, eps ramp over epochs 3-62, Adam 5e-4 halving every 10 epochs after the ramp, batch 256) from the same fresh
# weights, one after the other, keeping a checkpoint every CKPT_EVERY epochs (and the starting weights). Every
# checkpoint of both tools is then judged by cf-gpu: clean accuracy and verified accuracy (EVAL_METHODS) on the full test set.
#
#   bash run_time_curve.sh OUT_DIR       (summarize with: python report_time_curve.py OUT_DIR)
#   NET=, DATA=, EPS=, EPOCHS=, CKPT_EVERY=, BOUND=, SEED=, BATCH=, NUM_IMAGES= override. Finished steps are skipped.
set -u
HARNESS=/home/db50/cf-harness
HERE=$(cd "$(dirname "$0")" && pwd)
DRIVER=$HERE/compare_driver.py
LIRPA_PY=$HARNESS/auto_LiRPA/lrpa311/bin/python
CF_PY=$HARNESS/constraintflow-gpu/cfg/bin/python3

NET=${NET:-cifar/convBigRELU__DiffAI}   # N32
DATA=${DATA:-CIFAR}
EPS=${EPS:-0.00784313725490196}         # 2/255
EPOCHS=${EPOCHS:-100}
CKPT_EVERY=${CKPT_EVERY:-5}
BOUND=${BOUND:-IBP}
SEED=${SEED:-100}
BATCH=${BATCH:-256}
NUM_IMAGES=${NUM_IMAGES:-10000}         # full test set
BUDGET=${BUDGET:-0}                     # timed seconds per tool; 0 = run all EPOCHS (#FEAT)
TRAIN_EXTRA=${TRAIN_EXTRA:-}            # more simple_training.py args, e.g. "--scheduler_name FixedScheduler --no_lr_decay"
CF_EXTRA=${CF_EXTRA:-}                  # cf-gpu-only simple_training.py args (training and testing), e.g. --cf_no_cudagraphs
EVAL_METHODS=${EVAL_METHODS:-IBP}       # certifier(s) the checkpoints are tested with, e.g. CROWN-IBP for CROWN-FAST

OUT=$(mkdir -p "$1" && cd "$1" && pwd)
mkdir -p "$OUT/data"
ln -sfn "$HARNESS/constraintflow-gpu/MNIST" "$OUT/data/MNIST"
ln -sfn "$HARNESS/constraintflow-gpu/cifar-10-batches-py" "$OUT/data/cifar-10-batches-py"
cd "$OUT"
export CF_DISABLE_TF32=1
ONNX=$HARNESS/nets/$NET.onnx
COMMON=(--data "$DATA" --eps "$EPS" --seed "$SEED" --batch_size "$BATCH")
DATA_ARGS=(--gpu_data)
[ "$DATA" = MNIST ] && DATA_ARGS+=(--normalize)

drive() {  # done-file tool out log driver-args... -- script-args...; skipped when done-file exists
    local done=$1 tool=$2 out=$3 log=$4; shift 4
    [ -e "$done" ] && return
    echo "$(date +%T) $(basename "$log" .log) ($tool)"
    if [ "$tool" = lirpa ]; then
        "$LIRPA_PY" "$DRIVER" --tool lirpa --onnx "$ONNX" --out "$out" "$@"
    else
        PYTHONPATH=$HARNESS/constraintflow-gpu "$CF_PY" "$DRIVER" --tool cf --onnx "$ONNX" --out "$out" "$@" \
            --cf_cache_dir "$OUT/cf_cache" $CF_EXTRA
    fi > "$log" 2>&1 || echo "$(date +%T) FAILED: $log"
}

# 1. training, one tool at a time
for tool in lirpa cf; do
    mkdir -p "$tool"
    drive "$tool/record.json" $tool "$tool" "$tool/train.log" --reinit --ckpt_every "$CKPT_EVERY" --time_budget "$BUDGET" \
        "${DATA_ARGS[@]}" -- "${COMMON[@]}" --bound_type "$BOUND" --num_epochs "$EPOCHS" $TRAIN_EXTRA
done
# 2. every checkpoint of both tools judged by cf-gpu (clean + IBP verified)
for tool in lirpa cf; do
    drive "$tool/precision_curve_cf.json" cf "$tool" "$tool/curve.log" --methods "$EVAL_METHODS" --num_images "$NUM_IMAGES" \
        --eval_batch_size 100 --precision_ckpt "$tool/ckpts" "${DATA_ARGS[@]}" -- "${COMMON[@]}"
done
echo "$(date +%T) done"

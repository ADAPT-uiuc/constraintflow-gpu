#!/bin/bash
# Certified training per certifier (IBP, CROWN-FAST = pure CROWN-IBP, zid, CROWN; on nets marked "stops" also zono and
# DeepPoly stopped KK layers back, DP-stopKK), auto_LiRPA vs cf-gpu (zid, zono, DP-stopKK: cf-gpu only),
# trained and evaluated at MNIST 0.1 / CIFAR 2/255. Every model is judged by the same evaluators (clean accuracy and
# verified precision under every certifier, IBP, CROWN-IBP, CROWN, zid, on the full test set): on the ONNX weights, on a checkpoint every
# CKPT_EVERY epochs (by cf-gpu), and on the final weights (by both tools); final weights are also judged by every
# certifier (cross product: + zono and the net's distinct DP-stopKK, by cf-gpu). Bounds and gradients of IBP, CROWN-IBP
# and CROWN are compared between the tools on the ONNX weights and on each tool's trained weights.
#
#   bash run_grid.sh OUT_DIR          (summarize with: python report_grid.py OUT_DIR)
#   EPOCHS=, CKPT_EVERY=, SCHEDULER=, SCHEDULER_OPTS=, BATCH=, SEED=, STOPS= override; ONLY=<net name> runs one net.
#   Finished steps are skipped, so a killed run resumes where it stopped.
set -u
HARNESS=/home/db50/cf-harness
HERE=$(cd "$(dirname "$0")" && pwd)
DRIVER=$HERE/compare_driver.py
LIRPA_PY=$HARNESS/auto_LiRPA/lrpa311/bin/python
CF_PY=$HARNESS/constraintflow-gpu/cfg/bin/python3

EPOCHS=${EPOCHS:-100}
CKPT_EVERY=${CKPT_EVERY:-10}
SCHEDULER=${SCHEDULER:-SmoothedScheduler}
SCHEDULER_OPTS=${SCHEDULER_OPTS:-start=3,length=60}  # auto_LiRPA simple_training.py default: eps ramps over epochs 3-62
BATCH=${BATCH:-256}
SEED=${SEED:-100}
CERTIFIERS=(IBP CROWN-FAST zid CROWN)
CORE=IBP,CROWN-IBP,CROWN,zid  # curve evaluators (+ the run's own certifier)
EVAL_OPTS="--num_images 10000 --eval_batch_size 100"  # full test set
EVAL="--methods $CORE $EVAL_OPTS"
STOPS=${STOPS:-$(printf 'DP-stop%02d,' $(seq 0 18) | sed 's/,$//')}  # by increasing depth; deeper than the net = duplicates
COMPARE="--methods IBP,CROWN-IBP,CROWN"  # bounds and gradients, both tools
# net, dataset, eps, extra training certifiers (stops: zono + distinct DP-stopKK; core: none); the FC net last
NETS=("mnist/convSmallRELU__Point MNIST 0.1 stops"
      "cifar/convSmallRELU__Point CIFAR 0.00784313725490196 stops"
      "mnist/ffnnRELU__PGDK_w_0.3_6_500 MNIST 0.1 core")

OUT=$(mkdir -p "$1" && cd "$1" && pwd)
mkdir -p "$OUT/data"
ln -sfn "$HARNESS/constraintflow-gpu/MNIST" "$OUT/data/MNIST"
ln -sfn "$HARNESS/constraintflow-gpu/cifar-10-batches-py" "$OUT/data/cifar-10-batches-py"
cd "$OUT"
export CF_DISABLE_TF32=1

drive() {  # done-file tool onnx out log driver-args... -- script-args...; skipped when done-file exists
    local done=$1 tool=$2 onnx=$3 out=$4 log=$5; shift 5
    [ -e "$done" ] && return
    local cache=()
    [ "$tool" = cf ] && cache=(--cf_cache_dir "$OUT/cf_cache")
    echo "$(date +%T) $(basename "$log" .log)"
    if [ "$tool" = lirpa ]; then
        "$LIRPA_PY" "$DRIVER" --tool lirpa --onnx "$onnx" --out "$out" "$@"
    else
        PYTHONPATH=$HARNESS/constraintflow-gpu "$CF_PY" "$DRIVER" --tool cf --onnx "$onnx" --out "$out" "$@" "${cache[@]}"
    fi > "$log" 2>&1 || echo "$(date +%T) FAILED: $log"
}

for spec in "${NETS[@]}"; do
    read -r net data eps extra <<< "$spec"
    onnx=$HARNESS/nets/$net.onnx
    name=$(basename "$net")_$data
    [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ] && continue
    dir=$OUT/$name
    mkdir -p "$dir/before"
    data_args=(--gpu_data)  # datasets on the GPU for both tools (CIFAR augmentation too)
    [ "$data" = MNIST ] && data_args+=(--normalize)
    common=(--data "$data" --eps "$eps" --seed "$SEED" --batch_size "$BATCH")
    for tool in lirpa cf; do
        drive "$dir/before/precision_onnx_$tool.json" $tool "$onnx" "$dir/before" "$dir/before/precision.$tool.log" \
            $EVAL --precision_ckpt onnx "${data_args[@]}" -- "${common[@]}"
        drive "$dir/before/bounds_onnx_$tool.pt" $tool "$onnx" "$dir/before" "$dir/before/bounds.$tool.log" \
            $COMPARE --bounds_ckpt onnx "${data_args[@]}" -- "${common[@]}"
    done
    # the custom stops that differ on this net, and the cross-product evaluators
    drive "$dir/distinct_methods.json" cf "$onnx" "$dir" "$dir/distinct.log" --distinct_methods --methods "$STOPS" \
        $EVAL_OPTS "${data_args[@]}" -- "${common[@]}"
    stops=($(python3 -c "import json; print(' '.join(json.load(open('$dir/distinct_methods.json'))))"))
    ALL="$CORE,zono,$(IFS=,; echo "${stops[*]}")"
    drive "$dir/before/precision_all_cf.json" cf "$onnx" "$dir/before" "$dir/before/precision_all.cf.log" \
        --methods "$ALL" $EVAL_OPTS --name precision_all --precision_ckpt onnx "${data_args[@]}" -- "${common[@]}"
    certs=("${CERTIFIERS[@]}")
    [ "$extra" = stops ] && certs+=(zono "${stops[@]}")
    for cert in "${certs[@]}"; do
        tools=(lirpa cf)
        case $cert in zid|zono|DP-stop*) tools=(cf) ;; esac
        curve=$CORE
        case ",$CORE," in *",$cert,"*) ;; *) curve=$CORE,$cert ;; esac
        for tool in "${tools[@]}"; do
            run=$dir/$cert/$tool
            mkdir -p "$run"
            drive "$run/record.json" $tool "$onnx" "$run" "$run/train.log" --ckpt_every "$CKPT_EVERY" "${data_args[@]}" -- \
                "${common[@]}" --bound_type "$cert" --num_epochs "$EPOCHS" --scheduler_name "$SCHEDULER" --scheduler_opts "$SCHEDULER_OPTS"
            # the curve: every kept checkpoint judged by cf-gpu's evaluators
            drive "$run/precision_curve_cf.json" cf "$onnx" "$run" "$run/curve.log" \
                --methods "$curve" $EVAL_OPTS --precision_ckpt "$run/ckpts" "${data_args[@]}" -- "${common[@]}"
            # the final weights judged by every certifier
            drive "$run/precision_all_cf.json" cf "$onnx" "$run" "$run/precision_all.log" \
                --methods "$ALL" $EVAL_OPTS --name precision_all --precision_ckpt "$run/ckpt" "${data_args[@]}" -- "${common[@]}"
        done
        [ ${#tools[@]} = 1 ] && continue
        # final weights of each tool, judged and bounded by both tools
        for trained in lirpa cf; do
            for tool in lirpa cf; do
                drive "$dir/$cert/$trained/precision_ckpt_$tool.json" $tool "$onnx" "$dir/$cert/$trained" \
                    "$dir/$cert/$trained/precision_final.$tool.log" $EVAL --precision_ckpt "$dir/$cert/$trained/ckpt" "${data_args[@]}" -- "${common[@]}"
                mkdir -p "$dir/$cert/bounds_$trained"
                drive "$dir/$cert/bounds_$trained/bounds_ckpt_$tool.pt" $tool "$onnx" "$dir/$cert/bounds_$trained" \
                    "$dir/$cert/bounds_$trained/bounds.$tool.log" $COMPARE --bounds_ckpt "$dir/$cert/$trained/ckpt" "${data_args[@]}" -- "${common[@]}"
            done
        done
    done
    echo "$(date +%T) $name done"
done

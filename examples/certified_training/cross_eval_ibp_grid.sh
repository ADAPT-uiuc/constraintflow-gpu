#!/bin/bash
# Cross-evaluation for run_ibp_grid.sh: each tool's IBP precision on the other tool's trained weights.
# Writes precision_ckpt_<evaluating tool>.json next to the trained checkpoint (OUT/<net>/eps_<e>/<trained>/).
#
#   bash cross_eval_ibp_grid.sh OUT_DIR       (only finished runs; rerun later for the rest)
set -u
HARNESS=/home/db50/cf-harness
HERE=$(cd "$(dirname "$0")" && pwd)
DRIVER=$HERE/compare_driver.py
LIRPA_PY=$HARNESS/auto_LiRPA/lrpa311/bin/python
CF_PY=$HARNESS/constraintflow-gpu/cfg/bin/python3
PRECISION="--num_images 10000 --eval_batch_size 100"

OUT=$(cd "$1" && pwd)
cd "$OUT"
export CF_DISABLE_TF32=1

for dir in "$OUT"/*/eps_*; do
    name=$(basename "$(dirname "$dir")")
    eps=${dir##*eps_}
    #FIX FOR: some nets share a file name across nets/mnist and nets/cifar; take the dataset from the run.
    data=$(grep -o '"data": "[A-Z]*"' "$dir"/*/record.json | head -1 | grep -o '[A-Z]*"$' | tr -d '"')
    [ -n "$data" ] || { echo "skip $name: no finished run"; continue; }
    onnx=$HARNESS/nets/${data,,}/$name.onnx
    data_args=(); [ "$data" = MNIST ] && data_args=(--normalize)
    for pair in "lirpa cf" "cf lirpa"; do
        read -r trained tool <<< "$pair"
        # record.json is written when training ends; ckpt is rewritten every epoch
        [ -f "$dir/$trained/record.json" ] || { echo "skip $name: $trained training not finished"; continue; }
        [ -f "$dir/$trained/precision_ckpt_$tool.json" ] && { echo "skip $name: $trained-by-$tool done"; continue; }
        log=$dir/cross.$trained-by-$tool.log
        if [ "$tool" = lirpa ]; then
            "$LIRPA_PY" "$DRIVER" --tool lirpa --onnx "$onnx" --out "$dir/$trained" --methods IBP \
                --precision_ckpt "$dir/$trained/ckpt" $PRECISION "${data_args[@]}" -- --data $data --eps "$eps"
        else
            PYTHONPATH=$HARNESS/constraintflow-gpu "$CF_PY" "$DRIVER" --tool cf --onnx "$onnx" --out "$dir/$trained" --methods IBP \
                --precision_ckpt "$dir/$trained/ckpt" $PRECISION "${data_args[@]}" -- --data $data --eps "$eps" \
                --cf_cache_dir "$OUT/cf_cache"
        fi > "$log" 2>&1 || echo "FAILED: $log"
        echo "$(date +%T) $name: $trained-trained weights evaluated by $tool"
    done
done

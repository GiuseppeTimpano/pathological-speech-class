#!/usr/bin/env bash
# Regenerate the XAI results for the multiclass CNN-BiGRU (ECA+TAM) at 8kHz, on all 5 folds
# and all 3 classes, Runs up to
# MAX_PARALLEL explain.py processes at once. --common_sr 8000 and --ig_eval_mode are
# mandatory: see the flags passed to explain.py below for why.
# Output per (fold, class): results/multiclass/cnn_bigru_eca_tam/explanations/fold<f>/<HS|PD|ALS>/
#   attribution_sample_<i>.png  (one |IG| map per correctly classified recording)
#   attribution_sample_0.npy    (normalised map of sample 0, used by make_ig_figure.py)
#   xai_metrics.csv             (FC, FE, FM, SPS, COMP for that fold+class)
# Skips any (fold, class) whose xai_metrics.csv already exists, so this script can be
# re-run safely after an interruption without redoing completed work.
#
# MAX_PARALLEL: how many explain.py processes to run at once. The model is tiny (2.1M params,
# efficiency table), so GPU memory is not the bottleneck; start at 3 and raise it if `nvidia-smi` (or
# Activity Monitor / GPU history on Mac) shows plenty of headroom, or lower it if you see
# out-of-memory errors or the machine becoming unresponsive.
MAX_PARALLEL="${MAX_PARALLEL:-3}"

set -uo pipefail
mkdir -p logs

CLASS_NAMES=(HS PD ALS)
pids=()

wait_for_slot() {
    while [[ "${#pids[@]}" -ge "$MAX_PARALLEL" ]]; do
        sleep 5
        alive=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        pids=("${alive[@]}")
    done
}

for fold in 0 1 2 3 4; do
    for target_label in 0 1 2; do
        class_name="${CLASS_NAMES[$target_label]}"
        out_dir="results/multiclass/cnn_bigru_eca_tam/explanations/fold${fold}/${class_name}"
        if [[ -f "${out_dir}/xai_metrics.csv" ]]; then
            echo "=== multiclass fold ${fold} / ${class_name}: already done, skipping ==="
            continue
        fi
        wait_for_slot
        echo "=== multiclass fold ${fold} / ${class_name}: launching (log: logs/xai_fold${fold}_${class_name}.log) ==="
        (
            python explain.py --task multiclass --eca --tam --common_sr 8000 --ig_eval_mode \
                --fold "$fold" --target_label "$target_label" \
                > "logs/xai_fold${fold}_${class_name}.log" 2>&1
            echo "OK   multiclass fold ${fold} / ${class_name}"
        ) &
        pids+=("$!")
    done
done

echo "All combinations launched, waiting for the last ${#pids[@]} to finish..."
wait

echo "All XAI runs done. Next steps:"
echo "  python scripts/make_xai_table.py                                # XAI metrics table"
echo "  python scripts/make_ig_figure.py                                 # IG example figure"

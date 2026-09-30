#!/usr/bin/env bash
# CNN-BiGRU ablation (baseline / +ECA / +TAM / +ECA+TAM) - multiclass task, run CONCURRENTLY:
# each variant uses ~2.7GB VRAM, so all 4 fit comfortably on a 32GB GPU together.
# --common_sr 8000: resamples every clip to 8kHz before the log-Mel front-end, so PD
# (native 16/44.1kHz) and ALS (native 8kHz) no longer differ in bandwidth - removes
# sampling rate as a trivial dataset-identity cue for this task.
# Requires: python preprocess.py --raw_dir data/raw --out_dir data/processed
# Output: results/multiclass/cnn_bigru<_eca><_tam>/{kfold_results.csv, kfold_summary.csv, ...}
# Per-variant logs: logs/multiclass_<run_name>.log
# Skips any run whose kfold_summary.csv already exists. NOTE: the run folder name does
# not encode --common_sr, so don't also run a native-rate multiclass variant under the
# same run names or the skip check will mistake one for the other.
set -uo pipefail
mkdir -p logs
declare -A pids

run() {
    local run_name="$1"; shift
    local summary="results/multiclass/${run_name}/kfold_summary.csv"
    if [[ -f "$summary" ]]; then
        echo "=== multiclass / ${run_name}: already done, skipping (${summary} exists) ==="
        return
    fi
    echo "=== multiclass / ${run_name}: starting in background, log at logs/multiclass_${run_name}.log ==="
    python train.py --task multiclass --model cnn-bigru --weighted_loss --common_sr 8000 "$@" \
        > "logs/multiclass_${run_name}.log" 2>&1 &
    pids["$run_name"]=$!
}

echo "=== multiclass: warming up mapped-dataset cache (sequential, once) ==="
python scripts/prepare_features.py --task multiclass --common_sr 8000

run cnn_bigru
run cnn_bigru_eca --eca
run cnn_bigru_tam --tam
run cnn_bigru_eca_tam --eca --tam

echo "Launched: ${!pids[@]} - waiting for them to finish..."
failed=0
for run_name in "${!pids[@]}"; do
    if wait "${pids[$run_name]}"; then
        echo "OK   multiclass / ${run_name}"
    else
        echo "FAIL multiclass / ${run_name} - see logs/multiclass_${run_name}.log"
        failed=1
    fi
done
[[ $failed -eq 0 ]] && echo "All multiclass CNN-BiGRU variants finished successfully."
exit $failed

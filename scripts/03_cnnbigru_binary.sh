#!/usr/bin/env bash
# CNN-BiGRU ablation (baseline / +ECA / +TAM / +ECA+TAM) - binary task, run CONCURRENTLY:
# each variant uses ~2.7GB VRAM, so all 4 fit comfortably on a 32GB GPU together.
# Native per-file sampling rate (8/16/44.1kHz): no --common_sr here, decided on purpose,
# since the binary task does not confound class with source dataset.
# Requires: python preprocess.py --raw_dir data/raw --out_dir data/processed
# Output: results/binary/cnn_bigru<_eca><_tam>/{kfold_results.csv, kfold_summary.csv, ...}
# Per-variant logs: logs/binary_<run_name>.log
# Skips any run whose kfold_summary.csv already exists.
set -uo pipefail
mkdir -p logs
declare -A pids

run() {
    local run_name="$1"; shift
    local summary="results/binary/${run_name}/kfold_summary.csv"
    if [[ -f "$summary" ]]; then
        echo "=== binary / ${run_name}: already done, skipping (${summary} exists) ==="
        return
    fi
    echo "=== binary / ${run_name}: starting in background, log at logs/binary_${run_name}.log ==="
    python train.py --task binary --model cnn-bigru --weighted_loss "$@" \
        > "logs/binary_${run_name}.log" 2>&1 &
    pids["$run_name"]=$!
}

echo "=== binary: warming up mapped-dataset cache (sequential, once) ==="
python scripts/prepare_features.py --task binary

run cnn_bigru
run cnn_bigru_eca --eca
run cnn_bigru_tam --tam
run cnn_bigru_eca_tam --eca --tam

echo "Launched: ${!pids[@]} - waiting for them to finish..."
failed=0
for run_name in "${!pids[@]}"; do
    if wait "${pids[$run_name]}"; then
        echo "OK   binary / ${run_name}"
    else
        echo "FAIL binary / ${run_name} - see logs/binary_${run_name}.log"
        failed=1
    fi
done
[[ $failed -eq 0 ]] && echo "All binary CNN-BiGRU variants finished successfully."
exit $failed

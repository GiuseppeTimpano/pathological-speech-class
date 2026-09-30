#!/usr/bin/env bash
# Fair inference-speed re-measurement - multiclass task, run SEQUENTIALLY on purpose:
# same reasoning as 05_speedtest_binary.sh. Run this only after that script has
# finished (never at the same time), so the GPU is never shared between measurements.
# --common_sr 8000: must match how each run was actually trained (04_cnnbigru_multiclass.sh
# and 02_baselines_multiclass.sh both used it) - a mismatch here does not silently give a
# wrong number, it makes the checkpoint fail to load (shape mismatch).
# Reuses the checkpoints already saved in results/multiclass/<run_name>/best_model_fold*/
# (no retraining). Does NOT touch results/ at all: everything this script produces goes
# into its own tree, results_inference_speed/multiclass/<run_name>/, kept separate on purpose.
# Output: results_inference_speed/multiclass/<run_name>/{inference_speed.csv, inference_speed_summary.csv}
# Skips any run whose inference_speed_summary.csv already exists.
set -euo pipefail
mkdir -p logs

run() {
    local run_name="$1"; shift
    local summary="results_inference_speed/multiclass/${run_name}/inference_speed_summary.csv"
    if [[ -f "$summary" ]]; then
        echo "=== multiclass / ${run_name}: already measured, skipping (${summary} exists) ==="
        return
    fi
    echo "=== multiclass / ${run_name}: measuring inference speed, log at logs/speedtest_multiclass_${run_name}.log ==="
    python scripts/measure_inference_speed.py --task multiclass --common_sr 8000 --repeats 3 "$@" \
        > "logs/speedtest_multiclass_${run_name}.log" 2>&1
    echo "OK   multiclass / ${run_name}"
}

# Proposed model: same flags as the trained run (04_cnnbigru_multiclass.sh).
run cnn_bigru_eca_tam --model cnn-bigru --eca --tam

# Baselines: same flags as the trained run (02_baselines_multiclass.sh).
run wav2vec2 --model wav2vec2
run hubert   --model hubert
run ast      --model ast

echo "All multiclass inference-speed measurements done."
echo "Summaries: results_inference_speed/multiclass/{cnn_bigru_eca_tam,wav2vec2,hubert,ast}/inference_speed_summary.csv"

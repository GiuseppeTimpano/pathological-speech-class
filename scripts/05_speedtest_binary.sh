#!/usr/bin/env bash
# Fair inference-speed re-measurement - binary task, run SEQUENTIALLY on purpose:
# unlike the training scripts, these numbers feed the efficiency table and are only
# comparable if every model gets the GPU to itself, one at a time. Do NOT run this
# alongside 06_speedtest_multiclass.sh or anything else that touches the GPU.
# Reuses the checkpoints already saved in results/binary/<run_name>/best_model_fold*/
# (no retraining). Does NOT touch results/ at all: everything this script produces goes
# into its own tree, results_inference_speed/binary/<run_name>/, kept separate on purpose.
# Output: results_inference_speed/binary/<run_name>/{inference_speed.csv, inference_speed_summary.csv}
# Skips any run whose inference_speed_summary.csv already exists, so this script can be
# re-run safely after an interruption without redoing completed work.
set -euo pipefail
mkdir -p logs

run() {
    local run_name="$1"; shift
    local summary="results_inference_speed/binary/${run_name}/inference_speed_summary.csv"
    if [[ -f "$summary" ]]; then
        echo "=== binary / ${run_name}: already measured, skipping (${summary} exists) ==="
        return
    fi
    echo "=== binary / ${run_name}: measuring inference speed, log at logs/speedtest_binary_${run_name}.log ==="
    python scripts/measure_inference_speed.py --task binary --repeats 3 "$@" \
        > "logs/speedtest_binary_${run_name}.log" 2>&1
    echo "OK   binary / ${run_name}"
}

# Proposed model: same flags as the trained run (03_cnnbigru_binary.sh), native sampling rate.
run cnn_bigru_eca_tam --model cnn-bigru --eca --tam

# Baselines: same flags as the trained run (01_baselines_binary.sh), native sampling rate.
run wav2vec2 --model wav2vec2
run hubert   --model hubert
run ast      --model ast

echo "All binary inference-speed measurements done."
echo "Summaries: results_inference_speed/binary/{cnn_bigru_eca_tam,wav2vec2,hubert,ast}/inference_speed_summary.csv"

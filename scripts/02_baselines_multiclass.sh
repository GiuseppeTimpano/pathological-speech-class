#!/usr/bin/env bash
# Pretrained baselines (Wav2Vec2, HuBERT, AST) - multiclass task (HS vs PD vs ALS).
# --common_sr 8000: audio band-limited to 8kHz, then resampled to the 16kHz the models expect -
# same effective bandwidth for PD and ALS, as for the multiclass CNN-BiGRU (04_cnnbigru_multiclass.sh).
# Requires: python preprocess.py --raw_dir data/raw --out_dir data/processed
# Output: results/multiclass/<model>/{kfold_results.csv, kfold_summary.csv, ...}
# Skips any run whose kfold_summary.csv already exists.
set -euo pipefail

run() {
    local task="$1"; shift
    local run_name="$1"; shift
    local summary="results/${task}/${run_name}/kfold_summary.csv"
    if [[ -f "$summary" ]]; then
        echo "=== ${task} / ${run_name}: already done, skipping (${summary} exists) ==="
        return
    fi
    echo "=== ${task} / ${run_name} ==="
    python train.py --task "$task" "$@"
}

for model in wav2vec2 hubert ast; do
    run multiclass "$model" --model "$model" --weighted_loss --common_sr 8000
done

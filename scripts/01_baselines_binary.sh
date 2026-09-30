#!/usr/bin/env bash
# Pretrained baselines (Wav2Vec2, HuBERT, AST) - binary task (HS vs pathological).
# Requires: python preprocess.py --raw_dir data/raw --out_dir data/processed
# Output: results/binary/<model>/{kfold_results.csv, kfold_summary.csv, ...}
# Skips any run whose kfold_summary.csv already exists, so this script can be re-run
# safely after an interruption without redoing completed work.
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
    run binary "$model" --model "$model" --weighted_loss
done

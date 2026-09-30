#!/usr/bin/env bash
# Corpus-confound control experiment: quantifies how much of the
# multiclass PD-vs-ALS separation could actually be the model learning to tell apart the
# two source corpora, rather than the two diseases.
#
# Uses only the healthy subjects (HS) of both datasets, relabelled by SOURCE CORPUS instead
# of disease (0 = Italian Parkinson's Voice and Speech, 1 = VOC-ALS), and trains the same
# CNN-BiGRU (ECA+TAM, 8kHz) as a binary classifier on that label. A high AUC here means the
# two corpora are easy to tell apart even among healthy speakers (recording setup, mic,
# room, etc.), so part of what looks like disease-specific signal in the multiclass task
# could be corpus-specific signal instead; a low AUC (near chance, 0.5) means the corpora
# are not trivially separable and strengthens the multiclass PD/ALS result.
#
# Output: results_confound/binary/cnn_bigru_eca_tam/kfold_summary.csv (same format as any
# other train.py run); the patient-level AUC in that file is the result of this control.
# Skips both steps if their output already exists, so this script is safe to re-run.
set -euo pipefail
mkdir -p logs

if [[ -f data/processed_confound/dataset.csv ]]; then
    echo "=== confound dataset: already built, skipping (data/processed_confound/dataset.csv exists) ==="
else
    echo "=== confound dataset: building from data/processed, log at logs/confound_dataset.log ==="
    python scripts/make_confound_dataset.py --data_dir data/processed --out_dir data/processed_confound \
        > logs/confound_dataset.log 2>&1
    echo "OK   confound dataset built"
fi

summary="results_confound/binary/cnn_bigru_eca_tam/kfold_summary.csv"
if [[ -f "$summary" ]]; then
    echo "=== confound control training: already done, skipping (${summary} exists) ==="
else
    echo "=== confound control training: running train.py, log at logs/confound_train.log ==="
    python train.py --task binary --model cnn-bigru --eca --tam --weighted_loss --common_sr 8000 \
        --data_dir data/processed_confound --output_dir results_confound \
        > logs/confound_train.log 2>&1
    echo "OK   confound control training done"
fi

echo "Done. Patient-level AUC:"
echo "  results_confound/binary/cnn_bigru_eca_tam/kfold_summary.csv"

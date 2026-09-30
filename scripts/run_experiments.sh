#!/usr/bin/env bash
# Runs every experiment, in order, then builds the LaTeX tables and figures.
# Each step's script skips work it has already done, so this is safe to re-run after an
# interruption. Run the baselines with the GPU otherwise idle: their eval_runtime feeds the
# efficiency table. Steps 5-6 (inference-speed re-measurement) must be run one at
# a time with nothing else on the GPU - see the comments in those scripts.
#   1-2. pretrained baselines (Wav2Vec2, HuBERT, AST), binary and multiclass (multiclass band-limited to 8kHz)
#   3-4. CNN-BiGRU ablation (no attention / ECA / TAM / ECA+TAM), binary (native rates) and multiclass (8kHz)
#   5-6. fair inference-speed re-measurement (idle GPU, one model at a time) - feeds the efficiency table
#   7.   Integrated Gradients for each class of the multiclass CNN-BiGRU (ECA+TAM), all 5 folds
#   8.   corpus-confound control (healthy subjects only, label = source corpus)
set -euo pipefail

bash scripts/01_baselines_binary.sh
bash scripts/02_baselines_multiclass.sh
bash scripts/03_cnnbigru_binary.sh
bash scripts/04_cnnbigru_multiclass.sh
bash scripts/05_speedtest_binary.sh
bash scripts/06_speedtest_multiclass.sh
MAX_PARALLEL="${MAX_PARALLEL:-3}" bash scripts/07_xai_multiclass.sh
bash scripts/08_confound_control.sh

python scripts/make_xai_table.py                                  # XAI metrics table
python scripts/make_ig_figure.py                                    # IG example figure
python make_tables.py --results_dir results > results/tables.tex
echo "Tables written to results/tables.tex"

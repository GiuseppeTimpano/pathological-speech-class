# Efficient CNN–BiGRU with Dual Attention for Parkinson's and ALS Speech Classification

Code for the paper *"Efficient CNN–BiGRU with Dual Attention for Parkinson's and ALS Speech Classification:
A Lightweight Alternative to Self-Supervised Models"* (G. Timpano, M. E. Caligiuri, M. Cannataro, P. H. Guzzi,
P. Veltri, P. Vizza).

A 2.1M-parameter CNN–BiGRU classifies log-Mel spectrograms of short vocal tasks as healthy (HS), Parkinson's
disease (PD) or amyotrophic lateral sclerosis (ALS). Two attention modules are added: Efficient Channel
Attention (ECA) after the convolutional blocks and a Temporal Attention Module (TAM) after the BiGRU. The model
is compared with Wav2Vec2, HuBERT and the Audio Spectrogram Transformer (86–96M parameters) under subject-level
5-fold cross-validation, and explained with Integrated Gradients. A corpus-confound control experiment checks
how much of the multiclass PD-vs-ALS separation could be explained by the two diseases coming from different
source corpora rather than by disease-specific signal.

```
log-Mel (1001 × 128) → 3 × [Conv3×3 → BN → ReLU → MaxPool] → ECA → 2-layer BiGRU (128) → TAM → linear
```

## Repository layout

```
preprocess.py          VAD + Wiener filtering of the raw recordings, builds data/processed/dataset.csv
train.py               subject-level k-fold training and clip/subject-level evaluation
explain.py             Integrated Gradients maps and Quantus faithfulness/complexity metrics
calibration.py         Expected/Maximum Calibration Error, Brier score, reliability diagrams
make_tables.py         LaTeX tables from the k-fold summaries
pathospeech/
    preprocessing.py   voice activity detection, Wiener filter, dataset index
    features.py        log-Mel spectrogram (STFT parameters adapted to the sampling rate)
    data.py            subject-level stratified folds, feature extraction for every model
    models/            CNN-BiGRU, ECA and temporal attention, pretrained baselines
    training.py        weighted loss, clip-level metrics, subject-level aggregation
    explainability.py  Integrated Gradients and Quantus metrics
scripts/
    run_experiments.sh          runs every step below in order, then builds tables/figures
    01_baselines_binary.sh          pretrained baselines (Wav2Vec2, HuBERT, AST), binary
    02_baselines_multiclass.sh      pretrained baselines, multiclass (band-limited to 8kHz)
    03_cnnbigru_binary.sh           CNN-BiGRU ablation (no attention / ECA / TAM / ECA+TAM), binary, native rates
    04_cnnbigru_multiclass.sh       CNN-BiGRU ablation, multiclass (8kHz)
    05_speedtest_binary.sh          fair inference-speed re-measurement, binary (idle GPU, one model at a time)
    06_speedtest_multiclass.sh      fair inference-speed re-measurement, multiclass
    measure_inference_speed.py      shared helper used by 05/06, no retraining
    07_xai_multiclass.sh            Integrated Gradients for each class of CNN-BiGRU (ECA+TAM), all 5 folds
    08_confound_control.sh          corpus-confound control experiment (see "Confound control" below)
    make_xai_table.py               aggregates xai_metrics.csv across folds/classes into an XAI metrics table
    make_ig_figure.py               builds the Integrated Gradients example figure (HS/PD/ALS)
    prepare_features.py             pre-computes and caches the log-Mel features (used by 03/04)
    make_confound_dataset.py        builds the healthy-only, corpus-labelled dataset used by 08
```

Each numbered script skips work whose output already exists, so `scripts/run_experiments.sh` can be re-run
safely after an interruption.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Experiments were run with Python 3.12, PyTorch 2.7.1 and Transformers 4.53.3 on a single NVIDIA RTX 5000 Ada GPU.

## Data

The recordings are not redistributed here; both corpora are publicly available from their providers:

| Corpus | Subjects | Access |
|---|---|---|
| Italian Parkinson's Voice and Speech | 28 PD, 37 HS | IEEE DataPort, [doi:10.21227/aw6b-tg17](https://doi.org/10.21227/aw6b-tg17) |
| VOC-ALS | 102 ALS, 51 HS | Synapse, [syn53009474](https://www.synapse.org/Synapse:syn53009474/wiki/624730) (registration required) |

Place the recordings in one folder per subject, grouping the healthy speakers of both corpora together:

```
data/raw/
├── Healthy/<subject_id>/*.wav
├── Disease_PD/<subject_id>/*.wav
└── Disease_ALS/<subject_id>/*.wav
```

Then clean the signals and build the index:

```bash
python preprocess.py --raw_dir data/raw --out_dir data/processed
```

`data/processed/dataset.csv` lists every recording with its subject and multiclass label (0 = HS, 1 = PD,
2 = ALS). The binary task relabels PD and ALS as pathological (1).

## Training

```bash
# proposed model, binary (native sampling rates)
python train.py --task binary --model cnn-bigru --eca --tam --weighted_loss

# proposed model, multiclass (band-limited to 8kHz, the two corpora
# were recorded at different native rates, so multiclass runs fix a common rate)
python train.py --task multiclass --model cnn-bigru --eca --tam --weighted_loss --common_sr 8000

# baselines: wav2vec2 | hubert | ast
python train.py --task multiclass --model hubert --weighted_loss --common_sr 8000

# every experiment (baselines, ablation, explanations, confound control, tables)
bash scripts/run_experiments.sh
```

`--common_sr <Hz>` resamples/band-limits all audio to a common rate before feature extraction; it is required
for every multiclass run (see `scripts/02_baselines_multiclass.sh` and `scripts/04_cnnbigru_multiclass.sh`)
and optional for binary runs, which use each corpus's native rate.

Subjects are split with a stratified group 5-fold (seed 101), so all recordings of a speaker are in the same
fold. Training uses AdamW (Hugging Face `Trainer` defaults, no weight decay) with learning rate 3·10⁻⁵ and a
linear schedule with 10% warm-up, batch size 16, at most 200 epochs, early
stopping on the validation clip-level weighted F1 (patience 30) and, with `--weighted_loss`, inverse-frequency
class weights. Subject-level predictions average the clip probabilities of each speaker.

Each run writes to `<output_dir>/<task>/<run>/` (`<output_dir>` defaults to `results/`; `scripts/run_experiments.sh`
uses `results/` and `results_confound/` for the main and confound-control runs respectively):

| File | Content |
|---|---|
| `best_model_fold<k>/` | best checkpoint of fold *k* |
| `predictions_fold<k>.csv` | clip-level class probabilities with subject id |
| `kfold_results.csv` | clip- and subject-level metrics, inference time, parameter count per fold |
| `kfold_summary.csv` | mean and standard deviation over folds |

## Explainability

```bash
python explain.py --task multiclass --eca --tam --fold 0 --target_label 2 --common_sr 8000 --ig_eval_mode
```

Integrated Gradients (zero baseline, 50 steps) are computed for the correctly classified validation recordings
of the requested class. Maps are saved to `<output_dir>/<task>/<run>/explanations/fold<k>/<class>/` together with
`xai_metrics.csv` (Faithfulness Correlation, Faithfulness Estimate, Monotonicity, Sparseness, Complexity).

`--ig_eval_mode` keeps the model in evaluation mode and disables cuDNN, giving deterministic attributions; this
is the mode used for all reported XAI results. Without it, on GPU the attributions are computed
with the model in training mode instead, because cuDNN does not back-propagate through a GRU in evaluation mode.

`scripts/07_xai_multiclass.sh` runs `explain.py` with the settings above for all 5 folds and all 3 classes of
the multiclass CNN-BiGRU (ECA+TAM), up to `MAX_PARALLEL` processes at once (default 3; raise or lower it
depending on GPU headroom). `scripts/make_xai_table.py` then aggregates the resulting `xai_metrics.csv` files
into an XAI metrics table, and `scripts/make_ig_figure.py` builds the Integrated Gradients
example figure from the saved `.npy` attribution maps.

## Confound control

```bash
bash scripts/08_confound_control.sh
```

Trains the same CNN-BiGRU (ECA+TAM, 8kHz) as a binary classifier on the healthy subjects (HS) of both corpora
only, relabelled by *source corpus* instead of disease (0 = Italian Parkinson's Voice and Speech, 1 = VOC-ALS).
A high patient-level AUC means the two corpora are easy to tell apart even among healthy speakers, so part of
what looks like disease-specific signal in the multiclass PD/ALS task could instead be corpus-specific; a low
AUC (near chance) strengthens the multiclass result. The script builds the relabelled dataset with
`scripts/make_confound_dataset.py` and writes results to `results_confound/binary/cnn_bigru_eca_tam/kfold_summary.csv`,
skipping either step if its output already exists.

## Tables

```bash
python make_tables.py --results_dir results > results/tables.tex
```

## Calibration

```bash
python calibration.py --results_dir results/binary/cnn_bigru_eca_tam
```

Reports Expected/Maximum Calibration Error and the Brier score at clip level and at subject
level (probabilities averaged per patient), and saves a reliability diagram (observed accuracy
vs. predicted confidence, binned) to `results/<task>/<run>/calibration/`. Uses the
`predictions_fold<k>.csv` files already written by `train.py`, so no retraining is needed.

## Citation

If you use this code, please cite:

```bibtex
@article{timpano2026cnnbigru,
  title   = {Efficient CNN--BiGRU with Dual Attention for Parkinson's and ALS Speech Classification:
             A Lightweight Alternative to Self-Supervised Models},
  author  = {Timpano, Giuseppe and Caligiuri, Maria Eugenia and Cannataro, Mario and Guzzi, Pietro Hiram
             and Veltri, Pierangelo and Vizza, Patrizia},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT (see `LICENSE`). The speech corpora are subject to the terms of their providers.

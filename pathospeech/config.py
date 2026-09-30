"""Experiment configuration shared by all entry-point scripts."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SEED = 101

# Labels of the combined PD + ALS dataset (folder name -> multiclass label).
CLASS_FOLDERS = {"Healthy": 0, "Disease_PD": 1, "Disease_ALS": 2}
CLASS_NAMES = {
    "binary": {0: "HS", 1: "Pathological"},
    "multiclass": {0: "HS", 1: "PD", 2: "ALS"},
}

# Short names accepted on the command line -> Hugging Face checkpoints.
PRETRAINED_MODELS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
}
MODEL_CHOICES = ["cnn-bigru", *PRETRAINED_MODELS]


@dataclass
class ExperimentConfig:
    task: str = "binary"                 # "binary" (HS vs pathological) or "multiclass" (HS/PD/ALS)
    model: str = "cnn-bigru"             # "cnn-bigru", "wav2vec2", "hubert" or "ast"
    eca: bool = False                    # Efficient Channel Attention after the CNN
    tam: bool = False                    # Temporal Attention Module after the BiGRU
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("results")
    n_folds: int = 5
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-5
    weighted_loss: bool = False
    patience: int = 30
    common_sr: int | None = None      # if set, band-limit/resample all audio to this rate (Hz) (all models)
    num_workers: int = 4
    max_duration: float = 10.0           # seconds; longer clips are truncated, shorter ones zero-padded
    n_mels: int = 128
    hidden_size: int = 128
    num_layers: int = 2

    @property
    def is_cnn_bigru(self) -> bool:
        return self.model == "cnn-bigru"

    @property
    def is_ast(self) -> bool:
        return self.model == "ast"

    @property
    def pretrained_name(self) -> str | None:
        return PRETRAINED_MODELS.get(self.model)

    @property
    def run_name(self) -> str:
        if not self.is_cnn_bigru:
            return self.model
        return "cnn_bigru" + ("_eca" if self.eca else "") + ("_tam" if self.tam else "")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.task / self.run_name

    @property
    def folds_dir(self) -> Path:
        return Path(self.data_dir) / f"folds_{self.task}"

    @property
    def dataset_csv(self) -> Path:
        return Path(self.data_dir) / "dataset.csv"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="cnn-bigru")
    parser.add_argument("--eca", action="store_true", help="enable Efficient Channel Attention (CNN-BiGRU only)")
    parser.add_argument("--tam", action="store_true", help="enable the Temporal Attention Module (CNN-BiGRU only)")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"),
                        help="folder produced by preprocess.py (contains dataset.csv)")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4, help="processes used for feature extraction")
    parser.add_argument("--common_sr", type=int, default=None,
                        help="resample all audio to this common rate in Hz, e.g. 8000 to equalise bandwidth "
                        "across the native 8/16/44.1kHz source corpora and remove sampling rate as a "
                        "dataset-identity cue. CNN-BiGRU: the log-Mel front-end then runs at this rate. "
                        "Pretrained baselines: audio is band-limited to this rate and then resampled to the "
                        "16kHz their feature extractor expects. Omit to keep each recording at its native "
                        "sampling rate (default, unchanged behaviour).")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    fields = ExperimentConfig.__dataclass_fields__
    cfg = ExperimentConfig(**{k: v for k, v in vars(args).items() if k in fields})
    if not cfg.is_cnn_bigru and (cfg.eca or cfg.tam):
        raise ValueError("--eca/--tam only apply to the CNN-BiGRU model")
    return cfg


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

"""Subject-level k-fold splits and Hugging Face dataset construction."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import disable_caching, load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoFeatureExtractor

from pathospeech.config import SEED, ExperimentConfig
from pathospeech.features import log_mel_spectrogram

disable_caching()


def load_index(cfg: ExperimentConfig) -> pd.DataFrame:
    """Read dataset.csv, resolve file paths and apply the label scheme of the task."""
    df = pd.read_csv(cfg.dataset_csv)
    root = Path(cfg.dataset_csv).parent
    df["filename"] = [str(p) if Path(p).is_absolute() else str(root / p) for p in df["filename"]]
    if cfg.task == "binary":
        df["label"] = (df["label"] != 0).astype(int)  # PD and ALS -> pathological
    return df


def make_subject_folds(cfg: ExperimentConfig) -> list[Path]:
    """Split subjects into stratified folds and write one tab-separated CSV per fold.

    All recordings of a subject fall in the same fold, so no speaker appears in both the
    training and the validation set of any split. The split is deterministic (fixed SEED),
    so if all fold files already exist they are reused as-is: several train.py processes
    for the same task (e.g. the 4 CNN-BiGRU ablation variants) can be launched concurrently
    and share one fold split instead of racing to rewrite it.
    """
    cfg.folds_dir.mkdir(parents=True, exist_ok=True)
    paths = [cfg.folds_dir / f"fold{i}.csv" for i in range(cfg.n_folds)]
    if all(p.exists() for p in paths):
        return paths

    data = load_index(cfg)
    subjects = data.groupby("patient_id").first().reset_index()
    kfold = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=SEED)

    splits = kfold.split(subjects, subjects["label"], groups=subjects["patient_id"])
    for fold_idx, (_, subject_idx) in enumerate(splits):
        fold_subjects = subjects.iloc[subject_idx]["patient_id"]
        fold_df = data[data["patient_id"].isin(fold_subjects)].reset_index(drop=True)
        path = paths[fold_idx]
        tmp_path = path.with_suffix(f".tmp{os.getpid()}.csv")
        fold_df.to_csv(tmp_path, sep="\t", encoding="utf-8", index=False)
        os.replace(tmp_path, path)  # atomic: a concurrent reader never sees a partial file
    return paths


class FeatureExtractor:
    """Turns one dataset row into model inputs (log-Mel for CNN-BiGRU, HF processors otherwise).

    Log-Mel features for the CNN-BiGRU model are cached to disk, keyed by (file path, common_sr,
    max_duration, n_mels): the same recording produces the same features regardless of which
    fold or which attention ablation (eca/tam) it is used in, so this avoids recomputing the
    spectrogram from scratch for every one of the 5 folds x 4 ablation variants. This is the
    *only* on-disk cache: each unique recording is stored exactly once (~500KB/file), unlike a
    per-fold dataset cache, which would duplicate the same recordings many times over since
    folds' train splits overlap heavily.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.hf_extractor = None if cfg.is_cnn_bigru else AutoFeatureExtractor.from_pretrained(cfg.pretrained_name)
        if cfg.is_cnn_bigru:
            self.cache_dir = Path(cfg.data_dir) / ".feature_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _cache_path(self, filename: str) -> Path:
        key = f"{filename}|sr={self.cfg.common_sr}|dur={self.cfg.max_duration}|nmels={self.cfg.n_mels}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def __call__(self, row: dict) -> dict:
        if self.cfg.is_cnn_bigru:
            cache_path = self._cache_path(row["filename"])
            if cache_path.exists():
                features_np = np.load(cache_path)
            else:
                waveform, sampling_rate = torchaudio.load(row["filename"])
                if self.cfg.common_sr is not None and sampling_rate != self.cfg.common_sr:
                    waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.cfg.common_sr)(waveform)
                    sampling_rate = self.cfg.common_sr
                features_np = log_mel_spectrogram(waveform.squeeze(), sampling_rate, self.cfg.max_duration,
                                                  self.cfg.n_mels).numpy()
                tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}.npy")
                np.save(tmp_path, features_np)
                os.replace(tmp_path, cache_path)  # atomic, safe if two processes race on the same file
            inputs = {"input_values": features_np}
            sampling_rate = torchaudio.info(row["filename"]).sample_rate  # header only, cheap, for the column below
        else:
            waveform, sampling_rate = torchaudio.load(row["filename"])
            target_sr = self.hf_extractor.sampling_rate
            if self.cfg.common_sr is not None and sampling_rate != self.cfg.common_sr:
                # band-limit to common_sr first (e.g. 8kHz, as for the multiclass CNN-BiGRU), then
                # resample to the rate the pretrained model expects: every corpus then has the same
                # effective bandwidth even though the model input is at target_sr.
                waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.cfg.common_sr)(waveform)
                sampling_rate = self.cfg.common_sr
            if sampling_rate != target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)(waveform)
            waveform = waveform.squeeze().numpy()
            if self.cfg.is_ast:
                # AST computes its own log-Mel filterbank (fixed 1024-frame window)
                inputs = self.hf_extractor(waveform, sampling_rate=target_sr, return_tensors="np")
            else:
                # Wav2Vec2 / HuBERT: raw waveform, padded/truncated to max_duration
                inputs = self.hf_extractor(
                    waveform,
                    sampling_rate=target_sr,
                    return_attention_mask=True,
                    max_length=int(target_sr * self.cfg.max_duration),
                    truncation=True,
                    padding="max_length",
                    return_tensors="np",
                )
            inputs = dict(inputs)
            inputs["input_values"] = inputs["input_values"].squeeze()
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].squeeze()

        inputs["labels"] = row["label"]
        inputs["patient_id"] = row["patient_id"]
        inputs["sampling_rate"] = sampling_rate
        return inputs


def load_fold_datasets(cfg: ExperimentConfig) -> list[dict]:
    """Return, for each fold k, {"train": folds != k, "validation": fold k} as feature datasets.

    Only per-file features are cached to disk (see FeatureExtractor._cache_path); the fold-level
    HF Datasets built from them are always kept in memory only (keep_in_memory=True) and never
    written to disk. An earlier version of this function also persisted the mapped fold/split
    datasets under data/.mapped_cache/, but since a fold's train split overlaps ~80% with every
    other fold's train split, that duplicated the same recordings' features many times over on
    disk and was what exceeded the per-user disk quota - even running this function once,
    sequentially, for a single task. Once the per-file cache above is warm, rebuilding a fold's
    dataset from it is cheap (just numpy loads, no audio decoding or STFT), so nothing is lost
    by not caching at this level too.
    """
    fold_paths = make_subject_folds(cfg)
    extractor = FeatureExtractor(cfg)

    per_fold = []
    for k, valid_path in enumerate(fold_paths):
        fold_data = {}
        for split in ("train", "validation"):
            files = (
                [str(p) for j, p in enumerate(fold_paths) if j != k] if split == "train"
                else [str(valid_path)]
            )
            raw = load_dataset("csv", data_files={split: files}, delimiter="\t", cache_dir=None,
                                keep_in_memory=True)[split]
            fold_data[split] = raw.map(
                extractor,
                remove_columns=[c for c in raw.column_names if c not in ("patient_id", "label")],
                num_proc=cfg.num_workers,
                keep_in_memory=True,
            )
        per_fold.append(fold_data)
    return per_fold


def stack_inputs(dataset) -> torch.Tensor:
    """Stack the log-Mel inputs of a feature dataset into a (N, 1, frames, n_mels) tensor."""
    return torch.stack([torch.tensor(x, dtype=torch.float32) for x in dataset["input_values"]]).unsqueeze(1)

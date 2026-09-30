#!/usr/bin/env python3
"""Warm up the on-disk per-file feature cache for one task, sequentially, once.

FeatureExtractor (pathospeech/data.py) caches each recording's log-Mel spectrogram under
data/.feature_cache/<hash>.npy, keyed on (file path, common_sr, max_duration, n_mels) -
not on --eca/--tam, since those only change the model, not the features. Running this
script once, before launching several train.py processes for the same task in parallel
(e.g. the 4 CNN-BiGRU ablation variants), computes every recording's features in a single
sequential pass. When the parallel processes then start, they mostly just np.load() the
cached .npy files instead of racing each other to decode audio and compute STFTs, which
avoids both redundant CPU work and concurrent processes hammering the disk at once. Note
that fold-level datasets themselves are never written to disk (see load_fold_datasets),
only these per-file features are - that is what keeps this well under the disk quota.

Usage:
    python scripts/prepare_features.py --task binary
    python scripts/prepare_features.py --task multiclass --common_sr 8000
"""
import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/prepare_features.py`: add the repo root (this file's
# parent directory) to sys.path, since Python otherwise only puts scripts/ itself on it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathospeech.config import add_common_args, config_from_args
from pathospeech.data import load_fold_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)

    print(f"Preparing per-file feature cache: task={cfg.task}, model={cfg.model}, "
          f"common_sr={cfg.common_sr}, n_folds={cfg.n_folds} ...")
    t0 = time.time()
    per_fold = load_fold_datasets(cfg)
    elapsed = time.time() - t0
    n_train = sum(len(f["train"]) for f in per_fold)
    n_val = sum(len(f["validation"]) for f in per_fold)
    print(f"Done in {elapsed / 60:.1f} min: {len(per_fold)} folds built "
          f"({n_train} train rows + {n_val} validation rows total across folds, "
          f"features now cached per-file under data/.feature_cache/).")

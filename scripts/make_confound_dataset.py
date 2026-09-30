"""Build the dataset-origin confound-control set: healthy subjects only, relabelled by
source corpus (0 = Italian Parkinson's Voice and Speech, 1 = VOC-ALS) instead of disease.

Requires data/processed/dataset.csv to already exist (run preprocess.py first): reuses the
already VAD+Wiener-cleaned recordings, no re-processing.

Usage:
    python scripts/make_confound_dataset.py --data_dir data/processed --out_dir data/processed_confound
"""

import argparse
import re
from pathlib import Path

import pandas as pd

VOC_ALS_PATIENT_ID = re.compile(r"^(PZ|CT)\d+", re.IGNORECASE)  # VOC-ALS patient/control id convention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"),
                        help="folder written by preprocess.py (contains dataset.csv)")
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed_confound"))
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "dataset.csv")
    hs = df[df["label"] == 0].copy()

    # Resolve to absolute paths now, since processed_confound/dataset.csv lives in a
    # different folder and load_index() would otherwise resolve relative paths against it.
    root = args.data_dir.resolve()
    hs["filename"] = [str((root / p).resolve()) for p in hs["filename"]]

    # VOC-ALS controls use CT-prefixed ids; PZ prefix marks VOC-ALS patients (not present
    # among healthy subjects, kept in the regex for clarity/robustness only).
    hs["label"] = hs["patient_id"].apply(lambda pid: 1 if VOC_ALS_PATIENT_ID.match(str(pid)) else 0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hs[["filename", "label", "patient_id"]].to_csv(args.out_dir / "dataset.csv", index=False)

    counts = hs.groupby("label")["patient_id"].nunique()
    print(f"{len(hs)} recordings from {hs['patient_id'].nunique()} healthy subjects "
          f"(origin=Parkinson-corpus: {counts.get(0, 0)} subjects, origin=VOC-ALS: {counts.get(1, 0)} subjects)")
    print(f"Index written to {args.out_dir / 'dataset.csv'}")

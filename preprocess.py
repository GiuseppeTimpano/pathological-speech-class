"""Clean the raw recordings (VAD + Wiener filter) and build data/processed/dataset.csv.

Usage:
    python preprocess.py --raw_dir data/raw --out_dir data/processed
"""

import argparse
from pathlib import Path

from pathospeech.preprocessing import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"),
                        help="folder with Healthy/, Disease_PD/ and Disease_ALS/ sub-folders (one folder per subject)")
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    df = build_dataset(args.raw_dir, args.out_dir)
    counts = df.groupby("label")["patient_id"].nunique()
    print(f"{len(df)} recordings from {df['patient_id'].nunique()} subjects "
          f"(HS={counts.get(0, 0)}, PD={counts.get(1, 0)}, ALS={counts.get(2, 0)})")
    print(f"Index written to {args.out_dir / 'dataset.csv'}")

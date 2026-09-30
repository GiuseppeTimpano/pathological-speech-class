"""Build the XAI metrics table from explain.py's per-fold, per-class xai_metrics.csv.

Reads results/<task>/<run>/explanations/fold<f>/<class>/xai_metrics.csv for f in 0..n_folds-1
(one row per Quantus metric: FaithfulnessCorrelation, FaithfulnessEstimate, Monotonicity,
Sparseness, Complexity - written by explain.py) and aggregates each metric across folds as
mean +/- std (columns FC/FE/FM/SPS/COMP).

Usage:
    python scripts/make_xai_table.py
    python scripts/make_xai_table.py --run_dir results/multiclass/cnn_bigru_eca_tam --n_folds 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathospeech.config import CLASS_NAMES  # noqa: E402

# Quantus metric name (as written by explain.py) -> table column.
METRIC_TO_COLUMN = {
    "FaithfulnessCorrelation": "FC",
    "FaithfulnessEstimate": "FE",
    "Monotonicity": "FM",
    "Sparseness": "SPS",
    "Complexity": "COMP",
}
COLUMNS = ["FC", "FE", "FM", "SPS", "COMP"]


def fmt(value: float) -> str:
    sign = r"$-$" if value < 0 else ""
    return f"{sign}{abs(value):.3f}"


def load_class_scores(run_dir: Path, class_name: str, n_folds: int) -> dict[str, list[float]]:
    scores = {col: [] for col in COLUMNS}
    for fold in range(n_folds):
        csv_path = run_dir / "explanations" / f"fold{fold}" / class_name / "xai_metrics.csv"
        if not csv_path.exists():
            print(f"  missing: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path).set_index("Metric")["Score"]
        for metric_name, column in METRIC_TO_COLUMN.items():
            if metric_name in df.index:
                scores[column].append(df[metric_name])
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, default=Path("results/multiclass/cnn_bigru_eca_tam"))
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    class_names = list(CLASS_NAMES["multiclass"].values())  # HS, PD, ALS, in label order

    rows = []
    for class_name in class_names:
        scores = load_class_scores(args.run_dir, class_name, args.n_folds)
        n_found = {col: len(vals) for col, vals in scores.items()}
        if any(n < args.n_folds for n in n_found.values()):
            print(f"WARNING: {class_name} has incomplete folds per metric: {n_found}", file=sys.stderr)
        cells = [fmt(np.mean(scores[col])) + r" $\pm$ " + f"{np.std(scores[col]):.3f}" for col in COLUMNS]
        rows.append((class_name, cells))

    print(r"% XAI metrics, mean $\pm$ std over folds, one row per class.")
    for class_name, cells in rows:
        print(f"{class_name}  & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()

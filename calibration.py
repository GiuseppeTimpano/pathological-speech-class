"""Calibration analysis (ECE, Brier score, reliability diagram) from saved fold predictions.

Calibration asks a different question than accuracy: when the model says "80% probability
of ALS", is it right about 80% of the time on such cases? A model can be accurate yet
overconfident (probabilities pushed toward 0/1 regardless of correctness), which matters for
any clinical use of the output probabilities, not just the predicted label.

Reads the ``predictions_fold<k>.csv`` files that ``train.py`` already writes to a run
directory (clip-level probabilities, with a patient_id column), and reports, at both clip
level and subject level (probabilities averaged per patient, as in the
subject-level evaluation of train.py):

  * Expected Calibration Error (ECE) -- mean gap between confidence and accuracy across bins.
  * Maximum Calibration Error (MCE)  -- worst-case gap across bins.
  * Brier score -- mean squared error between predicted probabilities and one-hot labels.
  * A reliability diagram (confidence vs. observed accuracy per bin, with a bin-count panel).

Usage:
    python calibration.py --results_dir results/binary/cnn_bigru_eca_tam
    python calibration.py --results_dir results/multiclass/hubert --n_bins 15 --out calib_hubert

Each run directory is one model/task combination (i.e. what train.py wrote for a single
``--task``/``--model``/attention configuration), matching the layout in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_clip_predictions(results_dir: Path) -> pd.DataFrame:
    """Concatenate every predictions_fold<k>.csv found in results_dir, tagging the fold."""
    files = sorted(results_dir.glob("predictions_fold*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No predictions_fold*.csv found in {results_dir}. Run train.py for this "
            f"task/model first (it writes one such file per fold).")
    frames = []
    for f in files:
        fold = int("".join(ch for ch in f.stem if ch.isdigit()))
        df = pd.read_csv(f)
        df["fold"] = fold
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_subject_level(clip_df: pd.DataFrame, prob_cols: list[str]) -> pd.DataFrame:
    """Average clip probabilities per (fold, patient_id), as in the subject-level score of train.py."""
    subjects = clip_df.groupby(["fold", "patient_id"])[prob_cols].mean().reset_index()
    subjects["true_label"] = clip_df.groupby(["fold", "patient_id"])["true_label"].first().to_numpy()
    return subjects


def confidence_and_correctness(df: pd.DataFrame, prob_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Top-class confidence and whether the top class matches the true label, per row."""
    probs = df[prob_cols].to_numpy()
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (preds == df["true_label"].to_numpy()).astype(float)
    return confidence, correct


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    """Equal-width binning over [0, 1]. Returns (ECE, MCE, per-bin table)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    ece, mce = 0.0, 0.0
    n = len(confidence)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_lower": bin_edges[b], "bin_upper": bin_edges[b + 1],
                        "count": 0, "avg_confidence": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        avg_conf = confidence[mask].mean()
        acc = correct[mask].mean()
        gap = abs(avg_conf - acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append({"bin_lower": bin_edges[b], "bin_upper": bin_edges[b + 1],
                    "count": count, "avg_confidence": avg_conf, "accuracy": acc, "gap": gap})
    return ece, mce, pd.DataFrame(rows)


def brier_score(df: pd.DataFrame, prob_cols: list[str]) -> float:
    """Multiclass Brier score: mean squared error between predicted probabilities and one-hot labels.

    Reduces to the standard binary Brier score (using prob_1) when there are two classes.
    """
    probs = df[prob_cols].to_numpy()
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(df)), df["true_label"].to_numpy()] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def plot_reliability_diagram(bin_table: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, (ax_rel, ax_count) = plt.subplots(
        2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    centers = (bin_table["bin_lower"] + bin_table["bin_upper"]) / 2
    ax_rel.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax_rel.bar(centers, bin_table["accuracy"], width=1 / len(bin_table), edgecolor="black",
              color="tab:blue", alpha=0.7, label="Observed accuracy")
    ax_rel.plot(centers, bin_table["avg_confidence"], marker="o", color="tab:red",
               label="Avg. confidence")
    ax_rel.set_ylabel("Accuracy")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_title(title)
    ax_rel.legend(loc="upper left", fontsize=8)

    ax_count.bar(centers, bin_table["count"], width=1 / len(bin_table), color="tab:gray", edgecolor="black")
    ax_count.set_xlabel("Confidence")
    ax_count.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def analyze(df: pd.DataFrame, prob_cols: list[str], level: str, n_bins: int, out_dir: Path,
           run_name: str) -> dict:
    confidence, correct = confidence_and_correctness(df, prob_cols)
    ece, mce, bin_table = expected_calibration_error(confidence, correct, n_bins=n_bins)
    brier = brier_score(df, prob_cols)

    bin_table.to_csv(out_dir / f"calibration_bins_{level}.csv", index=False)
    plot_reliability_diagram(
        bin_table, title=f"{run_name} -- {level}-level reliability diagram",
        out_path=out_dir / f"reliability_diagram_{level}.png")

    return {"level": level, "n_samples": len(df), "ece": ece, "mce": mce, "brier_score": brier}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True, type=Path,
                        help="run directory written by train.py, e.g. results/binary/cnn_bigru_eca_tam")
    parser.add_argument("--n_bins", type=int, default=10, help="number of equal-width confidence bins")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory for the CSVs/plots (default: <results_dir>/calibration)")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out or (results_dir / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = results_dir.name

    clip_df = load_clip_predictions(results_dir)
    prob_cols = sorted([c for c in clip_df.columns if c.startswith("prob_")],
                       key=lambda c: int(c.split("_")[1]))
    if not prob_cols:
        raise ValueError(f"No prob_* columns found in predictions files under {results_dir}")
    subject_df = to_subject_level(clip_df, prob_cols)

    summary = [
        analyze(clip_df, prob_cols, "clip", args.n_bins, out_dir, run_name),
        analyze(subject_df, prob_cols, "subject", args.n_bins, out_dir, run_name),
    ]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "calibration_summary.csv", index=False)

    print(f"\nCalibration -- {run_name}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved bin tables, reliability diagrams and summary to {out_dir}")


if __name__ == "__main__":
    main()

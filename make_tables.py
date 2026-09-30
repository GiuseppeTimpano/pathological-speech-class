"""Build LaTeX result tables from results/<task>/<run>/kfold_summary.csv.

Usage:
    python make_tables.py --results_dir results > tables.tex
"""

import argparse
from pathlib import Path

import pandas as pd

BASELINES = [("wav2vec2", "Wav2Vec2"), ("hubert", "HuBERT"), ("ast", "AST"),
             ("cnn_bigru_eca_tam", "CNN-BiGRU (proposed)")]
ABLATION = [("cnn_bigru", "Base"), ("cnn_bigru_eca", "+ ECA"), ("cnn_bigru_tam", "+ TAM"),
            ("cnn_bigru_eca_tam", "+ ECA + TAM")]
CLIP_ROWS = [("eval_accuracy", "Acc (clip)"), ("eval_f1", "F1 (clip)"), ("eval_recall", "Recall (clip)"),
             ("eval_precision", "Precision (clip)"), ("eval_auc", "AUC (clip)")]
PATIENT_ROWS = [("acc_patient", "Acc (patient)"), ("f1_patient", "F1 (patient)"), ("auc_patient", "AUC (patient)")]
TASK_TITLES = {"binary": "Binary classification (HS vs. pathological)",
               "multiclass": "Multiclass classification (HS vs. PD vs. ALS)"}


def load(results_dir: Path, task: str, run: str):
    path = results_dir / task / run / "kfold_summary.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


def cell(summary, metric: str, bold: bool = False) -> str:
    if summary is None or metric not in summary:
        return "--"
    mean = f"{summary.loc['mean', metric]:.3f}"
    if bold:
        mean = r"\textbf{" + mean + "}"
    return mean + r" $\pm$ " + f"{summary.loc['std', metric]:.3f}"


def metric_table(results_dir, task, runs, caption, label, bold_best=False) -> str:
    summaries = [load(results_dir, task, run) for run, _ in runs]
    lines = [
        r"\begin{table}[h!]", r"\renewcommand{\arraystretch}{1.3}", r"\centering",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(runs) + "}", r"\hline",
        r"\textbf{Metric} & " + " & ".join(rf"\textbf{{{name}}}" for _, name in runs) + r" \\", r"\hline",
    ]
    for block in (CLIP_ROWS, PATIENT_ROWS):
        for metric, name in block:
            means = [s.loc["mean", metric] if s is not None and metric in s else float("-inf") for s in summaries]
            best = max(means)
            cells = [cell(s, metric, bold_best and m == best and m != float("-inf")) for s, m in zip(summaries, means)]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(lines)


def efficiency_table(results_dir, task) -> str:
    rows = [("eval_runtime", "Eval runtime (s)", 2), ("eval_samples_per_second", "Samples/s", 2),
            ("eval_steps_per_second", "Steps/s", 3)]
    summaries = [load(results_dir, task, run) for run, _ in BASELINES]
    lines = [r"\begin{table}[h!]", r"\centering", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lcccc}", r"\hline",
             "Metric & " + " & ".join(name for _, name in BASELINES) + r" \\", r"\hline"]
    for metric, name, digits in rows:
        cells = ["--" if s is None else f"{s.loc['mean', metric]:.{digits}f} $\\pm$ {s.loc['std', metric]:.{digits}f}"
                 for s in summaries]
        lines.append(f"\\textbf{{{name}}} & " + " & ".join(cells) + r" \\")
    params = ["--" if s is None else f"{s.loc['mean', 'total_params'] / 1e6:.1f}M" for s in summaries]
    lines += [r"\textbf{Total params} & " + " & ".join(params) + r" \\", r"\hline", r"\end{tabular}}",
              rf"\caption{{Inference efficiency ({task} models), mean $\pm$ std over folds.}}",
              rf"\label{{tab:efficiency_{task}}}", r"\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    results_dir = parser.parse_args().results_dir

    for task in ("binary", "multiclass"):
        print(metric_table(results_dir, task, BASELINES,
                           f"{TASK_TITLES[task]}: clip- and patient-level metrics, mean $\\pm$ std over 5 folds.",
                           f"tab:{task}_main", bold_best=True))
    for task in ("binary", "multiclass"):
        print(metric_table(results_dir, task, ABLATION,
                           f"Ablation study ({TASK_TITLES[task]}): mean $\\pm$ std over 5 folds.",
                           f"tab:{task}_ablation"))
    print(efficiency_table(results_dir, "binary"))

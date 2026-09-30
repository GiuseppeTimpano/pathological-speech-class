"""Re-measure inference speed (eval_runtime / samples_per_second / steps_per_second) of an
already-trained k-fold run, WITHOUT retraining. Loads the best_model_fold<k>/ checkpoints
saved by train.py and re-runs trainer.evaluate() on each fold's validation split.

Use this to get timing numbers for different models under comparable GPU conditions
(e.g. re-time the CNN-BiGRU on a free GPU, right after the baselines finished on it),
instead of relying on numbers measured with the GPU shared with other jobs.

Does NOT touch kfold_results.csv / kfold_summary.csv, and does NOT write anything under the
training run's own directory (results/<task>/<run_name>/): everything this script produces
goes into a separate --speedtest_dir tree (default: results_inference_speed/<task>/<run_name>/),
so a speed-test run can never overwrite or get mixed up with training output.

Example:
    python scripts/measure_inference_speed.py --task binary --model cnn-bigru --eca --tam
    python scripts/measure_inference_speed.py --task multiclass --model wav2vec2 --common_sr 8000
    python scripts/measure_inference_speed.py --task binary --model ast --speedtest_dir results_inference_speed_run2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import DefaultDataCollator, Trainer, TrainingArguments

# scripts/ is not the project root: add the parent directory (which contains the
# pathospeech package) to sys.path, since Python otherwise only puts scripts/ itself on it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathospeech.config import add_common_args, config_from_args
from pathospeech.data import load_fold_datasets
from pathospeech.models import load_trained_model
from pathospeech.training import compute_clip_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1,
                         help="re-run trainer.evaluate() this many times per fold and average, "
                              "to smooth out first-batch CUDA warm-up / caching noise.")
    parser.add_argument("--speedtest_dir", type=Path, default=Path("results_inference_speed"),
                         help="root directory for this script's own output (kept separate from "
                              "--output_dir, which is only read from to locate the checkpoints).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = config_from_args(args)

    # cfg.run_dir (under --output_dir, e.g. results/binary/wav2vec2) is only used to find the
    # already-trained checkpoints. Everything this script writes goes under speedtest_dir instead,
    # in the same task/run_name layout, so it never touches the training run's own files.
    speed_dir = args.speedtest_dir / cfg.task / cfg.run_name
    speed_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_idx, dataset in enumerate(load_fold_datasets(cfg)):
        model_dir = cfg.run_dir / f"best_model_fold{fold_idx}"
        if not model_dir.exists():
            print(f"skipping fold {fold_idx}: no checkpoint at {model_dir}")
            continue

        num_labels = len(set(dataset["validation"]["labels"]))
        model = load_trained_model(cfg, model_dir, num_labels)

        eval_args = TrainingArguments(
            output_dir=speed_dir / f"fold{fold_idx}_speedtest",
            per_device_eval_batch_size=args.eval_batch_size,
            fp16=not cfg.is_cnn_bigru,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=dataset["validation"],
            data_collator=DefaultDataCollator(return_tensors="pt"),
            compute_metrics=compute_clip_metrics,
        )

        for rep in range(args.repeats):
            results = trainer.evaluate(dataset["validation"])
            results["fold"] = fold_idx
            results["repeat"] = rep
            results["total_params"] = sum(p.numel() for p in model.parameters())
            rows.append(results)
            print(f"fold {fold_idx} rep {rep}: eval_runtime={results['eval_runtime']:.3f}s  "
                  f"samples/s={results['eval_samples_per_second']:.3f}  "
                  f"steps/s={results['eval_steps_per_second']:.3f}")

        del trainer, model

    if not rows:
        print("No checkpoints found - nothing measured.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(speed_dir / "inference_speed.csv", index=False)
    df.describe().loc[["mean", "std"]].to_csv(speed_dir / "inference_speed_summary.csv")
    print(f"\nSaved to {speed_dir / 'inference_speed.csv'} and inference_speed_summary.csv")


if __name__ == "__main__":
    main()

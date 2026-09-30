"""Integrated Gradients explanations for a trained CNN-BiGRU and their quantitative evaluation.

Explains the validation recordings of one fold that belong to --target_label and are
correctly classified, saves one |IG| map per recording and the Quantus metrics
(Faithfulness Correlation/Estimate, Monotonicity, Sparseness, Complexity).

Example (HS, PD and ALS in turn):
    for c in 0 1 2; do
        python explain.py --task multiclass --eca --tam --fold 0 --target_label $c
    done
"""

import argparse

import torch

from pathospeech.config import CLASS_NAMES, add_common_args, config_from_args, set_seed
from pathospeech.data import load_fold_datasets, stack_inputs
from pathospeech.explainability import (LogitsWrapper, evaluate_attributions, integrated_gradients,
                                        save_attribution_maps)
from pathospeech.models import load_trained_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--fold", type=int, default=0, help="validation fold to explain")
    parser.add_argument("--target_label", type=int, default=0, help="class to explain (0 = HS)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--ig_eval_mode", action="store_true",
                        help="keep the model in eval mode during IG (cuDNN disabled); gives deterministic attributions")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = config_from_args(args)
    if not cfg.is_cnn_bigru:
        raise ValueError("explain.py supports the CNN-BiGRU model only")
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    valid = load_fold_datasets(cfg)[args.fold]["validation"]
    num_labels = len(set(valid["labels"]))
    model_dir = cfg.run_dir / f"best_model_fold{args.fold}"

    def fresh_model():
        return LogitsWrapper(load_trained_model(cfg, model_dir, num_labels)).to(device).eval()

    # 1. keep correctly classified recordings of the target class
    inputs = stack_inputs(valid)
    labels = torch.tensor(valid["labels"])
    model = fresh_model()
    with torch.no_grad():
        preds = torch.cat([model(batch).argmax(-1).cpu() for batch in inputs.split(args.batch_size)])
    mask = (preds == labels) & (labels == args.target_label)
    x, y = inputs[mask], labels[mask]
    class_name = CLASS_NAMES[cfg.task][args.target_label]
    print(f"{len(x)} correctly classified '{class_name}' recordings to explain")
    if len(x) == 0:
        return

    # 2. Integrated Gradients
    out_dir = cfg.run_dir / "explanations" / f"fold{args.fold}" / class_name
    attributions = []
    for start in range(0, len(x), args.batch_size):
        batch_attr = integrated_gradients(model, x[start:start + args.batch_size],
                                          y[start:start + args.batch_size], eval_mode=args.ig_eval_mode)
        save_attribution_maps(batch_attr, out_dir, start_index=start)
        attributions.append(batch_attr.cpu())
    attributions = torch.cat(attributions)

    # 3. Quantus metrics (fresh model: IG in train mode alters BatchNorm statistics)
    scores = evaluate_attributions(fresh_model(), x.numpy(), y.numpy(), attributions.numpy())
    scores.to_csv(out_dir / "xai_metrics.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()

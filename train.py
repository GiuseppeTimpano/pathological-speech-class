"""Subject-level k-fold training and evaluation.

For each fold the best checkpoint (validation clip-level weighted F1) is evaluated at clip
and subject level; subject predictions average the clip probabilities of each speaker.
Only the best-F1 checkpoint of each fold is kept (as best_model_fold<k>/); the
per-epoch checkpoints used to select it are deleted right after each fold finishes.

Examples:
    # proposed model (CNN-BiGRU + ECA + TAM)
    python train.py --task binary --model cnn-bigru --eca --tam --weighted_loss
    # baselines
    python train.py --task multiclass --model hubert --weighted_loss

Outputs in results/<task>/<run>/: best_model_fold<k>/, predictions_fold<k>.csv,
kfold_results.csv (one row per fold), kfold_summary.csv (mean and std), and fold<k>/logs/.
"""

import argparse
import shutil
import time

import pandas as pd
from transformers import DefaultDataCollator, EarlyStoppingCallback, Trainer, TrainingArguments

from pathospeech.config import add_common_args, config_from_args, set_seed
from pathospeech.data import load_fold_datasets
from pathospeech.models import build_model
from pathospeech.training import (WeightedLossTrainer, compute_class_weights, compute_clip_metrics,
                                  subject_level_evaluation)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    parser.add_argument("--weighted_loss", action="store_true", help="inverse-frequency weighted cross-entropy")
    return parser.parse_args()


def main():
    cfg = config_from_args(parse_args())
    set_seed()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fold_idx, dataset in enumerate(load_fold_datasets(cfg)):
        print(f"\n=== {cfg.task} | {cfg.run_name} | fold {fold_idx} ===")
        labels = sorted(set(dataset["train"]["labels"]))
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}

        model = build_model(cfg, label2id, id2label)
        fold_dir = cfg.run_dir / f"fold{fold_idx}"
        args = TrainingArguments(
            output_dir=fold_dir,
            logging_dir=fold_dir / "logs",
            report_to="tensorboard",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            warmup_ratio=0.1,
            gradient_checkpointing=not cfg.is_cnn_bigru,  # large pretrained models only
            fp16=not cfg.is_cnn_bigru,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
        )
        trainer_kwargs = dict(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=DefaultDataCollator(return_tensors="pt"),
            compute_metrics=compute_clip_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience, early_stopping_threshold=0.01)],
        )
        if cfg.weighted_loss:
            trainer = WeightedLossTrainer(class_weights=compute_class_weights(dataset["train"]), **trainer_kwargs)
        else:
            trainer = Trainer(**trainer_kwargs)

        start = time.time()
        trainer.train()
        training_time = time.time() - start

        # clip level (includes eval_runtime / samples_per_second used for the efficiency table)
        results = trainer.evaluate(dataset["validation"])
        results["fold"] = fold_idx

        # subject level
        pred = trainer.predict(dataset["validation"])
        subject_metrics, cm, clip_predictions = subject_level_evaluation(
            pred.predictions, pred.label_ids, dataset["validation"]["patient_id"])
        clip_predictions.to_csv(cfg.run_dir / f"predictions_fold{fold_idx}.csv", index=False)
        print(f"Subject level: {subject_metrics}\nConfusion matrix:\n{cm}")

        results.update({
            "acc_patient": subject_metrics["acc"],
            "f1_patient": subject_metrics["f1"],
            "auc_patient": subject_metrics["auc"],
            "training_time": training_time,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        all_results.append(results)

        # trainer.model is already the best-F1 checkpoint here (load_best_model_at_end=True);
        # save only that one, then drop the per-epoch checkpoints used to pick it - they take
        # far more disk than the final model and are never needed again.
        trainer.save_model(cfg.run_dir / f"best_model_fold{fold_idx}")
        del trainer, model
        for ckpt in fold_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt, ignore_errors=True)

    results = pd.DataFrame(all_results)
    results.to_csv(cfg.run_dir / "kfold_results.csv", index=False)
    results.describe().loc[["mean", "std"]].to_csv(cfg.run_dir / "kfold_summary.csv")
    print(f"\nResults saved to {cfg.run_dir}")


if __name__ == "__main__":
    main()

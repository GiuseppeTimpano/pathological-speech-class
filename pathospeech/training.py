"""Loss, metrics and subject-level aggregation used during k-fold training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from transformers import Trainer


def compute_class_weights(train_dataset) -> torch.Tensor:
    """Inverse-frequency class weights: n_samples / (n_classes * n_samples_in_class)."""
    counts = np.bincount(train_dataset["labels"])
    return torch.tensor([counts.sum() / (len(counts) * c) for c in counts], dtype=torch.float)


class WeightedLossTrainer(Trainer):
    """Trainer using class-weighted cross-entropy to compensate class imbalance."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        return (loss, outputs) if return_outputs else loss


def softmax(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()


def compute_clip_metrics(pred) -> dict:
    """Clip-level metrics (weighted averages over classes); "f1" drives model selection."""
    logits, labels = pred.predictions, pred.label_ids
    preds = logits.argmax(-1)
    probs = softmax(logits)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
    }
    try:
        if probs.shape[1] == 2:
            metrics["auc"] = roc_auc_score(labels, probs[:, 1])
        else:
            metrics["auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except ValueError as err:  # e.g. a single class in the evaluation set
        print(f"AUC not computed: {err}")
    return metrics


def subject_level_evaluation(logits: np.ndarray, labels: np.ndarray, patient_ids) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Average clip probabilities per subject and score the resulting subject-level predictions.

    Returns (metrics, confusion matrix, clip-level prediction table).
    """
    probs = softmax(logits)
    n_classes = probs.shape[1]
    clips = pd.DataFrame({"patient_id": patient_ids, "true_label": labels})
    for c in range(n_classes):
        clips[f"prob_{c}"] = probs[:, c]

    prob_cols = [f"prob_{c}" for c in range(n_classes)]
    subjects = clips.groupby("patient_id")[prob_cols].mean()
    subjects["pred_label"] = subjects[prob_cols].to_numpy().argmax(axis=1)
    subjects["true_label"] = clips.groupby("patient_id")["true_label"].first()

    y_true, y_pred = subjects["true_label"], subjects["pred_label"]
    binary = n_classes == 2
    try:
        if binary:
            auc = roc_auc_score(y_true, subjects["prob_1"])
        else:
            auc = roc_auc_score(y_true, subjects[prob_cols], multi_class="ovo")
    except ValueError:
        auc = np.nan

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary" if binary else "macro"),
        "auc": auc,
    }
    return metrics, confusion_matrix(y_true, y_pred), clips

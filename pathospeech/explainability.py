"""Integrated Gradients attributions on log-Mel inputs and their quantitative evaluation (Quantus)."""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import quantus  # noqa: E402
import torch  # noqa: E402
from captum.attr import IntegratedGradients  # noqa: E402


class LogitsWrapper(torch.nn.Module):
    """Expose a plain logits tensor and accept (B, 1, T, F) inputs, as Captum and Quantus expect."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(next(self.model.parameters()).device)
        if x.dim() == 4:
            x = x.squeeze(1)
        return self.model(x).logits


def integrated_gradients(model: LogitsWrapper, inputs: torch.Tensor, targets: torch.Tensor,
                         n_steps: int = 50, eval_mode: bool = False) -> torch.Tensor:
    """Integrated Gradients with an all-zero baseline (= the mean of the standardised log-Mel).

    cuDNN only back-propagates through a GRU in training mode. By default (as for the
    published results) the model is therefore switched to train mode on GPU, which also
    activates dropout and batch statistics in BatchNorm. With ``eval_mode=True`` the model
    stays in eval mode and cuDNN is disabled instead, giving deterministic attributions.
    Train mode updates the BatchNorm running statistics: use a fresh copy of the model for
    anything computed afterwards.
    """
    device = next(model.parameters()).device
    use_train_mode = device.type == "cuda" and not eval_mode
    model.train(use_train_mode)
    no_cudnn = torch.backends.cudnn.flags(enabled=False) if eval_mode else contextlib.nullcontext()
    with no_cudnn:
        attributions = IntegratedGradients(model).attribute(
            inputs.detach().clone().to(device).requires_grad_(True),
            baselines=torch.zeros_like(inputs).to(device),
            target=targets.to(device),
            n_steps=n_steps,
            internal_batch_size=1,
        )
    model.eval()
    return attributions.detach()


def save_attribution_maps(attributions: torch.Tensor, out_dir: Path, start_index: int = 0,
                          save_npy_for_index: int | None = 0) -> None:
    """Save |IG| maps, min-max normalised per sample, as time x frequency images.

    ``save_npy_for_index``: in addition to the PNG, also save the normalised (n_mels, frames)
    array as attribution_sample_<that index>.npy - the global sample index within this fold
    and class, not the index within the current batch. Used by make_ig_figure.py to rebuild
    the example figure from the exact normalised map shown in the per-sample PNGs, with a shared color
    scale across classes, without having to recompute Integrated Gradients. Pass None to skip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, attribution in enumerate(attributions):
        global_index = start_index + i
        a = np.abs(attribution.squeeze().cpu().numpy()).T  # (n_mels, frames)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        if save_npy_for_index is not None and global_index == save_npy_for_index:
            np.save(out_dir / f"attribution_sample_{global_index}.npy", a)
        plt.imshow(a, cmap="inferno", aspect="auto", origin="lower")
        plt.title("Normalized |Integrated Gradients|")
        plt.colorbar(label="Importance intensity")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bands")
        plt.savefig(out_dir / f"attribution_sample_{global_index}.png", bbox_inches="tight")
        plt.close()


def evaluate_attributions(model: LogitsWrapper, inputs: np.ndarray, targets: np.ndarray,
                          attributions: np.ndarray) -> pd.DataFrame:
    """Faithfulness (Correlation, Estimate, Monotonicity) and complexity (Sparseness, Complexity) metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perturb = quantus.functions.perturb_func.batch_baseline_replacement_by_indices
    metrics = {
        "FaithfulnessCorrelation": quantus.FaithfulnessCorrelation(
            nr_runs=50, subset_size=5000, perturb_baseline="black", perturb_func=perturb,
            similarity_func=quantus.similarity_func.correlation_pearson, abs=True, return_aggregate=True),
        "FaithfulnessEstimate": quantus.FaithfulnessEstimate(
            features_in_step=224, perturb_baseline="black", perturb_func=perturb,
            similarity_func=quantus.similarity_func.correlation_pearson, abs=True, return_aggregate=True),
        "Monotonicity": quantus.MonotonicityCorrelation(
            features_in_step=224, perturb_baseline="black", perturb_func=perturb, abs=True, return_aggregate=True),
        "Sparseness": quantus.Sparseness(abs=True, return_aggregate=True),
        "Complexity": quantus.Complexity(abs=True, return_aggregate=True),
    }
    scores = {
        name: metric(model=model, x_batch=inputs, y_batch=targets, a_batch=attributions, device=device)
        for name, metric in metrics.items()
    }
    return pd.DataFrame({"Metric": list(scores), "Score": [np.ravel(s)[0] for s in scores.values()]})

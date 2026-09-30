"""Build the example Integrated Gradients figure (saliency maps for HS/PD/ALS) from the
normalised attribution arrays saved by explain.py / save_attribution_maps (attribution_sample_0.npy,
one representative recording per class), on the retrained multiclass CNN-BiGRU (ECA+TAM, 8kHz).

Requires scripts/07_xai_multiclass.sh to have been run first for the chosen fold (default: fold 0),
so that results/multiclass/cnn_bigru_eca_tam/explanations/fold<fold>/<HS|PD|ALS>/attribution_sample_0.npy
exists for all three classes.

Usage:
    python scripts/make_ig_figure.py
    python scripts/make_ig_figure.py --fold 2 --out results/multiclass/cnn_bigru_eca_tam/ig_examples.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathospeech.config import CLASS_NAMES  # noqa: E402

# (grid row, grid col, panel label) for HS, PD, ALS:
# a) HS top-left, b) PD top-right, c) ALS bottom-left, colorbar in the remaining bottom-right cell.
PANELS = {"HS": (0, 0, "a)"), "PD": (0, 1, "b)"), "ALS": (1, 0, "c)")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, default=Path("results/multiclass/cnn_bigru_eca_tam"))
    parser.add_argument("--fold", type=int, default=0, help="which fold's sample_0 attribution map to show")
    parser.add_argument("--out", type=Path, default=Path("results/multiclass/cnn_bigru_eca_tam/ig_examples.png"),
                        help="output image path")
    args = parser.parse_args()

    class_names = list(CLASS_NAMES["multiclass"].values())  # HS, PD, ALS
    maps = {}
    for class_name in class_names:
        npy_path = args.run_dir / "explanations" / f"fold{args.fold}" / class_name / "attribution_sample_0.npy"
        if not npy_path.exists():
            sys.exit(f"Missing {npy_path} - run scripts/07_xai_multiclass.sh first "
                     f"(it saves attribution_sample_0.npy for each fold/class).")
        maps[class_name] = np.load(npy_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Normalized |Integrated Gradients|", fontsize=16)

    for class_name, (row, col, label) in PANELS.items():
        ax = axes[row, col]
        im = ax.imshow(maps[class_name], cmap="inferno", aspect="auto", origin="lower", vmin=0, vmax=1)
        ax.set_xlabel("Time frames")
        if col == 0:
            ax.set_ylabel("Mel bands")
        ax.text(-0.12, -0.18, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Fourth cell (bottom-right): shared colorbar only, no axes/frame.
    cbar_ax = axes[1, 1]
    cbar_ax.axis("off")
    fig.colorbar(im, ax=cbar_ax, fraction=0.6, label="Importance intensity")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

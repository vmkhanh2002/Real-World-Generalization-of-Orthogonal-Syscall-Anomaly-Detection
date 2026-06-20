"""Generate score-distribution figure from the refreshed Phase-5 z-matrix."""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_config import DEFAULT_DATASET, normalize_dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    parser.add_argument("--z-path", type=Path, default=None, help="Optional refreshed z-matrix input path.")
    parser.add_argument("--out", type=Path, default=None, help="Main figure output path.")
    parser.add_argument("--paper-figures-dir", type=Path, default=None, help="Optional paper figures directory.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, str]:
    dataset = normalize_dataset_name(args.dataset)
    z_path = args.z_path or ROOT / "experiments" / dataset / "logs" / "z_matrix_refreshed.pkl"
    out_root = args.out or ROOT / "figures" / "score_distributions.png"
    out_paper = args.paper_figures_dir / "score_distributions.png" if args.paper_figures_dir else None
    return z_path, out_root, out_paper, dataset


def main() -> None:
    args = parse_args()
    z_path, out_root, out_paper, dataset = resolve_paths(args)

    if not z_path.exists():
        raise FileNotFoundError(
            f"Missing refreshed z-matrix: {z_path}. "
            f"Run `python scripts/train_phase5_fusion.py --dataset {dataset}` first."
        )

    with open(z_path, "rb") as handle:
        data = pickle.load(handle)

    z_all = np.vstack([data["z_train"], data["z_val"], data["z_test"]])
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]]).astype(int)

    path_configs = [
        ("Path A (SGD-OCSVM)", 0, "Scores"),
        ("Path B (CNN-1D AE)", 1, "Scores"),
        ("Path C (GRU Predictor)", 2, "Scores"),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    spans = []
    for ax, (title, col, xlabel) in zip(axes, path_configs):
        values = z_all[:, col]
        lo = float(values.min())
        hi = float(np.quantile(values, 0.99))
        spans.append(float(values.max() - values.min()))
        benign = values[y_all == 0]
        attack = values[y_all == 1]
        bins = np.linspace(lo, hi, 80)

        ax.hist(benign, bins=bins, alpha=0.62, color="#4878d0", label="Benign", density=True)
        ax.hist(attack, bins=bins, alpha=0.62, color="#d65f5f", label="Attack", density=True)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density (log scale)")
        ax.legend()
        ax.set_xlim(lo, hi)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    out_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_root, dpi=300, bbox_inches="tight")

    print(f"Saved {out_root}")
    if out_paper is not None:
        out_paper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_root, out_paper)
        print(f"Saved {out_paper}")
    print("Full spans: " + ", ".join(f"{span:.4f}" for span in spans))


if __name__ == "__main__":
    main()

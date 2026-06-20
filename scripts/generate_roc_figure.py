"""
Generate the ROC comparison figure from the refreshed Phase-5 ROC export.

This script intentionally does not read z_matrix_splits.pkl or
z_matrix_eval_full.pkl. Those artifacts can be stale relative to the refreshed
Phase-5 protocol. The ground-truth input is produced by:

 python scripts/train_phase5_fusion.py --dataset DATASET

which writes the refreshed ROC JSON under the selected dataset log directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_config import DEFAULT_DATASET, normalize_dataset_name

STYLES = {
    "Path_A": ("Path A (SGD-OCSVM)", "#d62728", "--", 1.8),
    "Path_B": ("Path B (CNN-1D AE)", "#ff7f0e", "-.", 1.8),
    "Path_C": ("Path C (GRU)", "#2ca02c", ":", 2.0),
    "XGBoost_Fusion": ("XGBoost Fusion", "#1f77b4", "-", 2.2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    parser.add_argument("--roc-json", type=Path, default=None, help="Optional ROC JSON input path.")
    parser.add_argument("--phase5-results", type=Path, default=None, help="Optional Phase 5 results JSON path.")
    parser.add_argument("--out-metrics", type=Path, default=None, help="Optional metrics summary output path.")
    parser.add_argument(
        "--out",
        type=Path,
        action="append",
        default=None,
        help="Figure output path. Can be passed more than once.",
    )
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=None,
        help="Optional paper figures directory for an extra PDF copy.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, list[Path], str]:
    dataset = normalize_dataset_name(args.dataset)
    logs_dir = ROOT / "experiments" / dataset / "logs"
    roc_json = args.roc_json or logs_dir / "roc_curves_refreshed.json"
    phase5_results = args.phase5_results or logs_dir / "phase5_fusion_results.json"
    out_metrics = args.out_metrics or logs_dir / "roc_exact_metrics.json"
    out_paths = args.out or [
        ROOT / "figures" / "roc_comparison.pdf",
        ROOT / "figures" / "roc_comparison.png",
    ]
    if args.paper_figures_dir is not None:
        out_paths.append(args.paper_figures_dir / "roc_comparison.pdf")
    return roc_json, phase5_results, out_metrics, out_paths, dataset


def load_roc_data(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing refreshed ROC export: {path}\n"
            "Run `python scripts/train_phase5_fusion.py --dataset DATASET` first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    missing = [key for key in STYLES if key not in data]
    if missing:
        raise KeyError(f"ROC export is missing required curves: {missing}")
    return data


def validate_against_phase5_results(roc_data: dict, phase5_results: Path) -> None:
    with open(phase5_results, "r", encoding="utf-8") as handle:
        phase5 = json.load(handle)

    expected_xgb = phase5["XGBoost"]
    curve = roc_data["XGBoost_Fusion"]
    op = curve.get("operating_point", {})
    checks = [
        ("XGBoost AUC", curve["auc"], expected_xgb["AUC_test"]),
        ("XGBoost FPR", op.get("fpr"), expected_xgb["FPR_test"]),
        ("XGBoost TPR", op.get("tpr"), expected_xgb["Recall_test"]),
        ("XGBoost threshold", op.get("threshold"), expected_xgb["threshold_tau"]),
    ]
    failures = []
    for label, actual, expected in checks:
        if actual is None or abs(float(actual) - float(expected)) > 5e-4:
            failures.append(f"{label}: actual={actual}, expected={expected}")

    if failures:
        details = "\n ".join(failures)
        raise ValueError(
            "Refreshed ROC JSON does not match phase5_fusion_results.json.\n"
            f" {details}\n"
            "Re-export ROC curves from the same Phase-5 run used for the paper tables."
        )


def write_metrics_summary(roc_data: dict, out_metrics: Path) -> None:
    summary = {
        key: {
            "AUC": round(float(value["auc"]), 4),
            **(
                {"operating_point": value["operating_point"]}
                if "operating_point" in value
                else {}
            ),
        }
        for key, value in roc_data.items()
    }
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def plot_roc(roc_data: dict, out_paths: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for key, (label, color, linestyle, linewidth) in STYLES.items():
        curve = roc_data[key]
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            label=f"{label} (AUC={float(curve['auc']):.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")

    op = roc_data["XGBoost_Fusion"].get("operating_point")
    if op:
        ax.plot(op["fpr"], op["tpr"], "r*", markersize=12, zorder=5)
        ax.annotate(
            f"Operating point\n(FPR={op['fpr'] * 100:.1f}%, TPR={op['tpr'] * 100:.1f}%)",
            xy=(op["fpr"], op["tpr"]),
            xytext=(op["fpr"] + 0.12, op["tpr"] - 0.12),
            fontsize=8,
            arrowprops=dict(arrowstyle=" to ", color="black", lw=0.8),
        )

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    for output_path in out_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    roc_json, phase5_results, out_metrics, out_paths, dataset = resolve_paths(args)

    roc_data = load_roc_data(roc_json)
    validate_against_phase5_results(roc_data, phase5_results)
    write_metrics_summary(roc_data, out_metrics)
    plot_roc(roc_data, out_paths)

    print(f"\nROC AUC summary for {dataset}")
    for key in STYLES:
        print(f" {key}: {float(roc_data[key]['auc']):.4f}")
    op = roc_data["XGBoost_Fusion"].get("operating_point", {})
    if op:
        print(
            " Operating point: "
            f"FPR={float(op['fpr']):.4f}, "
            f"TPR={float(op['tpr']):.4f}, "
            f"threshold={float(op['threshold']):.6f}"
        )


if __name__ == "__main__":
    main()

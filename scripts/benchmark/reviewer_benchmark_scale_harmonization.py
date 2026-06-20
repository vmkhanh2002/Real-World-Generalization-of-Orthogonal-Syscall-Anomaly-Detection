"""
Load-only review of score scale heterogeneity on the selected Z-matrix.

This script does not fit synthetic LOF/IsolationForest baselines. It loads the
saved Z-matrix split artifact and reports per-dimension scale spans so the
review can reason about whether score harmonization is needed.
"""

import json
import os
import pickle
import sys

import numpy as np

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

from benchmark_common import DEFAULT_DATASET, PROJECT_ROOT, resolve_phase5_selection


def describe_scale(values):
    lo = float(np.min(values))
    hi = float(np.max(values))
    mean = float(np.mean(values))
    std = float(np.std(values))
    span = hi - lo
    return {"min": lo, "max": hi, "mean": mean, "std": std, "span": span}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    z_file = os.path.join(log_dir, "z_matrix_splits.pkl")
    out_path = os.path.join(log_dir, "reviewer_benchmark_scale_harmonization.py.out.txt")

    if not os.path.exists(z_file):
        raise FileNotFoundError(
            f"Missing Z-matrix artifact: {z_file}\nRun benchmark_ablation_score_space.py first "
            "to rebuild the selected score-space splits."
        )

    with open(z_file, "rb") as f:
        data = pickle.load(f)

    z_all = np.vstack([data["z_train"], data["z_val"], data["z_test"]])
    selection = resolve_phase5_selection(args.dataset)
    labels = [
        f"Path A ({selection['path_a_label']})",
        f"Path B ({selection['path_b_label']})",
        f"Path C ({selection['path_c_label']})",
    ]

    stats = {label: describe_scale(z_all[:, idx]) for idx, label in enumerate(labels)}
    spans = [stats[label]["span"] for label in labels]
    heterogeneity_ratio = max(spans) / max(min(spans), 1e-12)

    lines = [
        "==============================================",
        " SCALE HARMONIZATION REVIEW (LOAD-ONLY) ",
        "==============================================",
    ]
    for label in labels:
        entry = stats[label]
        lines.append(
            f"{label}: min={entry['min']:.6f} max={entry['max']:.6f} "
            f"mean={entry['mean']:.6f} std={entry['std']:.6f} span={entry['span']:.6f}"
        )
    lines.append("----------------------------------------------")
    lines.append(f"Span heterogeneity ratio (max/min): {heterogeneity_ratio:.4f}")
    lines.append("Note: this script intentionally does not fit scalers or synthetic IF baselines.")

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    json_out = os.path.join(log_dir, "scale_harmonization_review.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "stats": stats,
                "span_heterogeneity_ratio": heterogeneity_ratio,
                "source": z_file,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()

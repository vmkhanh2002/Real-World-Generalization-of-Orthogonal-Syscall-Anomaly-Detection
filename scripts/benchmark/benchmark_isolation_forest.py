"""
Load-only benchmark: exported Isolation Forest baseline vs saved Phase 5 model.

This script does not train inside scripts/benchmark. It loads:
 - the saved selected Phase 5 meta-model metrics
 - the exported full-eval Z-matrix artifact
 - the exported Isolation Forest score-space artifact
"""

import json
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from benchmark_common import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_phase5_baseline_payload,
    load_phase5_meta_payload,
    load_phase5_z_eval,
)


ISO_VS_XGB_IF_ARTIFACT = "iso_vs_xgb_isolation_forest"


def apply_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    if preds.sum() == 0:
        return 0.0
    return float(f1_score(y_true, preds, zero_division=0))


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Load-only benchmark: exported Isolation Forest vs saved Phase 5 meta-model"
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "experiments" / args.dataset / "logs"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 65)
    print(f" BENCHMARK: Isolation Forest vs XGBoost ({args.dataset.upper()})")
    print("=" * 65)

    print("\nA. Loading selected Phase 5 meta-classifier artifact...")
    phase5_payload = load_phase5_meta_payload(args.dataset)
    xgb_label = phase5_payload.get("model_name", "Phase5_Meta")
    xgb_auc = float(phase5_payload["metrics"]["AUC_test"])
    xgb_f1 = float(phase5_payload["metrics"]["F1_test_final"])
    print(f" {xgb_label} AUC: {xgb_auc:.4f} | F1: {xgb_f1:.4f}")

    print("\nB. Loading exported Isolation Forest benchmark artifact...")
    z_eval = load_phase5_z_eval(args.dataset)
    z_full = np.asarray(z_eval["z_eval"])
    y_full = np.asarray(z_eval["y_eval"])
    iso_payload = load_phase5_baseline_payload(args.dataset, ISO_VS_XGB_IF_ARTIFACT)
    iso_scores = -iso_payload["model"].score_samples(z_full)
    iso_auc = float(roc_auc_score(y_full, iso_scores))
    iso_f1 = apply_threshold(y_full, iso_scores, float(iso_payload["threshold_tau"]))
    print(f" to Isolation Forest AUC: {iso_auc:.4f} | F1: {iso_f1:.4f}")

    delta_auc = xgb_auc - iso_auc
    delta_f1 = xgb_f1 - iso_f1

    print("\n" + "=" * 65)
    print(" COMPARISON RESULTS")
    print("=" * 65)
    print(f" {'Model':<30} {'AUC':>8} {'F1':>8}")
    print(f" {'-'*46}")
    print(f" {xgb_label:<30} {xgb_auc:>8.4f} {xgb_f1:>8.4f}")
    print(f" {'Isolation Forest (Artifact)':<30} {iso_auc:>8.4f} {iso_f1:>8.4f}")
    print(f" {'-'*46}")
    print(f" Performance Gap ({xgb_label} - ISO) dAUC: {delta_auc:+.4f} dF1: {delta_f1:+.4f}")

    if delta_f1 <= 0.03:
        verdict = "Option 2 RECOMMENDED: Isolation Forest within 3% of XGBoost."
    elif delta_f1 <= 0.06:
        verdict = "BORDERLINE: Gap is 3-6%."
    else:
        verdict = "Option 1 REQUIRED: Gap > 6%."

    print(f"\n VERDICT: {verdict}")

    result = {
        "XGBoost_Supervised": {"AUC": round(xgb_auc, 4), "F1": round(xgb_f1, 4)},
        "IsolationForest_Unsupervised": {"AUC": round(iso_auc, 4), "F1": round(iso_f1, 4)},
        "Delta_AUC": round(delta_auc, 4),
        "Delta_F1": round(delta_f1, 4),
        "Sources": {
            "meta_model": f"models/{args.dataset}/phase5_fusion/meta_best_model.pkl",
            "z_eval": f"experiments/{args.dataset}/logs/z_matrix_eval_full.pkl",
            "isolation_forest_artifact": (
                f"models/{args.dataset}/phase5_fusion/baselines/{ISO_VS_XGB_IF_ARTIFACT}.pkl"
            ),
        },
        "Verdict": verdict,
    }

    out_path = log_dir / "benchmark_iso_vs_xgb.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)

    print(f"\n Full results saved to {out_path}")


if __name__ == "__main__":
    main()

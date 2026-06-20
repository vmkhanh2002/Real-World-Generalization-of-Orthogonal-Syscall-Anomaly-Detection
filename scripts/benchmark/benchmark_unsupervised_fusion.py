"""
Load-only benchmark for unsupervised fusion baselines.

This script evaluates exported Phase 5 score-space artifacts on the saved
Z-matrix split artifact. It does not fit QuantileTransformer or GMM inside
scripts/benchmark.
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
    load_phase5_z_splits,
)


UNSUP_CDF_MAX_ARTIFACT = "unsupervised_cdf_max"
UNSUP_GMM_ARTIFACT = "unsupervised_gmm"


def apply_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    if preds.sum() == 0:
        return 0.0
    return float(f1_score(y_true, preds, zero_division=0))


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "experiments" / args.dataset / "logs"
    os.makedirs(log_dir, exist_ok=True)

    z_splits = load_phase5_z_splits(args.dataset)
    z_test = np.asarray(z_splits["z_test"])
    y_test = np.asarray(z_splits["y_test"])

    cdf_payload = load_phase5_baseline_payload(args.dataset, UNSUP_CDF_MAX_ARTIFACT)
    cdf_scores = cdf_payload["transformer"].transform(z_test).max(axis=1)
    cdf_auc = float(roc_auc_score(y_test, cdf_scores))
    cdf_f1 = apply_threshold(y_test, cdf_scores, float(cdf_payload["threshold_tau"]))

    gmm_payload = load_phase5_baseline_payload(args.dataset, UNSUP_GMM_ARTIFACT)
    gmm_scores = -gmm_payload["model"].score_samples(z_test)
    gmm_auc = float(roc_auc_score(y_test, gmm_scores))
    gmm_f1 = apply_threshold(y_test, gmm_scores, float(gmm_payload["threshold_tau"]))

    results = {
        "Unsupervised_CDF_Max": {"AUC": round(cdf_auc, 4), "F1": round(cdf_f1, 4)},
        "Unsupervised_GMM": {"AUC": round(gmm_auc, 4), "F1": round(gmm_f1, 4)},
        "Sources": {
            "z_splits": str(PROJECT_ROOT / "experiments" / args.dataset / "logs" / "z_matrix_splits.pkl"),
            "cdf_max_artifact": str(
                PROJECT_ROOT
                / "models"
                / args.dataset
                / "phase5_fusion"
                / "baselines"
                / f"{UNSUP_CDF_MAX_ARTIFACT}.pkl"
            ),
            "gmm_artifact": str(
                PROJECT_ROOT
                / "models"
                / args.dataset
                / "phase5_fusion"
                / "baselines"
                / f"{UNSUP_GMM_ARTIFACT}.pkl"
            ),
        },
    }

    out_file = log_dir / "benchmark_unsupervised_baselines.json"
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    phase5_payload = load_phase5_meta_payload(args.dataset)
    print(f"INFO Loaded saved Phase 5 meta-model: {phase5_payload.get('model_name', 'Phase5_Meta')}")
    print(f"INFO Evaluated exported unsupervised baselines from models/{args.dataset}/phase5_fusion/baselines")
    print(json.dumps(results, indent=4))
    print(f"INFO Saved results to {out_file}")


if __name__ == "__main__":
    main()

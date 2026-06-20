"""
Phase 5 ablation benchmark: load-only score-space baselines.

This script does not train Isolation Forest or GMM. It loads:
 - exported Z-matrix split artifacts
 - exported score-space baseline model artifacts

and evaluates them on the saved test split.
"""

import json
import sys

import numpy as np
from sklearn.metrics import f1_score

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from benchmark_common import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_phase5_baseline_payload,
    load_phase5_meta_payload,
    load_phase5_z_splits,
)


ABLATION_IF_ARTIFACT = "ablation_isolation_forest"
ABLATION_GMM_ARTIFACT = "ablation_gmm"


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

    print("=" * 60)
    print("PHASE 5 ABLATION: LOAD-ONLY SCORE-SPACE BASELINES")
    print("=" * 60)

    z_splits = load_phase5_z_splits(args.dataset)
    z_test = np.asarray(z_splits["z_test"])
    y_test = np.asarray(z_splits["y_test"])

    iso_payload = load_phase5_baseline_payload(args.dataset, ABLATION_IF_ARTIFACT)
    iso_scores = -iso_payload["model"].score_samples(z_test)
    iso_f1 = apply_threshold(y_test, iso_scores, float(iso_payload["threshold_tau"]))

    gmm_payload = load_phase5_baseline_payload(args.dataset, ABLATION_GMM_ARTIFACT)
    gmm_scores = -gmm_payload["model"].score_samples(z_test)
    gmm_f1 = apply_threshold(y_test, gmm_scores, float(gmm_payload["threshold_tau"]))

    results = {
        "IsolationForest_ScoreSpace": {"F1_test_final": round(iso_f1, 4)},
        "GMM_ScoreSpace": {"F1_test_final": round(gmm_f1, 4)},
        "Sources": {
            "z_splits": str(PROJECT_ROOT / "experiments" / args.dataset / "logs" / "z_matrix_splits.pkl"),
            "isolation_forest_artifact": str(
                PROJECT_ROOT
                / "models"
                / args.dataset
                / "phase5_fusion"
                / "baselines"
                / f"{ABLATION_IF_ARTIFACT}.pkl"
            ),
            "gmm_artifact": str(
                PROJECT_ROOT
                / "models"
                / args.dataset
                / "phase5_fusion"
                / "baselines"
                / f"{ABLATION_GMM_ARTIFACT}.pkl"
            ),
        },
    }

    out_file = log_dir / "ablation_unsupervised_fusion.json"
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    phase5_payload = load_phase5_meta_payload(args.dataset)
    print(f"INFO Loaded saved Phase 5 model: {phase5_payload.get('model_name', 'Phase5_Meta')}")
    print(f"INFO Loaded exported baselines from models/{args.dataset}/phase5_fusion/baselines")
    print(json.dumps(results, indent=4))
    print(f"INFO Saved results to {out_file}")


if __name__ == "__main__":
    main()

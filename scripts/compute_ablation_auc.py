"""
Compute missing AUC values for the two unsupervised score-space ablation methods:
 - IsolationForest (score space only)
 - GMM (score space only)

Reads the Phase 5 score split and exported score-space baselines for the
selected dataset.

Writes the ablation summary JSON with AUC values filled in.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_config import DEFAULT_DATASET, normalize_dataset_name


def resolve_paths(dataset: str) -> dict[str, Path]:
    logs_dir = ROOT / "experiments" / dataset / "logs"
    baselines_dir = ROOT / "models" / dataset / "phase5_fusion" / "baselines"
    return {
        "z_splits": logs_dir / "z_matrix_splits.pkl",
        "isolation_forest": baselines_dir / "ablation_isolation_forest.pkl",
        "gmm": baselines_dir / "ablation_gmm.pkl",
        "summary": logs_dir / "ablation_unsupervised_fusion.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    return parser.parse_args()


def load_test_split(z_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(z_path, "rb") as f:
        data = pickle.load(f)
    Z_test = np.array(data["z_test"])
    y_test = np.array(data["y_test"]).astype(int)
    print(f"Test split: {Z_test.shape[0]} windows, {y_test.sum()} attack")
    return Z_test, y_test


def compute_auc_for_model(model_path: Path, Z_test: np.ndarray, y_test: np.ndarray, name: str) -> float:
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    # Some artifacts wrap the estimator under the model key.
    model = artifact["model"] if isinstance(artifact, dict) else artifact

    if hasattr(model, "decision_function"):
        # IsolationForest: higher score = more normal invert for anomaly score
        raw_scores = model.decision_function(Z_test)
        anomaly_scores = -raw_scores
    elif hasattr(model, "score_samples"):
        # GaussianMixture: log likelihood, higher = more normal invert
        raw_scores = model.score_samples(Z_test)
        anomaly_scores = -raw_scores
    else:
        raise ValueError(f"Model {name} has neither decision_function nor score_samples")

    auc = roc_auc_score(y_test, anomaly_scores)
    print(f"{name}: AUC = {auc:.4f}")
    return round(float(auc), 4)


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    paths = resolve_paths(dataset)

    Z_test, y_test = load_test_split(paths["z_splits"])

    if_auc  = compute_auc_for_model(paths["isolation_forest"], Z_test, y_test, "IsolationForest_ScoreSpace")
    gmm_auc = compute_auc_for_model(paths["gmm"], Z_test, y_test, "GMM_ScoreSpace")

    # Load existing JSON and update with AUC values
    with open(paths["summary"], "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing["IsolationForest_ScoreSpace"]["AUC_test"] = if_auc
    existing["GMM_ScoreSpace"]["AUC_test"]             = gmm_auc

    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

    print(f"\nUpdated {paths['summary']}")
    print(f" IF (score space only) AUC = {if_auc}")
    print(f" GMM (score space only) AUC = {gmm_auc}")


if __name__ == "__main__":
    main()

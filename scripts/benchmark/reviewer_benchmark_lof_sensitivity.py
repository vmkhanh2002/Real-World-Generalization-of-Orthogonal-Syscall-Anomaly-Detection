"""
Load-only review of LOF evidence for Path A.

This script does not refit LOF. It loads:
 - the saved LOF model artifact
 - the saved Path A result log
 - the saved best-config metadata

and summarizes the currently stored LOF evidence.
"""

import json
import os
import pickle
import sys

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

from benchmark_common import DEFAULT_DATASET, PROJECT_ROOT


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset, "path_a")
    os.makedirs(log_dir, exist_ok=True)

    results_path = os.path.join(log_dir, "path_a_results.json")
    best_cfg_path = os.path.join(model_dir, "path_a_best_config_per_model_type.json")
    lof_model_path = os.path.join(model_dir, "lof.pkl")
    out_path = os.path.join(log_dir, "reviewer_benchmark_lof_sensitivity.py.out.txt")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing Path A results log: {results_path}")
    if not os.path.exists(best_cfg_path):
        raise FileNotFoundError(f"Missing Path A best-config metadata: {best_cfg_path}")
    if not os.path.exists(lof_model_path):
        raise FileNotFoundError(f"Missing LOF model artifact: {lof_model_path}")

    with open(results_path, "r", encoding="utf-8-sig") as f:
        path_a_results = json.load(f)
    with open(best_cfg_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)
    with open(lof_model_path, "rb") as f:
        lof_model = pickle.load(f)

    lof_metrics = path_a_results.get("results", {}).get("LOF", {})
    lof_cfg = best_cfg.get("LOF", {})
    n_neighbors = getattr(lof_model, "n_neighbors", None)
    contamination = getattr(lof_model, "contamination", None)

    lines = [
        "==================================",
        "LOF EVIDENCE REVIEW (LOAD-ONLY)",
        "==================================",
        f"Model artifact: {lof_model_path}",
        f"Configured feature: {lof_cfg.get('feature')}",
        f"Configured analyzer: {lof_cfg.get('tfidf_analyzer')}",
        f"Configured ngram_range: {lof_cfg.get('tfidf_ngram_range')}",
        f"Configured reduction: {lof_cfg.get('reduction')}",
        f"Configured hyperparam k: {lof_cfg.get('hyperparam', {}).get('k')}",
        f"Loaded model n_neighbors: {n_neighbors}",
        f"Loaded model contamination: {contamination}",
        f"Stored AUC_ROC: {lof_metrics.get('AUC_ROC')}",
        f"Stored AUPR: {lof_metrics.get('AUPR')}",
        f"Stored F1_calib: {lof_metrics.get('F1_calib')}",
        f"Stored FPR_at_TPR95: {lof_metrics.get('FPR_at_TPR95')}",
        "",
        "Note: this script intentionally does not refit LOF or sweep contamination values.",
    ]

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()

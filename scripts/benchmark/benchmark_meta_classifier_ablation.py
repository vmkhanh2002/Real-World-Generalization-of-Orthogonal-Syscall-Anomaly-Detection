"""
Benchmark: Meta-classifier ablation (NICE TO HAVE #9).
Compares all 5 trained meta-classifiers on the single shared eval pool.

UNIFIED: uses pre-computed fusion probabilities in the pool artifact (inference
already done in the builder). Reports AUC/AUPR, F1 at the locked tau*, and F1 at
the dynamic best threshold.

Output:
    experiments/32bit/logs/unified/meta_classifier_ablation.json
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))
from pool_io import load_pool, save_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()

    pool = load_pool(args.dataset)
    y = pool["y_eval"]
    tau = pool["tau_star"]

    results = {}
    for name, proba in pool["proba"].items():
        auc = roc_auc_score(y, proba)
        aupr = average_precision_score(y, proba)
        f1_tau = f1_score(y, (proba >= tau).astype(int))
        p, r, thr = precision_recall_curve(y, proba)
        f1v = 2 * p * r / (p + r + 1e-12)
        bi = int(np.argmax(f1v[:-1]))
        results[name] = {
            "AUC": round(float(auc), 4),
            "AUPR": round(float(aupr), 4),
            "F1_at_tau_star": round(float(f1_tau), 4),
            "F1_dynamic": round(float(f1v[bi]), 4),
            "dynamic_threshold": round(float(thr[bi]), 6),
        }

    out = {"pool": "unified_eval_pool", "tau_star": tau, "n": int(len(y)), "models": results}
    path = save_json(out, args.dataset, "meta_classifier_ablation.json")
    print(f"{'Classifier':<24}{'AUC':>8}{'AUPR':>8}{'F1@tau*':>9}{'F1_dyn':>8}")
    for n, r in sorted(results.items(), key=lambda x: -x[1]["AUC"]):
        print(f"{n:<24}{r['AUC']:>8.4f}{r['AUPR']:>8.4f}{r['F1_at_tau_star']:>9.4f}{r['F1_dynamic']:>8.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

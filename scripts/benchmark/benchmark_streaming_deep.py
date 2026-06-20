"""
Benchmark: Deep per-path / per-family detection analysis (SHOULD FIX #6).

UNIFIED: single shared pool. NOTE these are WINDOW-LEVEL detection rates at
P95-on-benign thresholds (NOT the episode-level hysteresis recall of the
realtime_eval chapter). Keys are named window_recall to avoid confusion.

Output:
    experiments/32bit/logs/unified/streaming_per_path.json
    experiments/32bit/logs/unified/streaming_failure_analysis.json
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))
from pool_io import load_pool, save_json  # noqa: E402


def per_path_block(scores, thr, y, fam):
    flags = (scores >= thr).astype(int)
    atk = y == 1
    ben = y == 0
    fam_rec = {}
    for f in sorted(set(fam[atk])):
        m = atk & (fam == f)
        fam_rec[str(f)] = {
            "window_recall": round(float(flags[m].mean()), 4),
            "n_windows": int(m.sum()),
            "mean_score": round(float(scores[m].mean()), 4),
        }
    return {
        "window_recall": round(float(flags[atk].mean()), 4),
        "false_alert_rate_window": round(float(flags[ben].mean()), 4),
        "threshold": round(float(thr), 6),
        "n_attack_windows": int(atk.sum()),
        "missed_pct": round(float(1 - flags[atk].mean()) * 100, 2),
        "per_family_window_recall": fam_rec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()

    pool = load_pool(args.dataset)
    y = pool["y_eval"]
    fam = np.asarray(pool["family_eval"])
    th = pool["thresholds"]
    sa, sb, sc = pool["scores"]["path_a"], pool["scores"]["path_b"], pool["scores"]["path_c"]
    fus = pool["proba"]["XGBoost"]

    per_path = {
        "Path_A": per_path_block(sa, th["path_a_p95_benign"], y, fam),
        "Path_B": per_path_block(sb, th["path_b_p95_benign"], y, fam),
        "Path_C": per_path_block(sc, th["path_c_p95_benign"], y, fam),
        "Fusion_XGBoost": per_path_block(fus, th["fusion_p95_benign"], y, fam),
        "_note": "WINDOW-LEVEL recall at P95-benign thresholds; not episode-level hysteresis recall.",
    }
    save_json(per_path, args.dataset, "streaming_per_path.json")

    # failure analysis per family: detection by each path / by any / missed-by-all
    atk = y == 1
    ta, tb, tc = th["path_a_p95_benign"], th["path_b_p95_benign"], th["path_c_p95_benign"]
    tf = th["fusion_p95_benign"]
    fail = {}
    for f in sorted(set(fam[atk])):
        m = atk & (fam == f)
        da = (sa[m] >= ta); db = (sb[m] >= tb); dc = (sc[m] >= tc); dfu = (fus[m] >= tf)
        any_single = da | db | dc
        fail[str(f)] = {
            "n_windows": int(m.sum()),
            "detected_by_a": round(float(da.mean()), 4),
            "detected_by_b": round(float(db.mean()), 4),
            "detected_by_c": round(float(dc.mean()), 4),
            "detected_by_fusion": round(float(dfu.mean()), 4),
            "detected_by_any_single": round(float(any_single.mean()), 4),
            "missed_by_all_single": round(float((~any_single).mean()), 4),
        }
    save_json(fail, args.dataset, "streaming_failure_analysis.json")
    print("Per-path window recall:",
          {k: per_path[k]["window_recall"] for k in ["Path_A", "Path_B", "Path_C", "Fusion_XGBoost"]})
    print("Saved streaming_per_path.json + streaming_failure_analysis.json")


if __name__ == "__main__":
    main()

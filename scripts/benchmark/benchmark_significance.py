"""
Benchmark: Statistical significance of fusion gain (MUST FIX #1).
Runs DeLong test comparing XGBoost fusion vs. each single path.

UNIFIED: reads experiments/<ds>/logs/unified/unified_eval_pool.pkl (inference-only,
single shared pool). Reports BOTH window-level and trace-level DeLong; trace-level
removes the pseudo-replication of overlapping sliding windows. Fixes p-value
underflow via scipy.stats.norm.logsf (adds neg_log10_p).

Output:
    experiments/32bit/logs/unified/significance_tests.json
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))

from delong_test import delong_test  # noqa: E402
from pool_io import load_pool, save_json  # noqa: E402


def robust_p_from_z(z):
    """Two-sided p and -log10(p) that don't underflow to 0 for large |z|."""
    logsf = stats.norm.logsf(abs(z))  # log P(Z > |z|)
    log_p = np.log(2.0) + logsf  # natural log of two-sided p
    p = float(np.exp(log_p)) if log_p > -700 else 0.0
    neg_log10_p = float(-(log_p / np.log(10.0)))
    return p, neg_log10_p


def aggregate_by_trace(values, trace_ids, y):
    """Mean score per trace + trace label (traces are pure benign or pure attack)."""
    uniq = np.unique(trace_ids)
    sv = np.empty(len(uniq))
    yv = np.empty(len(uniq))
    for i, t in enumerate(uniq):
        m = trace_ids == t
        sv[i] = values[m].mean()
        yv[i] = y[m].mean()  # 0 or 1
    return sv, yv


def run_block(y, fusion, paths):
    out = {}
    for name, s in paths.items():
        r = delong_test(y, fusion, s)
        p, nlp = robust_p_from_z(r["z_statistic"])
        r["p_value"] = p
        r["neg_log10_p"] = nlp
        r["p_display"] = "<1e-300" if p == 0.0 else f"{p:.3e}"
        out[f"fusion_vs_{name}"] = r
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()

    pool = load_pool(args.dataset)
    y = pool["y_eval"]
    tid = pool["trace_ids_eval"]
    fusion = pool["proba"]["XGBoost"]
    paths = {"path_a": pool["scores"]["path_a"],
             "path_b": pool["scores"]["path_b"],
             "path_c": pool["scores"]["path_c"]}

    print("=" * 60)
    print("DeLong significance (unified pool)")
    print("=" * 60)

    window_level = run_block(y, fusion, paths)
    print("\nWindow-level:")
    for k, r in window_level.items():
        print(f"  {k}: dAUC={r['auc_diff']:+.4f} z={r['z_statistic']:.2f} p={r['p_display']}")

    # trace-level (removes overlapping-window pseudo-replication)
    fusion_t, y_t = aggregate_by_trace(fusion, tid, y)
    paths_t = {}
    for name, s in paths.items():
        st, _ = aggregate_by_trace(s, tid, y)
        paths_t[name] = st
    trace_level = run_block(y_t, fusion_t, paths_t)
    print(f"\nTrace-level (n_traces={len(y_t)}, pos={int(y_t.sum())}):")
    for k, r in trace_level.items():
        print(f"  {k}: dAUC={r['auc_diff']:+.4f} z={r['z_statistic']:.2f} p={r['p_display']}")

    result = {
        "pool": "unified_eval_pool",
        "n_windows": int(len(y)),
        "n_traces": int(len(y_t)),
        "fusion_model": "XGBoost",
        "window_level": window_level,
        "trace_level": trace_level,
        "note": "Trace-level is the defensible test (windows overlap 90% at stride 2; "
                "window-level z is inflated by pseudo-replication). p-values via logsf.",
    }
    out = save_json(result, args.dataset, "significance_tests.json")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

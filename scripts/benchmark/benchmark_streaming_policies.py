"""
Benchmark: Episode-decision policy comparison (NICE TO HAVE #12).

UNIFIED: single shared pool, grouped into episodes by trace_ids. Because the
pool is a sampled set (windows are NOT in temporal order), policies here are
ORDER-AGNOSTIC count/fraction rules on per-trace fusion probabilities, not the
temporal rolling-hysteresis of the realtime_eval chapter (which remains the
authoritative streaming evaluation). This is a relative comparison of decision
rules on one consistent pool.

Output:
    experiments/32bit/logs/unified/streaming_policy_comparison.json
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))
from pool_io import load_pool, save_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()

    pool = load_pool(args.dataset)
    y = pool["y_eval"]
    tid = pool["trace_ids_eval"]
    tau = pool["tau_star"]
    proba = pool["proba"]["XGBoost"]
    pos = (proba >= tau).astype(int)

    # group into episodes (one per trace)
    uniq = np.unique(tid)
    n_win, n_pos, pos_rate, mean_p, label = [], [], [], [], []
    for t in uniq:
        m = tid == t
        n_win.append(int(m.sum()))
        n_pos.append(int(pos[m].sum()))
        pos_rate.append(float(pos[m].mean()))
        mean_p.append(float(proba[m].mean()))
        label.append(int(y[m].mean() >= 0.5))
    n_win = np.array(n_win); n_pos = np.array(n_pos)
    pos_rate = np.array(pos_rate); mean_p = np.array(mean_p); label = np.array(label)
    atk = label == 1; ben = label == 0

    policies = {
        "single_threshold_any": n_pos >= 1,
        "count_2_positive": n_pos >= 2,
        "count_3_positive": n_pos >= 3,
        "fraction_30pct": pos_rate >= 0.30,
        "fraction_60pct": pos_rate >= 0.60,
        "mean_proba_ge_tau": mean_p >= tau,
    }
    results = {}
    for name, fired in policies.items():
        results[name] = {
            "episode_recall": round(float(fired[atk].mean()), 4),
            "false_alert_rate": round(float(fired[ben].mean()), 4),
        }

    out = {
        "pool": "unified_eval_pool",
        "n_episodes": int(len(uniq)),
        "n_attack_episodes": int(atk.sum()),
        "n_benign_episodes": int(ben.sum()),
        "tau_star": tau,
        "policies": results,
        "note": "Order-agnostic count/fraction episode rules (pool has no temporal order). "
                "Authoritative temporal hysteresis recall (0.7895) is in realtime_eval, not here.",
    }
    path = save_json(out, args.dataset, "streaming_policy_comparison.json")
    for n, r in results.items():
        print(f"  {n:<22} recall={r['episode_recall']:.4f} FAR={r['false_alert_rate']:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

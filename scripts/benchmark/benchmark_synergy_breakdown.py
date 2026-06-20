"""
Benchmark: Cross-path synergy breakdown (MUST FIX #2).
Tabulates all 2^3 detection patterns on attack windows + independence baseline.

UNIFIED: reads the single shared pool. Thresholds = per-path P95-on-benign,
defined once in the pool artifact. Eval-pool attack side (50k) = same basis as
the paper's synergy figure.

Output:
    experiments/32bit/logs/unified/synergy_breakdown.json
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))
from pool_io import load_pool, save_json  # noqa: E402


def breakdown(fa, fb, fc, n_ref):
    pattern = 4 * fa + 2 * fb + fc
    per = {}
    for val in range(8):
        nm = "+".join([c for c, b in zip("ABC", [(val >> 2) & 1, (val >> 1) & 1, val & 1]) if b]) or "None"
        cnt = int(np.sum(pattern == val))
        per[nm] = {"pattern_code": val, "count": cnt, "pct": round(cnt / n_ref * 100, 2)}
    s = fa + fb + fc
    p_a, p_b, p_c = fa.mean(), fb.mean(), fc.mean()
    exp = (p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c + (1 - p_a) * p_b * p_c + p_a * p_b * p_c) * n_ref
    syn = int(np.sum(s >= 2))
    # Paper's original definition (reviewer_benchmark_orthogonality.py): A-missed but B or C caught
    a_missed_bc_caught = int(np.sum((fa == 0) & ((fb == 1) | (fc == 1))))
    return {
        "per_pattern": per,
        "total_detected": int(np.sum(pattern > 0)),
        "synergy_count_ge2_paths": syn,
        "synergy_count": syn,
        "paper_def_a_missed_bc_caught": a_missed_bc_caught,
        "synergy_pct": round(float(np.mean(s >= 2)) * 100, 2),
        "all_three": int(np.sum(pattern == 7)),
        "none_detected": int(np.sum(pattern == 0)),
        "marginal_detection_rates": {"p_a": round(float(p_a), 4), "p_b": round(float(p_b), 4), "p_c": round(float(p_c), 4)},
        "independence_expected_synergy": int(round(exp)),
        "observed_vs_expected_ratio": round(syn / max(round(exp), 1), 2),
        "n_total": int(n_ref),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()

    pool = load_pool(args.dataset)
    y = pool["y_eval"]
    th = pool["thresholds"]
    sa, sb, sc = pool["scores"]["path_a"], pool["scores"]["path_b"], pool["scores"]["path_c"]
    ta, tb, tc = th["path_a_p95_benign"], th["path_b_p95_benign"], th["path_c_p95_benign"]
    fa, fb, fc = (sa >= ta).astype(int), (sb >= tb).astype(int), (sc >= tc).astype(int)
    atk = y == 1

    result = {
        "pool": "unified_eval_pool",
        "thresholds": {"a": ta, "b": tb, "c": tc, "rule": "per-path P95 on benign windows"},
        "attack_windows": breakdown(fa[atk], fb[atk], fc[atk], int(atk.sum())),
        "all_windows": breakdown(fa, fb, fc, len(y)),
    }
    out = save_json(result, args.dataset, "synergy_breakdown.json")
    aw = result["attack_windows"]
    print(f"Attack synergy (>=2 paths): {aw['synergy_count']}  "
          f"(indep {aw['independence_expected_synergy']}, ratio {aw['observed_vs_expected_ratio']})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

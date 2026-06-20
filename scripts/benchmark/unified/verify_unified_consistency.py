"""
Cross-consistency gate: assert every unified phase output reconciles to ONE pool.
Reads experiments/<ds>/logs/unified/*.json and checks fusion/per-path AUC,
pool sizes, and synergy arithmetic agree. Writes unified_consistency_report.json.
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from pool_io import unified_dir  # noqa: E402


def load(d, name):
    p = os.path.join(unified_dir(d), name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="32bit")
    args = ap.parse_args()
    d = args.dataset
    summary = load(d, "unified_eval_pool_summary.json")
    sig = load(d, "significance_tests.json")
    abl = load(d, "meta_classifier_ablation.json")
    abi = load(d, "abi_impact_analysis.json")
    syn = load(d, "synergy_breakdown.json")

    checks = []
    def chk(name, a, b, tol=1e-3):
        ok = (a is not None and b is not None and abs(a - b) <= tol)
        checks.append({"check": name, "a": a, "b": b, "pass": bool(ok)})

    fusion_pool = summary["per_model_fusion_auc"]["XGBoost"]
    pa = summary["per_path_auc"]["path_a"]

    if sig:
        chk("fusion AUC: pool vs DeLong(window auc_1)", fusion_pool, sig["window_level"]["fusion_vs_path_a"]["auc_1"])
        chk("path_a AUC: pool vs DeLong(window auc_2)", pa, sig["window_level"]["fusion_vs_path_a"]["auc_2"])
    if abl:
        chk("fusion AUC: pool vs ablation XGBoost", fusion_pool, abl["models"]["XGBoost"]["AUC"], tol=2e-3)
    if abi:
        chk("fusion AUC: pool vs ABI correct baseline", fusion_pool, abi["configs"]["correct_i386"]["auc_fusion"], tol=2e-3)
        chk("path_a AUC: pool vs ABI correct baseline", pa, abi["configs"]["correct_i386"]["auc_path_a"], tol=2e-3)
    if syn:
        aw = syn["attack_windows"]
        s = sum(v["count"] for k, v in aw["per_pattern"].items() if k != "None")
        chk("synergy: total_detected == sum(non-None patterns)", float(aw["total_detected"]), float(s), tol=0.5)

    n_pass = sum(c["pass"] for c in checks)
    report = {"dataset": d, "n_checks": len(checks), "n_pass": n_pass,
              "all_pass": n_pass == len(checks), "fusion_auc_pool": fusion_pool, "checks": checks}
    out = os.path.join(unified_dir(d), "unified_consistency_report.json")
    json.dump(report, open(out, "w"), indent=2)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['a']} vs {c['b']}")
    print(f"\n{n_pass}/{len(checks)} checks pass. Saved {out}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()

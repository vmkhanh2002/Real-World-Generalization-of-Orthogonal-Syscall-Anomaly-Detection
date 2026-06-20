# real_attack_eval/adduser/scripts/compute_metrics.py
"""
Window-level evaluation for real Adduser traces — LOAO-replicated protocol.

LOAO reference numbers are paper constants (not recomputed):
  Adduser: AUC=0.9052  AUCPR=0.6830  AUCPR(1:1)=0.8676  F1=0.7061
           TPR@FPR1%=0.0806  TPR@FPR0.1%=0.0118
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

_HERE         = Path(__file__).resolve().parent
_FAMILY_DIR   = _HERE.parent
_SHARED_DIR   = _FAMILY_DIR.parent / "shared"

REAL_ATK_W20  = _FAMILY_DIR / "artifacts" / "window_scores_w20.jsonl"
ADFA_BENIGN   = _SHARED_DIR / "adfa_benign_w20.jsonl"
REAL_MANIFEST = _FAMILY_DIR / "artifacts" / "manifest.json"

# Paper constants — never recomputed, used only as reference row
LOAO_PAPER = {
    "Adduser": {
        "AUC": 0.9052, "AUCPR": 0.6830, "AUCPR_1to1": 0.8676,
        "F1": 0.7061, "TPR@FPR1%": 0.0806, "TPR@FPR01%": 0.0118,
    },
}

TAU_STAR         = 0.401214
BENIGN_POOL_SIZE = 50_000
BENIGN_TEST_FRAC = 0.4
SEED             = 42


def load_benign_pool(jsonl_path, pool_size, seed):
    all_probs, all_tids = [], []
    for tid, line in enumerate(jsonl_path.open(encoding="utf-8")):
        row = json.loads(line)
        probs = row.get("phase5_probs", [])
        if probs:
            all_probs.extend(probs)
            all_tids.extend([tid] * len(probs))
    all_probs = np.array(all_probs, dtype=np.float32)
    all_tids  = np.array(all_tids,  dtype=np.int32)
    rng = np.random.default_rng(seed)
    n   = min(pool_size, len(all_probs))
    idx = rng.choice(len(all_probs), size=n, replace=False)
    return all_probs[idx], all_tids[idx]


def benign_test_split(probs, tids, test_frac, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    _, test_idx = next(gss.split(np.arange(len(probs)), groups=tids))
    return test_idx


def load_probs_from_jsonl(jsonl_path):
    probs = []
    for line in jsonl_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        probs.extend(json.loads(line).get("phase5_probs", []))
    return np.array(probs, dtype=np.float32)


def tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    idx = max(0, min(np.searchsorted(fpr, fpr_target, side="right") - 1, len(tpr) - 1))
    return float(tpr[idx])


def aucpr_equal_pool(y, s, seed=SEED):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    rng = np.random.default_rng(seed)
    if len(pos_idx) > n:
        pos_idx = rng.choice(pos_idx, size=n, replace=False)
    if len(neg_idx) > n:
        neg_idx = rng.choice(neg_idx, size=n, replace=False)
    return float(average_precision_score(y[np.concatenate([pos_idx, neg_idx])],
                                         s[np.concatenate([pos_idx, neg_idx])]))


def compute_metrics(pos, neg, tau):
    y    = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s    = np.concatenate([pos, neg])
    pred = (s > tau).astype(int)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    return {
        "AUC":        round(float(roc_auc_score(y, s)), 4),
        "AUCPR":      round(float(average_precision_score(y, s)), 4),
        "AUCPR_1to1": round(float(aucpr_equal_pool(y, s)), 4),
        "F1":         round(float((2 * prec * rec) / (prec + rec + 1e-9)), 4),
        "TPR@FPR1%":  round(float(tpr_at_fpr(y, s, 0.01)), 4),
        "TPR@FPR01%": round(float(tpr_at_fpr(y, s, 0.001)), 4),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
    }


def detection_rate(jsonl_path, manifest_path, tau):
    manifest = json.loads(manifest_path.read_text())
    file_to_family = {
        e["source_file"].replace("\\", "/").lower(): e["target_preset"]
        for e in manifest["entries"]
    }
    families = {}
    for line in jsonl_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row    = json.loads(line)
        key    = row["file"].replace("\\", "/").lower()
        family = file_to_family.get(key, "Unknown")
        probs  = np.array(row.get("phase5_probs", []), dtype=np.float32)
        if not len(probs):
            continue
        flagged = int((probs >= tau).sum())
        if family not in families:
            families[family] = {"n_traces": 0, "detected": 0, "total_w": 0, "flagged_w": 0, "rates": []}
        d = families[family]
        d["n_traces"] += 1; d["total_w"] += len(probs); d["flagged_w"] += flagged
        d["rates"].append(flagged / len(probs))
        if flagged >= 1:
            d["detected"] += 1
    out = {}
    for fam, d in families.items():
        rates = np.array(d["rates"])
        out[fam] = {
            "n_traces":         d["n_traces"],
            "detected_traces":  d["detected"],
            "trace_dr":         round(d["detected"] / d["n_traces"], 4),
            "total_windows":    d["total_w"],
            "flagged_windows":  d["flagged_w"],
            "window_dr":        round(d["flagged_w"] / d["total_w"], 4),
            "median_per_trace": round(float(np.median(rates)), 4),
            "p10_per_trace":    round(float(np.percentile(rates, 10)), 4),
        }
    return out


def print_detection_rate_table(dr, tau):
    W = 106
    print("=" * W)
    print(f"  Recall-Only Detection Rate (ALL real malware traces, tau*={tau})")
    print(f"  No benign pool required — pure TPR measurement")
    print("=" * W)
    hdr = (f"  {'Family':<22}  {'Traces':>7}  {'Detected':>9}  {'Trace-DR':>9}"
           f"  {'Windows':>10}  {'Flagged':>9}  {'Win-DR':>7}  {'Median%':>8}  {'P10%':>6}")
    print(hdr)
    print("-" * W)
    totals = {"n_traces": 0, "detected": 0, "total_w": 0, "flagged_w": 0}
    for fam in sorted(dr):
        d = dr[fam]
        print(f"  {fam:<22}  {d['n_traces']:>7,}  {d['detected_traces']:>9,}  {d['trace_dr']:>8.1%}"
              f"  {d['total_windows']:>10,}  {d['flagged_windows']:>9,}  {d['window_dr']:>6.1%}"
              f"  {d['median_per_trace']:>7.1%}  {d['p10_per_trace']:>5.1%}")
        totals["n_traces"] += d["n_traces"]; totals["detected"] += d["detected_traces"]
        totals["total_w"]  += d["total_windows"]; totals["flagged_w"] += d["flagged_windows"]
    print("-" * W)
    tdr = totals["detected"] / totals["n_traces"]
    wdr = totals["flagged_w"] / totals["total_w"]
    print(f"  {'TOTAL':<22}  {totals['n_traces']:>7,}  {totals['detected']:>9,}  {tdr:>8.1%}"
          f"  {totals['total_w']:>10,}  {totals['flagged_w']:>9,}  {wdr:>6.1%}"
          f"  {'':>8}  {'':>6}")
    print()
    print("  Trace-DR = % traces with >= 1 window flagged  (trace-level recall)")
    print("  Win-DR   = % of all windows flagged           (window-level recall)")
    print("  Median%  = median per-trace window detection rate")
    print("  P10%     = 10th-percentile per-trace rate (worst 10% of traces)")


def main():
    print(f"Building benign pool ({BENIGN_POOL_SIZE:,} windows, seed={SEED})...")
    b_probs, b_tids = load_benign_pool(ADFA_BENIGN, BENIGN_POOL_SIZE, SEED)
    print(f"  sampled {len(b_probs):,} windows")
    neg_test = b_probs[benign_test_split(b_probs, b_tids, BENIGN_TEST_FRAC, SEED)]
    print(f"  test negatives = {len(neg_test):,}\n")

    real_pos = load_probs_from_jsonl(REAL_ATK_W20)
    print(f"Real Adduser: {len(real_pos):,} windows\n")
    real_m = compute_metrics(real_pos, neg_test, TAU_STAR)

    W = 110
    loao = LOAO_PAPER["Adduser"]
    print("=" * W)
    print("  Window-Level Metrics (w=20) -- LOAO-Replicated Protocol")
    print(f"  Neg pool: {len(b_probs):,} windows, {BENIGN_TEST_FRAC:.0%} test => {len(neg_test):,} negatives")
    print(f"  Raw AUCPR and F1@tau* shown as n/a for Real (class-imbalance artifact)")
    print("=" * W)
    print(f"  {'Source':<30}  {'AUC':>7}  {'AUCPR':>7}  {'AUCPR(1:1)':>10}  {'F1@tau*':>8}  {'TPR@FPR1%':>10}  {'TPR@FPR0.1%':>12}")
    print("-" * W)
    print(f"  {'LOAO Adduser (paper)':<30}  {loao['AUC']:>7.4f}  {loao['AUCPR']:>7.4f}  {loao['AUCPR_1to1']:>10.4f}  {loao['F1']:>8.4f}  {loao['TPR@FPR1%']:>10.4f}  {loao['TPR@FPR01%']:>12.4f}")
    print(f"  {'Real Adduser':<30}  {real_m['AUC']:>7.4f}  {'n/a':>7}  {real_m['AUCPR_1to1']:>10.4f}  {'n/a':>8}  {real_m['TPR@FPR1%']:>10.4f}  {real_m['TPR@FPR01%']:>12.4f}")
    print()
    print("  LOAO row: paper constants — not recomputed.")
    print("  n/a: AUCPR and F1@tau* excluded for Real (positive class << negative pool size)")

    dr = detection_rate(REAL_ATK_W20, REAL_MANIFEST, TAU_STAR)
    print()
    print_detection_rate_table(dr, TAU_STAR)

    out = {
        "protocol":          f"LOAO-replicated (50K benign pool, 40% test split, tau*={TAU_STAR})",
        "window_size":       20,
        "benign_pool_size":  int(len(b_probs)),
        "neg_test_size":     int(len(neg_test)),
        "tau_star":          TAU_STAR,
        "seed":              SEED,
        "loao_adduser":      loao,
        "real_adduser":      real_m,
        "detection_rate_by_family": dr,
    }
    out_path = _FAMILY_DIR / "artifacts" / "window_metrics_w20.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

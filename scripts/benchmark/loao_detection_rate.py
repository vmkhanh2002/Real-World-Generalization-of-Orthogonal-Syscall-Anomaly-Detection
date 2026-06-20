"""
LOAO Trace-Level Detection Rate — Recall-Only Analysis.

For each ADFA-LD attack family, computes what fraction of attack traces
and windows the Phase 5 meta-classifier flags at tau* — without any benign
pool.  This is a pure TPR measurement unaffected by class imbalance.

Protocol:
  - Source: artifacts/realtime_eval/window_scores.jsonl (full ADFA-LD scores)
  - Threshold: tau* = 0.401214 (Phase 5 XGBoost, from phase5 training)
  - Positive detection: trace has >= 1 window with phase5_prob >= tau*

Output:
  experiments/32bit/logs/loao_detection_rate.json
  experiments/32bit/logs/loao_detection_rate.txt
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

WINDOW_SCORES = PROJECT_ROOT / "artifacts" / "realtime_eval" / "window_scores.jsonl"
MANIFEST      = PROJECT_ROOT / "artifacts" / "realtime_eval" / "adfa_ld_realtime_full_manifest.json"
OUT_DIR       = PROJECT_ROOT / "experiments" / "32bit" / "logs"

TAU_STAR = 0.401214

ATTACK_FAMILIES = {"Adduser", "Hydra_FTP", "Hydra_SSH", "Java_Meterpreter", "Meterpreter", "Web_Shell"}


def build_file_to_family(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for e in manifest["entries"]:
        src = e.get("source_file", "")
        fam = e.get("target_preset") or e.get("source_group") or "Normal"
        if src:
            out[src.replace("\\", "/").lower()] = fam
    return out


def compute_detection_rates(
    jsonl_path: Path, file_to_family: dict[str, str], tau: float
) -> dict[str, dict]:
    families: dict[str, dict] = defaultdict(lambda: {
        "n_traces": 0, "detected_traces": 0,
        "total_windows": 0, "flagged_windows": 0,
        "per_trace_rates": [],
    })

    for line in jsonl_path.open(encoding="utf-8"):
        row = json.loads(line)
        key    = row["file"].replace("\\", "/").lower()
        family = file_to_family.get(key, "Unknown")
        if family not in ATTACK_FAMILIES:
            continue
        probs  = np.array(row.get("phase5_probs", []), dtype=np.float32)
        if len(probs) == 0:
            continue
        flagged = int((probs >= tau).sum())
        d = families[family]
        d["n_traces"]        += 1
        d["total_windows"]   += len(probs)
        d["flagged_windows"] += flagged
        d["per_trace_rates"].append(flagged / len(probs))
        if flagged >= 1:
            d["detected_traces"] += 1

    out = {}
    for fam in sorted(families):
        d     = families[fam]
        rates = np.array(d["per_trace_rates"])
        out[fam] = {
            "n_traces":         d["n_traces"],
            "detected_traces":  d["detected_traces"],
            "trace_dr":         round(d["detected_traces"] / d["n_traces"], 4),
            "total_windows":    d["total_windows"],
            "flagged_windows":  d["flagged_windows"],
            "window_dr":        round(d["flagged_windows"] / d["total_windows"], 4),
            "median_per_trace": round(float(np.median(rates)), 4),
            "p10_per_trace":    round(float(np.percentile(rates, 10)), 4),
            "p90_per_trace":    round(float(np.percentile(rates, 90)), 4),
        }
    return out


def format_table(dr: dict[str, dict], tau: float) -> str:
    lines = []
    W = 112
    lines.append("=" * W)
    lines.append("  LOAO ADFA-LD: Trace-Level Detection Rate (Recall-Only)")
    lines.append(f"  tau* = {tau}  |  Positive = >= 1 window flagged per trace")
    lines.append(f"  Source: artifacts/realtime_eval/window_scores.jsonl")
    lines.append("=" * W)
    hdr = (
        f"  {'Family':<22}  {'Traces':>7}  {'Detected':>9}  {'Trace-DR':>9}"
        f"  {'Windows':>9}  {'Flagged':>8}  {'Win-DR':>7}  {'Median%':>8}  {'P10%':>6}  {'P90%':>6}"
    )
    lines.append(hdr)
    lines.append("-" * W)

    tot = {"n_traces": 0, "detected_traces": 0, "total_windows": 0, "flagged_windows": 0}
    for fam, d in dr.items():
        lines.append(
            f"  {fam:<22}  {d['n_traces']:>7,}  {d['detected_traces']:>9,}  {d['trace_dr']:>8.1%}"
            f"  {d['total_windows']:>9,}  {d['flagged_windows']:>8,}  {d['window_dr']:>6.1%}"
            f"  {d['median_per_trace']:>7.1%}  {d['p10_per_trace']:>5.1%}  {d['p90_per_trace']:>5.1%}"
        )
        for k in tot:
            tot[k] += d[k]

    lines.append("-" * W)
    tdr = tot["detected_traces"] / tot["n_traces"]
    wdr = tot["flagged_windows"]  / tot["total_windows"]
    lines.append(
        f"  {'TOTAL':<22}  {tot['n_traces']:>7,}  {tot['detected_traces']:>9,}  {tdr:>8.1%}"
        f"  {tot['total_windows']:>9,}  {tot['flagged_windows']:>8,}  {wdr:>6.1%}"
        f"  {'':>8}  {'':>6}  {'':>6}"
    )
    lines.append("")
    lines.append("  Trace-DR  = % of attack traces with >= 1 window flagged (trace-level recall)")
    lines.append("  Win-DR    = % of all windows flagged across all traces (window-level recall)")
    lines.append("  Median%   = median per-trace window detection rate")
    lines.append("  P10%/P90% = 10th/90th percentile per-trace rate")
    return "\n".join(lines)


def main() -> None:
    print(f"Loading manifest: {MANIFEST}")
    file_to_family = build_file_to_family(MANIFEST)
    print(f"Loading scores:   {WINDOW_SCORES}")
    dr = compute_detection_rates(WINDOW_SCORES, file_to_family, TAU_STAR)

    table = format_table(dr, TAU_STAR)
    print()
    print(table)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_path = OUT_DIR / "loao_detection_rate.txt"
    txt_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved: {txt_path}")

    json_path = OUT_DIR / "loao_detection_rate.json"
    json_path.write_text(
        json.dumps({"tau_star": TAU_STAR, "families": dr}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

# real_attack_eval/reverse_shell/scripts/build_manifest.py
"""Scan ReverseShell_Real capture directory → artifacts/manifest.json.

Capture root resolution (in order):
  1. $CAPTURE_ROOT env var  (set for standalone use)
  2. <real_attack_eval>/../datasets  (default: works inside the main repo)
"""
import json
import os
from pathlib import Path

_HERE          = Path(__file__).resolve().parent
_REAL_ATK_ROOT = _HERE.parents[1]                    # real_attack_eval/
_FAMILY_DIR    = _HERE.parent                         # real_attack_eval/reverse_shell/
OUT_PATH       = _FAMILY_DIR / "artifacts" / "manifest.json"

_default_capture = _REAL_ATK_ROOT.parent / "datasets"
CAPTURE_BASE = Path(os.environ.get("CAPTURE_ROOT", str(_default_capture))) \
               / "reverse_shell_capture" / "adfa_ld_out" / "Attack_Data_Master"

entries = []
d = CAPTURE_BASE / "ReverseShell_Real"
if not d.exists():
    print(f"  WARNING: {d} not found")
else:
    for f in sorted(d.glob("*.txt")):
        try:
            source_file = str(f.resolve().relative_to(_REAL_ATK_ROOT.resolve()))
        except ValueError:
            source_file = str(f.resolve())
        entries.append({
            "sample_id":            f.stem,
            "intended_label":       "malware_targeted",
            "target_preset":        "ReverseShell_Real",
            "source_group":         "ReverseShell_Real",
            "source_file":          source_file,
            "expected_detect_file": source_file,
        })
    print(f"  ReverseShell_Real: {len(entries)} traces")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "schema_version": 2,
    "manifest_type":  "reverse_shell_real_attack_eval",
    "note":           "Victim-side real Reverse Shell traces (Python bind shell, bash subprocess strace)",
    "entries":        entries,
}, indent=2, ensure_ascii=False))
print(f"\nWrote {len(entries)} entries -> {OUT_PATH}")

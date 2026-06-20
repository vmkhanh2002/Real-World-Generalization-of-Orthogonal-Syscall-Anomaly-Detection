# real_attack_eval/hydra/scripts/build_manifest.py
"""Scan Hydra_FTP_Real + Hydra_SSH_Real capture directories → artifacts/manifest.json.

Capture root resolution (in order):
  1. $CAPTURE_ROOT env var  (set for standalone use)
  2. <real_attack_eval>/../datasets  (default: works inside the main repo)
"""
import json
import os
from pathlib import Path

_HERE          = Path(__file__).resolve().parent
_REAL_ATK_ROOT = _HERE.parents[1]                    # real_attack_eval/
_HYDRA_DIR     = _HERE.parent                         # real_attack_eval/hydra/
OUT_PATH       = _HYDRA_DIR / "artifacts" / "manifest.json"

_default_capture = _REAL_ATK_ROOT.parent / "datasets"
CAPTURE_BASE = Path(os.environ.get("CAPTURE_ROOT", str(_default_capture))) \
               / "hydra_capture" / "adfa_ld_out" / "Attack_Data_Master"

FAMILIES = [
    ("Hydra_FTP_Real", "Hydra_FTP_Real"),
    ("Hydra_SSH_Real", "Hydra_SSH_Real"),
]

entries = []
for dirname, preset in FAMILIES:
    d = CAPTURE_BASE / dirname
    if not d.exists():
        print(f"  WARNING: {d} not found, skipping")
        continue
    files = sorted(d.glob("*.txt"))
    for f in files:
        try:
            source_file = str(f.resolve().relative_to(_REAL_ATK_ROOT.resolve()))
        except ValueError:
            source_file = str(f.resolve())
        entries.append({
            "sample_id":            f.stem,
            "intended_label":       "malware_targeted",
            "target_preset":        preset,
            "source_group":         dirname,
            "source_file":          source_file,
            "expected_detect_file": source_file,
        })
    print(f"  {dirname}: {len(files)} traces")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "schema_version": 2,
    "manifest_type":  "hydra_real_attack_eval",
    "note":           "Victim-side real Hydra traces (vsftpd/sshd strace during attack)",
    "entries":        entries,
}, indent=2, ensure_ascii=False))
print(f"\nWrote {len(entries)} entries -> {OUT_PATH}")

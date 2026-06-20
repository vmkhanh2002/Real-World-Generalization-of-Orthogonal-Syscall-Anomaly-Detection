#!/usr/bin/env bash
# run_eval.sh -- Hydra eval: manifest → score traces → compute metrics
#
# Usage (from anywhere):
#   bash real_attack_eval/hydra/scripts/run_eval.sh
#
# Environment variables:
#   CAPTURE_ROOT  Directory containing <family>_capture/ subdirs.
#                 Default: sibling datasets/ dir (works inside the main repo).
#                 For standalone use: export CAPTURE_ROOT=/path/to/your/captures
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"   # real_attack_eval/ root
cd "$ROOT"

# Resolve project root: real_attack_eval/ is 2 levels below project root
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
unset MSYS_NO_PATHCONV 2>/dev/null || true
PYTHON="$PROJECT_ROOT/venv311/Scripts/python.exe"
[ -f "$PYTHON" ] || PYTHON="$(command -v python3 2>/dev/null || command -v python)"

: "${CAPTURE_ROOT:="$ROOT/../datasets"}"

echo "=== [Hydra] Build manifest ==="
"$PYTHON" hydra/scripts/build_manifest.py

echo ""
echo "=== [Hydra] Score traces (w=20, i386) ==="
# Scans both Hydra_FTP_Real/ and Hydra_SSH_Real/ via rglob
"$PYTHON" scripts/realtime_eval/export_window_scores.py \
    --dataset     32bit \
    --syscall-abi i386 \
    --input-dir   "$CAPTURE_ROOT/hydra_capture/adfa_ld_out/Attack_Data_Master" \
    --output-jsonl "hydra/artifacts/window_scores_w20.jsonl" \
    --window-size 20 \
    --stride      2

echo ""
echo "=== [Hydra] Compute metrics ==="
"$PYTHON" hydra/scripts/compute_metrics.py

#!/usr/bin/env bash
# run_eval.sh -- SQLMap eval: manifest → score traces → compute metrics
#
# Usage (from anywhere):
#   bash real_attack_eval/sqlmap/scripts/run_eval.sh
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

echo "=== [SQLMap] Build manifest ==="
"$PYTHON" sqlmap/scripts/build_manifest.py

echo ""
echo "=== [SQLMap] Score traces (w=20, i386) ==="
"$PYTHON" scripts/realtime_eval/export_window_scores.py \
    --dataset     32bit \
    --syscall-abi i386 \
    --input-dir   "$CAPTURE_ROOT/sqlmap_capture/adfa_ld_out/Attack_Data_Master/SQLMap_Real" \
    --output-jsonl "sqlmap/artifacts/window_scores_w20.jsonl" \
    --window-size 20 \
    --stride      2

echo ""
echo "=== [SQLMap] Compute metrics ==="
"$PYTHON" sqlmap/scripts/compute_metrics.py

#!/usr/bin/env bash
# run_all.sh -- Run evaluation for all attack families
#
# Usage (from real_attack_eval/ root):
#   bash run_all.sh [families...]
#
# Examples:
#   bash run_all.sh                                    # all 10 families
#   bash run_all.sh hydra meterpreter                  # selected families
#   bash run_all.sh reverse_shell sqlmap nikto         # novel families only
#
# Environment variables:
#   CAPTURE_ROOT  Parent directory containing <family>_capture/ subdirs.
#                 Default: sibling datasets/ dir (works inside the main repo).
#                 Standalone: export CAPTURE_ROOT=/path/to/captures
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

export CAPTURE_ROOT="${CAPTURE_ROOT:-"$HERE/../datasets"}"

ALL_FAMILIES=(hydra adduser web_shell meterpreter java_meterpreter reverse_shell sqlmap nikto shellshock cryptominer)
FAMILIES=("${@:-${ALL_FAMILIES[@]}}")

echo "======================================================"
echo "  Real-Attack Evaluation — $(date '+%Y-%m-%d %H:%M')"
echo "  CAPTURE_ROOT = $CAPTURE_ROOT"
echo "  Families     = ${FAMILIES[*]}"
echo "======================================================"
echo ""

for fam in "${FAMILIES[@]}"; do
    script="$HERE/$fam/scripts/run_eval.sh"
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found, skipping $fam"
        continue
    fi
    echo ">>> [$fam] starting"
    bash "$script"
    echo ">>> [$fam] done"
    echo ""
done

echo "======================================================"
echo "  ALL DONE"
echo "======================================================"

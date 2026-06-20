# Realtime Evaluation Module

## Overview
This module contains the canonical streaming evaluation pipeline for the project. The older file-level end-of-file detection workflow is now deprecated. Realtime evaluation answers a different operational question: it measures how well the system behaves while a trace is still actively unfolding.

## Directory Structure
```text
scripts/realtime_eval/
 defaults.py # Shared paths and configurations
 build_adfa_ld_realtime_manifest.py
 inference_backend.py # Shared model loading and per-window inference
 export_window_scores.py # Runs models on converted traces
 build_stream_episodes.py # Builds episodes from window array scores
 report_builders.py # Tools to build final metric reports
 evaluate_realtime_detect.py # Policy-gate metric runner
 utils.py # Shared utility functions (path resolution, sequence helpers)
```

## Dependencies
- `torch`, `numpy`, `argparse`
- Requires numeric trace inputs generated upstream by the `syscall_abi` module.

## Usage
### Quickstart
```powershell
python scripts/realtime_eval/export_window_scores.py `
 --dataset 32bit `
 --input-dir tests\syscall_abi\organized\i386\converted `
 --input-glob "*.name_to_number.txt" `
 --output-jsonl artifacts\realtime_eval\window_scores.jsonl

python scripts/realtime_eval/build_stream_episodes.py `
 --window-scores-jsonl artifacts\realtime_eval\window_scores.jsonl `
 --manifest artifacts\realtime_eval\adfa_ld_realtime_full_manifest.json `
 --output-json artifacts\realtime_eval\stream_episodes.json

python scripts/realtime_eval/evaluate_realtime_detect.py `
 --window-scores-jsonl artifacts\realtime_eval\window_scores.jsonl `
 --episodes-json artifacts\realtime_eval\stream_episodes.json `
 --policy-json artifacts\realtime_eval\POLICY.canonical.json `
 --output-json artifacts\realtime_eval\realtime_eval_report.json `
 --output-md artifacts\realtime_eval\realtime_eval_report.md
```

## Configuration
- Workspace configuration and CLI defaults live in `defaults.py`.
- Operational policies (e.g. onset calibration, hysteresis gating) are driven by JSON configurations like `POLICY.canonical.json`.

## Pipeline Integration
Realtime evaluation DOES NOT convert syscall names to numbers. Conversion must happen upstream via `scripts/syscall_abi/process_drop.py`.
**Data Flow:**
numeric trace to `export_window_scores.py` to `build_stream_episodes.py` to `evaluate_realtime_detect.py`

## Metrics
- Early attack detection effectiveness.
- Detection delay.
- Benign false-alert burden per episode and over time.
- Family-specific behavioral tracking under different operational policy gates.

## Limitations
- High OOV (out-of-vocabulary) rates can cause Path B and Path C anomaly scores to be underestimated because undefined syscalls are explicitly mapped to token 0 without triggering structural anomalies.
- The pipeline assumes purely numeric syscall traces; textual syscall names will cause failures.

## Changelog
- **v1**: Reverted file-level evaluation in favor of realtime-first metrics.
- **v1 (Phase 1)**: Corrected module structuring by shifting `__init__.py` behavior ensuring `pytest` resolves identical packages successfully. Introduced utility consolidation to `utils.py`.

# Realtime Evaluation Artifacts

This directory stores reference files and generated outputs for the realtime evaluation pipeline.

## What Lives Here

- protocol and schema references
- canonical policy JSON
- family onset calibration
- generated window-score exports
- generated episode manifests
- generated realtime reports

## Source Code vs Generated Outputs

- Source code lives in `scripts/realtime_eval/`
- Generated outputs live in `artifacts/realtime_eval/`

This split is intentional so the repo stays easier to navigate:

- `scripts/` answers "how does the pipeline run?"
- `artifacts/` answers "what did the pipeline produce?"

## Key Reference Files

- `PROTOCOL.md`
- `POLICY.canonical.json`
- `family_onset_calibration.full_100_manifest.json`
- `STREAM_SCHEMA.example.json`
- `REPORT_SCHEMA.example.json`

## Canonical Script Entry Points

```powershell
python scripts/realtime_eval/export_window_scores.py
python scripts/realtime_eval/build_stream_episodes.py
python scripts/realtime_eval/evaluate_realtime_detect.py
```

## Important Input Assumption

The realtime pipeline expects numeric syscall traces as input. It does not perform syscall name-to-number conversion internally.

If a trace still needs ABI-aware conversion, run:

```powershell
python scripts/syscall_abi/process_drop.py
```

before starting the realtime pipeline.

## Canonical Status

This directory is the canonical home for realtime manifests, the selected deployment policy, family onset calibration, and final reports.


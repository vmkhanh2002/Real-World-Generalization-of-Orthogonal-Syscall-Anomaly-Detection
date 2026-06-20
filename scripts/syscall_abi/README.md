# Syscall ABI Module

## Overview
This module contains the syscall ABI utilities used by trace conversion and detector benchmarking. It facilitates detecting raw string trace structures, interpreting syscalls into normalized architectures, and organizing mapped numeric inputs to be evaluated by the ML pipelines natively.

## Directory Structure
```text
scripts/syscall_abi/
 defaults.py # CLI defaults and workspace paths
 tables.py # JSON mapping loading and lookup
 detection.py # ABI auto-detection logic
 tool.py # Public CLI logic
 verify_dataset.py # Dataset-level ABI check wrapper
 process_drop.py # Automates trace batch conversion
 validate_organized.py # Validates organized conversion batches
 generate_synthetic_inbox.py # Generates labeled synthetic traces
 mappings/ # Canonical JSON ABI mapping tables
```

## Dependencies
- Standard Python built-ins (`argparse`, `json`, `pathlib`, `collections`).
- Relies implicitly on ABI JSON mapping arrays located in the `mappings/` subdirectory.

## Usage
### Diagnostics & Lookup
```powershell
python scripts/syscall_abi/tool.py detect --dataset-root datasets/32bit/ADFA-LD/ADFA-LD --top-k 20
python scripts/syscall_abi/tool.py lookup-name --abi i386 --name openat
python scripts/syscall_abi/tool.py lookup-number --dataset-root datasets/32bit/ADFA-LD/ADFA-LD --number 265
```

### Trace Batch Conversion
```powershell
python scripts/syscall_abi/process_drop.py
python scripts/syscall_abi/validate_organized.py
```

## Configuration
Local workspace defaults are explicitly centralized in `defaults.py`:
- inbox: `tests/syscall_abi/inbox`
- organized: `tests/syscall_abi/organized`
- default `top_k`: `20`
- default synthetic seed: `20260325`

## Pipeline Integration
Serves as the authoritative data ingestion and preprocessing stage. Raw inbox strings drop into an `inbox`, are mapped logically via `process_drop.py`, then emitted as numeric payloads (e.g., `*.name_to_number.txt`), enabling immediate ingestion by `scripts/realtime_eval/export_window_scores.py`.

## Metrics
This module also houses benchmarking tooling for mapping efficiency. Current `i386` synthetic baseline metrics (`synthetic_labeled_i386_round8_mixed_strict_seed20260325`):
- `precision = 0.968627`
- `recall = 0.988`
- `FPR = 0.032`
- `accuracy = 0.978`
Recall benchmarks scale smoothly per operational scenario (e.g., `exec_kill_chain = 1.0`, `ptrace_loader_chain = 1.0`).

## Limitations
- The detector can automatically resolve numeric mapping or semantic strings, but mixed files containing both will throw processing errors.
- **Note:** Missing structural dependencies and deferred unit test gap (Option B) are currently acknowledged. Testing leans on integration sanity (`test_dummy.py`) rather than explicit mocking.

## Changelog
- **v1**: Baseline ABI schema loading for i386, x86_64, arm, arm64.
- **v1 (Phase 1)**: Integrated placeholder structural unit testing to unblock automated CI tasks. Identified absence of complete centralized `utils.py` usage compared to `realtime_eval`, acknowledged as a pending structural consolidation task.

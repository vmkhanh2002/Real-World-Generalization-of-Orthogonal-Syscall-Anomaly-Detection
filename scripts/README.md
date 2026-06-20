# Scripts Directory

This folder contains the training, benchmark, inference, and documentation-refresh scripts for the project.

## High-Level Layout

```text
scripts/
+-- benchmark/ Benchmark and reviewer-response scripts
+-- optimization/ Historical sweep scripts
+-- realtime_eval/ Streaming / realtime evaluation pipeline
+-- syscall_abi/ ABI detection, mapping tables, lookups, and inbox batch conversion
+-- build_inference_bundle.py Package deployable Path B/C artifacts
+-- compare_path_bc_eval_protocols.py
+-- data_pipeline.py Phase 1 for 32bit ADFA-LD
+-- data_pipeline_64bit.py Phase 1 for 64bit DongTing
+-- export_paper_protocol_table.py
+-- export_phase5_benchmark_artifacts.py
+-- train_path_a.py
+-- train_path_b.py
+-- train_path_c.py
+-- train_phase5_fusion.py
+-- pipeline_config.py
+-- models.py
+-- model_metadata.py
+-- preprocess_dongting.py
+```

## Current 32bit Training Flow

> Pre-trained models are included in `models/32bit/` — retraining is optional.
> The ADFA-LD dataset (not in this repo) is required to retrain. Download from the
> [ADFA-LD project page](https://research.unsw.edu.au/projects/adfa-ids-datasets)
> and place at `datasets/32bit/ADFA-LD/ADFA-LD/`.
> Real-attack capture datasets are on Zenodo: [10.5281/zenodo.20776518](https://doi.org/10.5281/zenodo.20776518).

```bash
python scripts/data_pipeline.py
python scripts/train_path_a.py --dataset 32bit --n_strata 5 --min_per_stratum 200
python scripts/train_path_b.py --dataset 32bit --embed_dim 8 --fraction 0.2 --epochs 15
python scripts/train_path_c.py --dataset 32bit --hidden_dim 192 --fraction 0.5 --num_layers 1 --epochs 15
python scripts/train_phase5_fusion.py --dataset 32bit --max_windows_per_split 50000
```

The canonical evaluation workflow is now the realtime pipeline under `scripts/realtime_eval/`, with generated outputs stored under `artifacts/realtime_eval/`.

Current engineering-reference results from those scripts:

- Path A `SGD-OCSVM`: `AUC 0.7975`, `AUPR 0.3702`, `F1_calib 0.2892`
- Path B `CNN1D_AE`: `AUC 0.7822`, `F1 0.4599`
- Path C `GRU_Predictor`: `AUC 0.8151`, `F1 0.4313`
- Phase 5 `XGBoost`: `AUC 0.9122`, `AUPR 0.8979`, `F1 0.8514`

## Protocol Lock and Paper Export

```bash
python scripts/compare_path_bc_eval_protocols.py --dataset 32bit --paths both --outer_folds 5 --inner_folds 3 --batch_size 1024
python scripts/export_paper_protocol_table.py --dataset 32bit
```

Current paper-lock values:

- Path B nested CV: `AUC 0.7813 +/- 0.0159`, `F1 0.4581 +/- 0.0355`
- Path C nested CV: `AUC 0.8140 +/- 0.0137`, `F1 0.4311 +/- 0.0248`

## Deployment and Verification Utilities

```bash
python scripts/syscall_abi/verify_dataset.py --dataset-root datasets/32bit/ADFA-LD/ADFA-LD --output-json experiments/32bit/logs/verify_dataset_syscall_abi.json
python scripts/syscall_abi/tool.py detect --dataset-root datasets/32bit/ADFA-LD/ADFA-LD --top-k 20
python scripts/syscall_abi/tool.py lookup-number --abi i386 --number 295
python scripts/syscall_abi/process_drop.py
python scripts/realtime_eval/build_adfa_ld_realtime_manifest.py
python scripts/realtime_eval/export_window_scores.py --dataset 32bit
python scripts/realtime_eval/build_stream_episodes.py
python scripts/realtime_eval/evaluate_realtime_detect.py
python scripts/build_inference_bundle.py --dataset 32bit --paths both --selection_source nested_majority --calib_ratio 0.5 --batch_size 1024
```

These commands support the deployment-facing claims:

- ABI verification (`i386` ranked first for `32bit`)
- direct name/number syscall lookups
- inbox-based batch conversion into `tests/syscall_abi/organized/`
- realtime manifest construction and policy-gate evaluation
- inference bundle packaging

## Syscall ABI Tooling

The ABI-specific conversion logic now lives entirely under `scripts/syscall_abi/`:

- `defaults.py`: shared workspace paths and CLI defaults
- `tool.py`: direct CLI for `detect`, `lookup-name`, and `lookup-number`
- `process_drop.py`: batch processor for dropped `.txt` files in `tests/syscall_abi/inbox/`
- `verify_dataset.py`: wrapper CLI for ABI ranking reports
- `tables.py`: canonical registry loader for the 4 supported mapping JSON files
- `mappings/`: official-only syscall tables for `i386`, `x86_64`, `arm`, and `arm64_asm_generic`

Preferred batch workflow:

```text
tests/syscall_abi/inbox/
 i386/
 x86_64/
 arm/
 arm64/
```

- Drop `number to syscall` files at the inbox root for auto-detection.
- Drop `syscall to number` files inside `inbox/<abi>/` for deterministic conversion.
- Converted files and reports are written under `tests/syscall_abi/organized/<abi>/`.

See [scripts/syscall_abi/README.md](./syscall_abi/README.md) for the detailed module reference.

## Hardcoded Audit

The repository is not fully hardcoded-free across all scripts.

There are two distinct cases:

- Generic CLIs:
 These are expected to take inputs from arguments or shared defaults. The `scripts/syscall_abi/` workspace now centralizes its local defaults in `scripts/syscall_abi/defaults.py`.
- Dataset-specific pipelines:
 Scripts such as `data_pipeline.py`, `data_pipeline_64bit.py`, and `preprocess_dongting.py` are intentionally tied to one dataset family or one preprocessing contract. Their names already reflect that scope.

For the publishable syscall ABI flow, the remaining defaults are limited to workspace paths and convenience values. Inference model selection is resolved from the phase 5 manifest rather than from hidden hardcoded artifact names inside the syscall ABI scripts.

## Benchmarks

See [scripts/benchmark/README.md](./benchmark/README.md) for the refreshed `32bit` benchmark set.

## Historical Sweeps

Path-level training sweeps remain archived under [scripts/optimization/README.md](./optimization/README.md).
Realtime policy sweeps have been retired in favor of the fixed canonical operating point stored in `artifacts/realtime_eval/POLICY.canonical.json`.



# Real-World Generalization of Orthogonal Syscall Anomaly Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20776518.svg)](https://doi.org/10.5281/zenodo.20776518)

This project builds a syscall-sequence anomaly detection pipeline. It includes three component detection paths, a Phase 5 fusion model, ADFA-LD benchmarking, realtime evaluation, and real-world attack evaluation across ten attack families.

> Run capture and attack workflows only in an authorized lab environment. The capture scripts in this repository target an internal Docker lab setup.

## Repository Layout

```text
.
+-- artifacts/             # Realtime evaluation outputs, policies, reports, window scores
+-- models/                # Pre-trained Path A/B/C checkpoints, Phase 5 models, inference bundles
+-- real_attack_eval/      # Real attack evaluations by family (Hydra, Meterpreter, etc.)
+-- scripts/               # Training, benchmarking, realtime evaluation, syscall ABI tools
+-- requirements.txt       # Python dependencies
```

Key modules:

- `scripts/data_pipeline.py`: prepares the 32-bit ADFA-LD dataset.
- `scripts/train_path_a.py`: Path A, frequency/rarity modeling with TF-IDF and anomaly detection.
- `scripts/train_path_b.py`: Path B, topology/geometry modeling with a CNN autoencoder.
- `scripts/train_path_c.py`: Path C, chronology/grammar modeling with a GRU predictor.
- `scripts/train_phase5_fusion.py`: Phase 5 fusion model, defaulting to XGBoost.
- `scripts/realtime_eval/`: streaming/realtime evaluation pipeline.
- `scripts/syscall_abi/`: ABI detection, syscall name/number mapping, and trace conversion.
- `real_attack_eval/hydra/`: real Hydra FTP/SSH evaluation.

## Requirements

- Python 3.11 is recommended.
- Git.
- Docker Desktop or Docker Engine for the Hydra capture lab.
- CUDA 12.1 GPU support is pinned in `requirements.txt`. CPU-only runs are supported with a separate PyTorch CPU install.

## Environment Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

For CPU-only usage:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Then remove or comment the `torch==...+cu121`, `torchaudio==...+cu121`, and `torchvision==...+cu121` lines in `requirements.txt`, then run:

```powershell
pip install -r requirements.txt
```

## Datasets

Attack capture datasets (raw strace logs + ADFA-LD formatted traces) are published on Zenodo:

> **DOI: [10.5281/zenodo.20776518](https://doi.org/10.5281/zenodo.20776518)**

Ten attack families are available as individual archives:
`adduser`, `cryptominer`, `hydra` (FTP+SSH), `java_meterpreter`, `meterpreter`, `nikto`, `reverse_shell`, `shellshock`, `sqlmap`, `webshell`.

Each archive contains `raw_strace/`, `adfa_ld_out/` (ADFA-LD i386 format), and `docker/` (Docker Compose lab setup for re-capture).

The **ADFA-LD training dataset** (benign + original attack families) is a separate academic release — download it from the [ADFA-LD project page](https://research.unsw.edu.au/projects/adfa-ids-datasets) and place it at `datasets/32bit/ADFA-LD/ADFA-LD/`.

## Pre-trained Models

All models are included in `models/32bit/` — **no retraining is required** to reproduce the paper results.

| Directory | Contents |
|-----------|----------|
| `models/32bit/path_a/` | TF-IDF vectorizer, PCA, SGD-OCSVM, IForest, LOF, HBOS |
| `models/32bit/path_b/` | CNN autoencoder and variant checkpoints |
| `models/32bit/path_c/` | GRU predictor and variant checkpoints |
| `models/32bit/phase5_fusion/` | XGBoost meta-classifier and baseline variants |
| `models/32bit/inference/` | Packaged inference bundles for Path B and C |

## Quick Reproduction

Set up the environment (see [Environment Setup](#environment-setup)), then run:

```powershell
# Reproduce all real-attack evaluation reports
python real_attack_eval/hydra/scripts/build_manifest.py
python real_attack_eval/hydra/scripts/compute_metrics.py

# Reproduce ADFA-LD benchmark metrics
python scripts/export_phase5_benchmark_artifacts.py --dataset 32bit
python scripts/benchmark/benchmark_trace_metrics.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_latency.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_loao_generalization.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_orthogonality.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_scale_harmonization.py --dataset 32bit
```

> Benchmarks require the ADFA-LD dataset. See [Training Pipeline](#training-pipeline) for dataset setup.

## Training Pipeline

Retrain all models from scratch using the ADFA-LD dataset (optional — pre-trained models are already included).

Canonical training flow for the `32bit` dataset:

```powershell
python scripts/data_pipeline.py
python scripts/train_path_a.py --dataset 32bit --n_strata 5 --min_per_stratum 200
python scripts/train_path_b.py --dataset 32bit --embed_dim 8 --fraction 0.2 --epochs 15
python scripts/train_path_c.py --dataset 32bit --hidden_dim 192 --fraction 0.5 --num_layers 1 --epochs 15
python scripts/train_phase5_fusion.py --dataset 32bit --max_windows_per_split 50000
```

Main outputs:

- `models/32bit/path_a/`
- `models/32bit/path_b/`
- `models/32bit/path_c/`
- `models/32bit/phase5_fusion/`
- `experiments/32bit/logs/`

## Benchmarks

Re-run individual benchmark scripts against the pre-trained models:

```powershell
python scripts/export_phase5_benchmark_artifacts.py --dataset 32bit
python scripts/benchmark/benchmark_trace_metrics.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_latency.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_loao_generalization.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_orthogonality.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_scale_harmonization.py --dataset 32bit
```

See `scripts/benchmark/README.md` for benchmark details.

## Realtime Evaluation

Realtime evaluation measures how the detector behaves while a trace is still unfolding. This differs from file-level classification at the end of a trace.

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

Realtime evaluation expects numeric syscall traces. If you have syscall-name traces, convert them first with `scripts/syscall_abi/`.

## Syscall Name/Number Conversion

Detect the likely ABI:

```powershell
python scripts/syscall_abi/tool.py detect --dataset-root datasets/32bit/ADFA-LD/ADFA-LD --top-k 20
```

Look up syscalls:

```powershell
python scripts/syscall_abi/tool.py lookup-number --abi i386 --number 295
python scripts/syscall_abi/tool.py lookup-name --abi i386 --name open
```

Batch conversion:

```powershell
python scripts/syscall_abi/process_drop.py
```

Default batch workflow:

```text
tests/syscall_abi/inbox/              # Drop files here
tests/syscall_abi/inbox/i386/         # Deterministic conversion by ABI
tests/syscall_abi/organized/i386/     # Converted outputs
```

See `scripts/syscall_abi/README.md` for details.

## Hydra Real Attack Evaluation

The Hydra real attack evaluation uses victim-side `vsftpd` and `sshd` traces converted to ADFA-LD i386 syscall numbers.

Rebuild the manifest and metrics:

```powershell
python real_attack_eval/hydra/scripts/build_manifest.py
python real_attack_eval/hydra/scripts/compute_metrics.py
```

To rescore FTP/SSH trace windows:

```powershell
python scripts/realtime_eval/export_window_scores.py `
  --input-dir datasets/hydra_capture/adfa_ld_out/Attack_Data_Master/Hydra_FTP_Real `
  --output-jsonl real_attack_eval/hydra/artifacts/window_scores_ftp.jsonl

python scripts/realtime_eval/export_window_scores.py `
  --input-dir datasets/hydra_capture/adfa_ld_out/Attack_Data_Master/Hydra_SSH_Real `
  --output-jsonl real_attack_eval/hydra/artifacts/window_scores_ssh.jsonl
```

Then rebuild the reports:

```powershell
python real_attack_eval/hydra/scripts/build_manifest.py
python real_attack_eval/hydra/scripts/compute_metrics.py
```

See `real_attack_eval/README.md` and `real_attack_eval/hydra/README.md` for details.

## Hydra Lab Capture With Docker

Use this only in an authorized lab environment. The script creates internal FTP/SSH targets, runs Hydra from a separate container, attaches `strace` to the victim-side daemon, and parses logs into ADFA-LD format.

From Git Bash, WSL, or another Bash-capable shell:

```bash
bash datasets/hydra_capture/run_all.sh 40 50
```

Arguments:

- `40`: number of SSH runs.
- `50`: number of FTP runs.

Outputs:

- Raw strace logs: `datasets/hydra_capture/raw_strace/`
- ADFA-LD formatted traces: `datasets/hydra_capture/adfa_ld_out/Attack_Data_Master/`

Docker Compose file:

```text
datasets/hydra_capture/docker/docker-compose.yml
```

## Build an Inference Bundle

Package inference artifacts for Path B/C:

```powershell
python scripts/build_inference_bundle.py `
  --dataset 32bit `
  --paths both `
  --selection_source nested_majority `
  --calib_ratio 0.5 `
  --batch_size 1024
```

Outputs:

- `models/32bit/inference/`
- `experiments/32bit/logs/inference_bundle_summary.json`

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` and verify dependencies are installed.
- PyTorch CUDA wheel errors: add `--extra-index-url https://download.pytorch.org/whl/cu121` when installing requirements.
- CPU-only setup: do not use the `+cu121` dependency lines in `requirements.txt`.
- Realtime evaluation fails on syscall-name input: convert to numeric syscall traces with `scripts/syscall_abi/process_drop.py`.
- Docker capture fails on Windows Git Bash: the script sets `MSYS_NO_PATHCONV=1`; if issues persist, run it from WSL.
- Missing `models/32bit/phase5_fusion/` or other model files: run `git clone` again — model checkpoints are stored directly in the repository as regular files.
- Missing `datasets/`: download from [Zenodo DOI 10.5281/zenodo.20776518](https://doi.org/10.5281/zenodo.20776518) and extract each archive under `datasets/`.

## Additional Documentation

- `scripts/README.md`
- `scripts/realtime_eval/README.md`
- `scripts/benchmark/README.md`
- `scripts/syscall_abi/README.md`
- `real_attack_eval/README.md`
- `real_attack_eval/hydra/README.md`

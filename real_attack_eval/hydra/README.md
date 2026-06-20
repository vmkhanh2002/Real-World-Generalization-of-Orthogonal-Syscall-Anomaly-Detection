# Hydra Real-Attack Evaluation

This folder evaluates victim-side syscall traces collected during live Hydra brute-force attacks. FTP and SSH are kept in the same capture family, but the results are reported separately because vsftpd and OpenSSH produce different syscall profiles.

## Data

Pre-captured Hydra FTP and SSH traces are available on Zenodo:

> **DOI: [10.5281/zenodo.20776518](https://doi.org/10.5281/zenodo.20776518)** — download `hydra_capture.zip`

Extract to `datasets/hydra_capture/`. The archive includes `raw_strace/`, `adfa_ld_out/` (ADFA-LD i386 format), and `docker/` for re-capture in an authorized lab environment.

## Capture Setup

| Item | FTP | SSH |
|---|---:|---:|
| Target service | vsftpd | OpenSSH |
| Capture method | `strace -f` on victim worker | `strace -f` on victim worker |
| Trace format | ADFA-LD-style i386 syscall numbers | ADFA-LD-style i386 syscall numbers |
| Traces | 15,272 | 5,360 |
| Attack windows | 358,565 | 2,738,916 |
| Scenario count | 1 | 1 |

## Latest Metrics

Protocol: 50,000 ADFA-LD benign windows, 40% grouped test split, seed 42, window size 20, stride 2, `tau* = 0.401214`.

| Source | AUC | AUCPR(1:1) | TPR@FPR1% | TPR@FPR0.1% | Trace-DR | Window-DR |
|---|---:|---:|---:|---:|---:|---:|
| LOAO FTP paper row | 0.9112 | 0.8786 | 0.0909 | 0.0092 | n/a | n/a |
| Real Hydra FTP | 0.8684 | 0.7945 | 0.0638 | 0.0000 | 100.0% | 93.8% |
| LOAO SSH paper row | 0.8970 | 0.8694 | 0.1366 | 0.0094 | n/a | n/a |
| Real Hydra SSH | 0.7451 | 0.6460 | 0.0035 | 0.0000 | 100.0% | 58.6% |

Trace-level totals:

| Family | Traces | Detected | Windows | Flagged windows | Median per-trace | P10 per-trace |
|---|---:|---:|---:|---:|---:|---:|
| Hydra FTP | 15,272 | 15,272 | 358,565 | 336,336 | 100.0% | 78.8% |
| Hydra SSH | 5,360 | 5,360 | 2,738,916 | 1,603,741 | 57.9% | 49.5% |
| Total | 20,632 | 20,632 | 3,097,481 | 1,940,077 | n/a | n/a |

## Interpretation

FTP transfers better than SSH. FTP traces are shorter and have denser attack-specific behavior. SSH traces contain modern OpenSSH key-exchange, crypto, and file-handling activity, so many windows look closer to benign ADFA-LD behavior.

The important distinction is trace-level versus window-level behavior. Hydra has 100% Trace-DR for both protocols, but SSH has much weaker Window-DR and low-FPR TPR. This means the detector usually flags the attack session, but it does not classify every SSH window as anomalous.

## Files

| File | Purpose |
|---|---|
| `scripts/build_manifest.py` | Builds `artifacts/manifest.json` for FTP and SSH traces |
| `scripts/run_eval.sh` | Runs manifest build, scoring, and metric computation |
| `scripts/compute_metrics.py` | Computes AUC, AUCPR(1:1), low-FPR TPR, and detection rates |
| `artifacts/manifest.json` | Current file manifest |
| `artifacts/window_metrics_w20.json` | Source JSON for the metrics above |
| `artifacts/RESULTS.md` | Generated human-readable result table |
| `report.md` | Hydra-specific analysis note |

## Run

```bash
bash real_attack_eval/hydra/scripts/run_eval.sh
```

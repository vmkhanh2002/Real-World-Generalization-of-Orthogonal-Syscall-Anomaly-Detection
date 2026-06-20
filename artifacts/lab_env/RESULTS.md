# Lab Environment Results

> **Scope:** ADFA-LD 32-bit dataset only. All results here are from controlled lab conditions using the standard ADFA-LD corpus. Do NOT mix with realtime evaluation numbers — see `artifacts/realtime_eval/` for those.

---

## Benchmark 1 — LOAO Generalization (Paper Table 5)

**Protocol:** Leave-One-Attack-Out cross-validation. One attack family held out at a time; meta-classifier trained on remaining families. Window-level classification.

| Config | Value |
|--------|-------|
| Dataset | ADFA-LD 32-bit |
| window_size | **20** |
| stride | 2 |
| Meta-classifier | XGBoost (tau\* = 0.439345) |
| Metric | AUC / AUCPR / F1 (window-level) |
| Source | `experiments/32bit/logs/loao_generalization_results.txt` |

### Per-Family Results

| Held-Out Family | AUC | AUCPR | F1 |
|----------------|-----|-------|-----|
| Adduser | 0.9048 | 0.6891 | 0.7093 |
| Hydra FTP | 0.9078 | 0.7311 | 0.7268 |
| Hydra SSH | 0.8979 | 0.7942 | 0.7745 |
| Java Meterpreter | 0.9159 | 0.7732 | 0.7783 |
| Meterpreter | 0.9128 | 0.6988 | 0.6491 |
| Web Shell | 0.9046 | 0.7757 | 0.7456 |
| **Cross-Val Avg** | **0.9073** | **0.7437** | **0.7306** |

---

## Benchmark 2 — Phase 5 Fusion (Held-Out Test Set)

**Protocol:** Single train/val/test split. Meta-classifier trained on all families. Window-level classification against held-out test windows.

| Config | Value |
|--------|-------|
| Dataset | ADFA-LD 32-bit |
| window_size | 20 |
| stride | 2 |
| Source | `experiments/32bit/logs/phase5_fusion_results.json` |

### Meta-Classifier Comparison (Test Set)

| Model | AUC | AUCPR | F1 | Precision | Recall | tau |
|-------|-----|-------|----|-----------|--------|-----|
| XGBoost (**selected**) | **0.9113** | **0.8950** | **0.8523** | 0.8044 | 0.9063 | 0.401214 |
| Random Forest | 0.8877 | 0.8665 | 0.8420 | 0.7744 | 0.9226 | 0.398545 |
| SVM | 0.9004 | 0.8799 | 0.8379 | 0.7816 | 0.9029 | 0.341122 |
| MLP | 0.8952 | 0.8444 | 0.8402 | 0.7905 | 0.8965 | 0.407780 |
| Logistic Regression | 0.8503 | 0.7787 | 0.8031 | 0.7344 | 0.8859 | 0.495211 |

### Component Path Metrics (Phase 5 Individual Paths)

Source: `experiments/32bit/logs/phase5_individual_path_metrics.json`

| Path | Model | AUC | Notes |
|------|-------|-----|-------|
| Path A | SGD-OCSVM | — | see path_a_results.json |
| Path B | CNN1D_AE | — | see path_b_results.json |
| Path C | GRU_Predictor | AUC 0.8151 | hidden_dim=192, num_layers=1, fraction=0.5 |

---

## Paper-Locked Nested Protocol

**Protocol:** Nested cross-validation with trace-level grouping (no window leakage across traces).

| Metric | Value |
|--------|-------|
| AUC | 0.8140 ± 0.0137 |
| F1 | 0.4311 ± 0.0248 |
| Source | `experiments/32bit/logs/paper_nested_protocol_table.json` |

---

## Key Artifact Files

| File | Description |
|------|-------------|
| `experiments/32bit/logs/loao_generalization_results.txt` | Raw LOAO benchmark output (Paper Table 5) |
| `experiments/32bit/logs/phase5_fusion_results.json` | All meta-classifier comparison results |
| `experiments/32bit/logs/paper_nested_protocol_table.json` | Nested CV paper lock |
| `experiments/32bit/logs/paper_nested_protocol_table.md` | Human-readable nested CV table |
| `experiments/32bit/logs/path_a_results.json` | Path A sweep results |
| `experiments/32bit/logs/path_b_results.json` | Path B sweep results |
| `experiments/32bit/logs/path_c_results.json` | Path C sweep results (active model) |

---

## What's NOT Here

- **Realtime episode-level evaluation** → see `artifacts/realtime_eval/RESULTS.md` (window_size=32, hysteresis policy)
- **Hydra real-attack evaluation** → pending (see `docs/plans/2026-05-19-hydra-real-trace-collection.md`)

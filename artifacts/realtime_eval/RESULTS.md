# Realtime Evaluation Results

> **Scope:** Realtime episode-level detection on ADFA-LD 32-bit dataset (ADFA-LD corpus, simulated streaming). Do NOT mix with lab env window-level metrics — see `artifacts/lab_env/RESULTS.md` for those.

---

## Benchmark — Realtime Episode-Level Detection (Baseline)

**Protocol:** Sliding-window streaming with rolling hysteresis policy. Detection is episode-level (did the model raise an alert within the episode?), not window-level.

| Config | Value |
|--------|-------|
| Dataset | ADFA-LD 32-bit |
| window_size | **32** |
| stride | 2 |
| syscall_abi | i386 |
| Policy | `rolling_hysteresis_canonical_v1` |
| rolling_window | 25 |
| trigger threshold (rolling_mean_prob) | ≥ 0.60 |
| trigger threshold (rolling_positive_rate) | ≥ 0.60 |
| consecutive_positive | ≥ 3 |
| phase5_tau | 0.401214 (XGBoost) |
| Generated | 2026-04-12 |
| Source | `artifacts/realtime_eval/realtime_eval_report.full_100_manifest.json` |

### Overall Results

| Metric | Value |
|--------|-------|
| Episodes evaluated | 5118 |
| Attack episodes | 746 |
| Benign episodes | 4372 |
| **Episode recall** | **0.7895** |
| Median detection delay (windows) | 0.00 |
| P90 detection delay (windows) | 0.00 |
| False alert rate / benign episode | 0.2710 |
| False alert windows / 1000 benign | 191.10 |

### Per-Family Recall

| Family | Attack Episodes | Recall | Recall≤5w | Recall≤10w | Recall≤20w | Median Delay |
|--------|----------------|--------|-----------|------------|------------|--------------|
| Adduser | 91 | 0.8462 | 0.8352 | 0.8352 | 0.8352 | 0.00 |
| Hydra FTP | 162 | 0.7593 | 0.6852 | 0.6975 | 0.7346 | 0.00 |
| Hydra SSH | 176 | 0.6761 | 0.6307 | 0.6477 | 0.6648 | 0.00 |
| Java Meterpreter | 124 | 0.9113 | 0.8952 | 0.8952 | 0.9032 | 0.00 |
| Meterpreter | 75 | 0.7733 | 0.7600 | 0.7600 | 0.7733 | 0.00 |
| Web Shell | 118 | 0.8390 | 0.8136 | 0.8136 | 0.8220 | 0.00 |

---

## Why window_size=32 Here (Not 20)

The realtime pipeline uses **window_size=32** deliberately — this is a different benchmark from LOAO. The larger window provides more context per inference call in the streaming setting. The lab env LOAO benchmark uses window_size=20 to match the training pipeline default.

| Benchmark | window_size | Metric type | Source |
|-----------|-------------|-------------|--------|
| LOAO (lab) | 20 | AUC / AUCPR / F1 (window-level) | `artifacts/lab_env/RESULTS.md` |
| Realtime (this file) | 32 | Episode recall + detection delay | this file |

---

## Pending: Hydra Real-Attack Evaluation

The numbers above are from ADFA-LD corpus replays (lab traces). A separate evaluation using **real Hydra attack syscall traces** captured in a live environment is planned.

- Plan: `docs/plans/2026-05-19-hydra-real-trace-collection.md`
- Output will go to: `artifacts/hydra_real_eval/`
- Target: compare real Hydra recall vs. corpus Hydra recall (FTP: 0.7593, SSH: 0.6761)

---

## Key Artifact Files

| File | Description |
|------|-------------|
| `realtime_eval_report.full_100_manifest.json` | Full JSON report (100% manifest) |
| `realtime_eval_report.full_100_manifest.md` | Human-readable report |
| `stream_episodes.full_100_manifest.json` | Episode stream manifest |
| `window_scores.full_100_manifest.jsonl` | Per-window scores (5118 lines) |
| `family_onset_calibration.full_100_manifest.json` | Family onset offsets |
| `POLICY.canonical.json` | Canonical hysteresis policy spec |
| `PROTOCOL.md` | Full protocol documentation |

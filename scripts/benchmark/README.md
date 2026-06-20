# Benchmark Scripts

This directory contains the scripts used to regenerate the current `32bit` benchmark and analysis artifacts.

## Main Commands

```bash
python scripts/export_phase5_benchmark_artifacts.py --dataset 32bit
python scripts/benchmark/benchmark_trace_metrics.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_latency.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_loao_generalization.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_orthogonality.py --dataset 32bit
python scripts/benchmark/reviewer_benchmark_scale_harmonization.py --dataset 32bit
```

## Current 32bit Outputs

- `experiments/32bit/logs/benchmark_iso_vs_xgb.json`
- `experiments/32bit/logs/benchmark_unsupervised_baselines.json`
- `experiments/32bit/logs/ablation_unsupervised_fusion.json`
- `experiments/32bit/logs/trace_level_metrics.json`
- `experiments/32bit/logs/trace_attrition_metrics.json`
- `experiments/32bit/logs/score_sensitivity_stats.json`
- `experiments/32bit/logs/score_distributions.png`
- `experiments/32bit/logs/scale_harmonization_review.json`
- `experiments/32bit/logs/loao_generalization_results.txt`
- `experiments/32bit/orthogonality_analysis.png`

## Current 32bit Highlights

- XGBoost vs Isolation Forest: `AUC 0.9122 vs 0.8107`, `F1 0.8514 vs 0.7937`
- Unsupervised baselines: `CDF+Max AUC 0.6818 / F1 0.7644`, `GMM AUC 0.5851 / F1 0.7379`
- Score-space ablation: `IF F1 0.2045`, `GMM F1 0.1500`
- Trace attrition: `0 / 5951`
- Trace metrics: strict `TPR 1.0000`, `FPR 0.6977`; majority `TPR 0.9804`, `FPR 0.5116`
- Latency: Path A `0.0295`, Path B `0.0140`, Path C `0.1767`, total `0.2217 ms/window`
- LOAO average: `AUC 0.9073`, `AUCPR 0.7437`, `F1 0.7306`
- Scale heterogeneity ratio: `553.3949`
- Orthogonality correlations: `A-B 0.436`, `B-C 0.457`, `A-C 0.429`, synergy `29568`

## Notes

- These benchmarks load the current selected stack from `models/<dataset>/phase5_fusion/meta_best_model_manifest.json` when available.
- `score_distributions.png` and `orthogonality_analysis.png` are the figures mirrored into `figures/` and `latex/*/figures/` for the write-up refresh.
- If `models/32bit/` files appear missing, re-clone the repository — model checkpoints are stored as regular git files and download automatically with `git clone`.
- Retraining from scratch requires the ADFA-LD dataset: [ADFA-LD project page](https://research.unsw.edu.au/projects/adfa-ids-datasets).

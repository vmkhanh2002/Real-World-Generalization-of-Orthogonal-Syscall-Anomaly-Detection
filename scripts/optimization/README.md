# Optimization Scripts

This directory preserves the historical sweep scripts used to identify the current path defaults.

## Purpose

These scripts are not required for the standard training or benchmark refresh. They exist for:

- sweep provenance
- configuration archaeology
- optional re-exploration of archived search spaces

## Current Status

| Path | Script(s) | Current selection / latest 32-bit reference |
|---|---|---|
| Path A | `optimize_path_a.py` | `SGD-OCSVM`, `word(1,2)`, `max_features=1000`, `PCA(100)`, `nu=0.01`, holdout `AUC 0.7975`, `F1 0.2892` |
| Path B | `optimize_path_b.py` | `CNN1D_AE`, `embed_dim=8`, `fraction=0.2`, holdout `AUC 0.7822`, nested paper-lock `AUC 0.7813 +/- 0.0159`, `F1 0.4581 +/- 0.0355` |
| Path C | `optimize_path_c.py`, `optimize_path_c_v2.py` | active holdout selection `GRU_Predictor`, `hidden_dim=192`, `fraction=0.5`, holdout `AUC 0.8151`, `F1 0.4313`, nested paper-lock `AUC 0.8140 +/- 0.0137`, `F1 0.4311 +/- 0.0248` |

## Important Nuance for Path C

- The current engineering log still shows a high holdout F1 for `CBOW_Predictor`.
- The write-up and paper now lock Path C to the leak-safe nested trace-level protocol.
- Under that protocol, `GRU` is selected in all `5/5` outer folds and remains the active Path C model.

## Outputs Referenced During the Refresh

- Active benchmark outputs:
- `experiments/32bit/logs/path_a_results.json`
- `experiments/32bit/logs/path_b_topology_results.json`
- `experiments/32bit/logs/path_c_results.json`
- `experiments/32bit/logs/paper_nested_protocol_table.json`
- Active metadata mirrors:
- `models/32bit/path_a/path_a_best_config_per_model_type.json`
- `models/32bit/path_b/path_b_best_config_per_model_type.json`
- `models/32bit/path_c/path_c_best_config_per_model_type.json`
- Historical sweep provenance:
- `experiments/32bit/logs/path_a_v7_optimization.json`
- `experiments/32bit/logs/path_b_sweep_results.json`
- `experiments/32bit/logs/path_c_sweep_results.json`
- `experiments/32bit/logs/path_c_v2_sweep_results.json`

## Re-run Warning

These scripts are much slower than the main training path and are not needed for routine reproduction:

```bash
python scripts/optimization/optimize_path_a.py --dataset 32bit
python scripts/optimization/optimize_path_b.py --dataset 32bit
python scripts/optimization/optimize_path_c_v2.py --dataset 32bit
```

Use `run_all_paths.bat` or `run_all_paths.sh` for the standard refreshed pipeline.


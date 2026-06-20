"""
Export loadable Phase 5 benchmark artifacts outside scripts/benchmark.

This script is the only place that trains score-space baseline models for the
benchmark suite. The benchmark scripts under scripts/benchmark must remain
load-only and consume the artifacts produced here.
"""

import json
import os
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
BENCHMARK_ROOT = SCRIPTS_ROOT / "benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_common import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_path_b_model,
    load_path_c_model,
    load_selected_path_a_bundle,
    resolve_phase5_baseline_dir,
    resolve_phase5_selection,
    resolve_phase5_z_eval_path,
    resolve_phase5_z_splits_path,
    score_path_a_bundle,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ABLATION_IF_ARTIFACT = "ablation_isolation_forest"
ABLATION_GMM_ARTIFACT = "ablation_gmm"
UNSUP_CDF_MAX_ARTIFACT = "unsupervised_cdf_max"
UNSUP_GMM_ARTIFACT = "unsupervised_gmm"
ISO_VS_XGB_IF_ARTIFACT = "iso_vs_xgb_isolation_forest"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def fast_sample(matrix, trace_ids, size=50000):
    if len(matrix) <= size:
        return matrix, trace_ids

    unique_traces = np.unique(trace_ids)
    rng = np.random.RandomState(42)
    shuffled_traces = rng.permutation(unique_traces)

    selected_mask = np.zeros(len(matrix), dtype=bool)
    current_size = 0

    for trace_id in shuffled_traces:
        trace_mask = trace_ids == trace_id
        trace_size = int(np.sum(trace_mask))
        if current_size + trace_size <= size:
            selected_mask |= trace_mask
            current_size += trace_size
            continue

        remainder = size - current_size
        idx_of_trace = np.where(trace_mask)[0]
        selected_mask[idx_of_trace[:remainder]] = True
        break

    return matrix[selected_mask], trace_ids[selected_mask]


def get_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    return float(threshold), float(f1_scores[best_idx])


def get_edr_threshold(scores_benign, percentile=99.5):
    return float(np.percentile(scores_benign, percentile))


def apply_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    if preds.sum() == 0:
        return 0.0
    return float(f1_score(y_true, preds, zero_division=0))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_phase1_arrays(dataset):
    data_file = PROJECT_ROOT / "data" / "processed" / dataset / "phase1_base_arrays.pkl"
    with open(data_file, "rb") as handle:
        return pickle.load(handle)


def build_z_artifacts(dataset, sample_size=50000):
    data = load_phase1_arrays(dataset)
    selection = resolve_phase5_selection(dataset)

    trace_ids_train = data.get("trace_ids_train", np.arange(len(data["X_train_w"])))
    trace_ids_val = data["trace_ids_val"]
    trace_ids_attack = data["trace_ids_attack"]

    X_train_w, trace_ids_train = fast_sample(data["X_train_w"], trace_ids_train, sample_size)
    X_val_w, trace_ids_val = fast_sample(data["X_val_w"], trace_ids_val, sample_size)
    X_attack_eval, trace_ids_attack_eval = fast_sample(data["X_attack_w"], trace_ids_attack, sample_size)

    y_eval = np.concatenate([np.zeros(len(X_val_w)), np.ones(len(X_attack_eval))])
    trace_ids_eval = np.concatenate(
        [trace_ids_val, trace_ids_attack_eval + int(trace_ids_val.max()) + 1]
    )

    z_scores = np.zeros((len(y_eval), 3), dtype=np.float64)

    path_a_bundle = load_selected_path_a_bundle(dataset, X_train_w, X_val_w, X_attack_eval)
    scores_a_val, scores_a_attack = score_path_a_bundle(path_a_bundle)
    z_scores[:, 0] = np.concatenate([scores_a_val, scores_a_attack])

    t_val = torch.tensor(X_val_w, dtype=torch.long).to(device)
    t_attack = torch.tensor(X_attack_eval, dtype=torch.long).to(device)

    path_b_model, path_b_ckpt = load_path_b_model(dataset, device)
    with torch.no_grad():
        val_pred_b, val_emb_b = path_b_model(t_val)
        atk_pred_b, atk_emb_b = path_b_model(t_attack)
        mse_val = torch.mean((val_emb_b - val_pred_b) ** 2, dim=[1, 2]).cpu().numpy()
        mse_attack = torch.mean((atk_emb_b - atk_pred_b) ** 2, dim=[1, 2]).cpu().numpy()
    z_scores[:, 1] = np.concatenate([mse_val, mse_attack])

    path_c_model, path_c_ckpt = load_path_c_model(dataset, device)
    crit = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        nll_val = crit(path_c_model(t_val[:, :-1]).transpose(1, 2), t_val[:, 1:]).mean(dim=1).cpu().numpy()
        nll_attack = crit(
            path_c_model(t_attack[:, :-1]).transpose(1, 2), t_attack[:, 1:]
        ).mean(dim=1).cpu().numpy()
    z_scores[:, 2] = np.concatenate([nll_val, nll_attack])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, heldout_idx = next(gss1.split(z_scores, y_eval, groups=trace_ids_eval))
    heldout_groups = trace_ids_eval[heldout_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_local_idx, test_local_idx = next(
        gss2.split(z_scores[heldout_idx], y_eval[heldout_idx], groups=heldout_groups)
    )
    val_idx = heldout_idx[val_local_idx]
    test_idx = heldout_idx[test_local_idx]

    z_splits = {
        "artifact_type": "phase5_z_splits",
        "artifact_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "sample_size": sample_size,
        "z_train": z_scores[train_idx],
        "y_train": y_eval[train_idx],
        "z_val": z_scores[val_idx],
        "y_val": y_eval[val_idx],
        "z_test": z_scores[test_idx],
        "y_test": y_eval[test_idx],
    }

    z_eval = {
        "artifact_type": "phase5_z_eval_full",
        "artifact_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "sample_size": sample_size,
        "z_eval": z_scores,
        "y_eval": y_eval,
        "trace_ids_eval": trace_ids_eval,
    }

    provenance = {
        "path_a_model": str(path_a_bundle["model_path"]),
        "path_a_vectorizer": str(path_a_bundle["vectorizer_path"]),
        "path_a_reducer": str(path_a_bundle["reducer_path"]) if path_a_bundle["reducer_path"] else None,
        "path_b_model": str(path_b_ckpt),
        "path_c_model": str(path_c_ckpt),
    }
    z_splits["source_artifacts"] = provenance
    z_eval["source_artifacts"] = provenance

    return z_splits, z_eval


def save_z_artifacts(dataset, z_splits, z_eval):
    z_splits_path = resolve_phase5_z_splits_path(dataset)
    z_eval_path = resolve_phase5_z_eval_path(dataset)

    with open(z_splits_path, "wb") as handle:
        pickle.dump(z_splits, handle)
    with open(z_eval_path, "wb") as handle:
        pickle.dump(z_eval, handle)

    return z_splits_path, z_eval_path


def build_ablation_isolation_forest(dataset, selection, z_splits):
    z_train_benign = z_splits["z_train"][z_splits["y_train"] == 0]
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(z_train_benign)

    val_scores = -model.score_samples(z_splits["z_val"])
    tau = get_edr_threshold(val_scores[z_splits["y_val"] == 0], percentile=99.5)
    test_scores = -model.score_samples(z_splits["z_test"])
    f1_test = apply_threshold(z_splits["y_test"], test_scores, tau)

    return {
        "artifact_type": "phase5_score_space_baseline",
        "artifact_version": 1,
        "artifact_name": ABLATION_IF_ARTIFACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "model_name": "IsolationForest",
        "protocol": "ablation_score_space_edr",
        "training_scope": "benign_z_train_only",
        "threshold_method": "edr_percentile_on_benign_z_val",
        "threshold_tau": tau,
        "threshold_percentile": 99.5,
        "metrics": {
            "F1_test_final": round(f1_test, 4),
        },
        "model": model,
    }


def build_ablation_gmm(dataset, selection, z_splits):
    z_train_benign = z_splits["z_train"][z_splits["y_train"] == 0]
    model = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
    model.fit(z_train_benign)

    val_scores = -model.score_samples(z_splits["z_val"])
    tau = get_edr_threshold(val_scores[z_splits["y_val"] == 0], percentile=99.5)
    test_scores = -model.score_samples(z_splits["z_test"])
    f1_test = apply_threshold(z_splits["y_test"], test_scores, tau)

    return {
        "artifact_type": "phase5_score_space_baseline",
        "artifact_version": 1,
        "artifact_name": ABLATION_GMM_ARTIFACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "model_name": "GaussianMixture",
        "protocol": "ablation_score_space_edr",
        "training_scope": "benign_z_train_only",
        "threshold_method": "edr_percentile_on_benign_z_val",
        "threshold_tau": tau,
        "threshold_percentile": 99.5,
        "metrics": {
            "F1_test_final": round(f1_test, 4),
        },
        "model": model,
    }


def build_unsupervised_cdf_max(dataset, selection, z_splits):
    z_train_benign = z_splits["z_train"][z_splits["y_train"] == 0]
    n_quantiles = int(min(1000, max(10, len(z_train_benign))))
    transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=42)
    transformer.fit(z_train_benign)

    z_val_cdf = transformer.transform(z_splits["z_val"])
    z_test_cdf = transformer.transform(z_splits["z_test"])
    scores_val = z_val_cdf.max(axis=1)
    scores_test = z_test_cdf.max(axis=1)

    tau, f1_val = get_best_threshold(z_splits["y_val"], scores_val)
    f1_test = apply_threshold(z_splits["y_test"], scores_test, tau)
    auc_test = float(roc_auc_score(z_splits["y_test"], scores_test))

    return {
        "artifact_type": "phase5_score_space_baseline",
        "artifact_version": 1,
        "artifact_name": UNSUP_CDF_MAX_ARTIFACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "model_name": "QuantileTransformer_Max",
        "protocol": "unsupervised_cdf_max",
        "training_scope": "benign_z_train_only",
        "threshold_method": "best_f1_on_z_val",
        "threshold_tau": tau,
        "metrics": {
            "AUC": round(auc_test, 4),
            "F1": round(f1_test, 4),
            "F1_val": round(f1_val, 4),
        },
        "transformer": transformer,
    }


def build_unsupervised_gmm(dataset, selection, z_splits):
    z_train_benign = z_splits["z_train"][z_splits["y_train"] == 0]
    model = GaussianMixture(n_components=3, random_state=42)
    model.fit(z_train_benign)

    scores_val = -model.score_samples(z_splits["z_val"])
    scores_test = -model.score_samples(z_splits["z_test"])
    tau, f1_val = get_best_threshold(z_splits["y_val"], scores_val)
    f1_test = apply_threshold(z_splits["y_test"], scores_test, tau)
    auc_test = float(roc_auc_score(z_splits["y_test"], scores_test))

    return {
        "artifact_type": "phase5_score_space_baseline",
        "artifact_version": 1,
        "artifact_name": UNSUP_GMM_ARTIFACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "model_name": "GaussianMixture",
        "protocol": "unsupervised_gmm_best_f1",
        "training_scope": "benign_z_train_only",
        "threshold_method": "best_f1_on_z_val",
        "threshold_tau": tau,
        "metrics": {
            "AUC": round(auc_test, 4),
            "F1": round(f1_test, 4),
            "F1_val": round(f1_val, 4),
        },
        "model": model,
    }


def build_iso_vs_xgb_if(dataset, selection, z_eval):
    z_benign_full = z_eval["z_eval"][z_eval["y_eval"] == 0]
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(z_benign_full)

    scores_eval = -model.score_samples(z_eval["z_eval"])
    tau, f1_eval = get_best_threshold(z_eval["y_eval"], scores_eval)
    auc_eval = float(roc_auc_score(z_eval["y_eval"], scores_eval))

    return {
        "artifact_type": "phase5_score_space_baseline",
        "artifact_version": 1,
        "artifact_name": ISO_VS_XGB_IF_ARTIFACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": selection,
        "model_name": "IsolationForest",
        "protocol": "iso_vs_xgb_full_eval_best_f1",
        "training_scope": "all_benign_z_eval_only",
        "threshold_method": "best_f1_on_full_eval",
        "threshold_tau": tau,
        "metrics": {
            "AUC": round(auc_eval, 4),
            "F1": round(f1_eval, 4),
        },
        "model": model,
    }


def write_payload_and_manifest(out_dir, payload):
    artifact_name = payload["artifact_name"]
    payload_path = out_dir / f"{artifact_name}.pkl"
    manifest_path = out_dir / f"{artifact_name}_manifest.json"

    with open(payload_path, "wb") as handle:
        pickle.dump(payload, handle)

    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "transformer"}
    }
    manifest["model_file"] = str(payload_path)
    manifest["manifest_file"] = str(manifest_path)
    manifest = _json_ready(manifest)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=4)

    return {
        "artifact_name": artifact_name,
        "model_name": payload["model_name"],
        "model_file": str(payload_path),
        "manifest_file": str(manifest_path),
        "protocol": payload["protocol"],
    }


def build_registry(dataset, selection, z_splits_path, z_eval_path, artifact_entries):
    return {
        "artifact_type": "phase5_score_space_baseline_registry",
        "artifact_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_paths": _json_ready(selection),
        "z_splits_file": str(z_splits_path),
        "z_eval_file": str(z_eval_path),
        "artifacts": artifact_entries,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export Phase 5 benchmark artifacts outside scripts/benchmark")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Trace-aware sample size for eval pools")
    args = parser.parse_args()

    set_seed(42)

    baseline_dir = resolve_phase5_baseline_dir(args.dataset)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"EXPORT PHASE 5 BENCHMARK ARTIFACTS ({args.dataset.upper()})")
    print("=" * 72)

    selection = resolve_phase5_selection(args.dataset)
    print(f"INFO Selected stack: {selection['path_a_label']} + {selection['path_b_label']} + {selection['path_c_label']}")

    print("INFO Building Z-matrix artifacts from saved Path A/B/C models...")
    z_splits, z_eval = build_z_artifacts(args.dataset, sample_size=args.sample_size)
    z_splits_path, z_eval_path = save_z_artifacts(args.dataset, z_splits, z_eval)
    print(f" Saved splits to {z_splits_path}")
    print(f" Saved eval to {z_eval_path}")

    print("INFO Training/exporting score-space baseline artifacts...")
    payloads = [
        build_ablation_isolation_forest(args.dataset, selection, z_splits),
        build_ablation_gmm(args.dataset, selection, z_splits),
        build_unsupervised_cdf_max(args.dataset, selection, z_splits),
        build_unsupervised_gmm(args.dataset, selection, z_splits),
        build_iso_vs_xgb_if(args.dataset, selection, z_eval),
    ]

    artifact_entries = []
    for payload in payloads:
        entry = write_payload_and_manifest(baseline_dir, payload)
        artifact_entries.append(entry)
        print(f" Exported {payload['artifact_name']} to {entry['model_file']}")

    registry = build_registry(args.dataset, selection, z_splits_path, z_eval_path, artifact_entries)
    registry_path = baseline_dir / "baseline_models_index.json"
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(registry), handle, indent=4)
    print(f"INFO Registry saved to {registry_path}")

    print("\nINFO Summary")
    for payload in payloads:
        print(f" {payload['artifact_name']}: {json.dumps(_json_ready(payload['metrics']))}")


if __name__ == "__main__":
    main()

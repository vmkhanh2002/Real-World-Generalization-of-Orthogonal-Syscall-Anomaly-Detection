"""
Compare Path B/C Evaluation Protocols
=====================================

This script compares two threshold-selection protocols on Path B/C anomaly scores:

1) Holdout trace split
 - Split traces into calibration/test (stratified by class at trace level)
 - Select tau* on calibration split (PR-curve best F1)
 - Report final metrics on holdout test split

2) Nested CV (trace-level, leakage-safe)
 - Outer folds: unbiased test estimate
 - Inner folds: model-variant + tau* selection
 - Path B variant set: cnn / dense / lstm / vae / svdd
 - Path C variant set: markov / hmm / gru / cbow / lstm_ae

Outputs:
 experiments/{dataset}/logs/path_bc_eval_protocol_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from model_metadata import load_torch_with_metadata
from models import CBOWPredictor, Conv1DAE, DeepSVDD, DenseAE, GRUPredictor, LSTMAE, LSTMAESequence, VAE
from pipeline_config import (
    get_path_b_defaults,
    get_path_c_defaults,
    normalize_dataset_name,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH_B_VARIANTS = ("cnn", "dense", "lstm", "vae", "svdd")
PATH_C_VARIANTS = ("markov", "hmm", "gru", "cbow", "lstm_ae")

AGGREGATE_METRIC_KEYS = (
    "AUC_ROC",
    "AUPR",
    "F1",
    "Precision",
    "Recall",
    "Specificity",
    "Balanced_Accuracy",
    "MCC",
    "FPR",
    "FNR",
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file for {label}: {path}")


def fast_sample(matrix: np.ndarray, trace_ids: np.ndarray, size: int | None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
 Subsample windows while roughly preserving trace distribution.
 Mirrors the utility used in phase-5 scripts.
 """
    if size is None or len(matrix) <= size:
        return matrix, trace_ids

    rng = np.random.RandomState(seed)
    unique_traces = np.unique(trace_ids)
    shuffled_traces = rng.permutation(unique_traces)

    selected_mask = np.zeros(len(matrix), dtype=bool)
    current_size = 0

    for t_id in shuffled_traces:
        t_mask = trace_ids == t_id
        t_size = int(np.sum(t_mask))
        if current_size + t_size <= size:
            selected_mask |= t_mask
            current_size += t_size
            continue

        remainder = size - current_size
        if remainder > 0:
            idx_of_trace = np.where(t_mask)[0]
            selected_mask[idx_of_trace[:remainder]] = True
        break

    return matrix[selected_mask], trace_ids[selected_mask]


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def get_best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
 PR-curve best-F1 threshold selection.
 Same indexing style as training scripts:
 threshold = thresholds[min(best_idx, len(thresholds)-1)]
 """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    if len(f1_scores) == 0:
        return 0.5, 0.0

    best_idx = int(np.argmax(f1_scores))
    if len(thresholds) == 0:
        return 0.5, float(f1_scores[best_idx])
    thr_idx = min(best_idx, len(thresholds) - 1)
    return float(thresholds[thr_idx]), float(f1_scores[best_idx])


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "AUC_ROC": safe_roc_auc(y_true, scores),
        "AUPR": safe_aupr(y_true, scores),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, preds)),
        "MCC": float(matthews_corrcoef(y_true, preds)),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def _agg_series(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def aggregate_metrics(metric_rows: Sequence[Mapping[str, float]]) -> dict:
    out: dict = {}
    for key in AGGREGATE_METRIC_KEYS:
        out[key] = _agg_series([float(row[key]) for row in metric_rows])
    return out


def _safe_float(v: float | int | np.number | None) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if not math.isfinite(fv):
        return None
    return fv


def sanitize_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    cap = float(np.nanmax(arr[finite])) if finite.any() else 1e6
    arr = np.where(np.isposinf(arr), cap, arr)
    arr = np.where(np.isneginf(arr), -cap, arr)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr

def _ensure_min_fold_size(requested: int, n_norm: int, n_attack: int, name: str) -> int:
    max_feasible = min(n_norm, n_attack)
    if max_feasible < 2:
        raise ValueError(
            f"Not enough traces for {name}: normal={n_norm}, attack={n_attack}. Need at least 2 per class."
        )
    return max(2, min(requested, max_feasible))


def combine_scores_with_trace_ids(
    scores_val: np.ndarray,
    scores_atk: np.ndarray,
    trace_ids_val: np.ndarray,
    trace_ids_atk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_all = np.concatenate(
        [
            np.zeros(len(scores_val), dtype=np.int64),
            np.ones(len(scores_atk), dtype=np.int64),
        ]
    )

    scores_all = np.concatenate([scores_val, scores_atk]).astype(np.float64, copy=False)
    offset = int(trace_ids_val.max()) + 1 if len(trace_ids_val) > 0 else 0
    trace_ids_all = np.concatenate([trace_ids_val, trace_ids_atk + offset]).astype(np.int64, copy=False)

    return y_all, scores_all, trace_ids_all


def _split_class_trace_ids(ids: np.ndarray, calib_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(ids, dtype=np.int64, copy=False)
    rng = np.random.RandomState(seed)
    ids = ids.copy()
    rng.shuffle(ids)

    if len(ids) < 2:
        return ids, ids

    n_calib = int(round(len(ids) * calib_ratio))
    n_calib = max(1, min(n_calib, len(ids) - 1))
    return ids[:n_calib], ids[n_calib:]


def run_holdout_trace_split(
    scores_val: np.ndarray,
    scores_atk: np.ndarray,
    trace_ids_val: np.ndarray,
    trace_ids_atk: np.ndarray,
    calib_ratio: float,
    seed: int,
) -> dict:
    y_all, scores_all, trace_ids_all = combine_scores_with_trace_ids(scores_val, scores_atk, trace_ids_val, trace_ids_atk)

    norm_trace_ids = np.unique(trace_ids_all[y_all == 0])
    atk_trace_ids = np.unique(trace_ids_all[y_all == 1])

    calib_norm, test_norm = _split_class_trace_ids(norm_trace_ids, calib_ratio=calib_ratio, seed=seed + 1)
    calib_atk, test_atk = _split_class_trace_ids(atk_trace_ids, calib_ratio=calib_ratio, seed=seed + 2)

    calib_trace_ids = np.concatenate([calib_norm, calib_atk])
    test_trace_ids = np.concatenate([test_norm, test_atk])

    calib_mask = np.isin(trace_ids_all, calib_trace_ids)
    test_mask = np.isin(trace_ids_all, test_trace_ids)

    y_calib = y_all[calib_mask]
    s_calib = scores_all[calib_mask]
    y_test = y_all[test_mask]
    s_test = scores_all[test_mask]

    tau_star, f1_calib = get_best_f1(y_calib, s_calib)

    calib_metrics = compute_binary_metrics(y_calib, s_calib, tau_star)
    test_metrics = compute_binary_metrics(y_test, s_test, tau_star)

    sizes = {
        "calib_windows": int(calib_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "calib_norm_windows": int(np.sum((y_all == 0) & calib_mask)),
        "calib_attack_windows": int(np.sum((y_all == 1) & calib_mask)),
        "test_norm_windows": int(np.sum((y_all == 0) & test_mask)),
        "test_attack_windows": int(np.sum((y_all == 1) & test_mask)),
        "calib_norm_traces": int(len(calib_norm)),
        "calib_attack_traces": int(len(calib_atk)),
        "test_norm_traces": int(len(test_norm)),
        "test_attack_traces": int(len(test_atk)),
    }

    return {
        "protocol": "holdout_trace_split",
        "calib_ratio": float(calib_ratio),
        "threshold_tau": float(tau_star),
        "sizes": sizes,
        "F1_calib": float(f1_calib),
        "calib_metrics": calib_metrics,
        "test_metrics": test_metrics,
    }


def run_nested_cv_variant_selection(
    variant_to_scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    trace_ids_val: np.ndarray,
    trace_ids_atk: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict:
    if not variant_to_scores:
        raise ValueError("No variants provided for nested CV.")

    variants = sorted(variant_to_scores.keys())

    first_variant = variants[0]
    first_val, first_atk = variant_to_scores[first_variant]
    y_all, _, trace_ids_all = combine_scores_with_trace_ids(first_val, first_atk, trace_ids_val, trace_ids_atk)

    combined_variant_scores: dict[str, np.ndarray] = {}
    for name in variants:
        s_val, s_atk = variant_to_scores[name]
        if len(s_val) != len(first_val) or len(s_atk) != len(first_atk):
            raise ValueError(
                f"Variant '{name}' length mismatch. "
                f"Expected val/atk=({len(first_val)}, {len(first_atk)}), got ({len(s_val)}, {len(s_atk)})."
            )
        combined_variant_scores[name] = np.concatenate([s_val, s_atk]).astype(np.float64, copy=False)

    norm_trace_ids = np.unique(trace_ids_all[y_all == 0])
    atk_trace_ids = np.unique(trace_ids_all[y_all == 1])

    outer_k = _ensure_min_fold_size(outer_folds, len(norm_trace_ids), len(atk_trace_ids), "outer CV")
    outer_norm_kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed + 1000)
    outer_atk_kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed + 2000)

    outer_norm_splits = list(outer_norm_kf.split(norm_trace_ids))
    outer_atk_splits = list(outer_atk_kf.split(atk_trace_ids))

    fold_results: list[dict] = []
    selected_taus: list[float] = []
    selected_variants: list[str] = []

    for outer_idx in range(outer_k):
        norm_train_i, norm_test_i = outer_norm_splits[outer_idx]
        atk_train_i, atk_test_i = outer_atk_splits[outer_idx]

        outer_train_norm = norm_trace_ids[norm_train_i]
        outer_test_norm = norm_trace_ids[norm_test_i]
        outer_train_atk = atk_trace_ids[atk_train_i]
        outer_test_atk = atk_trace_ids[atk_test_i]

        outer_train_ids = np.concatenate([outer_train_norm, outer_train_atk])
        outer_test_ids = np.concatenate([outer_test_norm, outer_test_atk])

        outer_train_mask = np.isin(trace_ids_all, outer_train_ids)
        outer_test_mask = np.isin(trace_ids_all, outer_test_ids)

        inner_norm_trace_ids = np.unique(trace_ids_all[outer_train_mask & (y_all == 0)])
        inner_atk_trace_ids = np.unique(trace_ids_all[outer_train_mask & (y_all == 1)])

        inner_k = _ensure_min_fold_size(
            inner_folds,
            len(inner_norm_trace_ids),
            len(inner_atk_trace_ids),
            f"inner CV (outer fold {outer_idx})",
        )

        inner_norm_kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed + 3000 + outer_idx)
        inner_atk_kf = KFold(n_splits=inner_k, shuffle=True, random_state=seed + 4000 + outer_idx)

        inner_norm_splits = list(inner_norm_kf.split(inner_norm_trace_ids))
        inner_atk_splits = list(inner_atk_kf.split(inner_atk_trace_ids))

        variant_inner_rows: dict[str, list[dict]] = {v: [] for v in variants}

        for inner_idx in range(inner_k):
            norm_in_train_i, norm_in_val_i = inner_norm_splits[inner_idx]
            atk_in_train_i, atk_in_val_i = inner_atk_splits[inner_idx]

            _inner_train_ids = np.concatenate(
                [
                    inner_norm_trace_ids[norm_in_train_i],
                    inner_atk_trace_ids[atk_in_train_i],
                ]
            )
            inner_val_ids = np.concatenate(
                [
                    inner_norm_trace_ids[norm_in_val_i],
                    inner_atk_trace_ids[atk_in_val_i],
                ]
            )

            inner_val_mask = np.isin(trace_ids_all, inner_val_ids)
            y_inner_val = y_all[inner_val_mask]

            for variant in variants:
                s_inner_val = combined_variant_scores[variant][inner_val_mask]
                tau, val_f1 = get_best_f1(y_inner_val, s_inner_val)
                val_auc = safe_roc_auc(y_inner_val, s_inner_val)

                variant_inner_rows[variant].append(
                    {
                        "fold": int(inner_idx),
                        "tau": float(tau),
                        "val_f1": float(val_f1),
                        "val_auc": float(val_auc),
                    }
                )

        variant_summaries: list[dict] = []
        for variant in variants:
            rows = variant_inner_rows[variant]
            tau_values = [r["tau"] for r in rows]
            f1_values = [r["val_f1"] for r in rows]
            auc_values = [r["val_auc"] for r in rows]

            summary = {
                "variant": variant,
                "inner_mean_val_f1": float(np.mean(f1_values)),
                "inner_std_val_f1": float(np.std(f1_values)),
                "inner_mean_val_auc": float(np.nanmean(auc_values)),
                "inner_std_val_auc": float(np.nanstd(auc_values)),
                "inner_tau_mean": float(np.mean(tau_values)),
                "inner_tau_median": float(np.median(tau_values)),
                "inner_candidates": rows,
            }
            variant_summaries.append(summary)

        variant_summaries_sorted = sorted(
            variant_summaries,
            key=lambda s: (
                s["inner_mean_val_f1"],
                s["inner_mean_val_auc"],
                -s["inner_std_val_f1"],
            ),
            reverse=True,
        )

        selected_variant_summary = variant_summaries_sorted[0]
        selected_variant = str(selected_variant_summary["variant"])
        selected_tau = float(selected_variant_summary["inner_tau_median"])

        y_outer_test = y_all[outer_test_mask]
        s_outer_test = combined_variant_scores[selected_variant][outer_test_mask]
        test_metrics = compute_binary_metrics(y_outer_test, s_outer_test, selected_tau)

        selected_taus.append(selected_tau)
        selected_variants.append(selected_variant)

        fold_results.append(
            {
                "outer_fold": int(outer_idx),
                "selection_mode": "inner_cv_select_variant_and_tau",
                "selected_variant": selected_variant,
                "selected_tau": selected_tau,
                "outer_test_norm_traces": int(len(outer_test_norm)),
                "outer_test_attack_traces": int(len(outer_test_atk)),
                "outer_test_norm_windows": int(np.sum(outer_test_mask & (y_all == 0))),
                "outer_test_attack_windows": int(np.sum(outer_test_mask & (y_all == 1))),
                "variant_summaries": variant_summaries_sorted,
                "test_metrics": test_metrics,
            }
        )

    aggregate_test = aggregate_metrics([row["test_metrics"] for row in fold_results])
    variant_counts = dict(sorted(Counter(selected_variants).items(), key=lambda x: x[0]))

    return {
        "protocol": "nested_cv_trace_level",
        "outer_folds_requested": int(outer_folds),
        "outer_folds_used": int(outer_k),
        "inner_folds_requested": int(inner_folds),
        "selected_tau_stats": _agg_series(selected_taus),
        "selected_variant_counts": variant_counts,
        "aggregate_test_metrics": aggregate_test,
        "fold_results": fold_results,
    }


def infer_conv1dae_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    emb = state_dict["embedding.weight"]
    vocab_size, embed_dim = emb.shape
    return {"vocab_size": int(vocab_size), "embed_dim": int(embed_dim)}


def infer_dense_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    input_dim = int(state_dict["encoder.0.weight"].shape[1])
    latent_dim = int(state_dict["encoder.2.weight"].shape[0])
    return {"input_dim": input_dim, "latent_dim": latent_dim}


def infer_lstm_ae_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor], seq_len_hint: int) -> dict:
    hidden_dim = int(state_dict["encoder.weight_ih_l0"].shape[0] // 4)
    return {"seq_len": int(seq_len_hint), "hidden_dim": hidden_dim}


def infer_vae_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    input_dim = int(state_dict["fc1.weight"].shape[1])
    latent_dim = int(state_dict["fc_mu.weight"].shape[0])
    return {"input_dim": input_dim, "latent_dim": latent_dim}


def infer_svdd_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    input_dim = int(state_dict["net.0.weight"].shape[1])
    hidden = int(state_dict["net.4.weight"].shape[0])
    return {"input_dim": input_dim, "hidden": hidden}


def infer_gru_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    emb_weight = state_dict["embedding.weight"]
    vocab_size = int(emb_weight.shape[0] - 1)
    embed_dim = int(emb_weight.shape[1])
    hidden_dim = int(state_dict["gru.weight_ih_l0"].shape[0] // 3)

    num_layers = 0
    while f"gru.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1
    num_layers = max(num_layers, 1)

    return {
        "vocab_size": max(vocab_size, 1),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


def infer_cbow_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    emb_weight = state_dict["embedding.weight"]
    vocab_size = int(emb_weight.shape[0] - 1)
    embed_dim = int(emb_weight.shape[1])
    return {"vocab_size": max(vocab_size, 1), "embed_dim": embed_dim}


def infer_lstm_ae_seq_kwargs_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    emb_weight = state_dict["embed.weight"]
    vocab_size = int(emb_weight.shape[0] - 1)
    embed_dim = int(emb_weight.shape[1])
    hidden_dim = int(state_dict["encoder.weight_ih_l0"].shape[0] // 4)
    return {
        "vocab_size": max(vocab_size, 1),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
    }


def load_torch_model(
    checkpoint_path: str,
    model_label: str,
    class_registry: Mapping[str, type[nn.Module]],
    fallback_ctor: Callable[[], nn.Module],
    infer_kwargs_fn: Callable[[Mapping[str, torch.Tensor]], dict] | None = None,
) -> nn.Module:
    try:
        payload = load_torch_with_metadata(
            checkpoint_path,
            map_location=DEVICE,
            class_registry=class_registry,
            fallback_ctor=fallback_ctor,
            strict=True,
            require_metadata=False,
        )
        model = payload["model"].to(DEVICE)
        model.eval()
        if payload.get("metadata_used"):
            model_class = payload.get("metadata", {}).get("model_class", "<unknown>")
            print(f" - {model_label}: loaded with metadata ({model_class}).")
        else:
            print(f" - {model_label}: loaded via fallback ctor (metadata missing).")
        return model
    except Exception as exc:
        if infer_kwargs_fn is None:
            raise RuntimeError(f"{model_label} load failed: {exc}") from exc

        print(f" [warning] {model_label}: metadata/fallback load failed ({exc}).")
        print(f" [warning] {model_label}: trying state_dict shape inference.")

        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        inferred_kwargs = infer_kwargs_fn(state_dict)
        model_cls = next(iter(class_registry.values()))
        model = model_cls(**inferred_kwargs).to(DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f" - {model_label}: loaded via inferred kwargs {inferred_kwargs}.")
        return model


@dataclass
class PreparedData:
    dataset: str
    x_train_w: np.ndarray
    x_val_w: np.ndarray
    x_attack_w: np.ndarray
    trace_ids_train: np.ndarray
    trace_ids_val: np.ndarray
    trace_ids_attack: np.ndarray
    max_id: int

def load_prepared_data(dataset: str, max_windows_per_split: int | None, seed: int) -> PreparedData:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed", dataset, "phase1_base_arrays.pkl")
    require_file(data_file, "phase1 arrays")

    import pickle

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    for key in ("X_train_w", "X_val_w", "X_attack_w", "trace_ids_train", "trace_ids_val", "trace_ids_attack"):
        if key not in data:
            raise KeyError(
                f"Missing '{key}' in {data_file}. "
                "Please regenerate arrays with scripts/data_pipeline.py."
            )

    x_train_raw = data["X_train_w"]
    x_val_raw = data["X_val_w"]
    x_attack_raw = data["X_attack_w"]
    trace_ids_train = data["trace_ids_train"]
    trace_ids_val = data["trace_ids_val"]
    trace_ids_attack = data["trace_ids_attack"]

    max_id = int(max(x_train_raw.max(), x_val_raw.max(), x_attack_raw.max()))

    x_train_w, trace_ids_train = fast_sample(x_train_raw, trace_ids_train, max_windows_per_split, seed=seed + 11)
    x_val_w, trace_ids_val = fast_sample(x_val_raw, trace_ids_val, max_windows_per_split, seed=seed + 12)
    x_attack_w, trace_ids_attack = fast_sample(x_attack_raw, trace_ids_attack, max_windows_per_split, seed=seed + 13)

    return PreparedData(
        dataset=dataset,
        x_train_w=x_train_w,
        x_val_w=x_val_w,
        x_attack_w=x_attack_w,
        trace_ids_train=trace_ids_train,
        trace_ids_val=trace_ids_val,
        trace_ids_attack=trace_ids_attack,
        max_id=max_id,
    )


def _path_b_selected_to_variant(selected_model_key: str) -> str:
    key = selected_model_key.strip().lower()
    mapping = {
        "cnn1d_ae": "cnn",
        "cnn": "cnn",
        "dense_ae": "dense",
        "dense": "dense",
        "lstm_ae": "lstm",
        "lstm": "lstm",
        "vae": "vae",
        "deep_svdd": "svdd",
        "deepsvdd": "svdd",
        "svdd": "svdd",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported Path B selected_model='{selected_model_key}'.")
    return mapping[key]


def _path_c_selected_to_variant(selected_model_key: str) -> str:
    key = selected_model_key.strip().lower()
    mapping = {
        "markov": "markov",
        "hmm": "hmm",
        "gru_predictor": "gru",
        "gru": "gru",
        "cbow_predictor": "cbow",
        "cbow": "cbow",
        "lstm_ae_sequence": "lstm_ae",
        "lstm_ae": "lstm_ae",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported Path C selected_model='{selected_model_key}'.")
    return mapping[key]


def _load_pickle(path: str, label: str):
    require_file(path, label)
    with open(path, "rb") as f:
        return pickle.load(f)


def _score_markov_sequences(model: Mapping[str, object], matrix: np.ndarray) -> np.ndarray:
    order = int(model["order"])
    probs_dict = model["probs"]
    scores: list[float] = []
    for seq in matrix:
        seq = np.asarray(seq)
        denom = max(len(seq) - order, 1)
        log_prob = 0.0
        for i in range(len(seq) - order):
            state = tuple(int(x) for x in seq[i : i + order])
            nxt = int(seq[i + order])
            if state in probs_dict and nxt in probs_dict[state]:
                log_prob += np.log(probs_dict[state][nxt] + 1e-9)
            else:
                log_prob += -10.0
        scores.append(-log_prob / denom)
    return sanitize_scores(scores)


def _score_hmm_sequences(model, matrix: np.ndarray) -> np.ndarray:
    scores: list[float] = []
    for seq in matrix:
        try:
            score = -model.score(np.asarray(seq).reshape(-1, 1)) / max(len(seq), 1)
        except Exception:
            score = np.inf
        scores.append(score)
    return sanitize_scores(scores)


def score_path_b_variants(
    prepared: PreparedData,
    model_dir_root: str,
    path_b_defaults: Mapping,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    print("\nPath B Scoring all variants...")
    x_train_norm = (prepared.x_train_w / max(prepared.max_id, 1)).astype(np.float32)
    x_val_norm = (prepared.x_val_w / max(prepared.max_id, 1)).astype(np.float32)
    x_atk_norm = (prepared.x_attack_w / max(prepared.max_id, 1)).astype(np.float32)

    train_tensor = torch.tensor(x_train_norm, dtype=torch.float32, device=DEVICE)
    val_tensor = torch.tensor(x_val_norm, dtype=torch.float32, device=DEVICE)
    atk_tensor = torch.tensor(x_atk_norm, dtype=torch.float32, device=DEVICE)
    val_long = torch.tensor(prepared.x_val_w, dtype=torch.long, device=DEVICE)
    atk_long = torch.tensor(prepared.x_attack_w, dtype=torch.long, device=DEVICE)

    model_files = path_b_defaults.get("model_files", {})
    path_b_dir = os.path.join(model_dir_root, "path_b")

    ckpt_map = {
        "cnn": os.path.join(path_b_dir, model_files.get("cnn", "cnn.pth")),
        "dense": os.path.join(path_b_dir, model_files.get("dense", "dense.pth")),
        "lstm": os.path.join(path_b_dir, model_files.get("lstm", "lstm.pth")),
        "vae": os.path.join(path_b_dir, model_files.get("vae", "vae.pth")),
        "svdd": os.path.join(path_b_dir, model_files.get("svdd", "svdd.pth")),
    }
    for variant, ckpt in ckpt_map.items():
        require_file(ckpt, f"Path B/{variant} checkpoint")

    variant_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    cnn = load_torch_model(
        ckpt_map["cnn"],
        "Path B/cnn",
        class_registry={"Conv1DAE": Conv1DAE},
        fallback_ctor=lambda: Conv1DAE(
            vocab_size=int(prepared.max_id + 1),
            embed_dim=int(path_b_defaults.get("embed_dim", 8)),
        ),
        infer_kwargs_fn=infer_conv1dae_kwargs_from_state_dict,
    )
    with torch.no_grad():
        val_pred, val_emb = cnn(val_long)
        atk_pred, atk_emb = cnn(atk_long)
        s_val = torch.mean((val_emb - val_pred) ** 2, dim=[1, 2]).cpu().numpy()
        s_atk = torch.mean((atk_emb - atk_pred) ** 2, dim=[1, 2]).cpu().numpy()
    variant_scores["cnn"] = (s_val, s_atk)

    dense = load_torch_model(
        ckpt_map["dense"],
        "Path B/dense",
        class_registry={"DenseAE": DenseAE},
        fallback_ctor=lambda: DenseAE(input_dim=prepared.x_train_w.shape[1], latent_dim=int(path_b_defaults.get("dense_latent_dim", 16))),
        infer_kwargs_fn=infer_dense_kwargs_from_state_dict,
    )
    with torch.no_grad():
        s_val = torch.mean((val_tensor - dense(val_tensor)) ** 2, dim=1).cpu().numpy()
        s_atk = torch.mean((atk_tensor - dense(atk_tensor)) ** 2, dim=1).cpu().numpy()
    variant_scores["dense"] = (s_val, s_atk)

    lstm = load_torch_model(
        ckpt_map["lstm"],
        "Path B/lstm",
        class_registry={"LSTMAE": LSTMAE},
        fallback_ctor=lambda: LSTMAE(seq_len=prepared.x_train_w.shape[1], hidden_dim=int(path_b_defaults.get("lstm_hidden_dim", 16))),
        infer_kwargs_fn=lambda sd: infer_lstm_ae_kwargs_from_state_dict(sd, seq_len_hint=prepared.x_train_w.shape[1]),
    )
    with torch.no_grad():
        val_seq = val_tensor.unsqueeze(2)
        atk_seq = atk_tensor.unsqueeze(2)
        s_val = torch.mean((val_seq - lstm(val_seq)) ** 2, dim=[1, 2]).cpu().numpy()
        s_atk = torch.mean((atk_seq - lstm(atk_seq)) ** 2, dim=[1, 2]).cpu().numpy()
    variant_scores["lstm"] = (s_val, s_atk)

    vae = load_torch_model(
        ckpt_map["vae"],
        "Path B/vae",
        class_registry={"VAE": VAE},
        fallback_ctor=lambda: VAE(input_dim=prepared.x_train_w.shape[1], latent_dim=int(path_b_defaults.get("vae_latent_dim", 16))),
        infer_kwargs_fn=infer_vae_kwargs_from_state_dict,
    )
    with torch.no_grad():
        val_recon, _, _ = vae(val_tensor)
        atk_recon, _, _ = vae(atk_tensor)
        s_val = torch.mean((val_tensor - val_recon) ** 2, dim=1).cpu().numpy()
        s_atk = torch.mean((atk_tensor - atk_recon) ** 2, dim=1).cpu().numpy()
    variant_scores["vae"] = (s_val, s_atk)

    svdd = load_torch_model(
        ckpt_map["svdd"],
        "Path B/svdd",
        class_registry={"DeepSVDD": DeepSVDD},
        fallback_ctor=lambda: DeepSVDD(input_dim=prepared.x_train_w.shape[1], hidden=int(path_b_defaults.get("svdd_hidden_dim", 32))),
        infer_kwargs_fn=infer_svdd_kwargs_from_state_dict,
    )
    with torch.no_grad():
        center_batch = train_tensor[: min(1000, len(train_tensor))]
        c = torch.mean(svdd(center_batch), dim=0, keepdim=True)
        s_val = torch.sum((svdd(val_tensor) - c) ** 2, dim=1).cpu().numpy()
        s_atk = torch.sum((svdd(atk_tensor) - c) ** 2, dim=1).cpu().numpy()
    variant_scores["svdd"] = (s_val, s_atk)

    for variant in PATH_B_VARIANTS:
        s_val, s_atk = variant_scores[variant]
        print(
            f" - {variant:<5} "
            f"val(mean={s_val.mean():.6f}, std={s_val.std():.6f}) | "
            f"atk(mean={s_atk.mean():.6f}, std={s_atk.std():.6f})"
        )

    return variant_scores

def _score_gru_windows(model: GRUPredictor, seq_tensor: torch.Tensor, batch_size: int) -> np.ndarray:
    criterion = nn.CrossEntropyLoss(reduction="none")
    x_seq = seq_tensor[:, :-1]
    y_seq = seq_tensor[:, 1:]
    ds = TensorDataset(x_seq, y_seq)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out_scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb)
            token_loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)).view(xb.size(0), -1)
            out_scores.append(token_loss.mean(dim=1).cpu().numpy())
    return np.concatenate(out_scores)


def _score_cbow_windows(model: CBOWPredictor, seq_tensor: torch.Tensor, batch_size: int, context_size: int = 5) -> np.ndarray:
    criterion = nn.CrossEntropyLoss(reduction="none")
    ds = TensorDataset(seq_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out_scores: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for (xb,) in dl:
            seq_len = xb.shape[1]
            losses = torch.zeros(xb.size(0), device=xb.device)
            n_positions = 0
            for i in range(context_size, seq_len - context_size):
                left = xb[:, i - context_size : i]
                right = xb[:, i + 1 : i + context_size + 1]
                context = torch.cat([left, right], dim=1)
                target = xb[:, i]
                logits = model(context)
                losses += criterion(logits, target)
                n_positions += 1

            if n_positions > 0:
                losses = losses / float(n_positions)
            out_scores.append(losses.cpu().numpy())

    return np.concatenate(out_scores)


def _score_lstm_ae_seq_windows(model: LSTMAESequence, seq_tensor: torch.Tensor, batch_size: int) -> np.ndarray:
    criterion = nn.CrossEntropyLoss(reduction="none")
    ds = TensorDataset(seq_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out_scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb)
            token_loss = criterion(logits.reshape(-1, logits.size(-1)), xb.reshape(-1)).view(xb.size(0), -1)
            out_scores.append(token_loss.mean(dim=1).cpu().numpy())
    return np.concatenate(out_scores)


def score_path_c_variants(
    prepared: PreparedData,
    model_dir_root: str,
    path_c_defaults: Mapping,
    batch_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    print("\nPath C Scoring all variants...")
    val_long = torch.tensor(prepared.x_val_w, dtype=torch.long, device=DEVICE)
    atk_long = torch.tensor(prepared.x_attack_w, dtype=torch.long, device=DEVICE)

    model_files = path_c_defaults.get("model_files", {})
    path_c_dir = os.path.join(model_dir_root, "path_c")
    ckpt_map = {
        "markov": os.path.join(path_c_dir, model_files.get("markov", "markov.pkl")),
        "hmm": os.path.join(path_c_dir, model_files.get("hmm", "hmm.pkl")),
        "gru": os.path.join(path_c_dir, model_files.get("gru", "gru.pth")),
        "cbow": os.path.join(path_c_dir, model_files.get("cbow", "cbow.pth")),
        "lstm_ae": os.path.join(path_c_dir, model_files.get("lstm_ae", "lstm_ae.pth")),
    }
    for variant, ckpt in ckpt_map.items():
        require_file(ckpt, f"Path C/{variant} checkpoint")

    variant_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    markov_model = _load_pickle(ckpt_map["markov"], "Path C/markov")
    s_val = _score_markov_sequences(markov_model, prepared.x_val_w)
    s_atk = _score_markov_sequences(markov_model, prepared.x_attack_w)
    variant_scores["markov"] = (s_val, s_atk)

    hmm_model = _load_pickle(ckpt_map["hmm"], "Path C/hmm")
    s_val = _score_hmm_sequences(hmm_model, prepared.x_val_w)
    s_atk = _score_hmm_sequences(hmm_model, prepared.x_attack_w)
    variant_scores["hmm"] = (s_val, s_atk)

    gru = load_torch_model(
        ckpt_map["gru"],
        "Path C/gru",
        class_registry={"GRUPredictor": GRUPredictor},
        fallback_ctor=lambda: GRUPredictor(
            vocab_size=int(prepared.max_id + 1),
            embed_dim=int(path_c_defaults.get("gru_embed_dim", 16)),
            hidden_dim=int(path_c_defaults.get("hidden_dim", 32)),
            num_layers=int(path_c_defaults.get("num_layers", 1)),
        ),
        infer_kwargs_fn=infer_gru_kwargs_from_state_dict,
    )
    s_val = _score_gru_windows(gru, val_long, batch_size=batch_size)
    s_atk = _score_gru_windows(gru, atk_long, batch_size=batch_size)
    variant_scores["gru"] = (s_val, s_atk)

    cbow = load_torch_model(
        ckpt_map["cbow"],
        "Path C/cbow",
        class_registry={"CBOWPredictor": CBOWPredictor},
        fallback_ctor=lambda: CBOWPredictor(
            vocab_size=int(prepared.max_id + 1),
            embed_dim=int(path_c_defaults.get("cbow_embed_dim", 32)),
        ),
        infer_kwargs_fn=infer_cbow_kwargs_from_state_dict,
    )
    s_val = _score_cbow_windows(cbow, val_long, batch_size=batch_size, context_size=5)
    s_atk = _score_cbow_windows(cbow, atk_long, batch_size=batch_size, context_size=5)
    variant_scores["cbow"] = (s_val, s_atk)

    lstm_ae = load_torch_model(
        ckpt_map["lstm_ae"],
        "Path C/lstm_ae",
        class_registry={"LSTMAESequence": LSTMAESequence},
        fallback_ctor=lambda: LSTMAESequence(
            vocab_size=int(prepared.max_id + 1),
            embed_dim=int(path_c_defaults.get("lstm_ae_embed_dim", 16)),
            hidden_dim=int(path_c_defaults.get("lstm_ae_hidden_dim", 32)),
        ),
        infer_kwargs_fn=infer_lstm_ae_seq_kwargs_from_state_dict,
    )
    s_val = _score_lstm_ae_seq_windows(lstm_ae, val_long, batch_size=batch_size)
    s_atk = _score_lstm_ae_seq_windows(lstm_ae, atk_long, batch_size=batch_size)
    variant_scores["lstm_ae"] = (s_val, s_atk)

    for variant in PATH_C_VARIANTS:
        s_val, s_atk = variant_scores[variant]
        print(
            f" - {variant:<8} "
            f"val(mean={s_val.mean():.6f}, std={s_val.std():.6f}) | "
            f"atk(mean={s_atk.mean():.6f}, std={s_atk.std():.6f})"
        )

    return variant_scores


def compare_for_path(
    path_name: str,
    variant_scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    selected_variant: str,
    trace_ids_val: np.ndarray,
    trace_ids_atk: np.ndarray,
    calib_ratio: float,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict:
    if selected_variant not in variant_scores:
        raise ValueError(
            f"{path_name}: selected variant '{selected_variant}' is not available in scored variants "
            f"{sorted(variant_scores.keys())}."
        )

    sel_val, sel_atk = variant_scores[selected_variant]
    holdout = run_holdout_trace_split(
        sel_val,
        sel_atk,
        trace_ids_val=trace_ids_val,
        trace_ids_atk=trace_ids_atk,
        calib_ratio=calib_ratio,
        seed=seed,
    )

    nested = run_nested_cv_variant_selection(
        variant_to_scores=variant_scores,
        trace_ids_val=trace_ids_val,
        trace_ids_atk=trace_ids_atk,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )

    summary_by_variant = {}
    for variant, (s_val, s_atk) in variant_scores.items():
        summary_by_variant[variant] = {
            "val_mean": float(np.mean(s_val)),
            "val_std": float(np.std(s_val)),
            "atk_mean": float(np.mean(s_atk)),
            "atk_std": float(np.std(s_atk)),
        }

    holdout_f1 = _safe_float(holdout["test_metrics"]["F1"])
    nested_f1_mean = _safe_float(nested["aggregate_test_metrics"]["F1"]["mean"])
    delta = None
    if holdout_f1 is not None and nested_f1_mean is not None:
        delta = holdout_f1 - nested_f1_mean

    return {
        "selected_variant": selected_variant,
        "available_variants": sorted(list(variant_scores.keys())),
        "scores_summary_by_variant": summary_by_variant,
        "protocols": {
            "holdout_trace_split": holdout,
            "nested_cv_trace_level": nested,
            "comparison": {
                "holdout_f1_test": holdout_f1,
                "nested_f1_test_mean": nested_f1_mean,
                "delta_f1_holdout_minus_nested": delta,
            },
        },
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Path B/C evaluation protocols. "
            "Nested CV selects both model variant and threshold tau* in the inner loop."
        )
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset namespace: 32bit or 64bit")
    parser.add_argument(
        "--paths",
        type=str,
        default="both",
        choices=["path_b", "path_c", "both"],
        help="Which path(s) to evaluate.",
    )
    parser.add_argument(
        "--calib_ratio",
        type=float,
        default=0.5,
        help="Calibration ratio for holdout protocol (trace-level, per class).",
    )
    parser.add_argument("--outer_folds", type=int, default=5, help="Outer folds for nested CV.")
    parser.add_argument("--inner_folds", type=int, default=3, help="Inner folds for nested CV.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for model inference.")
    parser.add_argument(
        "--max_windows_per_split",
        type=int,
        default=None,
        help="Optional cap per split (train/val/attack) before scoring.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    if not (0.0 < args.calib_ratio < 1.0):
        raise ValueError("--calib_ratio must be in (0, 1).")
    if args.outer_folds < 2:
        raise ValueError("--outer_folds must be >= 2.")
    if args.inner_folds < 2:
        raise ValueError("--inner_folds must be >= 2.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

    set_seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir_root = os.path.join(project_root, "models", args.dataset)
    out_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(out_dir, exist_ok=True)

    path_b_defaults = get_path_b_defaults(args.dataset)
    path_c_defaults = get_path_c_defaults(args.dataset)

    print("=" * 72)
    print(f"PATH B/C EVAL PROTOCOL COMPARISON ({args.dataset.upper()})")
    print(f"Device={DEVICE} | paths={args.paths} | seed={args.seed}")
    print(
        f"Config: calib_ratio={args.calib_ratio}, outer_folds={args.outer_folds}, "
        f"inner_folds={args.inner_folds}, batch_size={args.batch_size}, "
        f"max_windows_per_split={args.max_windows_per_split}"
    )
    print("=" * 72)

    prepared = load_prepared_data(
        dataset=args.dataset,
        max_windows_per_split=args.max_windows_per_split,
        seed=args.seed,
    )

    print(
        "Data loaded: "
        f"train={prepared.x_train_w.shape}, val={prepared.x_val_w.shape}, attack={prepared.x_attack_w.shape}, "
        f"max_id={prepared.max_id}"
    )
    print(
        "Trace counts: "
        f"train={len(np.unique(prepared.trace_ids_train))}, "
        f"val={len(np.unique(prepared.trace_ids_val))}, "
        f"attack={len(np.unique(prepared.trace_ids_attack))}"
    )

    final_results: dict = {}

    if args.paths in {"path_b", "both"}:
        path_b_variant_scores = score_path_b_variants(
            prepared=prepared,
            model_dir_root=model_dir_root,
            path_b_defaults=path_b_defaults,
        )
        path_b_selected_key = str(path_b_defaults.get("selected_model", "cnn1d_ae"))
        path_b_selected_variant = _path_b_selected_to_variant(path_b_selected_key)
        print(f"Path B Holdout baseline selected model: {path_b_selected_key} to variant '{path_b_selected_variant}'")

        final_results["path_b"] = compare_for_path(
            path_name="path_b",
            variant_scores=path_b_variant_scores,
            selected_variant=path_b_selected_variant,
            trace_ids_val=prepared.trace_ids_val,
            trace_ids_atk=prepared.trace_ids_attack,
            calib_ratio=args.calib_ratio,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )

    if args.paths in {"path_c", "both"}:
        path_c_variant_scores = score_path_c_variants(
            prepared=prepared,
            model_dir_root=model_dir_root,
            path_c_defaults=path_c_defaults,
            batch_size=args.batch_size,
        )
        path_c_selected_key = str(path_c_defaults.get("selected_model", "gru_predictor"))
        path_c_selected_variant = _path_c_selected_to_variant(path_c_selected_key)
        print(f"Path C Holdout baseline selected model: {path_c_selected_key} to variant '{path_c_selected_variant}'")

        final_results["path_c"] = compare_for_path(
            path_name="path_c",
            variant_scores=path_c_variant_scores,
            selected_variant=path_c_selected_variant,
            trace_ids_val=prepared.trace_ids_val,
            trace_ids_atk=prepared.trace_ids_attack,
            calib_ratio=args.calib_ratio,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )

    payload = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "config": {
            "paths": args.paths,
            "calib_ratio": float(args.calib_ratio),
            "outer_folds": int(args.outer_folds),
            "inner_folds": int(args.inner_folds),
            "batch_size": int(args.batch_size),
            "max_windows_per_split": int(args.max_windows_per_split) if args.max_windows_per_split is not None else None,
        },
        "results": final_results,
    }

    out_file = os.path.join(out_dir, "path_bc_eval_protocol_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved comparison report:")
    print(f" {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Torch Checkpoint Metadata Utilities
==================================

Utilities for saving/loading torch checkpoints with JSON sidecar metadata.
This enables model reconstruction without hardcoded architecture defaults.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type

import torch

SCHEMA_VERSION = "1.0"


def _to_serializable(obj: Any) -> Any:
    """Convert common Python/NumPy/Torch scalar-like values into JSON-safe types."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    return str(obj)


def _default_meta_path(checkpoint_path: str | os.PathLike) -> str:
    path = Path(checkpoint_path)
    return str(path.with_suffix(path.suffix + ".meta.json"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_metadata(
    *,
    model_class: str,
    init_kwargs: Optional[Mapping[str, Any]] = None,
    dataset: Optional[str] = None,
    source_script: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    schema_version: str = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Build normalized metadata dictionary for model checkpoints."""
    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "model_class": model_class,
        "init_kwargs": _to_serializable(dict(init_kwargs or {})),
        "dataset": dataset,
        "source_script": source_script,
        "saved_at_utc": _utc_now_iso(),
    }
    if extra:
        payload["extra"] = _to_serializable(dict(extra))
    return payload


def write_metadata(metadata: Mapping[str, Any], meta_path: str | os.PathLike) -> str:
    """Write metadata JSON to disk and return absolute meta path."""
    meta_path = str(meta_path)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(dict(metadata)), f, indent=2)
    return meta_path


def read_metadata(meta_path: str | os.PathLike) -> Dict[str, Any]:
    """Read metadata JSON and return parsed dict."""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_torch_with_metadata(
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike,
    *,
    model_class: Optional[str] = None,
    init_kwargs: Optional[Mapping[str, Any]] = None,
    dataset: Optional[str] = None,
    source_script: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    meta_path: Optional[str | os.PathLike] = None,
) -> Dict[str, str]:
    """
 Save a torch state_dict and its sidecar metadata file.

 Returns:
 {"checkpoint_path": ..., "meta_path": ...}
 """
    checkpoint_path = str(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    metadata = build_metadata(
        model_class=model_class or model.__class__.__name__,
        init_kwargs=init_kwargs,
        dataset=dataset,
        source_script=source_script,
        extra=extra,
    )
    resolved_meta_path = str(meta_path) if meta_path else _default_meta_path(checkpoint_path)
    write_metadata(metadata, resolved_meta_path)
    return {"checkpoint_path": checkpoint_path, "meta_path": resolved_meta_path}


def _instantiate_model(
    *,
    metadata: Optional[Mapping[str, Any]],
    class_registry: Optional[Mapping[str, Type[torch.nn.Module]]] = None,
    fallback_ctor: Optional[callable] = None,
) -> torch.nn.Module:
    """Instantiate model from metadata registry or fallback constructor."""
    if metadata and class_registry:
        model_class_name = metadata.get("model_class")
        init_kwargs = metadata.get("init_kwargs") or {}
        if model_class_name in class_registry:
            return class_registry[model_class_name](**init_kwargs)
    if fallback_ctor is not None:
        return fallback_ctor()
    cls_name = metadata.get("model_class") if metadata else "<unknown>"
    raise ValueError(
        f"Unable to instantiate model '{cls_name}'. Provide class_registry or fallback_ctor."
    )


def load_torch_with_metadata(
    checkpoint_path: str | os.PathLike,
    *,
    map_location: Optional[torch.device | str] = None,
    class_registry: Optional[Mapping[str, Type[torch.nn.Module]]] = None,
    fallback_ctor: Optional[callable] = None,
    strict: bool = True,
    require_metadata: bool = False,
    meta_path: Optional[str | os.PathLike] = None,
) -> Dict[str, Any]:
    """
 Load checkpoint and instantiate model using sidecar metadata when available.

 Returns dict with keys:
 - model
 - metadata
 - checkpoint_path
 - meta_path
 - metadata_used
 - warning (optional)
 """
    checkpoint_path = str(checkpoint_path)
    resolved_meta_path = str(meta_path) if meta_path else _default_meta_path(checkpoint_path)

    metadata: Optional[Dict[str, Any]] = None
    metadata_used = False
    warning = None

    if os.path.exists(resolved_meta_path):
        metadata = read_metadata(resolved_meta_path)
        metadata_used = True
    elif require_metadata:
        raise FileNotFoundError(
            f"Metadata sidecar missing for checkpoint: {resolved_meta_path}"
        )
    else:
        warning = (
            f"Metadata sidecar missing for checkpoint '{checkpoint_path}'. "
            "Using fallback constructor."
        )

    model = _instantiate_model(
        metadata=metadata,
        class_registry=class_registry,
        fallback_ctor=fallback_ctor,
    )

    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict, strict=strict)

    result: Dict[str, Any] = {
        "model": model,
        "metadata": metadata,
        "checkpoint_path": checkpoint_path,
        "meta_path": resolved_meta_path,
        "metadata_used": metadata_used,
    }
    if warning:
        result["warning"] = warning
    return result



# CALIBRATION HELPERS Matches optimize_path_*.py


def find_best_threshold_pr(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """
 Find optimal threshold using PR-curve F1 maximization.
 EXACT same implementation as optimize_path_*.py get_best_f1()

 Args:
 y_true: Binary ground truth (0=benign, 1=anomaly)
 scores: Anomaly scores (higher = more anomalous)

 Returns:
 (best_threshold, best_f1): Optimal threshold and corresponding F1 score
 """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # Compute F1 for each threshold (avoid division by zero)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    # Find best F1 (exclude last point where threshold = max(score))
    if len(f1_scores) > 1:
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_f1 = float(f1_scores[best_idx])
    else:
        best_threshold = 0.5
        best_f1 = 0.0

    return best_threshold, best_f1


def compute_fpr_at_tpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
 Compute False Positive Rate at True Positive Rate = 95%.
 Matches optimize_path_a.py implementation.
 """
    sort_idx = np.argsort(-scores)
    y_sorted = y_true[sort_idx]
    n_pos = max(y_true.sum(), 1)
    n_neg = max(len(y_true) - y_true.sum(), 1)

    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg

    above95 = np.where(tpr >= 0.95)[0]
    return float(fpr[above95[0]]) if len(above95) > 0 else 1.0

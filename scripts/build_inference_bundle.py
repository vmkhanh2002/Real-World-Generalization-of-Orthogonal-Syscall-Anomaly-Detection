"""
Build Inference Bundle for Path B/C (Selected Variant + tau*)
==============================================================

Goal:
 Produce deployment-ready, inference-only artifacts for Path B/C without
 re-running training each deployment cycle.

What this script does:
 1) Load prepared arrays (val + attack trace pools).
 2) Score all Path B/C variants using existing checkpoints.
 3) Select variant per path:
 - default: majority selection from nested CV report
 - fallback: pipeline_config selected_model
 4) Calibrate tau* on a trace-level calibration split (holdout protocol),
 then keep holdout-test metrics as reference.
 5) Bundle artifacts in models/{dataset}/inference/:
 - model checkpoint copy
 - sidecar metadata copy (if present)
 - threshold_tau.json
 - manifest.json
 6) Write summary log:
 experiments/{dataset}/logs/inference_bundle_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from compare_path_bc_eval_protocols import (
    _path_b_selected_to_variant,
    _path_c_selected_to_variant,
    load_prepared_data,
    run_holdout_trace_split,
    score_path_b_variants,
    score_path_c_variants,
    set_seed,
)
from pipeline_config import get_path_b_defaults, get_path_c_defaults, normalize_dataset_name


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_variant_from_report(
    *,
    report_payload: Mapping[str, Any] | None,
    path_key: str,
    fallback_variant: str,
) -> tuple[str, str, Mapping[str, int]]:
    if report_payload is None:
        return fallback_variant, "pipeline_selected_fallback", {}

    nested = (
        report_payload.get("results", {})
        .get(path_key, {})
        .get("protocols", {})
        .get("nested_cv_trace_level", {})
    )
    counts = nested.get("selected_variant_counts", {}) or {}
    if not counts:
        return fallback_variant, "pipeline_selected_fallback", {}

    # Deterministic tie-break: highest count, then lexical variant key
    items = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))
    selected_variant = items[0][0]
    normalized_counts = {k: v for k, v in sorted(items, key=lambda kv: kv[0])}
    return selected_variant, "nested_majority", normalized_counts


def _resolve_source_checkpoint(
    *,
    dataset: str,
    path_key: str,
    variant: str,
    project_root: str,
    path_defaults: Mapping[str, Any],
) -> str:
    model_files = path_defaults.get("model_files", {})
    rel_name = str(model_files.get(variant, f"{variant}.pth"))
    ckpt = os.path.join(project_root, "models", dataset, path_key, rel_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint for {path_key}/{variant}: {ckpt}")
    return ckpt


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    shutil.copy2(src, dst)
    return True


def _bundle_one_path(
    *,
    dataset: str,
    path_key: str,
    selected_variant: str,
    selection_source: str,
    selected_variant_counts: Mapping[str, int],
    variant_scores: Mapping[str, tuple[Any, Any]],
    trace_ids_val,
    trace_ids_attack,
    calib_ratio: float,
    seed: int,
    source_checkpoint: str,
    bundle_root: str,
) -> dict:
    if selected_variant not in variant_scores:
        raise ValueError(
            f"{path_key}: selected variant '{selected_variant}' is missing in scored variants "
            f"{sorted(variant_scores.keys())}."
        )

    scores_val, scores_atk = variant_scores[selected_variant]
    holdout = run_holdout_trace_split(
        scores_val=scores_val,
        scores_atk=scores_atk,
        trace_ids_val=trace_ids_val,
        trace_ids_atk=trace_ids_attack,
        calib_ratio=calib_ratio,
        seed=seed,
    )

    path_bundle_dir = os.path.join(bundle_root, path_key)
    os.makedirs(path_bundle_dir, exist_ok=True)

    bundle_model_path = os.path.join(path_bundle_dir, "model.pth")
    bundle_meta_path = bundle_model_path + ".meta.json"
    bundle_tau_path = os.path.join(path_bundle_dir, "threshold_tau.json")
    bundle_manifest_path = os.path.join(path_bundle_dir, "manifest.json")

    shutil.copy2(source_checkpoint, bundle_model_path)
    source_meta = source_checkpoint + ".meta.json"
    meta_copied = _copy_if_exists(source_meta, bundle_meta_path)

    tau_payload = {
        "dataset": dataset,
        "path": path_key,
        "selected_variant": selected_variant,
        "selection_source": selection_source,
        "selected_variant_counts": dict(selected_variant_counts),
        "calibration_protocol": "trace_holdout_split",
        "calib_ratio": float(calib_ratio),
        "seed": int(seed),
        "tau_star": float(holdout["threshold_tau"]),
        "F1_calib": float(holdout["F1_calib"]),
        "calib_metrics": holdout["calib_metrics"],
        "test_metrics_reference": holdout["test_metrics"],
        "sizes": holdout["sizes"],
        "created_at_utc": _utc_now_iso(),
    }
    with open(bundle_tau_path, "w", encoding="utf-8") as f:
        json.dump(tau_payload, f, indent=2, default=_json_default)

    manifest_payload = {
        "dataset": dataset,
        "path": path_key,
        "selected_variant": selected_variant,
        "selection_source": selection_source,
        "selected_variant_counts": dict(selected_variant_counts),
        "source_checkpoint": source_checkpoint,
        "bundle_checkpoint": bundle_model_path,
        "source_meta": source_meta if os.path.exists(source_meta) else None,
        "bundle_meta": bundle_meta_path if meta_copied else None,
        "tau_file": bundle_tau_path,
        "calibration_protocol": "trace_holdout_split",
        "calib_ratio": float(calib_ratio),
        "seed": int(seed),
        "reference_test_metrics": holdout["test_metrics"],
        "created_at_utc": _utc_now_iso(),
    }
    with open(bundle_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, default=_json_default)

    print(
        f"Bundle {path_key}: variant={selected_variant} | "
        f"tau*={holdout['threshold_tau']:.6f} | "
        f"F1_test_ref={holdout['test_metrics']['F1']:.4f} | "
        f"AUC_ref={holdout['test_metrics']['AUC_ROC']:.4f}"
    )

    return {
        "path": path_key,
        "selected_variant": selected_variant,
        "selection_source": selection_source,
        "selected_variant_counts": dict(selected_variant_counts),
        "tau_star": float(holdout["threshold_tau"]),
        "F1_calib": float(holdout["F1_calib"]),
        "test_metrics_reference": holdout["test_metrics"],
        "sizes": holdout["sizes"],
        "source_checkpoint": source_checkpoint,
        "bundle_checkpoint": bundle_model_path,
        "bundle_meta": bundle_meta_path if meta_copied else None,
        "threshold_file": bundle_tau_path,
        "manifest_file": bundle_manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inference-only bundle for Path B/C.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset namespace: 32bit or 64bit.")
    parser.add_argument(
        "--paths",
        type=str,
        default="both",
        choices=["path_b", "path_c", "both"],
        help="Which paths to include in bundle.",
    )
    parser.add_argument(
        "--selection_source",
        type=str,
        default="nested_majority",
        choices=["nested_majority", "pipeline_selected"],
        help="How selected variants are chosen.",
    )
    parser.add_argument(
        "--protocol_report",
        type=str,
        default=None,
        help="Optional protocol comparison JSON path. Used when selection_source=nested_majority.",
    )
    parser.add_argument("--calib_ratio", type=float, default=0.5, help="Trace-level calibration ratio per class.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for Path C scoring.")
    parser.add_argument(
        "--max_windows_per_split",
        type=int,
        default=None,
        help="Optional cap per split before scoring.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    if not (0.0 < args.calib_ratio < 1.0):
        raise ValueError("--calib_ratio must be in (0, 1).")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

    set_seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    bundle_root = os.path.join(project_root, "models", args.dataset, "inference")
    os.makedirs(bundle_root, exist_ok=True)

    report_path = args.protocol_report or os.path.join(logs_dir, "path_bc_eval_protocol_comparison.json")
    report_payload = None
    if args.selection_source == "nested_majority":
        if os.path.exists(report_path):
            report_payload = _safe_load_json(report_path)
        else:
            print(
                f"[warning] Protocol report not found at {report_path}. "
                "Falling back to pipeline selected_model."
            )

    path_b_defaults = get_path_b_defaults(args.dataset)
    path_c_defaults = get_path_c_defaults(args.dataset)

    print("=" * 72)
    print(f"BUILD INFERENCE BUNDLE ({args.dataset.upper()})")
    print(f"paths={args.paths} | selection_source={args.selection_source} | seed={args.seed}")
    print(f"calib_ratio={args.calib_ratio} | batch_size={args.batch_size} | max_windows={args.max_windows_per_split}")
    print("=" * 72)

    prepared = load_prepared_data(
        dataset=args.dataset,
        max_windows_per_split=args.max_windows_per_split,
        seed=args.seed,
    )
    print(
        f"Data val={prepared.x_val_w.shape} (traces={len(set(prepared.trace_ids_val.tolist()))}) | "
        f"attack={prepared.x_attack_w.shape} (traces={len(set(prepared.trace_ids_attack.tolist()))})"
    )

    bundle_results: dict[str, Any] = {}

    if args.paths in {"path_b", "both"}:
        scores_b = score_path_b_variants(
            prepared=prepared,
            model_dir_root=os.path.join(project_root, "models", args.dataset),
            path_b_defaults=path_b_defaults,
        )
        fallback_variant = _path_b_selected_to_variant(str(path_b_defaults.get("selected_model", "cnn1d_ae")))
        if args.selection_source == "nested_majority":
            selected_variant, sel_source, selected_counts = _select_variant_from_report(
                report_payload=report_payload,
                path_key="path_b",
                fallback_variant=fallback_variant,
            )
        else:
            selected_variant, sel_source, selected_counts = fallback_variant, "pipeline_selected", {}

        source_ckpt = _resolve_source_checkpoint(
            dataset=args.dataset,
            path_key="path_b",
            variant=selected_variant,
            project_root=project_root,
            path_defaults=path_b_defaults,
        )
        bundle_results["path_b"] = _bundle_one_path(
            dataset=args.dataset,
            path_key="path_b",
            selected_variant=selected_variant,
            selection_source=sel_source,
            selected_variant_counts=selected_counts,
            variant_scores=scores_b,
            trace_ids_val=prepared.trace_ids_val,
            trace_ids_attack=prepared.trace_ids_attack,
            calib_ratio=args.calib_ratio,
            seed=args.seed,
            source_checkpoint=source_ckpt,
            bundle_root=bundle_root,
        )

    if args.paths in {"path_c", "both"}:
        scores_c = score_path_c_variants(
            prepared=prepared,
            model_dir_root=os.path.join(project_root, "models", args.dataset),
            path_c_defaults=path_c_defaults,
            batch_size=args.batch_size,
        )
        fallback_variant = _path_c_selected_to_variant(str(path_c_defaults.get("selected_model", "gru_predictor")))
        if args.selection_source == "nested_majority":
            selected_variant, sel_source, selected_counts = _select_variant_from_report(
                report_payload=report_payload,
                path_key="path_c",
                fallback_variant=fallback_variant,
            )
        else:
            selected_variant, sel_source, selected_counts = fallback_variant, "pipeline_selected", {}

        source_ckpt = _resolve_source_checkpoint(
            dataset=args.dataset,
            path_key="path_c",
            variant=selected_variant,
            project_root=project_root,
            path_defaults=path_c_defaults,
        )
        bundle_results["path_c"] = _bundle_one_path(
            dataset=args.dataset,
            path_key="path_c",
            selected_variant=selected_variant,
            selection_source=sel_source,
            selected_variant_counts=selected_counts,
            variant_scores=scores_c,
            trace_ids_val=prepared.trace_ids_val,
            trace_ids_attack=prepared.trace_ids_attack,
            calib_ratio=args.calib_ratio,
            seed=args.seed,
            source_checkpoint=source_ckpt,
            bundle_root=bundle_root,
        )

    summary_payload = {
        "dataset": args.dataset,
        "config": {
            "paths": args.paths,
            "selection_source": args.selection_source,
            "protocol_report": report_path if args.selection_source == "nested_majority" else None,
            "calib_ratio": float(args.calib_ratio),
            "batch_size": int(args.batch_size),
            "max_windows_per_split": int(args.max_windows_per_split) if args.max_windows_per_split is not None else None,
            "seed": int(args.seed),
        },
        "bundle_root": bundle_root,
        "created_at_utc": _utc_now_iso(),
        "results": bundle_results,
    }

    summary_log_path = os.path.join(logs_dir, "inference_bundle_summary.json")
    with open(summary_log_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=_json_default)

    top_manifest_path = os.path.join(bundle_root, "inference_bundle_manifest.json")
    with open(top_manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=_json_default)

    print("\nBundle build complete:")
    print(f" Summary log: {summary_log_path}")
    print(f" Top manifest: {top_manifest_path}")


if __name__ == "__main__":
    main()


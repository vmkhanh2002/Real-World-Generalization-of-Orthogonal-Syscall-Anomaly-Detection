"""
Export Paper-Ready Nested CV Table for Path B/C
===============================================

Reads:
 experiments/{dataset}/logs/path_bc_eval_protocol_comparison.json

Writes:
 - experiments/{dataset}/logs/paper_nested_protocol_table.md
 - experiments/{dataset}/logs/paper_nested_protocol_table.json

Table fields (nested_cv_trace_level):
 AUC, AUPR, F1, Precision, Recall, selected_variant_counts
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Mapping

from pipeline_config import normalize_dataset_name


def _fmt_mean_std(metric_node: Mapping[str, Any], digits: int = 4) -> str:
    mean = metric_node.get("mean") if isinstance(metric_node, Mapping) else None
    std = metric_node.get("std") if isinstance(metric_node, Mapping) else None
    if mean is None:
        return "n/a"
    if std is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def _compact_variant_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "n/a"
    parts = []
    for key in sorted(counts.keys()):
        parts.append(f"{key}:{counts[key]}")
    return ", ".join(parts)


def _extract_row(path_name: str, path_payload: Mapping[str, Any]) -> dict:
    nested = path_payload["protocols"]["nested_cv_trace_level"]
    agg = nested["aggregate_test_metrics"]
    return {
        "Path": path_name,
        "AUC_ROC": _fmt_mean_std(agg.get("AUC_ROC", {})),
        "AUPR": _fmt_mean_std(agg.get("AUPR", {})),
        "F1": _fmt_mean_std(agg.get("F1", {})),
        "Precision": _fmt_mean_std(agg.get("Precision", {})),
        "Recall": _fmt_mean_std(agg.get("Recall", {})),
        "selected_variant_counts": dict(nested.get("selected_variant_counts", {})),
        "selected_variant_counts_compact": _compact_variant_counts(nested.get("selected_variant_counts", {})),
    }


def _build_markdown(dataset: str, rows: list[dict], src_json: str) -> str:
    lines = []
    lines.append(f"# Nested CV Paper Table ({dataset})")
    lines.append("")
    lines.append("Protocol: `nested_cv_trace_level` (trace-level outer/inner CV)")
    lines.append("")
    lines.append(f"Source: `{src_json}`")
    lines.append("")
    lines.append(
        "| Path | AUC_ROC (mean +/- std) | AUPR (mean +/- std) | F1 (mean +/- std) | Precision (mean +/- std) | Recall (mean +/- std) | selected_variant_counts |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {Path} | {AUC_ROC} | {AUPR} | {F1} | {Precision} | {Recall} | {selected_variant_counts_compact} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("> Holdout metrics are retained in the comparison JSON for reference only.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready nested CV table for Path B/C.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset namespace (32bit or 64bit).")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional explicit input JSON path. Defaults to experiments/{dataset}/logs/path_bc_eval_protocol_comparison.json",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "experiments", dataset, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    input_json = args.input or os.path.join(logs_dir, "path_bc_eval_protocol_comparison.json")
    if not os.path.exists(input_json):
        raise FileNotFoundError(
            f"Missing protocol comparison file: {input_json}\n"
            f"Run scripts/compare_path_bc_eval_protocols.py first."
        )

    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for path_key, label in (("path_b", "Path B"), ("path_c", "Path C")):
        if path_key in payload.get("results", {}):
            rows.append(_extract_row(label, payload["results"][path_key]))

    if not rows:
        raise ValueError(f"No path_b/path_c entries found in {input_json}")

    md_output = os.path.join(logs_dir, "paper_nested_protocol_table.md")
    json_output = os.path.join(logs_dir, "paper_nested_protocol_table.json")

    md_content = _build_markdown(dataset=dataset, rows=rows, src_json=input_json)
    with open(md_output, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_payload = {
        "dataset": dataset,
        "source": input_json,
        "rows": rows,
    }
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print("Export complete:")
    print(f" Markdown: {md_output}")
    print(f" JSON: {json_output}")


if __name__ == "__main__":
    main()

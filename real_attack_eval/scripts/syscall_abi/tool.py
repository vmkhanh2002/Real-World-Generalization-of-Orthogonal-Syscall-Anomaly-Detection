"""CLI entry point for ABI detection and syscall lookup helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .detection import build_detection_report, print_detection_report, save_detection_report
    from .tables import (
        load_syscall_abi_mapping,
        lookup_syscall_name,
        lookup_syscall_number,
    )
except ImportError:
    from detection import build_detection_report, print_detection_report, save_detection_report  # type: ignore
    from tables import (  # type: ignore
        load_syscall_abi_mapping,
        lookup_syscall_name,
        lookup_syscall_number,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect syscall ABI and perform forward/reverse syscall lookups using ABI-specific JSON mappings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect the best matching ABI for a numeric syscall dataset.")
    detect_parser.add_argument(
 "--dataset-root",
        required=True,
        help="Root directory containing numeric syscall trace .txt files.",
    )
    detect_parser.add_argument(
 "--sample-files",
        type=int,
        default=0,
        help="Optional cap on number of files to inspect. 0 means all files.",
    )
    detect_parser.add_argument(
 "--top-k",
        type=int,
        default=20,
        help="How many most-frequent numeric IDs to include in evidence tables.",
    )
    detect_parser.add_argument(
 "--output-json",
        default=None,
        help="Optional path to save the ABI scoring report as JSON.",
    )

    for subcommand in ("lookup-name", "lookup-number"):
        lookup_parser = subparsers.add_parser(
            subcommand,
            help="Resolve a syscall name or number using either an explicit ABI or an auto-detected dataset ABI.",
        )
        lookup_parser.add_argument(
 "--abi",
            default=None,
            help="Explicit ABI to use (i386, x86_64, arm, arm64/asm-generic). If omitted, --dataset-root is used for auto-detection.",
        )
        lookup_parser.add_argument(
 "--dataset-root",
            default=None,
            help="Optional dataset root to auto-detect ABI when --abi is not provided.",
        )
        lookup_parser.add_argument(
 "--sample-files",
            type=int,
            default=0,
            help="Optional cap on number of files to inspect during auto-detection.",
        )
        lookup_parser.add_argument(
 "--top-k",
            type=int,
            default=20,
            help="How many most-frequent numeric IDs to include in the auto-detection evidence tables.",
        )
        lookup_parser.add_argument(
 "--output-json",
            default=None,
            help="Optional path to save the lookup result as JSON.",
        )

    subparsers.choices["lookup-name"].add_argument("--name", required=True, help="Syscall name to resolve.")
    subparsers.choices["lookup-number"].add_argument("--number", required=True, type=int, help="Syscall number to resolve.")
    return parser


def resolve_mapping_from_args(
    project_root: Path,
    abi: str | None,
    dataset_root: str | None,
    sample_files: int,
    top_k: int,
) -> tuple[Any, dict[str, Any] | None]:
    if abi:
        return load_syscall_abi_mapping(abi, project_root=project_root), None
    if not dataset_root:
        raise ValueError("Either --abi or --dataset-root must be provided.")

    resolved_dataset_root = (project_root / dataset_root).resolve()
    if not resolved_dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {resolved_dataset_root}")
    detection_report = build_detection_report(
        dataset_root=resolved_dataset_root,
        sample_files=sample_files,
        top_k=top_k,
        project_root=project_root,
    )
    mapping = load_syscall_abi_mapping(detection_report["recommended_abi"], project_root=project_root)
    return mapping, detection_report


def print_detection_banner(detection_report: dict[str, Any] | None) -> None:
    if not detection_report:
        return
    print("ABI auto-detection:")
    print(
        f" resolved={detection_report['recommended_abi_display']} "
        f"({detection_report['recommended_abi']})"
    )
    print(f" mapping_json={detection_report['recommended_mapping_json']}")
    print("")


def save_json_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.command == "detect":
        dataset_root = (project_root / args.dataset_root).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        report = build_detection_report(
            dataset_root=dataset_root,
            sample_files=int(args.sample_files),
            top_k=int(args.top_k),
            project_root=project_root,
        )
        print_detection_report(report)
        if args.output_json:
            output_path = (project_root / args.output_json).resolve()
            save_detection_report(report, output_path)
            print("")
            print(f"Saved JSON report to: {output_path}")
        return

    mapping, detection_report = resolve_mapping_from_args(
        project_root=project_root,
        abi=getattr(args, "abi", None),
        dataset_root=getattr(args, "dataset_root", None),
        sample_files=int(getattr(args, "sample_files", 0)),
        top_k=int(getattr(args, "top_k", 20)),
    )
    print_detection_banner(detection_report)
    print(f"Resolved ABI: {mapping.display_name} ({mapping.abi})")
    print(f"Resolved mapping JSON: {mapping.source_path}")
    print("")

    if args.command == "lookup-name":
        numbers = sorted(lookup_syscall_name(mapping.abi, args.name, project_root=project_root))
        if not numbers:
            print(f"No syscall numbers found for name '{args.name}'.")
        else:
            print(f"Name '{args.name}' maps to: {numbers}")
            for number in numbers:
                aliases = sorted(lookup_syscall_number(mapping.abi, number, project_root=project_root))
                print(f" {number}: aliases={aliases}")
        if args.output_json:
            payload = {
                "command": args.command,
                "query_name": args.name,
                "resolved_abi": mapping.abi,
                "resolved_abi_display": mapping.display_name,
                "mapping_json": str(mapping.source_path),
                "numbers": numbers,
                "auto_detection": detection_report,
            }
            output_path = (project_root / args.output_json).resolve()
            save_json_payload(payload, output_path)
            print("")
            print(f"Saved JSON report to: {output_path}")
        return

    names = sorted(lookup_syscall_number(mapping.abi, args.number, project_root=project_root))
    if not names:
        print(f"No syscall names found for number '{args.number}'.")
    else:
        print(f"Number '{args.number}' maps to: {names}")
        for name in names:
            numbers = sorted(lookup_syscall_name(mapping.abi, name, project_root=project_root))
            print(f" {name}: numbers={numbers}")
    if args.output_json:
        payload = {
            "command": args.command,
            "query_number": int(args.number),
            "resolved_abi": mapping.abi,
            "resolved_abi_display": mapping.display_name,
            "mapping_json": str(mapping.source_path),
            "names": names,
            "auto_detection": detection_report,
        }
        output_path = (project_root / args.output_json).resolve()
        save_json_payload(payload, output_path)
        print("")
        print(f"Saved JSON report to: {output_path}")


if __name__ == "__main__":
    main()

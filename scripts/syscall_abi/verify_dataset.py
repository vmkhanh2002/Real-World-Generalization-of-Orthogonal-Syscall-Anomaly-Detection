"""Compatibility wrapper for ABI scoring on numeric syscall datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .defaults import DEFAULT_TOP_K
    from .detection import build_detection_report, print_detection_report, save_detection_report
except ImportError:
    from defaults import DEFAULT_TOP_K  # type: ignore
    from detection import build_detection_report, print_detection_report, save_detection_report  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a numeric syscall dataset against candidate ABI syscall tables."
    )
    parser.add_argument(
 "--dataset-root",
        required=True,
        help="Root directory containing numeric syscall trace .txt files.",
    )
    parser.add_argument(
 "--sample-files",
        type=int,
        default=0,
        help="Optional cap on number of files to inspect. 0 means all files.",
    )
    parser.add_argument(
 "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many most-frequent numeric IDs to include in evidence tables.",
    )
    parser.add_argument(
 "--output-json",
        default=None,
        help="Optional path to save the ABI scoring report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
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


if __name__ == "__main__":
    main()

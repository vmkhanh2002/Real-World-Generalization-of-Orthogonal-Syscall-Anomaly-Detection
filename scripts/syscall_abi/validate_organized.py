"""Validate the organized syscall ABI batch produced by `process_drop.py`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .defaults import DEFAULT_ORGANIZED_ROOT, DEFAULT_TOP_K
    from .detection import build_detection_report
    from .tables import get_supported_abis, normalize_abi_name
except ImportError:
    from defaults import DEFAULT_ORGANIZED_ROOT, DEFAULT_TOP_K  # type: ignore
    from detection import build_detection_report  # type: ignore
    from tables import get_supported_abis, normalize_abi_name  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
 "Validate organized syscall ABI batch outputs after process_drop.py. "
 "Checks report consistency and, for name_to_number batches, re-runs ABI detection "
 "on the converted numeric outputs."
        )
    )
    parser.add_argument(
 "--organized-root",
        default=DEFAULT_ORGANIZED_ROOT.as_posix(),
        help="Root folder created by process_drop.py.",
    )
    parser.add_argument(
 "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many frequent numeric IDs to include in validation detection evidence.",
    )
    parser.add_argument(
 "--output-json",
        default=None,
        help="Optional path to save the validation summary as JSON.",
    )
    parser.add_argument(
 "--prefix",
        default=None,
        help="Optional filename prefix filter, for example synthetic_mixed_.",
    )
    return parser


def load_report(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def matches_prefix(path: Path, prefix: str | None) -> bool:
    if not prefix:
        return True
    return path.name.startswith(prefix)


def summarize_abi_folder(
    base_dir: Path,
    abi: str,
    project_root: Path,
    top_k: int,
    prefix: str | None,
) -> dict[str, Any]:
    source_dir = base_dir / "source"
    converted_dir = base_dir / "converted"
    reports_dir = base_dir / "reports"

    source_files = sorted(path for path in source_dir.glob("*.txt") if path.is_file() and matches_prefix(path, prefix))
    converted_files = sorted(path for path in converted_dir.glob("*.txt") if path.is_file() and matches_prefix(path, prefix))
    report_files = sorted(path for path in reports_dir.glob("*.json") if path.is_file() and matches_prefix(path, prefix))

    report_payloads = [load_report(path) for path in report_files]
    non_converted_reports = [payload for payload in report_payloads if payload.get("status") != "converted"]
    wrong_abi_reports = [payload for payload in report_payloads if payload.get("resolved_abi") != abi]
    unknown_reports = [payload for payload in report_payloads if payload.get("unknown_tokens")]

    directions = sorted({payload.get("direction", "unknown") for payload in report_payloads})
    numeric_detection: dict[str, Any] | None = None
    if converted_files and all(path.name.endswith(".name_to_number.txt") for path in converted_files):
        numeric_detection = build_detection_report(
            dataset_root=converted_dir,
            sample_files=0,
            top_k=int(top_k),
            project_root=project_root,
            candidates=get_supported_abis(),
        )

    passed = (
        len(source_files) == len(converted_files) == len(report_files)
        and not non_converted_reports
        and not wrong_abi_reports
        and not unknown_reports
        and (numeric_detection is None or normalize_abi_name(numeric_detection["recommended_abi"]) == abi)
    )

    return {
 "abi": abi,
 "source_count": len(source_files),
 "converted_count": len(converted_files),
 "report_count": len(report_files),
 "directions": directions,
 "non_converted_report_count": len(non_converted_reports),
 "wrong_abi_report_count": len(wrong_abi_reports),
 "unknown_report_count": len(unknown_reports),
 "numeric_detection": numeric_detection,
 "passed": passed,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Organized root: {summary['organized_root']}")
    print("")
    for item in summary["abis"]:
        print(
            f"[{ 'PASS' if item['passed'] else 'FAIL' }] {item['abi']} "
            f"source={item['source_count']} converted={item['converted_count']} reports={item['report_count']}"
        )
        print(
            f" directions={item['directions']} "
            f"wrong_abi_reports={item['wrong_abi_report_count']} "
            f"unknown_reports={item['unknown_report_count']}"
        )
        detection = item.get("numeric_detection")
        if detection:
            print(
                f" autodetect={detection['recommended_abi']} "
                f"score={detection['ranking'][0]['score']}"
            )
    print("")
    print(f"Overall status: {'PASS' if summary['passed'] else 'FAIL'}")


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    organized_root = (project_root / args.organized_root).resolve()
    if not organized_root.exists():
        raise FileNotFoundError(f"Organized root not found: {organized_root}")

    abi_dirs = [
        path for path in sorted(organized_root.iterdir())
        if path.is_dir() and path.name in get_supported_abis()
    ]
    if not abi_dirs:
        raise ValueError(f"No ABI folders found under organized root: {organized_root}")

    abi_summaries = [
        summarize_abi_folder(
            base_dir=abi_dir,
            abi=abi_dir.name,
            project_root=project_root,
            top_k=int(args.top_k),
            prefix=args.prefix,
        )
        for abi_dir in abi_dirs
    ]
    summary = {
        "organized_root": str(organized_root),
        "prefix": args.prefix,
        "abis": abi_summaries,
        "passed": all(item["passed"] for item in abi_summaries),
    }
    print_summary(summary)

    if args.output_json:
        output_path = (project_root / args.output_json).resolve()
        save_json(output_path, summary)
        print(f"Saved JSON report to: {output_path}")


if __name__ == "__main__":
    main()

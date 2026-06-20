"""Convert inbox syscall traces into ABI-specific organized outputs and reports."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .defaults import DEFAULT_INBOX_ROOT, DEFAULT_ORGANIZED_ROOT, DEFAULT_TOP_K
    from .detection import score_candidate
    from .tables import (
        ABI_DISPLAY_NAMES,
        get_supported_abis,
        load_syscall_abi_mapping,
        normalize_abi_name,
    )
except ImportError:
    from defaults import DEFAULT_INBOX_ROOT, DEFAULT_ORGANIZED_ROOT, DEFAULT_TOP_K  # type: ignore
    from detection import score_candidate  # type: ignore
    from tables import ABI_DISPLAY_NAMES, get_supported_abis, load_syscall_abi_mapping, normalize_abi_name  # type: ignore


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
 "Process dropped syscall trace files from an inbox folder: infer conversion direction, "
 "auto-detect ABI when possible, honor optional inbox/<abi>/ routing hints, move the "
 "source file into the resolved ABI folder, and write a converted output plus a JSON report."
        )
    )
    parser.add_argument(
 "--inbox",
        default=DEFAULT_INBOX_ROOT.as_posix(),
        help="Folder to scan for input .txt files.",
    )
    parser.add_argument(
 "--output-root",
        default=DEFAULT_ORGANIZED_ROOT.as_posix(),
        help="Root folder where ABI-specific output folders will be created.",
    )
    parser.add_argument(
 "--abi",
        default=None,
        help="Optional explicit ABI override. The preferred batch hint is putting files under inbox/<abi>/.",
    )
    parser.add_argument(
 "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many frequent numeric IDs to keep in numeric auto-detection evidence.",
    )
    return parser


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def infer_direction(tokens: list[str]) -> str:
    if not tokens:
        raise ValueError("No syscall tokens found in file.")
    numeric_count = sum(token.isdigit() for token in tokens)
    if numeric_count == len(tokens):
        return "number_to_name"
    if numeric_count == 0:
        return "name_to_number"
    raise ValueError("Mixed numeric and syscall-name tokens are not supported in one file.")


def build_numeric_detection_report(
    tokens: list[str],
    top_k: int,
    project_root: Path,
) -> dict[str, Any]:
    counts = Counter(int(token) for token in tokens)
    ranking = sorted(
        (
            score_candidate(counts=counts, top_k=top_k, abi=abi, project_root=project_root)
            for abi in get_supported_abis()
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    best = ranking[0]
    return {
 "method": "numeric_auto_detect",
 "numeric_tokens": len(tokens),
 "unique_numeric_ids": len(counts),
 "ranking": ranking,
 "resolved_abi": best["abi"],
 "resolved_abi_display": best["abi_display"],
 "resolved_mapping_json": best["mapping_json"],
    }


def build_name_detection_report(
    tokens: list[str],
    project_root: Path,
) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    for abi in get_supported_abis():
        mapping = load_syscall_abi_mapping(abi, project_root=project_root)
        known_count = sum(1 for token in tokens if mapping.primary_number_for_name(token) is not None)
        ranking.append(
            {
 "abi": mapping.abi,
 "abi_display": mapping.display_name,
 "known_tokens": known_count,
 "coverage": round(known_count / max(1, len(tokens)), 6),
 "mapping_json": str(mapping.source_path),
            }
        )
    ranking.sort(key=lambda item: (item["known_tokens"], item["coverage"]), reverse=True)
    best = ranking[0]
    tied = [item for item in ranking if item["known_tokens"] == best["known_tokens"]]
    return {
 "method": "name_coverage_auto_detect",
 "name_tokens": len(tokens),
 "ranking": ranking,
 "resolved_abi": best["abi"] if best["known_tokens"] > 0 and len(tied) == 1 else None,
 "resolved_abi_display": best["abi_display"] if best["known_tokens"] > 0 and len(tied) == 1 else None,
 "resolved_mapping_json": best["mapping_json"] if best["known_tokens"] > 0 and len(tied) == 1 else None,
 "status": "resolved" if best["known_tokens"] > 0 and len(tied) == 1 else "ambiguous",
 "ambiguous_abis": [item["abi"] for item in tied] if len(tied) > 1 else [],
    }


def build_explicit_detection_report(
    normalized_abi: str,
    project_root: Path,
    method: str,
    source: str,
) -> dict[str, Any]:
    mapping = load_syscall_abi_mapping(normalized_abi, project_root=project_root)
    return {
 "method": method,
 "source": source,
 "resolved_abi": normalized_abi,
 "resolved_abi_display": ABI_DISPLAY_NAMES[normalized_abi],
 "resolved_mapping_json": str(mapping.source_path),
    }


def infer_abi_from_inbox_subpath(file_path: Path, inbox: Path) -> str | None:
    try:
        relative_path = file_path.resolve().relative_to(inbox.resolve())
    except ValueError:
        return None

    parent = relative_path.parent
    if parent == Path("."):
        return None

    for part in parent.parts:
        try:
            return normalize_abi_name(part)
        except ValueError:
            continue
    return None


def resolve_explicit_abi_hint(
    file_path: Path,
    inbox: Path,
    cli_abi: str | None,
    project_root: Path,
) -> tuple[str | None, dict[str, Any] | None]:
    cli_hint = normalize_abi_name(cli_abi) if cli_abi else None
    inbox_hint = infer_abi_from_inbox_subpath(file_path=file_path, inbox=inbox)

    if cli_hint and inbox_hint and cli_hint != inbox_hint:
        raise ValueError(
 f"Conflicting ABI hints for {file_path.name}: --abi={cli_hint} but inbox subfolder implies {inbox_hint}."
        )

    if cli_hint:
        return cli_hint, build_explicit_detection_report(
            normalized_abi=cli_hint,
            project_root=project_root,
            method="explicit_abi",
            source="cli",
        )

    if inbox_hint:
        return inbox_hint, build_explicit_detection_report(
            normalized_abi=inbox_hint,
            project_root=project_root,
            method="inbox_subdir_abi",
            source=str(file_path.parent.relative_to(inbox)),
        )

    return None, None


def make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}__{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def resolve_output_dirs(output_root: Path, abi_folder: str) -> dict[str, Path]:
    base_dir = output_root / abi_folder
    source_dir = base_dir / "source"
    converted_dir = base_dir / "converted"
    reports_dir = base_dir / "reports"
    for directory in (source_dir, converted_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
 "base": base_dir,
 "source": source_dir,
 "converted": converted_dir,
 "reports": reports_dir,
    }


def convert_tokens(tokens: list[str], direction: str, mapping: Any) -> tuple[list[str], list[str]]:
    converted: list[str] = []
    unknown_tokens: list[str] = []
    if direction == "number_to_name":
        for token in tokens:
            name = mapping.primary_name_for_number(int(token))
            if name is None:
                unknown_tokens.append(token)
                converted.append(f"UNKNOWN_NUMBER_{token}")
            else:
                converted.append(name)
        return converted, unknown_tokens

    for token in tokens:
        number = mapping.primary_number_for_name(token)
        if number is None:
            unknown_tokens.append(token)
            converted.append(f"UNKNOWN_SYSCALL_{token}")
        else:
            converted.append(str(number))
    return converted, unknown_tokens


def process_file(
    file_path: Path,
    inbox: Path,
    output_root: Path,
    project_root: Path,
    explicit_abi: str | None,
    top_k: int,
) -> dict[str, Any]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize_text(text)
    direction = infer_direction(tokens)
    normalized_abi, detection_report = resolve_explicit_abi_hint(
        file_path=file_path,
        inbox=inbox,
        cli_abi=explicit_abi,
        project_root=project_root,
    )

    if not normalized_abi:
        if direction == "number_to_name":
            detection_report = build_numeric_detection_report(tokens=tokens, top_k=top_k, project_root=project_root)
            normalized_abi = str(detection_report["resolved_abi"])
        else:
            detection_report = build_name_detection_report(tokens=tokens, project_root=project_root)
            normalized_abi = detection_report["resolved_abi"]

    if not normalized_abi:
        output_dirs = resolve_output_dirs(output_root=output_root, abi_folder="unresolved")
        source_target = make_unique_path(output_dirs["source"] / file_path.name)
        report_target = make_unique_path(output_dirs["reports"] / f"{file_path.stem}.report.json")
        shutil.move(str(file_path), str(source_target))
        report_payload = {
 "source_file": str(source_target),
 "direction": direction,
 "status": "unresolved",
 "reason": "Could not uniquely determine ABI from syscall-name input. Re-run with --abi or place the file under inbox/<abi>/.",
 "detection": detection_report,
        }
        report_target.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
        return report_payload

    mapping = load_syscall_abi_mapping(normalized_abi, project_root=project_root)
    converted_tokens, unknown_tokens = convert_tokens(tokens=tokens, direction=direction, mapping=mapping)

    output_dirs = resolve_output_dirs(output_root=output_root, abi_folder=normalized_abi)
    source_target = make_unique_path(output_dirs["source"] / file_path.name)
    converted_target = make_unique_path(output_dirs["converted"] / f"{file_path.stem}.{direction}.txt")
    report_target = make_unique_path(output_dirs["reports"] / f"{file_path.stem}.report.json")

    shutil.move(str(file_path), str(source_target))
    converted_target.write_text(" ".join(converted_tokens) + "\n", encoding="utf-8")

    report_payload = {
 "source_file": str(source_target),
 "converted_file": str(converted_target),
 "direction": direction,
 "status": "converted",
 "resolved_abi": mapping.abi,
 "resolved_abi_display": mapping.display_name,
 "resolved_mapping_json": str(mapping.source_path),
 "unknown_tokens": unknown_tokens,
 "detection": detection_report,
    }
    report_target.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    return report_payload


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    inbox = (project_root / args.inbox).resolve()
    output_root = (project_root / args.output_root).resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in inbox.rglob("*.txt") if path.is_file())
    if not files:
        print(f"No .txt files found in inbox: {inbox}")
        return

    for file_path in files:
        try:
            report = process_file(
                file_path=file_path,
                inbox=inbox,
                output_root=output_root,
                project_root=project_root,
                explicit_abi=args.abi,
                top_k=int(args.top_k),
            )
        except Exception as exc:
            print(f"FAILED {file_path.name}: {exc}")
            continue

        if report["status"] == "unresolved":
            print(f"UNRESOLVED {Path(report['source_file']).name} to {report['reason']}")
            continue

        print(
            f"OK {Path(report['source_file']).name} to "
            f"{report['resolved_abi']} to {Path(report['converted_file']).name}"
        )


if __name__ == "__main__":
    main()

"""Score numeric syscall traces against the supported ABI mapping tables."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .tables import (
        ABI_DISPLAY_NAMES,
        COMMON_USERSPACE_SYSCALLS,
        get_supported_abis,
        get_syscall_maps,
        load_syscall_abi_mapping,
        normalize_abi_name,
    )
except ImportError:
    from tables import (  # type: ignore
        ABI_DISPLAY_NAMES,
        COMMON_USERSPACE_SYSCALLS,
        get_supported_abis,
        get_syscall_maps,
        load_syscall_abi_mapping,
        normalize_abi_name,
    )

ANCHOR_NAME_GROUPS = (
    ("read",),
    ("write",),
    ("open", "openat"),
    ("close",),
    ("mmap", "mmap2"),
    ("fstat", "fstat64", "newfstatat", "fstatat64"),
    ("brk",),
    ("ioctl",),
    ("poll", "ppoll"),
    ("dup", "dup2", "dup3"),
    ("pipe", "pipe2"),
    ("exit_group", "exit"),
    ("clock_gettime",),
    ("futex",),
    ("set_tid_address",),
    ("getdents64", "getdents"),
    ("set_robust_list",),
    ("prlimit64",),
)


def load_numeric_counts(dataset_root: Path, sample_files: int = 0) -> tuple[Counter[int], int, int]:
    counter: Counter[int] = Counter()
    total_files = 0
    total_tokens = 0
    txt_files = sorted(path for path in dataset_root.rglob("*.txt") if path.is_file())
    if sample_files and sample_files > 0:
        txt_files = txt_files[:sample_files]
    for path in txt_files:
        total_files += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in text.split():
            if token.isdigit():
                number = int(token)
                counter[number] += 1
                total_tokens += 1
    return counter, total_files, total_tokens


def is_common_name(name: str) -> bool:
    return name in COMMON_USERSPACE_SYSCALLS


def score_candidate(
    counts: Counter[int],
    top_k: int,
    abi: str,
    project_root: Path,
) -> dict[str, Any]:
    normalized_abi = normalize_abi_name(abi)
    name_to_nums, num_to_names = get_syscall_maps(normalized_abi, project_root=project_root)
    table_ids = set(num_to_names)
    observed_ids = set(counts)
    total_tokens = sum(counts.values())
    known_token_count = sum(freq for num, freq in counts.items() if num in table_ids)
    known_unique_count = sum(1 for num in observed_ids if num in table_ids)

    top_ids = counts.most_common(top_k)
    top_token_total = sum(freq for _, freq in top_ids) or 1
    common_top_token_count = 0
    common_top_id_count = 0
    top_translation_rows: list[dict[str, Any]] = []
    for number, freq in top_ids:
        names = sorted(num_to_names.get(number, set()))
        common_here = any(is_common_name(name) for name in names)
        if common_here:
            common_top_token_count += freq
            common_top_id_count += 1
        top_translation_rows.append(
            {
 "id": number,
 "count": freq,
 "names": names,
 "is_common_userspace": common_here,
            }
        )

    candidate_anchor_ids: set[int] = set()
    for group in ANCHOR_NAME_GROUPS:
        for name in group:
            candidate_anchor_ids.update(name_to_nums.get(name, set()))

    anchor_rows: list[dict[str, Any]] = []
    anchor_total = 0
    for number in sorted(candidate_anchor_ids):
        if number not in counts:
            continue
        freq = counts[number]
        names = sorted(num_to_names.get(number, set()))
        common_here = any(is_common_name(name) for name in names)
        anchor_total += freq
        anchor_rows.append(
            {
 "id": number,
 "count": freq,
 "names": names,
 "is_common_userspace": common_here,
            }
        )

    known_token_coverage = known_token_count / max(1, total_tokens)
    known_unique_coverage = known_unique_count / max(1, len(observed_ids))
    common_top_token_coverage = common_top_token_count / top_token_total
    common_top_id_coverage = common_top_id_count / max(1, len(top_ids))
    anchor_token_coverage = anchor_total / max(1, total_tokens)

    final_score = 100.0 * (
        0.35 * common_top_token_coverage
        + 0.30 * known_token_coverage
        + 0.20 * anchor_token_coverage
        + 0.10 * common_top_id_coverage
        + 0.05 * known_unique_coverage
    )

    resolved_mapping = load_syscall_abi_mapping(normalized_abi, project_root=project_root)
    return {
 "abi": normalized_abi,
 "abi_display": ABI_DISPLAY_NAMES[normalized_abi],
 "mapping_json": str(resolved_mapping.source_path),
 "known_token_coverage": round(known_token_coverage, 6),
 "known_unique_coverage": round(known_unique_coverage, 6),
 "common_top_token_coverage": round(common_top_token_coverage, 6),
 "common_top_id_coverage": round(common_top_id_coverage, 6),
 "anchor_token_coverage": round(anchor_token_coverage, 6),
 "score": round(final_score, 3),
 "top_translations": top_translation_rows,
 "anchor_translations": anchor_rows,
 "table_size": len(name_to_nums),
    }


def build_detection_report(
    dataset_root: Path,
    sample_files: int,
    top_k: int,
    project_root: Path,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    counts, files_inspected, numeric_tokens = load_numeric_counts(
        dataset_root=dataset_root,
        sample_files=int(sample_files),
    )
    if not counts:
        raise ValueError(f"No numeric syscall tokens found under: {dataset_root}")

    normalized_candidates = [normalize_abi_name(abi) for abi in (candidates or get_supported_abis())]
    ranking = sorted(
        (
            score_candidate(counts, top_k=int(top_k), abi=abi, project_root=project_root)
            for abi in normalized_candidates
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    best = ranking[0]
    return {
 "dataset_root": str(dataset_root),
 "files_inspected": files_inspected,
 "numeric_tokens": numeric_tokens,
 "unique_numeric_ids": len(counts),
 "top_numeric_ids": counts.most_common(int(top_k)),
 "ranking": ranking,
 "recommended_abi": best["abi"],
 "recommended_abi_display": best["abi_display"],
 "recommended_mapping_json": best["mapping_json"],
    }


def print_detection_report(report: dict[str, Any]) -> None:
    print(f"Dataset root: {report['dataset_root']}")
    print(
        f"Files inspected: {report['files_inspected']} | "
        f"Numeric tokens: {report['numeric_tokens']} | "
        f"Unique numeric IDs: {report['unique_numeric_ids']}"
    )
    print("")
    print("Ranking:")
    for rank, item in enumerate(report["ranking"], start=1):
        print(
            f" {rank}. {item['abi_display']}: "
            f"score={item['score']:.3f}, "
            f"known_token_coverage={item['known_token_coverage']:.3f}, "
            f"common_top_token_coverage={item['common_top_token_coverage']:.3f}, "
            f"anchor_token_coverage={item['anchor_token_coverage']:.3f}"
        )
        print(f" mapping_json={item['mapping_json']}")
    print("")
    best = report["ranking"][0]
    print(f"Recommended ABI: {best['abi_display']} ({best['abi']})")
    print(f"Resolved mapping JSON: {best['mapping_json']}")
    print("")
    print(f"Top {len(best['top_translations'])} ID translations for the best match:")
    for row in best["top_translations"]:
        names = ", ".join(row["names"]) if row["names"] else "<unknown>"
        print(f" id={row['id']:>4} count={row['count']:>8} names={names}")
    print("")
    print("Discriminative anchor IDs under the best match:")
    for row in best["anchor_translations"]:
        names = ", ".join(row["names"]) if row["names"] else "<unknown>"
        print(f" id={row['id']:>4} count={row['count']:>8} names={names}")


def save_detection_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

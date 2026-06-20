"""Build a deterministic realtime evaluation manifest from an ADFA-LD subset."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ATTACK_DIR_SUFFIX_RE = re.compile(r"_(\d+)$")


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import defaults as rt_defaults  # pylint: disable=import-error

    parser = argparse.ArgumentParser(
        description=(
 "Sample a realtime ADFA-LD benchmark subset and write a manifest plus "
 "an input list for the realtime evaluation pipeline."
        )
    )
    parser.add_argument(
 "--dataset-root",
        default=str(rt_defaults.DEFAULT_DATASET_ROOT),
        help="Path to the ADFA-LD dataset root.",
    )
    parser.add_argument(
 "--validation-fraction",
        type=float,
        default=0.5,
        help="Fraction of Validation_Data_Master files to keep.",
    )
    parser.add_argument(
 "--attack-fraction",
        type=float,
        default=0.5,
        help="Fraction of files to keep inside each attack family.",
    )
    parser.add_argument(
 "--seed",
        type=int,
        default=20260325,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
 "--manifest-out",
        default=str(rt_defaults.DEFAULT_HALF_MANIFEST),
        help="Output manifest JSON path.",
    )
    parser.add_argument(
 "--input-list-out",
        default=str(rt_defaults.DEFAULT_HALF_INPUTS),
        help="Output text file listing sampled input paths.",
    )
    return parser


def resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path)


def path_for_output(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def ensure_fraction(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return float(value)


def sample_count_floor(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return total
    return int(total * fraction)


def sample_files(files: list[Path], keep_count: int, rng: random.Random) -> list[Path]:
    if keep_count <= 0:
        return []
    if keep_count >= len(files):
        return list(files)
    selected = rng.sample(files, keep_count)
    return sorted(selected)


def attack_family_from_dir(dir_name: str) -> str:
    return ATTACK_DIR_SUFFIX_RE.sub("", dir_name)


def build_entry(
    *,
    project_root: Path,
    file_path: Path,
    intended_label: str,
    target_preset: str | None,
    source_group: str,
) -> dict[str, Any]:
    relative_path = path_for_output(project_root, file_path)
    return {
 "sample_id": file_path.stem,
 "intended_label": intended_label,
 "target_preset": target_preset,
 "source_group": source_group,
 "source_file": relative_path,
 "expected_detect_file": relative_path,
    }


def allocate_family_keep_counts(
    *,
    subdir_file_map: dict[str, list[Path]],
    family_keep: int,
    fraction: float,
) -> dict[str, int]:
    quota_by_dir = {dir_name: 0 for dir_name in subdir_file_map}
    if family_keep <= 0:
        return quota_by_dir

    fractional_parts: list[tuple[float, str]] = []
    base_total = 0
    for dir_name, files in sorted(subdir_file_map.items()):
        ideal = len(files) * fraction
        base = min(len(files), int(ideal))
        quota_by_dir[dir_name] = base
        base_total += base
        fractional_parts.append((ideal - base, dir_name))

    remaining = family_keep - base_total
    if remaining <= 0:
        return quota_by_dir

    ranked_dirs = sorted(
        fractional_parts,
        key=lambda item: (-item[0], item[1]),
    )
    for _, dir_name in ranked_dirs:
        if remaining <= 0:
            break
        if quota_by_dir[dir_name] >= len(subdir_file_map[dir_name]):
            continue
        quota_by_dir[dir_name] += 1
        remaining -= 1

    if remaining != 0:
        raise ValueError("Family quota allocation failed to distribute the full requested count.")
    return quota_by_dir


def main() -> None:
    args = build_parser().parse_args()
    project_root = project_root_from_here()
    dataset_root = resolve_project_path(project_root, args.dataset_root).resolve()

    validation_fraction = ensure_fraction("validation_fraction", args.validation_fraction)
    attack_fraction = ensure_fraction("attack_fraction", args.attack_fraction)

    validation_dir = dataset_root / "Validation_Data_Master"
    attack_root = dataset_root / "Attack_Data_Master"

    if not validation_dir.is_dir():
        raise FileNotFoundError(f"Validation_Data_Master not found: {validation_dir}")
    if not attack_root.is_dir():
        raise FileNotFoundError(f"Attack_Data_Master not found: {attack_root}")

    rng = random.Random(int(args.seed))

    validation_files = sorted(validation_dir.glob("*.txt"))
    validation_keep = sample_count_floor(len(validation_files), validation_fraction)
    selected_validation = sample_files(validation_files, validation_keep, rng)

    attack_dirs = sorted(p for p in attack_root.iterdir() if p.is_dir())
    family_to_subdirs: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    family_available_counter = Counter()

    for attack_dir in attack_dirs:
        attack_files = sorted(attack_dir.glob("*.txt"))
        family = attack_family_from_dir(attack_dir.name)
        family_to_subdirs[family][attack_dir.name] = attack_files
        family_available_counter[family] += len(attack_files)

    selected_attack_entries: list[tuple[str, str, list[Path]]] = []
    per_attack_dir_summary: list[dict[str, Any]] = []
    per_attack_family_summary: list[dict[str, Any]] = []
    family_selected_counter = Counter()

    for family in sorted(family_to_subdirs):
        subdir_map = family_to_subdirs[family]
        family_total = int(sum(len(files) for files in subdir_map.values()))
        family_keep = sample_count_floor(family_total, attack_fraction)
        quota_by_dir = allocate_family_keep_counts(
            subdir_file_map=subdir_map,
            family_keep=family_keep,
            fraction=attack_fraction,
        )

        family_subdir_summary: list[dict[str, Any]] = []
        family_selected_total = 0
        for dir_name in sorted(subdir_map):
            files = subdir_map[dir_name]
            quota = int(quota_by_dir[dir_name])
            selected_files = sample_files(files, quota, rng)
            selected_attack_entries.append((dir_name, family, selected_files))
            family_selected_total += len(selected_files)

            ideal = len(files) * attack_fraction
            dir_summary = {
 "attack_dir": dir_name,
 "attack_family": family,
 "available_files": len(files),
 "selected_files": len(selected_files),
 "allocated_quota": quota,
 "ideal_fractional_target": round(float(ideal), 6),
            }
            per_attack_dir_summary.append(dir_summary)
            family_subdir_summary.append(dir_summary)

        family_selected_counter[family] = family_selected_total
        per_attack_family_summary.append(
            {
 "attack_family": family,
 "available_files": family_total,
 "selected_files": family_selected_total,
 "requested_fraction": attack_fraction,
 "rounding_mode": "floor",
 "allocation_policy": "largest_remainder_across_subdirs",
 "subdir_count": len(subdir_map),
 "subdir_sampling": family_subdir_summary,
            }
        )

    entries: list[dict[str, Any]] = []
    input_files: list[str] = []

    for file_path in selected_validation:
        output_path = path_for_output(project_root, file_path)
        entries.append(
            build_entry(
                project_root=project_root,
                file_path=file_path,
                intended_label="benign_control",
                target_preset=None,
                source_group="Validation_Data_Master",
            )
        )
        input_files.append(output_path)

    for attack_dir_name, family, selected_files in sorted(selected_attack_entries, key=lambda item: (item[1], item[0])):
        for file_path in selected_files:
            output_path = path_for_output(project_root, file_path)
            entries.append(
                build_entry(
                    project_root=project_root,
                    file_path=file_path,
                    intended_label="malware_targeted",
                    target_preset=family,
                    source_group=attack_dir_name,
                )
            )
            input_files.append(output_path)

    manifest_payload = {
 "schema_version": 2,
 "manifest_type": "adfa_ld_realtime_subset",
 "dataset_root": path_for_output(project_root, dataset_root),
 "sampling": {
 "seed": int(args.seed),
 "validation_fraction": validation_fraction,
 "attack_fraction": attack_fraction,
 "rounding_mode": "floor",
 "validation_sampling_unit": "pool_overall",
 "attack_sampling_unit": "family",
 "family_allocation_policy": "largest_remainder_across_subdirs",
 "expected_path_field": "expected_detect_file",
        },
 "summary": {
 "entries_total": len(entries),
 "benign_control_count": len(selected_validation),
 "malware_targeted_count": int(sum(family_selected_counter.values())),
 "validation_available_total": len(validation_files),
 "validation_selected_total": len(selected_validation),
 "attack_available_total": int(sum(family_available_counter.values())),
 "attack_selected_total": int(sum(family_selected_counter.values())),
 "malware_target_preset_counts": dict(sorted(family_selected_counter.items())),
 "malware_target_preset_available_counts": dict(sorted(family_available_counter.items())),
        },
 "attack_family_sampling": per_attack_family_summary,
 "attack_dir_sampling": per_attack_dir_summary,
 "entries": entries,
    }

    manifest_out = resolve_project_path(project_root, args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    input_list_out = resolve_project_path(project_root, args.input_list_out)
    input_list_out.parent.mkdir(parents=True, exist_ok=True)
    input_list_out.write_text("".join(f"{line}\n" for line in input_files), encoding="utf-8")

    print("=== ADFA_LD_REALTIME_SUBSET ===")
    print(json.dumps(manifest_payload["summary"], ensure_ascii=False))
    print(f"manifest_out={path_for_output(project_root, manifest_out.resolve())}")
    print(f"input_list_out={path_for_output(project_root, input_list_out.resolve())}")


if __name__ == "__main__":
    main()

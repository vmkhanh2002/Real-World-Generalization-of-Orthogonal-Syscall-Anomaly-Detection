"""Export per-window Path A/B/C scores and Phase 5 probabilities for streaming evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import io_utils
from utils import path_for_output, resolve_project_path  # type: ignore


def project_root_from_here() -> Path:
    # .../scripts/realtime_eval/export_window_scores.py > repo root
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import defaults as rt_defaults  # pylint: disable=import-error

    parser = argparse.ArgumentParser(
        description=(
 "Run the hybrid stack and export per-window scores for real-time/streaming evaluation."
        )
    )
    parser.add_argument("--dataset", default="32bit", help="Dataset key under models/<dataset>/.")
    parser.add_argument("--syscall-abi", default=None, help="Override syscall ABI (e.g. i386, x64).")
    parser.add_argument(
 "--official-syscalls",
        default=None,
        help="Optional path to an official syscall list file used for mapping.",
    )
    parser.add_argument("--manifest", default=None, help="Path to phase5 manifest JSON.")
    parser.add_argument("--meta-best", default=None, help="Path to phase5 model pickle.")
    parser.add_argument("--path-a-model-file", default=None, help="Optional override for Path A model.")
    parser.add_argument("--path-b-ckpt-file", default=None, help="Optional override for Path B checkpoint.")
    parser.add_argument("--path-c-ckpt-file", default=None, help="Optional override for Path C checkpoint.")
    parser.add_argument("--input-list-file", default=None, help="Text file listing input trace files.")
    parser.add_argument("--input-dir", default=None, help="Directory containing trace files.")
    parser.add_argument("--input-glob", default="*.txt", help="Glob for --input-dir (default: *.txt).")
    parser.add_argument(
 "--window-size",
        type=int,
        default=20,
        help="Sliding window size for sequence tokenization.",
    )
    parser.add_argument(
 "--stride",
        type=int,
        default=2,
        help="Sliding window stride.",
    )
    parser.add_argument(
 "--max-files",
        type=int,
        default=0,
        help="Optional cap for quick runs (0 means no cap).",
    )
    parser.add_argument(
 "--output-jsonl",
        default=str(rt_defaults.artifact_path("window_scores.jsonl")),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def read_input_list_file(path: Path) -> List[str]:
    lines: List[str] = []
    for raw_line in io_utils.read_text_allow_bom(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def resolve_project_path(project_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (project_root / path)


def path_for_output(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def dedup_keep_order(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        canonical = str(path.resolve()).lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(path)
    return out


def resolve_inputs(project_root: Path, args: argparse.Namespace) -> List[Path]:
    raw_paths: List[Path] = []
    if args.input_list_file:
        list_path = resolve_project_path(project_root, args.input_list_file)
        if not list_path.exists():
            raise FileNotFoundError(f"Input list file not found: {list_path}")
        if not list_path.is_file():
            raise FileNotFoundError(f"Input list path is not a file: {list_path}")
        listed_paths = [resolve_project_path(project_root, line) for line in read_input_list_file(list_path)]
        missing = [str(path) for path in listed_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing input files from --input-list-file ({len(missing)}): {missing}")
        non_files = [str(path) for path in listed_paths if path.exists() and not path.is_file()]
        if non_files:
            raise FileNotFoundError(f"Non-file paths found in --input-list-file ({len(non_files)}): {non_files}")
        raw_paths.extend(listed_paths)

    if args.input_dir:
        input_dir = resolve_project_path(project_root, args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        raw_paths.extend(sorted(input_dir.rglob(args.input_glob)))

    resolved = [p.resolve() for p in raw_paths if p.exists()]
    deduped = dedup_keep_order(resolved)
    if args.max_files and args.max_files > 0:
        deduped = deduped[: args.max_files]
    return deduped


def main() -> None:
    args = parse_args()
    project_root = project_root_from_here()
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from realtime_eval import inference_backend as backend  # pylint: disable=import-error

    manifest_rel = Path(args.manifest) if args.manifest else backend.default_manifest_rel(args.dataset)
    meta_best_rel = Path(args.meta_best) if args.meta_best else backend.default_meta_best_rel(args.dataset)
    official_syscalls_rel = Path(args.official_syscalls) if args.official_syscalls else None

    manifest_path = resolve_project_path(project_root, manifest_rel)
    meta_best_path = resolve_project_path(project_root, meta_best_rel)
    inputs = resolve_inputs(project_root, args)
    if not inputs:
        raise ValueError("No input files resolved. Provide --input-list-file or --input-dir.")

    loaded = backend.load_selected_models(
        project_root=project_root,
        dataset=args.dataset,
        manifest_path=manifest_path,
        meta_best_path=meta_best_path,
        need_path_a=True,
        need_path_b=True,
        need_path_c=True,
        need_phase5=True,
        path_a_model_override=args.path_a_model_file,
        path_b_ckpt_override=args.path_b_ckpt_file,
        path_c_ckpt_override=args.path_c_ckpt_file,
    )
    max_vocab_id = int(loaded["max_vocab_id"])

    name_to_nums, num_to_names, syscall_lookup_meta = backend.resolve_syscall_lookup(
        project_root=project_root,
        dataset=args.dataset,
        official_syscalls_rel=official_syscalls_rel,
        syscall_abi=args.syscall_abi,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded["model_b"] = loaded["model_b"].to(device).eval()
    loaded["model_c"] = loaded["model_c"].to(device).eval()

    output_path = resolve_project_path(project_root, args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for input_path in inputs:
            raw_seq, unknown_names = backend.parse_raw_trace_tokens(input_path, name_to_nums, max_vocab_id)
            mapped_seq, stats = backend.normalize_numeric_sequence(raw_seq, num_to_names, name_to_nums, max_vocab_id)
            windows = backend.make_windows(mapped_seq, args.window_size, args.stride)

            rec = {
 "file": path_for_output(project_root, input_path),
 "dataset": args.dataset,
 "syscall_mapping_source": syscall_lookup_meta["syscall_mapping_source"],
 "syscall_mapping_ref": syscall_lookup_meta["syscall_mapping_ref"],
 "syscall_abi": syscall_lookup_meta["syscall_abi"],
 "window_size": int(args.window_size),
 "stride": int(args.stride),
 "raw_tokens": int(len(raw_seq)),
 "mapped_tokens": int(len(mapped_seq)),
 "compat_num_remap": int(stats["compat_num_remap"]),
 "name_from_num_remap": int(stats["name_from_num_remap"]),
 "oov_to_zero": int(stats["oov_to_zero"]),
 "oov_rate": float(stats.get("oov_rate", 0.0)),
 "unknown_name_count": int(len(unknown_names)),
 "unknown_name_examples": sorted(set(unknown_names))[:8],
 "n_windows": int(len(windows)),
 "phase5_tau": float(loaded["phase5_tau"]),
 "selected_paths": loaded["selected_paths"],
            }

            _oov_rate = float(stats.get("oov_rate", 0.0))
            if _oov_rate > 0.10:
                warnings.warn(
 f"[oov_rate={_oov_rate:.1%}] High OOV rate in "
 f"{path_for_output(project_root, input_path)!r}. "
 "Path B/C anomaly scores may be underestimated for this trace "
 "(exotic syscalls silently mapped to token 0).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if len(windows) == 0:
                rec["status"] = f"too_short_for_window_{args.window_size}"
                rec["path_a_scores"] = []
                rec["path_b_scores"] = []
                rec["path_c_scores"] = []
                rec["phase5_probs"] = []
                rec["phase5_positive"] = []
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            s_a = backend.score_path_a(
                loaded["path_a_selected"],
                loaded["vec"],
                loaded["pca"],
                loaded["model_a"],
                windows,
            )
            s_b = backend.score_path_b(loaded["model_b"], windows, device)
            s_c = backend.score_path_c(loaded["model_c"], windows, device)

            z = np.column_stack([s_a, s_b, s_c])
            p = loaded["phase5_model"].predict_proba(z)[:, 1]
            y = p >= loaded["phase5_tau"]

            rec.update(
                {
 "status": "ok",
 "path_a_scores": [float(x) for x in s_a.tolist()],
 "path_b_scores": [float(x) for x in s_b.tolist()],
 "path_c_scores": [float(x) for x in s_c.tolist()],
 "phase5_probs": [float(x) for x in p.tolist()],
 "phase5_positive": [bool(x) for x in y.tolist()],
                }
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== WINDOW_SCORE_EXPORT ===")
    print(
        json.dumps(
            {
                "inputs_count": len(inputs),
                "output_jsonl": path_for_output(project_root, output_path),
                "window_size": int(args.window_size),
                "stride": int(args.stride),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

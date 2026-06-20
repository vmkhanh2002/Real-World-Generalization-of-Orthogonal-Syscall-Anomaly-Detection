"""Shared inference backend for realtime evaluation."""

from __future__ import annotations

import json
import pickle
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

try:
    from scipy.sparse import coo_matrix as _coo_matrix
    from sklearn.preprocessing import normalize as _sk_normalize
    _SCIPY_AVAILABLE = True
except ImportError:  # scipy not installed - fall back to original string path
    _SCIPY_AVAILABLE = False
    _coo_matrix = None  # type: ignore
    _sk_normalize = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# Add parent scripts/ dir so scripts/models.py (Conv1DAE, GRUPredictor) is importable
_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import io_utils
from utils import (  # type: ignore
    canonical_path,
    max_consecutive_true,
    path_for_output,
    resolve_project_path,
)

from models import Conv1DAE, GRUPredictor  # type: ignore[attr-defined]
from syscall_abi.tables import (
    ABI_DISPLAY_NAMES,
    get_default_abi_for_dataset,
    get_syscall_maps,
    normalize_abi_name,
)


PATH_SELECTION_ALIASES = {
 "isolation_forest": "ifo",
 "one_class_svm": "sgd_ocsvm",
 "ocsvm": "sgd_ocsvm",
 "cnn1d_ae": "cnn",
 "gru_predictor": "gru",
}

# Compiled once at module load; avoids re-compilation on every parse_raw_trace_tokens() call.
_NAME_PAT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class MappedTrace:
    file: str
    raw_tokens: int
    mapped_tokens: int
    compat_num_remap: int
    name_from_num_remap: int
    oov_to_zero: int
    unknown_name_count: int
    unknown_name_examples: List[str]
    seq: List[int]


def load_json(path: Path) -> dict:
    payload = io_utils.load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def torch_load_checkpoint_safe(path: Path):
    """Load a checkpoint with `weights_only=True` when the runtime supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch releases do not support weights_only.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(path, map_location="cpu")


def extract_state_dict_from_checkpoint(payload):
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if payload and all(isinstance(k, str) for k in payload.keys()) and all(
            torch.is_tensor(v) for v in payload.values()
        ):
            return payload
    raise ValueError(
 "Unsupported torch checkpoint format. Expected a state_dict or payload containing state_dict/model_state_dict."
    )


def parse_official_syscall_list(path: Path) -> tuple[dict, dict]:
    name_to_nums = defaultdict(set)
    num_to_names = defaultdict(set)
    pat = re.compile(r"^#define __NR_=([A-Za-z0-9_]+)\s+(\d+)$")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            raw_name = m.group(1)
            num = int(m.group(2))
            base_name = raw_name[5:] if raw_name.startswith("3264_") else raw_name
            name_to_nums[base_name].add(num)
            num_to_names[num].add(base_name)
    return name_to_nums, num_to_names


def resolve_syscall_lookup(
    project_root: Path,
    dataset: str,
    official_syscalls_rel: Path | None,
    syscall_abi: str | None,
) -> tuple[dict, dict, dict]:
    if official_syscalls_rel is not None:
        official_syscalls_path = resolve_project_path(project_root, official_syscalls_rel)
        name_to_nums, num_to_names = parse_official_syscall_list(official_syscalls_path)
        return name_to_nums, num_to_names, {
 "syscall_mapping_source": "explicit_file",
 "syscall_mapping_ref": path_for_output(project_root, official_syscalls_path),
 "syscall_abi": "custom_file",
 "syscall_abi_display": "Custom syscall list file",
        }

    resolved_abi = normalize_abi_name(syscall_abi or get_default_abi_for_dataset(dataset))
    name_to_nums, num_to_names = get_syscall_maps(resolved_abi, project_root=project_root)
    return name_to_nums, num_to_names, {
 "syscall_mapping_source": "builtin_abi",
 "syscall_mapping_ref": resolved_abi,
 "syscall_abi": resolved_abi,
 "syscall_abi_display": ABI_DISPLAY_NAMES[resolved_abi],
    }


def map_name_to_id(
    name: str,
    name_to_nums: dict,
    max_vocab_id: int,
) -> int | None:
    name = name.strip()
    if not name:
        return None
    nums = sorted(name_to_nums.get(name, []))
    if not nums:
        return None
    nums_in_vocab = [n for n in nums if n <= max_vocab_id]
    if nums_in_vocab:
        return nums_in_vocab[0]
    return nums[0]


def parse_raw_trace_tokens(path: Path, name_to_nums: dict, max_vocab_id: int) -> tuple[List[int], List[str]]:
    values: List[int] = []
    unknown_names: List[str] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Fast path for numeric only traces.
    if re.fullmatch(r"[\s0-9]+", text):
        for tok in text.split():
            if tok.isdigit():
                values.append(int(tok))
        return values, unknown_names

    for line in text.splitlines():
        tok = line.strip()
        if not tok:
            continue
        if tok.isdigit():
            values.append(int(tok))
            continue
        # Accept bare syscall names and lines like "openat(...)".
        m = _NAME_PAT.match(tok)
        if not m:
            continue
        name = m.group(1)
        mapped = map_name_to_id(name, name_to_nums, max_vocab_id)
        if mapped is None:
            unknown_names.append(name)
        else:
            values.append(mapped)

    return values, unknown_names


def normalize_numeric_sequence(
    seq: Sequence[int],
    num_to_names: dict,
    name_to_nums: dict,
    max_vocab_id: int,
) -> tuple[List[int], Counter]:
    out: List[int] = []
    stats = Counter()
    for n in seq:
        if 0 <= n <= max_vocab_id:
            out.append(n)
            continue

        mapped = None
        for nm in sorted(num_to_names.get(n, [])):
            cand = map_name_to_id(nm, name_to_nums, max_vocab_id)
            if cand is not None and 0 <= cand <= max_vocab_id:
                mapped = cand
                break
        if mapped is not None:
            out.append(mapped)
            stats["name_from_num_remap"] += 1
        else:
            out.append(0)
            stats["oov_to_zero"] += 1
    # Expose OOV rate so callers can warn when exotic syscalls are silently zeroed.
    stats["oov_rate"] = stats["oov_to_zero"] / max(1, len(seq))
    return out, stats


def make_windows(seq: Sequence[int], w: int, s: int) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.int64)
    if len(arr) < w:
        return np.empty((0, w), dtype=np.int64)
    n_windows = (len(arr) - w) // s + 1
    shape   = (n_windows, w)
    strides = (arr.strides[0] * s, arr.strides[0])
    # as_strided creates a zero-copy view; np.array() copies it so the result
    # is safe even after the source array is garbage-collected.
    return np.array(np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides))


def max_consecutive_true(mask: Sequence[bool]) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def get_decision_bucket(n_windows: int, min_windows: int, short_max: int, medium_max: int) -> str:
    if n_windows < min_windows:
        return "too_short"
    if n_windows <= short_max:
        return "short"
    if n_windows <= medium_max:
        return "medium"
    return "long"


def parse_binary_label(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "malware", "malicious", "attack", "positive"}:
        return True
    if text in {"0", "false", "benign", "normal", "negative"}:
        return False
    return None


def load_calibration_records(
    calibration_json_path: Path,
    min_windows: int,
    short_max: int,
    medium_max: int,
    label_key: str,
) -> list[dict]:
    payload = load_json(calibration_json_path)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = payload["results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(
 "Unsupported calibration JSON format. Expect either a list of records "
 "or an object with key 'results'."
        )

    records: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        label_raw = row.get(label_key)
        if label_raw is None:
            for alt in ("label", "y_true", "ground_truth", "is_malware", "target"):
                if alt in row:
                    label_raw = row.get(alt)
                    break
        y_true = parse_binary_label(label_raw)
        if y_true is None:
            continue

        n_windows = int(row.get("n_windows", 0))
        if n_windows < min_windows:
            continue

        prob_mean = row.get("phase5_prob_mean")
        pos_rate = row.get("phase5_positive_rate")
        if prob_mean is None or pos_rate is None:
            continue
        prob_mean = float(prob_mean)
        pos_rate = float(pos_rate)

        max_consec = row.get("phase5_max_consecutive_positive_windows")
        if max_consec is None:
            max_consec = row.get("phase5_positive_windows", 0)
        max_consec = int(max(0, max_consec))

        bucket = get_decision_bucket(
            n_windows=n_windows,
            min_windows=min_windows,
            short_max=short_max,
            medium_max=medium_max,
        )
        if bucket == "too_short":
            continue

        records.append(
            {
 "bucket": bucket,
 "is_malware": y_true,
 "prob_mean": prob_mean,
 "pos_rate": pos_rate,
 "max_consecutive": max_consec,
            }
        )
    return records


def calibrate_bucket_rules_for_target_fpr(
    records: list[dict],
    base_rules: dict,
    target_fpr: float,
    min_samples_per_bucket: int,
) -> tuple[dict, dict]:
    calibrated = {k: dict(v) for k, v in base_rules.items()}
    report: dict[str, dict] = {}
    quantile_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    for bucket in ("short", "medium", "long"):
        rows = [r for r in records if r["bucket"] == bucket]
        benign = [r for r in rows if not r["is_malware"]]
        malware = [r for r in rows if r["is_malware"]]
        report[bucket] = {
 "rows": len(rows),
 "benign_rows": len(benign),
 "malware_rows": len(malware),
 "status": "ok",
        }

        if len(benign) < min_samples_per_bucket or len(malware) == 0:
            report[bucket]["status"] = "insufficient_samples_keep_default"
            report[bucket]["selected_rule"] = dict(calibrated[bucket])
            continue

        prob_all = np.asarray([r["prob_mean"] for r in rows], dtype=np.float64)
        rate_all = np.asarray([r["pos_rate"] for r in rows], dtype=np.float64)
        max_consec_all = [int(r["max_consecutive"]) for r in rows]
        max_consec_cap = max(max_consec_all) if max_consec_all else 1

        prob_candidates = sorted(
            set(
                [float(np.quantile(prob_all, q)) for q in quantile_grid]
                + [float(calibrated[bucket]["min_mean_prob"])]
            )
        )
        rate_candidates = sorted(
            set(
                [float(np.quantile(rate_all, q)) for q in quantile_grid]
                + [float(calibrated[bucket]["min_positive_rate"])]
            )
        )
        consec_seed = [1, 2, 3, 4, 5, 6, 8, 10, 12, int(calibrated[bucket]["min_consecutive_positive"])]
        consec_candidates = sorted(set(k for k in consec_seed if 1 <= k <= max_consec_cap))
        if not consec_candidates:
            consec_candidates = [1]

        best_ok = None
        best_fallback = None
        # Pre-sort benign/malware arrays once; searchsorted replaces O(N) inner loops
        # with O(log N), reducing calibration from O(P*R*K*N) to O(P*R*K*log N).
        benign_probs_s   = np.sort(np.array([r["prob_mean"] for r in benign]))
        benign_rates_s   = np.sort(np.array([r["pos_rate"]  for r in benign]))
        benign_consec_s  = np.sort(np.array([int(r["max_consecutive"]) for r in benign]))
        malware_probs_s  = np.sort(np.array([r["prob_mean"] for r in malware]))
        malware_rates_s  = np.sort(np.array([r["pos_rate"]  for r in malware]))
        malware_consec_s = np.sort(np.array([int(r["max_consecutive"]) for r in malware]))
        n_benign  = len(benign)
        n_malware = len(malware)

        for tp in prob_candidates:
            for tr in rate_candidates:
                for tk in consec_candidates:
                    # O(log N) per threshold FP satisfies ALL three conditions.
                    fp = min(
                        n_benign - int(np.searchsorted(benign_probs_s,   tp,     side="left")),
                        n_benign - int(np.searchsorted(benign_rates_s,   tr,     side="left")),
                        n_benign - int(np.searchsorted(benign_consec_s,  tk - 1, side="right")),
                    )
                    fp = max(0, fp)
                    tp_hit = min(
                        n_malware - int(np.searchsorted(malware_probs_s,   tp,     side="left")),
                        n_malware - int(np.searchsorted(malware_rates_s,   tr,     side="left")),
                        n_malware - int(np.searchsorted(malware_consec_s,  tk - 1, side="right")),
                    )
                    tp_hit = max(0, tp_hit)
                    fpr = fp / max(1, n_benign)
                    tpr = tp_hit / max(1, n_malware)
                    cand = {
 "min_mean_prob": float(tp),
 "min_positive_rate": float(tr),
 "min_consecutive_positive": int(tk),
 "fpr": float(fpr),
 "tpr": float(tpr),
                    }
                    if best_fallback is None:
                        best_fallback = cand
                    else:
                        if (
                            cand["fpr"] < best_fallback["fpr"]
                            or (
                                np.isclose(cand["fpr"], best_fallback["fpr"])
                                and cand["tpr"] > best_fallback["tpr"]
                            )
                        ):
                            best_fallback = cand

                    if fpr <= target_fpr:
                        if best_ok is None:
                            best_ok = cand
                        else:
                            if (
                                cand["tpr"] > best_ok["tpr"]
                                or (
                                    np.isclose(cand["tpr"], best_ok["tpr"])
                                    and cand["fpr"] < best_ok["fpr"]
                                )
                            ):
                                best_ok = cand

        selected = best_ok if best_ok is not None else best_fallback
        if selected is None:
            report[bucket]["status"] = "search_failed_keep_default"
            report[bucket]["selected_rule"] = dict(calibrated[bucket])
            continue

        calibrated[bucket]["min_mean_prob"] = float(selected["min_mean_prob"])
        calibrated[bucket]["min_positive_rate"] = float(selected["min_positive_rate"])
        calibrated[bucket]["min_consecutive_positive"] = int(selected["min_consecutive_positive"])

        report[bucket]["status"] = (
 "target_fpr_met" if best_ok is not None else "target_fpr_not_met_best_fallback"
        )
        report[bucket]["selected_rule"] = dict(calibrated[bucket])
        report[bucket]["selected_metrics"] = {
 "fpr": float(selected["fpr"]),
 "tpr": float(selected["tpr"]),
        }

    return calibrated, report


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (project_root / p)


def path_for_output(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def default_manifest_rel(dataset: str) -> Path:
    return Path("models") / dataset / "phase5_fusion" / "meta_best_model_manifest.json"


def default_meta_best_rel(dataset: str) -> Path:
    return Path("models") / dataset / "phase5_fusion" / "meta_best_model.pkl"

def natural_sort_key(path_like: str):
    parts = re.split(r"(\d+)", path_like.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def read_input_list_file(path: Path) -> List[str]:
    rows: List[str] = []
    for line in io_utils.read_text_allow_bom(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def dedup_paths_keep_order(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        k = str(p.resolve()).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def resolve_input_paths(project_root: Path, args) -> List[Path]:
    explicit_inputs: List[str] = []
    if args.input_list_file:
        list_file_path = resolve_project_path(project_root, args.input_list_file)
        if not list_file_path.exists():
            raise FileNotFoundError(f"--input-list-file not found: {list_file_path}")
        explicit_inputs.extend(read_input_list_file(list_file_path))
    if args.input_files:
        explicit_inputs.extend(args.input_files)

    if explicit_inputs:
        inputs = [resolve_project_path(project_root, p) for p in explicit_inputs]
        missing = [str(p) for p in inputs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing input files ({len(missing)}): {missing}")
        return dedup_paths_keep_order([p for p in inputs if p.is_file()])

    input_dir = resolve_project_path(project_root, args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    pattern = f"**/{args.input_glob}" if args.input_recursive else args.input_glob
    discovered = [p for p in input_dir.glob(pattern) if p.is_file()]
    discovered = sorted(
        discovered,
        key=lambda p: natural_sort_key(path_for_output(project_root, p).replace("\\", "/")),
    )
    if not discovered:
        raise FileNotFoundError(
 f"No input files found in {input_dir} with pattern '{pattern}'. "
 "Use --input-files or --input-list-file to pass files explicitly."
        )
    return dedup_paths_keep_order(discovered)


def resolve_override_file(
    project_root: Path,
    default_dir: Path,
    override: str | None,
    expected_suffix: str,
    label: str,
) -> Path | None:
    if not override:
        return None
    raw = Path(override)
    candidates = [raw] if raw.is_absolute() else [project_root / raw, default_dir / raw]
    for cand in candidates:
        if cand.exists():
            if expected_suffix and cand.suffix.lower() != expected_suffix.lower():
                raise ValueError(
 f"{label} override must end with '{expected_suffix}', got: {cand.name}"
                )
            return cand
    raise FileNotFoundError(f"{label} override file not found from: {candidates}")


def choose_artifact_by_selected_name(candidates: Sequence[Path], selected_name: str, label: str) -> Path:
    if not candidates:
        raise FileNotFoundError(f"No candidate artifacts found for {label}")
    sel_raw = selected_name.lower().strip()
    sel_norm = normalize_key(sel_raw)
    alias_norm = normalize_key(PATH_SELECTION_ALIASES.get(sel_raw, ""))

    scored = []
    for p in candidates:
        stem_norm = normalize_key(p.stem)
        score = 0
        if stem_norm == sel_norm:
            score = 100
        elif alias_norm and stem_norm == alias_norm:
            score = 95
        elif stem_norm and sel_norm.startswith(stem_norm):
            score = 80
        elif stem_norm and stem_norm in sel_norm:
            score = 70
        elif sel_norm and sel_norm in stem_norm:
            score = 60
        if score > 0:
            scored.append((score, len(stem_norm), p.name.lower(), p))

    if not scored:
        if len(candidates) == 1:
            return candidates[0]
        cand_names = ", ".join(sorted(p.name for p in candidates))
        raise ValueError(
 f"Could not auto-resolve {label} artifact for selected='{selected_name}'. "
 f"Candidates: {cand_names}. Use explicit override CLI option."
        )

    scored.sort(reverse=True)
    best_score = scored[0][0]
    best = [it for it in scored if it[0] == best_score]
    if len(best) == 1:
        return best[0][-1]

    cand_names = ", ".join(it[-1].name for it in best)
    raise ValueError(
 f"Ambiguous auto-resolve for {label}, selected='{selected_name}', "
 f"best-score candidates: {cand_names}. Use explicit override CLI option."
    )


def resolve_path_a_model_file(
    project_root: Path,
    path_a_dir: Path,
    selected_name: str,
    override: str | None,
) -> Path:
    override_path = resolve_override_file(
        project_root=project_root,
        default_dir=path_a_dir,
        override=override,
        expected_suffix=".pkl",
        label="Path A model",
    )
    if override_path is not None:
        return override_path

    ignore_names = {"vec.pkl", "pca.pkl", "pca100.pkl"}
    candidates = sorted(
        [
            p
            for p in path_a_dir.glob("*.pkl")
            if p.name not in ignore_names and "best_config" not in p.name.lower()
        ]
    )
    return choose_artifact_by_selected_name(candidates, selected_name, "Path A")


def resolve_path_torch_ckpt_file(
    project_root: Path,
    path_dir: Path,
    selected_name: str,
    override: str | None,
    label: str,
) -> Path:
    override_path = resolve_override_file(
        project_root=project_root,
        default_dir=path_dir,
        override=override,
        expected_suffix=".pth",
        label=f"{label} checkpoint",
    )
    if override_path is not None:
        return override_path
    candidates = sorted(path_dir.glob("*.pth"))
    return choose_artifact_by_selected_name(candidates, selected_name, label)


def load_selected_models(
    project_root: Path,
    dataset: str,
    manifest_path: Path,
    meta_best_path: Path,
    need_path_a: bool,
    need_path_b: bool,
    need_path_c: bool,
    need_phase5: bool,
    path_a_model_override: str | None,
    path_b_ckpt_override: str | None,
    path_c_ckpt_override: str | None,
):
    manifest = load_json(manifest_path)
    selected = manifest["selected_paths"]
    path_a_sel = selected["path_a"]
    path_b_sel = selected["path_b"]
    path_c_sel = selected["path_c"]

    models_root = project_root / "models" / dataset

    vec = None
    pca = None
    model_a = None
    path_a_file = None
    if need_path_a or need_phase5:
        path_a_dir = models_root / "path_a"
        vec = load_pickle(path_a_dir / "vec.pkl")
        pca = load_pickle(path_a_dir / "pca.pkl")
        path_a_file = resolve_path_a_model_file(
            project_root=project_root,
            path_a_dir=path_a_dir,
            selected_name=path_a_sel,
            override=path_a_model_override,
        )
        model_a = load_pickle(path_a_file)

    # Path B metadata defines vocab_size even when the model is skipped.
    path_b_dir = models_root / "path_b"
    path_b_ckpt = resolve_path_torch_ckpt_file(
        project_root=project_root,
        path_dir=path_b_dir,
        selected_name=path_b_sel,
        override=path_b_ckpt_override,
        label="Path B",
    )
    path_b_meta = load_json(Path(str(path_b_ckpt) + ".meta.json"))
    max_vocab_id = int(path_b_meta["init_kwargs"]["vocab_size"]) - 1
    model_b = None
    if need_path_b or need_phase5:
        model_b = Conv1DAE(**path_b_meta["init_kwargs"])
        model_b_payload = torch_load_checkpoint_safe(path_b_ckpt)
        model_b.load_state_dict(extract_state_dict_from_checkpoint(model_b_payload))
        model_b.eval()

    model_c = None
    path_c_ckpt = None
    if need_path_c or need_phase5:
        path_c_dir = models_root / "path_c"
        path_c_ckpt = resolve_path_torch_ckpt_file(
            project_root=project_root,
            path_dir=path_c_dir,
            selected_name=path_c_sel,
            override=path_c_ckpt_override,
            label="Path C",
        )
        path_c_meta = load_json(Path(str(path_c_ckpt) + ".meta.json"))
        model_c = GRUPredictor(**path_c_meta["init_kwargs"])
        model_c_payload = torch_load_checkpoint_safe(path_c_ckpt)
        model_c.load_state_dict(extract_state_dict_from_checkpoint(model_c_payload))
        model_c.eval()

    payload = load_pickle(meta_best_path) if need_phase5 else None
    phase5_model = (
        payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    ) if need_phase5 else None
    phase5_tau = float(
        manifest.get(
 "threshold_tau",
            payload.get("threshold_tau", 0.5) if isinstance(payload, dict) else 0.5,
        )
    )
    if hasattr(phase5_model, "n_jobs"):
        phase5_model.n_jobs = 1

    return {
 "manifest": manifest,
 "selected_paths": selected,
 "path_a_selected": path_a_sel,
 "path_b_selected": path_b_sel,
 "path_c_selected": path_c_sel,
 "path_a_file": path_for_output(project_root, path_a_file) if path_a_file else None,
 "path_b_file": path_for_output(project_root, path_b_ckpt),
 "path_c_file": path_for_output(project_root, path_c_ckpt) if path_c_ckpt else None,
 "vec": vec,
 "pca": pca,
 "model_a": model_a,
 "model_b": model_b,
 "model_c": model_c,
 "phase5_model": phase5_model,
 "phase5_tau": phase5_tau,
 "max_vocab_id": max_vocab_id,
    }


def _vec_transform_fast(vec, windows: np.ndarray):
    """Equivalent to ``vec.transform([" ".join(map(str, row)) for row in windows])``.

    Uses ``scipy.sparse.coo_matrix`` + manual IDF/norm weighting instead of
    O(NAW) string allocations + TfidfVectorizer tokenisation. Falls back to
    the original string path when scipy is unavailable.
    """
    if not _SCIPY_AVAILABLE or windows.size == 0:
        texts = [" ".join(map(str, row.tolist())) for row in windows]
        return vec.transform(texts)

    vocab: dict = vec.vocabulary_   # {str_token: tfidf_col_idx}
    n_vocab = len(vocab)
    n_win, w = windows.shape

    # Build integer syscall-ID to TF-IDF column index lookup (one-time per call).
    max_id = int(windows.max())
    id_to_col = np.full(max(max_id + 1, 1), -1, dtype=np.int32)
    for str_tok, col_idx in vocab.items():
        try:
            tok_int = int(str_tok)
            if 0 <= tok_int <= max_id:
                id_to_col[tok_int] = col_idx
        except (ValueError, IndexError):
            pass

    flat    = windows.flatten()                           # (n_win * w,)
    row_idx = np.repeat(np.arange(n_win, dtype=np.int32), w)
    clipped = np.clip(flat, 0, len(id_to_col) - 1).astype(np.int32)
    col_arr = id_to_col[clipped]
    mask    = col_arr >= 0

    x_counts = _coo_matrix(
        (np.ones(int(mask.sum()), dtype=np.float64),
         (row_idx[mask], col_arr[mask])),
        shape=(n_win, n_vocab),
    ).tocsr()

    # Honour sublinear TF if the fitted vectoriser uses it.
    if getattr(vec, "sublinear_tf", False):
        x_counts = x_counts.copy()
        x_counts.data = 1.0 + np.log(np.maximum(x_counts.data, 1e-10))

    # Apply IDF weights (vec.idf_ is always present when use_idf=True, the default).
    if hasattr(vec, "idf_"):
        x_tfidf = x_counts.multiply(vec.idf_)
    else:
        x_tfidf = x_counts

    # L2 normalise (TfidfVectorizer default norm='l2').
    norm = getattr(vec, "norm", "l2") or "l2"
    if norm:
        x_tfidf = _sk_normalize(x_tfidf, norm=norm, copy=False)

    return x_tfidf


def score_path_a(path_a_selected: str, vec, pca, model_a, windows: np.ndarray) -> np.ndarray:
    x_tfidf = _vec_transform_fast(vec, windows)

    if path_a_selected == "pca_err":
        # todense() / asarray(): guard against np.matrix returned by sparse ops or sklearn,
        # which breaks np.column_stack() and np.mean() return type contracts.
        x_dense = np.asarray(x_tfidf.todense() if hasattr(x_tfidf, "todense") else x_tfidf)
        x_recon = model_a.inverse_transform(model_a.transform(x_dense))
        return np.asarray(np.mean((x_dense - x_recon) ** 2, axis=1)).ravel()

    x_red = pca.transform(x_tfidf)
    if path_a_selected in {"sgd_ocsvm", "lof"}:
        return -model_a.decision_function(x_red)
    if path_a_selected in {"ifo", "isolation_forest"}:
        return -model_a.score_samples(x_red)
    if path_a_selected == "hbos":
        return model_a.decision_function(x_red)
    raise ValueError(f"Unsupported path_a selected model: {path_a_selected}")


def score_path_b(model_b: Conv1DAE, windows: np.ndarray, device: torch.device) -> np.ndarray:
    t = torch.tensor(windows, dtype=torch.long, device=device)
    with torch.no_grad():
        pred, emb = model_b(t)
        mse = torch.mean((emb - pred) ** 2, dim=(1, 2))
    return mse.cpu().numpy()


def score_path_c(model_c: GRUPredictor, windows: np.ndarray, device: torch.device) -> np.ndarray:
    t = torch.tensor(windows, dtype=torch.long, device=device)
    if t.size(1) < 2:
        return np.zeros((t.size(0),), dtype=np.float64)
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        out = model_c(t[:,:-1])
        loss = crit(out.transpose(1, 2), t[:, 1:]).mean(dim=1)
    return loss.cpu().numpy()



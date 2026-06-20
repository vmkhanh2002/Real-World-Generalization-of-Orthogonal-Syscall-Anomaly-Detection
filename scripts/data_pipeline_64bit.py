"""
Phase 1: 64-bit (DongTing) Data Pipeline

Extracts arrays tailored for scalable 64-bit Kernel Exploit workloads mapping to
10 vulnerability families. Enforces rigorous Domain-Shift training mechanics.
"""

# Dataset defaults.
from pipeline_config import get_data_pipeline_defaults, normalize_dataset_name

DATASET_CFG = get_data_pipeline_defaults("64bit")
DATASET_NAME = DATASET_CFG["dataset_name"]
WINDOW_SIZE = DATASET_CFG["window_size"]
STRIDE = DATASET_CFG["stride"]
ARRAY_DTYPE = DATASET_CFG["array_dtype"]  # Safe for syscall IDs and trace counts in this dataset.
MAX_WINDOWS_PER_TRACE = DATASET_CFG["max_windows_per_trace"]  # Cap windows extracted from one trace.


def configure_dataset(dataset_name):
    global DATASET_CFG, DATASET_NAME, WINDOW_SIZE, STRIDE, ARRAY_DTYPE, MAX_WINDOWS_PER_TRACE
    DATASET_CFG = get_data_pipeline_defaults(normalize_dataset_name(dataset_name))
    DATASET_NAME = DATASET_CFG["dataset_name"]
    WINDOW_SIZE = DATASET_CFG["window_size"]
    STRIDE = DATASET_CFG["stride"]
    ARRAY_DTYPE = DATASET_CFG["array_dtype"]
    MAX_WINDOWS_PER_TRACE = DATASET_CFG["max_windows_per_trace"]

# Filename rules for attack families. First match wins.
ATTACK_FAMILY_RULES = [
    ("use_after_free",   "Use-After-Free"),
    ("out_of_bounds",    "Out-of-Bounds"),
    ("memory_leak",      "Memory-Leak"),
    ("uninit_value",     "Uninitialized-Memory"),
    ("WARNING",          "Kernel-Warning"),
    ("BUG",              "Kernel-BUG"),
    ("null_ptr_deref",   "Null-Pointer-Deref"),
    ("deadlock",         "Deadlock"),
    ("double_free",      "Double-Free"),
    ("overflow",         "Overflow"),
]
DEFAULT_FAMILY = "Other-Exploit"

import argparse
import os, pickle, sys, re
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

# Force UTF 8 Output on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')


def classify_attack_family(filename):
    """Classifies an attack log filename into a vulnerability family."""
    for pattern, family in ATTACK_FAMILY_RULES:
        if pattern in filename:
            return family
    return DEFAULT_FAMILY

def iter_traces_from_dir(dirpath):
    """Yield parsed syscall traces from one directory."""
    if not os.path.exists(dirpath):
        print(f"Warning: Directory not found {dirpath}")
        return

    files = sorted([f for f in os.listdir(dirpath) if f.endswith('.txt')])
    print(f" INFO Scanning {len(files)} files in {os.path.basename(dirpath)}...")

    for i, fname in enumerate(files):
        fpath = os.path.join(dirpath, fname)
        try:
            if i > 0 and i % 2000 == 0:
                print(f" INFO Loading file {i}/{len(files)}: {fname}")

            # Fast path for standard numeric trace files.
            trace = np.loadtxt(fpath, dtype=np.int16)
            trace = trace.flatten()  # Handles empty or column shaped arrays.

            if trace.size > 0:
                yield fname, trace

        except ValueError:
            # Fallback for files with unusual whitespace or formatting.
            number_pattern = re.compile(rb'\d+')
            nums = []
            with open(fpath, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    nums.extend(int(m.group()) for m in number_pattern.finditer(chunk))
            if nums:
                yield fname, np.array(nums, dtype=np.int16)
        except Exception as e:
            print(f" WARNING Error parsing {fname}: {e}")


def generate_sliding_windows(traces, w, s, trace_families=None):
    """
 Transforms integer traces into fixed-length matrices.
 Forces INT16 dtype and performs pre-allocation to eliminate OOM faults.

 Args:
 traces: Array of variable-length numpy lists.
 w: Window size constraint.
 s: Stride length.
 trace_families: Array of attack classification tags.

 Returns:
 windows: (N, w) trace extraction blocks.
 trace_ids: (N,) index mapping block to parent trace.
 window_families: (N,) mapped family tags.
 """
    if not traces:
        if trace_families is not None:
            return np.empty((0, w), dtype=ARRAY_DTYPE), np.empty(0, dtype=np.int32), np.empty(0, dtype=str)
        return np.empty((0, w), dtype=ARRAY_DTYPE), np.empty(0, dtype=np.int32)

    total_windows = sum(min((len(t) - w) // s + 1, MAX_WINDOWS_PER_TRACE) for t in traces if len(t) >= w)

    windows = np.empty((total_windows, w), dtype=ARRAY_DTYPE)
    trace_ids = np.empty(total_windows, dtype=np.int32)
    if trace_families is not None:
        window_families = np.empty(total_windows, dtype=object)

    idx = 0
    for trace_idx, trace in enumerate(traces):
        if len(trace) < w:
            continue

        # Use a view instead of copying each window.
        sub_windows = np.lib.stride_tricks.sliding_window_view(trace, window_shape=w)[::s, :]
        n_sub = len(sub_windows)

        if n_sub > MAX_WINDOWS_PER_TRACE:
            # Downsample long traces while preserving window order.
            rng = np.random.RandomState(42 + trace_idx)
            chosen_indices = rng.choice(n_sub, size=MAX_WINDOWS_PER_TRACE, replace=False)
            chosen_indices.sort()
            sub_windows = sub_windows[chosen_indices]
            n_sub = MAX_WINDOWS_PER_TRACE

        windows[idx : idx + n_sub] = sub_windows
        trace_ids[idx : idx + n_sub] = trace_idx
        if trace_families is not None:
            window_families[idx : idx + n_sub] = trace_families[trace_idx]

        idx += n_sub

    if trace_families is not None:
        return windows, trace_ids, window_families
    return windows, trace_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, help="Dataset namespace.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional raw dataset root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional processed output directory.")
    args = parser.parse_args()
    configure_dataset(args.dataset)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build paths based on dataset configuration
    dataset_dir = args.dataset_root or os.path.join(project_root, "datasets", DATASET_NAME)
    output_dir = args.output_dir or os.path.join(project_root, "data", "processed", DATASET_NAME)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 1: DATA PIPELINE INITIALIZATION ({DATASET_NAME.upper()})")
    print("=" * 60)

    # 1. Load Domain Shift Traces
    print(f"\n1. Loading Raw Traces from {dataset_dir}...")
    train_dir = os.path.join(dataset_dir, "Training_Data_Master")
    val_dir   = os.path.join(dataset_dir, "Validation_Data_Master")
    attack_base = os.path.join(dataset_dir, "Attack_Data_Master")

    train_traces = [trace for _, trace in iter_traces_from_dir(train_dir)]
    val_traces   = [trace for _, trace in iter_traces_from_dir(val_dir)]

    # Load attack traces with vulnerability family categorization
    attack_traces = []
    attack_families_per_trace = []
    family_counts = {}

    if os.path.exists(attack_base):
        for fname, trace in iter_traces_from_dir(attack_base):
            attack_traces.append(trace)
            family = classify_attack_family(fname)
            attack_families_per_trace.append(family)
            family_counts[family] = family_counts.get(family, 0) + 1

    print(f" Train traces: {len(train_traces)}")
    print(f" Val traces: {len(val_traces)}")
    print(f" Attack traces: {len(attack_traces)}")

    if family_counts:
        print(f"\n Attack Family Distribution ({len(family_counts)} families):")
        for fam, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True):
            print(f" - {fam}: {count} traces")

    # 2. Generate Sliding Windows (with trace provenance IDs)
    print(f"\n2. Generating Sliding Windows (W={WINDOW_SIZE}, S={STRIDE}, dtype={ARRAY_DTYPE})...")
    X_train_w,  trace_ids_train  = generate_sliding_windows(train_traces,  WINDOW_SIZE, STRIDE)
    X_val_w,    trace_ids_val    = generate_sliding_windows(val_traces,    WINDOW_SIZE, STRIDE)
    X_attack_w, trace_ids_attack, window_families_attack = generate_sliding_windows(
        attack_traces, WINDOW_SIZE, STRIDE, trace_families=attack_families_per_trace
    )

    print(f" Train windows: {X_train_w.shape} | Unique traces: {len(np.unique(trace_ids_train))}")
    print(f" Val windows: {X_val_w.shape} | Unique traces: {len(np.unique(trace_ids_val))}")
    print(f" Attack windows: {X_attack_w.shape} | Unique traces: {len(np.unique(trace_ids_attack))}")

    # Memory usage report
    total_bytes = (X_train_w.nbytes + X_val_w.nbytes + X_attack_w.nbytes +
                   trace_ids_train.nbytes + trace_ids_val.nbytes + trace_ids_attack.nbytes)
    print(f"\n Total array memory: {total_bytes / (1024**3):.2f} GB")

    # 3. Export Baseline Numpy Arrays
    print("\n3. Exporting Payload Arrays...")
    out_file = os.path.join(output_dir, "phase1_base_arrays.pkl")
    payload = {
        "X_train_w":            X_train_w,
        "X_val_w":              X_val_w,
        "X_attack_w":           X_attack_w,
        "trace_ids_train":      trace_ids_train,
        "trace_ids_val":        trace_ids_val,
        "trace_ids_attack":     trace_ids_attack,
        "attack_family_labels": window_families_attack,
        "raw_train_traces":     train_traces,
        "raw_val_traces":       val_traces,
        "raw_attack_traces":    attack_traces
    }
    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    print(f" Successfully saved to {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

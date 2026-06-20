"""
Phase 1: 32-bit (ADFA-LD) Data Pipeline

Generates root numpy arrays and sliding windows for the 32-bit dataset.
Embeds trace provenance IDs per window to enforce strict separation
during Phase 5 Leave-One-Attack-Out (LOAO) validation.
"""

# Configuration
from pipeline_config import DEFAULT_DATASET, get_data_pipeline_defaults, normalize_dataset_name

DATASET_CFG = get_data_pipeline_defaults(DEFAULT_DATASET)
DATASET_NAME = DATASET_CFG["dataset_name"]
WINDOW_SIZE = DATASET_CFG["window_size"]
STRIDE = DATASET_CFG["stride"]
MAX_WINDOWS_PER_TRACE = DATASET_CFG["max_windows_per_trace"]
ARRAY_DTYPE = DATASET_CFG["array_dtype"]


def configure_dataset(dataset_name):
    global DATASET_CFG, DATASET_NAME, WINDOW_SIZE, STRIDE, MAX_WINDOWS_PER_TRACE, ARRAY_DTYPE
    DATASET_CFG = get_data_pipeline_defaults(normalize_dataset_name(dataset_name))
    DATASET_NAME = DATASET_CFG["dataset_name"]
    WINDOW_SIZE = DATASET_CFG["window_size"]
    STRIDE = DATASET_CFG["stride"]
    MAX_WINDOWS_PER_TRACE = DATASET_CFG["max_windows_per_trace"]
    ARRAY_DTYPE = DATASET_CFG["array_dtype"]

import os, pickle, sys, argparse
import random

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

import numpy as np

# Force UTF 8 Output on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_traces_from_dir(dirpath):
    """Loads all contiguous integer traces from a given directory."""
    traces = []
    if not os.path.exists(dirpath):
        print(f"Warning: Directory not found {dirpath}")
        return traces

    for fname in sorted(os.listdir(dirpath)):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath) and fname.endswith('.txt'):
            with open(fpath, 'r') as f:
                content = f.read().strip().split()
                traces.append([int(x) for x in content if x.isdigit()])
    return traces

def generate_sliding_windows(traces, w=20, s=2, trace_families=None):
    """
 Transforms variable-length integer traces into fixed-length sliding windows.

 Args:
 traces: Array of variable-length integer lists (one per discrete process).
 w: Window size constraint.
 s: Stride length.
 trace_families: Array of attack classification tags (LOAO context).

 Returns:
 windows (np.ndarray): (N, w) trace extraction blocks.
 trace_ids (np.ndarray): (N,) index mapping block to parent trace.
 window_families (np.ndarray, optional): (N,) mapped family tags.
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

        trace_arr = np.array(trace, dtype=np.int16)
        sub_windows = np.lib.stride_tricks.sliding_window_view(trace_arr, window_shape=w)[::s, :]
        n_sub = len(sub_windows)

        if n_sub > MAX_WINDOWS_PER_TRACE:
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
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional raw dataset root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional processed output directory.")
    args = parser.parse_args()
    configure_dataset(args.dataset)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build paths based on dataset configuration
    dataset_dir = args.dataset_root or os.path.join(project_root, "datasets", DATASET_NAME, "ADFA-LD", "ADFA-LD")
    output_dir = args.output_dir or os.path.join(project_root, "data", "processed", DATASET_NAME)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 1: DATA PIPELINE INITIALIZATION ({DATASET_NAME.upper()})")
    print("=" * 60)

    # 1. Load Raw Traces
    print(f"1. Loading Raw Traces from {dataset_dir}...")
    train_dir = os.path.join(dataset_dir, "Training_Data_Master")
    val_dir   = os.path.join(dataset_dir, "Validation_Data_Master")
    attack_base = os.path.join(dataset_dir, "Attack_Data_Master")

    train_traces = load_traces_from_dir(train_dir)
    val_traces   = load_traces_from_dir(val_dir)

    attack_traces = []
    attack_families_per_trace = []
    if os.path.exists(attack_base):
        for atype in sorted(os.listdir(attack_base)):
            apath = os.path.join(attack_base, atype)
            if os.path.isdir(apath):
                loaded = load_traces_from_dir(apath)
                attack_traces.extend(loaded)

                parts = atype.split("_")
                if parts[-1].isdigit():
                    base_family = "_".join(parts[:-1])
                else:
                    base_family = atype
                attack_families_per_trace.extend([base_family] * len(loaded))

    print(f" Train traces: {len(train_traces)}")
    print(f" Val traces: {len(val_traces)}")
    print(f" Attack traces: {len(attack_traces)}")

    # Generate sliding windows mapped to trace provenance
    # trace_ids enforce rigid split boundaries in downstream pipeline phases
    print(f"\n2. Generating Sliding Windows (W={WINDOW_SIZE}, S={STRIDE})...")
    X_train_w,  trace_ids_train  = generate_sliding_windows(train_traces,  WINDOW_SIZE, STRIDE)
    X_val_w,    trace_ids_val    = generate_sliding_windows(val_traces,    WINDOW_SIZE, STRIDE)
    X_attack_w, trace_ids_attack, window_families_attack = generate_sliding_windows(attack_traces, WINDOW_SIZE, STRIDE, trace_families=attack_families_per_trace)

    print(f" Train windows: {X_train_w.shape} | Unique traces: {len(np.unique(trace_ids_train))}")
    print(f" Val windows: {X_val_w.shape} | Unique traces: {len(np.unique(trace_ids_val))}")
    print(f" Attack windows: {X_attack_w.shape} | Unique traces: {len(np.unique(trace_ids_attack))}")

    # Export base payload
    print("\n3. Exporting Payload Arrays...")
    out_file = os.path.join(output_dir, "phase1_base_arrays.pkl")
    payload = {
        "X_train_w":        X_train_w,
        "X_val_w":          X_val_w,
        "X_attack_w":       X_attack_w,
        "trace_ids_train":  trace_ids_train,
        "trace_ids_val":    trace_ids_val,
        "trace_ids_attack": trace_ids_attack,
        "attack_family_labels": window_families_attack,
        "raw_train_traces": train_traces,
        "raw_val_traces":   val_traces,
        "raw_attack_traces": attack_traces
    }
    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    print(f" Successfully saved to {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

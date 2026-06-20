"""
Benchmark: ABI mismatch impact on detection (NICE TO HAVE #10).

FIXES vs old version:
  - remaps ALL paths (incl. Path A), not just B/C
  - uses REAL ABI permutations (i386 -> x86_64 / -> arm, by syscall identity)
    instead of an arbitrary additive (X+offset)%341; keeps a random "scrambled"
    control as worst case.
UNIFIED: raw windows from the shared pool; re-scores all paths + fusion. Inference only.

Output:
    experiments/32bit/logs/unified/abi_impact_analysis.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))

from benchmark_common import (  # noqa: E402
    PROJECT_ROOT, load_selected_path_a_bundle, load_path_b_model,
    load_path_c_model, load_phase5_meta_payload, score_path_a_bundle,
)
from pool_io import load_pool, save_json  # noqa: E402

MAP_DIR = os.path.join(str(PROJECT_ROOT), "scripts", "syscall_abi", "mappings")


def num2name(abi):
    d = json.load(open(os.path.join(MAP_DIR, f"{abi}.json")))["name_to_numbers"]
    out = {}
    for name, nums in d.items():
        for n in nums:
            out[int(n)] = name
    return out, {name: int(nums[0]) for name, nums in d.items()}


def build_remap(src_abi, dst_abi, vocab):
    """remap[id] = dst number for the same syscall name as src id (else 0=OOV)."""
    s_n2nm, _ = num2name(src_abi)
    _, d_nm2n = num2name(dst_abi)
    remap = np.zeros(vocab, dtype=np.int64)
    for i in range(vocab):
        nm = s_n2nm.get(i)
        tgt = d_nm2n.get(nm, 0) if nm else 0
        remap[i] = tgt if tgt < vocab else 0
    return remap


def apply_remap(X, remap):
    Xc = np.clip(X.astype(np.int64), 0, len(remap) - 1)
    return remap[Xc].astype(np.int16)


def score_all(args, X, device, bundle_train_X):
    # Path A: score X in both val/atk slots; take the (identical) val-slot scores
    bundle = load_selected_path_a_bundle(args.dataset, bundle_train_X, X, X)
    sa, _ = score_path_a_bundle(bundle)
    cnn, _ = load_path_b_model(args.dataset, device)
    gru, _ = load_path_c_model(args.dataset, device)
    crit = nn.CrossEntropyLoss(reduction="none")

    def cnn_s(Xs):
        o = []
        for i in range(0, len(Xs), 512):
            t = torch.tensor(Xs[i:i+512], dtype=torch.long, device=device)
            with torch.no_grad():
                p, e = cnn(t); o.append(torch.mean((e-p)**2, dim=[1,2]).cpu().numpy())
        return np.concatenate(o)

    def gru_s(Xs):
        o = []
        for i in range(0, len(Xs), 512):
            t = torch.tensor(Xs[i:i+512], dtype=torch.long, device=device)
            with torch.no_grad():
                o.append(crit(gru(t[:, :-1]).transpose(1, 2), t[:, 1:]).mean(dim=1).cpu().numpy())
        return np.concatenate(o)

    sb, sc = cnn_s(X), gru_s(X)
    meta = load_phase5_meta_payload(args.dataset)["model"]
    fus = meta.predict_proba(np.column_stack([sa, sb, sc]))[:, 1]
    return sa, sb, sc, fus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()
    device = torch.device("cpu")

    pool = load_pool(args.dataset)
    X = pool["X_eval"]
    y = pool["y_eval"]
    vocab = int(X.max()) + 1

    data_file = os.path.join(str(PROJECT_ROOT), "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    import pickle
    train_X = pickle.load(open(data_file, "rb"))["X_train_w"]

    rng = np.random.RandomState(42)
    scramble = rng.permutation(vocab).astype(np.int64)

    configs = {
        "correct_i386": X,
        "mismatch_as_x86_64": apply_remap(X, build_remap("i386", "x86_64", vocab)),
        "mismatch_as_arm_eabi": apply_remap(X, build_remap("i386", "arm_eabi", vocab)),
        "scrambled_control": apply_remap(X, scramble),
    }

    results = {}
    base = None
    for name, Xc in configs.items():
        sa, sb, sc, fus = score_all(args, Xc, device, train_X)
        m = {"auc_path_a": round(float(roc_auc_score(y, sa)), 4),
             "auc_path_b": round(float(roc_auc_score(y, sb)), 4),
             "auc_path_c": round(float(roc_auc_score(y, sc)), 4),
             "auc_fusion": round(float(roc_auc_score(y, fus)), 4)}
        if name == "correct_i386":
            base = m["auc_fusion"]
        results[name] = m
        print(f"  {name:<22} fusion AUC={m['auc_fusion']:.4f}")

    for name, m in results.items():
        m["fusion_degradation"] = round(base - m["auc_fusion"], 4)
        m["fusion_degradation_pct"] = round((base - m["auc_fusion"]) / base * 100, 2)

    out = {"pool": "unified_eval_pool", "vocab": vocab,
           "method": "real ABI permutation by syscall identity (i386->x86_64/arm) + random scramble control; ALL paths remapped",
           "configs": results}
    path = save_json(out, args.dataset, "abi_impact_analysis.json")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

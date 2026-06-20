"""
Build ONE canonical evaluation pool for all reviewer-response phases.

INFERENCE ONLY — loads already-trained models and scores them. Never calls
.fit(), never writes to models/. Writes a single artifact:
    experiments/<ds>/logs/unified/unified_eval_pool.pkl

Pool construction replicates reviewer_benchmark_loao_generalization.py exactly
(seed 42, np.random.choice 50k benign + 50k attack) so the published per-family
(avg AUC 0.9081) and synergy (29,568) numbers are reproduced.

Usage:
    python scripts/benchmark/unified/build_unified_eval_pool.py --dataset 32bit
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# scripts/benchmark on path for benchmark_common
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BENCH_DIR)

from benchmark_common import (  # noqa: E402
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_selected_path_a_bundle,
    score_path_a_bundle,
    load_path_b_model,
    load_path_c_model,
    load_phase5_meta_payload,
    resolve_phase5_selection,
)

META_FILES = {
    "Logistic_Regression": "meta_logistic_regression.pkl",
    "Random_Forest": "meta_random_forest.pkl",
    "XGBoost": "meta_xgboost.pkl",
    "Support_Vector_Machine": "meta_support_vector_machine.pkl",
    "Neural_Network_MLP": "meta_neural_network_mlp.pkl",
}


def unified_dir(dataset):
    d = os.path.join(str(PROJECT_ROOT), "experiments", dataset, "logs", "unified")
    os.makedirs(d, exist_ok=True)
    return d


def pool_path(dataset):
    return os.path.join(unified_dir(dataset), "unified_eval_pool.pkl")


def load_unified_pool(dataset=DEFAULT_DATASET):
    """Shared loader used by every phase script."""
    with open(pool_path(dataset), "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--pool_size", type=int, default=50000)
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    out_dir = unified_dir(args.dataset)
    device = torch.device("cpu")

    print("=" * 64)
    print(f"BUILD UNIFIED EVAL POOL ({args.dataset}) — INFERENCE ONLY")
    print("=" * 64)

    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    fam_labels = data.get("attack_family_labels")
    if fam_labels is None:
        raise SystemExit("ERROR: attack_family_labels missing; rerun data_pipeline.py")

    # --- Replicate LOAO eval-pool construction EXACTLY (seed 42) ---
    np.random.seed(42)
    n_benign = min(args.pool_size, len(data["X_val_w"]))
    idx_val = np.random.choice(len(data["X_val_w"]), n_benign, replace=False)
    X_val_w = data["X_val_w"][idx_val]
    trace_ids_val = data["trace_ids_val"][idx_val]

    attack_raw = data["X_attack_w"]
    ids_atk_raw = data["trace_ids_attack"]
    n_atk = min(args.pool_size, len(attack_raw))
    idx_atk = np.random.choice(len(attack_raw), n_atk, replace=False)
    X_attack_eval = attack_raw[idx_atk]
    trace_ids_attack_eval = ids_atk_raw[idx_atk]
    family_attack_eval = np.asarray(fam_labels)[idx_atk]

    X_eval = np.concatenate([X_val_w, X_attack_eval], axis=0)
    y_eval = np.concatenate([np.zeros(len(X_val_w)), np.ones(len(X_attack_eval))]).astype(np.int64)
    n_val_traces = int(trace_ids_val.max()) + 1 if len(trace_ids_val) else 0
    trace_ids_eval = np.concatenate([trace_ids_val, trace_ids_attack_eval + n_val_traces])
    family_eval = np.concatenate([np.array(["Benign"] * len(X_val_w)), family_attack_eval])

    print(f" Benign windows: {len(X_val_w)} | Attack windows: {len(X_attack_eval)}")
    print(f" Total: {len(X_eval)} | unique traces: {len(np.unique(trace_ids_eval))}")

    selection = resolve_phase5_selection(args.dataset)

    # --- Path A (inference) ---
    print(f"\nScoring Path A ({selection['path_a_label']})...")
    bundle_a = load_selected_path_a_bundle(args.dataset, data["X_train_w"], X_val_w, X_attack_eval)
    sa_val, sa_atk = score_path_a_bundle(bundle_a)
    s_a = np.concatenate([sa_val, sa_atk])

    # --- Path B (inference) ---
    print(f"Scoring Path B ({selection['path_b_label']})...")
    cnn, _ = load_path_b_model(args.dataset, device)

    def score_cnn(X):
        out = []
        for i in range(0, len(X), 512):
            t = torch.tensor(X[i:i + 512], dtype=torch.long, device=device)
            with torch.no_grad():
                pred, emb = cnn(t)
                out.append(torch.mean((emb - pred) ** 2, dim=[1, 2]).cpu().numpy())
        return np.concatenate(out)

    s_b = score_cnn(X_eval)

    # --- Path C (inference) ---
    print(f"Scoring Path C ({selection['path_c_label']})...")
    gru, _ = load_path_c_model(args.dataset, device)
    crit = nn.CrossEntropyLoss(reduction="none")

    def score_gru(X):
        out = []
        for i in range(0, len(X), 512):
            t = torch.tensor(X[i:i + 512], dtype=torch.long, device=device)
            with torch.no_grad():
                nll = crit(gru(t[:, :-1]).transpose(1, 2), t[:, 1:]).mean(dim=1)
                out.append(nll.cpu().numpy())
        return np.concatenate(out)

    s_c = score_gru(X_eval)

    Z_eval = np.column_stack([s_a, s_b, s_c]).astype(np.float64)

    # --- Fusion: all 5 saved meta-models (inference) ---
    print("\nScoring 5 meta-classifiers (inference)...")
    meta_dir = os.path.join(project_root, "models", args.dataset, "phase5_fusion")
    probas = {}
    for name, fname in META_FILES.items():
        fpath = os.path.join(meta_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARN missing {fname}; skipping {name}")
            continue
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        probas[name] = model.predict_proba(Z_eval)[:, 1].astype(np.float64)
        print(f"  {name:<22} AUC={roc_auc_score(y_eval, probas[name]):.4f}")

    tau_star = float(load_phase5_meta_payload(args.dataset).get("threshold_tau", 0.401214))

    # --- Thresholds (defined ONCE here) ---
    benign = y_eval == 0
    thresholds = {
        "path_a_p95_benign": float(np.percentile(s_a[benign], 95)),
        "path_b_p95_benign": float(np.percentile(s_b[benign], 95)),
        "path_c_p95_benign": float(np.percentile(s_c[benign], 95)),
        "fusion_tau_star": tau_star,
        "fusion_p95_benign": float(np.percentile(probas["XGBoost"][benign], 95)) if "XGBoost" in probas else None,
    }

    per_path_auc = {
        "path_a": float(roc_auc_score(y_eval, s_a)),
        "path_b": float(roc_auc_score(y_eval, s_b)),
        "path_c": float(roc_auc_score(y_eval, s_c)),
    }

    pool = {
        "artifact_type": "unified_eval_pool",
        "dataset": args.dataset,
        "selected_model": "XGBoost",
        "z_feature_order": ["path_a", "path_b", "path_c"],
        "X_eval": X_eval.astype(np.int16),
        "Z_eval": Z_eval,
        "scores": {"path_a": s_a, "path_b": s_b, "path_c": s_c},
        "proba": probas,
        "y_eval": y_eval,
        "trace_ids_eval": trace_ids_eval.astype(np.int64),
        "family_eval": family_eval.astype(object),
        "tau_star": tau_star,
        "thresholds": thresholds,
        "per_path_auc": per_path_auc,
        "n_benign": int(benign.sum()),
        "n_attack": int((~benign).sum()),
        "seed": 42,
    }
    with open(pool_path(args.dataset), "wb") as f:
        pickle.dump(pool, f)

    # --- Sanity gate ---
    print("\n--- SANITY ---")
    xgb_auc = roc_auc_score(y_eval, probas["XGBoost"]) if "XGBoost" in probas else None
    print(f" XGBoost full-pool fusion AUC: {xgb_auc:.4f}  (expected ~0.9322)")
    # per-family avg (benign vs each family) — expect ~0.9081
    fams = sorted(f for f in np.unique(family_eval) if f != "Benign")
    fam_aucs = {}
    for fam in fams:
        m = (family_eval == "Benign") | (family_eval == fam)
        fam_aucs[fam] = float(roc_auc_score((family_eval[m] == fam).astype(int), probas["XGBoost"][m]))
    avg_fam = float(np.mean(list(fam_aucs.values())))
    print(f" Per-family avg AUC: {avg_fam:.4f}  (paper LOAO avg 0.9081)")

    # compare Z vs saved z_matrix_eval_full (warn only)
    zsaved_path = os.path.join(project_root, "experiments", args.dataset, "logs", "z_matrix_eval_full.pkl")
    z_match = None
    if os.path.exists(zsaved_path):
        try:
            zs = pickle.load(open(zsaved_path, "rb"))["z_eval"]
            if zs.shape == Z_eval.shape:
                z_match = float(np.mean(np.abs(np.sort(zs[:, 0]) - np.sort(Z_eval[:, 0]))))
                print(f" Z[:,0] sorted-MAE vs saved z_matrix_eval_full: {z_match:.2e}")
        except Exception as e:
            print(f" (Z compare skipped: {e})")

    summary = {
        "dataset": args.dataset,
        "n_benign": pool["n_benign"],
        "n_attack": pool["n_attack"],
        "n_total": len(y_eval),
        "unique_traces": int(len(np.unique(trace_ids_eval))),
        "per_path_auc": per_path_auc,
        "per_model_fusion_auc": {k: float(roc_auc_score(y_eval, v)) for k, v in probas.items()},
        "per_family_auc": fam_aucs,
        "per_family_avg_auc": avg_fam,
        "thresholds": thresholds,
        "tau_star": tau_star,
        "z_sorted_mae_vs_saved": z_match,
        "note": "Inference-only pool. Replicates LOAO seed-42 construction. No retraining.",
    }
    with open(os.path.join(out_dir, "unified_eval_pool_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved pool: {pool_path(args.dataset)}")
    print(f"Saved summary: {os.path.join(out_dir, 'unified_eval_pool_summary.json')}")


if __name__ == "__main__":
    main()

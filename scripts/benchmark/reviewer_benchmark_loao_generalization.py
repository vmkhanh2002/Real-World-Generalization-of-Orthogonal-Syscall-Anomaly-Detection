"""
Per-family generalization benchmark for the saved Phase 5 meta-classifier.
Evaluates the loaded meta-classifier on each attack family slice without
retraining inside the benchmark script.
"""

import os, pickle, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import GroupShuffleSplit


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def aucpr_equal_pool(y_true: np.ndarray, scores: np.ndarray, seed: int = 42) -> float:
    """AUCPR after subsampling the majority class to match the minority class (1:1)."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    rng = np.random.default_rng(seed)
    if len(pos_idx) > n:
        pos_idx = rng.choice(pos_idx, size=n, replace=False)
    if len(neg_idx) > n:
        neg_idx = rng.choice(neg_idx, size=n, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    return float(average_precision_score(y_true[idx], scores[idx]))
import warnings

# To reuse models
from benchmark_common import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_phase5_meta_payload,
    load_path_b_model,
    load_path_c_model,
    load_selected_path_a_bundle,
    resolve_phase5_selection,
    score_path_a_bundle,
)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stringify_windows(X):
    return [" ".join(map(str, row)) for row in X]

def extract_class_metrics_dynamic_threshold(y_val_benign_proba, y_test, y_pred_proba):
    if len(np.unique(y_test)) < 2:
        return 0, 0, 0
    auc = roc_auc_score(y_test, y_pred_proba)
    aucpr = average_precision_score(y_test, y_pred_proba)

    # EDR Dynamic Thresholding logic: 99.5th percentile of normal traffic
    tau_edr = np.percentile(y_val_benign_proba, 99.5)

    y_pred_binary = (y_pred_proba > tau_edr).astype(int)

    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, y_pred_binary, zero_division=0)
    rec = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)

    return auc, aucpr, f1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, 'loao_generalization_results.txt'))

    print("=" * 60)
    print(f"PHASE 5: TRUE LOAO GENERALIZATION BENCHMARK ({args.dataset.upper()})")
    print("=" * 60)

    set_seed(42)

    # 1. Load data
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    model_dir = os.path.join(project_root, "models", args.dataset)

    print(f"\nINFO Loading {args.dataset} numpy arrays from {data_file}...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    attack_family_labels = data.get("attack_family_labels")
    if attack_family_labels is None:
        print("ERROR: array 'attack_family_labels' missing. Rerun data_pipeline.py.")
        return

    # Grab 50K benign validating windows
    np.random.seed(42)
    benign_pool_size = min(50000, len(data["X_val_w"]))
    idx_val = np.random.choice(len(data["X_val_w"]), benign_pool_size, replace=False)
    X_val_w = data["X_val_w"][idx_val]
    trace_ids_val = data["trace_ids_val"][idx_val]

    # Grab 50K Attack Windows (offset by 5000 exactly like phase 5)
    attack_raw = data["X_attack_w"]
    ids_atk_raw = data["trace_ids_attack"]
    fam_atk_raw = attack_family_labels
    assert len(attack_raw) > 1000, f"Attack pool too small: {len(attack_raw)}"

    atk_pool_size = min(50000, len(attack_raw))
    idx_atk = np.random.choice(len(attack_raw), atk_pool_size, replace=False)
    X_attack_eval = attack_raw[idx_atk]
    trace_ids_attack_eval = ids_atk_raw[idx_atk]
    family_attack_eval = fam_atk_raw[idx_atk]

    Y_eval = np.concatenate([np.zeros(len(X_val_w)), np.ones(len(X_attack_eval))])

    n_val_traces = int(trace_ids_val.max()) + 1 if len(trace_ids_val) > 0 else 0
    trace_ids_eval = np.concatenate([trace_ids_val, trace_ids_attack_eval + n_val_traces])

    # 0 for benign, rest string
    family_eval = np.concatenate([np.array(["Benign"]*len(X_val_w)), family_attack_eval])

    print(f" Eval Normal Pool: {X_val_w.shape}")
    print(f" Eval Attack Pool: {X_attack_eval.shape}")

    Z_scores = np.zeros((len(Y_eval), 3))
    selection = resolve_phase5_selection(args.dataset)

    # --- PATH A ---
    print(f"\nINFO Inferring Path A ({selection['path_a_label']})...")
    path_a_bundle = load_selected_path_a_bundle(args.dataset, data["X_train_w"], X_val_w, X_attack_eval)
    scores_a_val, scores_a_attack = score_path_a_bundle(path_a_bundle)
    Z_scores[:, 0] = np.concatenate([scores_a_val, scores_a_attack])

    # --- PATH B ---
    print(f"INFO Inferring Path B ({selection['path_b_label']})...")
    cnn, _ = load_path_b_model(args.dataset, device)

    val_t = torch.tensor(X_val_w, dtype=torch.long).to(device)
    atk_t = torch.tensor(X_attack_eval, dtype=torch.long).to(device)
    with torch.no_grad():
        v_pred, v_emb = cnn(val_t)
        a_pred, a_emb = cnn(atk_t)
        mse_v = torch.mean((v_emb - v_pred)**2, dim=[1,2]).cpu().numpy()
        mse_a = torch.mean((a_emb - a_pred)**2, dim=[1,2]).cpu().numpy()
    Z_scores[:, 1] = np.concatenate([mse_v, mse_a])

    # --- PATH C ---
    print(f"INFO Inferring Path C ({selection['path_c_label']})...")
    gru, _ = load_path_c_model(args.dataset, device)
    crit = nn.CrossEntropyLoss(reduction='none')
    def score_gru(t):
        with torch.no_grad():
            return crit(gru(t[:, :-1]).transpose(1, 2), t[:, 1:]).mean(dim=1).cpu().numpy()
    Z_scores[:, 2] = np.concatenate([score_gru(val_t), score_gru(atk_t)])

    print("\nINFO Loading saved Phase 5 meta-classifier for per-family evaluation")
    phase5_payload = load_phase5_meta_payload(args.dataset)
    meta_model = phase5_payload["model"]
    tau_star = float(phase5_payload["threshold_tau"])
    print(f" Model: {phase5_payload.get('model_name', 'Phase5_Meta')} | tau*={tau_star:.6f}")

    unique_families = sorted([f for f in np.unique(family_eval) if f != "Benign"])
    print(f" Detected Families: {unique_families}")

    W = 100
    print("\n" + "=" * W)
    print(" PER-FAMILY GENERALIZATION METRICS (Loaded Meta-Classifier)")
    print("=" * W)
    hdr = f"{'Held-Out Family':<20} | {'AUC':>7} | {'AUCPR':>7} | {'AUCPR(1:1)':>10} | {'F1@tau*':>8} | {'TPR@FPR1%':>10} | {'TPR@FPR0.1%':>12}"
    print(hdr)
    print("-" * W)

    aucs, aucprs, aucprs_eq, f1s, tprs_1, tprs_01 = [], [], [], [], [], []

    from sklearn.metrics import precision_score, recall_score

    for held_out in unique_families:
        benign_indices = np.where(family_eval == "Benign")[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        _, b_test_idx = next(gss.split(benign_indices, groups=trace_ids_eval[benign_indices]))

        real_b_test_idx  = benign_indices[b_test_idx]
        heldout_atk_indices = np.where(family_eval == held_out)[0]
        test_idx = np.concatenate([real_b_test_idx, heldout_atk_indices])
        X_test, y_test = Z_scores[test_idx], Y_eval[test_idx]

        y_pred_proba = meta_model.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) < 2:
            auc = aucpr = aucpr_eq = tpr1 = tpr01 = 0.0
        else:
            auc      = roc_auc_score(y_test, y_pred_proba)
            aucpr    = average_precision_score(y_test, y_pred_proba)
            aucpr_eq = aucpr_equal_pool(y_test, y_pred_proba)
            tpr1     = tpr_at_fpr(y_test, y_pred_proba, 0.01)
            tpr01    = tpr_at_fpr(y_test, y_pred_proba, 0.001)

        y_pred_binary = (y_pred_proba > tau_star).astype(int)
        prec = precision_score(y_test, y_pred_binary, zero_division=0)
        rec  = recall_score(y_test, y_pred_binary, zero_division=0)
        f1   = (2 * prec * rec) / (prec + rec + 1e-9)

        n_pos = int(y_test.sum())
        n_neg = int((y_test == 0).sum())
        print_name = held_out.replace("_", " ")
        print(f"{print_name:<20} | {auc:>7.4f} | {aucpr:>7.4f} | {aucpr_eq:>10.4f} | {f1:>8.4f} | {tpr1:>10.4f} | {tpr01:>12.4f}   n_pos={n_pos} n_neg={n_neg}")

        aucs.append(auc); aucprs.append(aucpr); aucprs_eq.append(aucpr_eq)
        f1s.append(f1);   tprs_1.append(tpr1);  tprs_01.append(tpr01)

    print("-" * W)
    print(f"{'Cross-Val Avg':<20} | {np.mean(aucs):>7.4f} | {np.mean(aucprs):>7.4f} | {np.mean(aucprs_eq):>10.4f} | {np.mean(f1s):>8.4f} | {np.mean(tprs_1):>10.4f} | {np.mean(tprs_01):>12.4f}")
    print("=" * W)
    print()
    print("Metrics:")
    print("  AUC          : ROC-AUC, threshold-free")
    print("  AUCPR        : Average Precision at natural pos/neg ratio")
    print("  AUCPR(1:1)   : Average Precision after subsampling majority class to 1:1")
    print("  F1@tau*      : F1 at fixed threshold tau* (from phase5 training)")
    print("  TPR@FPR1%    : True Positive Rate when False Positive Rate <= 1%")
    print("  TPR@FPR0.1%  : True Positive Rate when False Positive Rate <= 0.1%")
    print("\nINFO Operation Complete.")

if __name__ == "__main__":
    main()

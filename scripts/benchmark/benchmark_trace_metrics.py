import os, pickle, sys, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings("ignore")

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_edr_threshold(scores_benign, percentile=99.5):
    return np.percentile(scores_benign, percentile)

def stringify_windows(windows):
    return [" ".join(map(str, row)) for row in windows]

def fast_sample(matrix, trace_ids, size=50000):
    if len(matrix) <= size:
        return matrix, trace_ids
    unique_traces = np.unique(trace_ids)
    rng = np.random.RandomState(42)
    shuffled_traces = rng.permutation(unique_traces)

    selected_mask = np.zeros(len(matrix), dtype=bool)
    current_size = 0
    for t_id in shuffled_traces:
        t_mask = (trace_ids == t_id)
        t_size = np.sum(t_mask)
        if current_size + t_size <= size:
            selected_mask |= t_mask
            current_size += t_size
        else:
            remainder = size - current_size
            idx_of_trace = np.where(t_mask)[0]
            selected_mask[idx_of_trace[:remainder]] = True
            break
    return matrix[selected_mask], trace_ids[selected_mask]

# Model classes imported from models.py (Conv1DAE, GRUPredictor)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset)

    os.makedirs(log_dir, exist_ok=True)

    print("1. Loading Data...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    trace_ids_val = data["trace_ids_val"]
    trace_ids_attack = data["trace_ids_attack"]

    # Subsample data exactly as in phase5
    X_train_w, _ = fast_sample(data["X_train_w"], data.get("trace_ids_train", np.arange(len(data["X_train_w"]))), 50000)
    X_val_w, trace_ids_val = fast_sample(data["X_val_w"], trace_ids_val, 50000)
    X_attack_eval, trace_ids_attack_eval = fast_sample(data["X_attack_w"], trace_ids_attack, 50000)

    Y_eval = np.concatenate([np.zeros(len(X_val_w)), np.ones(len(X_attack_eval))])
    trace_ids_eval = np.concatenate([trace_ids_val, trace_ids_attack_eval + trace_ids_val.max() + 1])

    Z_scores = np.zeros((len(Y_eval), 3))
    selection = resolve_phase5_selection(args.dataset)

    print("2. Loading pre-trained models and extracting sub-scores...")
    # PATH A (selected Path A model)
    print(f" - Path A: {selection['path_a_label']} ({selection['source']})")
    path_a_bundle = load_selected_path_a_bundle(args.dataset, X_train_w, X_val_w, X_attack_eval)
    scores_a_val, scores_a_attack = score_path_a_bundle(path_a_bundle)
    Z_scores[:, 0] = np.concatenate([scores_a_val, scores_a_attack])

    # PATH B (selected Path B model)
    val_t = torch.tensor(X_val_w, dtype=torch.long).to(device)
    atk_t = torch.tensor(X_attack_eval, dtype=torch.long).to(device)
    cnn, _ = load_path_b_model(args.dataset, device)
    with torch.no_grad():
        val_pred, val_emb = cnn(val_t)
        atk_pred, atk_emb = cnn(atk_t)
        mse_v = torch.mean((val_emb - val_pred)**2, dim=[1,2]).cpu().numpy()
        mse_a = torch.mean((atk_emb - atk_pred)**2, dim=[1,2]).cpu().numpy()
    Z_scores[:, 1] = np.concatenate([mse_v, mse_a])

    # PATH C (selected Path C model)
    gru, _ = load_path_c_model(args.dataset, device)
    crit = nn.CrossEntropyLoss(reduction='none')
    def score_gru(t):
        with torch.no_grad():
            return crit(gru(t[:, :-1]).transpose(1, 2), t[:, 1:]).mean(dim=1).cpu().numpy()
    Z_scores[:, 2] = np.concatenate([score_gru(val_t), score_gru(atk_t)])

    print("3. Trace-level GroupShuffleSplit...")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, heldout_idx = next(gss1.split(Z_scores, Y_eval, groups=trace_ids_eval))

    heldout_groups = trace_ids_eval[heldout_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    local_val_idx, local_test_idx = next(gss2.split(Z_scores[heldout_idx], Y_eval[heldout_idx], groups=heldout_groups))
    val_idx  = heldout_idx[local_val_idx]
    test_idx = heldout_idx[local_test_idx]

    z_train, y_train = Z_scores[train_idx], Y_eval[train_idx]
    z_val,   y_val   = Z_scores[val_idx],   Y_eval[val_idx]
    z_test,  y_test  = Z_scores[test_idx],  Y_eval[test_idx]
    test_trace_ids = trace_ids_eval[test_idx]

    print("4. Loading selected Phase 5 meta-classifier...")
    phase5_payload = load_phase5_meta_payload(args.dataset)
    model = phase5_payload["model"]
    tau_star = float(phase5_payload["threshold_tau"])
    print(f" - Model: {phase5_payload.get('model_name', 'Phase5_Meta')}")
    print(f" - Loaded tau*: {tau_star:.6f}")

    test_preds = model.predict_proba(z_test)[:, 1]

    print("5. Evaluating trace-level metrics on test split...")
    # Group predictions by trace
    unique_test_traces = np.unique(test_trace_ids)

    trace_labels = []
    trace_anom_fracs = []

    for t_id in unique_test_traces:
        mask = (test_trace_ids == t_id)
        trace_y = y_test[mask]
        trace_p = (test_preds[mask] >= tau_star).astype(int)

        # A trace has a single ground truth label
        t_label = trace_y[0]
        # Calculate fraction of anomalous windows
        anom_frac = trace_p.mean()

        trace_labels.append(t_label)
        trace_anom_fracs.append(anom_frac)

    trace_labels = np.array(trace_labels)
    trace_anom_fracs = np.array(trace_anom_fracs)

    results = {}

    def eval_threshold(name, theta):
        trace_preds = (trace_anom_fracs >= theta).astype(int)
        TP = np.sum((trace_preds == 1) & (trace_labels == 1))
        FP = np.sum((trace_preds == 1) & (trace_labels == 0))
        TN = np.sum((trace_preds == 0) & (trace_labels == 0))
        FN = np.sum((trace_preds == 0) & (trace_labels == 1))

        TPR = TP / (TP + FN + 1e-8)
        FPR = FP / (FP + TN + 1e-8)

        print(f" --- {name} (Theta={theta:.2f}) ---")
        print(f" Trace TPR: {TPR:.4f} Trace FPR: {FPR:.4f}")

        # Operational PPV (Positive Predictive Value)
        # PPV = (TPR * BaseRate) / (TPR * BaseRate + FPR * (1 - BaseRate))
        base_rates = [0.01, 0.001] # 1%, 0.1% attacks
        ppv_results = {}
        for br in base_rates:
            ppv = (TPR * br) / (TPR * br + FPR * (1 - br) + 1e-8)
            print(f" Operational PPV @ {br*100}% Base Rate: {ppv:.4f}")
            ppv_results[str(br)] = ppv

        return {
            "TPR": TPR,
            "FPR": FPR,
            "PPV": ppv_results
        }

    # theta=0 means ANY anomalous window makes the trace anomalous (Strict Detection)
    results["Any_Window_1"] = eval_threshold("Strict - 1 Anomalous Window", 0.0001)

    # theta=0.05 means 5% of trace must be anomalous
    results["Threshold_5Pct"] = eval_threshold("Majority - 5% Windows", 0.05)

    out_file = os.path.join(log_dir, "trace_level_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nINFO Saved trace metrics to {out_file}")

if __name__ == "__main__":
    main()

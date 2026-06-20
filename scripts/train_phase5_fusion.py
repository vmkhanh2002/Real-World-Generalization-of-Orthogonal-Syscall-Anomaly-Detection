"""
Phase 5: The Meta-Classifier Assembly (Late Fusion)

Builds anomaly score-space representations from the three selected path models
(loaded from pipeline_config.py) and trains five meta-classifiers using a strict
trace-level 3-way split (60% train / 20% val / 20% test) with GroupShuffleSplit
to prevent data leakage from overlapping sliding windows.

Notes:
The meta models use balanced class weights where supported. XGBoost gets its
positive-class weight after the trace split, and threshold selection keeps a
guard for small validation sets.
"""

import os, pickle, sys, json, time
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
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_threshold(y_true, scores, threshold):
    """Apply a fixed threshold (selected on val set) to test scores. Returns F1."""
    preds = (scores >= threshold).astype(int)
    if preds.sum() == 0:
        # Warn when the validation threshold sits above every test score.
        print(f" WARNING tau*={threshold:.4f} exceeds max score={scores.max():.4f} "
              f"and gives zero positive predictions, F1=0.0")
        return 0.0
    return f1_score(y_true, preds, zero_division=0)


def require_file(path, label, fix_hint=None):
    """Raise a clear, actionable error when a required artifact is missing."""
    if os.path.exists(path):
        return
    msg = f"Missing required file for {label}: {path}"
    if fix_hint:
        msg += f"\n Suggestion: {fix_hint}"
    raise FileNotFoundError(msg)


def compute_fpr_at_tpr95(y_true, scores):
    """Compute FPR at TPR=95% from score distribution."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    if len(fpr) == 0:
        return float("nan")
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return float(fpr[-1])
    return float(fpr[idx[0]])


def compute_binary_metrics(y_true, scores, threshold):
    """
 Compute thresholded and threshold-free binary classification metrics.

 Thresholded metrics are evaluated at `threshold`, while AUC/AUPR/FPR95 are
 derived from score ranking.
 """
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    auc_roc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)
    fpr95 = compute_fpr_at_tpr95(y_true, scores)
    pos_rate = float(preds.mean())

    return {
        "AUC_ROC": float(auc_roc),
        "AUPR": float(aupr),
        "FPR_at_TPR95": float(fpr95),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "NPV": float(npv),
        "F1": float(f1),
        "Balanced_Accuracy": float(bal_acc),
        "MCC": float(mcc),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Positive_Pred_Rate": float(pos_rate),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def export_refreshed_z_splits(
    out_path,
    args,
    selected_paths,
    z_train,
    y_train,
    z_val,
    y_val,
    z_test,
    y_test,
    source_artifacts,
):
    payload = {
        "artifact_type": "phase5_refreshed_z_splits",
        "artifact_version": 1,
        "dataset": args.dataset,
        "protocol": "trace_level_manual_60_20_20_from_train_phase5_fusion",
        "max_windows_per_split": args.max_windows_per_split,
        "selected_paths": selected_paths,
        "z_train": z_train,
        "y_train": y_train,
        "z_val": z_val,
        "y_val": y_val,
        "z_test": z_test,
        "y_test": y_test,
        "source_artifacts": source_artifacts,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def export_refreshed_roc_curves(out_path, y_test, z_test, trained_meta_models, threshold_by_model):
    export = {}
    for i, path_name in enumerate(["Path_A", "Path_B", "Path_C"]):
        fpr, tpr, _ = roc_curve(y_test, z_test[:, i])
        export[path_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y_test, z_test[:, i])),
        }

    xgb_model = trained_meta_models["XGBoost"]
    xgb_scores = xgb_model.predict_proba(z_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, xgb_scores)
    tau = float(threshold_by_model["XGBoost"])
    preds = (xgb_scores >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr_op = fp / (fp + tn) if (fp + tn) else 0.0
    tpr_op = tp / (tp + fn) if (tp + fn) else 0.0

    export["XGBoost_Fusion"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(roc_auc_score(y_test, xgb_scores)),
        "operating_point": {
            "fpr": float(fpr_op),
            "tpr": float(tpr_op),
            "threshold": tau,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

def fast_sample(matrix, trace_ids, size=50000):
    """Subsample windows while preserving trace boundaries to stay within memory limits."""
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


def stringify_windows(windows):
    return [" ".join(map(str, row)) for row in windows]


from models import Conv1DAE, GRUPredictor
from model_metadata import load_torch_with_metadata
from pipeline_config import (
    apply_optional_overrides,
    get_fusion_defaults,
    get_path_a_best_configs,
    get_path_a_defaults,
    get_path_b_defaults,
    get_path_c_defaults,
    normalize_dataset_name,
)


def infer_conv1dae_kwargs_from_state_dict(state_dict):
    """Infer Conv1DAE architecture from state_dict tensor shapes."""
    if "embedding.weight" not in state_dict:
        raise KeyError("Conv1DAE state_dict missing 'embedding.weight' for fallback inference")
    vocab_size, embed_dim = state_dict["embedding.weight"].shape
    return {
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
    }


def build_meta_models(pos_weight: float) -> dict:
    """
 Instantiate five meta-classifiers with class imbalance handling.

 Design decisions:
 LR / SVM: QuantileTransformer pipeline because Z-scores span many orders of magnitude.
 (LOF ~O(10^9), MSE ~O(10^-4), NLL ~O(10^0)), making Euclidean-based
 optimizers ill-conditioned. QT maps to N(0,1) by rank, outlier-invariant.
 class_weight='balanced' corrects for benign/attack window imbalance.
 RF: tree splits are scale-invariant; 'balanced_subsample' resamples per tree,
 more robust than global 'balanced' for deep forest ensembles.
 XGBoost: rank-based splits are scale-invariant; scale_pos_weight dynamically
 set from actual train split (computed after GroupShuffleSplit).
 MLP: alpha=1e-3 (L2) + early_stopping prevents overconfident predictions
 that previously caused tau*=1.0 (no F1 peak in [0,1] probability range).
 Note: MLPClassifier does not support class_weight natively;
 PR-curve threshold selection compensates by shifting tau* downward.
 """
    return {
        # [FIX 2] class_weight='balanced'
        "Logistic_Regression": Pipeline([
            ("qt",  QuantileTransformer(output_distribution='normal', random_state=42)),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight='balanced'
            ))
        ]),

        # [FIX 3] class_weight='balanced_subsample'
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        ),

        # [FIX 4] scale_pos_weight from actual train ratio
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=4,
            random_state=42,
            scale_pos_weight=pos_weight
        ),

        # [FIX 2] class_weight='balanced'
        "Support_Vector_Machine": Pipeline([
            ("qt",  QuantileTransformer(output_distribution='normal', random_state=42)),
            ("clf", SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                class_weight='balanced'
            ))
        ]),

        "Neural_Network_MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            random_state=42,
            alpha=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
    }


def select_threshold_pr(model, z_val, y_val) -> tuple:
    """
 Select tau* via PR-curve F1 peak with Savitzky-Golay smoothing.

 Why not 99.5th-percentile=
 - Uses only benign scores (ignores attack signal entirely)
 - Overfits to the validation tail and becomes too conservative on test data.
 - Caused 54-84% valatest F1 drop on small attack val pools

 PR-curve argmax advantages:
 - Uses both benign AND attack val predictions
 - Directly optimizes F1 (primary paper metric)
 - SG smoothing prevents noise-peak traps on small val pools

 Returns: (tau_star, f1_at_tau_star)
 """
    val_preds = model.predict_proba(z_val)[:, 1]
    precision_pr, recall_pr, pr_thresholds = precision_recall_curve(y_val, val_preds)
    f1_pr      = 2 * precision_pr * recall_pr / (precision_pr + recall_pr + 1e-8)
    f1_pr_body = f1_pr[:-1]  # len(thresholds) = len(precision) 1

    # Savitzky-Golay needs an odd window of at least five points.
    if len(f1_pr_body) >= 5:
        raw_win = max(5, (len(f1_pr_body) // 10) * 2 + 1)
        win = min(21, raw_win)
        if win % 2 == 0:
            win += 1  # force odd
        try:
            from scipy.signal import savgol_filter
            f1_smooth = savgol_filter(f1_pr_body, window_length=win, polyorder=2)
        except Exception:
            f1_smooth = f1_pr_body
    else:
        f1_smooth = f1_pr_body  # too few points to smooth safely

    if len(pr_thresholds) > 0:
        best_idx  = np.argmax(f1_smooth)
        tau_star  = float(pr_thresholds[best_idx])
        f1_at_tau = float(f1_pr_body[best_idx])
    else:
        tau_star, f1_at_tau = 0.5, 0.0

    return tau_star, f1_at_tau


def _count_recurrent_layers(state_dict, prefix):
    idx = 0
    while f"{prefix}.weight_ih_l{idx}" in state_dict:
        idx += 1
    return max(idx, 1)


def infer_gru_kwargs_from_state_dict(state_dict):
    if "embedding.weight" not in state_dict or "gru.weight_ih_l0" not in state_dict:
        raise KeyError("GRU state_dict missing required keys for fallback inference")

    emb_weight = state_dict["embedding.weight"]
    vocab_size = int(emb_weight.shape[0] - 1)
    embed_dim = int(emb_weight.shape[1])

    hidden_dim = int(state_dict["gru.weight_ih_l0"].shape[0] // 3)
    num_layers = _count_recurrent_layers(state_dict, "gru")

    return {
        "vocab_size": max(vocab_size, 1),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }


def load_torch_model_for_fusion(
    checkpoint_path,
    model_label,
    class_registry,
    fallback_ctor,
    infer_kwargs_fn,
):
    """
 Metadata-first model reconstruction with robust legacy fallback.

 Flow:
 1) Try metadata sidecar (`*.meta.json`) via `load_torch_with_metadata`.
 2) If sidecar missing, utility falls back to provided constructor.
 3) If fallback constructor mismatches checkpoint shapes, infer architecture
 directly from state_dict tensor shapes and reload.
 """
    try:
        payload = load_torch_with_metadata(
            checkpoint_path,
            map_location=device,
            class_registry=class_registry,
            fallback_ctor=fallback_ctor,
            strict=True,
            require_metadata=False,
        )
        model = payload["model"].to(device)
        model.eval()

        if payload.get("metadata_used"):
            meta_class = payload.get("metadata", {}).get("model_class", "<unknown>")
            print(f" - {model_label}: loaded from metadata sidecar ({meta_class}).")
        elif payload.get("warning"):
            print(f" - {model_label}: {payload['warning']}")

        return model, payload
    except Exception as exc:
        print(f" WARNING {model_label}: metadata/fallback constructor failed ({exc}).")
        print(f" WARNING {model_label}: inferring architecture from state_dict shapes.")

        state_dict = torch.load(checkpoint_path, map_location=device)
        inferred_kwargs = infer_kwargs_fn(state_dict)
        model_cls = next(iter(class_registry.values()))
        model = model_cls(**inferred_kwargs).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        print(f" - {model_label}: loaded via inferred kwargs {inferred_kwargs}")
        return model, {
            "metadata_used": False,
            "metadata": None,
            "inferred_kwargs": inferred_kwargs,
            "warning": "Loaded without sidecar metadata via inferred architecture.",
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset namespace (e.g., 32bit or 64bit)")
    parser.add_argument("--max_windows_per_split", type=int, default=None,
                        help="Max windows retained per split before fusion scoring")
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)
    fusion_defaults = get_fusion_defaults(args.dataset)
    apply_optional_overrides(args, fusion_defaults, ["max_windows_per_split"])
    path_a_defaults = get_path_a_defaults(args.dataset)
    path_a_best_cfgs = get_path_a_best_configs(args.dataset)
    path_b_defaults = get_path_b_defaults(args.dataset)
    path_c_defaults = get_path_c_defaults(args.dataset)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir   = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    require_file(
        data_file,
        "processed dataset arrays",
        f"run `python scripts/data_pipeline.py --dataset {args.dataset}` first",
    )

    print("=" * 60)
    print(f"PHASE 5: META-CLASSIFIER ASSEMBLY ({args.dataset.upper()})")
    path_a_selected = str(path_a_defaults.get("selected_model", "sgd_ocsvm"))
    print(f"Selected Models (from pipeline_config.py):")
    print(f" Path A: {path_a_selected}")
    if path_a_best_cfgs:
        print(f" Best configs per model type: {', '.join(sorted(path_a_best_cfgs.keys()))}")
    path_b_selected = str(path_b_defaults.get("selected_model", "cnn1d_ae")).lower()
    path_c_selected = str(path_c_defaults.get("selected_model", "gru_predictor")).lower()
    print(f" Path B: {path_b_selected}")
    print(f" Path C: {path_c_selected}")
    print(f"Config: max_windows_per_split={args.max_windows_per_split}")
    print("=" * 60)

    print(f"\n1. Loading Base OS Matrices from {data_file}...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    has_trace_ids = "trace_ids_val" in data and "trace_ids_attack" in data
    if has_trace_ids:
        print(" INFO trace_ids loaded; using trace-level 3-way split.")
        trace_ids_val    = data["trace_ids_val"]
        trace_ids_attack = data["trace_ids_attack"]
    else:
        print(" WARNING trace_ids missing; run data_pipeline.py first!")
        trace_ids_val    = np.arange(len(data["X_val_w"]))
        trace_ids_attack = np.arange(len(data["X_attack_w"]))

    set_seed(42)

    X_train_w, trace_ids_train = fast_sample(
        data["X_train_w"],
        data.get("trace_ids_train", np.arange(len(data["X_train_w"]))),
        args.max_windows_per_split,
    )
    X_val_w, trace_ids_val = fast_sample(data["X_val_w"], trace_ids_val, args.max_windows_per_split)

    # All path models are unsupervised, so the full attack pool is used for evaluation.
    attack_raw  = data["X_attack_w"]
    ids_atk_raw = trace_ids_attack if has_trace_ids else np.arange(len(data["X_attack_w"]))
    assert len(attack_raw) > 1000, f"Attack pool too small: {len(attack_raw)} windows"
    X_attack_eval, trace_ids_attack_eval = fast_sample(attack_raw, ids_atk_raw, args.max_windows_per_split)

    Y_eval = np.concatenate([np.zeros(len(X_val_w)), np.ones(len(X_attack_eval))])
    n_val_traces   = int(trace_ids_val.max()) + 1 if len(trace_ids_val) > 0 else 0
    trace_ids_eval = np.concatenate([trace_ids_val, trace_ids_attack_eval + n_val_traces])

    print(f" Train Pool: {X_train_w.shape}")
    print(f" Eval Normal Pool: {X_val_w.shape} | Unique traces: {len(np.unique(trace_ids_val))}")
    print(f" Eval Attack Pool: {X_attack_eval.shape} | Unique traces: {len(np.unique(trace_ids_attack_eval))}")

    Z_scores = np.zeros((len(Y_eval), 3))

    # Path A score.
    print(f"\nINFO Loading Path A Model ({path_a_selected}) from pipeline_config...")
    path_a_dir = os.path.join(model_dir, "path_a")
    path_a_files = path_a_defaults.get("model_files", {})
    vectorizer_path = os.path.join(path_a_dir, path_a_files.get("vectorizer", "vec.pkl"))
    require_file(
        vectorizer_path,
        "Path A vectorizer",
        f"run `python scripts/train_path_a.py --dataset {args.dataset}` first",
    )

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    x_v_tfidf = vectorizer.transform(stringify_windows(X_val_w)).toarray()
    x_a_tfidf = vectorizer.transform(stringify_windows(X_attack_eval)).toarray()
    selected_key_to_file = {
        "sgd_ocsvm": path_a_files.get("sgd_ocsvm", path_a_files.get("classifier", "sgd_ocsvm.pkl")),
        "ifo": path_a_files.get("isolation_forest", "ifo.pkl"),
        "isolation_forest": path_a_files.get("isolation_forest", "ifo.pkl"),
        "lof": path_a_files.get("lof", "lof.pkl"),
        "hbos": path_a_files.get("hbos", "hbos.pkl"),
        "pca_error": path_a_files.get("pca_error", "pca_err.pkl"),
        "pca-err": path_a_files.get("pca_error", "pca_err.pkl"),
    }
    selected_file = selected_key_to_file.get(path_a_selected, path_a_files.get("classifier", "sgd_ocsvm.pkl"))
    selected_model_path = os.path.join(path_a_dir, selected_file)

    if path_a_selected in {"pca_error", "pca-err"}:
        require_file(
            selected_model_path,
            f"Path A selected model ({path_a_selected})",
            f"run `python scripts/train_path_a.py --dataset {args.dataset}` first",
        )
        with open(selected_model_path, "rb") as f:
            pca_err = pickle.load(f)
        X_full = np.vstack([x_v_tfidf, x_a_tfidf])
        X_recon = pca_err.inverse_transform(pca_err.transform(X_full))
        Z_scores[:, 0] = np.mean((X_full - X_recon) ** 2, axis=1)
    else:
        pca_path = os.path.join(path_a_dir, path_a_files.get("pca", "pca.pkl"))
        require_file(
            pca_path,
            "Path A reducer",
            f"run `python scripts/train_path_a.py --dataset {args.dataset}` first",
        )
        require_file(
            selected_model_path,
            f"Path A selected model ({path_a_selected})",
            f"run `python scripts/train_path_a.py --dataset {args.dataset}` first",
        )
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(selected_model_path, "rb") as f:
            path_a_model = pickle.load(f)
        x_v_red = pca.transform(x_v_tfidf)
        x_a_red = pca.transform(x_a_tfidf)
        X_red = np.vstack([x_v_red, x_a_red])
        if path_a_selected in {"sgd_ocsvm", "lof"}:
            Z_scores[:, 0] = -path_a_model.decision_function(X_red)
        elif path_a_selected in {"ifo", "isolation_forest"}:
            Z_scores[:, 0] = -path_a_model.score_samples(X_red)
        elif path_a_selected == "hbos":
            Z_scores[:, 0] = path_a_model.decision_function(X_red)
        else:
            raise ValueError(f"Unsupported Path A selected_model: {path_a_selected}")
    print(f" - Path A inference complete with selected_model={path_a_selected}.")

    # Path B score.
    print(f"\nINFO Loading Path B Model ({path_b_selected}) from pipeline_config...")
    MAX_ID = max(data["X_train_w"].max(), data["X_val_w"].max(), data["X_attack_w"].max())
    vocab_size_hint = int(MAX_ID + 1)

    path_b_files = path_b_defaults.get("model_files", {})
    path_b_selected_to_file = {
        "cnn1d_ae": path_b_files.get("cnn", "cnn.pth"),
        "cnn": path_b_files.get("cnn", "cnn.pth"),
    }
    if path_b_selected not in path_b_selected_to_file:
        raise ValueError(
            f"Unsupported Path B selected_model='{path_b_selected}' for Phase 5. "
            "Currently supported: cnn1d_ae."
        )
    path_b_ckpt_name = path_b_selected_to_file[path_b_selected]
    path_b_ckpt = os.path.join(model_dir, "path_b", path_b_ckpt_name)
    require_file(
        path_b_ckpt,
        f"Path B checkpoint ({path_b_selected})",
        f"run `python scripts/train_path_b.py --dataset {args.dataset}` first",
    )
    cnn, _cnn_payload = load_torch_model_for_fusion(
        checkpoint_path=path_b_ckpt,
        model_label="Path B / Conv1DAE",
        class_registry={"Conv1DAE": Conv1DAE},
        fallback_ctor=lambda: Conv1DAE(
            vocab_size=vocab_size_hint,
            embed_dim=int(path_b_defaults.get("embed_dim", 8)),
        ),
        infer_kwargs_fn=infer_conv1dae_kwargs_from_state_dict,
    )

    val_t = torch.tensor(X_val_w,       dtype=torch.long).to(device)
    atk_t = torch.tensor(X_attack_eval, dtype=torch.long).to(device)

    with torch.no_grad():
        val_pred, val_emb = cnn(val_t)
        atk_pred, atk_emb = cnn(atk_t)
        # MSE in embedding space
        mse_v = torch.mean((val_emb - val_pred) ** 2, dim=[1, 2]).cpu().numpy()
        mse_a = torch.mean((atk_emb - atk_pred) ** 2, dim=[1, 2]).cpu().numpy()
    Z_scores[:, 1] = np.concatenate([mse_v, mse_a])
    print(f" - Path B inference complete with selected_model={path_b_selected}.")

    # Path C score.
    print(f"\nINFO Loading Path C Model ({path_c_selected}) from pipeline_config...")
    path_c_files = path_c_defaults.get("model_files", {})
    path_c_selected_to_file = {
        "gru_predictor": path_c_files.get("gru", "gru.pth"),
        "gru": path_c_files.get("gru", "gru.pth"),
    }
    if path_c_selected not in path_c_selected_to_file:
        raise ValueError(
            f"Unsupported Path C selected_model='{path_c_selected}' for Phase 5. "
            "Currently supported: gru_predictor."
        )
    path_c_ckpt_name = path_c_selected_to_file[path_c_selected]
    path_c_ckpt = os.path.join(model_dir, "path_c", path_c_ckpt_name)
    require_file(
        path_c_ckpt,
        f"Path C checkpoint ({path_c_selected})",
        f"run `python scripts/train_path_c.py --dataset {args.dataset}` first",
    )
    gru, _gru_payload = load_torch_model_for_fusion(
        checkpoint_path=path_c_ckpt,
        model_label="Path C / GRUPredictor",
        class_registry={"GRUPredictor": GRUPredictor},
        fallback_ctor=lambda: GRUPredictor(
            vocab_size=vocab_size_hint,
            embed_dim=int(path_c_defaults.get("gru_embed_dim", 16)),
            hidden_dim=int(path_c_defaults.get("hidden_dim", 32)),
            num_layers=int(path_c_defaults.get("num_layers", 1)),
        ),
        infer_kwargs_fn=infer_gru_kwargs_from_state_dict,
    )
    crit = nn.CrossEntropyLoss(reduction='none')

    def score_gru(t):
        with torch.no_grad():
            return crit(gru(t[:, :-1]).transpose(1, 2), t[:, 1:]).mean(dim=1).cpu().numpy()

    Z_scores[:, 2] = np.concatenate([score_gru(val_t), score_gru(atk_t)])
    print(f" - Path C inference complete with selected_model={path_c_selected}.")

    # Trace-level fusion split.
    print("\nINFO Phase 5 Meta-Classifier Fusion...")
    print(f" Meta-Matrix Z shape: {Z_scores.shape}")

    if has_trace_ids:
        print(" INFO Performing Strict Trace-Level Stratified Split...")
        np.random.seed(42)

        norm_traces = np.unique(trace_ids_eval[Y_eval == 0])
        np.random.shuffle(norm_traces)
        n_tr = int(len(norm_traces) * 0.6)
        n_v  = int(len(norm_traces) * 0.2)
        nt_train = norm_traces[:n_tr]
        nt_val   = norm_traces[n_tr:n_tr + n_v]
        nt_test  = norm_traces[n_tr + n_v:]

        atk_traces = np.unique(trace_ids_eval[Y_eval == 1])
        np.random.shuffle(atk_traces)
        a_tr = max(1, int(len(atk_traces) * 0.6))
        a_v  = max(1, int(len(atk_traces) * 0.2))
        at_train = atk_traces[:a_tr]
        at_val   = atk_traces[a_tr:a_tr + a_v]
        at_test  = atk_traces[a_tr + a_v:]

        train_traces = np.concatenate([nt_train, at_train])
        val_traces   = np.concatenate([nt_val,   at_val])
        test_traces  = np.concatenate([nt_test,  at_test])

        assert len(set(train_traces) & set(val_traces))  == 0, "Trace leakage between train and val"
        assert len(set(train_traces) & set(test_traces)) == 0, "Trace leakage between train and test"
        assert len(set(val_traces)   & set(test_traces)) == 0, "Trace leakage between val and test"

        train_idx = np.where(np.isin(trace_ids_eval, train_traces))[0]
        val_idx   = np.where(np.isin(trace_ids_eval, val_traces))[0]
        test_idx  = np.where(np.isin(trace_ids_eval, test_traces))[0]

        z_train, y_train = Z_scores[train_idx], Y_eval[train_idx]
        z_val,   y_val   = Z_scores[val_idx],   Y_eval[val_idx]
        z_test,  y_test  = Z_scores[test_idx],  Y_eval[test_idx]

        print(f" Train: {len(z_train)} windows | Val: {len(z_val)} | Test: {len(z_test)}")
    else:
        from sklearn.model_selection import train_test_split
        z_tmp,   z_test,  y_tmp,   y_test  = train_test_split(
            Z_scores, Y_eval, test_size=0.2, random_state=42, stratify=Y_eval)
        z_train, z_val,   y_train, y_val   = train_test_split(
            z_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)
        print(" FALLBACK Random 60/20/20 window split.")

    # Compute pos_weight after the split from the real train class distribution.
    n_benign = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    pos_weight = n_benign / max(1, n_attack)
    print(f" Train class ratio benign:attack = {n_benign}:{n_attack} "
          f"(XGBoost scale_pos_weight={pos_weight:.2f})")

    source_artifacts = {
        "phase1_base_arrays": data_file,
        "path_a_vectorizer": vectorizer_path,
        "path_a_reducer": locals().get("pca_path"),
        "path_a_model": selected_model_path,
        "path_b_model": path_b_ckpt,
        "path_c_model": path_c_ckpt,
    }
    refreshed_z_path = os.path.join(log_dir, "z_matrix_refreshed.pkl")
    export_refreshed_z_splits(
        refreshed_z_path,
        args,
        {
            "path_a": path_a_selected,
            "path_b": path_b_selected,
            "path_c": path_c_selected,
        },
        z_train,
        y_train,
        z_val,
        y_val,
        z_test,
        y_test,
        source_artifacts,
    )
    print(f" Refreshed Z split export: {refreshed_z_path}")

    # QuantileTransformer lives inside the LR and SVM pipelines.
    meta_models = build_meta_models(pos_weight)
    results = {}
    trained_meta_models = {}
    threshold_by_model = {}

    def _safe_round(value, ndigits=4):
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return round(v, ndigits)

    for name, model in meta_models.items():
        print(f"\n Training: {name}...")
        model.fit(z_train, y_train)
        trained_meta_models[name] = model

        tau_star, _ = select_threshold_pr(model, z_val, y_val)
        threshold_by_model[name] = float(tau_star)

        val_preds = model.predict_proba(z_val)[:, 1]
        val_metrics = compute_binary_metrics(y_val, val_preds, tau_star)

        test_preds = model.predict_proba(z_test)[:, 1]
        test_metrics = compute_binary_metrics(y_test, test_preds, tau_star)
        auc_test = test_metrics["AUC_ROC"]
        f1_val = val_metrics["F1"]
        f1_test = test_metrics["F1"]

        print(f" Val: PR-curve tau*={tau_star:.4f} | F1_val={f1_val:.4f}")
        print(
            f" Test: AUC={auc_test:.4f} | AUPR={test_metrics['AUPR']:.4f} | "
            f"F1_test={f1_test:.4f} | Precision={test_metrics['Precision']:.4f} | "
            f"Recall={test_metrics['Recall']:.4f}"
        )

        results[name] = {
            "AUC_test": _safe_round(test_metrics["AUC_ROC"], 4),
            "AUPR_test": _safe_round(test_metrics["AUPR"], 4),
            "F1_val_dynamic": _safe_round(f1_val, 4),
            "F1_test_final": _safe_round(f1_test, 4),
            "threshold_tau": _safe_round(tau_star, 6),
            "Precision_test": _safe_round(test_metrics["Precision"], 4),
            "Recall_test": _safe_round(test_metrics["Recall"], 4),
            "Specificity_test": _safe_round(test_metrics["Specificity"], 4),
            "BalancedAcc_test": _safe_round(test_metrics["Balanced_Accuracy"], 4),
            "MCC_test": _safe_round(test_metrics["MCC"], 4),
            "NPV_test": _safe_round(test_metrics["NPV"], 4),
            "FPR_test": _safe_round(test_metrics["FPR"], 4),
            "FNR_test": _safe_round(test_metrics["FNR"], 4),
            "FPR_at_TPR95_test": _safe_round(test_metrics["FPR_at_TPR95"], 4),
            "PositivePredRate_test": _safe_round(test_metrics["Positive_Pred_Rate"], 4),
            "TP_test": int(test_metrics["TP"]),
            "FP_test": int(test_metrics["FP"]),
            "TN_test": int(test_metrics["TN"]),
            "FN_test": int(test_metrics["FN"]),
            "Val_metrics": {
                k: (_safe_round(v, 4) if isinstance(v, float) else v)
                for k, v in val_metrics.items()
            },
            "Test_metrics": {
                k: (_safe_round(v, 4) if isinstance(v, float) else v)
                for k, v in test_metrics.items()
            },
        }

    print("\n Test Metrics Summary (tau* from validation PR-curve):")
    print(
        f" {'Model':<26} {'AUC':>7} {'AUPR':>7} {'F1':>7} "
        f"{'Prec':>7} {'Rec':>7} {'Spec':>7} {'BalAcc':>8} {'MCC':>7}"
    )
    print(" " + "-" * 95)
    for model_name, metric_row in results.items():
        print(
            f" {model_name:<26} "
            f"{(metric_row['AUC_test'] if metric_row['AUC_test'] is not None else float('nan')):7.4f} "
            f"{(metric_row['AUPR_test'] if metric_row['AUPR_test'] is not None else float('nan')):7.4f} "
            f"{(metric_row['F1_test_final'] if metric_row['F1_test_final'] is not None else float('nan')):7.4f} "
            f"{(metric_row['Precision_test'] if metric_row['Precision_test'] is not None else float('nan')):7.4f} "
            f"{(metric_row['Recall_test'] if metric_row['Recall_test'] is not None else float('nan')):7.4f} "
            f"{(metric_row['Specificity_test'] if metric_row['Specificity_test'] is not None else float('nan')):7.4f} "
            f"{(metric_row['BalancedAcc_test'] if metric_row['BalancedAcc_test'] is not None else float('nan')):8.4f} "
            f"{(metric_row['MCC_test'] if metric_row['MCC_test'] is not None else float('nan')):7.4f}"
        )

    def _rank_key(model_name):
        metric_row = results[model_name]
        auc = metric_row.get("AUC_test")
        f1 = metric_row.get("F1_test_final")
        auc = float(auc) if auc is not None else float("-inf")
        f1 = float(f1) if f1 is not None else float("-inf")
        return (auc, f1)

    best_model_name = max(results.keys(), key=_rank_key)

    fusion_model_dir = os.path.join(model_dir, "phase5_fusion")
    os.makedirs(fusion_model_dir, exist_ok=True)

    def build_meta_artifact_payload(model_name):
        return {
            "artifact_type": "phase5_meta_model",
            "artifact_version": 1,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": args.dataset,
            "model_name": model_name,
            "model": trained_meta_models[model_name],
            "threshold_tau": threshold_by_model[model_name],
            "z_feature_order": [
                "path_a_anomaly_score",
                "path_b_anomaly_score",
                "path_c_anomaly_score",
            ],
            "selected_paths": {
                "path_a": path_a_selected,
                "path_b": path_b_selected,
                "path_c": path_c_selected,
            },
            "metrics": results[model_name],
            "train_windows": int(len(z_train)),
            "val_windows": int(len(z_val)),
            "test_windows": int(len(z_test)),
        }

    def _slug_model_name(model_name):
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_")

    def _write_model_payload(model_name, model_path, manifest_path):
        payload = build_meta_artifact_payload(model_name)
        with open(model_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "artifact_type": payload["artifact_type"],
                    "artifact_version": payload["artifact_version"],
                    "created_utc": payload["created_utc"],
                    "dataset": payload["dataset"],
                    "model_name": payload["model_name"],
                    "threshold_tau": _safe_round(payload["threshold_tau"], 6),
                    "z_feature_order": payload["z_feature_order"],
                    "selected_paths": payload["selected_paths"],
                    "metrics": payload["metrics"],
                    "train_windows": payload["train_windows"],
                    "val_windows": payload["val_windows"],
                    "test_windows": payload["test_windows"],
                },
                f,
                indent=4,
            )

    saved_models = []
    for model_name in trained_meta_models.keys():
        model_slug = _slug_model_name(model_name)
        model_path = os.path.join(fusion_model_dir, f"meta_{model_slug}.pkl")
        manifest_path = os.path.join(fusion_model_dir, f"meta_{model_slug}_manifest.json")
        _write_model_payload(model_name, model_path, manifest_path)
        saved_models.append(
            {
                "model_name": model_name,
                "model_file": model_path,
                "manifest_file": manifest_path,
            }
        )

    best_model_path = os.path.join(fusion_model_dir, "meta_best_model.pkl")
    best_manifest_path = os.path.join(fusion_model_dir, "meta_best_model_manifest.json")
    _write_model_payload(best_model_name, best_model_path, best_manifest_path)

    registry_path = os.path.join(fusion_model_dir, "meta_models_index.json")
    with open(registry_path, "w") as f:
        json.dump(
            {
                "artifact_type": "phase5_meta_model_registry",
                "artifact_version": 1,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": args.dataset,
                "selected_model_name": best_model_name,
                "selected_model_file": best_model_path,
                "selected_model_manifest": best_manifest_path,
                "models": saved_models,
            },
            f,
            indent=4,
        )

    print("\n Fusion Artifact Export:")
    print(f" - Saved meta-models: {len(saved_models)}")
    print(
        f" - Best model by AUC/F1: {best_model_name} "
        f"(AUC={results[best_model_name]['AUC_test']:.4f}, "
        f"F1={results[best_model_name]['F1_test_final']:.4f})"
    )
    print(f" - Selected model alias: {best_model_path}")
    print(f" - Model registry: {registry_path}")

    out_file = os.path.join(log_dir, "phase5_fusion_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    roc_out_file = os.path.join(log_dir, "roc_curves_refreshed.json")
    export_refreshed_roc_curves(
        roc_out_file,
        y_test,
        z_test,
        trained_meta_models,
        threshold_by_model,
    )
    print(f"INFO Refreshed ROC curves written to {roc_out_file}")

    print(f"\nINFO Done. Results written to {out_file}")


if __name__ == "__main__":
    main()

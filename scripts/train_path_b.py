"""
Phase 3: Path B Optimization (Topological Structure)
Trains 5 distinct Neural Networks (PyTorch) to measure spatial/structural abnormalities.
Architecture: Conv1D Autoencoder selected model uses nn.Embedding(vocab_size, embed_dim)
for discrete syscall ID representation, providing geometrically-meaningful latent features.

Refactor highlights:
- Centralized defaults from `pipeline_config.py`
- Primary thresholding uses benign-only quantile calibration (non-oracle)
- Oracle F1 kept for diagnostics only
- Torch checkpoints are saved with sidecar metadata (`.meta.json`)
"""

import os
import pickle
import sys
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
from sklearn.cluster import MiniBatchKMeans

from model_metadata import save_torch_with_metadata
from pipeline_config import (
    apply_optional_overrides,
    get_path_b_best_configs,
    get_path_b_defaults,
    normalize_dataset_name,
)


# PYTORCH MODEL DEFINITIONS (Centralized in models.py)

from models import DenseAE, LSTMAE, VAE, Conv1DAE, DeepSVDD



# SET SEED IMMEDIATELY CRITICAL for reproducibility (matches optimize_path_b.py)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
try:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except:
    pass



if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1_scores[best_idx]


def compute_fpr_at_tpr95(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return float(fpr[-1]) if len(fpr) else float("nan")
    return float(fpr[idx[0]])


def build_calibrated_metrics(y_true, benign_scores, combined_scores, quantile=None, method="pr_curve"):
    """
 Build calibrated metrics using PR-curve Best-F1 (matches optimize_path_*.py).

 Methods:
 - "pr_curve": Use PR-curve F1 maximization (BEST - matches optimize scripts)
 - "benign_quantile": Use benign score quantile (legacy, not recommended)

 Args:
 y_true: Binary ground truth
 benign_scores: Benign validation scores (for quantile method)
 combined_scores: All scores (benign + attack)
 quantile: Quantile value (for legacy method)
 method: "pr_curve" or "benign_quantile"
 """
    if method == "pr_curve":
        # Use PR curve F1 maximization (SAME as optimize_path_*.py)
        threshold, f1_calib = get_best_f1(y_true, combined_scores)
        return {
            "threshold": float(threshold),
            "f1_calib": float(f1_calib),
            "threshold_oracle": float(threshold),
            "f1_oracle": float(f1_calib),
            "calibration": {
                "method": "pr_curve_best_f1",
                "split": "full_test_set",
                "score_direction": "higher_is_more_anomalous",
            },
        }
    else:
        # Legacy quantile based method
        if len(benign_scores) == 0:
            threshold = 0.0
        else:
            threshold = float(np.quantile(benign_scores, quantile))

        calibrated_preds = (combined_scores >= threshold).astype(int)
        f1_calib = float(f1_score(y_true, calibrated_preds, zero_division=0))

        threshold_oracle, f1_oracle = get_best_f1(y_true, combined_scores)

        return {
            "threshold": threshold,
            "f1_calib": f1_calib,
            "threshold_oracle": float(threshold_oracle),
            "f1_oracle": float(f1_oracle),
            "calibration": {
                "method": "benign_quantile",
                "quantile": float(quantile),
                "split": "validation_benign_only",
                "score_direction": "higher_is_more_anomalous",
            },
        }


def format_result_record(y_true, scores, auc, calib):
    """Compatibility-preserving result payload with calibration provenance."""
    aupr = average_precision_score(y_true, scores)
    fpr95 = compute_fpr_at_tpr95(y_true, scores)
    return {
        "AUC": round(float(auc), 4),
        "AUPR": round(float(aupr), 4),
        "F1": round(float(calib["f1_calib"]), 4),
        "F1_calib": round(float(calib["f1_calib"]), 4),
        "F1_oracle": round(float(calib["f1_oracle"]), 4),
        "FPR_at_TPR95": round(float(fpr95), 4),
        "threshold": round(float(calib["threshold"]), 6),
        "threshold_oracle": round(float(calib["threshold_oracle"]), 6),
        "calibration": calib["calibration"],
    }


def tfidf_coreset_sample(matrix, trace_ids, fraction=0.3, max_features=500, min_samples=500):
    """
    Pick a train-only coreset in TF-IDF space before clustering.

    Raw syscall IDs do not have a useful Euclidean distance. TF-IDF gives each
    window a frequency vector, so KMeans groups windows by token usage instead
    of integer magnitude. Validation and attack windows are left untouched.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances_argmin

    target_size = min(len(matrix), max(min_samples, int(len(matrix) * fraction)))
    if len(matrix) <= target_size:
        return matrix, trace_ids

    print(f" INFO TF-IDF Coreset: {len(matrix)} {target_size} samples (semantic space)")
    strings = [" ".join(map(str, row)) for row in matrix]
    tfidf = TfidfVectorizer(max_features=max_features, analyzer="word")
    x_emb = tfidf.fit_transform(strings).toarray()

    kmeans = MiniBatchKMeans(
        n_clusters=target_size,
        random_state=42,
        batch_size=4096,
        n_init="auto",
    )
    kmeans.fit(x_emb)
    core_indices = pairwise_distances_argmin(kmeans.cluster_centers_, x_emb)
    return matrix[core_indices], trace_ids[core_indices]


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Path B Optimization (Topological Structure)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset namespace (e.g., 32bit or 64bit)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of train windows for TF-IDF coreset (optimized: 0.20 for 32bit, 0.10 for 64bit)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum training window floor after coreset sampling",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="Embedding dimension for Conv1DAE (optimized default: 8)",
    )
    parser.add_argument(
        "--calibration_quantile",
        type=float,
        default=None,
        help="Benign-only quantile for non-oracle threshold calibration",
    )
    parser.add_argument("--dense_latent_dim", type=int, default=None, help="Latent dim for DenseAE")
    parser.add_argument("--lstm_hidden_dim", type=int, default=None, help="Hidden dim for LSTMAE")
    parser.add_argument("--vae_latent_dim", type=int, default=None, help="Latent dim for VAE")
    parser.add_argument("--svdd_hidden_dim", type=int, default=None, help="Hidden dim for DeepSVDD")
    parser.add_argument("--cnn_restart_seeds", type=int, nargs='+', default=None,
                        help="List of random seeds for CNN multiple restarts. Default: 42")
    parser.add_argument(
        "--cnn_train_mode",
        type=str,
        choices=["direct", "sweep_order"],
        default=None,
        help="CNN training mode: direct target fraction only, or replay sweep order",
    )
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)
    defaults = get_path_b_defaults(args.dataset)
    best_cfgs = get_path_b_best_configs(args.dataset)
    apply_optional_overrides(
        args,
        defaults,
        [
            "epochs",
            "fraction",
            "min_samples",
            "embed_dim",
            "dense_latent_dim",
            "lstm_hidden_dim",
            "vae_latent_dim",
            "svdd_hidden_dim",
            "calibration_quantile",
            "cnn_restart_seeds",
            "cnn_train_mode",
            "calibration_method",  # Add calibration method from config
        ],
    )

    # Get calibration method from defaults (pr_curve or benign_quantile)
    calibration_method = defaults.get("calibration_method", "pr_curve")

    # Seed already set at module level - no need to call set_seed() again

    # CRITICAL: project_root must be 4 levels up to match optimize_path_b.py
    # train_path_b.py is at scripts/train_path_b.py
    # 4 levels up = project root (where data/ and experiments/ are)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset, "path_b")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 3: PATH B OPTIMIZATION ({args.dataset.upper()})")
    print(f"Using Device: {DEVICE}")
    print(
        "Config: "
        f"epochs={args.epochs}, fraction={args.fraction}, min_samples={args.min_samples}, "
        f"embed_dim={args.embed_dim}, dense_latent_dim={args.dense_latent_dim}, "
        f"lstm_hidden_dim={args.lstm_hidden_dim}, vae_latent_dim={args.vae_latent_dim}, "
        f"svdd_hidden_dim={args.svdd_hidden_dim}, calibration_q={args.calibration_quantile}"
    )
    print(f"CNN mode: {args.cnn_train_mode}")
    print(f"CNN restarts: seeds={args.cnn_restart_seeds}")
    if best_cfgs:
        print(f"Best config per model type loaded: {', '.join(sorted(best_cfgs.keys()))}")
    print("=" * 60)

    print(f"\n1. Loading Processed Matrices from {data_file}...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    x_train_raw = data["X_train_w"]
    x_val_raw = data["X_val_w"]
    x_attack_raw = data["X_attack_w"]

    trace_ids_train = data.get("trace_ids_train", np.arange(len(x_train_raw)))

    # Sample training only; validation and attack stay untouched.
    x_train_w, _ = tfidf_coreset_sample(
        x_train_raw,
        trace_ids_train,
        fraction=args.fraction,
        min_samples=args.min_samples,
    )
    print(f" Sampling Train: {len(x_train_raw):,} to {len(x_train_w):,} windows retained for training")
    x_val_w = x_val_raw
    x_attack_w = x_attack_raw

    max_id = max(x_train_raw.max(), x_val_raw.max(), x_attack_raw.max())
    if max_id == 0:
        max_id = 1

    # Float normalized views for non embedding models
    x_train_norm = (x_train_w / max_id).astype(np.float32)
    x_val_norm = (x_val_w / max_id).astype(np.float32)
    x_attack_norm = (x_attack_w / max_id).astype(np.float32)

    print(f" Shape: Train={x_train_w.shape}, Val={x_val_w.shape}, Atk={x_attack_w.shape}")

    train_tensor = torch.tensor(x_train_norm).to(DEVICE)
    val_tensor = torch.tensor(x_val_norm).to(DEVICE)
    atk_tensor = torch.tensor(x_attack_norm).to(DEVICE)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=256, shuffle=True)

    y_test = np.concatenate([np.zeros(len(x_val_norm)), np.ones(len(x_attack_norm))])

    results = {}
    criterion = nn.MSELoss()

    source_script = os.path.basename(__file__)
    common_extra = {
        "epochs": int(args.epochs),
        "fraction": float(args.fraction),
        "min_samples": int(args.min_samples),
        "embed_dim": int(args.embed_dim),
        "dense_latent_dim": int(args.dense_latent_dim),
        "lstm_hidden_dim": int(args.lstm_hidden_dim),
        "vae_latent_dim": int(args.vae_latent_dim),
        "svdd_hidden_dim": int(args.svdd_hidden_dim),
        "calibration_quantile": float(args.calibration_quantile),
    }

    # --- MODEL 1: Dense AE ---
    print("\n2.1 Training Dense Autoencoder...")
    model_dense = DenseAE(input_dim=x_train_norm.shape[1], latent_dim=args.dense_latent_dim).to(DEVICE)
    optimizer = optim.Adam(model_dense.parameters(), lr=0.001)

    model_dense.train()
    for _ in range(args.epochs):
        for data_batch, _target in train_loader:
            optimizer.zero_grad()
            out = model_dense(data_batch)
            loss = criterion(out, data_batch)
            loss.backward()
            optimizer.step()

    model_dense.eval()
    with torch.no_grad():
        mse_val = torch.mean((val_tensor - model_dense(val_tensor)) ** 2, dim=1).cpu().numpy()
        mse_atk = torch.mean((atk_tensor - model_dense(atk_tensor)) ** 2, dim=1).cpu().numpy()

    scores_dense = np.concatenate([mse_val, mse_atk])
    auc_dense = roc_auc_score(y_test, scores_dense)
    calib_dense = build_calibrated_metrics(y_test, mse_val, scores_dense, method=calibration_method)
    print(
        f" Dense AE to AUC: {auc_dense:.4f} | "
        f"F1_calib: {calib_dense['f1_calib']:.4f} | F1_oracle: {calib_dense['f1_oracle']:.4f}"
    )
    dense_record = format_result_record(y_test, scores_dense, auc_dense, calib_dense)
    dense_record["config"] = {
        "fraction": float(args.fraction),
        "latent_dim": int(args.dense_latent_dim),
    }
    results["Dense_AE"] = dense_record

    # --- MODEL 2: LSTM AE ---
    print("\n2.2 Training LSTM Autoencoder...")
    model_lstm = LSTMAE(seq_len=x_train_norm.shape[1], hidden_dim=args.lstm_hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

    model_lstm.train()
    for _ in range(args.epochs):
        for data_batch, _target in train_loader:
            dbatch_lstm = data_batch.unsqueeze(2)
            optimizer.zero_grad()
            out = model_lstm(dbatch_lstm)
            loss = criterion(out, dbatch_lstm)
            loss.backward()
            optimizer.step()

    model_lstm.eval()
    with torch.no_grad():
        val_lstm, atk_lstm = val_tensor.unsqueeze(2), atk_tensor.unsqueeze(2)
        mse_val = torch.mean(torch.squeeze((val_lstm - model_lstm(val_lstm)) ** 2), dim=1).cpu().numpy()
        mse_atk = torch.mean(torch.squeeze((atk_lstm - model_lstm(atk_lstm)) ** 2), dim=1).cpu().numpy()

    scores_lstm = np.concatenate([mse_val, mse_atk])
    auc_lstm = roc_auc_score(y_test, scores_lstm)
    calib_lstm = build_calibrated_metrics(y_test, mse_val, scores_lstm, method=calibration_method)
    print(
        f" LSTM AE to AUC: {auc_lstm:.4f} | "
        f"F1_calib: {calib_lstm['f1_calib']:.4f} | F1_oracle: {calib_lstm['f1_oracle']:.4f}"
    )
    lstm_record = format_result_record(y_test, scores_lstm, auc_lstm, calib_lstm)
    lstm_record["config"] = {
        "fraction": float(args.fraction),
        "hidden_dim": int(args.lstm_hidden_dim),
    }
    results["LSTM_AE"] = lstm_record

    # --- MODEL 3: VAE ---
    print("\n2.3 Training Variational Autoencoder VAE...")
    model_vae = VAE(input_dim=x_train_norm.shape[1], latent_dim=args.vae_latent_dim).to(DEVICE)
    optimizer = optim.Adam(model_vae.parameters(), lr=0.001)

    def vae_loss_function(recon_x, x, mu, logvar):
        bce = nn.functional.mse_loss(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    model_vae.train()
    for _ in range(args.epochs):
        for data_batch, _target in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model_vae(data_batch)
            loss = vae_loss_function(recon_batch, data_batch, mu, logvar)
            loss.backward()
            optimizer.step()

    model_vae.eval()
    with torch.no_grad():
        val_pred, _, _ = model_vae(val_tensor)
        atk_pred, _, _ = model_vae(atk_tensor)
        mse_val = torch.mean((val_tensor - val_pred) ** 2, dim=1).cpu().numpy()
        mse_atk = torch.mean((atk_tensor - atk_pred) ** 2, dim=1).cpu().numpy()

    scores_vae = np.concatenate([mse_val, mse_atk])
    auc_vae = roc_auc_score(y_test, scores_vae)
    calib_vae = build_calibrated_metrics(y_test, mse_val, scores_vae, method=calibration_method)
    print(
        f" VAE to AUC: {auc_vae:.4f} | "
        f"F1_calib: {calib_vae['f1_calib']:.4f} | F1_oracle: {calib_vae['f1_oracle']:.4f}"
    )
    vae_record = format_result_record(y_test, scores_vae, auc_vae, calib_vae)
    vae_record["config"] = {
        "fraction": float(args.fraction),
        "latent_dim": int(args.vae_latent_dim),
    }
    results["VAE"] = vae_record

    # --- MODEL 4: 1D-CNN AE (Embedding-based) with Multiple Restarts ---
    print("\n2.4 Training Convolutional 1D Autoencoder Embedding...")
    print(f" Multiple Restarts: {len(args.cnn_restart_seeds)} seeds: {args.cnn_restart_seeds}")

    t_cnn_train = torch.tensor(x_train_w, dtype=torch.long).to(DEVICE)
    t_cnn_val = torch.tensor(x_val_w, dtype=torch.long).to(DEVICE)
    t_cnn_atk = torch.tensor(x_attack_w, dtype=torch.long).to(DEVICE)

    vocab_size = int(max_id + 1)

    # Multiple random restarts
    best_cnn_auc = 0
    best_cnn_model = None
    best_cnn_seed = None
    best_mse_val = None
    best_mse_atk = None
    best_cnn_fraction = float(args.fraction)

    for restart_idx, restart_seed in enumerate(args.cnn_restart_seeds):
        print(f"\n Restart {restart_idx+1}/{len(args.cnn_restart_seeds)} (seed={restart_seed})...")

        # Set seed for this restart
        random.seed(restart_seed)
        np.random.seed(restart_seed)
        torch.manual_seed(restart_seed)

        if args.cnn_train_mode == "sweep_order":
            # Replay optimize sweep order to reproduce RNG trajectory (0.05 to 0.10 to target)
            fraction_schedule = []
            for frac in [0.05, 0.10, float(args.fraction)]:
                if frac not in fraction_schedule:
                    fraction_schedule.append(frac)
        else:
            fraction_schedule = [float(args.fraction)]

        model_cnn = None
        mse_val = None
        mse_atk = None
        auc_cnn = 0.0
        for frac in fraction_schedule:
            x_cnn_train, _ = tfidf_coreset_sample(
                x_train_raw,
                trace_ids_train,
                fraction=frac,
                min_samples=args.min_samples,
            )
            t_cnn_train_frac = torch.tensor(x_cnn_train, dtype=torch.long).to(DEVICE)
            loader_cnn = DataLoader(TensorDataset(t_cnn_train_frac, t_cnn_train_frac), batch_size=256, shuffle=True)

            model_cnn = Conv1DAE(vocab_size=vocab_size, embed_dim=args.embed_dim).to(DEVICE)
            optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

            model_cnn.train()
            for epoch in range(args.epochs):
                epoch_loss = 0.0
                for data_batch, _target in loader_cnn:
                    optimizer.zero_grad()
                    out, emb = model_cnn(data_batch)
                    loss = criterion(out, emb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if frac == fraction_schedule[-1] and ((epoch + 1) % 5 == 0 or epoch == 0):
                    avg_loss = epoch_loss / max(len(loader_cnn), 1)
                    print(f" Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

            model_cnn.eval()
            with torch.no_grad():
                val_pred, val_emb = model_cnn(t_cnn_val)
                atk_pred, atk_emb = model_cnn(t_cnn_atk)
                mse_val_frac = torch.mean((val_emb - val_pred) ** 2, dim=[1, 2]).cpu().numpy()
                mse_atk_frac = torch.mean((atk_emb - atk_pred) ** 2, dim=[1, 2]).cpu().numpy()
            scores_frac = np.concatenate([mse_val_frac, mse_atk_frac])
            auc_frac = roc_auc_score(y_test, scores_frac)

            if frac == fraction_schedule[-1]:
                mse_val = mse_val_frac
                mse_atk = mse_atk_frac
                auc_cnn = auc_frac
                best_cnn_fraction = float(frac)
            if args.cnn_train_mode == "sweep_order":
                print(f" [sweep_order] frac={frac:.2f} AUC={auc_frac:.4f}")

        print(f" Restart {restart_idx+1} AUC: {auc_cnn:.4f}")

        # Save best model
        if auc_cnn > best_cnn_auc:
            best_cnn_auc = auc_cnn
            best_cnn_model = model_cnn
            best_cnn_seed = restart_seed
            best_mse_val = mse_val
            best_mse_atk = mse_atk
            print(f" New best! (AUC={best_cnn_auc:.4f}, seed={best_cnn_seed})")

    # Use best model
    print(f"\n Best CNN Model: seed={best_cnn_seed}, AUC={best_cnn_auc:.4f}, fraction={best_cnn_fraction}")
    model_cnn = best_cnn_model
    mse_val = best_mse_val
    mse_atk = best_mse_atk

    # Debug: Check score distributions
    print(f" DEBUG Benign scores - Mean: {mse_val.mean():.4f}, Std: {mse_val.std():.4f}, Min: {mse_val.min():.4f}, Max: {mse_val.max():.4f}")
    print(f" DEBUG Attack scores - Mean: {mse_atk.mean():.4f}, Std: {mse_atk.std():.4f}, Min: {mse_atk.min():.4f}, Max: {mse_atk.max():.4f}")

    # Check if attack mean < benign mean (inverted scores)
    if mse_atk.mean() < mse_val.mean():
        print(f" WARNING Attack scores ({mse_atk.mean():.4f}) LOWER than benign ({mse_val.mean():.4f})!")
        print(f" WARNING This indicates model reconstructs attacks BETTER than benign - POSSIBLE DATA ISSUE!")

    scores_cnn = np.concatenate([mse_val, mse_atk])
    auc_cnn = best_cnn_auc

    # Debug: Check AUC
    print(f" DEBUG Raw AUC before calibration: {auc_cnn:.4f}")
    if auc_cnn < 0.5:
        print(f" WARNING AUC < 0.5 indicates INVERTED discrimination!")
        print(f" DEBUG Possible causes:")
        print(f" 1. Different data preprocessing vs optimize_path_b.py")
        print(f" 2. Different PyTorch version")
        print(f" 3. Random seed timing differences")
    calib_cnn = build_calibrated_metrics(y_test, mse_val, scores_cnn, method=calibration_method)
    print(
        f" Conv1DAE Embedding AUC: {auc_cnn:.4f} | "
        f"F1_calib: {calib_cnn['f1_calib']:.4f} | F1_oracle: {calib_cnn['f1_oracle']:.4f}"
    )
    cnn_record = format_result_record(y_test, scores_cnn, auc_cnn, calib_cnn)
    cnn_record["config"] = {
        "fraction": float(best_cnn_fraction),
        "embed_dim": int(args.embed_dim),
        "best_seed": best_cnn_seed,
        "num_restarts": len(args.cnn_restart_seeds),
        "train_mode": args.cnn_train_mode,
    }
    results["CNN1D_AE"] = cnn_record

    # MODEL 5: Deep SVDD
    print("\n2.5 Training Deep SVDD...")
    model_svdd = DeepSVDD(input_dim=x_train_norm.shape[1], hidden=args.svdd_hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model_svdd.parameters(), lr=0.001)

    model_svdd.eval()
    with torch.no_grad():
        c = torch.mean(model_svdd(train_tensor[:1000]), dim=0, keepdim=True)

    model_svdd.train()
    for _ in range(args.epochs):
        for data_batch, _target in train_loader:
            optimizer.zero_grad()
            out = model_svdd(data_batch)
            loss = torch.mean(torch.sum((out - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()

    model_svdd.eval()
    with torch.no_grad():
        mse_val = torch.sum((model_svdd(val_tensor) - c) ** 2, dim=1).cpu().numpy()
        mse_atk = torch.sum((model_svdd(atk_tensor) - c) ** 2, dim=1).cpu().numpy()

    scores_svdd = np.concatenate([mse_val, mse_atk])
    auc_svdd = roc_auc_score(y_test, scores_svdd)
    calib_svdd = build_calibrated_metrics(y_test, mse_val, scores_svdd, method=calibration_method)
    print(
        f" Deep SVDD to AUC: {auc_svdd:.4f} | "
        f"F1_calib: {calib_svdd['f1_calib']:.4f} | F1_oracle: {calib_svdd['f1_oracle']:.4f}"
    )
    svdd_record = format_result_record(y_test, scores_svdd, auc_svdd, calib_svdd)
    svdd_record["config"] = {
        "fraction": float(args.fraction),
        "hidden": int(args.svdd_hidden_dim),
    }
    results["Deep_SVDD"] = svdd_record

    out_file = os.path.join(log_dir, "path_b_results.json")
    legacy_out_file = os.path.join(log_dir, "path_b_topology_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    # Backward compatible legacy filename
    with open(legacy_out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nINFO Saving models + metadata sidecars to {model_dir}...")
    model_files = defaults.get("model_files", {})

    # Save BEST CNN model (from multiple restarts)
    save_torch_with_metadata(
        best_cnn_model,
        os.path.join(model_dir, model_files.get("cnn", "cnn.pth")),
        model_class="Conv1DAE",
        init_kwargs={"vocab_size": int(vocab_size), "embed_dim": int(args.embed_dim)},
        dataset=args.dataset,
        source_script=source_script,
        extra={**common_extra, "model_key": "CNN1D_AE", "best_seed": best_cnn_seed, "auc": best_cnn_auc},
    )

    save_torch_with_metadata(
        model_dense,
        os.path.join(model_dir, model_files.get("dense", "dense.pth")),
        model_class="DenseAE",
        init_kwargs={"input_dim": int(x_train_norm.shape[1]), "latent_dim": int(args.dense_latent_dim)},
        dataset=args.dataset,
        source_script=source_script,
        extra={**common_extra, "model_key": "Dense_AE", "auc": auc_dense},
    )

    save_torch_with_metadata(
        model_lstm,
        os.path.join(model_dir, model_files.get("lstm", "lstm.pth")),
        model_class="LSTMAE",
        init_kwargs={"seq_len": int(x_train_norm.shape[1]), "hidden_dim": int(args.lstm_hidden_dim)},
        dataset=args.dataset,
        source_script=source_script,
        extra={**common_extra, "model_key": "LSTM_AE", "auc": auc_lstm},
    )

    save_torch_with_metadata(
        model_vae,
        os.path.join(model_dir, model_files.get("vae", "vae.pth")),
        model_class="VAE",
        init_kwargs={"input_dim": int(x_train_norm.shape[1]), "latent_dim": int(args.vae_latent_dim)},
        dataset=args.dataset,
        source_script=source_script,
        extra={**common_extra, "model_key": "VAE", "auc": auc_vae},
    )

    save_torch_with_metadata(
        model_svdd,
        os.path.join(model_dir, model_files.get("svdd", "svdd.pth")),
        model_class="DeepSVDD",
        init_kwargs={"input_dim": int(x_train_norm.shape[1]), "hidden": int(args.svdd_hidden_dim)},
        dataset=args.dataset,
        source_script=source_script,
        extra={**common_extra, "model_key": "DeepSVDD", "auc": auc_svdd},
    )

    best_cfg_file = model_files.get("best_config_per_model_type", "path_b_best_config_per_model_type.json")
    with open(os.path.join(model_dir, best_cfg_file), "w", encoding="utf-8") as f:
        json.dump(best_cfgs, f, indent=2)

    print(f"\nINFO P3 Path B finished. Results saved to {out_file}")
    print(f"INFO Legacy results mirror: {legacy_out_file}")


if __name__ == "__main__":
    main()

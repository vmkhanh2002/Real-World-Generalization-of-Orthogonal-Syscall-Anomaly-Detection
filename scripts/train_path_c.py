"""
Phase 3: Path C Optimization (Temporal Chronology)
Trains 5 chronological model families for syscall sequence anomaly detection.

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
from collections import Counter, defaultdict

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

try:
    from hmmlearn.hmm import CategoricalHMM
    HMM_AVAILABLE = True
except ImportError:
    CategoricalHMM = None
    HMM_AVAILABLE = False

from model_metadata import save_torch_with_metadata
from pipeline_config import (
    apply_optional_overrides,
    get_path_c_best_configs,
    get_path_c_defaults,
    normalize_dataset_name,
)

from models import GRUPredictor, CBOWPredictor, LSTMAESequence


if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
 Primary threshold protocol (non-oracle):
 threshold = Quantile(benign validation scores, q)

 Oracle threshold/F1 is logged only for diagnostics.
 """
    if method == "pr_curve":
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


def transition_aware_sample(matrix, trace_ids, fraction=0.5, min_samples=5000):
    """
 Transition-aware sampling used by optimize_path_c_.py for sweep reproducibility.
 """
    target_size = min(len(matrix), max(min_samples, int(len(matrix) * fraction)))
    if len(matrix) <= target_size:
        return matrix, trace_ids

    print(f" INFO Transition-Aware Sample: {len(matrix)} to {target_size} windows")

    corpus_bigrams = Counter()
    for seq in matrix:
        for i in range(len(seq) - 1):
            corpus_bigrams[(int(seq[i]), int(seq[i + 1]))] += 1
    total = sum(corpus_bigrams.values()) or 1

    scores = []
    for seq in matrix:
        tb = Counter((int(seq[i]), int(seq[i + 1])) for i in range(len(seq) - 1))
        tt = sum(tb.values()) or 1
        scores.append(sum(min(tb[b] / tt, corpus_bigrams[b] / total) for b in tb))

    scores = np.asarray(scores, dtype=np.float64)
    probs = scores / (scores.sum() + 1e-12)
    chosen = np.random.choice(len(matrix), size=target_size, replace=False, p=probs)
    chosen = np.sort(chosen)
    return matrix[chosen], trace_ids[chosen]


def sanitize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if finite.all():
        return scores
    cap = float(np.nanmax(scores[finite])) if finite.any() else 1e6
    scores = np.where(np.isposinf(scores), cap, scores)
    scores = np.where(np.isneginf(scores), -cap, scores)
    scores = np.where(np.isnan(scores), 0.0, scores)
    return scores


def train_markov_model(matrix, order):
    transitions = defaultdict(Counter)
    probs_dict = defaultdict(dict)
    for seq in matrix:
        for i in range(len(seq) - order):
            state = tuple(int(x) for x in seq[i : i + order])
            nxt = int(seq[i + order])
            transitions[state][nxt] += 1
    for state, counter in transitions.items():
        total = sum(counter.values()) or 1
        for nxt, cnt in counter.items():
            probs_dict[state][nxt] = cnt / total
    return {"order": int(order), "probs": dict(probs_dict)}


def score_markov_sequences(model, matrix):
    order = int(model["order"])
    probs_dict = model["probs"]
    scores = []
    for seq in matrix:
        seq = np.asarray(seq)
        denom = max(len(seq) - order, 1)
        log_prob = 0.0
        for i in range(len(seq) - order):
            state = tuple(int(x) for x in seq[i : i + order])
            nxt = int(seq[i + order])
            if state in probs_dict and nxt in probs_dict[state]:
                log_prob += np.log(probs_dict[state][nxt] + 1e-9)
            else:
                log_prob += -10.0
        scores.append(-log_prob / denom)
    return sanitize_scores(scores)


def train_hmm_model(matrix, n_components):
    if not HMM_AVAILABLE:
        raise RuntimeError("hmmlearn is not installed in the active environment")
    flat = matrix.flatten().reshape(-1, 1)
    lengths = [matrix.shape[1]] * len(matrix)
    model = CategoricalHMM(n_components=int(n_components), n_iter=50, random_state=42)
    model.fit(flat, lengths)
    return model


def score_hmm_sequences(model, matrix):
    scores = []
    for seq in matrix:
        try:
            score = -model.score(np.asarray(seq).reshape(-1, 1)) / max(len(seq), 1)
        except Exception:
            score = np.inf
        scores.append(score)
    return sanitize_scores(scores)



# CBOW helper

def create_cbow_pairs(seq, context_size=5):
    pairs = []
    for i in range(context_size, len(seq) - context_size):
        context = seq[i - context_size : i] + seq[i + 1 : i + context_size + 1]
        target = seq[i]
        pairs.append((context, target))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Path C Optimization (Temporal Chronology)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset namespace (e.g., 32bit or 64bit)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of train windows for transition-aware sampling (optimized default: 0.50)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="GRU hidden dimension (optimized default: 192)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="GRU layer count (optimized default: 1)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum training window floor after coreset sampling",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Evaluation batch size for full validation/attack scoring",
    )
    parser.add_argument(
        "--calibration_quantile",
        type=float,
        default=None,
        help="Benign-only quantile for non-oracle threshold calibration",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        choices=["pr_curve", "benign_quantile"],
        default=None,
        help="Calibration method: pr_curve (default) or benign_quantile",
    )
    parser.add_argument("--gru_embed_dim", type=int, default=None)
    parser.add_argument("--cbow_embed_dim", type=int, default=None)
    parser.add_argument("--lstm_ae_embed_dim", type=int, default=None)
    parser.add_argument("--lstm_ae_hidden_dim", type=int, default=None)
    parser.add_argument("--lstm_ae_epochs", type=int, default=None)
    parser.add_argument("--cbow_epochs", type=int, default=None)

    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)
    defaults = get_path_c_defaults(args.dataset)
    best_cfgs = get_path_c_best_configs(args.dataset)
    apply_optional_overrides(
        args,
        defaults,
        [
            "epochs",
            "fraction",
            "hidden_dim",
            "num_layers",
            "min_samples",
            "eval_batch_size",
            "calibration_quantile",
            "gru_embed_dim",
            "cbow_embed_dim",
            "lstm_ae_embed_dim",
            "lstm_ae_hidden_dim",
            "lstm_ae_epochs",
            "cbow_epochs",
            "calibration_method",
        ],
    )

    set_seed(42)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset, "path_c")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print(f"PHASE 3: PATH C OPTIMIZATION ({args.dataset.upper()})")
    print(f"Using Device: {DEVICE}")
    print(
        "Config: "
        f"epochs={args.epochs}, fraction={args.fraction}, hidden_dim={args.hidden_dim}, "
        f"num_layers={args.num_layers}, min_samples={args.min_samples}, eval_batch_size={args.eval_batch_size}, "
        f"calibration_method={args.calibration_method}, calibration_q={args.calibration_quantile}, "
        f"gru_embed_dim={args.gru_embed_dim}, cbow_embed_dim={args.cbow_embed_dim}, "
        f"lstm_ae_embed_dim={args.lstm_ae_embed_dim}, lstm_ae_hidden_dim={args.lstm_ae_hidden_dim}, "
        f"lstm_ae_epochs={args.lstm_ae_epochs}, cbow_epochs={args.cbow_epochs}"
    )
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

    # Keep full validation/attack untouched for robust evaluation
    x_val_w = x_val_raw
    x_attack_w = x_attack_raw

    max_id = max(x_train_raw.max(), x_val_raw.max(), x_attack_raw.max())
    if max_id == 0:
        max_id = 1

    val_tensor_long = torch.tensor(x_val_w, dtype=torch.long).to(DEVICE)
    atk_tensor_long = torch.tensor(x_attack_w, dtype=torch.long).to(DEVICE)
    print(f" Shape: Train={x_train_raw.shape}, Val={x_val_w.shape}, Atk={x_attack_w.shape}")

    y_test = np.concatenate([np.zeros(len(x_val_w)), np.ones(len(x_attack_w))])
    results = {}

    sample_cache = {}

    def get_train_windows_for_fraction(frac):
        key = round(float(frac), 4)
        if key not in sample_cache:
            sampled, _ = transition_aware_sample(
                x_train_raw,
                trace_ids_train,
                fraction=float(frac),
                min_samples=args.min_samples,
            )
            sample_cache[key] = sampled
            print(
                f" Sampling fraction={float(frac):.2f}: "
                f"{len(x_train_raw):,} to {len(sampled):,} windows"
            )
        return sample_cache[key]

    source_script = os.path.basename(__file__)
    common_extra = {
        "epochs": int(args.epochs),
        "fraction": float(args.fraction),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "min_samples": int(args.min_samples),
        "eval_batch_size": int(args.eval_batch_size),
        "calibration_quantile": float(args.calibration_quantile),
        "calibration_method": str(args.calibration_method),
        "gru_embed_dim": int(args.gru_embed_dim),
        "cbow_embed_dim": int(args.cbow_embed_dim),
        "lstm_ae_embed_dim": int(args.lstm_ae_embed_dim),
        "lstm_ae_hidden_dim": int(args.lstm_ae_hidden_dim),
        "lstm_ae_epochs": int(args.lstm_ae_epochs),
        "cbow_epochs": int(args.cbow_epochs),
    }


    # MODEL 0.1: Markov Transition Model

    print("\n2.0 Training Markov Model...")
    markov_cfg = best_cfgs.get("Markov", {})
    markov_fraction = float(markov_cfg.get("fraction", 0.3))
    markov_order = int(markov_cfg.get("order", 3))
    print(f" Config: fraction={markov_fraction}, order={markov_order}")

    x_train_markov = get_train_windows_for_fraction(markov_fraction)
    markov_model = train_markov_model(x_train_markov, markov_order)
    score_val_markov = score_markov_sequences(markov_model, x_val_w)
    score_atk_markov = score_markov_sequences(markov_model, x_attack_w)
    scores_markov = np.concatenate([score_val_markov, score_atk_markov])
    auc_markov = roc_auc_score(y_test, scores_markov)
    calib_markov = build_calibrated_metrics(
        y_test,
        score_val_markov,
        scores_markov,
        quantile=args.calibration_quantile,
        method=args.calibration_method,
    )
    print(
        f" Markov to AUC: {auc_markov:.4f} | "
        f"F1_calib: {calib_markov['f1_calib']:.4f} | F1_oracle: {calib_markov['f1_oracle']:.4f}"
    )
    markov_record = format_result_record(y_test, scores_markov, auc_markov, calib_markov)
    markov_record["config"] = {
        "fraction": markov_fraction,
        "order": markov_order,
    }
    results["Markov"] = markov_record


    # MODEL 0.2: Hidden Markov Model

    print("\n2.0b Training Hidden Markov Model...")
    hmm_cfg = best_cfgs.get("HMM", {})
    hmm_fraction = float(hmm_cfg.get("fraction", 0.1))
    hmm_components = int(hmm_cfg.get("n_components", 4))
    print(
        f" Config: fraction={hmm_fraction}, n_components={hmm_components}, "
        f"hmm_available={HMM_AVAILABLE}"
    )

    if HMM_AVAILABLE:
        x_train_hmm = get_train_windows_for_fraction(hmm_fraction)
        hmm_model = train_hmm_model(x_train_hmm, hmm_components)
        score_val_hmm = score_hmm_sequences(hmm_model, x_val_w)
        score_atk_hmm = score_hmm_sequences(hmm_model, x_attack_w)
        scores_hmm = np.concatenate([score_val_hmm, score_atk_hmm])
        auc_hmm = roc_auc_score(y_test, scores_hmm)
        calib_hmm = build_calibrated_metrics(
            y_test,
            score_val_hmm,
            scores_hmm,
            quantile=args.calibration_quantile,
            method=args.calibration_method,
        )
        print(
            f" HMM to AUC: {auc_hmm:.4f} | "
            f"F1_calib: {calib_hmm['f1_calib']:.4f} | F1_oracle: {calib_hmm['f1_oracle']:.4f}"
        )
        hmm_record = format_result_record(y_test, scores_hmm, auc_hmm, calib_hmm)
        hmm_record["config"] = {
            "fraction": hmm_fraction,
            "n_components": hmm_components,
        }
        results["HMM"] = hmm_record
    else:
        print(" WARNING HMM skipped: hmmlearn is unavailable in the active environment")


    # MODEL 1: GRU Predictor (Primary Path C model)

    print("\n2.1 Training GRU Predictor...")
    gru_cfg = best_cfgs.get("GRU_Predictor", {})
    gru_fraction = float(gru_cfg.get("fraction", args.fraction))
    gru_embed_dim = int(gru_cfg.get("embed_dim", args.gru_embed_dim))
    gru_hidden_dim = int(gru_cfg.get("hidden_dim", args.hidden_dim))
    gru_num_layers = int(gru_cfg.get("num_layers", args.num_layers))
    print(
        f" Config: fraction={gru_fraction}, embed_dim={gru_embed_dim}, "
        f"hidden_dim={gru_hidden_dim}, num_layers={gru_num_layers}"
    )

    x_train_gru = get_train_windows_for_fraction(gru_fraction)
    train_tensor_long_gru = torch.tensor(x_train_gru, dtype=torch.long).to(DEVICE)

    input_seq_train = train_tensor_long_gru[:, :-1]
    target_seq_train = train_tensor_long_gru[:, 1:]

    input_seq_val = val_tensor_long[:, :-1]
    target_seq_val = val_tensor_long[:, 1:]

    input_seq_atk = atk_tensor_long[:, :-1]
    target_seq_atk = atk_tensor_long[:, 1:]

    train_dataset_gru = TensorDataset(input_seq_train, target_seq_train)
    train_loader_gru = DataLoader(train_dataset_gru, batch_size=256, shuffle=True)

    model_gru = GRUPredictor(
        vocab_size=int(max_id + 1),
        embed_dim=gru_embed_dim,
        hidden_dim=gru_hidden_dim,
        num_layers=gru_num_layers,
    ).to(DEVICE)

    criterion_seq = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model_gru.parameters(), lr=0.001)

    model_gru.train()
    for _ in range(args.epochs):
        for x_batch, y_batch in train_loader_gru:
            optimizer.zero_grad()
            preds = model_gru(x_batch)  # (B, T, V)
            loss = criterion_seq(preds.reshape(-1, preds.size(-1)), y_batch.reshape(-1)).mean()
            loss.backward()
            optimizer.step()

    # Full eval helper to avoid truncation bias
    def sequence_nll_scores(model, x_seq, y_seq, batch_size=512):
        ds = TensorDataset(x_seq, y_seq)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_scores = []
        model.eval()
        with torch.no_grad():
            for xb, yb in dl:
                out = model(xb)
                loss_tokens = criterion_seq(out.reshape(-1, out.size(-1)), yb.reshape(-1)).view(xb.size(0), -1)
                all_scores.append(loss_tokens.mean(dim=1).cpu().numpy())
        return np.concatenate(all_scores)

    scores_val = sequence_nll_scores(model_gru, input_seq_val, target_seq_val, batch_size=args.eval_batch_size)
    scores_atk = sequence_nll_scores(model_gru, input_seq_atk, target_seq_atk, batch_size=args.eval_batch_size)

    scores_gru = np.concatenate([scores_val, scores_atk])
    auc_gru = roc_auc_score(y_test, scores_gru)
    calib_gru = build_calibrated_metrics(
        y_test,
        scores_val,
        scores_gru,
        quantile=args.calibration_quantile,
        method=args.calibration_method,
    )
    print(
        f" GRU Predictor to AUC: {auc_gru:.4f} | "
        f"F1_calib: {calib_gru['f1_calib']:.4f} | F1_oracle: {calib_gru['f1_oracle']:.4f}"
    )
    gru_record = format_result_record(y_test, scores_gru, auc_gru, calib_gru)
    gru_record["config"] = {
        "fraction": gru_fraction,
        "embed_dim": gru_embed_dim,
        "hidden_dim": gru_hidden_dim,
        "num_layers": gru_num_layers,
    }
    results["GRU_Predictor"] = gru_record


    # MODEL 2: CBOW Predictor (Context prediction model)

    print("\n2.2 Training CBOW Predictor...")
    cbow_cfg = best_cfgs.get("CBOW_Predictor", {})
    cbow_fraction = float(cbow_cfg.get("fraction", args.fraction))
    cbow_embed_dim = int(cbow_cfg.get("embed_dim", args.cbow_embed_dim))
    print(f" Config: fraction={cbow_fraction}, embed_dim={cbow_embed_dim}")

    x_train_cbow = get_train_windows_for_fraction(cbow_fraction)
    train_tensor_long_cbow = torch.tensor(x_train_cbow, dtype=torch.long).to(DEVICE)

    # Subsample windows for CBOW efficiency (as before)
    train_subset = train_tensor_long_cbow[: min(10000, len(train_tensor_long_cbow))].cpu().numpy().tolist()
    val_subset = val_tensor_long[: min(5000, len(val_tensor_long))].cpu().numpy().tolist()
    atk_subset = atk_tensor_long[: min(5000, len(atk_tensor_long))].cpu().numpy().tolist()

    train_pairs = []
    for seq in train_subset:
        train_pairs.extend(create_cbow_pairs(seq, context_size=5))

    val_pairs = []
    for seq in val_subset:
        val_pairs.extend(create_cbow_pairs(seq, context_size=5))

    atk_pairs = []
    for seq in atk_subset:
        atk_pairs.extend(create_cbow_pairs(seq, context_size=5))

    if train_pairs and val_pairs and atk_pairs:
        context_train = torch.tensor([p[0] for p in train_pairs], dtype=torch.long).to(DEVICE)
        target_train = torch.tensor([p[1] for p in train_pairs], dtype=torch.long).to(DEVICE)

        context_val = torch.tensor([p[0] for p in val_pairs], dtype=torch.long).to(DEVICE)
        target_val = torch.tensor([p[1] for p in val_pairs], dtype=torch.long).to(DEVICE)

        context_atk = torch.tensor([p[0] for p in atk_pairs], dtype=torch.long).to(DEVICE)
        target_atk = torch.tensor([p[1] for p in atk_pairs], dtype=torch.long).to(DEVICE)

        cbow_dataset = TensorDataset(context_train, target_train)
        cbow_loader = DataLoader(cbow_dataset, batch_size=512, shuffle=True)

        model_cbow = CBOWPredictor(vocab_size=int(max_id + 1), embed_dim=cbow_embed_dim).to(DEVICE)
        criterion_cbow = nn.CrossEntropyLoss(reduction="none")
        optimizer_cbow = optim.Adam(model_cbow.parameters(), lr=0.001)

        model_cbow.train()
        for _ in range(args.cbow_epochs):
            for x_batch, y_batch in cbow_loader:
                optimizer_cbow.zero_grad()
                preds = model_cbow(x_batch)
                loss = criterion_cbow(preds, y_batch).mean()
                loss.backward()
                optimizer_cbow.step()

        model_cbow.eval()
        with torch.no_grad():
            val_preds = model_cbow(context_val)
            atk_preds = model_cbow(context_atk)
            score_val_cbow = criterion_cbow(val_preds, target_val).cpu().numpy()
            score_atk_cbow = criterion_cbow(atk_preds, target_atk).cpu().numpy()

        # Aggregate token-level losses back to sample groups (if needed downstream)
        min_len = min(len(score_val_cbow), len(score_atk_cbow))
        score_val_cbow = score_val_cbow[:min_len]
        score_atk_cbow = score_atk_cbow[:min_len]

        y_test_cbow = np.concatenate([np.zeros(len(score_val_cbow)), np.ones(len(score_atk_cbow))])
        scores_cbow = np.concatenate([score_val_cbow, score_atk_cbow])

        auc_cbow = roc_auc_score(y_test_cbow, scores_cbow)
        calib_cbow = build_calibrated_metrics(
            y_test_cbow,
            score_val_cbow,
            scores_cbow,
            quantile=args.calibration_quantile,
            method=args.calibration_method,
        )
        print(
            f" CBOW Predictor to AUC: {auc_cbow:.4f} | "
            f"F1_calib: {calib_cbow['f1_calib']:.4f} | F1_oracle: {calib_cbow['f1_oracle']:.4f}"
        )
        cbow_record = format_result_record(y_test_cbow, scores_cbow, auc_cbow, calib_cbow)
        cbow_record["config"] = {
            "fraction": cbow_fraction,
            "embed_dim": cbow_embed_dim,
        }
        results["CBOW_Predictor"] = cbow_record
    else:
        print(" WARNING CBOW skipped: insufficient generated context-target pairs")


    # MODEL 3: LSTM Autoencoder Sequence Reconstruction

    print("\n2.3 Training LSTM Sequence Autoencoder...")
    lstm_cfg = best_cfgs.get("LSTM_AE_Sequence", {})
    lstm_fraction = float(lstm_cfg.get("fraction", args.fraction))
    lstm_embed_dim = int(lstm_cfg.get("embed_dim", args.lstm_ae_embed_dim))
    lstm_hidden_dim = int(lstm_cfg.get("hidden_dim", args.lstm_ae_hidden_dim))
    print(
        f" Config: fraction={lstm_fraction}, embed_dim={lstm_embed_dim}, "
        f"hidden_dim={lstm_hidden_dim}"
    )

    x_train_lstm = get_train_windows_for_fraction(lstm_fraction)
    train_tensor_long_lstm = torch.tensor(x_train_lstm, dtype=torch.long).to(DEVICE)

    model_lstm_ae = LSTMAESequence(
        vocab_size=int(max_id + 1),
        embed_dim=lstm_embed_dim,
        hidden_dim=lstm_hidden_dim,
    ).to(DEVICE)

    optimizer_lstm_ae = optim.Adam(model_lstm_ae.parameters(), lr=0.001)
    criterion_lstm_ae = nn.CrossEntropyLoss(reduction="none")

    train_loader_ae = DataLoader(TensorDataset(train_tensor_long_lstm), batch_size=256, shuffle=True)

    model_lstm_ae.train()
    for _ in range(args.lstm_ae_epochs):
        for (batch_seq,) in train_loader_ae:
            optimizer_lstm_ae.zero_grad()
            logits = model_lstm_ae(batch_seq)  # (B, W, V)
            loss = criterion_lstm_ae(logits.reshape(-1, logits.size(-1)), batch_seq.reshape(-1)).mean()
            loss.backward()
            optimizer_lstm_ae.step()

    def seq_reconstruction_scores(model, seq_tensor, batch_size=512):
        loader = DataLoader(TensorDataset(seq_tensor), batch_size=batch_size, shuffle=False)
        all_scores = []
        model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                logits = model(xb)
                token_loss = criterion_lstm_ae(logits.reshape(-1, logits.size(-1)), xb.reshape(-1)).view(xb.size(0), -1)
                all_scores.append(token_loss.mean(dim=1).cpu().numpy())
        return np.concatenate(all_scores)

    score_val_lstm_ae = seq_reconstruction_scores(model_lstm_ae, val_tensor_long, batch_size=args.eval_batch_size)
    score_atk_lstm_ae = seq_reconstruction_scores(model_lstm_ae, atk_tensor_long, batch_size=args.eval_batch_size)

    scores_lstm_ae = np.concatenate([score_val_lstm_ae, score_atk_lstm_ae])
    auc_lstm_ae = roc_auc_score(y_test, scores_lstm_ae)
    calib_lstm_ae = build_calibrated_metrics(
        y_test,
        score_val_lstm_ae,
        scores_lstm_ae,
        quantile=args.calibration_quantile,
        method=args.calibration_method,
    )
    print(
        f" LSTM_AE_Sequence to AUC: {auc_lstm_ae:.4f} | "
        f"F1_calib: {calib_lstm_ae['f1_calib']:.4f} | F1_oracle: {calib_lstm_ae['f1_oracle']:.4f}"
    )
    lstm_record = format_result_record(y_test, scores_lstm_ae, auc_lstm_ae, calib_lstm_ae)
    lstm_record["config"] = {
        "fraction": lstm_fraction,
        "embed_dim": lstm_embed_dim,
        "hidden_dim": lstm_hidden_dim,
    }
    results["LSTM_AE_Sequence"] = lstm_record

    # Persist results
    out_file = os.path.join(log_dir, "path_c_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nINFO Saving Path C models + metadata sidecars to {model_dir}...")
    model_files = defaults.get("model_files", {})

    markov_model_path = os.path.join(model_dir, model_files.get("markov", "markov.pkl"))
    with open(markov_model_path, "wb") as f:
        pickle.dump(markov_model, f)
    with open(os.path.join(model_dir, model_files.get("markov_meta", "markov.meta.json")), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": "Markov",
                "dataset": args.dataset,
                "source_script": source_script,
                "config": markov_record["config"],
                "metrics": markov_record,
            },
            f,
            indent=2,
        )

    if HMM_AVAILABLE and "HMM" in results:
        hmm_model_path = os.path.join(model_dir, model_files.get("hmm", "hmm.pkl"))
        with open(hmm_model_path, "wb") as f:
            pickle.dump(hmm_model, f)
        with open(os.path.join(model_dir, model_files.get("hmm_meta", "hmm.meta.json")), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_key": "HMM",
                    "dataset": args.dataset,
                    "source_script": source_script,
                    "config": results["HMM"]["config"],
                    "metrics": results["HMM"],
                },
                f,
                indent=2,
            )

    save_torch_with_metadata(
        model_gru,
        os.path.join(model_dir, model_files.get("gru", "gru.pth")),
        model_class="GRUPredictor",
        init_kwargs={
            "vocab_size": int(max_id + 1),
            "embed_dim": int(gru_embed_dim),
            "hidden_dim": int(gru_hidden_dim),
            "num_layers": int(gru_num_layers),
        },
        dataset=args.dataset,
        source_script=source_script,
        extra={
            **common_extra,
            "model_key": "GRU_Predictor",
            "fraction": float(gru_fraction),
            "embed_dim": int(gru_embed_dim),
            "hidden_dim": int(gru_hidden_dim),
            "num_layers": int(gru_num_layers),
        },
    )

    if "CBOW_Predictor" in results:
        save_torch_with_metadata(
            model_cbow,
            os.path.join(model_dir, model_files.get("cbow", "cbow.pth")),
            model_class="CBOWPredictor",
            init_kwargs={
                "vocab_size": int(max_id + 1),
                "embed_dim": int(cbow_embed_dim),
            },
            dataset=args.dataset,
            source_script=source_script,
            extra={
                **common_extra,
                "model_key": "CBOW_Predictor",
                "fraction": float(cbow_fraction),
                "embed_dim": int(cbow_embed_dim),
            },
        )

    save_torch_with_metadata(
        model_lstm_ae,
        os.path.join(model_dir, model_files.get("lstm_ae", "lstm_ae.pth")),
        model_class="LSTMAESequence",
        init_kwargs={
            "vocab_size": int(max_id + 1),
            "embed_dim": int(lstm_embed_dim),
            "hidden_dim": int(lstm_hidden_dim),
        },
        dataset=args.dataset,
        source_script=source_script,
        extra={
            **common_extra,
            "model_key": "LSTM_AE_Sequence",
            "fraction": float(lstm_fraction),
            "embed_dim": int(lstm_embed_dim),
            "hidden_dim": int(lstm_hidden_dim),
        },
    )

    refreshed_best_cfgs = {
        "Markov": {
            **markov_record["config"],
            "AUC": markov_record["AUC"],
            "F1_calib": markov_record["F1_calib"],
            "source": "path_c_results.json",
        },
        "GRU_Predictor": {
            **gru_record["config"],
            "AUC": gru_record["AUC"],
            "F1_calib": gru_record["F1_calib"],
            "source": "path_c_results.json",
        },
        "LSTM_AE_Sequence": {
            **lstm_record["config"],
            "AUC": lstm_record["AUC"],
            "F1_calib": lstm_record["F1_calib"],
            "source": "path_c_results.json",
        },
    }

    if "CBOW_Predictor" in results:
        refreshed_best_cfgs["CBOW_Predictor"] = {
            **results["CBOW_Predictor"]["config"],
            "AUC": results["CBOW_Predictor"]["AUC"],
            "F1_calib": results["CBOW_Predictor"]["F1_calib"],
            "source": "path_c_results.json",
        }

    if "HMM" in results:
        refreshed_best_cfgs["HMM"] = {
            **results["HMM"]["config"],
            "AUC": results["HMM"]["AUC"],
            "F1_calib": results["HMM"]["F1_calib"],
            "source": "path_c_results.json",
        }

    best_cfg_file = model_files.get("best_config_per_model_type", "path_c_best_config_per_model_type.json")
    with open(os.path.join(model_dir, best_cfg_file), "w", encoding="utf-8") as f:
        json.dump(refreshed_best_cfgs, f, indent=2)

    print(f"\nINFO P3 Path C finished. Results saved to {out_file}")


if __name__ == "__main__":
    main()


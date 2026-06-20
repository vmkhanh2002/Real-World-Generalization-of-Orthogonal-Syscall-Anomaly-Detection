"""
Path C unified sweep (historical merge of V1 + V2 search spaces)
================================================================
This script merges the broad and targeted Path C sweeps into one replayable
optimization entrypoint. It is preserved for provenance and exploratory reruns.
The latest 32-bit active benchmark and paper-facing numbers must be taken from
the refreshed result artifacts instead of this merged sweep output.

Current 32-bit active references:
  - active holdout model: GRU_Predictor
  - active holdout metrics: AUC 0.8151, F1_calib 0.4313
  - nested trace-level paper lock: AUC 0.8140 +/- 0.0137, F1 0.4311 +/- 0.0248
  - source of truth: experiments/32bit/logs/path_c_results.json
  - paper artifact: experiments/32bit/logs/paper_nested_protocol_table.json

This unified script still writes:
  - experiments/<dataset>/logs/path_c_sweep_results.json
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure imports work when running from project root:
# python scripts/optimization/optimize_path_c_.py ...
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(THIS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from models import CBOWPredictor, GRUPredictor, LSTMAESequence
from pipeline_config import apply_optional_overrides, get_path_c_defaults, normalize_dataset_name

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import CategoricalHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_MODELS = {"Markov", "HMM", "LSTM_AE", "CBOW_IF", "GRU"}


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_best_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.argmax(f1))
    return thresholds[min(idx, len(thresholds) - 1)], float(f1[idx])


def sanitize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if finite.all():
        return scores
    cap = float(np.nanmax(scores[finite])) if finite.any() else 1e6
    scores = np.where(np.isposinf(scores), cap, scores)
    scores = np.where(np.isneginf(scores), -cap, scores)
    scores = np.where(np.isnan(scores), 0.0, scores)
    print(f" SANITIZE {int((~finite).sum())} non-finite values capped at {cap:.2f}")
    return scores


def score_in_batches(score_fn, tensor_cpu, batch_size=512):
    out = []
    for s in range(0, tensor_cpu.size(0), batch_size):
        out.append(score_fn(tensor_cpu[s:s + batch_size].to(DEVICE)))
    return np.concatenate(out)


def transition_aware_sample(matrix, trace_ids, target_size):
    if len(matrix) <= target_size:
        return matrix, trace_ids
    print(f" INFO Transition-Aware Sample: {len(matrix):,} to {target_size:,}")
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


def run_markov(X_train, X_val, X_atk, y_test, order=1, **_):
    transitions = defaultdict(Counter)
    probs_dict = defaultdict(dict)
    for seq in X_train:
        for i in range(len(seq) - order):
            state = tuple(seq[i:i + order])
            nxt = seq[i + order]
            transitions[state][nxt] += 1
    for state, ctr in transitions.items():
        tot = sum(ctr.values())
        for nxt, cnt in ctr.items():
            probs_dict[state][nxt] = cnt / tot

    def score_seq(seq):
        lp = 0.0
        for i in range(len(seq) - order):
            st = tuple(seq[i:i + order])
            nxt = seq[i + order]
            if st in probs_dict and nxt in probs_dict[st]:
                lp += np.log(probs_dict[st][nxt] + 1e-9)
            else:
                lp += -10.0
        return -lp / max(len(seq) - order, 1)

    sv = np.array([score_seq(s) for s in X_val])
    sa = np.array([score_seq(s) for s in X_atk])
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return float(auc), float(f1)


def run_hmm(X_train, X_val, X_atk, y_test, n_components=8, **_):
    if not HMM_AVAILABLE:
        return None, None
    flat = X_train.flatten().reshape(-1, 1)
    lens = [X_train.shape[1]] * len(X_train)
    model = CategoricalHMM(n_components=n_components, n_iter=50, random_state=42)
    model.fit(flat, lens)
    sv = [-model.score(s.reshape(-1, 1)) / len(s) for s in X_val]
    sa = [-model.score(s.reshape(-1, 1)) / len(s) for s in X_atk]
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return float(auc), float(f1)


def run_lstm_ae(X_train, X_val, X_atk, y_test, vocab_size, hidden_dim=32, epochs=5, eval_bs=512, **_):
    model = LSTMAESequence(vocab_size=vocab_size, hidden_dim=hidden_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss(reduction="none")

    t_train = torch.tensor(X_train, dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(t_train), batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (bx,) in loader:
            opt.zero_grad()
            logits = model(bx)
            loss = crit(logits.transpose(1, 2), bx).mean()
            loss.backward()
            opt.step()
    del t_train

    model.eval()
    val_cpu = torch.tensor(X_val, dtype=torch.long)
    atk_cpu = torch.tensor(X_atk, dtype=torch.long)

    def _score(b):
        with torch.no_grad():
            return crit(model(b).transpose(1, 2), b).mean(dim=1).cpu().numpy()

    sv = score_in_batches(_score, val_cpu, eval_bs)
    sa = score_in_batches(_score, atk_cpu, eval_bs)
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return float(auc), float(f1)


def run_cbow_if(
    X_train,
    X_val,
    X_atk,
    y_test,
    vocab_size,
    embed_dim=32,
    epochs=5,
    ctx_len_override=None,
    if_trees=100,
    **_,
):
    model = CBOWPredictor(vocab_size=vocab_size, embed_dim=embed_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()

    half = X_train.shape[1] // 2
    ctx_len = ctx_len_override if ctx_len_override is not None else min(half, 10)
    ctx_len = min(ctx_len, X_train.shape[1])

    t_ctx = torch.tensor(X_train[:,:ctx_len], dtype=torch.long).to(DEVICE)
    t_tgt = torch.tensor(X_train[:, half], dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(t_ctx, t_tgt), batch_size=256, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            if bx.size(1) < 10:
                pad = torch.zeros(bx.size(0), 10 - bx.size(1), dtype=torch.long, device=bx.device)
                bx = torch.cat([bx, pad], dim=1)
            elif bx.size(1) > 10:
                bx = bx[:,:10]
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
    del t_ctx, t_tgt

    model.eval()

    def extract_emb(arr):
        cpu = torch.tensor(arr, dtype=torch.long)
        out = []
        for s in range(0, len(cpu), 512):
            b = cpu[s:s + 512,:ctx_len].to(DEVICE)
            if b.size(1) < 10:
                pad = torch.zeros(b.size(0), 10 - b.size(1), dtype=torch.long, device=b.device)
                b = torch.cat([b, pad], dim=1)
            elif b.size(1) > 10:
                b = b[:,:10]
            with torch.no_grad():
                out.append(model.embedding(b).mean(dim=1).cpu().numpy())
        return np.vstack(out)

    X_tr_emb = extract_emb(X_train)
    X_vl_emb = extract_emb(X_val)
    X_ak_emb = extract_emb(X_atk)

    iso = IsolationForest(n_estimators=if_trees, contamination=0.05, random_state=42)
    iso.fit(X_tr_emb)
    sv = -iso.score_samples(X_vl_emb)
    sa = -iso.score_samples(X_ak_emb)
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return float(auc), float(f1)


def run_gru(
    X_train,
    X_val,
    X_atk,
    y_test,
    vocab_size,
    hidden_dim=64,
    num_layers=1,
    epochs=10,
    eval_bs=512,
    **_,
):
    model = GRUPredictor(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss(reduction="none")

    t_train = torch.tensor(X_train, dtype=torch.long).to(DEVICE)
    t_feat = t_train[:,:-1]
    t_targ = t_train[:, 1:]
    loader = DataLoader(TensorDataset(t_feat, t_targ), batch_size=256, shuffle=True)
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx).transpose(1, 2), by).mean()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        print(f" [GRU L{num_layers} ep {ep + 1}/{epochs}] loss={ep_loss / max(len(loader),1):.4f}")
    del t_train, t_feat, t_targ

    model.eval()
    val_cpu = torch.tensor(X_val, dtype=torch.long)
    atk_cpu = torch.tensor(X_atk, dtype=torch.long)

    def _score(b):
        with torch.no_grad():
            nll = crit(model(b[:,:-1]).transpose(1, 2), b[:, 1:])
            return nll.mean(dim=1).cpu().numpy()

    sv = score_in_batches(_score, val_cpu, eval_bs)
    sa = score_in_batches(_score, atk_cpu, eval_bs)
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return float(auc), float(f1)


def build_merged_sweep_configs():
    sweep = []

    # V1 broad sweep
    base_fracs = [0.05, 0.10, 0.20]
    for order in [1, 2]:
        for frac in base_fracs:
            sweep.append({"model": "Markov", "fraction": frac, "order": order})
    for n_components in [4, 8, 16]:
        for frac in base_fracs:
            sweep.append({"model": "HMM", "fraction": frac, "n_components": n_components})
    for hidden_dim in [16, 32, 64]:
        for frac in base_fracs:
            sweep.append({"model": "LSTM_AE", "fraction": frac, "hidden_dim": hidden_dim})
    for embed_dim in [16, 32, 64]:
        for frac in base_fracs:
            sweep.append({"model": "CBOW_IF", "fraction": frac, "embed_dim": embed_dim, "ctx_len": 10, "if_trees": 100})
    for hidden_dim in [32, 64, 128]:
        for frac in base_fracs:
            sweep.append({"model": "GRU", "fraction": frac, "hidden_dim": hidden_dim, "num_layers": 1})

    # V2 deep sweep extensions
    for hidden_dim in [128, 192, 256]:
        for num_layers in [1, 2]:
            for frac in [0.20, 0.30, 0.50]:
                sweep.append({"model": "GRU", "fraction": frac, "hidden_dim": hidden_dim, "num_layers": num_layers})
    for frac in [0.20, 0.30]:
        sweep.append({"model": "Markov", "fraction": frac, "order": 3})
    for ctx_len in [10, 15]:
        for if_trees in [100, 200]:
            for frac in [0.20, 0.30]:
                sweep.append(
                    {
                    "model": "CBOW_IF",
    "fraction": frac,
    "embed_dim": 16,
    "ctx_len": ctx_len,
    "if_trees": if_trees,
                    }
                )

    # Deduplicate by canonical signature
    uniq = []
    seen = set()
    for cfg in sweep:
        if cfg.get("model") not in ALLOWED_MODELS:
            continue
        sig = (
            cfg["model"],
            float(cfg["fraction"]),
            json.dumps({k: v for k, v in cfg.items() if k not in ("model", "fraction")}, sort_keys=True),
        )
        if sig not in seen:
            seen.add(sig)
            uniq.append(cfg)
    return uniq


def model_tag(cfg):
    m = cfg["model"]
    if m == "Markov":
        return f"Markov_o{cfg.get('order')}_f{cfg.get('fraction')}"
    if m == "HMM":
        return f"HMM_nc{cfg.get('n_components')}_f{cfg.get('fraction')}"
    if m == "LSTM_AE":
        return f"LSTM_AE_h{cfg.get('hidden_dim')}_f{cfg.get('fraction')}"
    if m == "CBOW_IF":
        return (
        f"CBOW_IF_e{cfg.get('embed_dim')}_ctx{cfg.get('ctx_len',10)}"
    f"_t{cfg.get('if_trees',100)}_f{cfg.get('fraction')}"
        )
    if m == "GRU":
        return f"GRU_h{cfg.get('hidden_dim')}_L{cfg.get('num_layers',1)}_f{cfg.get('fraction')}"
    return m


def best_row(rows, model_name):
    candidates = [r for r in rows if r.get("model") == model_name and r.get("AUC") is not None]
    return max(candidates, key=lambda r: r["AUC"]) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Unified Path C hyperparameter sweep (V1+V2).")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs for GRU")
    parser.add_argument("--ae_epochs", type=int, default=None, help="Training epochs for LSTM_AE / CBOW_IF")
    parser.add_argument("--min_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)
    defaults = get_path_c_defaults(args.dataset)
    apply_optional_overrides(args, defaults, ["epochs", "min_samples", "eval_batch_size"])
    if args.ae_epochs is None:
        args.ae_epochs = int(defaults.get("lstm_ae_epochs", 5))

    sweep_configs = build_merged_sweep_configs()
    total = len(sweep_configs)

    # __file__ = <project>/scripts/optimization/optimize_path_c_.py
    # project_root must be three levels up.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(log_dir, exist_ok=True)

    out_file = os.path.join(log_dir, "path_c_sweep_results.json")
    partial_file = os.path.join(log_dir, "path_c_sweep_results.partial.json")

    print("=" * 70)
    print(f"PATH C UNIFIED SWEEP | Dataset: {args.dataset.upper()} | Device: {DEVICE}")
    print("=" * 70)
    print(f" Configs: {total} | epochs={args.epochs} ae_epochs={args.ae_epochs}")
    if not HMM_AVAILABLE:
        print(" WARNING hmmlearn not installed to HMM configs will be skipped.")

    with open(data_file, "rb") as f:
        data = pickle.load(f)
    X_train_raw = data["X_train_w"]
    X_val_raw = data["X_val_w"]
    X_attack_raw = data["X_attack_w"]
    trace_ids_train = data.get("trace_ids_train", np.arange(len(X_train_raw)))

    vocab_size = int(max(X_train_raw.max(), X_val_raw.max(), X_attack_raw.max())) + 1
    y_test = np.concatenate([np.zeros(len(X_val_raw)), np.ones(len(X_attack_raw))])
    print(f" Train: {X_train_raw.shape} Val: {X_val_raw.shape} Attack: {X_attack_raw.shape} vocab={vocab_size}")

    # Resume support
    results = []
    if os.path.exists(partial_file):
        with open(partial_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            results = loaded
        elif isinstance(loaded, dict):
            results = loaded.get("results", [])
            print(f" Resume Loaded {len(results)} partial results")
    elif os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            results = loaded
        elif isinstance(loaded, dict):
            results = loaded.get("results", [])
            print(f" Resume Loaded {len(results)} existing results")

    done = {
        (
            r["config"]["model"],
            float(r["config"]["fraction"]),
            json.dumps({k: v for k, v in r["config"].items() if k not in ("model", "fraction")}, sort_keys=True),
        )
        for r in results
        if "config" in r
    }

    for i, cfg in enumerate(sweep_configs, start=1):
        sig = (
            cfg["model"],
            float(cfg["fraction"]),
            json.dumps({k: v for k, v in cfg.items() if k not in ("model", "fraction")}, sort_keys=True),
        )
        if sig in done:
            print(f" [{i}/{total}] SKIP cached: {cfg}")
            continue

        print("\n" + "-" * 70)
        print(f" [{i}/{total}] {cfg['model']} | cfg={cfg}")
        print("-" * 70)

        set_seed(42)
        target = max(int(args.min_samples), int(len(X_train_raw) * float(cfg["fraction"])))
        X_train, _ = transition_aware_sample(X_train_raw, trace_ids_train, target_size=target)
        print(f" Train sampled: {len(X_train_raw):,} to {len(X_train):,}")

        t0 = time.time()
        try:
            common = {
            "X_train": X_train,
    "X_val": X_val_raw,
    "X_atk": X_attack_raw,
    "y_test": y_test,
    "vocab_size": vocab_size,
    "eval_bs": int(args.eval_batch_size),
            }
            if cfg["model"] == "Markov":
                auc, f1 = run_markov(**common, order=int(cfg.get("order", 1)))
            elif cfg["model"] == "HMM":
                auc, f1 = run_hmm(**common, n_components=int(cfg.get("n_components", 8)))
            elif cfg["model"] == "LSTM_AE":
                auc, f1 = run_lstm_ae(**common, hidden_dim=int(cfg.get("hidden_dim", 32)), epochs=int(args.ae_epochs))
            elif cfg["model"] == "CBOW_IF":
                auc, f1 = run_cbow_if(
                    **common,
                    embed_dim=int(cfg.get("embed_dim", 16)),
                    epochs=int(args.ae_epochs),
                    ctx_len_override=int(cfg.get("ctx_len", 10)),
                    if_trees=int(cfg.get("if_trees", 100)),
                )
            elif cfg["model"] == "GRU":
                auc, f1 = run_gru(
                    **common,
                    hidden_dim=int(cfg.get("hidden_dim", 64)),
                    num_layers=int(cfg.get("num_layers", 1)),
                    epochs=int(args.epochs),
                )
            else:
                raise ValueError(f"Unknown model: {cfg['model']}")

            row = {
            "model": cfg["model"],
    "config": cfg,
    "AUC": round(float(auc), 4) if auc is not None else None,
    "F1_calib": round(float(f1), 4) if f1 is not None else None,
    "F1": round(float(f1), 4) if f1 is not None else None,
    "time_s": round(time.time() - t0, 1),
            }
            print(f" OK AUC={row['AUC']} F1={row['F1']} ({row['time_s']}s)")
        except Exception as exc:
            row = {
            "model": cfg["model"],
    "config": cfg,
    "AUC": None,
    "F1_calib": None,
    "F1": None,
    "time_s": round(time.time() - t0, 1),
    "error": str(exc),
            }
            print(f" ERROR {exc}")

        results.append(row)
        done.add(sig)
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    valid = [r for r in results if r.get("AUC") is not None]
    if not valid:
        raise RuntimeError("No valid sweep results to summarize.")

    # Best rows
    best_markov = best_row(valid, "Markov")
    best_hmm = best_row(valid, "HMM")
    best_lstm = best_row(valid, "LSTM_AE")
    best_cbow = best_row(valid, "CBOW_IF")
    best_gru = best_row(valid, "GRU")

    candidates = [r for r in [best_markov, best_hmm, best_lstm, best_cbow, best_gru] if r is not None]
    champion = max(candidates, key=lambda r: r["AUC"])

    all_configs = {model_tag(r["config"]): r for r in valid}

    best_cfg_per_model_type = {}
    if best_gru:
        best_cfg_per_model_type["GRU_Predictor"] = {
        "fraction": float(best_gru["config"].get("fraction", 0.5)),
    "embed_dim": 16,
    "hidden_dim": int(best_gru["config"].get("hidden_dim", 192)),
    "num_layers": int(best_gru["config"].get("num_layers", 1)),
    "AUC": float(best_gru["AUC"]),
    "F1_calib": float(best_gru["F1_calib"]),
        }
    if best_cbow:
        best_cfg_per_model_type["CBOW_Predictor"] = {
        "fraction": float(best_cbow["config"].get("fraction", 0.2)),
    "embed_dim": int(best_cbow["config"].get("embed_dim", 16)),
    "AUC": float(best_cbow["AUC"]),
    "F1_calib": float(best_cbow["F1_calib"]),
        }
    if best_lstm:
        best_cfg_per_model_type["LSTM_AE_Sequence"] = {
        "fraction": float(best_lstm["config"].get("fraction", 0.2)),
    "embed_dim": 16,
    "hidden_dim": int(best_lstm["config"].get("hidden_dim", 64)),
    "AUC": float(best_lstm["AUC"]),
    "F1_calib": float(best_lstm["F1_calib"]),
        }
    if best_markov:
        best_cfg_per_model_type["Markov"] = {
        "fraction": float(best_markov["config"].get("fraction", 0.3)),
    "order": int(best_markov["config"].get("order", 3)),
    "AUC": float(best_markov["AUC"]),
    "F1_calib": float(best_markov["F1_calib"]),
        }
    if best_hmm:
        best_cfg_per_model_type["HMM"] = {
        "fraction": float(best_hmm["config"].get("fraction", 0.1)),
    "n_components": int(best_hmm["config"].get("n_components", 4)),
    "AUC": float(best_hmm["AUC"]),
    "F1_calib": float(best_hmm["F1_calib"]),
        }

    summary = {
    "config_used": {
    "dataset": args.dataset,
    "epochs": int(args.epochs),
    "ae_epochs": int(args.ae_epochs),
    "min_samples": int(args.min_samples),
    "eval_batch_size": int(args.eval_batch_size),
    "defaults_from_pipeline_config": True,
    "hmm_available": bool(HMM_AVAILABLE),
    "total_configs": int(total),
        },
        "Markov_best": best_markov,
    "HMM_best": best_hmm,
    "LSTM_AE_best": best_lstm,
    "CBOW_IF_best": best_cbow,
    "GRU_best": best_gru,
    "champion": {
    "model": champion["model"],
    "AUC": float(champion["AUC"]),
    "F1": float(champion["F1"]),
    "config": champion["config"],
        },
        "best_config_per_model_type": best_cfg_per_model_type,
    "all_configs": all_configs,
    "results": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

        print("\n" + "=" * 70)
    print("SWEEP COMPLETE - SUMMARY")
    print("=" * 70)
    print(
    f" Champion: {summary['champion']['model']} "
    f"AUC={summary['champion']['AUC']:.4f} F1={summary['champion']['F1']:.4f}"
    )
    print(f" Output: {out_file}")
    print(f" Partial: {partial_file}")


if __name__ == "__main__":
    set_seed(42)
    main()


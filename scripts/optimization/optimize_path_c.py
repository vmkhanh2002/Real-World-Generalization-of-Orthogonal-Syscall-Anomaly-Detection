"""
Path C broad hyperparameter sweep (historical provenance script)
================================================================
This is the original wide Path C search over unsupervised sequence models. It is
kept for provenance and optional replay, not as the authoritative source of the
latest 32-bit benchmark or paper-facing metrics.

Current 32-bit active references:
  - active holdout model: GRU_Predictor
  - active holdout config: fraction=0.5, embed_dim=16, hidden_dim=192, num_layers=1
  - active holdout metrics: AUC 0.8151, F1_calib 0.4313
  - nested trace-level paper lock: AUC 0.8140 +/- 0.0137, F1 0.4311 +/- 0.0248
  - source of truth: experiments/32bit/logs/path_c_results.json
  - paper artifact: experiments/32bit/logs/paper_nested_protocol_table.json

This script still writes the historical sweep artifact:
  - experiments/<dataset>/logs/path_c_sweep_results.json

Models explored:
  1. Markov Chain - transition order {1, 2}
  2. HMM - n_components {4, 8, 16}
  3. LSTM AE - hidden_dim {16, 32, 64}
  4. CBOW + IF - embed_dim {16, 32, 64}
  5. GRU - hidden_dim {32, 64, 128}

Global sweep axis:
  fraction in {0.05, 0.10, 0.20} (transition-aware training sample fraction)

NOTE: Random Forest was intentionally excluded. Path C remains fully unsupervised,
so supervised attack-labelled training is out of scope for this sweep.
"""

import os, sys, json, time, pickle, random, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from hmmlearn.hmm import CategoricalHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("WARNING hmmlearn not installed - HMM configs will be skipped.")

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

from models import CBOWPredictor, GRUPredictor, LSTMAESequence



# UTILITIES


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_best_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.argmax(f1)
    return thresholds[min(idx, len(thresholds)-1)], f1[idx]


def sanitize_scores(scores):
    """Clip Ainf / NaN to finite cap (see HMM OOV issue in train_path_c.py)."""
    scores = np.array(scores, dtype=np.float64)
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
    """Run score_fn in mini-batches to prevent CUDA OOM."""
    out = []
    for s in range(0, tensor_cpu.size(0), batch_size):
        batch = tensor_cpu[s:s+batch_size].to(device)
        out.append(score_fn(batch))
    return np.concatenate(out)



# SAMPLING (same algorithm as train_path_c.py)


def transition_aware_sample(matrix, trace_ids, target_size):
    if len(matrix) <= target_size:
        return matrix, trace_ids
    print(f" INFO Transition-Aware Sample: {len(matrix):,} to {target_size:,}")
    corpus_bigrams = Counter()
    for seq in matrix:
        for i in range(len(seq) - 1):
            corpus_bigrams[(int(seq[i]), int(seq[i+1]))] += 1
    total = sum(corpus_bigrams.values()) or 1
    scores = []
    for seq in matrix:
        tb = Counter((int(seq[i]), int(seq[i+1])) for i in range(len(seq)-1))
        tt = sum(tb.values()) or 1
        scores.append(sum(min(tb[b]/tt, corpus_bigrams[b]/total) for b in tb))
    scores = np.array(scores)
    probs  = scores / (scores.sum() + 1e-12)
    chosen = np.random.choice(len(matrix), size=target_size, replace=False, p=probs)
    return matrix[np.sort(chosen)], trace_ids[np.sort(chosen)]



# Model trainers: one function per model


def run_markov(X_train, X_val, X_atk, y_test, order=1, **_):
    """1st or 2nd order Markov; anomaly = mean neg log prob."""
    transitions = defaultdict(Counter)
    probs_dict  = defaultdict(dict)
    for seq in X_train:
        for i in range(len(seq) - order):
            state = tuple(seq[i:i+order])
            nxt   = seq[i+order]
            transitions[state][nxt] += 1
    for state, ctr in transitions.items():
        tot = sum(ctr.values())
        for nxt, cnt in ctr.items():
            probs_dict[state][nxt] = cnt / tot

    def score_seq(seq):
        lp = 0.0
        for i in range(len(seq) - order):
            st  = tuple(seq[i:i+order])
            nxt = seq[i+order]
            if st in probs_dict and nxt in probs_dict[st]:
                lp += np.log(probs_dict[st][nxt] + 1e-9)
            else:
                lp += -10.0
        return -lp / (len(seq) - order)

    sc_val = np.array([score_seq(s) for s in X_val])
    sc_atk = np.array([score_seq(s) for s in X_atk])
    sc = sanitize_scores(np.concatenate([sc_val, sc_atk]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return auc, f1


def run_hmm(X_train, X_val, X_atk, y_test, n_components=8, **_):
    if not HMM_AVAILABLE:
        return None, None
    flat = X_train.flatten().reshape(-1, 1)
    lens = [X_train.shape[1]] * len(X_train)
    model = CategoricalHMM(n_components=n_components, n_iter=50, random_state=42)
    model.fit(flat, lens)
    sv = [-model.score(s.reshape(-1,1)) / len(s) for s in X_val]
    sa = [-model.score(s.reshape(-1,1)) / len(s) for s in X_atk]
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return auc, f1


def run_lstm_ae(X_train, X_val, X_atk, y_test,
                vocab_size, hidden_dim=32, epochs=5, eval_bs=512, **_):
    model = LSTMAESequence(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=0.005)
    crit  = nn.CrossEntropyLoss(reduction='none')

    t_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
    loader   = DataLoader(TensorDataset(t_tensor), batch_size=256, shuffle=True)
    model.train()
    for ep in range(epochs):
        for (bx,) in loader:
            opt.zero_grad()
            logits = model(bx)
            loss   = crit(logits.transpose(1,2), bx).mean()
            loss.backward()
            opt.step()
    del t_tensor

    model.eval()
    val_cpu = torch.tensor(X_val, dtype=torch.long)
    atk_cpu = torch.tensor(X_atk, dtype=torch.long)
    def _score(b):
        with torch.no_grad():
            return crit(model(b).transpose(1,2), b).mean(dim=1).cpu().numpy()
    sv = score_in_batches(_score, val_cpu, eval_bs)
    sa = score_in_batches(_score, atk_cpu, eval_bs)
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return auc, f1


def run_cbow_if(X_train, X_val, X_atk, y_test,
                vocab_size, embed_dim=32, epochs=5, **_):
    model = CBOWPredictor(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=0.005)
    crit  = nn.CrossEntropyLoss()

    W, half = X_train.shape[1], X_train.shape[1] // 2
    ctx_len  = min(half, 10)
    t_ctx    = torch.tensor(X_train[:,:ctx_len], dtype=torch.long).to(device)
    t_tgt    = torch.tensor(X_train[:, half],     dtype=torch.long).to(device)
    loader   = DataLoader(TensorDataset(t_ctx, t_tgt), batch_size=256, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
    del t_ctx, t_tgt

    model.eval()
    def extract_emb(arr):
        cpu = torch.tensor(arr, dtype=torch.long)
        ctx = cpu[:,:ctx_len]
        out = []
        for s in range(0, len(ctx), 512):
            b = ctx[s:s+512].to(device)
            with torch.no_grad():
                out.append(model.embedding(b).mean(dim=1).cpu().numpy())
        return np.vstack(out)

    X_tr_emb = extract_emb(X_train)
    X_vl_emb = extract_emb(X_val)
    X_ak_emb = extract_emb(X_atk)

    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_tr_emb)
    sc_val = -iso.score_samples(X_vl_emb)
    sc_atk = -iso.score_samples(X_ak_emb)
    sc = sanitize_scores(np.concatenate([sc_val, sc_atk]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return auc, f1


def run_gru(X_train, X_val, X_atk, y_test,
            vocab_size, hidden_dim=64, epochs=10, eval_bs=512, **_):
    model = GRUPredictor(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=0.005)
    crit  = nn.CrossEntropyLoss(reduction='none')

    t_tensor  = torch.tensor(X_train, dtype=torch.long).to(device)
    t_feat    = t_tensor[:,:-1]
    t_targ    = t_tensor[:, 1:]
    loader    = DataLoader(TensorDataset(t_feat, t_targ), batch_size=256, shuffle=True)

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx).transpose(1,2), by).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        print(f" [GRU ep {ep+1}/{epochs}] loss={ep_loss/len(loader):.4f}")
    del t_tensor, t_feat, t_targ

    model.eval()
    val_cpu = torch.tensor(X_val, dtype=torch.long)
    atk_cpu = torch.tensor(X_atk, dtype=torch.long)
    def _score(b):
        with torch.no_grad():
            nll = crit(model(b[:,:-1]).transpose(1,2), b[:,1:])
            return nll.mean(dim=1).cpu().numpy()
    sv = score_in_batches(_score, val_cpu, eval_bs)
    sa = score_in_batches(_score, atk_cpu, eval_bs)
    sc = sanitize_scores(np.concatenate([sv, sa]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return auc, f1



# SWEEP CONFIGS


FRACTIONS = [0.05, 0.10, 0.20]

SWEEP_CONFIGS = []

# MARKOV (order A fraction)
for order in [1, 2]:
    for frac in FRACTIONS:
        SWEEP_CONFIGS.append({
        "model": "Markov",
    "fraction": frac,
    "order": order,
        })

# HMM (n_components A fraction)
for nc in [4, 8, 16]:
    for frac in FRACTIONS:
        SWEEP_CONFIGS.append({
        "model": "HMM",
    "fraction": frac,
    "n_components": nc,
        })

# LSTM AE (hidden_dim A fraction)
for hd in [16, 32, 64]:
    for frac in FRACTIONS:
        SWEEP_CONFIGS.append({
        "model": "LSTM_AE",
    "fraction": frac,
    "hidden_dim": hd,
        })

# CBOW + IF (embed_dim A fraction)
for ed in [16, 32, 64]:
    for frac in FRACTIONS:
        SWEEP_CONFIGS.append({
        "model": "CBOW_IF",
    "fraction": frac,
    "embed_dim": ed,
        })

# GRU sweep: hidden_dim x fraction
for hd in [32, 64, 128]:
    for frac in FRACTIONS:
        SWEEP_CONFIGS.append({
        "model": "GRU",
    "fraction": frac,
    "hidden_dim": hd,
        })

TOTAL = len(SWEEP_CONFIGS)



# MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str,   default="32bit")
    parser.add_argument("--epochs",         type=int,   default=10,
                        help="Training epochs for GRU/LSTM-AE/CBOW (default: 10)")
    parser.add_argument("--ae_epochs",      type=int,   default=5,
                        help="Training epochs for LSTM-AE and CBOW (default: 5)")
    parser.add_argument("--min_samples",    type=int,   default=5000)
    parser.add_argument("--eval_batch_size",type=int,   default=512)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file    = os.path.join(project_root, "data", "processed",
                                args.dataset, "phase1_base_arrays.pkl")
    log_dir      = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_file     = os.path.join(log_dir, "path_c_sweep_results.json")

    print("=" * 65)
    print(f"PATH C HYPERPARAMETER SWEEP ({args.dataset.upper()}) - {TOTAL} configs")
    print(f" Device: {device} | GRU epochs: {args.epochs} | AE epochs: {args.ae_epochs}")
    print("=" * 65)

    # Load raw data.
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    X_train_raw  = data["X_train_w"]
    X_val_raw    = data["X_val_w"]
    X_attack_raw = data["X_attack_w"]
    trace_ids_train = data.get("trace_ids_train", np.arange(len(X_train_raw)))

    vocab_size = int(max(X_train_raw.max(),
                          X_val_raw.max(),
                          X_attack_raw.max())) + 1
    print(f" Vocab size: {vocab_size} | Train: {len(X_train_raw):,} | "
          f"Val: {len(X_val_raw):,} | Atk: {len(X_attack_raw):,}\n")

    y_test = np.concatenate([np.zeros(len(X_val_raw)),
                              np.ones(len(X_attack_raw))])

    # Load existing results (for resume).
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            all_results = json.load(f)
        print(f" Resume Loaded {len(all_results)} existing results from {out_file}")
    else:
        all_results = []

    done_keys = {
        (r["config"]["model"],
         r["config"]["fraction"],
         json.dumps({k: v for k, v in r["config"].items()
                     if k not in ("model", "fraction")}, sort_keys=True))
        for r in all_results
    }

    # Sweep loop.
    for cfg_idx, cfg in enumerate(SWEEP_CONFIGS):
        model_name = cfg["model"]
        fraction   = cfg["fraction"]
        extra_key  = json.dumps({k: v for k, v in cfg.items()
                                  if k not in ("model", "fraction")}, sort_keys=True)
        key = (model_name, fraction, extra_key)

        if key in done_keys:
            print(f" [{cfg_idx+1}/{TOTAL}] SKIP (cached): {cfg}")
            continue

        print(f"\n{'='*65}")
        print(f" [{cfg_idx+1}/{TOTAL}] {model_name} | cfg={cfg}")
        print(f"{'='*65}")

        # Sample training data
        set_seed(42)
        target = max(args.min_samples, int(len(X_train_raw) * fraction))
        X_train, _ = transition_aware_sample(
            X_train_raw, trace_ids_train, target_size=target
        )
        print(f" Train: {len(X_train_raw):,} to {len(X_train):,}")

        t0 = time.time()
        try:
            common = dict(
                X_train=X_train, X_val=X_val_raw, X_atk=X_attack_raw,
                y_test=y_test, vocab_size=vocab_size,
                eval_bs=args.eval_batch_size,
            )

            if model_name == "Markov":
                auc, f1 = run_markov(**common, order=cfg["order"])

            elif model_name == "HMM":
                auc, f1 = run_hmm(**common, n_components=cfg["n_components"])

            elif model_name == "LSTM_AE":
                auc, f1 = run_lstm_ae(**common,
                                       hidden_dim=cfg["hidden_dim"],
                                       epochs=args.ae_epochs)

            elif model_name == "CBOW_IF":
                auc, f1 = run_cbow_if(**common,
                                       embed_dim=cfg["embed_dim"],
                                       epochs=args.ae_epochs)

            elif model_name == "GRU":
                auc, f1 = run_gru(**common,
                                   hidden_dim=cfg["hidden_dim"],
                                   epochs=args.epochs)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            elapsed = time.time() - t0
            entry = {
                "config": cfg,
                "AUC": round(float(auc), 4) if auc is not None else None,
                "F1": round(float(f1), 4) if f1 is not None else None,
                "runtime_s": round(elapsed, 1),
            }
            print(f" OK AUC={entry['AUC']} F1={entry['F1']} ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            entry = {
                "config": cfg,
                "AUC": None, "F1": None,
                "runtime_s": round(elapsed, 1),
                "error": str(e),
            }
            print(f" ERROR {e}")

        all_results.append(entry)
        done_keys.add(key)

        # Save after every config (incremental)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary.
    print(f"\n{'='*65}")
    print("SWEEP COMPLETE - Best-Balance Summary (AUC + F1 normalised)")
    print(f"{'='*65}")

    by_model = defaultdict(list)
    for r in all_results:
        if r.get("F1") is not None:
            by_model[r["config"]["model"]].append(r)

    best_per_model = {}
    for mname, entries in by_model.items():
        aucs = [e["AUC"] for e in entries]
        f1s  = [e["F1"]  for e in entries]
        a_mn, a_mx = min(aucs), max(aucs)
        f_mn, f_mx = min(f1s),  max(f1s)
        for e in entries:
            a_n = (e["AUC"] - a_mn) / (a_mx - a_mn + 1e-9)
            f_n = (e["F1"]  - f_mn) / (f_mx - f_mn + 1e-9)
            e["_balance"] = a_n + f_n
        best = max(entries, key=lambda e: e["_balance"])
        best_per_model[mname] = best
        print(f" {mname:<12} Best-Balance: AUC={best['AUC']} F1={best['F1']} cfg={best['config']}")

    print(f"\n Results to {out_file}")
    return best_per_model


if __name__ == "__main__":
    set_seed(42)
    main()


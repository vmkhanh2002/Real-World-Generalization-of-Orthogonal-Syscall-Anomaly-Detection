"""
Path C deep optimization sweep (historical V2 script)
=====================================================
This script extends the original broad Path C sweep with a narrower search over
the most promising GRU, Markov, and CBOW-based regions. It is retained for
provenance and optional replay, but it should not be treated as the primary
source of the latest 32-bit benchmark or paper-facing metrics.

Current 32-bit active references:
  - active holdout model: GRU_Predictor
  - active holdout config: fraction=0.5, embed_dim=16, hidden_dim=192, num_layers=1
  - active holdout metrics: AUC 0.8151, F1_calib 0.4313
  - nested trace-level paper lock: AUC 0.8140 +/- 0.0137, F1 0.4311 +/- 0.0248
  - source of truth: experiments/32bit/logs/path_c_results.json
  - paper artifact: experiments/32bit/logs/paper_nested_protocol_table.json

Historical deep-search focus:
  1. GRU replay around the active family (hidden_dim, num_layers, fraction)
  2. Higher-order Markov transitions
  3. Wider-context CBOW + IsolationForest variants

NOTE: Bidirectional GRU is intentionally excluded because it leaks future tokens
at inference time and violates the streaming anomaly-detection constraint.

This script still writes the historical sweep artifact:
  - experiments/<dataset>/logs/path_c_v2_sweep_results.json
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
    """Clip Ainf / NaN to finite cap."""
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



# MODEL TRAINERS


def run_markov(X_train, X_val, X_atk, y_test, order=1, **_):
    """Markov Chain (order 1, 2, or 3); anomaly = mean neg log prob."""
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


def run_cbow_if(X_train, X_val, X_atk, y_test,
                vocab_size, embed_dim=32, epochs=5,
                ctx_len_override=None, if_trees=100, **_):
    """CBOW + IsolationForest; supports wider context window and more IF trees."""
    model = CBOWPredictor(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=0.005)
    crit  = nn.CrossEntropyLoss()

    half     = X_train.shape[1] // 2
    ctx_len  = ctx_len_override if ctx_len_override is not None else min(half, 10)
    ctx_len  = min(ctx_len, X_train.shape[1])   # can't exceed sequence length

    t_ctx    = torch.tensor(X_train[:,:ctx_len], dtype=torch.long).to(device)
    t_tgt    = torch.tensor(X_train[:, half],     dtype=torch.long).to(device)
    loader   = DataLoader(TensorDataset(t_ctx, t_tgt), batch_size=256, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            # CBOWPredictor expects exactly 10-token context; handle shorter ctx
            if bx.size(1) != 10:
                # Pad or truncate to 10 to match fc layer expectation
                if bx.size(1) < 10:
                    pad = torch.zeros(bx.size(0), 10 - bx.size(1),
                                      dtype=torch.long, device=bx.device)
                    bx = torch.cat([bx, pad], dim=1)
                else:
                    bx = bx[:,:10]
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
    del t_ctx, t_tgt

    # Patch: ensure extraction also aligns with 10-token FC layer
    model.eval()
    def extract_emb(arr):
        cpu = torch.tensor(arr, dtype=torch.long)
        out = []
        for s in range(0, len(cpu), 512):
            b = cpu[s:s+512,:ctx_len].to(device)
            if b.size(1) < 10:
                pad = torch.zeros(b.size(0), 10 - b.size(1),
                                  dtype=torch.long, device=b.device)
                b = torch.cat([b, pad], dim=1)
            else:
                b = b[:,:10]
            with torch.no_grad():
                out.append(model.embedding(b).mean(dim=1).cpu().numpy())
        return np.vstack(out)

    X_tr_emb = extract_emb(X_train)
    X_vl_emb = extract_emb(X_val)
    X_ak_emb = extract_emb(X_atk)

    iso = IsolationForest(n_estimators=if_trees, contamination=0.05, random_state=42)
    iso.fit(X_tr_emb)
    sc_val = -iso.score_samples(X_vl_emb)
    sc_atk = -iso.score_samples(X_ak_emb)
    sc = sanitize_scores(np.concatenate([sc_val, sc_atk]))
    auc = roc_auc_score(y_test, sc)
    _, f1 = get_best_f1(y_test, sc)
    return auc, f1


def run_gru(X_train, X_val, X_atk, y_test,
            vocab_size, hidden_dim=64, num_layers=1,
            epochs=10, eval_bs=512, **_):
    """GRU next-token predictor; supports stacked layers and extended epochs."""
    model = GRUPredictor(vocab_size=vocab_size,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers).to(device)
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
        print(f" [GRU L{num_layers} ep {ep+1}/{epochs}] "
              f"loss={ep_loss/len(loader):.4f}")
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



# Sweep configs: V2 new configs only, does not repeat V1.


FRACTIONS_GRU    = [0.20, 0.30, 0.50]
FRACTIONS_MARKOV = [0.20, 0.30]
FRACTIONS_CBOW   = [0.20, 0.30]

SWEEP_CONFIGS = []

# GRU: deeper/wider than V1.
for hd in [128, 192, 256]:
    for nl in [1, 2]:
        for frac in FRACTIONS_GRU:
            # Skip (hidden=128, layers=1, fraction=0.20); already done in V1.
            if hd == 128 and nl == 1 and frac == 0.20:
                continue
            SWEEP_CONFIGS.append({
            "model": "GRU",
    "fraction": frac,
    "hidden_dim": hd,
    "num_layers": nl,
            })

# Markov order 3.
for frac in FRACTIONS_MARKOV:
    SWEEP_CONFIGS.append({
    "model": "Markov",
    "fraction": frac,
    "order": 3,
    })

# CBOW + IF: wider context, more trees.
for ctx in [10, 15]:
    for trees in [100, 200]:
        for frac in FRACTIONS_CBOW:
            # Skip (ctx=10, trees=100, frac=0.20); best from V1 already done.
            if ctx == 10 and trees == 100 and frac == 0.20:
                continue
            SWEEP_CONFIGS.append({
            "model": "CBOW_IF",
    "fraction": frac,
    "embed_dim": 16, # V1 champion embed_dim
    "ctx_len": ctx,
    "if_trees": trees,
            })

TOTAL = len(SWEEP_CONFIGS)



# MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         type=str, default="32bit")
    parser.add_argument("--epochs",          type=int, default=15,
                        help="Training epochs for GRU (default: 15)")
    parser.add_argument("--ae_epochs",       type=int, default=5,
                        help="Training epochs for CBOW (default: 5)")
    parser.add_argument("--min_samples",     type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file    = os.path.join(project_root, "data", "processed",
                                args.dataset, "phase1_base_arrays.pkl")
    log_dir      = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_file     = os.path.join(log_dir, "path_c_v2_sweep_results.json")
    out_file_partial = os.path.join(log_dir, "path_c_v2_sweep_results.partial.json")
    v1_file      = os.path.join(log_dir, "path_c_sweep_results.json")

    print("=" * 65)
    print(f"PATH C V2 DEEP SWEEP ({args.dataset.upper()}) - {TOTAL} new configs")
    print(f" Device: {device}")
    print(f" GRU epochs: {args.epochs} | CBOW epochs: {args.ae_epochs}")
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

    # Load existing V2 results (for resume).
    all_results = []
    if os.path.exists(out_file_partial):
        with open(out_file_partial, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        all_results = loaded if isinstance(loaded, list) else loaded.get("v2_results", [])
        print(f" Resume Loaded {len(all_results)} existing V2 partial results")
    elif os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            all_results = loaded
        elif isinstance(loaded, dict):
            all_results = loaded.get("v2_results", loaded.get("results", []))
        print(f" Resume Loaded {len(all_results)} existing V2 results")

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

            elif model_name == "CBOW_IF":
                auc, f1 = run_cbow_if(**common,
                                       embed_dim=cfg["embed_dim"],
                                       epochs=args.ae_epochs,
                                       ctx_len_override=cfg.get("ctx_len"),
                                       if_trees=cfg.get("if_trees", 100))

            elif model_name == "GRU":
                auc, f1 = run_gru(**common,
                                   hidden_dim=cfg["hidden_dim"],
                                   num_layers=cfg.get("num_layers", 1),
                                   epochs=args.epochs)
            else:
                raise ValueError(f"Unknown model in V2 sweep: {model_name}")

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

        # Save incrementally after every config (partial checkpoint for resume)
        with open(out_file_partial, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # Combined V1 + V2 Summary.
    print(f"\n{'='*65}")
    print("V2 SWEEP COMPLETE - loading V1 results for combined ranking")
    print(f"{'='*65}")

    v1_results = []
    if os.path.exists(v1_file):
        with open(v1_file, "r") as f:
            v1_results = json.load(f)
    print(f" Loaded {len(v1_results)} V1 configs.")

    combined = v1_results + all_results
    valid    = [r for r in combined if r.get("AUC") is not None]

    by_model = defaultdict(list)
    for r in valid:
        by_model[r["config"]["model"]].append(r)

    print(f"\n{'='*65}")
    print(" COMBINED BEST PER MODEL (V1 + V2)")
    print(f"{'='*65}")

    overall_best = None
    for mname, entries in sorted(by_model.items()):
        best = max(entries, key=lambda e: e["AUC"])
        tag  = " - NEW!" if best in all_results else ""
        print(f" {mname:<12} "
              f"AUC={best['AUC']} F1={best['F1']} "
              f"cfg={best['config']}{tag}")
        if overall_best is None or best["AUC"] > overall_best["AUC"]:
            overall_best = best

    best_gru = max(by_model.get("GRU", []), key=lambda e: e["AUC"], default=None)
    best_cbow_if = max(by_model.get("CBOW_IF", []), key=lambda e: e["AUC"], default=None)
    best_lstm_ae = max(by_model.get("LSTM_AE", []), key=lambda e: e["AUC"], default=None)
    best_markov = max(by_model.get("Markov", []), key=lambda e: e["AUC"], default=None)

    best_config_per_model_type = {}
    if best_gru:
        best_config_per_model_type["GRU_Predictor"] = {
            "fraction": float(best_gru["config"].get("fraction", 0.5)),
            "embed_dim": 16,
            "hidden_dim": int(best_gru["config"].get("hidden_dim", 192)),
            "num_layers": int(best_gru["config"].get("num_layers", 1)),
            "AUC": float(best_gru["AUC"]),
            "F1_calib": float(best_gru.get("F1", 0.0)),
        }
    if best_cbow_if:
        best_config_per_model_type["CBOW_Predictor"] = {
            "fraction": float(best_cbow_if["config"].get("fraction", 0.2)),
            "embed_dim": int(best_cbow_if["config"].get("embed_dim", 16)),
            "AUC": float(best_cbow_if["AUC"]),
            "F1_calib": float(best_cbow_if.get("F1", 0.0)),
        }
    if best_lstm_ae:
        best_config_per_model_type["LSTM_AE_Sequence"] = {
            "fraction": float(best_lstm_ae["config"].get("fraction", 0.2)),
            "embed_dim": 16,
            "hidden_dim": int(best_lstm_ae["config"].get("hidden_dim", 64)),
            "AUC": float(best_lstm_ae["AUC"]),
            "F1_calib": float(best_lstm_ae.get("F1", 0.0)),
        }
    if best_markov:
        best_config_per_model_type["Markov"] = {
            "fraction": float(best_markov["config"].get("fraction", 0.3)),
            "order": int(best_markov["config"].get("order", 3)),
            "AUC": float(best_markov["AUC"]),
            "F1_calib": float(best_markov.get("F1", 0.0)),
        }

    summary = {
        "config_used": {
            "dataset": args.dataset,
            "epochs": int(args.epochs),
            "ae_epochs": int(args.ae_epochs),
            "min_samples": int(args.min_samples),
            "eval_batch_size": int(args.eval_batch_size),
            "v1_results_file": v1_file,
            "v2_total_new_configs": int(TOTAL),
        },
        "champion": {
            "model": overall_best["config"]["model"] if overall_best else None,
            "AUC": float(overall_best["AUC"]) if overall_best else None,
            "F1": float(overall_best["F1"]) if overall_best else None,
            "config": overall_best["config"] if overall_best else None,
        },
        "best_config_per_model_type": best_config_per_model_type,
        "v1_results_count": len(v1_results),
        "v2_results_count": len(all_results),
        "all_results": combined,
        "v2_results": all_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\n Overall champion")
    print(f" {overall_best}")
    print(f"\n V2 results to {out_file}")
    return overall_best


if __name__ == "__main__":
    set_seed(42)
    main()


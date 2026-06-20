"""
Path B optimization sweep (historical provenance script)
========================================================
Systematically sweeps key architectural hyperparameters across the Path B model
family. This script is preserved for provenance and optional replay; the current
32-bit benchmark and paper-facing references now live in the refreshed artifacts.

Current 32-bit active references:
  - holdout champion: CNN1D_AE, embed_dim=8, fraction=0.2
  - holdout metrics: AUC 0.7822, F1_calib 0.4599
  - nested trace-level paper lock: AUC 0.7813 +/- 0.0159, F1 0.4581 +/- 0.0355
  - source of truth: experiments/32bit/logs/path_b_topology_results.json
  - paper artifact: experiments/32bit/logs/paper_nested_protocol_table.json

This script still writes the historical sweep artifact:
  - experiments/<dataset>/logs/path_b_sweep_results.json

Sweep Axes:
  Axis 1 - Sampling Fraction: [0.05, 0.10, 0.20]
  Axis 2 - Conv1D Embed Dim: [8, 16, 32] (Conv1DAE only)
  Axis 3 - Dense AE Latent Dim: [8, 16, 32]
  Axis 4 - VAE Latent Dim: [8, 10, 16]
  Axis 5 - LSTM Hidden Dim: [8, 16, 32]
  Axis 6 - Deep SVDD Hidden Dim: [16, 32]

Strategy:
  - Conv1D replay remains the primary focus because it is the current active Path B model.
  - Other models are sampled for direct comparability within the same search space.
  - Training: 15 epochs, lr=0.001, batch_size=256.
  - Evaluation: AUC-ROC and Best-F1 on full (un-sampled) val+attack set.
"""

import os, pickle, sys, json, time, argparse, random, itertools
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline_config import get_path_b_defaults, normalize_dataset_name

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

set_seed(42)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.cluster import MiniBatchKMeans

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inline model definitions (no dependency on external models.py)
class DenseAE(nn.Module):
    def __init__(self, input_dim=20, latent_dim=16):
        super().__init__()
        mid = max(latent_dim * 2, 32)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(),
            nn.Linear(mid, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid), nn.ReLU(),
            nn.Linear(mid, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMAE(nn.Module):
    def __init__(self, seq_len=20, hidden_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.out     = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        h = hidden[-1].unsqueeze(1)
        dummy = torch.zeros(x.size(0), self.seq_len, 1, device=x.device)
        decoded, _ = self.decoder(
            dummy,
            (h.transpose(0, 1).contiguous(), torch.zeros_like(h.transpose(0, 1).contiguous()))
        )
        return self.sigmoid(self.out(decoded))


class VAE(nn.Module):
    def __init__(self, input_dim=20, latent_dim=10):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, 32)
        self.fc_mu  = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.fc3  = nn.Linear(latent_dim, 32)
        self.fc4  = nn.Linear(32, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.fc4(torch.relu(self.fc3(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Conv1DAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        ch = max(embed_dim, 8)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, ch, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, max(ch // 2, 4), kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(max(ch // 2, 4), ch, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose1d(ch, embed_dim, kernel_size=2, stride=2)
        )

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)   # (B, embed_dim, 20)
        enc = self.encoder(emb)
        dec = self.decoder(enc)
        return dec, emb


class DeepSVDD(nn.Module):
    def __init__(self, input_dim=20, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, hidden)
        )
    def forward(self, x):
        return self.net(x)


# Helpers
def get_best_f1(y_true, scores):
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-8)
    idx = np.argmax(f1)
    return thr[min(idx, len(thr) - 1)], float(f1[idx])


def tfidf_coreset_sample(matrix, ids, fraction=0.10, max_features=500, min_samples=500):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances_argmin

    target = min(len(matrix), max(min_samples, int(len(matrix) * fraction)))
    if len(matrix) <= target:
        return matrix, ids

    strings = [" ".join(map(str, row)) for row in matrix]
    tfidf   = TfidfVectorizer(max_features=max_features, analyzer='word')
    X_emb   = tfidf.fit_transform(strings).toarray()
    km      = MiniBatchKMeans(n_clusters=target, random_state=42, batch_size=4096, n_init='auto')
    km.fit(X_emb)
    idx     = pairwise_distances_argmin(km.cluster_centers_, X_emb)
    return matrix[idx], ids[idx]


def evaluate_ae(model, val_tensor, atk_tensor, mode='dense'):
    """Returns concatenated anomaly scores for val+attack."""
    model.eval()
    with torch.no_grad():
        if mode == 'lstm':
            v, a = val_tensor.unsqueeze(2), atk_tensor.unsqueeze(2)
            sv = torch.mean(torch.squeeze((v - model(v))**2), dim=1).cpu().numpy()
            sa = torch.mean(torch.squeeze((a - model(a))**2), dim=1).cpu().numpy()
        elif mode == 'vae':
            vp, _, _ = model(val_tensor)
            ap, _, _ = model(atk_tensor)
            sv = torch.mean((val_tensor - vp)**2, dim=1).cpu().numpy()
            sa = torch.mean((atk_tensor - ap)**2, dim=1).cpu().numpy()
        else:  # dense
            sv = torch.mean((val_tensor - model(val_tensor))**2, dim=1).cpu().numpy()
            sa = torch.mean((atk_tensor - model(atk_tensor))**2, dim=1).cpu().numpy()
    return np.concatenate([sv, sa])


def evaluate_cnn(model, val_int, atk_int):
    model.eval()
    with torch.no_grad():
        vp, ve = model(val_int)
        ap, ae = model(atk_int)
        sv = torch.mean((ve - vp)**2, dim=[1, 2]).cpu().numpy()
        sa = torch.mean((ae - ap)**2, dim=[1, 2]).cpu().numpy()
    return np.concatenate([sv, sa])


def evaluate_svdd(model, val_tensor, atk_tensor, c):
    model.eval()
    with torch.no_grad():
        sv = torch.sum((model(val_tensor) - c)**2, dim=1).cpu().numpy()
        sa = torch.sum((model(atk_tensor) - c)**2, dim=1).cpu().numpy()
    return np.concatenate([sv, sa])


def train_generic(model, loader, optimizer, criterion, epochs, mode='dense'):
    model.train()
    for _ in range(epochs):
        for batch, _ in loader:
            optimizer.zero_grad()
            if mode == 'lstm':
                b = batch.unsqueeze(2)
                loss = criterion(model(b), b)
            elif mode == 'vae':
                recon, mu, logvar = model(batch)
                mse = nn.functional.mse_loss(recon, batch, reduction='sum')
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse + kld
            else:
                loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()


def train_cnn(model, loader, optimizer, criterion, epochs):
    model.train()
    for _ in range(epochs):
        for batch, _ in loader:
            optimizer.zero_grad()
            out, emb = model(batch)
            loss = criterion(out, emb)
            loss.backward()
            optimizer.step()


def train_svdd(model, loader, optimizer, c, epochs):
    model.train()
    for _ in range(epochs):
        for batch, _ in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.mean(torch.sum((out - c)**2, dim=1))
            loss.backward()
            optimizer.step()


# Main sweep
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str,   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--min_samples", type=int,   default=None)
    parser.add_argument("--base_fraction", type=float, default=None,
                        help="Fraction used for non-CNN sweeps (defaults to pipeline_config fraction)")
    parser.add_argument("--skip_svdd",   action="store_true",
                        help="Skip Deep SVDD (slow)  focus on AE family")
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    defaults = get_path_b_defaults(args.dataset)
    if args.epochs is None:
        args.epochs = int(defaults.get("epochs", 15))
    if args.min_samples is None:
        args.min_samples = int(defaults.get("min_samples", 5000))
    if args.base_fraction is None:
        args.base_fraction = float(defaults.get("fraction", 0.10))

    # project_root is 2 levels up from scripts/optimization/ repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir   = os.path.join(project_root, "experiments", args.dataset, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print(f"PATH B HYPERPARAMETER SWEEP | Dataset: {args.dataset.upper()} | Device: {device}")
    print("=" * 70)

    # Load data
    print(f"\n1. Loading {data_file} ...")
    with open(data_file, "rb") as fh:
        data = pickle.load(fh)

    X_train_raw  = data["X_train_w"]
    X_val_raw    = data["X_val_w"]
    X_attack_raw = data["X_attack_w"]
    trace_ids_tr = data.get("trace_ids_train", np.arange(len(X_train_raw)))

    MAX_ID = max(X_train_raw.max(), X_val_raw.max(), X_attack_raw.max())
    if MAX_ID == 0: MAX_ID = 1
    vocab_size = int(MAX_ID + 1)

    y_test = np.concatenate([np.zeros(len(X_val_raw)), np.ones(len(X_attack_raw))])

    # Full val/attack tensors (not sampled)
    val_norm  = (X_val_raw  / MAX_ID).astype(np.float32)
    atk_norm  = (X_attack_raw / MAX_ID).astype(np.float32)
    val_t  = torch.tensor(val_norm).to(device)
    atk_t  = torch.tensor(atk_norm).to(device)
    val_i  = torch.tensor(X_val_raw,    dtype=torch.long).to(device)
    atk_i  = torch.tensor(X_attack_raw, dtype=torch.long).to(device)

    print(f" Val: {X_val_raw.shape} | Attack: {X_attack_raw.shape}")
    print(f" Train raw: {X_train_raw.shape} | vocab_size: {vocab_size}")

    all_results = {}
    criterion   = nn.MSELoss()

    # SWEEP 1: Conv1D AE embed_dim x fraction
    print("\n" + "" * 70)
    print("SWEEP 1/4 Conv1D AE (Embedding): embed_dim fraction")
    print("" * 70)

    EMBED_DIMS  = [8, 16, 32]
    FRACTIONS   = [0.05, 0.10, 0.20]
    cnn_results = []

    for embed_dim, fraction in itertools.product(EMBED_DIMS, FRACTIONS):
        tag = f"CNN1D_emb{embed_dim}_frac{fraction}"
        t0  = time.time()

        # Sample training set
        X_tr, _ = tfidf_coreset_sample(X_train_raw, trace_ids_tr,
                                        fraction=fraction,
                                        min_samples=args.min_samples)
        tr_int  = torch.tensor(X_tr, dtype=torch.long).to(device)
        loader  = DataLoader(TensorDataset(tr_int, tr_int), batch_size=256, shuffle=True)

        model  = Conv1DAE(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
        opt    = optim.Adam(model.parameters(), lr=0.001)
        train_cnn(model, loader, opt, criterion, args.epochs)

        scores = evaluate_cnn(model, val_i, atk_i)
        auc    = roc_auc_score(y_test, scores)
        _, f1  = get_best_f1(y_test, scores)
        elapsed = time.time() - t0

        row = {"model": "CNN1D_AE", "embed_dim": embed_dim, "fraction": fraction,
               "AUC": round(auc, 4), "F1_calib": round(f1, 4), "F1": round(f1, 4), "train_samples": len(X_tr),
               "time_s": round(elapsed, 1)}
        cnn_results.append(row)
        all_results[tag] = row
        print(f" {tag:<35} AUC={auc:.4f} F1={f1:.4f} n={len(X_tr):,} ({elapsed:.0f}s)")

    # Best CNN config
    best_cnn = max(cnn_results, key=lambda r: r["AUC"])
    print(f"\n Best CNN1D_AE: embed_dim={best_cnn['embed_dim']} fraction={best_cnn['fraction']}"
          f" AUC={best_cnn['AUC']} F1={best_cnn['F1']}")

    # SWEEP 2: Dense AE latent_dim (baseline fraction=0.10)
    print("\n" + "" * 70)
    print("SWEEP 2/4 Dense AE: latent_dim")
    print("" * 70)

    BASE_FRACTION = float(args.base_fraction)
    X_tr_base, _ = tfidf_coreset_sample(X_train_raw, trace_ids_tr,
                                         fraction=BASE_FRACTION,
                                         min_samples=args.min_samples)
    tr_norm = (X_tr_base / MAX_ID).astype(np.float32)
    tr_t    = torch.tensor(tr_norm).to(device)
    loader_norm = DataLoader(TensorDataset(tr_t, tr_t), batch_size=256, shuffle=True)

    dense_results = []
    for latent_dim in [8, 16, 32]:
        tag = f"Dense_AE_latent{latent_dim}"
        t0  = time.time()
        model = DenseAE(input_dim=20, latent_dim=latent_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=0.001)
        train_generic(model, loader_norm, opt, criterion, args.epochs, 'dense')
        scores = evaluate_ae(model, val_t, atk_t, 'dense')
        auc = roc_auc_score(y_test, scores)
        _, f1 = get_best_f1(y_test, scores)
        elapsed = time.time() - t0
        row = {"model": "Dense_AE", "latent_dim": latent_dim, "fraction": BASE_FRACTION,
        "AUC": round(auc, 4), "F1_calib": round(f1, 4), "F1": round(f1, 4), "time_s": round(elapsed, 1)}
        dense_results.append(row)
        all_results[tag] = row
        print(f" {tag:<35} AUC={auc:.4f} F1={f1:.4f} ({elapsed:.0f}s)")

    best_dense = max(dense_results, key=lambda r: r["AUC"])
    print(f"\n Best Dense AE: latent_dim={best_dense['latent_dim']} AUC={best_dense['AUC']} F1={best_dense['F1']}")

    # SWEEP 3: VAE latent_dim (baseline fraction=0.10)
    print("\n" + "" * 70)
    print("SWEEP 3/4 VAE: latent_dim")
    print("" * 70)

    vae_results = []
    for latent_dim in [8, 10, 16]:
        tag = f"VAE_latent{latent_dim}"
        t0  = time.time()
        model = VAE(input_dim=20, latent_dim=latent_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=0.001)
        train_generic(model, loader_norm, opt, criterion, args.epochs, 'vae')
        scores = evaluate_ae(model, val_t, atk_t, 'vae')
        auc = roc_auc_score(y_test, scores)
        _, f1 = get_best_f1(y_test, scores)
        elapsed = time.time() - t0
        row = {"model": "VAE", "latent_dim": latent_dim, "fraction": BASE_FRACTION,
        "AUC": round(auc, 4), "F1_calib": round(f1, 4), "F1": round(f1, 4), "time_s": round(elapsed, 1)}
        vae_results.append(row)
        all_results[tag] = row
        print(f" {tag:<35} AUC={auc:.4f} F1={f1:.4f} ({elapsed:.0f}s)")

    best_vae = max(vae_results, key=lambda r: r["AUC"])
    print(f"\n Best VAE: latent_dim={best_vae['latent_dim']} AUC={best_vae['AUC']} F1={best_vae['F1']}")

    # SWEEP 4: LSTM AE hidden_dim (baseline fraction=0.10)
    print("\n" + "" * 70)
    print("SWEEP 4/4 LSTM AE: hidden_dim")
    print("" * 70)

    lstm_results = []
    for hidden_dim in [8, 16, 32]:
        tag = f"LSTM_AE_hid{hidden_dim}"
        t0  = time.time()
        model = LSTMAE(seq_len=20, hidden_dim=hidden_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=0.001)
        train_generic(model, loader_norm, opt, criterion, args.epochs, 'lstm')
        scores = evaluate_ae(model, val_t, atk_t, 'lstm')
        auc = roc_auc_score(y_test, scores)
        _, f1 = get_best_f1(y_test, scores)
        elapsed = time.time() - t0
        row = {"model": "LSTM_AE", "hidden_dim": hidden_dim, "fraction": BASE_FRACTION,
        "AUC": round(auc, 4), "F1_calib": round(f1, 4), "F1": round(f1, 4), "time_s": round(elapsed, 1)}
        lstm_results.append(row)
        all_results[tag] = row
        print(f" {tag:<35} AUC={auc:.4f} F1={f1:.4f} ({elapsed:.0f}s)")

    best_lstm = max(lstm_results, key=lambda r: r["AUC"])
    print(f"\n Best LSTM AE: hidden_dim={best_lstm['hidden_dim']} AUC={best_lstm['AUC']} F1={best_lstm['F1']}")

    # OPTIONAL: Deep SVDD sweep (fast, hidden_dim only)
    if not args.skip_svdd:
        print("\n" + "" * 70)
        print("BONUS Deep SVDD: hidden_dim")
        print("" * 70)
        svdd_results = []
        for hidden in [16, 32]:
            tag = f"DeepSVDD_hid{hidden}"
            t0  = time.time()
            model = DeepSVDD(input_dim=20, hidden=hidden).to(device)
            opt   = optim.Adam(model.parameters(), lr=0.001)
            model.eval()
            with torch.no_grad():
                c = torch.mean(model(tr_t[:1000]), dim=0, keepdim=True)
            train_svdd(model, loader_norm, opt, c, args.epochs)
            scores = evaluate_svdd(model, val_t, atk_t, c)
            auc = roc_auc_score(y_test, scores)
            _, f1 = get_best_f1(y_test, scores)
            elapsed = time.time() - t0
            row = {"model": "DeepSVDD", "hidden": hidden, "fraction": BASE_FRACTION,
            "AUC": round(auc, 4), "F1_calib": round(f1, 4), "F1": round(f1, 4), "time_s": round(elapsed, 1)}
            svdd_results.append(row)
            all_results[tag] = row
            print(f" {tag:<35} AUC={auc:.4f} F1={f1:.4f} ({elapsed:.0f}s)")
        best_svdd = max(svdd_results, key=lambda r: r["AUC"])
        print(f"\n Best SVDD: hidden={best_svdd['hidden']} AUC={best_svdd['AUC']} F1={best_svdd['F1']}")
    else:
        svdd_results = []
        best_svdd    = None

    # SUMMARY REPORT
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE SUMMARY")
    print("=" * 70)

    # Refreshed 32 bit holdout references from path_b_topology_results.json.
    baseline = {"CNN1D_AE": 0.7822, "Dense_AE": 0.6044, "LSTM_AE": 0.5397,
    "VAE": 0.6303, "DeepSVDD": 0.4257}

    summary = {
    "config_used": {
    "dataset": args.dataset,
    "epochs": int(args.epochs),
    "min_samples": int(args.min_samples),
    "base_fraction_non_cnn": float(BASE_FRACTION),
    "cnn_fraction_sweep": [0.05, 0.10, 0.20],
    "defaults_from_pipeline_config": True,
        },
        "Conv1D_AE_best": best_cnn,
    "Dense_AE_best": best_dense,
    "LSTM_AE_best": best_lstm,
    "VAE_best": best_vae,
    "DeepSVDD_best": best_svdd,
    "all_configs": all_results,
    "baseline": baseline,
    }

    # Find overall champion
    candidates = [
        ("CNN1D_AE",  best_cnn["AUC"], best_cnn["F1"],
        f"embed_dim={best_cnn['embed_dim']}, fraction={best_cnn['fraction']}"),
        ("Dense_AE",  best_dense["AUC"], best_dense["F1"],
        f"latent_dim={best_dense['latent_dim']}"),
        ("LSTM_AE",   best_lstm["AUC"],  best_lstm["F1"],
        f"hidden_dim={best_lstm['hidden_dim']}"),
        ("VAE",       best_vae["AUC"],   best_vae["F1"],
        f"latent_dim={best_vae['latent_dim']}"),
    ]
    if best_svdd:
        candidates.append(
            ("DeepSVDD", best_svdd["AUC"], best_svdd["F1"],
            f"hidden={best_svdd['hidden']}")
        )

    champion = max(candidates, key=lambda x: x[1])
    summary["champion"] = {
    "model": champion[0], "AUC": champion[1],
    "F1": champion[2], "config": champion[3]
    }
    summary["best_config_per_model_type"] = {
    "CNN1D_AE": {
    "embed_dim": int(best_cnn["embed_dim"]),
    "fraction": float(best_cnn["fraction"]),
    "AUC": float(best_cnn["AUC"]),
    "F1_calib": float(best_cnn.get("F1_calib", best_cnn.get("F1", 0.0))),
        },
        "Dense_AE": {
    "latent_dim": int(best_dense["latent_dim"]),
    "fraction": float(BASE_FRACTION),
    "AUC": float(best_dense["AUC"]),
    "F1_calib": float(best_dense.get("F1_calib", best_dense.get("F1", 0.0))),
        },
        "LSTM_AE": {
    "hidden_dim": int(best_lstm["hidden_dim"]),
    "fraction": float(BASE_FRACTION),
    "AUC": float(best_lstm["AUC"]),
    "F1_calib": float(best_lstm.get("F1_calib", best_lstm.get("F1", 0.0))),
        },
        "VAE": {
    "latent_dim": int(best_vae["latent_dim"]),
    "fraction": float(BASE_FRACTION),
    "AUC": float(best_vae["AUC"]),
    "F1_calib": float(best_vae.get("F1_calib", best_vae.get("F1", 0.0))),
        },
    }
    if best_svdd is not None:
        summary["best_config_per_model_type"]["DeepSVDD"] = {
        "hidden": int(best_svdd["hidden"]),
    "fraction": float(BASE_FRACTION),
    "AUC": float(best_svdd["AUC"]),
    "F1_calib": float(best_svdd.get("F1_calib", best_svdd.get("F1", 0.0))),
        }

        print(f"\n {'Model':<20} {'Baseline AUC':>13} {'Best AUC':>10} {'Delta':>7} {'Best Config'}")
    print(f" {'-'*75}")
    for name, auc, f1, cfg in candidates:
        bsl = baseline.get(name, None)
        delta_str = f"{auc - bsl:+.4f}" if bsl else "N/A"
        print(f" {name:<20} {str(bsl) if bsl else 'N/A':>13} {auc:>10.4f} {delta_str:>7} {cfg}")

    print(f"\n Path B champion: {champion[0]} AUC={champion[1]:.4f} F1={champion[2]:.4f} config={champion[3]}")

    # Save results
    out_file = os.path.join(log_dir, "path_b_sweep_results.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4, default=str)
        print(f"\nINFO Full sweep log saved {out_file}")


if __name__ == "__main__":
    main()

"""
Measures CPU inference latency (ms/window) for the selected architecture.
Stage 1: selected Path A, Path B, Path C from the current Phase 5 manifest
Stage 2: selected Phase 5 meta-classifier

Provides reproducible benchmarking for Table 4 (Real-Time CPU Inference Latency Breakdown).
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
import warnings
import sys

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

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

warnings.filterwarnings('ignore')
torch.set_num_threads(1) # Ensure single thread CPU benchmarking for consistency

from benchmark_common import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_phase5_meta_payload,
    load_path_b_model,
    load_path_c_model,
    load_selected_path_a_bundle,
    resolve_phase5_selection,
)

# Model classes imported from models.py (Conv1DAE, GRUPredictor)

def stringify_windows(arr):
    return [" ".join(map(str, row)) for row in arr]

def measure_latency():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    data_path = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset)

    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, 'reviewer_benchmark_latency.py.out.txt'))
    print(f"INFO Loading Phase 1 Data arrays from {data_path}...")
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Use 1000 validation windows for latency testing
    X_val_w = data["X_val_w"][:1000]
    X_val_w_str = stringify_windows(X_val_w)
    X_val_tensor = torch.tensor(X_val_w, dtype=torch.long)
    N = len(X_val_w)
    selection = resolve_phase5_selection(args.dataset)

    print("INFO Loading pre-trained models for inference latency...")

    # Path A
    path_a_bundle = load_selected_path_a_bundle(args.dataset, data["X_train_w"], X_val_w, X_val_w)
    vectorizer = path_a_bundle["vectorizer"]
    reducer = path_a_bundle["reducer"]
    path_a_model = path_a_bundle["model"]
    path_a_model_type = path_a_bundle["model_type"]

    # Path B & Path C
    cnn, _ = load_path_b_model(args.dataset, torch.device("cpu"))
    gru, _ = load_path_c_model(args.dataset, torch.device("cpu"))

    # Stage 2: load the actual selected Phase 5 meta-classifier
    phase5_payload = load_phase5_meta_payload(args.dataset)
    meta_clf = phase5_payload["model"]
    meta_label = phase5_payload.get("model_name", "Phase5_Meta")

    print("\nINFO Benchmarking Inference Latency (Single Thread CPU)...")

    # --- PATH A ---
    t0 = time.perf_counter()
    x_val_tfidf = vectorizer.transform(X_val_w_str)

    if path_a_model_type == "PCA-Error":
        x_val_features = x_val_tfidf.toarray()
        recon = path_a_model.inverse_transform(path_a_model.transform(x_val_features))
        s_a = np.mean((x_val_features - recon) ** 2, axis=1)
    else:
        if reducer is None:
            x_val_features = x_val_tfidf.toarray()
        elif reducer.__class__.__name__ == "TruncatedSVD":
            x_val_features = reducer.transform(x_val_tfidf)
        else:
            x_val_features = reducer.transform(x_val_tfidf.toarray())

        if path_a_model_type in {"SGD-OCSVM", "LOF"}:
            s_a = -path_a_model.decision_function(x_val_features)
        elif path_a_model_type == "IsolationForest":
            s_a = -path_a_model.score_samples(x_val_features)
        elif path_a_model_type == "HBOS":
            s_a = path_a_model.decision_function(x_val_features)
        else:
            raise ValueError(f"Unsupported Path A model type for latency benchmark: {path_a_model_type}")
    t1 = time.perf_counter()
    path_a_ms = ((t1 - t0) / N) * 1000

    # --- PATH B ---
    t0 = time.perf_counter()
    with torch.no_grad():
        out_b, emb_b = cnn(X_val_tensor)
        s_b = torch.mean((out_b - emb_b)**2, dim=[1,2]).numpy()
    t1 = time.perf_counter()
    path_b_ms = ((t1 - t0) / N) * 1000

    # --- PATH C ---
    t0 = time.perf_counter()
    with torch.no_grad():
        preds_c = gru(X_val_tensor[:, :-1])
        target = X_val_tensor[:, 1:]
        ce = nn.CrossEntropyLoss(reduction='none')(preds_c.transpose(1, 2), target)
        s_c = ce.mean(dim=1).numpy()
    t1 = time.perf_counter()
    path_c_ms = ((t1 - t0) / N) * 1000

    stage1_total_ms = path_a_ms + path_b_ms + path_c_ms

    # --- STAGE 2 (XGBoost) ---
    Z_val = np.column_stack([s_a, s_b, s_c])
    t0 = time.perf_counter()
    preds = meta_clf.predict_proba(Z_val)[:, 1]
    t1 = time.perf_counter()
    stage2_ms = ((t1 - t0) / N) * 1000

    total_ms = stage1_total_ms + stage2_ms

    print("==============================================")
    print(" INFERENCE LATENCY REPORT ")
    print("==============================================")
    print(f"Path A (TFIDF+Reducer+{selection['path_a_label']}): {path_a_ms:.4f} ms / window")
    print(f"Path B ({selection['path_b_label']}): {path_b_ms:.4f} ms / window")
    print(f"Path C ({selection['path_c_label']}): {path_c_ms:.4f} ms / window")
    print("----------------------------------------------")
    print(f"Stage 1 Total: {stage1_total_ms:.4f} ms / window")
    print(f"Stage 2 ({meta_label}): {stage2_ms:.4f} ms / window")
    print("----------------------------------------------")
    print(f"Grand Total Latency: {total_ms:.4f} ms / window")
    print("==============================================")

if __name__ == "__main__":
    measure_latency()

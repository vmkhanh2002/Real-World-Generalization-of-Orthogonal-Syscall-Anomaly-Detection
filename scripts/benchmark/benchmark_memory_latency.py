"""
Benchmark: Memory footprint + latency (SHOULD FIX #8).

UNIFIED: windows drawn from the shared pool. Reports BOTH:
  - single-window latency (batch=1, p50/p95/p99) — honest streaming latency
  - batched throughput (per-window amortized) — reconciles the 0.2217 ms headline
Plus disk footprint and RAM delta. Inference only.

Output:
    experiments/32bit/logs/unified/detailed_latency_memory.json
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unified"))

from benchmark_common import (  # noqa: E402
    PROJECT_ROOT, load_selected_path_a_bundle, load_path_b_model,
    load_path_c_model, load_phase5_meta_payload,
)
from pool_io import load_pool, save_json  # noqa: E402


def size_mb(p):
    return round(os.path.getsize(p) / 1048576, 4) if os.path.exists(p) else None


def lat_stats(call, warmup=100, n=3000):
    for _ in range(warmup):
        call()
    t = np.empty(n)
    for i in range(n):
        a = time.perf_counter(); call(); t[i] = (time.perf_counter() - a) * 1000
    return {"mean_ms": round(float(t.mean()), 6), "p50_ms": round(float(np.percentile(t, 50)), 6),
            "p95_ms": round(float(np.percentile(t, 95)), 6), "p99_ms": round(float(np.percentile(t, 99)), 6),
            "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="32bit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    pool = load_pool(args.dataset)
    X = pool["X_eval"]
    proj = str(PROJECT_ROOT)
    mdl = os.path.join(proj, "models", args.dataset)

    # RAM baseline
    import psutil
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1048576

    bundle = load_selected_path_a_bundle(args.dataset, None, X[:10], X[:10])
    cnn, _ = load_path_b_model(args.dataset, device)
    gru, _ = load_path_c_model(args.dataset, device)
    meta = load_phase5_meta_payload(args.dataset)["model"]
    ram1 = proc.memory_info().rss / 1048576

    vec, red, ma, mtype = bundle["vectorizer"], bundle["reducer"], bundle["model"], bundle["model_type"]
    one = X[0:1]
    one_t = torch.tensor(one, dtype=torch.long, device=device)
    crit = nn.CrossEntropyLoss(reduction="none")

    def call_a():
        s = vec.transform([" ".join(map(str, one[0]))])
        if red is not None:
            s = red.transform(s.toarray() if hasattr(s, "toarray") else s)
        return ma.decision_function(s)

    def call_b():
        with torch.no_grad():
            pred, emb = cnn(one_t); return torch.mean((emb - pred) ** 2).item()

    def call_c():
        with torch.no_grad():
            return crit(gru(one_t[:, :-1]).transpose(1, 2), one_t[:, 1:]).mean().item()

    z1 = np.array([[0.5, 0.5, 0.5]])
    def call_5():
        return meta.predict_proba(z1)

    single = {"path_a": lat_stats(call_a), "path_b": lat_stats(call_b),
              "path_c": lat_stats(call_c), "phase5": lat_stats(call_5)}
    single_total = round(sum(v["mean_ms"] for v in single.values()), 4)

    # batched throughput (amortized per-window over the pool)
    B = 1024
    Xb = X[:20480]
    Xt = torch.tensor(Xb, dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(Xt), B):
            cnn(Xt[i:i + B])
    tb_b = (time.perf_counter() - t0) / len(Xb) * 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(Xt), B):
            gru(Xt[i:i + B, :-1])
    tb_c = (time.perf_counter() - t0) / len(Xb) * 1000

    disk = {
        "path_a_vectorizer": size_mb(os.path.join(mdl, "path_a", "vec.pkl")),
        "path_a_pca": size_mb(os.path.join(mdl, "path_a", "pca100.pkl")),
        "path_b_model": size_mb(os.path.join(mdl, "path_b", "cnn.pth")),
        "path_c_model": size_mb(os.path.join(mdl, "path_c", "gru.pth")),
        "phase5_meta": size_mb(os.path.join(mdl, "phase5_fusion", "meta_xgboost.pkl")),
    }
    disk["total_mb"] = round(sum(v for v in disk.values() if v), 4)

    out = {
        "pool": "unified_eval_pool", "device": "cpu_single_thread",
        "single_window_latency_ms": single,
        "single_window_total_mean_ms": single_total,
        "batched_throughput_ms_per_window": {
            "path_b_batch1024": round(tb_b, 6), "path_c_batch1024": round(tb_c, 6),
            "note": "amortized per-window in batch mode; reconciles the batched 0.2217 ms headline",
        },
        "disk_footprint_mb": disk,
        "ram_delta_mb": round(ram1 - ram0, 2),
        "methodology": {"cpu": "AMD Ryzen 9 6900HX", "threads": 1, "warmup": 100, "n_measure": 3000,
                        "note": "single-window = honest streaming latency; batched = throughput."},
    }
    path = save_json(out, args.dataset, "detailed_latency_memory.json")
    print(f"Single-window total mean: {single_total} ms | "
          f"batched B/C per-window: {tb_b:.4f}/{tb_c:.4f} ms | disk {disk['total_mb']} MB")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

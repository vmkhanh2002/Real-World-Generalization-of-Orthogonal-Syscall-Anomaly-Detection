"""
Reviewer Benchmark: Feature Orthogonality Analysis
=================================================
Quantitative validation that Path A, Path B, and Path C capture distinct
failure modes (Geometric, Topological, and Temporal).

This script identifies 'Hard Anomalies' that bypass one path but are
caught by others, proving the necessity of the multi-layered fusion.
"""

import os, pickle, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark_common import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    load_path_b_model,
    load_path_c_model,
    load_selected_path_a_bundle,
    resolve_phase5_selection,
    score_path_a_bundle,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    data_file = os.path.join(project_root, "data", "processed", args.dataset, "phase1_base_arrays.pkl")
    model_dir = os.path.join(project_root, "models", args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nWARNING Reviewer Benchmark: Orthogonality Analysis ({args.dataset.upper()})")

    # 1. Load Data
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    X_val = data["X_val_w"]
    X_atk = data["X_attack_w"]
    selection = resolve_phase5_selection(args.dataset)

    # 2. Path A: currently selected vector/geometric path
    print(f" INFO Computing Path A scores ({selection['path_a_label']})...")
    path_a_bundle = load_selected_path_a_bundle(args.dataset, data["X_train_w"], X_val, X_atk)

    # Batch size for evaluation
    BATCH_SIZE = 512

    # Initialize models
    # Path B
    cnn, _ = load_path_b_model(args.dataset, device)

    # Path C
    gru, _ = load_path_c_model(args.dataset, device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    def get_cnn_scores(X):
        all_scores = []
        for i in range(0, len(X), BATCH_SIZE):
            batch = torch.LongTensor(X[i:i+BATCH_SIZE]).to(device)
            with torch.no_grad():
                dec, tgt = cnn(batch)
                mse = torch.mean((dec - tgt)**2, dim=[1,2])
                all_scores.append(mse.cpu().numpy())
        return np.concatenate(all_scores)

    def get_gru_scores(X):
        all_scores = []
        for i in range(0, len(X), BATCH_SIZE):
            batch = torch.LongTensor(X[i:i+BATCH_SIZE]).to(device)
            with torch.no_grad():
                bx = batch[:, :-1]
                by = batch[:, 1:]
                preds = gru(bx).transpose(1, 2)
                nll = criterion(preds, by).mean(dim=1)
                all_scores.append(nll.cpu().numpy())
        return np.concatenate(all_scores)

    scores_a_val, scores_a_atk = score_path_a_bundle(path_a_bundle)

    print(" INFO Computing Path B scores...")
    scores_b_val = get_cnn_scores(X_val)
    scores_b_atk = get_cnn_scores(X_atk)

    print(" INFO Computing Path C scores...")
    scores_c_val = get_gru_scores(X_val)
    scores_c_atk = get_gru_scores(X_atk)

    # 5. Orthogonality Analysis
    # Normalize scores to percentiles for fair comparison
    def to_percentile(val_scores, atk_scores):
        combined = np.concatenate([val_scores, atk_scores])
        from scipy.stats import rankdata
        ranks = rankdata(combined) / len(combined)
        return ranks[:len(val_scores)], ranks[len(val_scores):]

    pct_a_val, pct_a_atk = to_percentile(scores_a_val, scores_a_atk)
    pct_b_val, pct_b_atk = to_percentile(scores_b_val, scores_b_atk)
    pct_c_val, pct_c_atk = to_percentile(scores_c_val, scores_c_atk)

    # Correlation Matrix (Benign Data) - Should be LOW if orthogonal
    corr_matrix = np.corrcoef([pct_a_val, pct_b_val, pct_c_val])
    print("\nINFO Score Correlation on Benign Data (Goal: < 0.4):")
    print(f" A and B: {corr_matrix[0,1]:.3f}")
    print(f" B and C: {corr_matrix[1,2]:.3f}")
    print(f" A and C: {corr_matrix[0,2]:.3f}")

    # Detection Synergy: Samples missed by A but caught by B or C
    missed_by_a = pct_a_atk < 0.90 # A thinks it's likely benign
    caught_by_b = pct_b_atk > 0.95
    caught_by_c = pct_c_atk > 0.95

    synergy_count = np.sum(missed_by_a & (caught_by_b | caught_by_c))
    print(f"\nINFO Detection Synergy: {synergy_count} attacks missed by Path A but caught by B/C.")

    # Save visualization
    plt.figure(figsize=(10, 6))
    data_to_plot = [corr_matrix[0,1], corr_matrix[1,2], corr_matrix[0,2]]
    labels = ["A-B", "B-C", "A-C"]
    bars = plt.bar(labels, data_to_plot, color=['deepskyblue', 'blueviolet', 'mediumspringgreen'])
    plt.title(f"Feature Orthogonality (Correlation) - {args.dataset.upper()}")
    plt.ylabel("Pearson Correlation")
    plt.ylim(0, 1)
    plt.axhline(0.4, color='red', linestyle='--', label="Orthogonality Threshold")
    for bar, value in zip(bars, data_to_plot):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.text(
        0.99,
        0.95,
        f"Synergy: {synergy_count}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    plt.legend()

    plot_path = os.path.join(project_root, "experiments", args.dataset, "orthogonality_analysis.png")
    plt.savefig(plot_path)
    print(f"\nINFO Analysis plot saved to {plot_path}")

if __name__ == "__main__":
    main()

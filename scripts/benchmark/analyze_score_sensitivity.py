import os, pickle, json
import numpy as np

from benchmark_common import DEFAULT_DATASET, PROJECT_ROOT, resolve_phase5_selection

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")
    z_file = os.path.join(log_dir, "z_matrix_splits.pkl")

    if not os.path.exists(z_file):
        print(f"Error: {z_file} not found. Run benchmark_ablation_score_space.py first.")
        return

    with open(z_file, "rb") as f:
        data = pickle.load(f)

    z_all = np.vstack([data['z_train'], data['z_val'], data['z_test']])
    y_all = np.concatenate([data['y_train'], data['y_val'], data['y_test']])

    # 0 = Benign, 1 = Attack
    z_benign = z_all[y_all == 0]
    z_attack = z_all[y_all == 1]
    selection = resolve_phase5_selection(args.dataset)

    stats = {}
    path_names = [
        f"Path A ({selection['path_a_label']})",
        f"Path B ({selection['path_b_label']})",
        f"Path C ({selection['path_c_label']})",
    ]

    for i, name in enumerate(path_names):
        b_scores = z_benign[:, i]
        a_scores = z_attack[:, i]
        stats[name] = {
            "Benign": {
                "min": float(np.min(b_scores)),
                "max": float(np.max(b_scores)),
                "mean": float(np.mean(b_scores)),
                "std": float(np.std(b_scores))
            },
            "Attack": {
                "min": float(np.min(a_scores)),
                "max": float(np.max(a_scores)),
                "mean": float(np.mean(a_scores)),
                "std": float(np.std(a_scores))
            }
        }

    out_json = os.path.join(log_dir, "score_sensitivity_stats.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=4)

    print(json.dumps(stats, indent=4))
    print(f"\nINFO Saved JSON statistics to {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, name in enumerate(path_names):
            b_scores = z_benign[:, i]
            a_scores = z_attack[:, i]

            # Handle extreme outliers by clipping to 99.9th percentile for visual clarity
            p99_9 = np.percentile(np.concatenate([b_scores, a_scores]), 99.9)
            b_clip = np.clip(b_scores, a_min=None, a_max=p99_9)
            a_clip = np.clip(a_scores, a_min=None, a_max=p99_9)

            axes[i].hist(b_clip, bins=50, alpha=0.5, label='Benign', density=True, color='blue')
            axes[i].hist(a_clip, bins=50, alpha=0.5, label='Attack', density=True, color='red')
            axes[i].set_title(name)
            axes[i].legend()

            # Log scale keeps the tail visible.
            axes[i].set_yscale('log')

            # Path A often dominates dynamic range, so annotate its clipped range explicitly.
            if name.startswith("Path A ("):
                # The percentile cap keeps large Path A values from flattening the plot.
                axes[i].set_xlabel(f"Scores (Clipped to {p99_9:.1f})")
            else:
                axes[i].set_xlabel("Scores")
        plt.tight_layout()
        plot_out = os.path.join(log_dir, "score_distributions.png")
        plt.savefig(plot_out)
        print(f"INFO Plot saved successfully to {plot_out}")
    except Exception as e:
        print(f"WARNING Could not generate plots: {e}")

if __name__ == "__main__":
    main()

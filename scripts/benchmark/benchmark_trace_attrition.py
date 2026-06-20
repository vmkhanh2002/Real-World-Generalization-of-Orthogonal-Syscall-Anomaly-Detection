import os
import sys
import json

from benchmark_common import DEFAULT_DATASET, PROJECT_ROOT, resolve_dataset_root

def measure_attrition(dirpath, w=20):
    total_traces = 0
    discarded_traces = 0

    if not os.path.exists(dirpath):
        print(f"Warning: Directory not found {dirpath}")
        return 0, 0

    for fname in sorted(os.listdir(dirpath)):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath) and fname.endswith('.txt'):
            total_traces += 1
            with open(fpath, 'r') as f:
                content = f.read().strip().split()
                trace_len = len([x for x in content if x.isdigit()])
                if trace_len < w:
                    discarded_traces += 1

    return total_traces, discarded_traces

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset namespace.")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    dataset_dir = str(resolve_dataset_root(args.dataset))
    log_dir = os.path.join(project_root, "experiments", args.dataset, "logs")

    train_dir = os.path.join(dataset_dir, "Training_Data_Master")
    val_dir   = os.path.join(dataset_dir, "Validation_Data_Master")
    attack_base = os.path.join(dataset_dir, "Attack_Data_Master")

    tr_total, tr_disc = measure_attrition(train_dir)
    v_total, v_disc = measure_attrition(val_dir)

    atk_total, atk_disc = 0, 0
    if os.path.exists(attack_base):
        for atype in sorted(os.listdir(attack_base)):
            apath = os.path.join(attack_base, atype)
            if os.path.isdir(apath):
                t, d = measure_attrition(apath)
                atk_total += t
                atk_disc += d

    grand_total = tr_total + v_total + atk_total
    grand_disc = tr_disc + v_disc + atk_disc

    results = {
        "train": {"total": tr_total, "discarded": tr_disc, "percent": tr_disc / (tr_total + 1e-8) * 100},
        "val": {"total": v_total, "discarded": v_disc, "percent": v_disc / (v_total + 1e-8) * 100},
        "attack": {"total": atk_total, "discarded": atk_disc, "percent": atk_disc / (atk_total + 1e-8) * 100},
        "overall": {"total": grand_total, "discarded": grand_disc, "percent": grand_disc / (grand_total + 1e-8) * 100}
    }

    os.makedirs(log_dir, exist_ok=True)
    out_file = os.path.join(log_dir, "trace_attrition_metrics.json")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Overall Attrition: {grand_disc} / {grand_total} ({results['overall']['percent']:.2f}%)")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()

import argparse
import os
from typing import List, Tuple

from pipeline_config import (
    DEFAULT_DATASET,
    get_path_a_defaults,
    get_path_b_defaults,
    get_path_c_defaults,
    normalize_dataset_name,
)


def exists_status(path: str) -> Tuple[bool, str]:
    ok = os.path.exists(path)
    return ok, ("OK" if ok else "MISSING")


def print_block(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check_files(base: str, files: List[str]):
    missing = []
    for f in files:
        p = os.path.join(base, f)
        ok, tag = exists_status(p)
        print(f" [{tag}] {p}")
        if not ok:
            missing.append(p)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Check Path A/B/C input-output structure.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    args = parser.parse_args()
    dataset = normalize_dataset_name(args.dataset)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed", dataset, "phase1_base_arrays.pkl")
    logs_dir = os.path.join(project_root, "experiments", dataset, "logs")
    models_root = os.path.join(project_root, "models", dataset)

    print_block(f"I/O CHECK | DATASET: {dataset.upper()}")
    ok, tag = exists_status(data_file)
    print(f" [{tag}] INPUT: {data_file}")

    missing_all = []
    if not ok:
        missing_all.append(data_file)

    # Path A
    path_a_cfg = get_path_a_defaults(dataset)
    path_a_files = path_a_cfg.get("model_files", {})
    path_a_model_dir = os.path.join(models_root, "path_a")
    path_a_expected = [
        path_a_files.get("vectorizer", "vec.pkl"),
        path_a_files.get("pca", "pca.pkl"),
        path_a_files.get("sgd_ocsvm", path_a_files.get("classifier", "sgd_ocsvm.pkl")),
        path_a_files.get("isolation_forest", "ifo.pkl"),
        path_a_files.get("lof", "lof.pkl"),
        path_a_files.get("hbos", "hbos.pkl"),
        path_a_files.get("pca_error", "pca_err.pkl"),
        path_a_files.get("best_config_per_model_type", "path_a_best_config_per_model_type.json"),
    ]
    print_block("PATH A")
    print(f" Model dir: {path_a_model_dir}")
    missing_all.extend(check_files(path_a_model_dir, path_a_expected))
    missing_all.extend(check_files(logs_dir, ["path_a_results.json"]))

    # Path B
    path_b_cfg = get_path_b_defaults(dataset)
    path_b_files = path_b_cfg.get("model_files", {})
    path_b_model_dir = os.path.join(models_root, "path_b")
    path_b_expected = [
        path_b_files.get("cnn", "cnn.pth"),
        path_b_files.get("dense", "dense.pth"),
        path_b_files.get("lstm", "lstm.pth"),
        path_b_files.get("vae", "vae.pth"),
        path_b_files.get("svdd", "svdd.pth"),
        path_b_files.get("best_config_per_model_type", "path_b_best_config_per_model_type.json"),
    ]
    print_block("PATH B")
    print(f" Model dir: {path_b_model_dir}")
    missing_all.extend(check_files(path_b_model_dir, path_b_expected))
    missing_all.extend(check_files(logs_dir, ["path_b_results.json", "path_b_sweep_results.json"]))

    # Path C
    path_c_cfg = get_path_c_defaults(dataset)
    path_c_files = path_c_cfg.get("model_files", {})
    path_c_model_dir = os.path.join(models_root, "path_c")
    path_c_expected = [
        path_c_files.get("gru", "gru.pth"),
        path_c_files.get("cbow", "cbow.pth"),
        path_c_files.get("lstm_ae", "lstm_ae.pth"),
        path_c_files.get("best_config_per_model_type", "path_c_best_config_per_model_type.json"),
    ]
    print_block("PATH C")
    print(f" Model dir: {path_c_model_dir}")
    missing_all.extend(check_files(path_c_model_dir, path_c_expected))
    missing_all.extend(check_files(logs_dir, ["path_c_results.json", "path_c_sweep_results.json"]))

    print_block("SUMMARY")
    if missing_all:
        print(f" STATUS: FAILED ({len(missing_all)} missing)")
        for p in missing_all:
            print(f" - {p}")
    else:
        print(" STATUS: PASS (all required input/output files exist)")


if __name__ == "__main__":
    main()

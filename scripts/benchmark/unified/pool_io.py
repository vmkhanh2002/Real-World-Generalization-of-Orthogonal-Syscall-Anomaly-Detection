"""Lightweight loader for the unified eval pool (no torch/sklearn import)."""
import os
import pickle

_THIS = os.path.dirname(os.path.abspath(__file__))  # scripts/benchmark/unified
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))  # repo root


def unified_dir(dataset="32bit"):
    d = os.path.join(PROJECT_ROOT, "experiments", dataset, "logs", "unified")
    os.makedirs(d, exist_ok=True)
    return d


def pool_path(dataset="32bit"):
    return os.path.join(unified_dir(dataset), "unified_eval_pool.pkl")


def load_pool(dataset="32bit"):
    with open(pool_path(dataset), "rb") as f:
        return pickle.load(f)


def save_json(obj, dataset, filename):
    import json
    out = os.path.join(unified_dir(dataset), filename)
    with open(out, "w") as f:
        json.dump(obj, f, indent=2)
    return out

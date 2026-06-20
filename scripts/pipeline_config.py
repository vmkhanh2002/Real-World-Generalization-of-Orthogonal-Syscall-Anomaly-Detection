"""
Shared Pipeline Configuration Defaults
====================================

Centralizes dataset/path defaults so training, preprocessing, and fusion scripts
no longer rely on duplicated hardcoded constants.

"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping
import json
import os

DEFAULT_DATASET = "32bit"


# Dataset-level defaults (used by data pipelines and script bootstrap)

DATASET_DEFAULTS = {
    # 32 bit ADFA LD Configuration
    "32bit": {
        "dataset_name": "32bit",
        "window_size": 20,
        "stride": 2,
        "max_windows_per_trace": 100_000,
        "array_dtype": "int16",
    },
    # 64 bit DongTing configuration
    "64bit": {
        "dataset_name": "64bit",
        "window_size": 20,
        "stride": 2,
        "max_windows_per_trace": 100_000,
        "array_dtype": "int16",
    },
}



# Path A Defaults (Frequency/Rarity - TF-IDF + SGD-OCSVM)

# Best configuration from latest Path A v7 optimization summary:
# Feature: word(1,2)_1k_sub
# Reduction: PCA(100)
# Model: SGD-OCSVM(nu=0.01)
# Results: AUC=0.7975, AUPR=0.3702, F1_calib=0.2892
PATH_A_DEFAULTS = {
    "base": {
        # Best model selection for Phase 5 fusion
        "selected_model": "sgd_ocsvm",
        "selected_model_nu": 0.01,

        # Sampling: stratified by trace length (train only)
        "n_strata": 5,
        "min_per_stratum": 200,  # 32bit: 200, 64bit: 500

        # Feature extraction (BEST AUC config from latest v7 optimization summary)
        "tfidf_max_features": 1000,
        "tfidf_ngram_range": (1, 2),
        "tfidf_analyzer": "word",
        "tfidf_sublinear_tf": True,
        "pca_n_components": 100,

        # Model hyperparameters
        "sgd_ocsvm_nu": 0.01,
        "sgd_ocsvm_max_iter": 1000,
        "sgd_ocsvm_learning_rate": "optimal",
        "sgd_ocsvm_tol": 1e-4,
        "isolation_forest_n_estimators": 200,
        "isolation_forest_contamination": 0.05,
        "isolation_forest_max_samples": 1024,
        "lof_n_neighbors": 50,
        "lof_contamination": 0.05,
        "hbos_n_bins": 80,
        "hbos_contamination": 0.13,
        "pca_error_n_components": 12,

        # Calibration - Use PR-curve Best-F1 (matches optimize scripts)
        "calibration_method": "pr_curve",       # Best-F1 from PR-curve
        "calibration_split_ratio": 0.2,
        "bootstrap_n": 200,

        # Model files (relative to models/{dataset}/path_a/)
        "model_files": {
            "vectorizer": "vec.pkl",
            "pca": "pca100.pkl",
            "classifier": "sgd_ocsvm.pkl",
            "sgd_ocsvm": "sgd_ocsvm.pkl",
            "isolation_forest": "ifo.pkl",
            "lof": "lof.pkl",
            "hbos": "hbos.pkl",
            "pca_error": "pca_err.pkl",
            "best_config_per_model_type": "path_a_best_config_per_model_type.json",
        },

        # Best config per model type from optimization sweep summary
        "best_config_per_model_type": {
            "SGD-OCSVM": {
                "feature": "word(1,2)_1k_sub",
                "tfidf_analyzer": "word",
                "tfidf_ngram_range": (1, 2),
                "tfidf_max_features": 1000,
                "tfidf_sublinear_tf": True,
                "reduction": "PCA(100)",
                "pca_n_components": 100,
                "hyperparam": {"nu": 0.01},
            },
            "IsolationForest": {
                "feature": "charwb(3,3)_2k_sub",
                "tfidf_analyzer": "char_wb",
                "tfidf_ngram_range": (3, 3),
                "tfidf_max_features": 2000,
                "tfidf_sublinear_tf": True,
                "reduction": "SVD(50)",
                "pca_n_components": 50,
                "hyperparam": {},
            },
            "LOF": {
                "feature": "charwb(3,5)_2k_sub",
                "tfidf_analyzer": "char_wb",
                "tfidf_ngram_range": (3, 5),
                "tfidf_max_features": 2000,
                "tfidf_sublinear_tf": True,
                "reduction": "PCA(50)",
                "pca_n_components": 50,
                "hyperparam": {"k": 50},
            },
            "PCA-Error": {
                "feature": "charwb(3,3)_2k_sub",
                "tfidf_analyzer": "char_wb",
                "tfidf_ngram_range": (3, 3),
                "tfidf_max_features": 2000,
                "tfidf_sublinear_tf": True,
                "reduction": "PCA-Error-auto",
                "pca_n_components": 12,
                "hyperparam": {"n_components": 12},
            },
            "HBOS": {
                "feature": "word(1,2)_1k_sub",
                "tfidf_analyzer": "word",
                "tfidf_ngram_range": (1, 2),
                "tfidf_max_features": 1000,
                "tfidf_sublinear_tf": True,
                "reduction": "PCA(100)",
                "pca_n_components": 100,
                "hyperparam": {"n_bins": 80},
            },
        },
    },
    "32bit": {
        "min_per_stratum": 200,
    },
    "64bit": {
        "min_per_stratum": 500,
    },
}



# Path B Defaults (Topology/Geometry CNN 1D Autoencoder)

# Best configurations from latest Path B benchmark refresh:
# Source: latest Path B topology result log.
#
# CNN1D_AE: embed_dim=8, fraction=0.2 to AUC=0.7822, F1=0.4599 from BEST
# Dense_AE: latent_dim=16, fraction=0.2 to AUC=0.6043, F1=0.2846
# LSTM_AE: hidden_dim=16, fraction=0.2 to AUC=0.5397, F1=0.2816
# VAE: latent_dim=16, fraction=0.2 to AUC=0.6303, F1=0.3746
# DeepSVDD: hidden=32, fraction=0.2 to AUC=0.5244, F1=0.2467

PATH_B_DEFAULTS = {
    "base": {
        "selected_model": "cnn1d_ae",
        "epochs": 15,
        "fraction": 0.2,
        "min_samples": 5000,
        "embed_dim": 8,
        "dense_latent_dim": 16,
        "lstm_hidden_dim": 16,
        "vae_latent_dim": 16,
        "svdd_hidden_dim": 32,
        "cnn_restart_seeds": [42],
        "cnn_train_mode": "sweep_order",
        "calibration_quantile": 0.95,
        "calibration_method": "pr_curve",
        "model_files": {
            "cnn": "cnn.pth",
            "dense": "dense.pth",
            "lstm": "lstm.pth",
            "vae": "vae.pth",
            "svdd": "svdd.pth",
            "best_config_per_model_type": "path_b_best_config_per_model_type.json",
        },
        "best_config_per_model_type": {
            "CNN1D_AE": {
                "embed_dim": 8,
                "fraction": 0.2,
                "AUC": 0.7822,
                "F1_calib": 0.4599,
            },
            "Dense_AE": {
                "latent_dim": 16,
                "fraction": 0.2,
                "AUC": 0.6043,
                "F1_calib": 0.2846,
            },
            "LSTM_AE": {
                "hidden_dim": 16,
                "fraction": 0.2,
                "AUC": 0.5397,
                "F1_calib": 0.2816,
            },
            "VAE": {
                "latent_dim": 16,
                "fraction": 0.2,
                "AUC": 0.6303,
                "F1_calib": 0.3746,
            },
            "DeepSVDD": {
                "hidden": 32,
                "fraction": 0.2,
                "AUC": 0.5244,
                "F1_calib": 0.2467,
            },
        },
    }
}



# Path C Defaults (Chronology/Grammar Markov/HMM/LSTM AE/CBOW/GRU)

PATH_C_DEFAULTS = {
    "base": {
        "selected_model": "gru_predictor",
        "epochs": 15,
        "fraction": 0.5,
        "hidden_dim": 192,
        "num_layers": 1,
        "min_samples": 5000,
        "eval_batch_size": 512,
        "calibration_quantile": 0.95,
        "calibration_method": "pr_curve",
        "gru_embed_dim": 16,
        "cbow_embed_dim": 16,
        "lstm_ae_embed_dim": 16,
        "lstm_ae_hidden_dim": 64,
        "lstm_ae_epochs": 5,
        "cbow_epochs": 5,
        "model_files": {
            "gru": "gru.pth",
            "cbow": "cbow.pth",
            "lstm_ae": "lstm_ae.pth",
            "markov": "markov.pkl",
            "markov_meta": "markov.meta.json",
            "hmm": "hmm.pkl",
            "hmm_meta": "hmm.meta.json",
            "best_config_per_model_type": "path_c_best_config_per_model_type.json",
        },
        "best_config_per_model_type": {
            "GRU_Predictor": {
                "fraction": 0.5,
                "embed_dim": 16,
                "hidden_dim": 192,
                "num_layers": 1,
                "AUC": 0.8151,
                "F1_calib": 0.4313,
                "source": "path_c_results.json",
            },
            "CBOW_Predictor": {
                "fraction": 0.2,
                "embed_dim": 16,
                "AUC": 0.7712,
                "F1_calib": 0.7624,
                "source": "path_c_results.json",
            },
            "LSTM_AE_Sequence": {
                "fraction": 0.2,
                "embed_dim": 16,
                "hidden_dim": 64,
                "AUC": 0.6606,
                "F1_calib": 0.3174,
                "source": "path_c_results.json",
            },
            "Markov": {
                "fraction": 0.3,
                "order": 3,
                "AUC": 0.7966,
                "F1_calib": 0.4287,
                "source": "path_c_results.json",
            },
            "HMM": {
                "fraction": 0.1,
                "n_components": 4,
                "AUC": 0.6792,
                "F1_calib": 0.3138,
                "source": "path_c_results.json",
            },
        },
    }
}



# Phase 5 / Fusion defaults

FUSION_DEFAULTS = {
    "base": {
        "selected_model": "xgboost",
        "max_windows_per_split": 50000,
    }
}


def _deep_merge(base: Mapping, override: Mapping) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_dataset_name(dataset_name: str | None) -> str:
    if not dataset_name:
        return DEFAULT_DATASET
    name = str(dataset_name).strip().lower()
    aliases = {
        "32": "32bit",
        "32-bit": "32bit",
        "32bit": "32bit",
        "adfa-ld-32": "32bit",
        "64": "64bit",
        "64-bit": "64bit",
        "64bit": "64bit",
        "adfa-ld-64": "64bit",
    }
    return aliases.get(name, name)


def _resolve_defaults(config_map: Mapping, dataset_name: str | None) -> dict:
    dataset = normalize_dataset_name(dataset_name)
    base = deepcopy(config_map.get("base", {}))
    if dataset in config_map:
        base = _deep_merge(base, config_map[dataset])
    return base


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_best_cfg_from_models(dataset: str, path_name: str, fallback: Mapping) -> dict:
    root = _project_root()
    cfg_path = os.path.join(
        root,
        "models",
        dataset,
        path_name,
        fallback.get("model_files", {}).get("best_config_per_model_type", f"{path_name}_best_config_per_model_type.json"),
    )
    loaded = _load_json_if_exists(cfg_path)
    return deepcopy(loaded) if isinstance(loaded, Mapping) else deepcopy(fallback.get("best_config_per_model_type", {}))


def get_data_pipeline_defaults(dataset_name: str | None = None) -> dict:
    return _resolve_defaults(DATASET_DEFAULTS, dataset_name)


def get_path_a_defaults(dataset_name: str | None = None) -> dict:
    return _resolve_defaults(PATH_A_DEFAULTS, dataset_name)


def get_path_b_defaults(dataset_name: str | None = None) -> dict:
    return _resolve_defaults(PATH_B_DEFAULTS, dataset_name)


def get_path_c_defaults(dataset_name: str | None = None) -> dict:
    return _resolve_defaults(PATH_C_DEFAULTS, dataset_name)


def get_fusion_defaults(dataset_name: str | None = None) -> dict:
    return _resolve_defaults(FUSION_DEFAULTS, dataset_name)


def get_path_a_best_configs(dataset_name: str | None = None) -> dict:
    dataset = normalize_dataset_name(dataset_name)
    return _load_best_cfg_from_models(dataset, "path_a", get_path_a_defaults(dataset))


def get_path_b_best_configs(dataset_name: str | None = None) -> dict:
    dataset = normalize_dataset_name(dataset_name)
    return _load_best_cfg_from_models(dataset, "path_b", get_path_b_defaults(dataset))


def get_path_c_best_configs(dataset_name: str | None = None) -> dict:
    dataset = normalize_dataset_name(dataset_name)
    defaults = get_path_c_defaults(dataset)
    loaded = _load_best_cfg_from_models(dataset, "path_c", defaults)
    for key, value in defaults.get("best_config_per_model_type", {}).items():
        loaded.setdefault(key, deepcopy(value))
    return loaded


def get_path_a_selected_model_key(dataset_name: str | None = None) -> str:
    return str(get_path_a_defaults(dataset_name).get("selected_model", "sgd_ocsvm"))


def get_path_b_selected_model_key(dataset_name: str | None = None) -> str:
    return str(get_path_b_defaults(dataset_name).get("selected_model", "cnn1d_ae"))


def get_path_c_selected_model_key(dataset_name: str | None = None) -> str:
    return str(get_path_c_defaults(dataset_name).get("selected_model", "gru_predictor"))


def apply_optional_overrides(args, defaults: Mapping, keys: Iterable[str]) -> None:
    for key in keys:
        if getattr(args, key, None) is None and key in defaults:
            setattr(args, key, deepcopy(defaults[key]))



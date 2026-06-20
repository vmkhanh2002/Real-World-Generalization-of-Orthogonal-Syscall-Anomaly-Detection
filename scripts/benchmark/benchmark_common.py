import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from models import Conv1DAE, GRUPredictor
from pipeline_config import (
    DEFAULT_DATASET,
    get_path_a_best_configs,
    get_path_a_defaults,
    get_path_a_selected_model_key,
    get_path_b_defaults,
    get_path_b_selected_model_key,
    get_path_c_defaults,
    get_path_c_selected_model_key,
)
from train_path_a import _resolve_model_feature_cfg


PATH_A_SELECTION_TO_MODEL_TYPE = {
    "sgd_ocsvm": "SGD-OCSVM",
    "ocsvm": "SGD-OCSVM",
    "isolation_forest": "IsolationForest",
    "ifo": "IsolationForest",
    "lof": "LOF",
    "hbos": "HBOS",
    "pca_error": "PCA-Error",
    "pca-err": "PCA-Error",
}

PATH_A_MODEL_TYPE_TO_FILE_KEY = {
    "SGD-OCSVM": "sgd_ocsvm",
    "IsolationForest": "isolation_forest",
    "LOF": "lof",
    "HBOS": "hbos",
    "PCA-Error": "pca_error",
}

PATH_B_SELECTION_TO_FILE_KEY = {
    "cnn1d_ae": "cnn",
    "cnn": "cnn",
}

PATH_B_SELECTION_TO_LABEL = {
    "cnn1d_ae": "CNN1D_AE",
    "cnn": "CNN1D_AE",
}

PATH_C_SELECTION_TO_FILE_KEY = {
    "gru_predictor": "gru",
    "gru": "gru",
}

PATH_C_SELECTION_TO_LABEL = {
    "gru_predictor": "GRU_Predictor",
    "gru": "GRU_Predictor",
}


def stringify_windows(windows):
    return [" ".join(map(str, row)) for row in windows]


def resolve_dataset_root(dataset):
    base = PROJECT_ROOT / "datasets" / dataset
    if dataset == "32bit":
        adfa_root = base / "ADFA-LD" / "ADFA-LD"
        if adfa_root.exists():
            return adfa_root
    return base


def _load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_phase5_manifest(dataset):
    manifest_path = PROJECT_ROOT / "models" / dataset / "phase5_fusion" / "meta_best_model_manifest.json"
    return _load_json(manifest_path)


def load_phase5_meta_payload(dataset):
    payload_path = PROJECT_ROOT / "models" / dataset / "phase5_fusion" / "meta_best_model.pkl"
    with open(payload_path, "rb") as handle:
        return pickle.load(handle)


def load_phase5_meta_registry(dataset):
    registry_path = PROJECT_ROOT / "models" / dataset / "phase5_fusion" / "meta_models_index.json"
    with open(registry_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_phase5_meta_payload_by_name(dataset, model_name):
    registry = load_phase5_meta_registry(dataset)
    for entry in registry.get("models", []):
        if entry.get("model_name") == model_name:
            with open(entry["model_file"], "rb") as handle:
                return pickle.load(handle)
    selected_name = registry.get("selected_model_name")
    if selected_name == model_name:
        return load_phase5_meta_payload(dataset)
    raise KeyError(f"Phase 5 meta model '{model_name}' not found for dataset '{dataset}'.")


def resolve_phase5_baseline_dir(dataset):
    return PROJECT_ROOT / "models" / dataset / "phase5_fusion" / "baselines"


def load_phase5_baseline_registry(dataset):
    registry_path = resolve_phase5_baseline_dir(dataset) / "baseline_models_index.json"
    with open(registry_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_phase5_baseline_payload(dataset, artifact_name):
    registry = load_phase5_baseline_registry(dataset)
    for entry in registry.get("artifacts", []):
        if entry.get("artifact_name") == artifact_name:
            with open(entry["model_file"], "rb") as handle:
                return pickle.load(handle)
    raise KeyError(f"Phase 5 baseline artifact '{artifact_name}' not found for dataset '{dataset}'.")


def resolve_phase5_z_splits_path(dataset):
    return PROJECT_ROOT / "experiments" / dataset / "logs" / "z_matrix_splits.pkl"


def resolve_phase5_z_eval_path(dataset):
    return PROJECT_ROOT / "experiments" / dataset / "logs" / "z_matrix_eval_full.pkl"


def load_phase5_z_splits(dataset):
    with open(resolve_phase5_z_splits_path(dataset), "rb") as handle:
        return pickle.load(handle)


def load_phase5_z_eval(dataset):
    with open(resolve_phase5_z_eval_path(dataset), "rb") as handle:
        return pickle.load(handle)


def _normalize_key(value, fallback):
    if value is None:
        return str(fallback).strip().lower()
    return str(value).strip().lower()


def _path_a_selection_to_model_type(selection_key):
    model_type = PATH_A_SELECTION_TO_MODEL_TYPE.get(selection_key)
    if model_type is None:
        raise ValueError(
            f"Unsupported Path A selection '{selection_key}'. "
            f"Known selections: {sorted(PATH_A_SELECTION_TO_MODEL_TYPE.keys())}"
        )
    return model_type


def resolve_phase5_selection(dataset):
    selected = {
        "path_a": _normalize_key(get_path_a_selected_model_key(dataset), "sgd_ocsvm"),
        "path_b": _normalize_key(get_path_b_selected_model_key(dataset), "cnn1d_ae"),
        "path_c": _normalize_key(get_path_c_selected_model_key(dataset), "gru_predictor"),
    }
    source = "pipeline_config"

    manifest = resolve_phase5_manifest(dataset)
    manifest_selected = (manifest or {}).get("selected_paths", {})
    if manifest_selected:
        for path_name in ("path_a", "path_b", "path_c"):
            if manifest_selected.get(path_name):
                selected[path_name] = _normalize_key(manifest_selected.get(path_name), selected[path_name])
        source = "phase5_manifest"

    return {
        "source": source,
        "path_a": selected["path_a"],
        "path_b": selected["path_b"],
        "path_c": selected["path_c"],
        "path_a_label": _path_a_selection_to_model_type(selected["path_a"]),
        "path_b_label": PATH_B_SELECTION_TO_LABEL.get(selected["path_b"], selected["path_b"]),
        "path_c_label": PATH_C_SELECTION_TO_LABEL.get(selected["path_c"], selected["path_c"]),
    }


def _resolve_path_a_best_configs(dataset):
    path_a_cfg = get_path_a_defaults(dataset)
    best_cfg_name = path_a_cfg.get("model_files", {}).get(
        "best_config_per_model_type",
        "path_a_best_config_per_model_type.json",
    )
    best_cfg_path = PROJECT_ROOT / "models" / dataset / "path_a" / best_cfg_name
    best_cfgs = _load_json(best_cfg_path)
    if best_cfgs:
        return best_cfgs
    return get_path_a_best_configs(dataset)


def _resolve_path_a_model_path(dataset, model_type):
    path_a_cfg = get_path_a_defaults(dataset)
    model_files = path_a_cfg.get("model_files", {})
    model_file_key = PATH_A_MODEL_TYPE_TO_FILE_KEY.get(model_type)
    if model_file_key is None:
        raise ValueError(
            f"Unsupported Path A model type '{model_type}'. "
            f"Known model types: {sorted(PATH_A_MODEL_TYPE_TO_FILE_KEY.keys())}"
        )

    model_file = model_files.get(model_file_key)
    if model_file is None and model_type == "SGD-OCSVM":
        model_file = model_files.get("classifier", "sgd_ocsvm.pkl")
    if model_file is None:
        raise FileNotFoundError(
            f"Could not resolve model filename for Path A model type '{model_type}' from pipeline_config."
        )

    return PROJECT_ROOT / "models" / dataset / "path_a" / model_file


def _resolve_path_a_vectorizer_path(dataset):
    path_a_cfg = get_path_a_defaults(dataset)
    model_dir = PROJECT_ROOT / "models" / dataset / "path_a"
    model_files = path_a_cfg.get("model_files", {})
    candidates = [
        model_dir / model_files.get("vectorizer", "vec.pkl"),
        model_dir / "vec.pkl",
        model_dir / "path_a_vec.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing Path A vectorizer artifact for dataset '{dataset}'. Checked: {[str(c) for c in candidates]}"
    )


def _resolve_path_a_reducer_path(dataset, model_type, cfg):
    if model_type == "PCA-Error":
        return None

    model_dir = PROJECT_ROOT / "models" / dataset / "path_a"
    reduction = str(cfg.get("reduction", "")).strip()
    n_components = cfg.get("pca_n_components")
    candidates = []

    if reduction.startswith("PCA("):
        if n_components is not None:
            candidates.extend(
                [
                    model_dir / f"pca{n_components}.pkl",
                    model_dir / f"path_a_pca{n_components}.pkl",
                ]
            )
        candidates.extend([model_dir / "pca.pkl", model_dir / "path_a_pca.pkl"])
    elif reduction.startswith("SVD("):
        if n_components is not None:
            candidates.extend(
                [
                    model_dir / f"svd{n_components}.pkl",
                    model_dir / f"path_a_svd{n_components}.pkl",
                ]
            )
        candidates.extend([model_dir / "svd.pkl", model_dir / "path_a_svd.pkl"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Missing Path A reducer artifact for dataset "
        f"'{dataset}' and model '{model_type}' with reduction '{reduction}'. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def _resolve_path_b_checkpoint_path(dataset, selection_key):
    cfg = get_path_b_defaults(dataset)
    file_key = PATH_B_SELECTION_TO_FILE_KEY.get(selection_key)
    if file_key is None:
        raise ValueError(
            f"Unsupported Path B selection '{selection_key}'. "
            f"Known selections: {sorted(PATH_B_SELECTION_TO_FILE_KEY.keys())}"
        )
    ckpt_name = cfg.get("model_files", {}).get(file_key)
    if ckpt_name is None:
        raise FileNotFoundError(f"Could not resolve checkpoint filename for Path B selection '{selection_key}'.")
    return PROJECT_ROOT / "models" / dataset / "path_b" / ckpt_name


def _resolve_path_c_checkpoint_path(dataset, selection_key):
    cfg = get_path_c_defaults(dataset)
    file_key = PATH_C_SELECTION_TO_FILE_KEY.get(selection_key)
    if file_key is None:
        raise ValueError(
            f"Unsupported Path C selection '{selection_key}'. "
            f"Known selections: {sorted(PATH_C_SELECTION_TO_FILE_KEY.keys())}"
        )
    ckpt_name = cfg.get("model_files", {}).get(file_key)
    if ckpt_name is None:
        raise FileNotFoundError(f"Could not resolve checkpoint filename for Path C selection '{selection_key}'.")
    return PROJECT_ROOT / "models" / dataset / "path_c" / ckpt_name


def resolve_path_bc_artifacts(dataset):
    manifest = resolve_phase5_manifest(dataset)
    if manifest:
        path_b_artifact = manifest.get("path_b_artifact")
        path_c_artifact = manifest.get("path_c_artifact")
        if path_b_artifact and path_c_artifact:
            path_b = PROJECT_ROOT / Path(path_b_artifact.replace("\\", "/"))
            path_c = PROJECT_ROOT / Path(path_c_artifact.replace("\\", "/"))
            if path_b.exists() and path_c.exists():
                return path_b, path_c

    selection = resolve_phase5_selection(dataset)
    return (
        _resolve_path_b_checkpoint_path(dataset, selection["path_b"]),
        _resolve_path_c_checkpoint_path(dataset, selection["path_c"]),
    )


def _infer_init_kwargs_from_state_dict(model_class, state_dict):
    if model_class is Conv1DAE:
        vocab_size, embed_dim = state_dict["embedding.weight"].shape
        return {"vocab_size": vocab_size, "embed_dim": embed_dim}
    if model_class is GRUPredictor:
        vocab_plus_one, embed_dim = state_dict["embedding.weight"].shape
        hidden_dim = state_dict["gru.weight_ih_l0"].shape[0] // 3
        return {
            "vocab_size": vocab_plus_one - 1,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": 1,
        }
    raise ValueError(f"Cannot infer init kwargs for {model_class.__name__}")


def load_checkpoint_model(checkpoint_path, model_class, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    meta_path = Path(f"{checkpoint_path}.meta.json")
    init_kwargs = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        init_kwargs = meta.get("init_kwargs", {})
    if not init_kwargs:
        init_kwargs = _infer_init_kwargs_from_state_dict(model_class, state_dict)

    model = model_class(**init_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_path_b_model(dataset, device):
    path_b_ckpt, _ = resolve_path_bc_artifacts(dataset)
    return load_checkpoint_model(path_b_ckpt, Conv1DAE, device), path_b_ckpt


def load_path_c_model(dataset, device):
    _, path_c_ckpt = resolve_path_bc_artifacts(dataset)
    return load_checkpoint_model(path_c_ckpt, GRUPredictor, device), path_c_ckpt


def load_path_a_bundle(dataset, model_type, X_train_w, X_val_w, X_attack_w):
    path_a_cfg = get_path_a_defaults(dataset)
    best_cfgs = _resolve_path_a_best_configs(dataset)
    cfg = _resolve_model_feature_cfg(path_a_cfg, best_cfgs, model_type)

    model_path = _resolve_path_a_model_path(dataset, model_type)
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    vectorizer_path = _resolve_path_a_vectorizer_path(dataset)
    with open(vectorizer_path, "rb") as handle:
        vectorizer = pickle.load(handle)

    reducer_path = _resolve_path_a_reducer_path(dataset, model_type, cfg)
    reducer = None
    if reducer_path is not None:
        with open(reducer_path, "rb") as handle:
            reducer = pickle.load(handle)

    val_tf = vectorizer.transform(stringify_windows(X_val_w))
    attack_tf = vectorizer.transform(stringify_windows(X_attack_w))

    if model_type == "PCA-Error":
        x_val = val_tf.toarray()
        x_attack = attack_tf.toarray()
    elif reducer is None:
        x_val = val_tf
        x_attack = attack_tf
    elif reducer.__class__.__name__ == "PCA":
        x_val = reducer.transform(val_tf.toarray())
        x_attack = reducer.transform(attack_tf.toarray())
    else:
        x_val = reducer.transform(val_tf)
        x_attack = reducer.transform(attack_tf)

    return {
        "cfg": cfg,
        "dataset": dataset,
        "model_type": model_type,
        "model_path": model_path,
        "model": model,
        "vectorizer_path": vectorizer_path,
        "reducer_path": reducer_path,
        "vectorizer": vectorizer,
        "reducer": reducer,
        "x_train": None,
        "x_val": x_val,
        "x_attack": x_attack,
    }


def load_selected_path_a_bundle(dataset, X_train_w, X_val_w, X_attack_w):
    selection = resolve_phase5_selection(dataset)
    model_type = _path_a_selection_to_model_type(selection["path_a"])
    bundle = load_path_a_bundle(dataset, model_type, X_train_w, X_val_w, X_attack_w)
    bundle["selected_key"] = selection["path_a"]
    bundle["selected_label"] = selection["path_a_label"]
    bundle["selection_source"] = selection["source"]
    return bundle


def score_path_a_bundle(path_a_bundle):
    model_type = path_a_bundle["model_type"]
    model = path_a_bundle["model"]
    x_val = path_a_bundle["x_val"]
    x_attack = path_a_bundle["x_attack"]

    if model_type in {"SGD-OCSVM", "LOF"}:
        scores_val = -model.decision_function(x_val)
        scores_attack = -model.decision_function(x_attack)
    elif model_type == "IsolationForest":
        scores_val = -model.score_samples(x_val)
        scores_attack = -model.score_samples(x_attack)
    elif model_type == "HBOS":
        scores_val = model.decision_function(x_val)
        scores_attack = model.decision_function(x_attack)
    elif model_type == "PCA-Error":
        recon_val = model.inverse_transform(model.transform(x_val))
        recon_attack = model.inverse_transform(model.transform(x_attack))
        scores_val = np.mean((x_val - recon_val) ** 2, axis=1)
        scores_attack = np.mean((x_attack - recon_attack) ** 2, axis=1)
    else:
        raise ValueError(f"Unsupported Path A scoring model type: {model_type}")

    return np.asarray(scores_val), np.asarray(scores_attack)

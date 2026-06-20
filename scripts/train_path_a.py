"""
Phase 2: Path A
"""

import os, pickle, sys, json, time, argparse, random
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
warnings.filterwarnings("error", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             average_precision_score, f1_score)

from pipeline_config import DEFAULT_DATASET, apply_optional_overrides, get_path_a_best_configs, get_path_a_defaults

try:
    from pyod.models.hbos import HBOS
    import pyod
    PYOD_VERSION = getattr(pyod, '__version__', 'unknown')
except ImportError:
    HBOS = None
    PYOD_VERSION = 'not installed'

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

#
# Reproducibility
#
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#
# Metrics (refactored)
#
def get_best_f1_debug(y_true, scores):
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-8)
    return float(f1[np.argmax(f1)])

def compute_fpr_at_tpr95(y_true, scores):
    sort_idx  = np.argsort(-scores)
    y_sorted  = y_true[sort_idx]
    n_pos     = max(y_true.sum(), 1)
    n_neg     = max(len(y_true) - y_true.sum(), 1)
    tpr = np.cumsum(y_sorted)  / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    above95 = np.where(tpr >= 0.95)[0]
    return float(fpr[above95[0]]) if len(above95) > 0 else 1.0

def compute_all_metrics(y_true, scores, scores_calib):
    auc_roc  = roc_auc_score(y_true, scores)
    aupr     = average_precision_score(y_true, scores)
    f1_dbg   = get_best_f1_debug(y_true, scores)
    # Calibrated threshold from separate calib split
    thr_95   = np.percentile(scores_calib, 95)
    y_pred   = (scores >= thr_95).astype(int)
    f1_cal   = f1_score(y_true, y_pred, zero_division=0)
    fpr95    = compute_fpr_at_tpr95(y_true, scores)
    return {
        "AUC_ROC":      round(auc_roc, 4),
        "AUPR":         round(aupr, 4),
        "F1_debug":     round(f1_dbg, 4),
        "F1_calib":     round(f1_cal, 4),
        "FPR_at_TPR95": round(fpr95, 4),
    }

def bootstrap_ci(y_true, scores, scores_calib, n_bootstrap=200, ci=95):
    results = {"AUC_ROC": [], "AUPR": [], "F1_calib": []}
    thr_95 = np.percentile(scores_calib, 95)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_b, sc_b = y_true[idx], scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        results["AUC_ROC"].append(roc_auc_score(y_b, sc_b))
        results["AUPR"].append(average_precision_score(y_b, sc_b))
        y_pred_b = (sc_b >= thr_95).astype(int)
        results["F1_calib"].append(f1_score(y_b, y_pred_b, zero_division=0))
    ci_lo = (100 - ci) / 2
    ci_hi = 100 - ci_lo
    out = {}
    for k, v in results.items():
        arr = np.array(v)
        out[k] = {
            "mean": round(np.mean(arr), 4),
            "lo":   round(np.percentile(arr, ci_lo), 4),
            "hi":   round(np.percentile(arr, ci_hi), 4),
        }
    return out

def reconstruction_error(pca_model, X):
    """Compute per-sample mean squared reconstruction error."""
    return np.mean((X - pca_model.inverse_transform(pca_model.transform(X)))**2, axis=1)

def stringify_windows(windows):
    """Convert syscall ID windows to space-separated token strings for TF-IDF."""
    return [" ".join(map(str, row)) for row in windows]

def fmt(m):
    return (f"AUC={m['AUC_ROC']:.4f} AUPR={m['AUPR']:.4f} "
            f"F1-cal={m['F1_calib']:.4f} FPR95={m['FPR_at_TPR95']:.4f}")

def fmt_ci(ci):
    return (f"AUC={ci['AUC_ROC']['mean']:.4f}"
            f"[{ci['AUC_ROC']['lo']:.4f},{ci['AUC_ROC']['hi']:.4f}] "
            f"F1-cal={ci['F1_calib']['mean']:.4f}"
            f"[{ci['F1_calib']['lo']:.4f},{ci['F1_calib']['hi']:.4f}]")

#
# Sampling: Stratified by trace length (replaces sample_asym)
#
def sample_stratified_length(matrix, trace_ids, n_strata=5, min_per_stratum=200):
    """
    Sample training windows across trace-length buckets.

    The old asymmetric sampler dropped most long traces. This keeps each length
    bucket represented in the training slice while validation and attack windows
    stay untouched.
    """
    unique = np.unique(trace_ids)
    lens   = np.array([int(np.sum(trace_ids == t)) for t in unique])
    sorted_idx = np.argsort(lens)
    strata = np.array_split(sorted_idx, n_strata)
    sel = []
    for stratum in strata:
        traces_in_stratum = unique[stratum]
        pool = np.where(np.isin(trace_ids, traces_in_stratum))[0]
        n = min(min_per_stratum, len(pool))
        if n > 0:
            sel.extend(np.random.choice(pool, size=n, replace=False))
    return matrix[np.sort(sel)]

#
# Evaluation
#
def _resolve_model_feature_cfg(path_a_cfg, best_cfgs, model_type):
    best_cfg = best_cfgs.get(model_type, {})
    return {
        "analyzer": best_cfg.get("tfidf_analyzer", path_a_cfg["tfidf_analyzer"]),
        "ngram_range": tuple(best_cfg.get("tfidf_ngram_range", path_a_cfg["tfidf_ngram_range"])),
        "max_features": int(best_cfg.get("tfidf_max_features", path_a_cfg["tfidf_max_features"])),
        "sublinear_tf": bool(best_cfg.get("tfidf_sublinear_tf", path_a_cfg.get("tfidf_sublinear_tf", False))),
        "reduction": str(best_cfg.get("reduction", f"PCA({path_a_cfg['pca_n_components']})")),
        "pca_n_components": int(best_cfg.get("pca_n_components", path_a_cfg["pca_n_components"])),
        "hyperparam": dict(best_cfg.get("hyperparam", {})),
    }


def _build_features_for_model(model_type, cfg, str_tr, str_calib, str_test_n, str_at):
    token_pattern = r"(=u)\b\w+\b" if cfg["analyzer"] == "word" else None
    vec = TfidfVectorizer(
        max_features=cfg["max_features"],
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        sublinear_tf=cfg["sublinear_tf"],
        token_pattern=token_pattern,
    )
    X_tr_tf = vec.fit_transform(str_tr)
    X_cal_tf = vec.transform(str_calib)
    X_test_n_tf = vec.transform(str_test_n)
    X_at_tf = vec.transform(str_at)

    if model_type == "PCA-Error":
        return {
            "vectorizer": vec,
            "reducer": None,
            "X_tr_raw": X_tr_tf.toarray(),
            "X_calib_raw": X_cal_tf.toarray(),
            "X_test_raw": np.vstack([X_test_n_tf.toarray(), X_at_tf.toarray()]),
        }

    reduction = cfg["reduction"].upper()
    if reduction.startswith("SVD"):
        reducer = TruncatedSVD(n_components=cfg["pca_n_components"], random_state=42)
        X_tr_red = reducer.fit_transform(X_tr_tf)
        X_calib_red = reducer.transform(X_cal_tf)
        X_test_n_red = reducer.transform(X_test_n_tf)
        X_at_red = reducer.transform(X_at_tf)
    else:
        reducer = PCA(n_components=cfg["pca_n_components"], random_state=42)
        X_tr_dense = X_tr_tf.toarray()
        X_cal_dense = X_cal_tf.toarray()
        X_test_n_dense = X_test_n_tf.toarray()
        X_at_dense = X_at_tf.toarray()
        X_tr_red = reducer.fit_transform(X_tr_dense)
        X_calib_red = reducer.transform(X_cal_dense)
        X_test_n_red = reducer.transform(X_test_n_dense)
        X_at_red = reducer.transform(X_at_dense)

    return {
        "vectorizer": vec,
        "reducer": reducer,
        "X_tr_red": X_tr_red,
        "X_calib_red": X_calib_red,
        "X_test_red": np.vstack([X_test_n_red, X_at_red]),
    }


def run_all_models(str_tr, str_calib, str_test_n, str_at, y_test, atk_ratio, path_a_cfg, best_cfgs):
    results = {}
    models = {}
    all_ci = {}
    artifacts = {}

    model_map = {
        "SGD-OCSVM": "sgd_ocsvm",
        "IsolationForest": "ifo",
        "LOF": "lof",
        "PCA-Error": "pca_error",
        "HBOS": "hbos",
    }
    selected_model = str(path_a_cfg.get("selected_model", "sgd_ocsvm")).lower()

    # SGD-OCSVM
    t = time.time()
    cfg = _resolve_model_feature_cfg(path_a_cfg, best_cfgs, "SGD-OCSVM")
    feat = _build_features_for_model("SGD-OCSVM", cfg, str_tr, str_calib, str_test_n, str_at)
    nu = float(cfg["hyperparam"].get("nu", path_a_cfg["sgd_ocsvm_nu"]))
    sgd_ocsvm = SGDOneClassSVM(
        nu=nu,
        max_iter=path_a_cfg.get("sgd_ocsvm_max_iter", 1000),
        learning_rate=path_a_cfg.get("sgd_ocsvm_learning_rate", "optimal"),
        random_state=42,
        tol=path_a_cfg.get("sgd_ocsvm_tol", 1e-4),
    )
    sgd_ocsvm.fit(feat["X_tr_red"])
    sc_test = -sgd_ocsvm.decision_function(feat["X_test_red"])
    sc_calib = -sgd_ocsvm.decision_function(feat["X_calib_red"])
    m = compute_all_metrics(y_test, sc_test, sc_calib)
    ci = bootstrap_ci(y_test, sc_test, sc_calib)
    print(f" SGD-OCSVM {fmt(m)} ({time.time()-t:.1f}s)")
    print(f" 95% CI: {fmt_ci(ci)}")
    results["SGD-OCSVM"] = m
    all_ci["SGD-OCSVM"] = ci
    models["sgd_ocsvm"] = sgd_ocsvm
    artifacts["sgd_ocsvm"] = {"vectorizer": feat["vectorizer"], "reducer": feat["reducer"]}

    # IsolationForest
    t = time.time()
    cfg = _resolve_model_feature_cfg(path_a_cfg, best_cfgs, "IsolationForest")
    feat = _build_features_for_model("IsolationForest", cfg, str_tr, str_calib, str_test_n, str_at)
    ifo = IsolationForest(
        n_estimators=path_a_cfg.get("isolation_forest_n_estimators", 200),
        contamination=path_a_cfg.get("isolation_forest_contamination", 0.05),
        max_samples=min(path_a_cfg.get("isolation_forest_max_samples", 1024), len(feat["X_tr_red"])),
        random_state=42,
    )
    ifo.fit(feat["X_tr_red"])
    sc_test = -ifo.score_samples(feat["X_test_red"])
    sc_calib = -ifo.score_samples(feat["X_calib_red"])
    m = compute_all_metrics(y_test, sc_test, sc_calib)
    ci = bootstrap_ci(y_test, sc_test, sc_calib)
    print(f" IsoForest {fmt(m)} ({time.time()-t:.1f}s)")
    print(f" 95% CI: {fmt_ci(ci)}")
    results["IsolationForest"] = m
    all_ci["IsolationForest"] = ci
    models["ifo"] = ifo
    artifacts["ifo"] = {"vectorizer": feat["vectorizer"], "reducer": feat["reducer"]}

    # LOF
    t = time.time()
    cfg = _resolve_model_feature_cfg(path_a_cfg, best_cfgs, "LOF")
    feat = _build_features_for_model("LOF", cfg, str_tr, str_calib, str_test_n, str_at)
    k = int(cfg["hyperparam"].get("k", path_a_cfg.get("lof_n_neighbors", 50)))
    lof = LocalOutlierFactor(
        n_neighbors=min(k, len(feat["X_tr_red"]) - 1),
        novelty=True,
        contamination=path_a_cfg.get("lof_contamination", 0.05),
    )
    lof.fit(feat["X_tr_red"])
    sc_test = -lof.decision_function(feat["X_test_red"])
    sc_calib = -lof.decision_function(feat["X_calib_red"])
    m = compute_all_metrics(y_test, sc_test, sc_calib)
    ci = bootstrap_ci(y_test, sc_test, sc_calib)
    print(f" LOF (k={k}) {fmt(m)} ({time.time()-t:.1f}s)")
    print(f" 95% CI: {fmt_ci(ci)}")
    results["LOF"] = m
    all_ci["LOF"] = ci
    models["lof"] = lof
    artifacts["lof"] = {"vectorizer": feat["vectorizer"], "reducer": feat["reducer"]}

    # PCA-Error
    t = time.time()
    cfg = _resolve_model_feature_cfg(path_a_cfg, best_cfgs, "PCA-Error")
    feat = _build_features_for_model("PCA-Error", cfg, str_tr, str_calib, str_test_n, str_at)
    n_comp = int(cfg["hyperparam"].get("n_components", path_a_cfg.get("pca_error_n_components", 12)))
    n_comp = max(2, min(n_comp, feat["X_tr_raw"].shape[1]))
    pca_err = PCA(n_components=n_comp, random_state=42)
    pca_err.fit(feat["X_tr_raw"])
    var_exp = pca_err.explained_variance_ratio_.sum()
    print(f" PCA-Error(n={n_comp}) explained variance: {var_exp:.3f}")
    err_calib = reconstruction_error(pca_err, feat["X_calib_raw"])
    err_test = reconstruction_error(pca_err, feat["X_test_raw"])
    m = compute_all_metrics(y_test, err_test, err_calib)
    ci = bootstrap_ci(y_test, err_test, err_calib)
    print(f" PCA-Error {fmt(m)} ({time.time()-t:.1f}s)")
    print(f" 95% CI: {fmt_ci(ci)}")
    m["n_components"] = n_comp
    m["variance_explained"] = round(var_exp, 4)
    results["PCA-Error"] = m
    all_ci["PCA-Error"] = ci
    models["pca_err"] = pca_err
    artifacts["pca_error"] = {"vectorizer": feat["vectorizer"], "reducer": pca_err}

    # HBOS
    if HBOS is not None:
        t = time.time()
        cfg = _resolve_model_feature_cfg(path_a_cfg, best_cfgs, "HBOS")
        feat = _build_features_for_model("HBOS", cfg, str_tr, str_calib, str_test_n, str_at)
        bins = int(cfg["hyperparam"].get("n_bins", path_a_cfg.get("hbos_n_bins", 80)))
        hbos = HBOS(
            n_bins=bins,
            contamination=path_a_cfg.get("hbos_contamination", 0.13),
        )
        hbos.fit(feat["X_tr_red"])
        sc_test = hbos.decision_function(feat["X_test_red"])
        sc_calib = hbos.decision_function(feat["X_calib_red"])
        m = compute_all_metrics(y_test, sc_test, sc_calib)
        ci = bootstrap_ci(y_test, sc_test, sc_calib)
        print(f" HBOS {fmt(m)} ({time.time()-t:.1f}s)")
        print(f" 95% CI: {fmt_ci(ci)}")
        results["HBOS"] = m
        all_ci["HBOS"] = ci
        models["hbos"] = hbos
        artifacts["hbos"] = {"vectorizer": feat["vectorizer"], "reducer": feat["reducer"]}

    print(f"\n --- Baselines ---")
    print(f" Random classifier: AUC=0.5000 AUPR={atk_ratio:.4f} "
          f"(attack fraction = {atk_ratio:.4f})")

    selected_artifact = artifacts.get(selected_model)
    if selected_artifact is None:
        fallback_key = model_map.get("SGD-OCSVM")
        selected_artifact = artifacts.get(fallback_key)
    return results, models, all_ci, selected_artifact

#
# Main
#
def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--n_strata", type=int, default=None,
                        help="Number of length strata (default 5)")
    parser.add_argument("--min_per_stratum", type=int, default=None,
                        help="Min windows per stratum (default 200 for 32bit, 500 for 64bit)")
    args = parser.parse_args()
    path_a_cfg = get_path_a_defaults(args.dataset)
    apply_optional_overrides(args, path_a_cfg, ["n_strata", "min_per_stratum"])

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "processed",
                             args.dataset, "phase1_base_arrays.pkl")
    log_dir   = os.path.join(project_root, "experiments", args.dataset, "logs")
    model_dir = os.path.join(project_root, "models", args.dataset, "path_a")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 75)
    print(f"PATH A CONFIG | {args.dataset.upper()}")
    print("=" * 75)
    print(f"\n pyod version: {PYOD_VERSION}")
    print(f" sklearn: {__import__('sklearn').__version__}")
    print(f" numpy: {np.__version__}")
    print("\n Config (from pipeline_config.py - v7 best):")
    print(f" TF-IDF: {path_a_cfg['tfidf_analyzer']} ngram{path_a_cfg['tfidf_ngram_range']}, max_features={path_a_cfg['tfidf_max_features']}, sublinear_tf={path_a_cfg.get('tfidf_sublinear_tf', False)}")
    print(f" PCA: n_components={path_a_cfg['pca_n_components']}")
    print(f" SGD-OCSVM: nu={path_a_cfg['sgd_ocsvm_nu']}, linear kernel [M1 fix]")
    print(f" IsoForest: cont={path_a_cfg.get('isolation_forest_contamination', 0.05)}, n_est={path_a_cfg.get('isolation_forest_n_estimators', 200)}, samples={path_a_cfg.get('isolation_forest_max_samples', 1024)}")
    print(f" LOF: k={path_a_cfg.get('lof_n_neighbors', 50)}, cont={path_a_cfg.get('lof_contamination', 0.05)}")
    print(f" HBOS: bins={path_a_cfg.get('hbos_n_bins', 30)}, cont={path_a_cfg.get('hbos_contamination', 0.13)}")
    print(f" PCA-Error: n_components={path_a_cfg.get('pca_error_n_components', 'auto>=70% var')}")
    print(f" Sampling: stratified_length n_strata={args.n_strata}, min_per_stratum={args.min_per_stratum} (train-only)")
    print(f" Threshold: P95 on separate calib split C2")
    print(f" Bootstrap: 200 rounds, 95% CI M4")
    print(f" NOTE: Each model type uses its own best feature/reduction config from best_config_per_model_type")

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    X_tr_raw = data["X_train_w"]
    X_va_raw = data["X_val_w"]
    X_at_raw = data["X_attack_w"]
    tid_tr   = data.get("trace_ids_train",  np.arange(len(X_tr_raw)))
    tid_va   = data.get("trace_ids_val",    np.arange(len(X_va_raw)))
    tid_at   = data.get("trace_ids_attack", np.arange(len(X_at_raw)))

    print(f"\n Raw Train:{X_tr_raw.shape} Val:{X_va_raw.shape} Atk:{X_at_raw.shape}")

    # Stratified sampling: TRAINING only (Peer Review fix)
    # sample_asym (lf=0.2) discarded 80% long traces including long attacks.
    # sample_stratified_length ensures every length bucket is equally represented.
    # Val and Attack are kept FULL for honest evaluation.
    set_seed(42)
    X_tr = sample_stratified_length(X_tr_raw, tid_tr,
                                     n_strata=args.n_strata,
                                     min_per_stratum=args.min_per_stratum)
    X_va = X_va_raw   # full val set no sampling
    X_at = X_at_raw   # full attack set no sampling
    print(f" Sampling Train: {len(X_tr_raw):,} to {len(X_tr):,} windows retained for training")
    print(f" After sampling Train:{X_tr.shape} Val:{X_va.shape} Atk:{X_at.shape}")

    # C2: Split val into calibration (20%) + test (80%)
    n_calib = int(0.2 * len(X_va))
    perm    = np.random.permutation(len(X_va))
    X_calib_w = X_va[perm[:n_calib]]
    X_test_normal_w = X_va[perm[n_calib:]]

    print(f" Val split Calib:{X_calib_w.shape} (threshold) | "
          f"Test-normal:{X_test_normal_w.shape} (eval)")

    # Stringify once for per model feature extraction
    print("\n [Feature Extraction]")
    t0 = time.time()
    str_tr = stringify_windows(X_tr)
    str_calib = stringify_windows(X_calib_w)
    str_test_n = stringify_windows(X_test_normal_w)
    str_at = stringify_windows(X_at)
    print(f" Stringified windows for per-model feature configs: {time.time()-t0:.1f}s")

    # Assemble test labels: test-normal (0) + attack (1)
    y_test = np.concatenate([np.zeros(len(X_test_normal_w)),
                             np.ones(len(X_at))])
    atk_ratio = float(len(X_at)) / float(len(y_test))
    print(f" Test set: {len(y_test)} samples "
          f"(normal={len(X_test_normal_w)}, attack={len(X_at)}, "
          f"atk_ratio={atk_ratio:.4f})")

    # Run models
    print("\n" + "" * 75)
    best_cfgs = get_path_a_best_configs(args.dataset)
    results, models, all_ci, selected_artifact = run_all_models(
        str_tr, str_calib, str_test_n, str_at,
        y_test, atk_ratio, path_a_cfg, best_cfgs
    )

    # Save results
    out_file = os.path.join(log_dir, "path_a_results.json")
    legacy_out_file = os.path.join(log_dir, "path_a_v6_reviewed.json")
    selected_reducer = selected_artifact.get("reducer") if selected_artifact else None
    selected_pca_var = None
    if selected_reducer is not None and hasattr(selected_reducer, "explained_variance_ratio_"):
        selected_pca_var = float(np.sum(selected_reducer.explained_variance_ratio_))
    save_data = {
        "config": {
            "version": "v7 CONFIG FROM pipeline_config.py",
            "tfidf": f"{path_a_cfg['tfidf_analyzer']} ngram{path_a_cfg['tfidf_ngram_range']} max_features={path_a_cfg['tfidf_max_features']} sublinear_tf={path_a_cfg.get('tfidf_sublinear_tf', False)}",
            "ocsvm": f"SGDOneClassSVM nu={path_a_cfg['sgd_ocsvm_nu']} linear",
            "pca_n_components": path_a_cfg['pca_n_components'],
            "pca_var_explained": round(selected_pca_var, 4) if selected_pca_var is not None else None,
            "sampling": f"stratified_length n_strata={args.n_strata} min_per_stratum={args.min_per_stratum} (train-only)",
            "calib_split_ratio": 0.2,
            "bootstrap_n": 200,
            "pyod_version": PYOD_VERSION,
            "best_config_per_model_type": best_cfgs,
        },
        "results": results,
        "bootstrap_95ci": all_ci,
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=4)
    # Backward compatible legacy filename
    with open(legacy_out_file, "w") as f:
        json.dump(save_data, f, indent=4)

    # Summary table
    metrics_primary = ["AUC_ROC", "AUPR", "F1_calib", "FPR_at_TPR95"]
    models_list     = ["SGD-OCSVM", "IsolationForest", "LOF", "PCA-Error", "HBOS"]

    print(f"\n{'='*75}")
    print("FINAL RESULTS (v6 all peer review fixes)")
    print(f"{'='*75}")
    print(f" {'Model':<18}" + "".join(f" {c:<14}" for c in metrics_primary))
    print(" " + "-" * (18 + 16 * len(metrics_primary)))
    for mn in models_list:
        r = results.get(mn, {})
        row = f" {mn:<18}"
        for c in metrics_primary:
            v = r.get(c, float("nan"))
            row += f" {v:<14.4f}" if not np.isnan(v) else f" {'N/A':<14}"
        if "WARNING" in r:
            row += " FAILED"
        print(row)
    print(f" {'Random':<18} {'0.5000':<14} {atk_ratio:<14.4f} {'---':<14} {'---':<14}")

    # Bootstrap CI table
    print(f"\n 95% Confidence Intervals:")
    for mn in models_list:
        ci = all_ci.get(mn)
        if ci:
            print(f" {mn:<18} {fmt_ci(ci)}")

    # Averages (excluding OC SVM if failed)
    valid_models = [mn for mn in models_list
                    if mn in results and results[mn].get("AUC_ROC", 0) >= 0.5]
    if valid_models:
        avg_auc  = np.mean([results[mn]["AUC_ROC"]  for mn in valid_models])
        avg_aupr = np.mean([results[mn]["AUPR"]     for mn in valid_models])
        avg_f1c  = np.mean([results[mn]["F1_calib"] for mn in valid_models])
        best_m   = max(valid_models, key=lambda mn: results[mn]["AUC_ROC"])
        print(f"\n Avg (valid models, excl. AUC<0.5):")
        print(f" AUC-ROC: {avg_auc:.4f}")
        print(f" AUPR: {avg_aupr:.4f}")
        print(f" F1-calib: {avg_f1c:.4f}")
        print(f" Best: {best_m} (AUC={results[best_m]['AUC_ROC']:.4f})")

    # Save selected feature pipeline + all trained model objects
    model_files = path_a_cfg.get("model_files", {})
    save_objs = {
        model_files.get("vectorizer", "vec.pkl"): selected_artifact.get("vectorizer") if selected_artifact else None,
        model_files.get("pca", "pca.pkl"): selected_artifact.get("reducer") if selected_artifact else None,
        "pca.pkl": selected_artifact.get("reducer") if selected_artifact else None,  # backward compatibility alias
        model_files.get("sgd_ocsvm", model_files.get("classifier", "sgd_ocsvm.pkl")): models.get("sgd_ocsvm"),
        model_files.get("isolation_forest", "ifo.pkl"): models.get("ifo"),
        model_files.get("lof", "lof.pkl"): models.get("lof"),
        model_files.get("hbos", "hbos.pkl"): models.get("hbos"),
        model_files.get("pca_error", "pca_err.pkl"): models.get("pca_err"),
    }
    # Filter out None values
    save_objs = {k: v for k, v in save_objs.items() if v is not None}

    for name, obj in save_objs.items():
        with open(os.path.join(model_dir, name), "wb") as f:
            pickle.dump(obj, f)

    best_cfg_file = model_files.get("best_config_per_model_type", "path_a_best_config_per_model_type.json")
    with open(os.path.join(model_dir, best_cfg_file), "w", encoding="utf-8") as f:
        json.dump(get_path_a_best_configs(args.dataset), f, indent=2)

    print(f"\n[] Results {out_file}")
    print(f"[] Legacy {legacy_out_file}")
    print(f"[] Models ({len(save_objs)}) {model_dir}")
    for name in save_objs:
        print(f" {name}")


if __name__ == "__main__":
    main()

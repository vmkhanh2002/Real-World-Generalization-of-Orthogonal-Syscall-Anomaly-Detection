"""
Path A optimization sweep (historical provenance script)

This script preserves the broad Path A grid search used to select the current
32-bit defaults. It is useful for search-space archaeology and optional replay,
but it is not the authoritative source for the latest paper-facing metrics.

Current 32-bit active reference:
  - model: SGD-OCSVM
  - feature: word(1,2), max_features=1000, sublinear_tf=True
  - reduction: PCA(100)
  - hyperparameter: nu=0.01
  - refreshed holdout metrics: AUC 0.7975, AUPR 0.3702, F1_calib 0.2892
  - source of truth: experiments/32bit/logs/path_a_results.json

This script still writes the historical sweep artifact:
  - experiments/<dataset>/logs/path_a_v7_optimization.json

Axes tested:
  Axis 1 (Feature Engineering):
  - max_features: [1000, 2000, 3000]
  - sublinear_tf: [False, True]
  - analyzer/ngram: word(1,2), word(1,3), char_wb(3,3), char_wb(3,5)

  Axis 2 (Dimensionality Reduction):
  - PCA(50), PCA(100), TruncatedSVD(50), TruncatedSVD(100)

  Axis 3 (Model Tuning):
  - SGD-OCSVM nu: [0.01, 0.03, 0.05, 0.08, 0.10]
  - HBOS n_bins: [20, 30, 50, 80]
"""

import os, pickle, sys, json, time, random, itertools
import numpy as np

# Single thread BLAS for reproducibility
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
# Metrics (same as v6)
#
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
    p, r, thr = precision_recall_curve(y_true, scores)
    f1_arr = 2 * p * r / (p + r + 1e-8)
    f1_dbg = float(f1_arr[np.argmax(f1_arr)])
    thr_95   = np.percentile(scores_calib, 95)
    y_pred   = (scores >= thr_95).astype(int)
    f1_cal   = f1_score(y_true, y_pred, zero_division=0)
    fpr95    = compute_fpr_at_tpr95(y_true, scores)
    return {
    "AUC_ROC": round(auc_roc, 4),
    "AUPR": round(aupr, 4),
    "F1_debug": round(f1_dbg, 4),
    "F1_calib": round(f1_cal, 4),
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
    "lo": round(np.percentile(arr, ci_lo), 4),
    "hi": round(np.percentile(arr, ci_hi), 4),
        }
    return out

def reconstruction_error_batched(pca_model, X, batch_size=50000):
    """MSE reconstruction error in batches handles both dense and sparse X."""
    from scipy import sparse as sp
    n = X.shape[0]
    errors = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        recon = pca_model.inverse_transform(pca_model.transform(chunk))
        errors[start:end] = np.mean((chunk - recon) ** 2, axis=1)
    return errors


#
# Helpers
#
def stringify_windows(windows):
    return [" ".join(map(str, row)) for row in windows]

def sample_stratified_length(matrix, trace_ids, n_strata=5, min_per_stratum=200):
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

def fmt(m):
    return (f"AUC={m['AUC_ROC']:.4f}  AUPR={m['AUPR']:.4f}  "
    f"F1-cal={m['F1_calib']:.4f} FPR95={m['FPR_at_TPR95']:.4f}")


#
# Feature Extraction Configs (Axis 1)
#
FEATURE_CONFIGS = [
    # Baseline (v6)
    {"name": "word(1,2)_1k",        "analyzer": "word", "ngram_range": (1,2),
    "max_features": 1000, "sublinear_tf": False,
    "token_pattern": r"(=u)\b\w+\b"},
    # Sublinear TF on baseline
    {"name": "word(1,2)_1k_sub",    "analyzer": "word", "ngram_range": (1,2),
    "max_features": 1000, "sublinear_tf": True,
    "token_pattern": r"(=u)\b\w+\b"},
    # max_features=2000
    {"name": "word(1,2)_2k_sub",    "analyzer": "word", "ngram_range": (1,2),
    "max_features": 2000, "sublinear_tf": True,
    "token_pattern": r"(=u)\b\w+\b"},
    # max_features=3000
    {"name": "word(1,2)_3k_sub",    "analyzer": "word", "ngram_range": (1,2),
    "max_features": 3000, "sublinear_tf": True,
    "token_pattern": r"(=u)\b\w+\b"},
    # Wider word n-grams
    {"name": "word(1,3)_2k_sub",    "analyzer": "word", "ngram_range": (1,3),
    "max_features": 2000, "sublinear_tf": True,
    "token_pattern": r"(=u)\b\w+\b"},
    # Char-level (3,3)
    {"name": "charwb(3,3)_2k_sub",  "analyzer": "char_wb", "ngram_range": (3,3),
    "max_features": 2000, "sublinear_tf": True,
    "token_pattern": None},
    # Char-level (3,5)
    {"name": "charwb(3,5)_2k_sub",  "analyzer": "char_wb", "ngram_range": (3,5),
    "max_features": 2000, "sublinear_tf": True,
    "token_pattern": None},
]

#
# Dimensionality Reduction Configs (Axis 2)
#
REDUCE_CONFIGS = [
    {"name": "PCA(50)",    "method": "pca",  "n_components": 50},
    {"name": "PCA(100)",   "method": "pca",  "n_components": 100},
    {"name": "SVD(50)",    "method": "svd",  "n_components": 50},
    {"name": "SVD(100)",   "method": "svd",  "n_components": 100},
]

#
# Model Hyper params (Axis 3)
#
SGD_NU_VALUES = [0.01, 0.03, 0.05, 0.08, 0.10]
HBOS_NBINS_VALUES = [20, 30, 50, 80]


#
# Main Experiment
#
def main():
    set_seed(42)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(project_root, "data", "processed",
                             "32bit", "phase1_base_arrays.pkl")
    log_dir   = os.path.join(project_root, "experiments", "32bit", "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print("PATH A OPTIMIZATION GRID SEARCH (v7-experiment)")
    print("=" * 80)

    # Load data
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    X_tr_raw = data["X_train_w"]
    X_va_raw = data["X_val_w"]
    X_at_raw = data["X_attack_w"]
    tid_tr   = data.get("trace_ids_train",  np.arange(len(X_tr_raw)))

    print(f"\n Raw Train:{X_tr_raw.shape} Val:{X_va_raw.shape} Atk:{X_at_raw.shape}")

    # Sampling (same as v6)
    set_seed(42)
    X_tr = sample_stratified_length(X_tr_raw, tid_tr, n_strata=5, min_per_stratum=200)
    X_va = X_va_raw
    X_at = X_at_raw
    print(f" Train after sampling: {X_tr.shape}")

    # Calibration/Test split (same as v6)
    set_seed(42)
    n_calib = int(0.2 * len(X_va))
    perm    = np.random.permutation(len(X_va))
    X_calib_w       = X_va[perm[:n_calib]]
    X_test_normal_w = X_va[perm[n_calib:]]
    print(f" Calib:{X_calib_w.shape} Test-normal:{X_test_normal_w.shape} Attack:{X_at.shape}")

    # Stringify once
    str_tr    = stringify_windows(X_tr)
    str_calib = stringify_windows(X_calib_w)
    str_test_n= stringify_windows(X_test_normal_w)
    str_at    = stringify_windows(X_at)

    # Experiment loop
    all_results = []
    exp_id = 0
    total_configs = len(FEATURE_CONFIGS) * len(REDUCE_CONFIGS)
    print(f"\n Total featurereduction combos: {total_configs}")
    print(f" Per combo: {len(SGD_NU_VALUES)} SGD-OCSVM + IsoForest + LOF + PCA-Error + {len(HBOS_NBINS_VALUES)} HBOS")
    print()

    for feat_cfg in FEATURE_CONFIGS:
        # TF IDF vectorization
        t_feat = time.time()
        vec_kwargs = {
        "max_features": feat_cfg["max_features"],
    "analyzer": feat_cfg["analyzer"],
    "ngram_range": feat_cfg["ngram_range"],
    "sublinear_tf": feat_cfg["sublinear_tf"],
        }
        if feat_cfg["token_pattern"] is not None:
            vec_kwargs["token_pattern"] = feat_cfg["token_pattern"]

        vec = TfidfVectorizer(**vec_kwargs)
        X_tr_tfidf    = vec.fit_transform(str_tr)
        X_calib_tfidf = vec.transform(str_calib)
        X_test_n_tf   = vec.transform(str_test_n)
        X_at_tfidf    = vec.transform(str_at)

        # Convert to dense for PCA (keep sparse for SVD)
        X_tr_dense    = X_tr_tfidf.toarray()
        X_calib_dense = X_calib_tfidf.toarray()
        X_test_n_dense= X_test_n_tf.toarray()
        X_at_dense    = X_at_tfidf.toarray()

        actual_feats = X_tr_tfidf.shape[1]
        feat_time = time.time() - t_feat
        print(f" Feature: {feat_cfg['name']} (actual_dims={actual_feats}, {feat_time:.1f}s) ")

        for red_cfg in REDUCE_CONFIGS:
            t_red = time.time()
            n_comp = min(red_cfg["n_components"], actual_feats)
            if n_comp < 2:
                print(f" Skipping {red_cfg['name']}: too few features ({actual_feats})")
                continue

            # Dimensionality reduction
            if red_cfg["method"] == "pca":
                reducer = PCA(n_components=n_comp, random_state=42)
                X_tr_red    = reducer.fit_transform(X_tr_dense)
                X_calib_red = reducer.transform(X_calib_dense)
                X_test_n_red= reducer.transform(X_test_n_dense)
                X_at_red    = reducer.transform(X_at_dense)
                var_exp = reducer.explained_variance_ratio_.sum()
            else:  # svd
                reducer = TruncatedSVD(n_components=n_comp, random_state=42)
                X_tr_red    = reducer.fit_transform(X_tr_tfidf)
                X_calib_red = reducer.transform(X_calib_tfidf)
                X_test_n_red= reducer.transform(X_test_n_tf)
                X_at_red    = reducer.transform(X_at_tfidf)
                var_exp = reducer.explained_variance_ratio_.sum()

            # Assemble test sets
            y_test   = np.concatenate([np.zeros(len(X_test_n_red)),
                                       np.ones(len(X_at_red))])
            X_test_red = np.vstack([X_test_n_red, X_at_red])

            atk_ratio = float(len(X_at_red)) / float(len(y_test))

            red_label = f"{red_cfg['name']}(actual={n_comp})"
            print(f"\n {red_label} var={var_exp:.3f} ({time.time()-t_red:.1f}s) ")

            #
            # MODEL 1: SGD-OCSVM with nu grid (Axis 3)
            #
            for nu in SGD_NU_VALUES:
                set_seed(42)
                exp_id += 1
                label = f"SGD-OCSVM(nu={nu})"
                try:
                    sgd = SGDOneClassSVM(nu=nu, max_iter=1000,
                                         learning_rate='optimal',
                                         random_state=42, tol=1e-4)
                    sgd.fit(X_tr_red)
                    sc_test  = -sgd.decision_function(X_test_red)
                    sc_calib = -sgd.decision_function(X_calib_red)
                    m = compute_all_metrics(y_test, sc_test, sc_calib)
                    print(f" [{exp_id:3d}] {label:<22} {fmt(m)}")
                    all_results.append({
                    "exp_id": exp_id, "feature": feat_cfg["name"],
    "reduction": red_cfg["name"], "n_comp": n_comp,
    "var_explained": round(var_exp, 4),
    "model": label, "model_type": "SGD-OCSVM",
    "hyperparam": {"nu": nu},
                        **m
                    })
                except ConvergenceWarning:
                    print(f" [{exp_id:3d}] {label:<22} ConvergenceWarning SKIPPED")
                except Exception as e:
                    print(f" [{exp_id:3d}] {label:<22} ERROR: {e}")

            #
            # MODEL 2: IsolationForest (fixed params)
            #
            set_seed(42)
            exp_id += 1
            label = "IsoForest"
            ifo = IsolationForest(n_estimators=200, contamination=0.05,
                                  max_samples=min(1024, len(X_tr_red)),
                                  random_state=42)
            ifo.fit(X_tr_red)
            sc_test  = -ifo.score_samples(X_test_red)
            sc_calib = -ifo.score_samples(X_calib_red)
            m = compute_all_metrics(y_test, sc_test, sc_calib)
            print(f" [{exp_id:3d}] {label:<22} {fmt(m)}")
            all_results.append({
            "exp_id": exp_id, "feature": feat_cfg["name"],
    "reduction": red_cfg["name"], "n_comp": n_comp,
    "var_explained": round(var_exp, 4),
    "model": label, "model_type": "IsolationForest",
    "hyperparam": {},
                **m
            })

            #
            # MODEL 3: LOF (fixed params)
            #
            set_seed(42)
            exp_id += 1
            label = "LOF(k=50)"
            lof = LocalOutlierFactor(n_neighbors=min(50, len(X_tr_red)-1),
                                     novelty=True, contamination=0.05)
            lof.fit(X_tr_red)
            sc_test  = -lof.decision_function(X_test_red)
            sc_calib = -lof.decision_function(X_calib_red)
            m = compute_all_metrics(y_test, sc_test, sc_calib)
            print(f" [{exp_id:3d}] {label:<22} {fmt(m)}")
            all_results.append({
            "exp_id": exp_id, "feature": feat_cfg["name"],
    "reduction": red_cfg["name"], "n_comp": n_comp,
    "var_explained": round(var_exp, 4),
    "model": label, "model_type": "LOF",
    "hyperparam": {"k": 50},
                **m
            })

            #
            # MODEL 4: PCA-Error (auto-tuned, batched to avoid OOM)
            #
            set_seed(42)
            exp_id += 1
            label = "PCA-Error"
            try:
                pca_full = PCA(random_state=42)
                pca_full.fit(X_tr_dense)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                nc = int(np.searchsorted(cumvar, 0.70) + 1)
                nc = max(nc, 5)
                nc = min(nc, X_tr_dense.shape[1])
                pca_err = PCA(n_components=nc, random_state=42)
                pca_err.fit(X_tr_dense)
                # Batched scoring avoids 7+ GiB dense allocation
                from scipy import sparse as sp
                X_test_sparse = sp.vstack([X_test_n_tf, X_at_tfidf])
                err_test  = reconstruction_error_batched(pca_err, X_test_sparse)
                err_calib = reconstruction_error_batched(pca_err, X_calib_tfidf)
                m = compute_all_metrics(y_test, err_test, err_calib)
                m["pca_err_n_comp"] = nc
                print(f" [{exp_id:3d}] {label}(n={nc}){'':<10} {fmt(m)}")
                all_results.append({
                "exp_id": exp_id, "feature": feat_cfg["name"],
    "reduction": "PCA-Error-auto", "n_comp": nc,
    "var_explained": round(pca_err.explained_variance_ratio_.sum(), 4),
    "model": label, "model_type": "PCA-Error",
    "hyperparam": {"n_components": nc},
                    **m
                })
            except Exception as e:
                print(f" [{exp_id:3d}] {label:<22} ERROR: {e}")

            #
            # MODEL 5: HBOS with n_bins grid (Axis 3)
            #
            if HBOS is not None:
                for nb in HBOS_NBINS_VALUES:
                    set_seed(42)
                    exp_id += 1
                    label = f"HBOS(bins={nb})"
                    try:
                        hbos = HBOS(n_bins=nb, contamination=0.13)
                        hbos.fit(X_tr_red)
                        sc_test  = hbos.decision_function(X_test_red)
                        sc_calib = hbos.decision_function(X_calib_red)
                        m = compute_all_metrics(y_test, sc_test, sc_calib)
                        print(f" [{exp_id:3d}] {label:<22} {fmt(m)}")
                        all_results.append({
                        "exp_id": exp_id, "feature": feat_cfg["name"],
    "reduction": red_cfg["name"], "n_comp": n_comp,
    "var_explained": round(var_exp, 4),
    "model": label, "model_type": "HBOS",
    "hyperparam": {"n_bins": nb},
                            **m
                        })
                    except Exception as e:
                        print(f" [{exp_id:3d}] {label:<22} ERROR: {e}")

    #
    # Find best configs
    #
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)

    # Sort by AUC_ROC
    sorted_by_auc = sorted(all_results, key=lambda x: x.get("AUC_ROC", 0), reverse=True)
    sorted_by_aupr = sorted(all_results, key=lambda x: x.get("AUPR", 0), reverse=True)
    sorted_by_f1 = sorted(all_results, key=lambda x: x.get("F1_calib", 0), reverse=True)
    sorted_by_fpr = sorted(all_results, key=lambda x: x.get("FPR_at_TPR95", 1))

    print(f"\n Total experiments: {len(all_results)}")

    print(f"\n TOP 10 by AUC-ROC ")
    print(f" {'Rank':>4} {'Feature':<22} {'Reduction':<12} {'Model':<22} "
    f"{'AUC':>7} {'AUPR':>7} {'F1-cal':>7} {'FPR95':>7}")
    print(f" " + "-" * 95)
    for i, r in enumerate(sorted_by_auc[:10]):
        print(f" {i+1:4d} {r['feature']:<22} {r['reduction']:<12} {r['model']:<22} "
    f"{r['AUC_ROC']:7.4f} {r['AUPR']:7.4f} {r['F1_calib']:7.4f} {r['FPR_at_TPR95']:7.4f}")

    print(f"\n TOP 10 by AUPR ")
    print(f" {'Rank':>4} {'Feature':<22} {'Reduction':<12} {'Model':<22} "
    f"{'AUC':>7} {'AUPR':>7} {'F1-cal':>7} {'FPR95':>7}")
    print(f" " + "-" * 95)
    for i, r in enumerate(sorted_by_aupr[:10]):
        print(f" {i+1:4d} {r['feature']:<22} {r['reduction']:<12} {r['model']:<22} "
    f"{r['AUC_ROC']:7.4f} {r['AUPR']:7.4f} {r['F1_calib']:7.4f} {r['FPR_at_TPR95']:7.4f}")

    print(f"\n TOP 10 by F1-calib ")
    print(f" {'Rank':>4} {'Feature':<22} {'Reduction':<12} {'Model':<22} "
    f"{'AUC':>7} {'AUPR':>7} {'F1-cal':>7} {'FPR95':>7}")
    print(f" " + "-" * 95)
    for i, r in enumerate(sorted_by_f1[:10]):
        print(f" {i+1:4d} {r['feature']:<22} {r['reduction']:<12} {r['model']:<22} "
    f"{r['AUC_ROC']:7.4f} {r['AUPR']:7.4f} {r['F1_calib']:7.4f} {r['FPR_at_TPR95']:7.4f}")

    print(f"\n TOP 10 by FPR@TPR95 (lowest is best) ")
    print(f" {'Rank':>4} {'Feature':<22} {'Reduction':<12} {'Model':<22} "
    f"{'AUC':>7} {'AUPR':>7} {'F1-cal':>7} {'FPR95':>7}")
    print(f" " + "-" * 95)
    for i, r in enumerate(sorted_by_fpr[:10]):
        print(f" {i+1:4d} {r['feature']:<22} {r['reduction']:<12} {r['model']:<22} "
    f"{r['AUC_ROC']:7.4f} {r['AUPR']:7.4f} {r['F1_calib']:7.4f} {r['FPR_at_TPR95']:7.4f}")

    # Bootstrap CI for top 1 overall
    best_overall = sorted_by_auc[0]
    print(f"\n BEST OVERALL (by AUC-ROC) ")
    print(f" Feature: {best_overall['feature']}")
    print(f" Reduction: {best_overall['reduction']} (n={best_overall['n_comp']})")
    print(f" Model: {best_overall['model']}")
    print(f" AUC-ROC: {best_overall['AUC_ROC']:.4f}")
    print(f" AUPR: {best_overall['AUPR']:.4f}")
    print(f" F1-calib: {best_overall['F1_calib']:.4f}")
    print(f" FPR@TPR95: {best_overall['FPR_at_TPR95']:.4f}")

    # Per model type best
    print(f"\n Best config per model type ")
    for mtype in ["SGD-OCSVM", "IsolationForest", "LOF", "PCA-Error", "HBOS"]:
        subset = [r for r in all_results if r["model_type"] == mtype]
        if not subset:
            continue
        best = max(subset, key=lambda x: x["AUC_ROC"])
        print(f" {mtype:<15} feature={best['feature']:<22} "
    f"reduce={best['reduction']:<12} "
    f"AUC={best['AUC_ROC']:.4f} AUPR={best['AUPR']:.4f} "
    f"F1={best['F1_calib']:.4f} {best.get('hyperparam',{})}")

    # Baseline vs Best comparison
    baseline_results = [r for r in all_results
                        if r["feature"] == "word(1,2)_1k"
                        and r["reduction"] == "PCA(50)"]
    if baseline_results:
        best_baseline = max(baseline_results, key=lambda x: x["AUC_ROC"])
        delta_auc  = best_overall["AUC_ROC"] - best_baseline["AUC_ROC"]
        delta_aupr = best_overall["AUPR"]    - best_baseline.get("AUPR", 0)
        delta_f1   = best_overall["F1_calib"] - best_baseline.get("F1_calib", 0)
        print(f"\n Improvement over v6 baseline ")
    print(f" Baseline: feature={best_baseline['feature']}, "
    f"reduce={best_baseline['reduction']}, model={best_baseline['model']}")
    print(f" AUC-ROC: {best_baseline['AUC_ROC']:.4f} {best_overall['AUC_ROC']:.4f} "
    f"( = {delta_auc:+.4f})")
    print(f" AUPR: {best_baseline.get('AUPR',0):.4f} {best_overall['AUPR']:.4f} "
    f"( = {delta_aupr:+.4f})")
    print(f" F1-calib: {best_baseline.get('F1_calib',0):.4f} {best_overall['F1_calib']:.4f} "
    f"( = {delta_f1:+.4f})")

    # Save all results
    out_file = os.path.join(log_dir, "path_a_v7_optimization.json")
    save_data = {
    "experiment": "Path A v7 Optimization Grid Search",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_experiments": len(all_results),
    "best_by_auc": sorted_by_auc[0] if sorted_by_auc else None,
    "best_by_aupr": sorted_by_aupr[0] if sorted_by_aupr else None,
    "best_by_f1": sorted_by_f1[0] if sorted_by_f1 else None,
    "best_by_fpr95": sorted_by_fpr[0] if sorted_by_fpr else None,
    "all_results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=4, default=str)
        print(f"\n[] Full results saved {out_file}")
    print(f"[] Total experiments completed: {len(all_results)}")


if __name__ == "__main__":
    main()

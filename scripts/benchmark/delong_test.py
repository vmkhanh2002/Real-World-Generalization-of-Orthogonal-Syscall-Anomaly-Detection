"""
DeLong test for comparing two correlated ROC AUCs (vectorized, memory-safe).
Reference: DeLong ER, DeLong DM, Clarke-Pearson DL (1988).
Approach: row-wise computation to avoid (n_pos, n_neg) full broadcast.
"""
import numpy as np
from scipy import stats


def _fast_auc(pos_s, neg_s):
    """Compute AUC without full (n_pos, n_neg) broadcast."""
    n_pos = len(pos_s)
    n_neg = len(neg_s)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Sort neg_scores for binary search
    neg_sorted = np.sort(neg_s)
    # For each pos: count neg < pos + 0.5 * count neg == pos
    indices = np.searchsorted(neg_sorted, pos_s, side="left")
    eq_counts = np.searchsorted(neg_sorted, pos_s, side="right") - indices
    vals = (indices + 0.5 * eq_counts) / n_neg
    return float(np.mean(vals))


def _placement_values(pos_s, neg_s):
    """V10[i] = psi(pos_i, neg) for each positive; V01[j] = psi(pos, neg_j) for each negative."""
    n_pos = len(pos_s)
    n_neg = len(neg_s)
    neg_sorted = np.sort(neg_s)
    pos_sorted = np.sort(pos_s)

    # V10: for each pos, fraction of neg it beats
    V10 = np.empty(n_pos)
    for i in range(n_pos):
        lt = np.searchsorted(neg_sorted, pos_s[i], side="left")
        eq = np.searchsorted(neg_sorted, pos_s[i], side="right") - lt
        V10[i] = (lt + 0.5 * eq) / n_neg

    # V01: for each neg, fraction of pos that beats it
    V01 = np.empty(n_neg)
    for j in range(n_neg):
        gt = n_pos - np.searchsorted(pos_sorted, neg_s[j], side="right")
        eq = np.searchsorted(pos_sorted, neg_s[j], side="right") - np.searchsorted(pos_sorted, neg_s[j], side="left")
        V01[j] = (gt + 0.5 * eq) / n_pos

    return V10, V01


def delong_test(y_true, scores_1, scores_2):
    """
    Compare two correlated ROC AUCs using DeLong's test (memory-safe).

    Args:
        y_true: Binary ground truth (0/1), shape (n,)
        scores_1: Anomaly scores from model 1 (higher = more anomalous), shape (n,)
        scores_2: Anomaly scores from model 2 (higher = more anomalous), shape (n,)

    Returns:
        dict with auc_1, auc_2, auc_diff, z_statistic, p_value, ci_lower, ci_upper, n_pos, n_neg
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_1 = np.asarray(scores_1, dtype=float)
    scores_2 = np.asarray(scores_2, dtype=float)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return {
            "auc_1": 0.5, "auc_2": 0.5, "auc_diff": 0.0,
            "z_statistic": 0.0, "p_value": 1.0,
            "ci_lower": 0.0, "ci_upper": 0.0,
            "n_pos": n_pos, "n_neg": n_neg, "note": "Only one class present"
        }

    pos_scores_1 = scores_1[pos_mask]
    neg_scores_1 = scores_1[neg_mask]
    pos_scores_2 = scores_2[pos_mask]
    neg_scores_2 = scores_2[neg_mask]

    auc_1 = _fast_auc(pos_scores_1, neg_scores_1)
    auc_2 = _fast_auc(pos_scores_2, neg_scores_2)
    auc_diff = auc_1 - auc_2

    V10_1, V01_1 = _placement_values(pos_scores_1, neg_scores_1)
    V10_2, V01_2 = _placement_values(pos_scores_2, neg_scores_2)

    S10 = np.cov(np.column_stack([V10_1, V10_2]), rowvar=False)
    S01 = np.cov(np.column_stack([V01_1, V01_2]), rowvar=False)

    var_diff = (S10[0, 0] - 2 * S10[0, 1] + S10[1, 1]) / n_pos + \
               (S01[0, 0] - 2 * S01[0, 1] + S01[1, 1]) / n_neg

    if var_diff <= 0:
        var_diff = 1e-10

    se = np.sqrt(var_diff)
    z_stat = auc_diff / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return {
        "auc_1": round(float(auc_1), 6),
        "auc_2": round(float(auc_2), 6),
        "auc_diff": round(float(auc_diff), 6),
        "z_statistic": round(float(z_stat), 4),
        "p_value": float(p_value),
        "ci_lower": round(float(auc_diff - 1.96 * se), 6),
        "ci_upper": round(float(auc_diff + 1.96 * se), 6),
        "n_pos": n_pos, "n_neg": n_neg,
    }

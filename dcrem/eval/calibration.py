"""Open-set score calibration: distance-based scoring with/without calibration.

Two approaches:
  1. Distance-based (replicates CREM): max-distance-to-reciprocal-point → top-K mean
  2. Calibrator-based (optional): MLP calibrator maps distances → scalar score

Both support K-search for optimal top-K.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def calibrate_by_distance(distances, K=3):
    """CREM-style calibration: top-K mean of per-label distances.

    distances : (N, q) array   per-sample, per-label distances to p_k
    K         : int             top-K aggregation

    Returns (N,) scores (higher = more likely known = farther from reciprocal pts).
    """
    sorted_dist = np.sort(distances, axis=1)[:, ::-1]
    return sorted_dist[:, :K].mean(axis=1)


def k_search_score(distances, osr_labels, ks=None):
    """Find best K for distance-based scoring.

    Returns (best_score, best_K, all_results).
    """
    N, q = distances.shape
    if ks is None:
        ks = list(range(1, q))
    best_auroc, best_K, all_res = -1.0, 1, {}
    for K in ks:
        scores = calibrate_by_distance(distances, K)
        score_neg = -scores
        try:
            auroc = float(roc_auc_score(osr_labels, score_neg))
        except Exception:
            auroc = float("nan")
        all_res[K] = auroc
        if not np.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_K = K
    return calibrate_by_distance(distances, best_K), best_K, all_res

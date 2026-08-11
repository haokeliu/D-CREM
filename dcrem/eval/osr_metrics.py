"""Open-set recognition metrics: AUROC, AUPR.

Reuses sklearn's roc_auc_score and average_precision_score.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_osr_metrics(scores, osr_labels):
    """Compute AUROC and AUPR for open-set detection.

    Parameters
    ----------
    scores     : (N,) array   open-set scores (higher = more likely known)
    osr_labels : (N,) array   1 = contains unknown labels, 0 = all known

    Returns dict with AUROC, AUPR.
    """
    # OSR: low score → unknown.  sklearn expects high score = positive.
    # We treat "contains unknown" as positive, so we negate the score.
    score_neg = -np.asarray(scores, dtype=float)
    lbl = np.asarray(osr_labels, dtype=int)
    mask = ~np.isnan(score_neg) & ~np.isnan(lbl)
    if mask.sum() < 2 or len(np.unique(lbl[mask])) < 2:
        return {"AUROC": float("nan"), "AUPR": float("nan")}
    return {
        "AUROC": float(roc_auc_score(lbl[mask], score_neg[mask])),
        "AUPR": float(average_precision_score(lbl[mask], score_neg[mask])),
    }


def candidate_ks(num_classes):
    """Shared CREM/D-CREM K grid; use K=1 for the single-label edge case."""
    if num_classes <= 1:
        return [1]
    return list(range(1, num_classes))


def top_k_scores(distances, K):
    """Top-K mean distance; higher means farther from reciprocal points."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2:
        raise ValueError("distances must have shape (N, q)")
    if K < 1 or K > distances.shape[1]:
        raise ValueError(f"K must be in [1, {distances.shape[1]}], got {K}")
    sorted_dist = np.sort(distances, axis=1)[:, ::-1]
    return sorted_dist[:, :K].mean(axis=1)


def evaluate_fixed_k(distances, osr_labels, K):
    """Evaluate a K selected without access to these labels."""
    result = compute_osr_metrics(top_k_scores(distances, K), osr_labels)
    return {**result, "best_K": int(K)}


def k_search_osr(distances, osr_labels, ks=None):
    """Select K on a validation fold using top-K mean distance.

    Do not call this function with test labels in final experiments.  Test
    metrics must be computed with :func:`evaluate_fixed_k`.

    Parameters
    ----------
    distances  : (N, q) array    per-sample per-label distances to reciprocal points
    osr_labels : (N,) array      1 = contains unknown labels
    ks         : list | None     K values to try; default [1, ..., q−1]

    Returns (best_result, k_detail).
    """
    N, q = distances.shape
    if ks is None:
        ks = candidate_ks(q)

    best_auroc = -1.0
    best_result = {}
    k_detail = {}
    for K in ks:
        # Top-K mean: higher = farther from reciprocal points = more likely known
        score = top_k_scores(distances, K)
        res = compute_osr_metrics(score, osr_labels)
        k_detail[K] = (res["AUROC"], res["AUPR"])
        if not np.isnan(res["AUROC"]) and res["AUROC"] > best_auroc:
            best_auroc = res["AUROC"]
            best_result = {"AUROC": res["AUROC"], "AUPR": res["AUPR"], "best_K": K}
    if not best_result:
        fallback = min(3, q)
        best_result = {"AUROC": float("nan"), "AUPR": float("nan"),
                       "best_K": int(fallback)}
    return best_result, k_detail

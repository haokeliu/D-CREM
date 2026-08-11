"""Testing / open-set recognition, port of CREM/CREM_test.m."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import metrics
from .firth import FirthLogisticRegression


def _fit_open_set_calibrators(train_target, model, Ktr):
    """Fit per-label Firth calibrators on training outputs only."""
    q = train_target.shape[0]
    Ytr = train_target.T
    P = model["P"]
    models_pos, models_neg = [], []
    for k in range(q):
        output_tr = Ktr @ P[:, k]
        ypos = (Ytr[:, k] == 1).astype(float)
        yneg = (Ytr[:, k] != 1).astype(float)
        models_pos.append(FirthLogisticRegression().fit(output_tr, ypos))
        models_neg.append(FirthLogisticRegression().fit(-output_tr, yneg))
    return models_pos, models_neg


def _predict_open_set_confidences(calibrators, model, K_eval):
    """Apply training-fitted calibrators to a validation or test fold."""
    models_pos, models_neg = calibrators
    P = model["P"]
    n_eval = K_eval.shape[0]
    q = P.shape[1]
    conf = np.zeros((n_eval, q))
    for k in range(q):
        output = K_eval @ P[:, k]
        conf[:, k] = np.maximum(
            models_pos[k].predict_proba(output),
            models_neg[k].predict_proba(-output),
        )
    return conf


def _compute_open_set_confidences(train_target, model, Ktr, Kte):
    """Backward-compatible fit-and-predict helper.

    Returns (conf_te, osr_labels) where conf_te is (nte, q) — each entry is
    max(positive_confidence, negative_confidence) for that label.
    """
    calibrators = _fit_open_set_calibrators(train_target, model, Ktr)
    return _predict_open_set_confidences(calibrators, model, Kte)


def _score_from_conf(conf_te, osr_labels, K):
    """Compute open-set AUROC/AUPR from a confidence matrix for a given K."""
    sort_conf_te = np.sort(conf_te, axis=1)[:, ::-1]
    score = sort_conf_te[:, :K].mean(axis=1)
    labels = np.asarray(osr_labels).ravel()
    if labels.size < 2 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    auroc = roc_auc_score(labels, -score)
    aupr = average_precision_score(labels, -score)
    return auroc, aupr


def crem_test(train_target, test_target, osr_labels, model, Ktr, Kte, param,
              verbose=True):
    """Standard CREM test (backward compatible).  Uses param['K']."""
    return _crem_test_impl(train_target, test_target, osr_labels, model,
                           Ktr, Kte, param, verbose)


def crem_test_with_k_search(train_target, test_target, osr_labels, model,
                            Ktr, Kte, param, verbose=True):
    """Legacy oracle K-search on the supplied evaluation labels.

    Retained only to reproduce the historical MATLAB protocol.  New
    experiments must use :func:`crem_validate_and_test`.
    """
    q = train_target.shape[0]
    ks_to_try = [1] if q <= 1 else list(range(1, q))

    # --- closed-set metrics (K-independent) ---
    base_result = _compute_closed_set(test_target, model, Ktr, Kte, param, verbose)

    # --- open-set confidences (K-independent) ---
    if verbose:
        print("Fitting Firth calibrators for K-search...")
    conf_te = _compute_open_set_confidences(train_target, model, Ktr, Kte)
    osr = np.asarray(osr_labels).ravel()

    # --- grid search over K ---
    best_auroc = -1.0
    best_result = dict(base_result)
    k_detail = {}
    for K in ks_to_try:
        auroc, aupr = _score_from_conf(conf_te, osr, K)
        k_detail[K] = (auroc, aupr)
        if auroc > best_auroc:
            best_auroc = auroc
            best_result["AUROC"] = auroc
            best_result["AUPR"] = aupr
            best_result["best_K"] = K

    if verbose:
        print(f"K-search: best K={best_result['best_K']} "
              f"(AUROC={best_auroc:.4f})")

    return best_result, k_detail


def crem_validate_and_test(train_target, val_target, test_target,
                           val_osr_labels, test_osr_labels, model,
                           Ktr, Kval, Kte, param, verbose=True, ks=None):
    """Select K on validation and evaluate the locked K on test.

    Firth models are fit once on training outputs.  Validation labels are used
    only for K selection; test labels are used only for the final metrics.
    """
    if val_target.shape[1] != Kval.shape[0]:
        raise ValueError("validation target/kernel sample counts do not match")
    selection, calibrators = crem_select_k_on_validation(
        train_target, val_osr_labels, model, Ktr, Kval, param, ks=ks)
    best_k = selection["selected_K"]
    result = crem_test_fixed_k(
        test_target, test_osr_labels, model, Ktr, Kte, param,
        calibrators, best_k, verbose=verbose)
    if verbose:
        print(f"Validation selected K={best_k}; test AUROC={result['AUROC']:.4f}")
    return result, selection


def crem_select_k_on_validation(train_target, val_osr_labels, model,
                                Ktr, Kval, param, ks=None):
    """Fit calibrators on train and choose K using validation labels only."""
    q = train_target.shape[0]
    if ks is None:
        ks = [1] if q <= 1 else list(range(1, q))
    calibrators = _fit_open_set_calibrators(train_target, model, Ktr)
    conf_val = _predict_open_set_confidences(calibrators, model, Kval)
    val_osr = np.asarray(val_osr_labels).ravel()
    best_k, best_val_auroc = None, -1.0
    k_detail = {}
    for K in ks:
        try:
            val_auroc, val_aupr = _score_from_conf(conf_val, val_osr, K)
        except ValueError:
            val_auroc, val_aupr = float("nan"), float("nan")
        k_detail[K] = (val_auroc, val_aupr)
        if np.isfinite(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc, best_k = val_auroc, K
    if best_k is None:
        best_k = int(np.clip(param.get("K", 3), 1, q))
    selection = {
        "selected_K": int(best_k),
        "validation_AUROC": float(best_val_auroc),
        "validation_k_search": k_detail,
    }
    return selection, calibrators


def crem_test_fixed_k(test_target, test_osr_labels, model, Ktr, Kte, param,
                      calibrators, K, verbose=True):
    """Evaluate test data using a K already locked on validation."""
    result = _compute_closed_set(test_target, model, Ktr, Kte, param, verbose)
    conf_test = _predict_open_set_confidences(calibrators, model, Kte)
    test_auroc, test_aupr = _score_from_conf(
        conf_test, np.asarray(test_osr_labels).ravel(), K)
    result.update({"AUROC": test_auroc, "AUPR": test_aupr,
                   "best_K": int(K)})
    return result


def _compute_closed_set(test_target, model, Ktr, Kte, param, verbose=True):
    """Compute closed-set multi-label metrics (K-independent)."""
    if verbose:
        print("Computing closed-set metrics...")
    q, nte = test_target.shape

    W = model["W"]
    b = model["b"]

    Outputs = Kte @ W + np.ones((nte, 1)) @ b.reshape(1, -1)

    Out_QN = Outputs.T
    T_QN = test_target
    result = {
        "RankingLoss": metrics.ranking_loss(Out_QN, T_QN),
        "Coverage": metrics.coverage(Out_QN, T_QN),
        "OneError": metrics.one_error(Out_QN, T_QN),
        "macroAUC": metrics.macro_auc(Out_QN, T_QN),
        "AveragePrecision": metrics.average_precision(Out_QN, T_QN),
    }
    return result


def _crem_test_impl(train_target, test_target, osr_labels, model, Ktr, Kte, param,
                    verbose):
    """Original single-K path, used by crem_test()."""
    result = _compute_closed_set(test_target, model, Ktr, Kte, param, verbose)

    conf_te = _compute_open_set_confidences(train_target, model, Ktr, Kte)
    osr = np.asarray(osr_labels).ravel()
    K = param["K"]
    auroc, aupr = _score_from_conf(conf_te, osr, K)
    result["AUROC"] = auroc
    result["AUPR"] = aupr
    return result

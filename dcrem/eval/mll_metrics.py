"""Multi-label classification metrics.

Thin wrapper around crem.metrics — the faithful MATLAB ports.
All functions take outputs (Q × N) and targets (Q × N, ±1).
"""

import numpy as np


def compute_mll_metrics(outputs_QN, targets_QN):
    """Compute 5 standard multi-label metrics.

    Parameters
    ----------
    outputs_QN : (Q, N) array   model outputs (logits)
    targets_QN : (Q, N) array   ±1 ground truth

    Returns dict with macroAUC, AveragePrecision, RankingLoss, Coverage, OneError.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from crem.metrics import (
        ranking_loss, coverage, one_error, macro_auc, average_precision,
    )

    Out = np.asarray(outputs_QN, dtype=float)
    Tgt = np.asarray(targets_QN, dtype=float)

    return {
        "macroAUC": float(macro_auc(Out, Tgt)),
        "AveragePrecision": float(average_precision(Out, Tgt)),
        "RankingLoss": float(ranking_loss(Out, Tgt)),
        "Coverage": float(coverage(Out, Tgt)),
        "OneError": float(one_error(Out, Tgt)),
    }

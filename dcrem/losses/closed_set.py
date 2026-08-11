"""MSE (mean squared error) closed-set loss.

Preserves the ridge-regression spirit of CREM: the original formulation
minimises ½‖FW + 1bᵀ − Y‖² (MSE in the primal/dual space).  D-CREM keeps
the same loss form in feature space.
"""

import torch
import torch.nn.functional as F


def mse_loss(logits, targets, known_mask=None, sample_average=False):
    """½‖FW + 1bᵀ − Y‖²_F (sum over elements, matching CREM convention).

    Parameters
    ----------
    logits      : (B, q) tensor   model output before sign
    targets     : (B, q) tensor   ground truth (±1)
    known_mask  : (q,) bool tensor | None   if given, only known-label dims

    ``sample_average=True`` returns ``0.5/N * ||logits-targets||²``.  D-CREM
    training uses this form so mini-batch SGD and the averaged Sylvester
    equations have the same regularisation scale.
    """
    if known_mask is not None:
        logits = logits[:, known_mask]
        targets = targets[:, known_mask]
    loss = 0.5 * F.mse_loss(logits, targets, reduction="sum")
    if sample_average:
        loss = loss / max(1, logits.shape[0])
    return loss

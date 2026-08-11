"""Hypersphere uniformity loss.

Reference: Wang & Isola, ICML 2020 — "Understanding Contrastive
Representation Learning through the Lens of Alignment and Uniformity
on the Hypersphere."

Standard form:
  L_unif = log E_{x,y ~ batch} [exp(τ · f(x)ᵀ f(y))]

Label-aware form (alternative):
  L_unif = log E_{(x,y),(x',y')} [I(y≠y') · exp(τ · f(x)ᵀ f(x'))]

The standard form pushes ALL sample pairs apart; the label-aware form
only pushes DIFFERENT-LABEL pairs apart, preserving intra-class tightness.
"""

import torch
import torch.nn.functional as F


def uniformity_loss(features, temperature=2.0):
    """Standard uniformity loss (all pairs).

    Parameters
    ----------
    features : (B, d') tensor   ℓ₂-normalised
    temperature : float           τ — repulsion strength (default 2)

    Returns scalar loss.
    """
    B = features.size(0)
    if B < 2:
        return features.new_zeros(())
    sim = features @ features.T                           # (B, B)
    # Exclude self-similarity
    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    sim = sim[mask].view(B, B - 1)                        # (B, B-1)
    # log E[exp(τ · sim)] = logsumexp(τ·sim) − log(B−1)
    import math
    loss = torch.logsumexp(temperature * sim, dim=1).mean() - math.log(B - 1)
    return loss


def label_aware_uniformity_loss(features, labels, temperature=2.0):
    """Label-aware uniformity: only repel pairs with DIFFERENT labels.

    Parameters
    ----------
    features : (B, d') tensor   ℓ₂-normalised
    labels   : (B,) tensor      integer class labels
    temperature : float

    Returns scalar loss.

    Notes
    -----
    Intended for single-label or simplified multi-label use.  Each instance
    is assigned its majority label.  For full multi-label, the all-pairs
    approach is simpler and usually sufficient.
    """
    B = features.size(0)
    if B < 2:
        return features.new_zeros(())
    sim = features @ features.T                           # (B, B)
    # Different-label mask
    diff_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()  # (B, B)
    # Fill self with -inf so they don't contribute to logsumexp
    sim = sim.masked_fill(~diff_mask.bool(), -float("inf"))
    sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=features.device),
                          -float("inf"))
    # log E_{pairs with y_i ≠ y_j}[exp(τ · sim)]
    n_pairs = diff_mask.sum(dim=1).clamp(min=1)           # (B,)
    loss_per_sample = torch.logsumexp(temperature * sim, dim=1) - n_pairs.log().to(sim.dtype)
    return loss_per_sample.mean()

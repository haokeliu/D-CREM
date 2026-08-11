"""Conditional reciprocal-point diversity loss.

Prevents multiple stored reciprocal coefficients from collapsing
on the hypersphere.  Only penalises pairs with cos > θ_div (conditional
repulsion), preserving the natural geometric structure induced by label
correlations.

  L_div = Σ_{i<j} max(0, ⟨p̂_i, p̂_j⟩ − θ_div)

where p̂_k = p_k / ‖p_k‖ is the unit-normed reciprocal point.

Why conditional (not global): the actual reciprocal point is -P_k and the
stored P_k aligns with w_k.  Classifiers should be close when labels are
highly correlated.  Forcing
ALL pairs apart would fight the label correlation structure.
"""

import torch


def diversity_loss(P, threshold=0.9):
    """Conditional reciprocal-point diversity loss.

    Parameters
    ----------
    P         : (d', q) tensor   reciprocal-point bank
    threshold : float            cosine-similarity threshold (0.9)

    Returns scalar loss (≥ 0).
    """
    _, q = P.shape
    if q < 2:
        return P.new_zeros(())

    # Normalise columns to unit hypersphere
    P_norm = P / (P.norm(p=2, dim=0, keepdim=True) + 1e-8)  # (d', q)

    # Pairwise cosine similarities: (q, q)
    cos = P_norm.T @ P_norm

    # Upper triangle only (i < j)
    triu_idx = torch.triu_indices(q, q, offset=1, device=P.device)
    cos_pairs = cos[triu_idx[0], triu_idx[1]]

    # Hinge: max(0, cos − θ)
    violations = torch.clamp(cos_pairs - threshold, min=0.0)

    return violations.sum()

"""Open-space risk: hinge loss on distances to reciprocal points.

Uses the CREM paper definition with one explicit storage convention:

  P_k is the classifier-aligned coefficient and the actual reciprocal point
  is -P_k.  Therefore d²(f_i, reciprocal_k) = ‖f_i + P_k‖² and W is coupled
  to P through ‖W-P‖².

  ℓ(R_k² − d²(f(x_i), -P_k))    where ℓ(t) = max(0, 1 + t)

For positive instances (y_{ik} = +1):
  L_open = (α / |pos_k|) Σ_k Σ_{i : y_{ik}=1} max(0, 1 + R_k² − d²(f(x_i), p_k))

This pushes each known-positive instance *away* from its class reciprocal point,
creating a margin for unknown-class detection.

The MATLAB implementation uses the opposite hinge direction as a preserved
legacy quirk.  D-CREM deliberately follows the paper objective; changing P's
sign cannot change the direction of the hinge inequality.
"""

import torch


def reciprocal_distances(features, P):
    """Squared Euclidean distances to the actual reciprocal points ``-P``.

    This general form remains correct when the L2-normalisation ablation is
    enabled; it never assumes ``||f||=1``.
    """
    feature_norms_sq = (features * features).sum(dim=1, keepdim=True)
    p_norms_sq = (P * P).sum(dim=0, keepdim=True)
    return feature_norms_sq + p_norms_sq + 2.0 * (features @ P)


def open_space_risk(features, P, R, targets, alpha=1.0,
                    positive_prevalence=None, reduction="sum",
                    radius_free=False, margin=1.0):
    """Compute open-space hinge risk.

    Parameters
    ----------
    features : (B, d') tensor   ℓ₂-normalised features
    P        : (d', q) tensor   stored coefficients; reciprocal points are -P
    R        : (q,) tensor      per-label margin radii (≥ 0)
    targets  : (B, q) tensor    ±1 ground truth
    alpha    : float            open-space risk weight (default 1, matches CREM)

    Returns scalar loss.
    """
    B, q = targets.shape
    d2 = reciprocal_distances(features, P)

    # Hinge: max(0, 1 + R_k² − d²)
    if radius_free:
        r2 = features.new_zeros((1, q))
    else:
        r2 = (R * R).unsqueeze(0)                 # (1, q)
    hinge = torch.clamp(float(margin) + r2 - d2, min=0.0)  # (B, q)

    # Per-class average over positive instances
    pos_mask = (targets == 1)                      # (B, q)
    if positive_prevalence is None:
        denom = pos_mask.sum(dim=0).clamp(min=1).to(features.dtype)
    else:
        prevalence = torch.as_tensor(
            positive_prevalence, device=features.device, dtype=features.dtype)
        denom = (features.shape[0] * prevalence).clamp(min=1e-12)
    per_class = (hinge * pos_mask.float()).sum(dim=0) / denom

    if reduction == "sum":
        reduced = per_class.sum()
    elif reduction == "mean":
        reduced = per_class.mean()
    else:
        raise ValueError(f"unknown open-space reduction: {reduction}")
    return alpha * reduced


def open_space_risk_per_class(features, P, R, targets, alpha=1.0,
                              positive_prevalence=None):
    """Like open_space_risk but returns per-class losses (for logging)."""
    B, q = targets.shape
    d2 = reciprocal_distances(features, P)
    r2 = (R * R).unsqueeze(0)
    hinge = torch.clamp(1.0 + r2 - d2, min=0.0)
    pos_mask = (targets == 1)
    if positive_prevalence is None:
        denom = pos_mask.sum(dim=0).clamp(min=1).to(features.dtype)
    else:
        prevalence = torch.as_tensor(
            positive_prevalence, device=features.device, dtype=features.dtype)
        denom = (features.shape[0] * prevalence).clamp(min=1e-12)
    per_class = (hinge * pos_mask.float()).sum(dim=0) / denom
    return alpha * per_class  # (q,)

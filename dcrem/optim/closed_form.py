"""Closed-form parameter updates for frozen-feature optimisation.

When the encoder is frozen, the objective reduces to a regularised
least-squares problem over (W, b, P, R).  Each admits a closed-form
or rapid iterative update, mirroring CREM's alternating optimisation
but in primal space.
"""

import torch


def closed_form_W(features, targets, P, lamda1=1.0, lamda3=10.0,
                  sample_average=True):
    """Ridge-regression closed form for W (encoder frozen, no correlation).

    Solves: min_W ½‖FW + 1bᵀ − Y‖² + ½λ₁‖W‖² + ½λ₃‖W−P‖²

    For fixed P, the optimal W satisfies:
      (FᵀF + (λ₁+λ₃)I) W = Fᵀ(Y − 1bᵀ) + λ₃P

    Parameters
    ----------
    features : (N, d') tensor
    targets  : (N, q) tensor   ±1
    P        : (d', q) tensor   current reciprocal bank
    lamda1   : float            ridge penalty (=1 in CREM)
    lamda3   : float            coupling weight (=10 in CREM)

    Returns
    -------
    W : (d', q) tensor
    """
    F = features
    d = F.size(1)
    Y = targets
    N = F.size(0)
    # Mean-centre for b
    F_mean = F.mean(dim=0, keepdim=True)     # (1, d')
    Y_mean = Y.mean(dim=0, keepdim=True)     # (1, q)
    Fc = F - F_mean                           # (N, d')
    Yc = Y - Y_mean                           # (N, q)

    scale = float(N) if sample_average else 1.0
    A = Fc.T @ Fc / scale + (lamda1 + lamda3) * torch.eye(
        d, device=F.device, dtype=F.dtype)
    B = Fc.T @ Yc / scale + lamda3 * P
    W = torch.linalg.solve(A, B)              # (d', q)
    return W


def closed_form_b(features, W, targets):
    """Closed-form bias (mean of residuals).

    b = mean(Y − FW) over the batch.
    """
    residuals = targets - features @ W        # (N, q)
    return residuals.mean(dim=0)              # (q,)


# There is intentionally no closed-form P/R update here.  Under the paper
# hinge max(0, 1 + R² - ||f+P||²), the active-region subproblem in P contains
# a negative quadratic and is not globally convex.  The old implementation
# copied a closed form derived for the opposite hinge direction, so Mode B did
# not optimise the loss reported by Mode A.  P and R are now updated by
# gradient descent on the shared objective in DCREMTrainer.

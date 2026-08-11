"""Learnable open-set calibrator (optional, independent of C1/C2/C3).

Replaces CREM's post-hoc Firth logistic calibration with a lightweight
MLP that maps per-label reciprocal-point distances to an open-set score.

  score(x) = σ( MLP_φ( [d(x, p_1), ..., d(x, p_q)] ) )

P stores classifier-aligned coefficients; the actual reciprocal point is -P.
Thus d²(x, reciprocal_k) = ‖f(x)+P_k‖².  The general norm is used so the
distance remains correct when L2 normalisation is disabled.

This is an *engineering improvement* — if it doesn't outperform Firth, we
fall back to the original calibrator.
"""

import torch
import torch.nn as nn


class OpenSetCalibrator(nn.Module):
    """MLP calibrator: per-sample distances → scalar open-set score.

    Architecture: q → q//2 → 1 with ReLU, no output activation (raw score).

    Parameters
    ----------
    num_classes : int   q — number of labels
    hidden_ratio : float   hidden dim = q // hidden_ratio (default 2 → q/2)
    """

    def __init__(self, num_classes: int, hidden_ratio: int = 2):
        super().__init__()
        hidden_dim = max(1, num_classes // hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, distances):
        """distances: (B, q) — per-label distances to reciprocal points.
        Returns (B,) — raw scores (higher = more likely known, i.e. farther
        from reciprocal points).
        """
        return self.mlp(distances).squeeze(-1)

    @staticmethod
    def compute_distances(features, P):
        """Compute squared distances from features to reciprocal points.

        features : (B, d')  ℓ₂-normalised
        P        : (d', q)  stored coefficients; reciprocal points are -P

        Returns (B, q): d²(f(x_i), -P_k) = ‖f(x_i)+P_k‖²
        """
        from dcrem.losses.open_space import reciprocal_distances
        return reciprocal_distances(features, P)

    @staticmethod
    def compute_relative_scores(features, W, P):
        """Per-label knownness from negative- minus positive-prototype distance.

        With unit directions this is
        ``||f+P_hat||² - ||f-W_hat||²``.  If ``P_hat == W_hat`` it reduces
        exactly to ``4 fᵀ W_hat``; therefore a non-collapsed residual bank is
        necessary for this score to contain information beyond a normalized
        linear classifier.
        """
        W_hat = torch.nn.functional.normalize(W, p=2, dim=0)
        P_hat = torch.nn.functional.normalize(P, p=2, dim=0)
        positive = ((features[:, :, None] - W_hat[None, :, :]) ** 2).sum(dim=1)
        negative = ((features[:, :, None] + P_hat[None, :, :]) ** 2).sum(dim=1)
        return negative - positive

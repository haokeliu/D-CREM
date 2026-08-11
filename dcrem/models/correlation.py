"""Learnable and train-fold static label-correlation modules.

Implements C2 (feature-space label correlation): label embeddings E ∈ R^{q×d_e}
are learned alongside the encoder, and the correlation matrix C adapts as the
feature space evolves during training.

C = 0.5 * (C_raw + C_rawᵀ)   where   C_raw = softmax(E Eᵀ / √d_e)
diag(C) = 0                     (no self-loops)
L = D − C   with   D = diag(C · 1)   (symmetric normalized Laplacian)

The explicit symmetrisation (C = 0.5*(C_raw + C_rawᵀ)) is essential: softmax
is row-wise and does not preserve symmetry even when the input is symmetric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _static_cooccurrence(train_target):
    """Build CREM's symmetric co-occurrence graph from a Q×N train target.

    The caller must supply the training fold only.  Keeping this transformation
    here, rather than deriving it from a loader during optimisation, makes the
    static-C ablation auditable and prevents validation/test labels from ever
    entering its graph.
    """
    target = torch.as_tensor(train_target, dtype=torch.float32)
    if target.ndim != 2:
        raise ValueError("train_target must be a Q×N matrix")
    if target.shape[1] == 0:
        raise ValueError("train_target must contain at least one sample")

    positive = target.eq(1).to(dtype=torch.float32)
    positive_counts = positive.sum(dim=1)             # (Q,)
    cooccurrence = positive @ positive.T              # (Q, Q)
    directed = torch.where(
        positive_counts.unsqueeze(0) > 0,
        cooccurrence / positive_counts.unsqueeze(0).clamp_min(1.0),
        torch.zeros_like(cooccurrence),
    )
    directed.fill_diagonal_(0.0)
    correlation = 0.5 * (directed + directed.T)
    degree = torch.diag(correlation.sum(dim=1))
    return correlation, degree - correlation


class CorrelationModule(nn.Module):
    """Learnable label correlation via scaled dot-product attention.

    Parameters
    ----------
    num_classes : int        q — number of labels
    embed_dim   : int        d_e — label embedding dimension (default 128)
    temperature : float      scaling factor for dot product (√d_e by default)
    """

    def __init__(self, num_classes: int, embed_dim: int = 128,
                 temperature: float = None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temperature = temperature or (embed_dim ** 0.5)
        # E ∈ R^{q × d_e}
        self.E = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

    def forward(self):
        """Return (C, L) — correlation matrix and Laplacian."""
        # Scaled dot-product
        scores = self.E @ self.E.T / self.temperature  # (q, q)
        # Row-wise softmax
        C_raw = F.softmax(scores, dim=1)
        # Explicit symmetrisation (critical — see §3.5 of the experimental plan)
        C = 0.5 * (C_raw + C_raw.T)
        # Zero diagonal
        C = C - torch.diag(torch.diag(C))
        # Laplacian: L = D − C
        D_diag = C.sum(dim=1)
        D = torch.diag(D_diag)
        L = D - C
        return C, L

    def get_static_reference(self, train_target):
        """Compute static co-occurrence C for comparison (M2 analysis).

        train_target: (Q, N) tensor with ±1 entries.
        Returns numpy (Q, Q) array.
        """
        C, _ = _static_cooccurrence(train_target)
        return C.cpu().numpy()


class StaticCorrelationModule(nn.Module):
    """Frozen label graph built exclusively from a training-fold target.

    Unlike ``CorrelationModule``, this module has no learnable parameters.
    Its buffers remain part of ``state_dict`` and move with ``.to(device)``,
    while gradients still flow through the W-dependent correlation loss.
    """

    def __init__(self, train_target):
        super().__init__()
        correlation, laplacian = _static_cooccurrence(train_target)
        self.register_buffer("correlation", correlation)
        self.register_buffer("laplacian", laplacian)

    def forward(self):
        return self.correlation, self.laplacian

"""Classifier head, reciprocal-point bank, and margin vector.

D-CREM's primary variables (primal space, d' × q):
  W — classifier weight matrix  (ClassifierHead)
  b — bias vector               (ClassifierHead)
  P — classifier-aligned reciprocal coefficients (actual points are -P)
  R — per-label margin radius   (MarginVector, softplus-param'd to be ≥ 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """Linear classifier W ∈ R^{d'×q}, b ∈ R^q.

    Output: s_k(x) = ⟨f(x), w_k⟩ + b_k  (logit for label k).
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, num_classes) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.num_classes = num_classes

    def forward(self, features):
        """features: (B, d') → logits: (B, q)."""
        return features @ self.W + self.b

    def closed_form_init(self, features, targets, lamda1=1.0,
                         sample_average=True):
        """Initialise W from ridge-regression closed form on *frozen* features.

        With the bias optimised out and the sample-averaged objective:
          W = (F_cᵀF_c/N + λ₁I)⁻¹ F_cᵀY_c/N
          b = mean(Y-FW)

        Parameters
        ----------
        features : (N, d') tensor          L2-normalised encoder outputs
        targets  : (N, q) tensor           ±1 labels
        lamda1   : float                   ridge penalty (CREM λ₁, = 1)
        """
        with torch.no_grad():
            F = features  # (N, d')
            Y = targets   # (N, q)
            n, d = F.shape
            Fc = F - F.mean(dim=0, keepdim=True)
            Yc = Y - Y.mean(dim=0, keepdim=True)
            scale = float(n) if sample_average else 1.0
            A = Fc.T @ Fc / scale + lamda1 * torch.eye(
                d, device=F.device, dtype=F.dtype)
            B = Fc.T @ Yc / scale
            W_init = torch.linalg.solve(A, B)  # (d', q)
            self.W.copy_(W_init)
            self.b.copy_((Y - F @ W_init).mean(dim=0))


class ReciprocalBank(nn.Module):
    """Classifier-aligned reciprocal coefficients P ∈ R^{d'×q}.

    The actual reciprocal point for class k is ``-P[:, k]``.  Initialising and
    coupling P to W therefore implements the classifier-induced relation
    reciprocal_k ≈ -w_k without mixing sign conventions.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.P = nn.Parameter(torch.randn(feature_dim, num_classes) * 0.01)

    def init_from_W(self, W):
        """Set stored coefficients P=W, hence actual reciprocal points=-W."""
        with torch.no_grad():
            self.P.copy_(W)

    def forward(self):
        return self.P

    def normalized(self):
        """Return P with each column ℓ₂-normalised (for L_div computation)."""
        return F.normalize(self.P, p=2, dim=0)


class MultiReciprocalBank(nn.Module):
    """Multiple reciprocal coefficients per label, ``P in R^(d x q x m)``.

    The actual reciprocal points remain ``-P``.  Development initialization
    clusters frozen-W hard negatives in feature space so different prototypes
    start from different observed extra-class modes.
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 num_prototypes: int = 2):
        super().__init__()
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")
        self.num_prototypes = int(num_prototypes)
        self.P = nn.Parameter(
            torch.randn(feature_dim, num_classes, num_prototypes) * 0.01)

    def forward(self):
        return self.P

    def init_from_W(self, W):
        with torch.no_grad():
            self.P.copy_(W.unsqueeze(-1).expand_as(self.P))

    @staticmethod
    def _farthest_first_centers(points, count, iterations=5):
        """Deterministic small k-means initialized by farthest-first points."""
        if points.shape[0] == 0:
            raise ValueError("cannot cluster an empty point set")
        centers = [points[0]]
        while len(centers) < count:
            stacked = torch.stack(centers)
            distances = torch.cdist(points, stacked).pow(2).min(dim=1).values
            centers.append(points[int(distances.argmax().item())])
        centers = torch.stack(centers)
        for _ in range(iterations):
            assignments = torch.cdist(points, centers).argmin(dim=1)
            updated = []
            for index in range(count):
                members = points[assignments == index]
                updated.append(
                    members.mean(dim=0) if members.shape[0] else centers[index])
            centers = torch.stack(updated)
        return F.normalize(centers, p=2, dim=1)

    def init_from_hard_negatives(self, features, targets, W,
                                 hard_fraction=0.25):
        """Initialize actual points ``-P`` from frozen-W hard-negative modes."""
        if not 0 < hard_fraction <= 1:
            raise ValueError("hard_fraction must be in (0, 1]")
        W_hat = F.normalize(W.detach(), p=2, dim=0)
        scores = features.detach() @ W_hat
        with torch.no_grad():
            for label_index in range(targets.shape[1]):
                negative_indices = torch.where(
                    targets[:, label_index] == -1)[0]
                if negative_indices.numel() == 0:
                    self.P[:, label_index, :].copy_(
                        W[:, label_index, None].expand(
                            -1, self.num_prototypes))
                    continue
                hard_count = min(
                    negative_indices.numel(),
                    max(self.num_prototypes, int(torch.ceil(torch.tensor(
                        negative_indices.numel() * hard_fraction)).item())))
                selected = torch.topk(
                    scores[negative_indices, label_index],
                    k=hard_count).indices
                hard_features = features[negative_indices[selected]].detach()
                centers = self._farthest_first_centers(
                    hard_features, self.num_prototypes)
                self.P[:, label_index, :].copy_(-centers.T)


class ResidualReciprocalBank(nn.Module):
    """A tangent residual bank that cannot collapse to ``P == W``.

    The stored direction is ``normalize(W_hat + rho * Delta_tangent_hat)``.
    ``Delta_tangent`` is projected onto the tangent plane of each classifier
    direction, giving the residual a distinct geometric role while keeping the
    classifier-induced reference explicit.
    """

    def __init__(self, feature_dim: int, num_classes: int, residual_scale=0.5):
        super().__init__()
        self.delta = nn.Parameter(torch.randn(feature_dim, num_classes) * 0.01)
        self.residual_scale = float(residual_scale)

    def forward(self, W):
        W_hat = F.normalize(W, p=2, dim=0)
        projection = (W_hat * self.delta).sum(dim=0, keepdim=True)
        tangent = self.delta - W_hat * projection
        tangent_hat = F.normalize(tangent, p=2, dim=0)
        return F.normalize(
            W_hat + self.residual_scale * tangent_hat, p=2, dim=0)

    def init_from_W(self, W):
        """Keep the random tangent directions; W only defines their plane."""
        del W


class MarginVector(nn.Module):
    """Per-label margin radius R ∈ R^q, parameterised via softplus to
    guarantee R_k ≥ 0.

    Initialised so that R_k = √2 (matching CREM default).
    """

    def __init__(self, num_classes: int, init_val=2.0 ** 0.5):
        super().__init__()
        # Solve softplus(raw) = init_val
        raw_init = torch.as_tensor(init_val).expm1().log().item()
        self.raw_R = nn.Parameter(torch.full((num_classes,), raw_init))

    def forward(self):
        """Return R ≥ 0 via softplus."""
        return F.softplus(self.raw_R)

    def clamp_min(self, min_val=1e-4):
        """Ensure raw_R stays above softplus⁻¹(min_val)."""
        floor = torch.as_tensor(min_val).expm1().log().item()
        with torch.no_grad():
            self.raw_R.clamp_(min=floor)

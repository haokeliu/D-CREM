"""Encoders for D-CREM: tabular MLP, frozen ResNet-50, frozen CLIP ViT-B/32,
and an identity encoder for equivalence testing.

All encoders output features with configurable dimension (default 128).
For non-identity encoders the L2Norm layer is *not* included here — it is
applied separately in the training loop so its presence can be ablated (A1).
"""

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """MLP encoder for tabular data (d → 512 → 256 → 128).

    With BatchNorm + ReLU at each hidden layer.
    Designed to be lightweight — pre-warmup runs MSE-only for 5-10 epochs.
    """

    def __init__(self, input_dim: int, hidden_dims=(512, 256), output_dim=128,
                 dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)


class ResNet50Encoder(nn.Module):
    """ImageNet-pretrained ResNet-50 with an optional learned projection."""

    def __init__(self, frozen=True, output_dim=2048):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Strip fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = (
            nn.Identity() if output_dim == 2048 else nn.Linear(2048, output_dim)
        )
        self.output_dim = output_dim
        self.backbone_frozen = frozen
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        # x: (B, 3, H, W), normalised with ImageNet stats
        feats = self.backbone(x).view(x.size(0), -1)
        return self.projection(feats)


class CLIPViTEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 encoder (512-d output, already ℓ₂-normalised).

    Uses open_clip for the ViT-B/32 variant with LAION-2B pretrained weights.
    Features come out already unit-normed; we keep that property.
    """

    def __init__(self, frozen=True):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.encoder = model.visual
        self.output_dim = self.encoder.output_dim  # 512
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, 3, H, W) normalised with CLIP stats
        return self.encoder(x)  # already ℓ₂-normed


class IdentityEncoder(nn.Module):
    """Identity encoder g_θ(x) = x (no parameters).

    Used for equivalence testing: D-CREM with identity encoder + linear
    kernel should be algebraically isomorphic to CREM with linear kernel,
    providing a correctness smoke-test for the primal-space Sylvester.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x):
        return x

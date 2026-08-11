"""Run a tiny, data-free D-CREM paper-core training smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dcrem.models.calibrator import OpenSetCalibrator
from dcrem.models.encoder import TabularMLP
from dcrem.models.heads import ClassifierHead, MarginVector, ReciprocalBank
from dcrem.optim.trainer import DCREMTrainer
from dcrem.reproducibility import seed_everything


def main() -> None:
    seed_everything(7)
    device = torch.device("cpu")
    sample_count, input_dim, class_count = 96, 12, 4

    generator = torch.Generator().manual_seed(7)
    features = torch.randn(sample_count, input_dim, generator=generator)
    targets = torch.where(
        torch.rand(sample_count, class_count, generator=generator) > 0.65,
        1.0,
        -1.0,
    )
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=24,
        shuffle=True,
        generator=generator,
    )

    embedding_dim = 16
    encoder = TabularMLP(input_dim, output_dim=embedding_dim)
    head = ClassifierHead(embedding_dim, class_count)
    reciprocal_bank = ReciprocalBank(embedding_dim, class_count)
    margins = MarginVector(class_count)
    trainer = DCREMTrainer(
        encoder,
        head,
        reciprocal_bank,
        margins,
        config={
            "lamda1": 1.0,
            "lamda2": 0.0,
            "lamda3": 0.0,
            "alpha": 0.0,
            "beta": 0.1,
            "gamma": 0.0,
            "lr": 1e-3,
            "backbone_lr": 1e-3,
            "T_sylvester": 1,
            "pre_warmup_epochs": 0,
            "no_warmup": True,
            "classifier_induced_reciprocal": True,
        },
    ).to(device)

    history = trainer.fit_mode_B(loader, num_epochs=2, log_every=1)
    trainer.eval()
    with torch.no_grad():
        encoded = trainer._forward_encoder(features.to(device))
        distances = OpenSetCalibrator.compute_distances(
            encoded, trainer.get_reciprocal_parameters()
        )

    assert np.isfinite(history["loss"]).all()
    assert distances.shape == (sample_count, class_count)
    print(
        "Synthetic smoke test passed: "
        f"final_loss={history['loss'][-1]:.6f}, "
        f"distance_shape={tuple(distances.shape)}"
    )


if __name__ == "__main__":
    main()

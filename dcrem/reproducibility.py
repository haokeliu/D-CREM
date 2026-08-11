"""Reproducibility helpers shared by D-CREM experiment entry points."""

import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy, PyTorch CPU/CUDA, and deterministic backends."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)

    return {
        "seed": seed,
        "deterministic_algorithms": bool(deterministic),
        "cudnn_deterministic": bool(
            deterministic and hasattr(torch.backends, "cudnn")),
    }

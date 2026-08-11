"""Tabular data loading — reuses crem.data for the CREM pipeline.

Provides torch DataLoader wrappers around CREM's known/unknown split.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TabularDataset:
    """PyTorch-compatible wrapper around CREM tabular data.

    Parameters
    ----------
    dataset_name : str          e.g. 'enron', 'bibtex'
    known_ratio  : float        fraction of labels known
    seed         : int          random seed for the split
    standardize  : bool         whether to standardise features
    train_ratio  : float        fraction for training (default 0.4)
    val_ratio    : float        fraction for validation (default 0.1)
    """

    def __init__(self, dataset_name, known_ratio=0.5, seed=0, standardize=True,
                 train_ratio=0.4, val_ratio=0.1):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from crem.data import get_dataset

        self.dataset_name = dataset_name
        self.known_ratio = known_ratio
        self.seed = seed
        self.standardize = standardize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        data = get_dataset(dataset_name, known_ratio=known_ratio, seed=seed,
                          standardize=standardize, train_ratio=train_ratio,
                          val_ratio=val_ratio)
        self.data = data
        self.input_dim = data["train_data"].shape[1]
        self.num_classes = data["train_target"].shape[0]

    def get_train_loader(self, batch_size=64, shuffle=True):
        X = torch.as_tensor(self.data["train_data"], dtype=torch.float32)
        # train_target: (Q, N, ±1) → (N, Q)
        Y = torch.as_tensor(self.data["train_target"].T, dtype=torch.float32)
        ds = TensorDataset(X, Y)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         drop_last=False, generator=generator)

    def get_val_data(self):
        """Return validation features, known-label targets, and OSR labels."""
        return (self.data["val_data"].astype(np.float64),
                self.data["val_target"],
                np.asarray(self.data["val_osr_labels"]).ravel())

    def get_test_data(self):
        """Return (X_test, Y_test) as numpy arrays (for evaluation)."""
        return (self.data["test_data"].astype(np.float64),
                self.data["test_target"])  # Q×N, ±1

    def get_osr_labels(self):
        return np.asarray(self.data["osr_labels"]).ravel()

    def get_label_names(self):
        return list(self.data.get("known_label_names", []))


def get_tabular_loaders(dataset_name, known_ratio=0.5, seed=0, standardize=True,
                        batch_size=64, train_ratio=0.4, val_ratio=0.1):
    """Convenience: create TabularDataset and return train/test data.

    Returns (train_loader, test_X, test_Y_QN, osr_labels, input_dim, num_classes).
    """
    ds = TabularDataset(dataset_name, known_ratio=known_ratio, seed=seed,
                        standardize=standardize, train_ratio=train_ratio,
                        val_ratio=val_ratio)
    train_loader = ds.get_train_loader(batch_size=batch_size, shuffle=True)
    test_X, test_Y = ds.get_test_data()
    osr = ds.get_osr_labels()
    return train_loader, test_X, test_Y, osr, ds.input_dim, ds.num_classes


def get_tabular_protocol(dataset_name, known_ratio=0.5, seed=0,
                         standardize=True, batch_size=64, train_ratio=0.4,
                         val_ratio=0.1):
    """Return all train/validation/test inputs for leakage-free experiments."""
    ds = TabularDataset(
        dataset_name, known_ratio=known_ratio, seed=seed,
        standardize=standardize, train_ratio=train_ratio, val_ratio=val_ratio)
    val_X, val_Y, val_osr = ds.get_val_data()
    test_X, test_Y = ds.get_test_data()
    return {
        "train_loader": ds.get_train_loader(batch_size=batch_size, shuffle=True),
        "val_X": val_X,
        "val_Y_QN": val_Y,
        "val_osr_labels": val_osr,
        "test_X": test_X,
        "test_Y_QN": test_Y,
        "test_osr_labels": ds.get_osr_labels(),
        "input_dim": ds.input_dim,
        "num_classes": ds.num_classes,
        "split": ds.data,
    }

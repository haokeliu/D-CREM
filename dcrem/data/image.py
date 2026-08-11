"""Image data loading for VOC2007 / MS-COCO multi-label datasets.

Loads pre-extracted .mat feature files from cache/ and splits them
into CREM-format train/test with known/unknown labels.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")


def _load_cache(dataset_name, feature_type):
    """Load a pre-extracted .mat cache file."""
    from scipy.io import loadmat
    path = os.path.join(CACHE_DIR, f"{dataset_name}_{feature_type}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature cache not found: {path}.  "
            f"Run scripts/run_m3_features.py --extract first.")
    mat = loadmat(path)
    X = mat["X"].astype(np.float64)
    Y = mat["Y"].astype(np.float64)
    raw_names = mat.get("label_names", [[]])
    label_names = [
        str(x[0]) if (hasattr(x, 'size') and x.size > 0) else str(x)
        for x in raw_names.ravel()
    ] if len(raw_names) > 0 else [f"L{i}" for i in range(Y.shape[1])]
    return X, Y, label_names


class ImageDataset:
    """PyTorch-compatible wrapper around pre-extracted image features.

    Uses the same CREM split protocol as tabular data (see crem.data.apply_crem_split).

    Parameters
    ----------
    dataset_name  : str         'voc2007' or 'coco2014'
    feature_type  : str         'pca', 'resnet50', or 'clip'
    known_ratio   : float       fraction of labels known
    seed          : int
    standardize   : bool        whether to standardise features (default True for image)
    train_ratio   : float       fraction for training
    val_ratio     : float       fraction for validation
    """

    def __init__(self, dataset_name, feature_type, known_ratio=0.5, seed=0,
                 standardize=True, train_ratio=0.4, val_ratio=0.1):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from crem.data import apply_crem_split

        X, Y, label_names = _load_cache(dataset_name, feature_type)
        data = apply_crem_split(X, Y, label_names, known_ratio, seed,
                               standardize=standardize, train_ratio=train_ratio,
                               val_ratio=val_ratio)
        self.data = data
        self.input_dim = data["train_data"].shape[1]
        self.num_classes = data["train_target"].shape[0]
        self.feature_type = feature_type
        self.dataset_name = dataset_name
        self.seed = seed

    def get_train_loader(self, batch_size=64, shuffle=True):
        X = torch.as_tensor(self.data["train_data"], dtype=torch.float32)
        Y = torch.as_tensor(self.data["train_target"].T, dtype=torch.float32)
        ds = TensorDataset(X, Y)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         drop_last=False, generator=generator)

    def get_val_data(self):
        return (self.data["val_data"].astype(np.float64),
                self.data["val_target"],
                np.asarray(self.data["val_osr_labels"]).ravel())

    def get_test_data(self):
        return (self.data["test_data"].astype(np.float64),
                self.data["test_target"])

    def get_osr_labels(self):
        return np.asarray(self.data["osr_labels"]).ravel()

    def get_label_names(self):
        return list(self.data.get("known_label_names", []))


def get_image_loaders(dataset_name, feature_type, known_ratio=0.5, seed=0,
                      standardize=True, batch_size=64, train_ratio=0.4,
                      val_ratio=0.1):
    """Convenience: create ImageDataset and return train/test data."""
    ds = ImageDataset(dataset_name, feature_type, known_ratio=known_ratio,
                      seed=seed, standardize=standardize,
                      train_ratio=train_ratio, val_ratio=val_ratio)
    train_loader = ds.get_train_loader(batch_size=batch_size, shuffle=True)
    test_X, test_Y = ds.get_test_data()
    osr = ds.get_osr_labels()
    return train_loader, test_X, test_Y, osr, ds.input_dim, ds.num_classes


def get_image_protocol(dataset_name, feature_type, known_ratio=0.5, seed=0,
                       standardize=True, batch_size=64, train_ratio=0.4,
                       val_ratio=0.1):
    """Return all train/validation/test inputs for frozen image features."""
    ds = ImageDataset(
        dataset_name, feature_type, known_ratio=known_ratio, seed=seed,
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


class _VOCSplitDataset(Dataset):
    """Lazy VOC image view with fold-specific transforms and fixed targets."""

    def __init__(self, base_dataset, indices, targets, transform):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, _ = self.base_dataset[int(self.indices[item])]
        image = self.transform(image.convert("RGB"))
        return image, torch.from_numpy(self.targets[item])


def get_voc2007_raw_protocol(known_ratio=0.5, seed=0, batch_size=32,
                             train_ratio=0.4, val_ratio=0.1,
                             num_workers=0):
    """Return leakage-free loaders over raw VOC2007 images.

    Label and instance partitions are generated by the same Protocol-v2 split
    routine as the frozen ResNet-50 feature control.  Random augmentation is
    confined to the training fold; validation and test transforms are fixed.
    """
    from torchvision import datasets, transforms
    from crem.data import apply_crem_split

    _, labels, label_names = _load_cache("voc2007", "resnet50")
    dummy = np.arange(labels.shape[0], dtype=np.float64).reshape(-1, 1)
    split = apply_crem_split(
        dummy, labels, label_names, known_ratio, seed, standardize=False,
        train_ratio=train_ratio, val_ratio=val_ratio)

    raw_root = os.path.join(CACHE_DIR, "images", "raw")
    base = datasets.VOCDetection(
        root=raw_root, year="2007", image_set="trainval", download=False)
    if len(base) != labels.shape[0]:
        raise ValueError(
            "VOC2007 raw-image order does not match the cached label matrix: "
            f"{len(base)} images versus {labels.shape[0]} label rows")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = _VOCSplitDataset(
        base, split["train_indices"], split["train_target"].T,
        train_transform)
    val_ds = _VOCSplitDataset(
        base, split["val_indices"], split["val_target"].T,
        eval_transform)
    test_ds = _VOCSplitDataset(
        base, split["test_indices"], split["test_target"].T,
        eval_transform)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train_loader": DataLoader(
            train_ds, shuffle=True, generator=generator, **loader_kwargs),
        "val_X": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "val_Y_QN": split["val_target"],
        "val_osr_labels": np.asarray(split["val_osr_labels"]).ravel(),
        "test_X": DataLoader(test_ds, shuffle=False, **loader_kwargs),
        "test_Y_QN": split["test_target"],
        "test_osr_labels": np.asarray(split["osr_labels"]).ravel(),
        "input_dim": None,
        "num_classes": split["train_target"].shape[0],
        "split": split,
    }

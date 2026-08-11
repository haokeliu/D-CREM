"""Dynamic data loading with configurable known_ratio protocol and caching.

Two-level cache strategy:
  Source ARFF/XML  --[once]-->  cache/{name}_full.mat  --[per-run]-->  (train, test, labels)

The full cache contains features + all labels after feature selection, label
selection and instance sampling, BEFORE the known/unknown split.  Subsequent
runs only redo the known_ratio split (cheap).
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import loadmat, savemat
from scipy import sparse

from .config import SOURCE_DIR, CACHE_DIR, DATASET_SPECS


# ═══════════════════════════════════════════════════════════════════════════
# ARFF / XML parsing (from scripts/preprocess_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_arff(arff_path):
    """Parse a Mulan-format ARFF file. Returns (X, attr_names, attr_types)."""
    attr_names, attr_types, data_tokens = [], [], []
    n_attrs, row_idx, sparse_mode = 0, 0, None
    with open(arff_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                parts = line.split(None, 2)
                attr_names.append(parts[1])
                at = parts[2].strip() if len(parts) > 2 else ""
                attr_types.append(
                    "numeric" if at.lower() in ("numeric", "real", "integer") else "nominal"
                )
            elif low.startswith("@data"):
                n_attrs = len(attr_names)
            elif n_attrs > 0:
                ls = line.strip()
                if sparse_mode is None:
                    sparse_mode = ls.startswith("{")
                if sparse_mode:
                    ls = ls.strip("{}")
                    if ls:
                        for token in ls.split(","):
                            token = token.strip()
                            if not token:
                                continue
                            parts = token.split()
                            if len(parts) >= 2:
                                ci, v = int(parts[0]), float(parts[1])
                                if ci < n_attrs:
                                    data_tokens.append((row_idx, ci, v))
                else:
                    for ci, v in enumerate(ls.split(",")):
                        if ci >= n_attrs:
                            break
                        v = v.strip().strip("'\"")
                        try:
                            val = float(v)
                        except ValueError:
                            val = np.nan
                        if not np.isnan(val):
                            data_tokens.append((row_idx, ci, val))
                row_idx += 1
    if sparse_mode:
        rows = [item[0] for item in data_tokens]
        cols = [item[1] for item in data_tokens]
        vals = [item[2] for item in data_tokens]
        X = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(row_idx, n_attrs), dtype=np.float64)
    else:
        X = np.zeros((row_idx, n_attrs), dtype=np.float64)
        for ri, ci, v in data_tokens:
            X[ri, ci] = v
    return X, attr_names, attr_types


def _parse_labels_xml(xml_path):
    """Return set of label names from Mulan XML."""
    tree = ET.parse(xml_path)
    names = set()
    for el in tree.getroot().iter():
        name = el.get("name")
        if name:
            names.add(name)
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Feature & label selection (from scripts/preprocess_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

def _select_features_tfidf_sum(X, target_d):
    """Top target_d features by total TF-IDF weight sum."""
    if X.shape[1] <= target_d:
        return X, np.arange(X.shape[1])
    weights = np.asarray(X.sum(axis=0)).ravel()
    top = np.argsort(-weights)[:target_d]
    selected = np.sort(top)
    return X[:, selected], selected


def _select_features_variance(X, target_d):
    """Top target_d features by variance."""
    if X.shape[1] <= target_d:
        return X, np.arange(X.shape[1])
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        variance = np.maximum(mean_sq - mean * mean, 0.0)
    else:
        variance = np.asarray(X.var(axis=0)).ravel()
    top = np.argsort(-variance)[:target_d]
    selected = np.sort(top)
    return X[:, selected], selected


def _select_labels_cardinality(Y, target_L, target_LCard):
    """Greedy label selection to match target label cardinality."""
    if Y.shape[1] <= target_L:
        return Y, np.arange(Y.shape[1])

    label_prob = np.asarray(Y.sum(axis=0)).ravel() / Y.shape[0]
    selected = []
    remaining = list(range(Y.shape[1]))
    ideal = target_LCard / target_L
    first = int(np.argmin(np.abs(label_prob - ideal)))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < target_L and remaining:
        cur_card = Y[:, selected].sum(axis=1).mean()
        best_i, best_gap = None, float("inf")
        for i in remaining:
            trial = selected + [i]
            gap = abs(Y[:, trial].sum(axis=1).mean() - target_LCard)
            if gap < best_gap:
                best_gap, best_i = gap, i
        if best_i is not None:
            selected.append(best_i)
            remaining.remove(best_i)
        else:
            break

    sel = np.array(sorted(selected))
    return Y[:, sel], sel


# ═══════════════════════════════════════════════════════════════════════════
# Full cache builder
# ═══════════════════════════════════════════════════════════════════════════

def build_full_cache(name):
    """Build cache/{name}_full.mat from source ARFF/XML (idempotent).

    The cached file contains:
      X        : (N, d)  float64  – features
      Y        : (N, L)  float64  – all L labels (0/1)
      label_names : (L,) object  – label name strings
    """
    # Protocol-v2 caches preserve the sampled feature matrix *before* feature
    # selection.  Feature ranking is fold-fitted later using training samples
    # only.  A new filename deliberately prevents reuse of the old leaky cache.
    cache_path = os.path.join(CACHE_DIR, f"{name}_protocol_v2.mat")
    if os.path.exists(cache_path):
        return cache_path

    target_N, target_d, target_L, target_LCard = (
        DATASET_SPECS[name]["N"],
        DATASET_SPECS[name]["d"],
        DATASET_SPECS[name]["L"],
        DATASET_SPECS[name]["LCard"],
    )

    arff = os.path.join(SOURCE_DIR, f"{name}.arff")
    xml = os.path.join(SOURCE_DIR, f"{name}.xml")
    if not os.path.exists(arff) or not os.path.exists(xml):
        raise FileNotFoundError(f"Missing source files: {arff} / {xml}")

    X, anames, atyp = _parse_arff(arff)
    xml_labels = _parse_labels_xml(xml)
    lidx = [i for i, a in enumerate(anames) if a in xml_labels]
    fidx = [i for i, a in enumerate(anames) if a not in xml_labels]

    Xf = X[:, fidx].astype(np.float64)
    Y = X[:, lidx].astype(np.float64)
    if sparse.issparse(Y):
        Y = Y.toarray()
    label_names = np.array([anames[i] for i in lidx], dtype=object)

    # 1. Label selection
    Y, lsel = _select_labels_cardinality(Y, target_L, target_LCard)
    label_names = label_names[lsel]

    # 2. Instance sampling (fixed seed=42 for cache stability)
    rng = np.random.default_rng(42)
    if Xf.shape[0] > target_N:
        idx = rng.choice(Xf.shape[0], size=target_N, replace=False)
        idx = np.sort(idx)
        Xf, Y = Xf[idx, :], Y[idx, :]

    os.makedirs(CACHE_DIR, exist_ok=True)
    savemat(cache_path, {
        "X": Xf.astype(np.float64),
        "Y": Y.astype(np.float64),
        "label_names": label_names,
        "cache_version": np.array([[2]], dtype=np.int16),
        "target_feature_dim": np.array([[target_d]], dtype=np.int32),
    }, do_compression=True)

    final_lcard = Y.sum(axis=1).mean()
    print(f"  Cache built: {name}  N={Xf.shape[0]} d={Xf.shape[1]} "
          f"L={Y.shape[1]} LCard={final_lcard:.3f}")
    return cache_path


def build_all_caches():
    """Build caches for all known datasets."""
    for name in DATASET_SPECS:
        try:
            build_full_cache(name)
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")


def load_full_data(name):
    """Load the cached full data for a dataset (builds cache if needed)."""
    cache_path = build_full_cache(name)
    mat = loadmat(cache_path)
    X = mat["X"]
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float64)
    else:
        X = X.astype(np.float64)
    return (
        X,
        mat["Y"].astype(np.float64),
        [str(x[0]) if hasattr(x, '__iter__') and len(x) > 0 else str(x)
         for x in mat["label_names"].ravel()],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic CREM protocol split
# ═══════════════════════════════════════════════════════════════════════════

def apply_crem_split(X, Y, label_names, known_ratio, seed,
                     standardize=False, train_ratio=0.4, val_ratio=0.1,
                     target_d=None, feature_selector=None):
    """Split full data into leakage-free train/validation/test folds.

    Parameters
    ----------
    X : (N, d) array
    Y : (N, L) array, 0/1 labels
    label_names : list of str
    known_ratio : float, fraction of labels to use as "known"
    seed : int, random seed for reproducible splits
    standardize : bool, fit scaling statistics on training samples only
    train_ratio : float, fraction of samples to use for training
    val_ratio : float, fraction of samples to use for validation
    target_d : int | None, number of features selected using training data
    feature_selector : {"tfidf_sum", "variance", None}

    Returns
    -------
    dict with keys: train_data, val_data, test_data, train_target,
    val_target, test_target, val_osr_labels, osr_labels,
    known_label_indices, unknown_label_indices,
    known_label_names, unknown_label_names, known_ratio, seed
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to < 1")

    split_seq, label_seq = np.random.SeedSequence(seed).spawn(2)
    split_rng = np.random.default_rng(split_seq)
    label_rng = np.random.default_rng(label_seq)
    n, d = X.shape
    q = Y.shape[1]

    # ── Sample split ──
    idx = np.arange(n)
    split_rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    tr_idx = np.sort(idx[:n_train])
    va_idx = np.sort(idx[n_train:n_train + n_val])
    te_idx = np.sort(idx[n_train + n_val:])
    Xtr, Xva, Xte = X[tr_idx], X[va_idx], X[te_idx]
    Ytr, Yva, Yte = Y[tr_idx], Y[va_idx], Y[te_idx]

    # ── Fold-fitted feature selection ──
    if target_d is not None and Xtr.shape[1] > target_d:
        if feature_selector == "tfidf_sum":
            _, feature_idx = _select_features_tfidf_sum(Xtr, target_d)
        elif feature_selector == "variance":
            _, feature_idx = _select_features_variance(Xtr, target_d)
        else:
            raise ValueError("feature_selector is required when target_d is set")
        Xtr, Xva, Xte = Xtr[:, feature_idx], Xva[:, feature_idx], Xte[:, feature_idx]
    else:
        feature_idx = np.arange(Xtr.shape[1], dtype=int)

    def _dense_float(array):
        if sparse.issparse(array):
            array = array.toarray()
        return np.asarray(array, dtype=np.float64)

    Xtr = _dense_float(Xtr)
    Xva = _dense_float(Xva)
    Xte = _dense_float(Xte)

    # ── Optional standardisation, fitted on training samples only ──
    scaler_mean = np.zeros((1, Xtr.shape[1]), dtype=np.float64)
    scaler_std = np.ones((1, Xtr.shape[1]), dtype=np.float64)
    if standardize:
        scaler_mean = Xtr.mean(axis=0, keepdims=True)
        scaler_std = Xtr.std(axis=0, keepdims=True)
        scaler_std[scaler_std == 0] = 1.0
        Xtr = (Xtr - scaler_mean) / scaler_std
        Xva = (Xva - scaler_mean) / scaler_std
        Xte = (Xte - scaler_mean) / scaler_std

    # ── known/unknown label split ──
    lidx = np.arange(q)
    label_rng.shuffle(lidx)
    n_known = min(q, max(1, int(q * known_ratio)))
    kset = set(lidx[:n_known].tolist())
    uset = set(lidx[n_known:].tolist())

    # Ensure known labels have >= 1 positive in train
    Ytr_pos = Ytr.sum(axis=0)
    for k in list(kset):
        if Ytr_pos[k] == 0:
            swapped = False
            for u in list(uset):
                if Ytr_pos[u] > 0:
                    kset.remove(k)
                    uset.add(k)
                    kset.add(u)
                    uset.remove(u)
                    swapped = True
                    break
            if not swapped:
                kset.remove(k)
                uset.add(k)

    kidx = np.array(sorted(kset), dtype=int)
    uidx = np.array(sorted(uset), dtype=int)

    # ── Build CREM-format targets (+/-1) ──
    Ytr_k = Ytr[:, kidx]
    train_target = (2.0 * Ytr_k - 1.0).T.astype(np.int16)
    Yva_k = Yva[:, kidx]
    val_target = (2.0 * Yva_k - 1.0).T.astype(np.int16)
    Yte_k = Yte[:, kidx]
    test_target = (2.0 * Yte_k - 1.0).T.astype(np.int16)
    Yva_u = Yva[:, uidx]
    val_osr_labels = (Yva_u.sum(axis=1) > 0).astype(np.uint8)
    Yte_u = Yte[:, uidx]
    osr_labels = (Yte_u.sum(axis=1) > 0).astype(np.uint8)

    ln = np.array(label_names, dtype=object)

    return {
        "train_data": Xtr.astype(np.float64),
        "val_data": Xva.astype(np.float64),
        "test_data": Xte.astype(np.float64),
        "train_target": train_target,
        "val_target": val_target,
        "test_target": test_target,
        "val_osr_labels": val_osr_labels.reshape(1, -1),
        "osr_labels": osr_labels.reshape(1, -1),
        "known_label_indices": kidx,
        "unknown_label_indices": uidx,
        "known_label_names": ln[kidx],
        "unknown_label_names": ln[uidx],
        "known_ratio": known_ratio,
        "seed": seed,
        "standardized": standardize,
        "train_indices": tr_idx,
        "val_indices": va_idx,
        "test_indices": te_idx,
        "feature_indices": feature_idx,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }


def get_dataset(name, known_ratio=0.5, seed=0, standardize=False,
                train_ratio=0.4, val_ratio=0.1):
    """Top-level API: load cache, apply split, return CREM-format dict."""
    X, Y, label_names = load_full_data(name)
    selector = "tfidf_sum" if ("yahoo" in name or "slashdot" in name) else "variance"
    return apply_crem_split(
        X, Y, label_names, known_ratio, seed,
        standardize=standardize, train_ratio=train_ratio, val_ratio=val_ratio,
        target_d=DATASET_SPECS[name]["d"], feature_selector=selector)

"""Label correlation, port of CREM/help_function/compute_label_correlation.m.

Also provides M2 ablation variants: identity, random, and GloVe semantic C.
"""
import os
import numpy as np


def compute_label_correlation(train_target):
    """train_target: Q x N with +/-1 entries. Returns symmetric Q x Q matrix.

    P(j,i) = |pos(i) ∩ pos(j)| / |pos(i)|, W = 0.5*(P + P^T), zero diagonal.
    """
    Y = train_target == 1  # Q x N boolean
    pos_counts = Y.sum(axis=1)  # per-class positive counts
    co = Y.astype(float) @ Y.T.astype(float)  # co[j,i] = |pos(i) ∩ pos(j)|
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(pos_counts[None, :] > 0,
                     co / pos_counts[None, :], 0.0)
    np.fill_diagonal(P, 0.0)
    W = 0.5 * (P + P.T)
    return W


# ═══════════════════════════════════════════════════════════════════════════
# M2 ablation variants
# ═══════════════════════════════════════════════════════════════════════════

def compute_identity_C(train_target):
    """C = I_q: no label correlation (λ₂ term still active but isotropic)."""
    q = train_target.shape[0]
    return np.eye(q)


def compute_random_C(train_target, seed=0):
    """Random symmetric C with same Frobenius norm as the real C.

    Generates C_rand = (R + Rᵀ)/2 with Gaussian entries, scaled so
    ||C_rand||_F = ||C_real||_F, then zeros the diagonal.

    This preserves the overall regularisation strength while destroying
    any meaningful correlation structure.
    """
    q = train_target.shape[0]
    C_real = compute_label_correlation(train_target)
    target_norm = np.linalg.norm(C_real, "fro")

    rng = np.random.default_rng(seed)
    R = rng.normal(0, 1, (q, q))
    C_rand = 0.5 * (R + R.T)
    np.fill_diagonal(C_rand, 0.0)

    current_norm = np.linalg.norm(C_rand, "fro")
    if current_norm > 1e-12:
        C_rand *= target_norm / current_norm

    return C_rand


def compute_glove_semantic_C(label_names, glove_path=None):
    """Label correlation from GloVe word-vector cosine similarity.

    Parameters
    ----------
    label_names : list of str  e.g. ['TAG_learning', 'TAG_network', ...]
    glove_path : str | None    path to glove.6B.*.txt; if None, tries common
                               locations and falls back to random init.

    Returns
    -------
    C : (q, q) symmetric, diagonally-zeroed, each row softmax-normalised.

    Notes
    -----
    Only meaningful when label names are natural-language words (bibtex).
    For coded labels (enron: C.C5, A.A3) the "semantic" content is nil and
    this function will log a warning.
    """
    q = len(label_names)
    # Try to find GloVe file
    if glove_path is None:
        candidates = [
            os.path.expanduser("~/glove/glove.6B.300d.txt"),
            os.path.expanduser("~/.cache/glove/glove.6B.300d.txt"),
        ]
        glove_path = next((p for p in candidates if os.path.exists(p)), None)

    if glove_path is None:
        print("  [glove_semantic_C] WARNING: GloVe vectors not found. "
              "Using identity matrix as fallback.")
        return np.eye(q)

    # Load GloVe
    print(f"  [glove_semantic_C] Loading GloVe from {glove_path} ...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            glove[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float64)

    # Extract label tokens (strip prefixes like 'TAG_' for bibtex)
    embeddings = []
    found_count = 0
    for name in label_names:
        name_str = str(name).strip().strip("'\"")
        # Try multiple token forms
        tokens_to_try = [name_str, name_str.lower()]
        # Strip common prefixes
        for prefix in ["TAG_", "tag_"]:
            if name_str.startswith(prefix):
                tokens_to_try.append(name_str[len(prefix):].lower())
        # Also try splitting on underscores and taking the last meaningful part
        parts = name_str.replace("_", " ").split()
        tokens_to_try.extend(parts)
        tokens_to_try.extend([p.lower() for p in parts])

        vec = None
        for tok in tokens_to_try:
            if tok in glove:
                vec = glove[tok]
                found_count += 1
                break
        if vec is None:
            vec = np.random.randn(300).astype(np.float64) * 0.01
        embeddings.append(vec)

    print(f"  [glove_semantic_C] Found {found_count}/{q} labels in GloVe")

    E = np.stack(embeddings, axis=0)  # q × 300
    # Cosine similarity
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    C = E_norm @ E_norm.T  # q × q, entries in [-1, 1]
    # Clip negative correlations to 0, softmax-normalise per row
    C = np.maximum(C, 0.0)
    C = C / (C.sum(axis=1, keepdims=True) + 1e-12)
    np.fill_diagonal(C, 0.0)
    # Symmetrise
    C = 0.5 * (C + C.T)
    return C


def get_C_matrix(c_mode, train_target, label_names=None, seed=0):
    """Dispatcher for M2 label correlation variants.

    Parameters
    ----------
    c_mode : str  'full' | 'identity' | 'random' | 'semantic'
    train_target : (Q, N) array, +/-1
    label_names : list of str | None, required for 'semantic'
    seed : int

    Returns
    -------
    C : (Q, Q) array, or None if c_mode=='full' (use default in train.py)
    """
    if c_mode == "full":
        return None  # signal: use default compute_label_correlation
    elif c_mode == "identity":
        return compute_identity_C(train_target)
    elif c_mode == "random":
        return compute_random_C(train_target, seed=seed)
    elif c_mode == "semantic":
        if label_names is None:
            raise ValueError("c_mode='semantic' requires label_names")
        return compute_glove_semantic_C(label_names)
    else:
        raise ValueError(f"Unknown c_mode: {c_mode}. "
                         f"Valid: full, identity, random, semantic")

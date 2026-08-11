"""Kernel functions, port of CREM/help_function/kernelization.m (RBF + linear)."""
import numpy as np


def kernelization(XA, XB, ker_type="RBF", para=(0.05,)):
    """Return the n(xA) x m(xB) kernel matrix.

    Mirrors kernelization.m: samples are rows of XA/XB.
    ker_type: 'RBF' (para=(gamma,)) or 'linear'.
    """
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    if ker_type == "RBF":
        gamma = para[0]
        # exp(-gamma * ||xa - xb||^2)
        sq = (np.sum(XA**2, axis=1, keepdims=True)
              - 2.0 * XA @ XB.T
              + np.sum(XB**2, axis=1, keepdims=True).T)
        np.maximum(sq, 0.0, out=sq)  # guard against tiny negative round-off
        return np.exp(-gamma * sq)
    elif ker_type == "linear":
        return XA @ XB.T
    else:
        raise ValueError(f"unsupported kernel type: {ker_type}")

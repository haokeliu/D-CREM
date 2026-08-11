"""Hinge / smooth loss, port of CREM/help_function/computer_smooth_loss.m
and the grad_loss sub-function of Update_theta_P_R.m."""
import numpy as np


def hinge_loss(dist):
    """max(1 + dist, 0) — computer_smooth_loss.m, hinge_loss()."""
    return np.maximum(1.0 + dist, 0.0)


def grad_loss(dist_vector):
    """Sub-gradient of the hinge part used in Update_theta_P_R.m:
    1 where dist >= -1, else 0."""
    return (dist_vector >= -1).astype(float)


def compute_smooth_loss(K, Y, R, W, P, alpha):
    """Port of computer_smooth_loss(K,Y,R,W,P,alpha).

    K: N x N kernel, Y: N x Q (+/-1), R: (Q,), W: N x Q (only used for shape),
    P: N x Q. Returns (loss_p, p_loss per class).
    """
    q = W.shape[1]
    p_loss = np.zeros(q)
    for k in range(q):
        p = P[:, k]
        r = R[k]
        y = Y[:, k] == 1
        K_k = K[y, :]
        dist_delta = (1.0 + 2.0 * K_k @ p + p @ K @ p) - r**2
        p_loss[k] = alpha / y.sum() * np.sum(hinge_loss(dist_delta))
    return p_loss.sum(), p_loss

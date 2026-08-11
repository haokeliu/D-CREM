"""Alternating-optimization sub-updates, ports of
CREM/help_function/Update_theta_W.m and Update_theta_P_R.m."""
import numpy as np
from scipy.linalg import solve_sylvester

from .losses import grad_loss


def update_theta_W(W, K, b, Y, P, C, lamda1, lamda2, lamda3):
    """Port of Update_theta_W.m: solve the Sylvester equation
    Sy_A V + V Sy_B = Sy_C (same argument order as MATLAB's sylvester)."""
    n = W.shape[0]
    Sy_A = K + lamda1 * np.eye(n) + lamda3 * np.eye(n)
    Sy_B = lamda2 * C
    Sy_C = Y - np.ones((n, 1)) @ b.reshape(1, -1) + lamda1 * P
    return solve_sylvester(Sy_A, Sy_B, Sy_C)


def update_theta_P_R(K, Y, R, P, W, J, alpha, lamda1):
    """Port of Update_theta_P_R.m: per-class closed-form p update and
    gradient step on r (lr=0.1, mu=1)."""
    q = W.shape[1]
    lr = 0.1
    mu = 1.0
    P = P.copy()
    R = R.copy()
    for k in range(q):
        r = R[k]
        y = Y[:, k] == 1
        p = P[:, k]
        w = W[:, k]

        num_p = int(y.sum())
        beta = alpha / num_p
        j = J[k]              # N x num_p indicator matrix
        K_p = K[y, :]         # num_p x N

        dist_vec_p = (1.0 + 2.0 * K_p @ p + p @ K @ p) - r * r
        delta = grad_loss(dist_vec_p)
        p = (lamda1 * w - beta * j @ delta) / (lamda1 + beta * delta.sum())
        P[:, k] = p

        dist_vec_r = (1.0 + 2.0 * K_p @ p + p @ K @ p) - r * r
        delta_r = grad_loss(dist_vec_r)
        r = r + mu * lr * beta * delta_r.sum() * r
        R[k] = r
    return P, R

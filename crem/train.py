"""Training loop, port of CREM/CREM_train.m (quirks preserved verbatim:
lamda1/lamda3 are swapped, alpha is hard-coded to 1)."""
import numpy as np
import scipy.linalg as spla

from .config import effective_from_nominal
from .label_correlation import compute_label_correlation
from .losses import compute_smooth_loss
from .updates import update_theta_P_R, update_theta_W


def crem_train(train_target, Ktr, param, verbose=True, C_override=None):
    """train_target: Q x N (+/-1); Ktr: N x N kernel; param: dict with
    lamda1, lamda2, lamda3, alpha, K. Returns model dict {W, b, P, R}.

    If C_override is not None, it is used as the label correlation matrix
    (already in Laplacian form: diag(rowsum) - W).  Otherwise the standard
    co-occurrence C is computed from train_target.
    """
    q, n = train_target.shape
    # CREM_train.m lines 15-18: apply the MATLAB swap exactly once.  Public
    # callers pass nominal parameters; never pass get_params(...)[1] here.
    effective = effective_from_nominal(param)
    lamda1 = effective["lamda1"]
    lamda2 = effective["lamda2"]
    lamda3 = effective["lamda3"]
    alpha = effective["alpha"]

    theta = 1e-4
    ITER = 200

    Y = train_target.T.astype(float)  # N x Q
    K = Ktr
    b = np.zeros(q)

    # Pseudo-inverse of the (symmetric PSD) regularised kernel.  np.linalg.pinv
    # uses LAPACK gesdd, which intermittently fails to converge on large
    # ill-conditioned bibtex kernels (LinAlgError: SVD did not converge).
    # scipy.linalg.pinvh uses an eigh decomposition, which is stable here.
    theta_W = spla.pinvh(K + lamda3 * np.eye(n)) @ Y
    theta_P = theta_W.copy()

    In = np.eye(n)
    J = [In[:, Y[:, i] == 1] for i in range(q)]

    R = np.sqrt(2.0) * np.ones(q)

    if C_override is not None:
        C = C_override
    else:
        C = compute_label_correlation(train_target)
        C = np.diag(C @ np.ones(q)) - C

    loss = np.zeros((ITER, 6))

    if verbose:
        print("CREM training...")
        print("Alternative optimization...")
        print("Iter | loss difference")
    for it in range(ITER):
        theta_W_old = theta_W
        b_old = b
        theta_W = update_theta_W(theta_W, K, b, Y, theta_P, C,
                                 lamda1, lamda2, lamda3)
        b = -1.0 / n * (K @ theta_W - Y).T @ np.ones(n)

        theta_P, R = update_theta_P_R(K, Y, R, theta_P, theta_W, J,
                                      alpha, lamda1)

        loss_wb = 0.5 * np.linalg.norm(
            K @ theta_W + np.ones((n, 1)) @ b.reshape(1, -1) - Y, "fro") ** 2
        d = theta_W - theta_P
        loss_wp = 0.5 * lamda1 * np.trace(d.T @ K @ d)
        loss_tr = 0.5 * lamda2 * np.trace(K @ theta_W @ C @ theta_W.T)
        loss_reg = 0.5 * lamda3 * np.trace(theta_W.T @ K @ theta_W)
        loss_p, _ = compute_smooth_loss(K, Y, R, theta_W, theta_P, alpha)

        loss[it] = [loss_wb + loss_wp + loss_p + loss_tr + loss_reg,
                    loss_wb, loss_wp, loss_tr, loss_reg, loss_p]

        if verbose and (it + 1) % 10 == 0:
            print(f"{it + 1:4d} | {abs(loss[it, 0] - loss[it - 1, 0]):10.5f}")
        if it > 0:
            if (abs(loss[it, 0] - loss[it - 1, 0]) < theta * abs(loss[it, 0])
                    and it + 1 > 50):
                if verbose:
                    print(f"CREM converged at {it + 1}-th iteration.")
                loss = loss[:it + 1]
                break

    return {"W": theta_W, "b": b, "P": theta_P, "R": R, "loss": loss}

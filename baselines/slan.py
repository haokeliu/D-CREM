"""Python port of the official MATLAB implementation of SLAN.

SLAN was introduced in "Multi-label Open Set Recognition" (NeurIPS 2024).
This module follows the authors' MATLAB update equations while using SciPy
linear solvers instead of explicit ``H^(-1)`` expressions. The third-party
MATLAB package and its sample data are intentionally not redistributed here.

Conventions
-----------
``X`` stores samples by row (N x d), ``Y`` is Q x N with values in {-1, +1},
and returned OSR scores are *knownness* scores (larger means more likely to
contain known labels only). Protocol-v2 OSR labels use 1 for an instance that
contains at least one unknown label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import linalg

from crem.kernels import kernelization


@dataclass(frozen=True)
class SLANParams:
    """Hyperparameters from the official ``main.m`` unless noted otherwise."""

    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 10.0
    mu1: float = 0.1
    tau: float = 0.8
    kernel_gamma: float = 0.01
    outer_iterations: int = 200
    z_iterations: int = 100
    f_iterations: int = 100
    admm_iterations: int = 25
    z_learning_rate: float = 1e-4
    f_learning_rate: float = 1e-2
    tolerance: float = 1e-4

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def shrinkage(x: np.ndarray, kappa: float) -> np.ndarray:
    """Element-wise soft thresholding (official ``shrinkage.m``)."""

    return np.maximum(0.0, x - kappa) - np.maximum(0.0, -x - kappa)


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left, ord="fro")
    numerator = np.linalg.norm(left - right, ord="fro")
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def estimate_train_structure(
    x: np.ndarray, iterations: int = 25, tolerance: float = 1e-4
) -> np.ndarray:
    """Port of ``estimateS.m``.

    The MATLAB function is named ``estimateS`` but returns its final ``V``
    variable. This seemingly unusual detail is intentionally preserved.
    """

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    gram = x @ x.T
    identity = np.eye(n, dtype=np.float64)
    v = np.zeros((n, n), dtype=np.float64)
    e = np.zeros_like(v)
    dual1 = np.zeros_like(v)
    dual2 = np.zeros_like(v)
    rho1 = 1.0
    rho2 = 1.0
    lam = 1.0
    mu0 = 0.1

    for _ in range(iterations):
        s_hat = shrinkage(v + dual1 / rho1, mu0 / rho1)
        coefficient = lam * gram + (rho1 + rho2) * identity
        rhs = lam * gram + rho1 * s_hat - dual1 + rho2 * e - dual2
        v_hat = linalg.solve(
            coefficient, rhs, assume_a="pos", check_finite=False,
            overwrite_a=True, overwrite_b=True,
        )
        e_hat = v_hat + dual2 / rho2
        np.fill_diagonal(e_hat, 0.0)

        e = e_hat
        v = v_hat
        dual1 += rho1 * (v - s_hat)
        dual2 += rho2 * (v - e)
        rho1 = min(rho1 * 2.0, 1e7)
        rho2 = min(rho2 * 2.0, 1e7)

        if (_relative_residual(s_hat, v) < tolerance and
                _relative_residual(v, e) < tolerance):
            break
    return v


def estimate_test_structure(
    train_x: np.ndarray,
    test_x: np.ndarray,
    iterations: int = 25,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Port of the nested ``estimate_testS`` function in ``SLAN_test.m``."""

    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    n_train = train_x.shape[0]
    n_test = test_x.shape[0]
    gram = train_x @ train_x.T
    cross = train_x @ test_x.T
    identity = np.eye(n_train, dtype=np.float64)
    v = np.zeros((n_train, n_test), dtype=np.float64)
    s = np.zeros_like(v)
    dual = np.zeros_like(v)
    rho = 1.0
    lam = 1.0
    mu0 = 0.1

    for _ in range(iterations):
        s_hat = shrinkage(v + dual / rho, mu0 / rho)
        coefficient = lam * gram + rho * identity
        rhs = lam * cross + rho * s_hat - dual
        v = linalg.solve(
            coefficient, rhs, assume_a="pos", check_finite=False,
            overwrite_a=True, overwrite_b=True,
        )
        s = s_hat
        dual += rho * (v - s)
        rho = min(rho * 2.0, 1e7)
        if _relative_residual(s, v) < tolerance:
            break
    return s


def _matlab_quantile_index(tau: float, count: int) -> int:
    """Translate ``round(tau*n)`` from one-based MATLAB to Python.

    MATLAB rounds positive half values away from zero, unlike NumPy's banker's
    rounding. The result is clipped because a user-supplied tau may be 0 or 1.
    """

    matlab_index = int(np.floor(float(tau) * count + 0.5))
    matlab_index = min(count, max(1, matlab_index))
    return matlab_index - 1


class SLAN:
    """Paper-faithful SLAN estimator with a scikit-learn-like interface."""

    def __init__(self, params: Optional[SLANParams] = None):
        self.params = params or SLANParams()
        self.train_x_: Optional[np.ndarray] = None
        self.train_y_: Optional[np.ndarray] = None
        self.a_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None
        self.f_pool_: Optional[Tuple[np.ndarray, ...]] = None
        self.train_outputs_: Optional[np.ndarray] = None
        self.positive_differences_: Optional[Tuple[np.ndarray, ...]] = None
        self.loss_history_: Optional[np.ndarray] = None

    def _validate_xy(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("X and Y must both be two-dimensional")
        if y.shape[1] != x.shape[0]:
            raise ValueError("Y must have shape (num_labels, num_samples)")
        if not np.all(np.isin(y, (-1, 1))):
            raise ValueError("Y must contain only -1 and +1")
        if x.shape[0] < 2 or y.shape[0] < 2:
            raise ValueError("SLAN requires at least two samples and two known labels")
        return x, y.astype(np.float64, copy=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SLAN":
        p = self.params
        x, y = self._validate_xy(x, y)
        q, n = y.shape

        z = np.zeros((q, n), dtype=np.float64)
        wxb = np.zeros_like(z)
        f_pool = [np.zeros((q - 1, n), dtype=np.float64) for _ in range(q)]
        label_complements = [np.r_[0:i, i + 1:q] for i in range(q)]
        negative_masks = [y[i] != 1 for i in range(q)]

        kernel = kernelization(
            x, x, ker_type="RBF", para=(p.kernel_gamma,))
        structure = estimate_train_structure(
            x, iterations=p.admm_iterations, tolerance=p.tolerance)
        ci = (structure - np.eye(n)) @ (structure - np.eye(n)).T

        # H is constant across all outer iterations. Factoring it once is
        # algebraically identical to repeatedly using H^(-1) in MATLAB.
        h = kernel / p.mu1 + np.eye(n)
        h_factor = linalg.cho_factor(h, lower=True, check_finite=False)
        ones = np.ones(n, dtype=np.float64)
        h_inv_ones = linalg.cho_solve(h_factor, ones, check_finite=False)
        b_denominator = float(ones @ h_inv_ones)

        losses = []
        a = np.zeros_like(z)
        b = np.zeros(q, dtype=np.float64)
        for _ in range(p.outer_iterations):
            # Update Z. Selection matrices P_i are implemented as label index
            # operations, avoiding q dense (q-1)xq matrices without changing
            # the equation in UpdateZ.m.
            previous_change = None
            for _z in range(p.z_iterations):
                accumulated = np.zeros_like(z)
                for i, complement in enumerate(label_complements):
                    residual = z[complement] - f_pool[i]
                    residual[:, ~negative_masks[i]] = 0.0
                    accumulated[complement] += residual
                gradient = ((z - wxb) + p.gamma * (z - y) +
                            p.alpha * accumulated)
                z_old = z
                z = z - p.z_learning_rate * gradient
                change = np.linalg.norm(z - z_old, ord="fro")
                if (_z > 0 and previous_change is not None and
                        abs(change - previous_change) <
                        p.tolerance * max(change, np.finfo(float).eps)):
                    break
                previous_change = change

            # Update each F_i (UpdateF.m).
            for i, complement in enumerate(label_complements):
                f = f_pool[i]
                mask = negative_masks[i][None, :]
                target = z[complement]
                previous_change = None
                for _f in range(p.f_iterations):
                    gradient = p.beta * (f @ ci) + p.alpha * mask * (f - target)
                    f_old = f
                    f = f - p.f_learning_rate * gradient
                    change = np.linalg.norm(f - f_old, ord="fro")
                    if (_f > 10 and previous_change is not None and
                            abs(change - previous_change) <
                            p.tolerance * max(change, np.finfo(float).eps)):
                        break
                    previous_change = change
                f_pool[i] = f

            b = (z @ h_inv_ones) / b_denominator
            centered = z - b[:, None]
            a = linalg.cho_solve(
                h_factor, centered.T, check_finite=False).T
            wxb = (a @ kernel) / p.mu1 + b[:, None]

            wb_loss = 0.5 * np.linalg.norm(wxb - z, ord="fro") ** 2
            zy_loss = 0.5 * p.gamma * np.linalg.norm(y - z, ord="fro") ** 2
            # Preserve the exact scaling written in official SLAN_train.m.
            reg_loss = (0.5 * p.mu1 * (1.0 / (2.0 * p.mu1)) ** 2 *
                        np.trace(a @ kernel @ a.T))
            current_loss = float(wb_loss + zy_loss + reg_loss)
            losses.append(current_loss)
            if (len(losses) > 1 and
                    abs(losses[-1] - losses[-2]) <
                    p.tolerance * max(current_loss, np.finfo(float).eps)):
                break

        self.train_x_ = x
        self.train_y_ = y
        self.a_ = a
        self.b_ = b
        self.f_pool_ = tuple(f_pool)
        self.train_outputs_ = wxb
        positive_differences = []
        for i, complement in enumerate(label_complements):
            difference = np.sum(
                np.abs(self.f_pool_[i] - wxb[complement]), axis=0)
            positive_differences.append(
                np.sort(difference[y[i] == 1])[::-1])
        self.positive_differences_ = tuple(positive_differences)
        self.loss_history_ = np.asarray(losses, dtype=np.float64)
        return self

    def _check_fitted(self) -> None:
        if any(value is None for value in (
                self.train_x_, self.train_y_, self.a_, self.b_, self.f_pool_,
                self.positive_differences_)):
            raise RuntimeError("SLAN.fit must be called before prediction")

    def _prediction_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the tau-independent classifier output and reconstruction."""

        self._check_fitted()
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.train_x_.shape[1]:
            raise ValueError("X has an incompatible feature dimension")
        test_kernel = kernelization(
            x, self.train_x_, ker_type="RBF", para=(p.kernel_gamma,))
        outputs = (self.a_ @ test_kernel.T) / p.mu1 + self.b_[:, None]
        structure = estimate_test_structure(
            self.train_x_, x, iterations=p.admm_iterations,
            tolerance=p.tolerance)
        return outputs, structure

    def _decision_from_state(
        self, outputs: np.ndarray, structure: np.ndarray, tau: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Apply a tau candidate without repeating the expensive ADMM solve."""

        q = self.train_y_.shape[0]
        n_test = outputs.shape[1]
        votes = np.zeros((n_test, q), dtype=np.float64)
        differences = np.full_like(votes, np.nan)
        thresholds = np.full(q, np.nan, dtype=np.float64)
        class_used = np.zeros(q, dtype=bool)
        for i in range(q):
            sorted_positive = self.positive_differences_[i]
            if len(sorted_positive) == 0:
                continue
            class_used[i] = True
            complement = np.r_[0:i, i + 1:q]
            threshold = sorted_positive[_matlab_quantile_index(
                tau, len(sorted_positive))]
            thresholds[i] = threshold
            test_f = np.clip(self.f_pool_[i] @ structure, -1.0, 1.0)
            test_difference = np.sum(
                np.abs(outputs[complement] - test_f), axis=0)
            differences[:, i] = test_difference
            votes[:, i] = test_difference > threshold

        if not np.any(class_used):
            raise RuntimeError("No class has positive training examples")
        unknown_vote_fraction = votes[:, class_used].mean(axis=1)
        diagnostics = {
            "thresholds": thresholds,
            "class_used": class_used,
            "unknown_vote_fraction": unknown_vote_fraction,
            "differences": differences,
        }
        return 1.0 - unknown_vote_fraction, diagnostics

    def decision_function(
        self, x: np.ndarray, tau: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Return known-label outputs, knownness scores, and diagnostics.

        The original implementation emits majority-vote binary decisions.
        ``known_scores`` retains that exact decision rule as a continuous vote
        fraction: 1 means every usable label voted known and 0 means every
        usable label voted unknown.
        """

        self._check_fitted()
        p = self.params
        tau = p.tau if tau is None else float(tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must lie in [0, 1]")

        outputs, structure = self._prediction_state(x)
        known_scores, diagnostics = self._decision_from_state(
            outputs, structure, tau)
        return outputs, known_scores, diagnostics

    def predict_known(self, x: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
        """Return the official majority decision (True means known-only)."""

        _, score, _ = self.decision_function(x, tau=tau)
        return score > 0.5

    def select_tau(
        self,
        val_x: np.ndarray,
        val_osr_labels: np.ndarray,
        candidates: Iterable[float],
    ) -> Tuple[float, Dict[float, Dict[str, float]]]:
        """Select tau using validation labels only; test labels are never read."""

        from dcrem.eval.osr_metrics import compute_osr_metrics

        details: Dict[float, Dict[str, float]] = {}
        best_tau = self.params.tau
        best_auroc = -np.inf
        outputs, structure = self._prediction_state(val_x)
        for candidate in candidates:
            candidate = float(candidate)
            if not 0.0 <= candidate <= 1.0:
                raise ValueError("tau candidates must lie in [0, 1]")
            known_score, _ = self._decision_from_state(
                outputs, structure, candidate)
            metrics = compute_osr_metrics(known_score, val_osr_labels)
            details[candidate] = metrics
            auroc = metrics["AUROC"]
            if np.isfinite(auroc) and auroc > best_auroc:
                best_auroc = auroc
                best_tau = candidate
        return best_tau, details

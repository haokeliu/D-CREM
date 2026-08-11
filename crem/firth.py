"""Self-contained Firth (Jeffreys-prior penalized) binomial logistic
regression, replacing MATLAB's
fitglm(x, y, 'Distribution','binomial','LikelihoodPenalty','jeffreys-prior').

Standard IRLS/Newton iteration on the design matrix X = [1, x]:
    p  = sigmoid(X beta)
    W  = diag(p(1-p))
    h  = leverage of W^1/2 X (X'WX)^{-1} X' W^1/2
    score = X' (y - p + h * (0.5 - p))        # Firth-modified score
    step  = (X'WX)^{-1} score
Linear systems are solved via Cholesky with a small jitter fallback.
"""
import numpy as np


class FirthLogisticRegression:
    def __init__(self, tol=1e-8, max_iter=200, jitter=1e-10):
        self.tol = tol
        self.max_iter = max_iter
        self.jitter = jitter
        self.coef_ = None

    @staticmethod
    def _solve(A, b, jitter):
        """Solve A x = b via Cholesky, escalating jitter if singular."""
        eps = jitter
        for _ in range(8):
            try:
                L = np.linalg.cholesky(A + eps * np.eye(A.shape[0]))
                z = np.linalg.solve(L, b)
                return np.linalg.solve(L.T, z)
            except np.linalg.LinAlgError:
                eps *= 100.0
        return np.linalg.solve(A + eps * np.eye(A.shape[0]), b)

    def _sigmoid(self, eta):
        """Numerically stable sigmoid with clipping."""
        eta = np.clip(np.asarray(eta, dtype=float), -100.0, 100.0)
        return np.where(eta >= 0,
                        1.0 / (1.0 + np.exp(-eta)),
                        np.exp(eta) / (1.0 + np.exp(eta)))

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        X = np.column_stack([np.ones_like(x), x])
        beta = np.zeros(2)
        for _ in range(self.max_iter):
            eta = X @ beta
            p = self._sigmoid(eta)
            w = np.clip(p * (1.0 - p), 1e-12, None)
            XtW = X.T * w                      # 2 x n
            XtWX = XtW @ X
            # leverage h_i = w_i * x_i' (X'WX)^{-1} x_i
            Z = self._solve(XtWX, X.T, self.jitter)   # 2 x n
            h = w * np.einsum("ni,in->n", X, Z)
            score = X.T @ (y - p + h * (0.5 - p))
            step = self._solve(XtWX, score, self.jitter)
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float).ravel()
        eta = self.coef_[0] + self.coef_[1] * x
        return self._sigmoid(eta)

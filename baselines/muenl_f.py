"""NumPy/scikit-learn port of the official MuENL forest detector.

The official MATLAB package (http://www.lamda.nju.edu.cn/files/MuENL.zip)
accompanies "Multi-label Learning with Emerging New Labels" (TKDE 2018).
This module ports the static MuENL-F detector used
as an MLOSR baseline: a pairwise-ranking linear classifier supplies label
predictions, and an ensemble of randomized clustering trees estimates whether
a sample lies outside the training support.

The third-party MATLAB package is intentionally not redistributed.  Samples
are rows (N x d); public targets are Q x N with values in {-1, +1}.  The
detector's raw score is the fraction of trees voting "unknown".
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


@dataclass(frozen=True)
class MuENLFParams:
    """Formal defaults from the CREM paper's MUENL-F comparison.

    The official standalone example uses a smaller ``psi=128``, three
    prediction attributes, and height five.  CREM Appendix A.1 instead fixes
    ``|q|=5, psi=256, g=100, e_m=9``; those comparison settings take priority
    for this repository's paper-facing runs.
    """

    weight_ranking_loss: float = 1.0
    weight_reg: float = 1e-4
    classifier_sweeps: int = 20
    classifier_inner_iterations: int = 10
    classifier_grad_tolerance: float = 1e-3
    psi: int = 256
    num_trees: int = 100
    num_features: int = 5
    num_predictions: int = 5
    max_height: int = 9
    min_leaf: int = 4
    radius_ratio: float = 1.0
    split_retries: int = 10
    random_state: int = 1

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class _Node:
    center: np.ndarray
    radius: float
    feature_indices: Optional[np.ndarray] = None
    prediction_indices: Optional[np.ndarray] = None
    left_center: Optional[np.ndarray] = None
    right_center: Optional[np.ndarray] = None
    left: Optional[int] = None
    right: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance with round-off clipped at zero."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.maximum(
        np.sum(a * a, axis=-1) + np.sum(b * b, axis=-1)
        - 2.0 * np.sum(a * b, axis=-1),
        0.0,
    )


class PairwiseRankingLinear:
    """Port of the official ``PLRTrain.m`` / ``PLRPredict.m`` classifier."""

    def __init__(self, params: MuENLFParams):
        self.params = params
        self.coef_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    @staticmethod
    def _validate(x: np.ndarray, y_qn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y_qn = np.asarray(y_qn)
        if x.ndim != 2 or y_qn.ndim != 2:
            raise ValueError("X and Y must both be two-dimensional")
        if y_qn.shape[1] != x.shape[0]:
            raise ValueError("Y must have shape (num_labels, num_samples)")
        if not np.all(np.isin(y_qn, (-1, 1))):
            raise ValueError("Y must contain only -1 and +1")
        if x.shape[0] < 2 or y_qn.shape[0] < 2:
            raise ValueError("MuENL-F requires at least two samples and labels")
        return x, y_qn.T.astype(np.float64, copy=False)

    def _cost_gradient(
        self, label: int, w: np.ndarray, x_aug: np.ndarray,
        y_nq: np.ndarray, models: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        p = self.params
        n, q = y_nq.shape
        margins = 1.0 - y_nq[:, label] * (x_aug @ w)
        hinge = np.maximum(margins, 0.0)
        active = margins > 0.0
        gradient = -(y_nq[active, label, None] * x_aug[active]).sum(axis=0) / n

        ranking_cost = 0.0
        ranking_gradient = np.zeros_like(w)
        for other in range(q):
            if other == label:
                continue
            delta = y_nq[:, label] - y_nq[:, other]
            ranking_margin = 1.0 - delta * (x_aug @ (w - models[other]))
            ranking_cost += np.maximum(ranking_margin, 0.0).sum()
            ranking_active = ranking_margin > 0.0
            ranking_gradient -= (
                delta[ranking_active, None] * x_aug[ranking_active]
            ).sum(axis=0) / (n * q)

        # These factors intentionally match UpdateWi.m, including its cost /
        # gradient scaling mismatch.
        cost = (
            hinge.mean()
            + p.weight_ranking_loss * ranking_cost / (n * (q - 1))
            + p.weight_reg * float(w @ w)
        )
        gradient += p.weight_ranking_loss * ranking_gradient + p.weight_reg * w
        return float(cost), gradient

    def fit(self, x: np.ndarray, y_qn: np.ndarray) -> "PairwiseRankingLinear":
        x, y_nq = self._validate(x, y_qn)
        p = self.params
        rng = np.random.default_rng(p.random_state)
        x_aug = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
        models = rng.random((y_nq.shape[1], x_aug.shape[1]))
        self.loss_history_ = []

        for _ in range(p.classifier_sweeps):
            sweep_cost = 0.0
            for label in range(y_nq.shape[1]):
                w = models[label].copy()
                for _inner in range(p.classifier_inner_iterations):
                    cost, gradient = self._cost_gradient(
                        label, w, x_aug, y_nq, models)
                    grad_norm = float(np.linalg.norm(gradient))
                    if grad_norm <= p.classifier_grad_tolerance:
                        break
                    step = 1.0
                    directional = grad_norm * grad_norm
                    while step > 1e-12:
                        candidate = w - step * gradient
                        candidate_cost, _ = self._cost_gradient(
                            label, candidate, x_aug, y_nq, models)
                        if candidate_cost <= cost - 1e-4 * step * directional:
                            w = candidate
                            break
                        step *= 0.5
                    if step <= 1e-12:
                        break
                models[label] = w
                sweep_cost += self._cost_gradient(
                    label, w, x_aug, y_nq, models)[0]
            self.loss_history_.append(sweep_cost)

        self.coef_ = models
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("PairwiseRankingLinear.fit must be called first")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] + 1 != self.coef_.shape[1]:
            raise ValueError("X has an incompatible feature dimension")
        return np.column_stack((x, np.ones(x.shape[0]))) @ self.coef_.T


class MuENLForest:
    """Randomized clustering forest from ``MuENLForest.m``."""

    def __init__(self, params: Optional[MuENLFParams] = None):
        self.params = params or MuENLFParams()
        self.trees_: Optional[List[List[_Node]]] = None
        self.feature_dim_: Optional[int] = None
        self.prediction_dim_: Optional[int] = None

    def _build_tree(
        self, features: np.ndarray, predictions: np.ndarray,
        root_indices: np.ndarray, rng: np.random.Generator,
    ) -> List[_Node]:
        p = self.params
        k_feature = min(p.num_features, features.shape[1])
        k_prediction = min(p.num_predictions, predictions.shape[1])
        nodes: List[_Node] = []
        pending: List[Tuple[np.ndarray, int, Optional[int], Optional[bool]]] = [
            (root_indices, 1, None, None)
        ]

        while pending:
            indices, height, parent, is_left = pending.pop(0)
            node_features = features[indices]
            center = node_features.mean(axis=0)
            radius = float(np.sqrt(_squared_distance(
                node_features, np.broadcast_to(center, node_features.shape)
            ).max()))
            node_index = len(nodes)
            nodes.append(_Node(center=center, radius=radius))
            if parent is not None:
                if is_left:
                    nodes[parent].left = node_index
                else:
                    nodes[parent].right = node_index

            if len(indices) < 2 * p.min_leaf or height >= p.max_height:
                continue
            for _ in range(p.split_retries):
                feature_indices = rng.choice(
                    features.shape[1], k_feature, replace=False)
                prediction_indices = rng.choice(
                    predictions.shape[1], k_prediction, replace=False)
                sample = np.column_stack((
                    features[indices][:, feature_indices],
                    predictions[indices][:, prediction_indices],
                ))
                kmeans_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                # MATLAB retries failed/degenerate k-means splits silently.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    labels = KMeans(
                        n_clusters=2, n_init=1, random_state=kmeans_seed,
                    ).fit_predict(sample)
                left_indices = indices[labels == 0]
                right_indices = indices[labels == 1]
                if min(len(left_indices), len(right_indices)) < p.min_leaf:
                    continue
                centers = np.vstack((sample[labels == 0].mean(axis=0),
                                     sample[labels == 1].mean(axis=0)))
                node = nodes[node_index]
                node.feature_indices = feature_indices
                node.prediction_indices = prediction_indices
                node.left_center = centers[0]
                node.right_center = centers[1]
                # Breadth-first insertion matches the MATLAB node traversal.
                pending.extend([
                    (left_indices, height + 1, node_index, True),
                    (right_indices, height + 1, node_index, False),
                ])
                break
        return nodes

    def fit(
        self, features: np.ndarray, predictions: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "MuENLForest":
        features = np.asarray(features, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        if features.ndim != 2 or predictions.ndim != 2:
            raise ValueError("features and predictions must be two-dimensional")
        if features.shape[0] != predictions.shape[0]:
            raise ValueError("features and predictions must share sample count")
        if features.shape[0] < 1 or features.shape[1] < 1 or predictions.shape[1] < 1:
            raise ValueError("empty feature or prediction matrices are unsupported")

        n = features.shape[0]
        if sample_weights is None:
            probability = None
        else:
            weights = np.asarray(sample_weights, dtype=np.float64).ravel()
            if weights.shape != (n,) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("sample_weights must be nonnegative and nonzero")
            probability = weights / weights.sum()

        p = self.params
        rng = np.random.default_rng(p.random_state)
        self.trees_ = []
        for _ in range(p.num_trees):
            replace = n < p.psi
            indices = rng.choice(n, size=p.psi, replace=replace, p=probability)
            self.trees_.append(self._build_tree(
                features, predictions, indices, rng))
        self.feature_dim_ = features.shape[1]
        self.prediction_dim_ = predictions.shape[1]
        return self

    def _check_inputs(
        self, features: np.ndarray, predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.trees_ is None:
            raise RuntimeError("MuENLForest.fit must be called before scoring")
        features = np.asarray(features, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        if (features.ndim != 2 or predictions.ndim != 2 or
                features.shape[0] != predictions.shape[0] or
                features.shape[1] != self.feature_dim_ or
                predictions.shape[1] != self.prediction_dim_):
            raise ValueError("incompatible features or predictions")
        return features, predictions

    def score_samples(
        self, features: np.ndarray, predictions: np.ndarray,
        radius_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Return unknown-vote fraction (larger means more likely unknown)."""

        features, predictions = self._check_inputs(features, predictions)
        ratio = self.params.radius_ratio if radius_ratio is None else float(radius_ratio)
        if ratio < 0:
            raise ValueError("radius_ratio must be nonnegative")
        votes = np.zeros((features.shape[0], len(self.trees_)), dtype=np.float64)
        for tree_index, tree in enumerate(self.trees_):
            for sample_index, (feature, prediction) in enumerate(
                    zip(features, predictions)):
                node_index = 0
                while not tree[node_index].is_leaf:
                    node = tree[node_index]
                    sample = np.concatenate((
                        feature[node.feature_indices],
                        prediction[node.prediction_indices],
                    ))
                    left_distance = _squared_distance(sample, node.left_center)
                    right_distance = _squared_distance(sample, node.right_center)
                    node_index = node.left if left_distance < right_distance else node.right
                leaf = tree[node_index]
                distance = float(np.sqrt(_squared_distance(feature, leaf.center)))
                votes[sample_index, tree_index] = distance > leaf.radius * ratio
        return votes.mean(axis=1)


class MuENLF:
    """End-to-end static MuENL-F baseline (PLR classifier + forest)."""

    def __init__(self, params: Optional[MuENLFParams] = None):
        self.params = params or MuENLFParams()
        self.classifier = PairwiseRankingLinear(self.params)
        self.forest = MuENLForest(self.params)

    def fit(self, x: np.ndarray, y_qn: np.ndarray) -> "MuENLF":
        self.classifier.fit(x, y_qn)
        predictions = self.classifier.decision_function(x)
        self.forest.fit(x, predictions)
        return self

    def decision_function(
        self, x: np.ndarray, radius_ratio: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Q x N classifier outputs and knownness scores."""

        predictions = self.classifier.decision_function(x)
        unknown_score = self.forest.score_samples(
            x, predictions, radius_ratio=radius_ratio)
        return predictions.T, 1.0 - unknown_score

    def select_radius_ratio(
        self, val_x: np.ndarray, val_osr_labels: np.ndarray,
        candidates: Iterable[float],
    ) -> Tuple[float, Dict[float, Dict[str, float]]]:
        from dcrem.eval.osr_metrics import compute_osr_metrics

        predictions = self.classifier.decision_function(val_x)
        details: Dict[float, Dict[str, float]] = {}
        best_ratio = self.params.radius_ratio
        best_auroc = -np.inf
        for candidate in candidates:
            candidate = float(candidate)
            if candidate < 0:
                raise ValueError("radius candidates must be nonnegative")
            unknown_score = self.forest.score_samples(
                val_x, predictions, radius_ratio=candidate)
            metrics = compute_osr_metrics(1.0 - unknown_score, val_osr_labels)
            details[candidate] = metrics
            auroc = metrics["AUROC"]
            if np.isfinite(auroc) and auroc > best_auroc:
                best_auroc = auroc
                best_ratio = candidate
        return best_ratio, details

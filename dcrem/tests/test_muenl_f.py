"""Numerical and protocol tests for the MuENL-F Python port."""

import numpy as np

from baselines.muenl_f import (
    MuENLF,
    MuENLForest,
    MuENLFParams,
    PairwiseRankingLinear,
)


def _toy_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(16, 4))
    y = np.where(rng.random((3, 16)) > 0.5, 1, -1)
    y[:, 0] = 1
    y[:, 1] = -1
    return x, y


def _quick_params(seed=7):
    return MuENLFParams(
        classifier_sweeps=1,
        classifier_inner_iterations=2,
        psi=12,
        num_trees=4,
        num_features=2,
        num_predictions=2,
        max_height=3,
        min_leaf=2,
        split_retries=3,
        random_state=seed,
    )


def test_plr_shapes_and_determinism():
    x, y = _toy_data()
    params = _quick_params()
    first = PairwiseRankingLinear(params).fit(x, y).decision_function(x[:3])
    second = PairwiseRankingLinear(params).fit(x, y).decision_function(x[:3])
    assert first.shape == (3, 3)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_forest_training_samples_score_inside_unit_interval():
    x, y = _toy_data()
    params = _quick_params()
    classifier = PairwiseRankingLinear(params).fit(x, y)
    predictions = classifier.decision_function(x)
    scores = MuENLForest(params).fit(x, predictions).score_samples(x, predictions)
    assert scores.shape == (len(x),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_larger_radius_cannot_create_more_unknown_votes():
    x, y = _toy_data()
    params = _quick_params()
    classifier = PairwiseRankingLinear(params).fit(x, y)
    predictions = classifier.decision_function(x)
    forest = MuENLForest(params).fit(x, predictions)
    small = forest.score_samples(x, predictions, radius_ratio=0.5)
    large = forest.score_samples(x, predictions, radius_ratio=2.0)
    assert np.all(large <= small)


def test_end_to_end_outputs_use_public_q_by_n_orientation():
    x, y = _toy_data()
    model = MuENLF(_quick_params()).fit(x, y)
    outputs, known_scores = model.decision_function(x[:5])
    assert outputs.shape == (3, 5)
    assert known_scores.shape == (5,)
    assert np.isfinite(outputs).all()


def test_radius_selection_only_consumes_supplied_validation_fold():
    x, y = _toy_data()
    model = MuENLF(_quick_params()).fit(x[:12], y[:, :12])
    labels = np.array([0, 0, 1, 1])
    selected, details = model.select_radius_ratio(
        x[12:], labels, [0.5, 1.0, 1.5])
    assert selected in details
    assert set(details) == {0.5, 1.0, 1.5}


def test_rejects_wrong_target_orientation():
    x, y = _toy_data()
    try:
        MuENLF(_quick_params()).fit(x, y.T)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("wrong target orientation was accepted")

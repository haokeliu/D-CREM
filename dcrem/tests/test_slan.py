"""Small numerical and protocol tests for the official SLAN Python port."""

import numpy as np

from baselines.slan import (
    SLAN,
    SLANParams,
    _matlab_quantile_index,
    estimate_test_structure,
    estimate_train_structure,
    shrinkage,
)


def _toy_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 3))
    y = np.where(rng.random((3, 8)) > 0.5, 1, -1)
    # Guarantee both signs and at least one positive for every label.
    y[:, 0] = 1
    y[:, 1] = -1
    return x, y


def test_slan_shrinkage_matches_soft_threshold():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(shrinkage(x, 0.5), [-1.5, 0, 0, 0, 1.5])


def test_slan_uses_matlab_rounding_and_one_based_index():
    assert _matlab_quantile_index(0.5, 5) == 2  # MATLAB round(2.5)=3
    assert _matlab_quantile_index(0.0, 5) == 0
    assert _matlab_quantile_index(1.0, 5) == 4


def test_slan_structure_shapes_are_finite():
    x, _ = _toy_data()
    train_s = estimate_train_structure(x, iterations=2)
    test_s = estimate_test_structure(x, x[:3], iterations=2)
    assert train_s.shape == (8, 8)
    assert test_s.shape == (8, 3)
    assert np.isfinite(train_s).all()
    assert np.isfinite(test_s).all()


def test_slan_fit_and_prediction_are_deterministic():
    x, y = _toy_data()
    params = SLANParams(
        outer_iterations=2, z_iterations=2, f_iterations=2,
        admm_iterations=2)
    first = SLAN(params).fit(x, y)
    second = SLAN(params).fit(x, y)
    out1, score1, diag1 = first.decision_function(x[:3])
    out2, score2, diag2 = second.decision_function(x[:3])
    assert out1.shape == (3, 3)
    assert score1.shape == (3,)
    assert np.all((score1 >= 0) & (score1 <= 1))
    np.testing.assert_allclose(out1, out2)
    np.testing.assert_allclose(score1, score2)
    np.testing.assert_array_equal(diag1["class_used"], diag2["class_used"])


def test_slan_rejects_wrong_target_orientation():
    x, y = _toy_data()
    try:
        SLAN().fit(x, y.T)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("wrong target orientation was accepted")

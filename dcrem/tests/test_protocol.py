"""Regression tests for the leakage-free protocol and experiment runners."""

import random

import numpy as np
import torch


def test_split_is_disjoint_deterministic_and_train_fitted():
    from crem.data import apply_crem_split

    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 8))
    # Make feature 7 high-variance only outside the eventual training fold.
    Y = (rng.random((100, 6)) > 0.7).astype(float)
    names = [f"L{i}" for i in range(Y.shape[1])]

    first = apply_crem_split(
        X, Y, names, known_ratio=0.5, seed=11, standardize=True,
        train_ratio=0.4, val_ratio=0.1, target_d=4,
        feature_selector="variance")
    second = apply_crem_split(
        X, Y, names, known_ratio=0.5, seed=11, standardize=True,
        train_ratio=0.4, val_ratio=0.1, target_d=4,
        feature_selector="variance")

    tr, va, te = map(set, (
        first["train_indices"], first["val_indices"], first["test_indices"]))
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == len(X)
    assert np.array_equal(first["feature_indices"], second["feature_indices"])
    assert np.allclose(first["train_data"], second["train_data"])

    raw_train = X[first["train_indices"]]
    expected = np.argsort(-raw_train.var(axis=0))[:4]
    assert set(first["feature_indices"]) == set(expected)
    selected_train = raw_train[:, first["feature_indices"]]
    assert np.allclose(first["scaler_mean"], selected_train.mean(axis=0, keepdims=True))
    assert np.allclose(first["train_data"].mean(axis=0), 0.0, atol=1e-10)


def test_test_changes_cannot_change_train_preprocessing():
    from crem.data import apply_crem_split

    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 6))
    Y = (rng.random((80, 5)) > 0.65).astype(float)
    names = [f"L{i}" for i in range(5)]
    kwargs = dict(known_ratio=0.6, seed=3, standardize=True,
                  train_ratio=0.4, val_ratio=0.1, target_d=3,
                  feature_selector="variance")
    base = apply_crem_split(X, Y, names, **kwargs)

    changed = X.copy()
    non_train = np.concatenate([base["val_indices"], base["test_indices"]])
    changed[non_train] += 10000.0
    rerun = apply_crem_split(changed, Y, names, **kwargs)

    assert np.array_equal(base["feature_indices"], rerun["feature_indices"])
    assert np.allclose(base["scaler_mean"], rerun["scaler_mean"])
    assert np.allclose(base["scaler_std"], rerun["scaler_std"])
    assert np.allclose(base["train_data"], rerun["train_data"])


def test_reproducibility_seeds_all_rngs():
    from dcrem.reproducibility import seed_everything

    seed_everything(123)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(123)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_crem_effective_parameters_are_swapped_once():
    from crem.config import effective_from_nominal, get_params

    nominal, logged = get_params(
        param_override={"lamda1": 2, "lamda2": 0.3, "lamda3": 7,
                        "alpha": 9, "gamma": 0.2, "K": 2})
    actual = effective_from_nominal(nominal)
    assert actual == logged
    assert actual["lamda1"] == 7       # W-P coupling
    assert actual["lamda3"] == 2       # ridge
    assert actual["alpha"] == 1        # faithful MATLAB quirk


def test_unified_crem_runner_passes_nominal_parameters(monkeypatch):
    """Regression guard for the former get_params -> crem_train double swap."""
    import run_crem

    nominal = {
        "lamda1": 2, "lamda2": 0.3, "lamda3": 7,
        "alpha": 1, "gamma": 0.2, "K": 1,
    }
    effective = {
        "lamda1": 7, "lamda2": 0.3, "lamda3": 2,
        "alpha": 1, "gamma": 0.2, "K": 1,
    }
    data = {
        "train_data": np.zeros((4, 2)),
        "val_data": np.zeros((2, 2)),
        "test_data": np.zeros((2, 2)),
        "train_target": np.array([[1, -1, 1, -1]], dtype=np.int16),
        "val_target": np.array([[1, -1]], dtype=np.int16),
        "test_target": np.array([[1, -1]], dtype=np.int16),
        "val_osr_labels": np.array([[0, 1]], dtype=np.uint8),
        "osr_labels": np.array([[0, 1]], dtype=np.uint8),
        "known_label_names": ["L0"],
    }
    captured = {}
    monkeypatch.setattr(
        run_crem, "get_params", lambda *args, **kwargs: (nominal, effective))
    monkeypatch.setattr(run_crem, "get_dataset", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        run_crem, "kernelization",
        lambda left, right, *args: np.zeros((len(left), len(right))))

    def fake_train(target, kernel, param, **kwargs):
        captured["param"] = param
        return {"W": np.zeros((4, 1)), "b": np.zeros(1),
                "P": np.zeros((4, 1)), "R": np.ones(1)}

    metrics = {
        "AUROC": 0.5, "AUPR": 0.5, "macroAUC": 0.5,
        "AveragePrecision": 0.5, "RankingLoss": 0.5,
        "Coverage": 0.5, "OneError": 0.5, "best_K": 1,
    }
    monkeypatch.setattr(run_crem, "crem_train", fake_train)
    monkeypatch.setattr(
        run_crem, "crem_validate_and_test",
        lambda *args, **kwargs: (metrics, {"selected_K": 1}))
    monkeypatch.setattr(run_crem, "save_run", lambda *args, **kwargs: "result.json")

    assert run_crem.run_single("dummy", 0.5, 0, verbose=False) == "result.json"
    assert captured["param"] == nominal


def test_ablation_runner_passes_run_tag(monkeypatch):
    from scripts import run_phase3

    captured = {}
    monkeypatch.setattr(run_phase3, "dcrem_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        run_phase3, "run_cmd",
        lambda cmd, desc="": captured.setdefault("cmd", cmd) is cmd)
    assert run_phase3.run_dcrem_single(
        "slashdot", "A", 0.5, 0, ablation_id="S1",
        extra_flags=["--primary-score", "logit"], python_bin="python")
    cmd = captured["cmd"]
    tag_index = cmd.index("--run-tag")
    assert cmd[tag_index + 1] == "ablation_modeA_core_S1"
    assert cmd[cmd.index("--mode") + 1] == "A"
    assert cmd[cmd.index("--batch-size") + 1] == "128"


def test_mode_b_ablation_runner_uses_separate_namespace(monkeypatch):
    from scripts import run_phase3

    captured = {}
    monkeypatch.setattr(run_phase3, "dcrem_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        run_phase3, "run_cmd",
        lambda cmd, desc="": captured.setdefault("cmd", cmd) is cmd)
    assert run_phase3.run_dcrem_single(
        "enron", "B", 0.3, 0, ablation_id="N1",
        extra_flags=["--no-l2norm"], python_bin="python",
        defer_summary=True)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--run-tag") + 1] == "ablation_modeB_core_N1"
    assert "--skip-summary-refresh" in cmd


def test_mode_b_sensitivity_matrix_is_pre_registered():
    from scripts.run_phase3 import SENSITIVITY_CONFIGS

    assert list(SENSITIVITY_CONFIGS) == [
        "REF", "T1", "T5", "T25", "T50", "L01", "L03", "L3", "L10",
        "B0", "B003", "B03", "B1", "D64", "D256"]
    assert SENSITIVITY_CONFIGS["REF"]["flags"] == []
    assert SENSITIVITY_CONFIGS["T1"]["flags"] == ["--block-interval", "1"]
    assert SENSITIVITY_CONFIGS["D256"]["flags"] == ["--embedding-dim", "256"]


def test_formal_ablation_config_is_pre_registered():
    from scripts.run_phase3 import (
        ABLATION_CONFIGS, ABLATION_DATASETS, PAPER_CORE_FLAGS)

    assert list(ABLATION_CONFIGS) == ["full", "N1", "E1", "S1", "U1"]
    assert "--no-correlation" in PAPER_CORE_FLAGS
    assert "--alpha" in PAPER_CORE_FLAGS
    assert "--gamma-div" in PAPER_CORE_FLAGS
    assert "--no-warmup" in PAPER_CORE_FLAGS
    assert "--classifier-induced-reciprocal" in PAPER_CORE_FLAGS
    assert "--primary-score" in PAPER_CORE_FLAGS
    assert ABLATION_DATASETS == ["enron", "slashdot", "bibtex"]


def test_lite_development_configs_are_validation_only_candidates():
    from scripts.run_phase3 import LITE_DEVELOPMENT_CONFIGS

    assert list(LITE_DEVELOPMENT_CONFIGS) == [
        "F0", "M1", "L1", "L2", "L3",
        "G1", "G2", "G3", "G4", "G5", "G6",
        "H1", "H2", "H3", "H4", "V31", "V32", "V33",
        "V34", "V35", "V36", "V37", "V38",
        "B0", "B1", "B2", "B3", "C1", "C2", "C3",
        "D1", "D2", "D3", "R1", "R2", "R3",
        "Q0", "Q1", "Q2", "Q3"]
    assert "--open-reduction" in LITE_DEVELOPMENT_CONFIGS["M1"]["flags"]
    assert "--radius-free-open" in LITE_DEVELOPMENT_CONFIGS["L1"]["flags"]
    assert "--label-rank-weight" in LITE_DEVELOPMENT_CONFIGS["D1"]["flags"]
    assert "--label-rank-hard-fraction" in LITE_DEVELOPMENT_CONFIGS["R1"]["flags"]
    assert "--reciprocal-prototypes" in LITE_DEVELOPMENT_CONFIGS["Q1"]["flags"]
    assert "--pseudo-variant" in LITE_DEVELOPMENT_CONFIGS["B3"]["flags"]


def test_validation_selects_k_and_test_keeps_it():
    from dcrem.eval.osr_metrics import evaluate_fixed_k, k_search_osr

    val_dist = np.array([[5.0, 0.0], [4.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    val_labels = np.array([0, 0, 1, 1])
    selected, _ = k_search_osr(val_dist, val_labels, ks=[1, 2])

    test_dist = np.array([[0.0, 5.0], [2.0, 2.0], [4.0, 0.0], [1.0, 1.0]])
    test_labels = np.array([0, 1, 0, 1])
    result = evaluate_fixed_k(test_dist, test_labels, selected["best_K"])
    assert result["best_K"] == selected["best_K"]


def test_image_loader_evaluation_batches_inputs_without_reordering():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from dcrem.scripts.train import _forward_inputs

    class _Head(torch.nn.Module):
        def forward(self, features):
            return torch.stack((features[:, 0], -features[:, 0]), dim=1)

    class _Trainer:
        head = _Head()

        @staticmethod
        def _forward_encoder(inputs):
            return inputs.mean(dim=(2, 3))

    images = torch.arange(
        5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    labels = torch.ones(5, 2)
    loader = DataLoader(
        TensorDataset(images, labels), batch_size=2, shuffle=False)
    features, logits = _forward_inputs(
        _Trainer(), loader, torch.device("cpu"))

    assert features.shape == (5, 3)
    assert logits.shape == (5, 2)
    assert torch.equal(features, images.mean(dim=(2, 3)))

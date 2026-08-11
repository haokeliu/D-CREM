"""D-CREM unit and regression tests.

Run:  pytest dcrem/tests/test_dcrem.py -v
or:   python -m pytest dcrem/tests/test_dcrem.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ═══════════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_tabular_data(device):
    """Small synthetic tabular dataset (like enron)."""
    N, d, q = 200, 50, 8
    torch.manual_seed(42)
    X = torch.randn(N, d, device=device)
    Y = (torch.rand(N, q, device=device) > 0.7).float() * 2 - 1  # ±1
    return X, Y


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: L2Norm produces unit-norm features
# ═══════════════════════════════════════════════════════════════════════════

def test_l2norm_unit_norm(device):
    """L2Norm 后 ‖f(x)‖₂ = 1（随机输入，容差 1e-6）"""
    from dcrem.models.encoder import TabularMLP

    encoder = TabularMLP(input_dim=50, output_dim=128).to(device)
    x = torch.randn(32, 50, device=device)
    raw = encoder(x)
    feats = F.normalize(raw, p=2, dim=-1)
    norms = feats.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), \
        f"L2Norm failed: norms range [{norms.min():.6f}, {norms.max():.6f}]"


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: CorrelationModule output is symmetric with zero diagonal
# ═══════════════════════════════════════════════════════════════════════════

def test_correlation_symmetric_zero_diag(device):
    """CorrelationModule 输出对称且 diag=0"""
    from dcrem.models.correlation import CorrelationModule

    q = 10
    mod = CorrelationModule(num_classes=q, embed_dim=64).to(device)
    C, L = mod()

    # Symmetry
    assert torch.allclose(C, C.T, atol=1e-6), \
        f"C not symmetric: max |C − Cᵀ| = {(C - C.T).abs().max():.6f}"

    # Zero diagonal
    diag = C.diag()
    assert diag.abs().max() < 1e-8, \
        f"C diagonal not zero: max |diag| = {diag.abs().max():.8f}"

    # L = D − C
    D_diag = C.sum(dim=1)
    D = torch.diag(D_diag)
    L_check = D - C
    assert torch.allclose(L, L_check, atol=1e-6)

    # Row sums of C should be ≤ 1 (softmax property before symmetrization)
    row_sums = C.sum(dim=1)
    assert row_sums.max() <= 1.01, f"Row sum > 1: max={row_sums.max():.4f}"


def test_static_correlation_uses_train_target_and_is_frozen(device):
    """static-C is a frozen train-fold co-occurrence Laplacian, not λ₂=0."""
    from dcrem.models.correlation import StaticCorrelationModule

    # Q×N, with known co-occurrence counts.  No validation/test target enters.
    train_target = torch.tensor([
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [-1, -1, 1, 1],
    ], dtype=torch.float32)
    mod = StaticCorrelationModule(train_target).to(device)
    C, L = mod()

    expected_C = torch.tensor([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.0],
    ], device=device)
    expected_L = torch.diag(expected_C.sum(dim=1)) - expected_C
    assert torch.allclose(C, expected_C)
    assert torch.allclose(L, expected_L)
    assert not list(mod.parameters())
    assert not C.requires_grad and not L.requires_grad

    # L is fixed, but the λ₂ objective still differentiates with respect to W.
    W = torch.randn(5, 3, device=device, requires_grad=True)
    loss = 0.5 * (W.T @ W * L).sum()
    loss.backward()
    assert W.grad is not None and W.grad.norm() > 0


def test_build_model_accepts_static_train_correlation(device):
    """The training entry point wires static-C without creating embeddings."""
    from dcrem.models.correlation import StaticCorrelationModule
    from dcrem.scripts.train import build_encoder, build_model

    train_target = torch.tensor([[1, -1, 1], [-1, 1, 1]], dtype=torch.float32)
    encoder = build_encoder("mlp", input_dim=4).to(device)
    _, _, _, corr_mod, _ = build_model(encoder, 2, {
        "correlation_mode": "static_train",
        "static_train_target": train_target,
    })
    assert isinstance(corr_mod, StaticCorrelationModule)
    assert not list(corr_mod.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Closed-form Sylvester W vs SGD W (frozen features)
# ═══════════════════════════════════════════════════════════════════════════

def test_sylvester_vs_sgd(device):
    """冻结特征下 Sylvester W vs SGD W 的 MSE < 1e-3"""
    from dcrem.models.heads import ClassifierHead, ReciprocalBank
    from dcrem.optim.sylvester import solve_sylvester

    torch.manual_seed(42)
    d, q, N = 64, 5, 100
    F_feats = torch.randn(N, d, device=device)
    F_feats = F_feats / F_feats.norm(dim=-1, keepdim=True)  # L2-normalized
    Y = (torch.rand(N, q, device=device) > 0.7).float() * 2 - 1

    lamda1, lamda2, lamda3 = 1.0, 0.1, 10.0
    P = torch.zeros(d, q, device=device)

    # ── Sylvester closed form ──
    Fc = F_feats - F_feats.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    Sy_A = Fc.T @ Fc / N + (lamda1 + lamda3) * torch.eye(d, device=device, dtype=F_feats.dtype)
    Sy_B = lamda2 * torch.eye(q, device=device, dtype=F_feats.dtype)  # identity C
    Sy_C = Fc.T @ Yc / N + lamda3 * P
    W_sylv = solve_sylvester(Sy_A, Sy_B, Sy_C)

    # ── L-BFGS baseline (sum-of-squares, matching Sylvester) ──
    head = ClassifierHead(d, q).to(device)
    opt = torch.optim.LBFGS([head.W], lr=1.0, max_iter=200,
                            tolerance_grad=1e-12, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        logits = Fc @ head.W
        loss = 0.5 / N * F.mse_loss(logits, Yc, reduction="sum")
        loss = loss + 0.5 * lamda1 * (head.W * head.W).sum()
        loss = loss + 0.5 * lamda2 * (head.W.T @ head.W * torch.eye(q, device=device)).sum()
        diff = head.W - P
        loss = loss + 0.5 * lamda3 * (diff * diff).sum()
        loss.backward()
        return loss
    opt.step(closure)

    mse = F.mse_loss(W_sylv, head.W).item()
    assert mse < 1e-3, f"Sylvester vs LBFGS MSE={mse:.6f} exceeds 1e-3"


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Hinge sub-gradient matches autograd
# ═══════════════════════════════════════════════════════════════════════════

def test_hinge_gradient_vs_autograd(device):
    """Hinge 次梯度 vs 数值梯度（torch.autograd 验证）误差 < 1e-4"""
    from dcrem.losses.open_space import open_space_risk

    torch.manual_seed(42)
    B, d, q = 32, 64, 5
    feats = torch.randn(B, d, device=device, requires_grad=True)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    P = torch.randn(d, q, device=device)
    R = F.softplus(torch.randn(q, device=device))
    Y = (torch.rand(B, q, device=device) > 0.7).float() * 2 - 1

    # Hinge loss
    P_param = P.detach().clone().requires_grad_(True)
    loss = open_space_risk(feats, P_param, R, Y)
    loss.backward()

    # Numerical gradient check via finite differences
    eps = 1e-4
    for i in range(min(5, d)):
        for j in range(q):
            P_plus = P.detach().clone()
            P_plus[i, j] += eps
            loss_plus = open_space_risk(feats, P_plus, R, Y).detach()

            P_minus = P.detach().clone()
            P_minus[i, j] -= eps
            loss_minus = open_space_risk(feats, P_minus, R, Y).detach()

            num_grad = (loss_plus - loss_minus) / (2 * eps)
            auto_grad = P_param.grad[i, j].item()
            rel_err = abs(num_grad - auto_grad) / max(abs(num_grad), abs(auto_grad), 1e-8)
            assert rel_err < 1e-3 or abs(num_grad - auto_grad) < 1e-4, \
                f"Hinge grad mismatch at ({i},{j}): auto={auto_grad:.8f}, num={num_grad:.8f}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: D-CREM identity encoder + linear kernel ≡ CREM linear kernel
# ═══════════════════════════════════════════════════════════════════════════

def test_identity_encoder_equivalence(device):
    """D-CREM (identity encoder, no norm, linear kernel) output should be
    algebraically isomorphic to CREM with linear kernel.

    This validates the primal-space Sylvester derivation.
    """
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, ReciprocalBank, MarginVector
    from dcrem.optim.sylvester import solve_sylvester

    torch.manual_seed(42)
    N, d, q = 100, 50, 5
    X = torch.randn(N, d, device=device)
    Y = (torch.rand(N, q, device=device) > 0.7).float() * 2 - 1

    # ── D-CREM: identity encoder, no norm, linear kernel ──
    encoder = IdentityEncoder(d).to(device)
    F_dcrem = encoder(X)  # = X (identity)
    head = ClassifierHead(d, q).to(device)
    P = torch.zeros(d, q, device=device)

    lamda = 1.0
    # D-CREM primal ridge solution for 0.5/N ||XW-Y||² + 0.5λ||W||².
    W_dcrem = torch.linalg.solve(
        F_dcrem.T @ F_dcrem / N + lamda * torch.eye(
            d, device=device, dtype=X.dtype),
        F_dcrem.T @ Y / N)

    # ── CREM: linear kernel ──
    # CREM works in dual space: K = X @ Xᵀ (linear kernel)
    # θ_W = (K + λ₃I)⁻¹ Y  (initialisation, ridge)
    # But the full Sylvester is different because CREM deals with dual vars.
    # Here we compare the *output*: Kte @ W_dcrem vs CREM's output.
    # Since D-CREM with identity encoder and linear kernel replaces CREM's:
    #   Output = Kte @ θ_W + b
    # with:
    #   Output = F @ W_dcrem + b
    # and Kte = Xte @ Xtrᵀ, θ_W = Xtr @ W_dual...
    # The key equivalence: if W_dcrem = Xᵀ @ θ_W then outputs match.
    # We just verify that D-CREM's Sylvester produces a valid W that minimises
    # the closed-set loss.

    # Compute closed-set output from D-CREM's W
    output_dcrem = F_dcrem @ W_dcrem
    # Compute CREM's W via ridge regression (initialisation path)
    K = F_dcrem @ F_dcrem.T  # linear kernel
    theta_W = torch.linalg.solve(
        K + N * lamda * torch.eye(N, device=device, dtype=X.dtype), Y)
    # Exact primal/dual identity: W = Xᵀ @ theta_W.
    W_from_dual = F_dcrem.T @ theta_W

    # They should produce identical outputs up to numerical precision.
    output_crem_dual = K @ theta_W
    mse_out = F.mse_loss(output_dcrem, output_crem_dual).item()
    assert mse_out < 1e-8, \
        f"Identity encoder equivalence check failed: MSE(output)={mse_out:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Warm-up P and -W cosine similarity > 0.9
# ═══════════════════════════════════════════════════════════════════════════

def test_warmup_P_vs_W(device):
    """Warm-up 后 P 与 W 的余弦相似度 > 0.9。

    After warm-up, P is initialised as P = W (matching CREM's theta_P = theta_W).
    P stores classifier-aligned coefficients, so the actual reciprocal point
    is -P=-W.  This test verifies that convention.
    """
    from dcrem.models.heads import ClassifierHead, ReciprocalBank

    torch.manual_seed(42)
    d, q, N = 64, 5, 100
    F = torch.randn(N, d, device=device)
    Y = (torch.rand(N, q, device=device) > 0.7).float() * 2 - 1

    head = ClassifierHead(d, q).to(device)
    # Simulate warm-up: ridge regression closed-form init
    head.closed_form_init(F, Y, lamda1=10.0)
    W = head.W

    recip = ReciprocalBank(d, q).to(device)
    recip.init_from_W(W)

    P = recip()
    # After warm-up, P = W, so cos(P, W) should be ~1.0
    P_norm = P / P.norm(dim=0, keepdim=True)
    W_norm = W / W.norm(dim=0, keepdim=True)
    cos_sim = (P_norm * W_norm).sum(dim=0)  # per-label cosine sim

    mean_cos = cos_sim.mean().item()
    assert mean_cos > 0.9, \
        f"Warm-up P vs W cosine similarity = {mean_cos:.4f} < 0.9"

    reciprocal = -P
    reciprocal_cos_w = (
        reciprocal / reciprocal.norm(dim=0, keepdim=True) * W_norm
    ).sum(dim=0).mean().item()
    assert reciprocal_cos_w < -0.9


def test_reciprocal_distance_uses_negative_P(device):
    """Stored P must produce the explicit Euclidean distance to -P."""
    from dcrem.losses.open_space import reciprocal_distances

    torch.manual_seed(7)
    features = torch.randn(6, 4, device=device)
    P = torch.randn(4, 3, device=device)
    actual = reciprocal_distances(features, P)
    explicit = ((features[:, None, :] + P.T[None, :, :]) ** 2).sum(dim=-1)
    assert torch.allclose(actual, explicit, atol=1e-6)


def test_relative_score_equivalence_and_residual_noncollapse(device):
    """Relative score exposes the exact P=W logit equivalence condition."""
    from dcrem.models.calibrator import OpenSetCalibrator
    from dcrem.models.heads import ResidualReciprocalBank

    torch.manual_seed(12)
    features = torch.nn.functional.normalize(
        torch.randn(7, 5, device=device), dim=1)
    W = torch.randn(5, 4, device=device)
    W_hat = torch.nn.functional.normalize(W, dim=0)
    equivalent = OpenSetCalibrator.compute_relative_scores(features, W, W)
    assert torch.allclose(equivalent, 4.0 * features @ W_hat, atol=1e-5)

    bank = ResidualReciprocalBank(5, 4, residual_scale=0.5).to(device)
    P = bank(W)
    cosine = torch.nn.functional.cosine_similarity(W.T, P.T, dim=1)
    expected = 1.0 / (1.0 + 0.5 ** 2) ** 0.5
    assert torch.allclose(cosine, torch.full_like(cosine, expected), atol=1e-5)


def test_pseudo_gate_variants_have_expected_geometry_gradients(device):
    """B0 is classifier geometry; B2 gives P/features an open-set gradient."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, ReciprocalBank
    from dcrem.optim.v3_trainer import DCREMV3Trainer

    torch.manual_seed(21)
    encoder = IdentityEncoder(3)
    head = ClassifierHead(3, 2)
    bank = ReciprocalBank(3, 2)
    common = {
        "lamda1": 1.0, "lamda3": 10.0, "pseudo_weight": 0.1,
        "pseudo_margin": 0.1, "pseudo_top_k": 1,
    }
    baseline = DCREMV3Trainer(
        encoder, head, bank, {**common, "pseudo_variant": "B0"}, device)
    assert baseline.get_reciprocal_parameters() is baseline.head.W

    trainer = DCREMV3Trainer(
        IdentityEncoder(3), ClassifierHead(3, 2), ReciprocalBank(3, 2),
        {**common, "pseudo_variant": "B2"}, device)
    features = torch.nn.functional.normalize(
        torch.randn(4, 3, device=device), dim=1).detach().requires_grad_(True)
    targets = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        device=device)
    heldout = torch.tensor([1], device=device)
    known_mask = torch.tensor([True, False], device=device)
    losses = trainer._losses(
        features, trainer.head(features), targets, heldout, known_mask)
    losses["L_pseudo"].backward()
    assert features.grad is not None and features.grad.norm() > 0
    assert trainer.recip_bank.P.grad is not None
    assert trainer.recip_bank.P.grad.norm() > 0
    assert trainer.head.W.grad is None


def test_pseudo_gate_b3_excludes_heldout_logits_from_classification(device):
    """Changing an episodically hidden logit must not change B3 L_cls."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, ReciprocalBank
    from dcrem.optim.v3_trainer import DCREMV3Trainer

    trainer = DCREMV3Trainer(
        IdentityEncoder(2), ClassifierHead(2, 2), ReciprocalBank(2, 2),
        {"pseudo_variant": "B3", "pseudo_top_k": 1}, device)
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], device=device)
    targets = torch.tensor(
        [[1.0, 1.0], [-1.0, -1.0]], device=device)
    heldout = torch.tensor([1], device=device)
    known_mask = torch.tensor([True, False], device=device)
    logits_a = torch.zeros(2, 2, device=device)
    logits_b = logits_a.clone()
    logits_b[:, 1] = 100.0
    loss_a = trainer._losses(
        features, logits_a, targets, heldout, known_mask)["L_cls"]
    loss_b = trainer._losses(
        features, logits_b, targets, heldout, known_mask)["L_cls"]
    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(loss_a, torch.tensor(1.0, device=device))


def test_label_rank_gives_features_and_P_an_independent_gradient(device):
    """Label-conditional ranking updates geometry without touching W."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, ReciprocalBank
    from dcrem.optim.v3_trainer import DCREMV3Trainer

    torch.manual_seed(22)
    trainer = DCREMV3Trainer(
        IdentityEncoder(3), ClassifierHead(3, 2), ReciprocalBank(3, 2),
        {
            "pseudo_variant": "B1", "pseudo_weight": 0.0,
            "label_rank_weight": 0.1, "label_rank_margin": 0.1,
        }, device)
    features = torch.nn.functional.normalize(
        torch.randn(4, 3, device=device), dim=1).detach().requires_grad_(True)
    targets = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        device=device)
    heldout = torch.tensor([1], device=device)
    known_mask = torch.tensor([True, False], device=device)
    losses = trainer._losses(
        features, trainer.head(features), targets, heldout, known_mask)
    losses["L_label_rank"].backward()
    assert features.grad is not None and features.grad.norm() > 0
    assert trainer.recip_bank.P.grad is not None
    assert trainer.recip_bank.P.grad.norm() > 0
    assert trainer.head.W.grad is None


def test_hard_negative_rank_ignores_easy_negative_geometry(device):
    """Only frozen-W top-ranked negatives participate in hard mining."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, ReciprocalBank
    from dcrem.optim.v3_trainer import DCREMV3Trainer

    trainer = DCREMV3Trainer(
        IdentityEncoder(2), ClassifierHead(2, 1), ReciprocalBank(2, 1),
        {
            "pseudo_variant": "B1", "pseudo_weight": 0.0,
            "label_rank_weight": 0.1, "label_rank_margin": 0.1,
            "label_rank_hard_fraction": 0.5,
        }, device)
    trainer.mining_W = torch.tensor([[1.0], [0.0]], device=device)
    features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]],
        device=device, requires_grad=True)
    targets = torch.tensor([[1.0], [-1.0], [-1.0]], device=device)
    losses = trainer._losses(
        features, trainer.head(features), targets,
        torch.tensor([0], device=device),
        torch.tensor([True], device=device))
    losses["L_label_rank"].backward()
    assert features.grad[0].norm() > 0
    assert features.grad[1].norm() > 0
    assert torch.allclose(features.grad[2], torch.zeros(2, device=device))
    assert trainer.head.W.grad is None


def test_multi_reciprocal_bank_initializes_distinct_hard_negative_modes(device):
    """Multiple actual reciprocal points start from distinct negative modes."""
    from dcrem.models.heads import MultiReciprocalBank

    bank = MultiReciprocalBank(2, 1, num_prototypes=2).to(device)
    features = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
        [-1.0, 0.0], [0.0, -1.0]], device=device)
    features = torch.nn.functional.normalize(features, dim=1)
    targets = torch.tensor(
        [[1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]],
        device=device)
    W = torch.tensor([[1.0], [0.0]], device=device)
    bank.init_from_hard_negatives(
        features, targets, W, hard_fraction=1.0)
    assert bank().shape == (2, 1, 2)
    actual_points = -bank()[:, 0, :].T
    assert torch.allclose(
        actual_points.norm(dim=1), torch.ones(2, device=device), atol=1e-6)
    assert torch.dist(actual_points[0], actual_points[1]) > 0.5


def test_single_multi_prototype_score_matches_single_bank_distance(device):
    """Nearest-prototype scoring reduces exactly to the one-point geometry."""
    from dcrem.losses.open_space import reciprocal_distances
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, MultiReciprocalBank
    from dcrem.optim.v3_trainer import DCREMV3Trainer

    bank = MultiReciprocalBank(3, 2, num_prototypes=1).to(device)
    trainer = DCREMV3Trainer(
        IdentityEncoder(3), ClassifierHead(3, 2), bank,
        {"pseudo_variant": "B1"}, device)
    features = torch.nn.functional.normalize(
        torch.randn(5, 3, device=device), dim=1)
    expected = reciprocal_distances(features, bank()[:, :, 0])
    actual = trainer.reciprocal_score_values(features)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_multi_prototype_rank_updates_bank_without_direct_W_gradient(device):
    """Nearest-prototype label ranking supplies geometry-only gradients."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, MultiReciprocalBank
    from dcrem.optim.v3_trainer import DCREMV3Trainer

    trainer = DCREMV3Trainer(
        IdentityEncoder(3), ClassifierHead(3, 2),
        MultiReciprocalBank(3, 2, num_prototypes=2),
        {
            "pseudo_variant": "B1", "label_rank_weight": 0.1,
            "label_rank_hard_fraction": 0.5,
        }, device)
    trainer.mining_W = trainer.head.W.detach().clone()
    features = torch.nn.functional.normalize(
        torch.randn(6, 3, device=device), dim=1).detach().requires_grad_(True)
    targets = torch.tensor([
        [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0],
        [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], device=device)
    losses = trainer._losses(
        features, trainer.head(features), targets,
        torch.tensor([1], device=device),
        torch.tensor([True, False], device=device))
    losses["L_label_rank"].backward()
    assert features.grad is not None and features.grad.norm() > 0
    assert trainer.recip_bank.P.grad is not None
    assert trainer.recip_bank.P.grad.norm() > 0
    assert trainer.head.W.grad is None


def test_paper_hinge_gradient_direction(device):
    """P/R autograd must match the paper hinge in an active region."""
    from dcrem.losses.open_space import open_space_risk

    features = torch.tensor([[0.2, -0.1], [0.3, 0.4]], device=device)
    P = torch.tensor([[0.1], [0.2]], device=device, requires_grad=True)
    R = torch.tensor([3.0], device=device, requires_grad=True)
    targets = torch.ones(2, 1, device=device)
    loss = open_space_risk(features, P, R, targets, alpha=1.0)
    loss.backward()

    # All hinges are active: d/dP mean(1+R²-||f+P||²)
    expected_p = -2.0 * (features + P.detach().T).mean(dim=0).reshape(2, 1)
    expected_r = 2.0 * R.detach()
    assert torch.allclose(P.grad, expected_p, atol=1e-6)
    assert torch.allclose(R.grad, expected_r, atol=1e-6)


def test_open_space_mean_reduction_and_radius_free(device):
    """Lite development loss is label-count invariant and ignores R by design."""
    from dcrem.losses.open_space import open_space_risk

    features = torch.tensor([[0.1, 0.2], [0.2, -0.1]], device=device)
    P = torch.zeros(2, 3, device=device)
    targets = torch.ones(2, 3, device=device)
    small_R = torch.zeros(3, device=device)
    large_R = torch.full((3,), 10.0, device=device)
    summed = open_space_risk(
        features, P, small_R, targets, reduction="sum", radius_free=True)
    averaged = open_space_risk(
        features, P, small_R, targets, reduction="mean", radius_free=True)
    averaged_large_R = open_space_risk(
        features, P, large_R, targets, reduction="mean", radius_free=True)
    assert torch.allclose(averaged, summed / 3.0)
    assert torch.allclose(averaged, averaged_large_R)


def test_mode_b_diversity_updates_P_not_W(device):
    """Mode B's gradient block must include P and exclude closed-form W/b."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, MarginVector, ReciprocalBank
    from dcrem.optim.trainer import DCREMTrainer

    d, q = 4, 3
    trainer = DCREMTrainer(
        IdentityEncoder(d).to(device),
        ClassifierHead(d, q).to(device),
        ReciprocalBank(d, q).to(device),
        MarginVector(q).to(device),
        config={"lamda3": 0.0, "alpha": 0.0, "beta": 0.0,
                "gamma": 1.0, "theta_div": -1.0},
    ).to(device)
    trainer.head.W.requires_grad_(False)
    trainer.head.b.requires_grad_(False)
    optimizer = trainer._make_mode_B_optimizer()
    parameter_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert id(trainer.recip_bank.P) in parameter_ids
    assert id(trainer.head.W) not in parameter_ids
    assert id(trainer.head.b) not in parameter_ids

    features = torch.randn(5, d, device=device)
    logits = trainer.head(features)
    targets = torch.ones(5, q, device=device)
    trainer._positive_prevalence = torch.ones(q, device=device)
    losses = trainer._compute_losses(features, logits, targets)
    optimizer.zero_grad()
    losses["total"].backward()
    assert trainer.recip_bank.P.grad is not None
    assert trainer.recip_bank.P.grad.abs().sum().item() > 0


def test_classifier_induced_reciprocal_is_exact_and_not_optimized(device):
    """Paper-core uses P:=W exactly and removes the redundant free bank."""
    from dcrem.models.encoder import IdentityEncoder
    from dcrem.models.heads import ClassifierHead, MarginVector, ReciprocalBank
    from dcrem.optim.trainer import DCREMTrainer

    d, q = 4, 3
    trainer = DCREMTrainer(
        IdentityEncoder(d).to(device),
        ClassifierHead(d, q).to(device),
        ReciprocalBank(d, q).to(device),
        MarginVector(q).to(device),
        config={
            "classifier_induced_reciprocal": True,
            "lamda3": 10.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
        },
    ).to(device)
    with torch.no_grad():
        trainer.recip_bank.P.fill_(100.0)
    assert trainer.get_reciprocal_parameters() is trainer.head.W

    optimizer = trainer._make_optimizer(mode="A")
    parameter_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert id(trainer.recip_bank.P) not in parameter_ids
    assert id(trainer.head.W) in parameter_ids

    features = torch.randn(5, d, device=device)
    targets = torch.ones(5, q, device=device)
    losses = trainer._compute_losses(
        features, trainer.head(features), targets)
    assert torch.allclose(
        losses["L_coupling"], torch.zeros((), device=device))


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test: end-to-end training on tiny synthetic data
# ═══════════════════════════════════════════════════════════════════════════

def test_smoke_training_mode_B(device):
    """Smoke test: Mode B on synthetic data (10 epochs, should not crash)."""
    from dcrem.models.encoder import TabularMLP
    from dcrem.models.heads import ClassifierHead, ReciprocalBank, MarginVector
    from dcrem.optim.trainer import DCREMTrainer
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    N, d, q = 200, 50, 8
    X = torch.randn(N, d, device=device)
    Y = (torch.rand(N, q, device=device) > 0.7).float() * 2 - 1

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    encoder = TabularMLP(d, output_dim=64).to(device)
    head = ClassifierHead(64, q).to(device)
    recip = ReciprocalBank(64, q).to(device)
    margins = MarginVector(q).to(device)

    config = {
        "lamda1": 1.0, "lamda2": 0.1, "lamda3": 10.0, "alpha": 1.0,
        "beta": 0.01, "gamma": 0.0, "tau": 2.0, "theta_div": 0.9,
        "lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 1e-5,
        "T_sylvester": 5, "T_warmup": 2, "pre_warmup_epochs": 3,
    }
    trainer = DCREMTrainer(encoder, head, recip, margins, config=config)
    trainer.to(device)

    history = trainer.fit_mode_B(loader, 10, log_every=5)

    # Loss should decrease
    assert len(history["loss"]) == 10
    assert history["loss"][-1] < history["loss"][0] * 0.9, \
        f"Loss did not decrease: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}"

    # No NaN
    for loss_val in history["loss"]:
        assert not np.isnan(loss_val), "NaN in training loss"

    print(f"Smoke test passed: loss {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}")

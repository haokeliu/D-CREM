"""D-CREM training loop with two modes and warm-up initialisation.

Mode A — end-to-end SGD (AdamW):
  All parameters (encoder, W, b, P, R, E) optimised jointly by gradient descent.

Mode B — hybrid optimisation:
  Every T_sylvester epochs: freeze encoder, solve W via Sylvester (using C from
  CorrelationModule), and update b in closed form.  Encoder, P, R, and E are
  updated by AdamW on the exact same objective between closed-form steps.

Warm-up (two-stage):
  Stage 0 (pre-warmup, tabular only): MSE-only training for ~10 epochs so the
    MLP encoder gains basic discriminative power.
  Stage 1 (warmup, all data): freeze encoder, accumulate full feature matrix,
    initialise W via ridge regression closed form, set P = W, R = √2·1.
"""

import time

import torch
import torch.nn.functional as F

from .sylvester import solve_sylvester


class DCREMTrainer:
    """D-CREM training orchestrator.

    Parameters
    ----------
    encoder    : nn.Module     g_θ(·), output dim = d'
    head       : ClassifierHead
    recip_bank : ReciprocalBank
    margins    : MarginVector
    corr_mod   : CorrelationModule | None
    calibrator : OpenSetCalibrator | None

    config     : dict with keys:
        lamda1, lamda2, lamda3, alpha   — loss weights (CREM legacy)
        beta, gamma                      — L_unif, L_div weights
        tau, theta_div                   — uniformity temperature, diversity threshold
        lr, backbone_lr                  — learning rates
        weight_decay                     — AdamW weight decay
        T_sylvester                      — sylvester interval for mode B
        T_warmup                         — warmup epochs
        pre_warmup_epochs                — tabular pre-warmup epochs (0 for image)
    """

    def __init__(self, encoder, head, recip_bank, margins,
                 corr_mod=None, calibrator=None, config=None):
        self.encoder = encoder
        self.head = head
        self.recip_bank = recip_bank
        self.margins = margins
        self.corr_mod = corr_mod
        self.calibrator = calibrator

        cfg = config or {}
        self.lamda1 = cfg.get("lamda1", 1.0)
        self.lamda2 = cfg.get("lamda2", 0.1)
        self.lamda3 = cfg.get("lamda3", 10.0)
        self.alpha = cfg.get("alpha", 1.0)
        self.open_reduction = cfg.get("open_reduction", "sum")
        self.radius_free_open = cfg.get("radius_free_open", False)
        self.open_margin = cfg.get("open_margin", 1.0)
        self.beta = cfg.get("beta", 0.1)
        self.gamma_div = cfg.get("gamma", 0.01)        # L_div weight
        self.tau = cfg.get("tau", 2.0)
        self.theta_div = cfg.get("theta_div", 0.9)
        self.lr = cfg.get("lr", 1e-4)
        self.backbone_lr = cfg.get("backbone_lr", 1e-5)
        self.weight_decay = cfg.get("weight_decay", 1e-4)
        self.T_sylvester = cfg.get("T_sylvester", 10)
        self.T_warmup = cfg.get("T_warmup", 5)
        self.pre_warmup_epochs = cfg.get("pre_warmup_epochs", 0)
        self.no_l2norm = cfg.get("no_l2norm", False)
        self.freeze_encoder = cfg.get("freeze_encoder", False)
        self.no_warmup = cfg.get("no_warmup", False)
        self.classifier_induced_reciprocal = cfg.get(
            "classifier_induced_reciprocal", False)
        self.primary_score = cfg.get("primary_score", "reciprocal")

        self.device = None
        self.mode = "B"          # "A" (end-to-end) or "B" (hybrid)
        self._positive_prevalence = None
        self.timing = {}

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.head.to(device)
        self.recip_bank.to(device)
        self.margins.to(device)
        if self.corr_mod is not None:
            self.corr_mod.to(device)
        if self.calibrator is not None:
            self.calibrator.to(device)
        return self

    def train(self):
        self.encoder.train()
        self.head.train()
        self.recip_bank.train()
        self.margins.train()
        if self.corr_mod is not None:
            self.corr_mod.train()
        if self.calibrator is not None:
            self.calibrator.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()
        self.recip_bank.eval()
        self.margins.eval()
        if self.corr_mod is not None:
            self.corr_mod.eval()
        if self.calibrator is not None:
            self.calibrator.eval()

    # ── forward helpers ─────────────────────────────────────────────────

    def _forward_encoder(self, x):
        """Encode and ℓ₂-normalise: f(x) = g_θ(x) / ‖g_θ(x)‖."""
        raw = self.encoder(x)
        if self.no_l2norm:
            return raw
        return F.normalize(raw, p=2, dim=-1)

    def _forward_full(self, x):
        """Full forward pass: features → logits."""
        feats = self._forward_encoder(x)
        return self.head(feats), feats

    def get_reciprocal_parameters(self):
        """Return the effective stored coefficients used by OSR scoring."""
        if self.classifier_induced_reciprocal:
            return self.head.W
        return self.recip_bank()

    # ── loss computation ────────────────────────────────────────────────

    def _compute_losses(self, features, logits, targets):
        """Return dict of all loss components."""
        from dcrem.losses import (
            mse_loss, open_space_risk, uniformity_loss, diversity_loss,
        )
        B, q = targets.shape
        W = self.head.W
        P = self.get_reciprocal_parameters()
        R = self.margins()

        L_cls = mse_loss(logits, targets, sample_average=True)
        L_reg_W = 0.5 * self.lamda1 * (W * W).sum()

        # Label correlation
        L_corr = features.new_zeros(())
        if self.corr_mod is not None and self.lamda2 > 0:
            C, L_lap = self.corr_mod()    # C: (q, q), L: (q, q)
            # tr(W L Wᵀ) = sum(W @ L * W) = trace(Wᵀ W L) or more directly:
            # trace(W.T @ W @ L) = (W.T @ W * L.T).sum()
            L_corr = 0.5 * self.lamda2 * (W.T @ W * L_lap).sum()

        # W-P coupling
        if self.classifier_induced_reciprocal:
            L_coupling = features.new_zeros(())
        else:
            diff = W - P
            L_coupling = 0.5 * self.lamda3 * (diff * diff).sum()

        # Open-space risk
        L_open = open_space_risk(
            features, P, R, targets, self.alpha,
            positive_prevalence=self._positive_prevalence,
            reduction=self.open_reduction,
            radius_free=self.radius_free_open,
            margin=self.open_margin)

        # Uniformity
        L_unif = uniformity_loss(features, self.tau) if self.beta > 0 else features.new_zeros(())
        L_unif = self.beta * L_unif

        # Diversity
        P_norm = P / (P.norm(p=2, dim=0, keepdim=True) + 1e-8)
        L_div = diversity_loss(P_norm, self.theta_div) if self.gamma_div > 0 else features.new_zeros(())
        L_div = self.gamma_div * L_div

        total = L_cls + L_reg_W + L_corr + L_coupling + L_open + L_unif + L_div

        return {
            "total": total,
            "L_cls": L_cls,
            "L_reg_W": L_reg_W,
            "L_corr": L_corr,
            "L_coupling": L_coupling,
            "L_open": L_open,
            "L_unif": L_unif,
            "L_div": L_div,
        }

    # ── optimiser setup ─────────────────────────────────────────────────

    def _make_optimizer(self, mode="A"):
        """Build AdamW optimiser(s) with differential learning rates."""
        # Encoder params
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        # Head params
        head_params = [p for p in (self.head.W, self.head.b) if p.requires_grad]
        # Other params
        other_params = list(self.margins.parameters())
        if not self.classifier_induced_reciprocal:
            other_params = list(self.recip_bank.parameters()) + other_params
        if self.corr_mod is not None:
            other_params += list(self.corr_mod.parameters())
        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": self.backbone_lr,
                "weight_decay": self.weight_decay,
            })
        if head_params or other_params:
            param_groups.append({
                "params": head_params + other_params,
                "lr": self.lr,
                # W/P/R/E already have explicit objective terms.  Applying
                # AdamW decay here would silently optimise a different loss.
                "weight_decay": 0.0,
            })

        return torch.optim.AdamW(param_groups)

    def _make_mode_B_optimizer(self):
        """Persistent optimiser for the gradient blocks of Mode B."""
        groups = []
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if encoder_params:
            groups.append({
                "params": encoder_params,
                "lr": self.backbone_lr,
                "weight_decay": self.weight_decay,
            })
        other = list(self.margins.parameters())
        if not self.classifier_induced_reciprocal:
            other = list(self.recip_bank.parameters()) + other
        if self.corr_mod is not None:
            other += list(self.corr_mod.parameters())
        groups.append({"params": other, "lr": self.lr, "weight_decay": 0.0})
        return torch.optim.AdamW(groups)

    def _prepare_objective_stats(self, train_loader):
        """Compute fold-level positive prevalences for unbiased batch risk."""
        total = 0
        positive = None
        for _, y_batch in train_loader:
            y = y_batch.detach()
            total += y.shape[0]
            counts = (y == 1).sum(dim=0).to(torch.float64)
            positive = counts if positive is None else positive + counts
        if total == 0:
            raise ValueError("empty training loader")
        prevalence = (positive / float(total)).clamp(min=1.0 / float(total))
        self._positive_prevalence = prevalence.to(
            device=self.device, dtype=torch.float32)

    def _set_encoder_training_mode(self):
        if self.freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()

    # ── Warm-up ─────────────────────────────────────────────────────────

    def _pre_warmup(self, train_loader):
        """Stage 0: MSE-only pre-warmup for tabular encoder (no L_open etc.)."""
        if self.pre_warmup_epochs <= 0:
            return
        print(f"  Pre-warmup ({self.pre_warmup_epochs} epochs, MSE only)...")
        opt = torch.optim.AdamW(list(self.encoder.parameters()) +
                                [self.head.W, self.head.b],
                                lr=self.lr, weight_decay=self.weight_decay)
        self.encoder.train()
        for epoch in range(self.pre_warmup_epochs):
            total_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                feats = self._forward_encoder(x_batch)
                logits = self.head(feats)
                loss = 0.5 * F.mse_loss(logits, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % max(1, self.pre_warmup_epochs // 3) == 0:
                print(f"    pre-warmup epoch {epoch+1}/{self.pre_warmup_epochs}  "
                      f"loss={total_loss / max(1, n_batches):.6f}")

    def _warmup(self, train_loader):
        """Stage 1: accumulate features, closed-form init W, set P=W, R=√2."""
        if self.no_warmup:
            print("  Warm-up skipped (--no-warmup). Using random init for W/P/R.")
            return
        print("  Warm-up (one deterministic full-fold feature pass)...")
        self.encoder.eval()
        all_features, all_targets = [], []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            with torch.no_grad():
                feats = self._forward_encoder(x_batch)
            all_features.append(feats)
            all_targets.append(y_batch.to(self.device))
        F_all = torch.cat(all_features, dim=0)   # (N, d')
        Y_all = torch.cat(all_targets, dim=0)     # (N, q)
        print(f"  Accumulated features: {F_all.shape}, targets: {Y_all.shape}")
        # Closed-form W init
        print("  Computing closed-form W init (ridge regression)...")
        self.head.closed_form_init(F_all, Y_all, lamda1=self.lamda1,
                                   sample_average=True)
        # P = W
        self.recip_bank.init_from_W(self.head.W)
        # R = √2 (default CREM init)
        with torch.no_grad():
            raw_init = torch.as_tensor(2.0 ** 0.5, device=self.device).expm1().log()
            self.margins.raw_R.fill_(raw_init.item())
        print("  Warm-up complete.")

    # ── Full training (mode B) ──────────────────────────────────────────

    def fit_mode_B(self, train_loader, num_epochs, val_loader=None,
                   log_every=5):
        """Hybrid optimisation: Sylvester closed-form interleaved with SGD.

        Every T_sylvester epochs:
          - Freeze encoder, compute full feature matrix
          - Solve W via Sylvester (with C from CorrelationModule)
          - Update b closed-form
          - Keep P and R fixed during this exact W,b block update
        Between Sylvester epochs:
          - AdamW on encoder, P, R, and E using the shared paper objective
        """
        print(f"Training mode B — {num_epochs} epochs, Sylvester every {self.T_sylvester}")
        self.to(self.device)

        # Pre-warmup (tabular only)
        stage_start = time.time()
        self._pre_warmup(train_loader)
        self.timing["pre_warmup_s"] = time.time() - stage_start
        # Warm-up
        stage_start = time.time()
        self._warmup(train_loader)
        self.timing["warmup_s"] = time.time() - stage_start

        # A frozen encoder is frozen *after* discriminative warm-up, and kept
        # in eval mode so BatchNorm running statistics cannot drift.
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            print("  Encoder frozen after warm-up (--freeze-encoder).")

        self._prepare_objective_stats(train_loader)
        # W and b are the closed-form block in Mode B.  Freezing their
        # autograd flags prevents stale gradient accumulation during SGD.
        self.head.W.requires_grad_(False)
        self.head.b.requires_grad_(False)
        opt = self._make_mode_B_optimizer()

        history = {"loss": [], "components": []}

        stage_start = time.time()
        for epoch in range(num_epochs):
            self._set_encoder_training_mode()
            epoch_losses = {k: 0.0 for k in [
                "total", "L_cls", "L_reg_W", "L_corr", "L_coupling",
                "L_open", "L_unif", "L_div"]}

            # ── Sylvester update ──
            if epoch % self.T_sylvester == 0:
                self._sylvester_update(train_loader)
                self._set_encoder_training_mode()

            # ── Gradient blocks: encoder + P + R + E ──
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                feats = self._forward_encoder(x_batch)
                logits = self.head(feats)
                loss_dict = self._compute_losses(feats, logits, y_batch)

                opt.zero_grad()
                loss_dict["total"].backward()
                opt.step()
                self.margins.clamp_min()

                for k in epoch_losses:
                    epoch_losses[k] += loss_dict[k].item()
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(1, n_batches)

            history["loss"].append(epoch_losses["total"])
            # Track ||W-P||_F for coupling analysis
            with torch.no_grad():
                wp_gap = (
                    self.head.W - self.get_reciprocal_parameters()
                ).norm(p="fro").item()
                history["components"].append({"wp_gap": wp_gap, **epoch_losses})

            if log_every and (epoch + 1) % log_every == 0:
                comps = {k: epoch_losses.get(k, 0) for k in
                         ["L_cls", "L_coupling", "L_open", "L_unif", "L_div"]}
                comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comps.items())
                print(f"  epoch {epoch+1:4d}/{num_epochs}  total={epoch_losses['total']:.6f}  "
                      + comp_str)

        self.timing["main_training_s"] = time.time() - stage_start
        return history

    def fit_mode_A(self, train_loader, num_epochs, val_loader=None,
                   log_every=5):
        """End-to-end SGD: all parameters optimised jointly."""
        print(f"Training mode A — {num_epochs} epochs, end-to-end AdamW")
        self.mode = "A"
        self.to(self.device)

        # Pre-warmup + Warm-up (same pipeline)
        stage_start = time.time()
        self._pre_warmup(train_loader)
        self.timing["pre_warmup_s"] = time.time() - stage_start
        stage_start = time.time()
        self._warmup(train_loader)
        self.timing["warmup_s"] = time.time() - stage_start

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            print("  Encoder frozen after warm-up (--freeze-encoder).")

        self._prepare_objective_stats(train_loader)
        opt = self._make_optimizer(mode="A")
        history = {"loss": [], "components": []}

        stage_start = time.time()
        for epoch in range(num_epochs):
            self.train()
            self._set_encoder_training_mode()
            epoch_losses = {k: 0.0 for k in [
                "total", "L_cls", "L_reg_W", "L_corr", "L_coupling",
                "L_open", "L_unif", "L_div"]}
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                feats = self._forward_encoder(x_batch)
                logits = self.head(feats)
                loss_dict = self._compute_losses(feats, logits, y_batch)

                opt.zero_grad()
                loss_dict["total"].backward()
                opt.step()
                self.margins.clamp_min()

                for k in epoch_losses:
                    epoch_losses[k] += loss_dict[k].item()
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(1, n_batches)

            history["loss"].append(epoch_losses["total"])
            with torch.no_grad():
                wp_gap = (
                    self.head.W - self.get_reciprocal_parameters()
                ).norm(p="fro").item()
                history["components"].append({"wp_gap": wp_gap, **epoch_losses})

            if log_every and (epoch + 1) % log_every == 0:
                comps = {k: epoch_losses.get(k, 0) for k in
                         ["L_cls", "L_coupling", "L_open", "L_unif", "L_div"]}
                comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comps.items())
                print(f"  epoch {epoch+1:4d}/{num_epochs}  total={epoch_losses['total']:.6f}  "
                      + comp_str)

        self.timing["main_training_s"] = time.time() - stage_start
        return history

    # ── Sylvester update (for Mode B) ───────────────────────────────────

    def _sylvester_update(self, train_loader):
        """Freeze encoder, accumulate features, solve W via Sylvester."""
        self.encoder.eval()
        all_features, all_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                all_features.append(self._forward_encoder(x_batch).cpu())
                all_targets.append(y_batch.cpu())

        F_all = torch.cat(all_features, dim=0).to(self.device)   # (N, d')
        Y_all = torch.cat(all_targets, dim=0).to(self.device)     # (N, q)
        N, d = F_all.shape
        q = Y_all.shape[1]

        W_old = self.head.W.data.clone()
        P = self.get_reciprocal_parameters().detach()

        # ── Solve Sylvester: S_A V + V S_B = S_C (all in centred space) ──
        # P2 fix: with b optimised out, the W-subproblem is
        #   min_W ½/N‖FcW−Yc‖² + ½λ₁‖W‖² + ½λ₃‖W−P‖²
        #         + ½λ₂tr(WLWᵀ),
        # whose first-order condition gives
        #   Sy_A = FcᵀFc + (λ₁+λ₃)I,  Sy_C = FcᵀYc + λ₃ P.
        # Previously Sy_A used the *uncentred* F_allᵀF_all while Sy_C used the
        # centred FcᵀYc — an inconsistent mix that made the closed form
        # non-optimal whenever mean(F) ≠ 0.  Both sides must use centred terms.
        Y_mean = Y_all.mean(dim=0, keepdim=True)       # (1, q)
        F_mean = F_all.mean(dim=0, keepdim=True)       # (1, d')
        Yc = Y_all - Y_mean
        Fc = F_all - F_mean

        effective_lamda3 = (
            0.0 if self.classifier_induced_reciprocal else self.lamda3)
        Sy_A = Fc.T @ Fc / float(N) + (
            self.lamda1 + effective_lamda3) * torch.eye(
            d, device=self.device, dtype=F_all.dtype)
        if self.corr_mod is not None and self.lamda2 > 0:
            with torch.no_grad():
                _, L_lap = self.corr_mod()
            Sy_B = self.lamda2 * L_lap
        else:
            Sy_B = torch.zeros(q, q, device=self.device, dtype=F_all.dtype)

        Sy_C = Fc.T @ Yc / float(N) + effective_lamda3 * P   # (d', q)

        # Solve
        V = solve_sylvester(Sy_A, Sy_B, Sy_C)            # (d', q)

        # ── Update parameters ──
        with torch.no_grad():
            self.head.W.copy_(V)
            # b = mean(Y) − mean(F) @ W
            b_new = (Y_all - F_all @ V).mean(dim=0)
            self.head.b.copy_(b_new)
            # P and R are intentionally not touched here.  They are the
            # gradient blocks for the paper hinge and shared Mode A/B loss.

        delta_W = (self.head.W - W_old).norm().item()
        print(f"    Sylvester update: ‖ΔW‖ = {delta_W:.6f}")

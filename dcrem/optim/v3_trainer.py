"""Validation-stage D-CREM v3 reciprocal-geometry trainer."""

import math
import time

import torch
import torch.nn.functional as F


class DCREMV3Trainer:
    """Classifier with a tangent residual reciprocal bank.

    Geometry gradients are isolated from ``W`` so open-set training cannot
    overwrite the known-label classifier.  Episodic held-out training labels
    provide pseudo-unknown samples using the training fold only.
    """

    def __init__(self, encoder, head, recip_bank, config, device):
        self.encoder = encoder.to(device)
        self.head = head.to(device)
        self.recip_bank = recip_bank.to(device)
        self.device = device
        self.no_l2norm = bool(config.get("no_l2norm", False))
        self.lamda1 = float(config.get("lamda1", 1.0))
        self.lamda3 = float(config.get("lamda3", 10.0))
        self.alpha = float(config.get("alpha", 0.1))
        self.pseudo_weight = float(config.get("pseudo_weight", 0.1))
        self.open_margin = float(config.get("open_margin", 2.0))
        self.pseudo_margin = float(config.get("pseudo_margin", 0.1))
        self.label_rank_weight = float(config.get("label_rank_weight", 0.0))
        self.label_rank_margin = float(config.get("label_rank_margin", 0.1))
        self.label_rank_hard_fraction = float(
            config.get("label_rank_hard_fraction", 1.0))
        if not 0 < self.label_rank_hard_fraction <= 1:
            raise ValueError("label_rank_hard_fraction must be in (0, 1]")
        self.holdout_fraction = float(config.get("holdout_fraction", 0.2))
        self.pseudo_target_fraction = float(
            config.get("pseudo_target_fraction", 0.3))
        self.top_k = int(config.get("pseudo_top_k", 3))
        self.lr = float(config.get("lr", 1e-4))
        self.backbone_lr = float(config.get("backbone_lr", 1e-5))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.pre_warmup_epochs = int(config.get("pre_warmup_epochs", 0))
        self.seed = int(config.get("seed", 0))
        self.pseudo_variant = str(config.get("pseudo_variant", "legacy"))
        self.development_primary_score = str(
            config.get("development_primary_score", "relative"))
        self.timing = {}

    def _forward_encoder(self, x):
        raw = self.encoder(x)
        return raw if self.no_l2norm else F.normalize(raw, p=2, dim=-1)

    def get_reciprocal_parameters(self):
        if self.pseudo_variant == "B0":
            return self.head.W
        if self.pseudo_variant == "legacy":
            return self.recip_bank(self.head.W)
        return self.recip_bank()

    def score_values(self, features):
        P = self.get_reciprocal_parameters()
        if P.ndim == 2:
            from dcrem.models.calibrator import OpenSetCalibrator
            return OpenSetCalibrator.compute_relative_scores(
                features, self.head.W, P)
        W_hat = F.normalize(self.head.W, p=2, dim=0)
        positive = (
            (features[:, :, None] - W_hat[None, :, :]) ** 2).sum(dim=1)
        return self.reciprocal_score_values(features) - positive

    def reciprocal_score_values(self, features):
        """Per-label distance to the nearest reciprocal prototype."""
        from dcrem.losses.open_space import reciprocal_distances

        P = self.get_reciprocal_parameters()
        if P.ndim == 2:
            return reciprocal_distances(features, P)
        distances = ((
            features[:, :, None, None] + P[None, :, :, :]) ** 2).sum(dim=1)
        return distances.min(dim=2).values

    def train(self):
        self.encoder.train()
        self.head.train()
        self.recip_bank.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()
        self.recip_bank.eval()

    def _pre_warmup(self, train_loader):
        if self.pre_warmup_epochs <= 0:
            return
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.pre_warmup_epochs):
            self.encoder.train()
            self.head.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.head(self._forward_encoder(x_batch))
                loss = 0.5 * F.mse_loss(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _warmup(self, train_loader):
        self.encoder.eval()
        features, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                features.append(self._forward_encoder(x_batch.to(self.device)))
                targets.append(y_batch.to(self.device))
        self.head.closed_form_init(
            torch.cat(features), torch.cat(targets),
            lamda1=self.lamda1, sample_average=True)
        # Freeze the warm-up classifier direction for hard-negative mining.
        # Selection is therefore independent of subsequent reciprocal updates.
        self.mining_W = self.head.W.detach().clone()
        if self.pseudo_variant in {"B1", "B2", "B3"}:
            if hasattr(self.recip_bank, "init_from_hard_negatives"):
                self.recip_bank.init_from_hard_negatives(
                    torch.cat(features), torch.cat(targets), self.head.W,
                    hard_fraction=self.label_rank_hard_fraction)
            else:
                self.recip_bank.init_from_W(self.head.W)

    def _holdout_schedule(self, train_loader, num_epochs):
        targets = torch.cat([batch_y for _, batch_y in train_loader], dim=0)
        q = targets.shape[1]
        max_count = min(q - 1, max(1, int(round(q * self.holdout_fraction))))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed * 100_003 + 71)
        schedule = []
        for _ in range(num_epochs):
            best = None
            best_gap = float("inf")
            for _ in range(64):
                count = int(torch.randint(
                    1, max_count + 1, (1,), generator=generator).item())
                candidate = torch.randperm(q, generator=generator)[:count]
                fraction = float(
                    (targets[:, candidate] == 1).any(dim=1).float().mean().item())
                gap = abs(fraction - self.pseudo_target_fraction)
                if gap < best_gap:
                    best, best_gap = candidate, gap
            heldout = best.to(self.device)
            known_mask = torch.ones(q, dtype=torch.bool, device=self.device)
            known_mask[heldout] = False
            schedule.append((heldout, known_mask))
        return schedule

    def _losses(self, features, logits, targets, heldout, known_mask):
        from dcrem.losses.open_space import reciprocal_distances

        known_count = int(known_mask.sum().item())
        if self.pseudo_variant == "B3":
            cls_logits = logits[:, known_mask]
            cls_targets = targets[:, known_mask]
        else:
            cls_logits = logits
            cls_targets = targets
        cls = 0.5 * F.mse_loss(
            cls_logits, cls_targets, reduction="sum")
        cls = cls / max(1, targets.shape[0])
        if self.pseudo_variant == "B3":
            # Keep the classification term on the same q-label scale as the
            # other variants after episodically hiding some label columns.
            cls = cls * (targets.shape[1] / max(1, known_count))
        reg = 0.5 * self.lamda1 * (self.head.W * self.head.W).sum()

        if self.pseudo_variant in {"B0", "B1", "B2", "B3"}:
            P = self.get_reciprocal_parameters()
            if P.ndim == 3:
                coupling_gap = self.head.W[:, :, None] - P
                coupling = (
                    0.5 * self.lamda3 * coupling_gap.pow(2).sum()
                    / P.shape[2])
                d_negative = self.reciprocal_score_values(features)
            else:
                coupling = (
                    0.5 * self.lamda3 * ((self.head.W - P) ** 2).sum()
                    if self.pseudo_variant != "B0"
                    else features.new_zeros(()))
                d_negative = reciprocal_distances(features, P)
            reciprocal = features.new_zeros(())

            # A label's observed negatives are train-fold examples of its
            # extra-class.  They should lie closer to reciprocal point -P_k
            # than its positives.  Pairwise ranking avoids a dataset-specific
            # absolute radius and supplies P with an independent signal.
            label_rank_terms = []
            if self.label_rank_weight > 0:
                mining_W = getattr(
                    self, "mining_W", self.head.W.detach())
                mining_W = F.normalize(mining_W, p=2, dim=0)
                mining_scores = features.detach() @ mining_W
                for label_index in range(targets.shape[1]):
                    positive_distances = d_negative[
                        targets[:, label_index] == 1, label_index]
                    negative_indices = torch.where(
                        targets[:, label_index] == -1)[0]
                    if (self.label_rank_hard_fraction < 1
                            and negative_indices.numel() > 0):
                        hard_count = max(1, math.ceil(
                            negative_indices.numel()
                            * self.label_rank_hard_fraction))
                        local_hard = torch.topk(
                            mining_scores[negative_indices, label_index],
                            k=hard_count).indices
                        negative_indices = negative_indices[local_hard]
                    negative_distances = d_negative[
                        negative_indices, label_index]
                    if (positive_distances.numel() > 0
                            and negative_distances.numel() > 0):
                        label_rank_terms.append(F.softplus(
                            self.label_rank_margin
                            + negative_distances[:, None]
                            - positive_distances[None, :]).mean())
            label_rank = (
                torch.stack(label_rank_terms).mean()
                if label_rank_terms else features.new_zeros(()))

            pseudo = features.new_zeros(())
            pseudo_mask = (targets[:, heldout] == 1).any(dim=1)
            normal_mask = ~pseudo_mask
            known_count = int(known_mask.sum().item())
            k = min(self.top_k, known_count)
            known_scores = torch.topk(
                d_negative[:, known_mask], k=k, dim=1).values.mean(dim=1)
            if (self.pseudo_variant in {"B2", "B3"}
                    and pseudo_mask.any() and normal_mask.any()):
                normal_scores = known_scores[normal_mask]
                pseudo_scores = known_scores[pseudo_mask]
                # Pairwise AUROC surrogate: every normal sample should be more
                # known-like than every pseudo-unknown sample by the margin.
                pseudo = F.softplus(
                    self.pseudo_margin
                    - normal_scores[:, None]
                    + pseudo_scores[None, :]).mean()

            total = (
                cls + reg + coupling + self.pseudo_weight * pseudo
                + self.label_rank_weight * label_rank)
            return {
                "total": total,
                "L_cls": cls,
                "L_reg_W": reg,
                "L_coupling": coupling,
                "L_reciprocal": reciprocal,
                "L_pseudo": self.pseudo_weight * pseudo,
                "L_label_rank": self.label_rank_weight * label_rank,
                "pseudo_fraction": pseudo_mask.float().mean(),
            }

        # Stop geometry gradients from changing the known-label classifier.
        W_geometry = self.head.W.detach()
        P = self.recip_bank(W_geometry)
        geometry_features = features.detach()
        d_negative = reciprocal_distances(geometry_features, P)
        W_hat = F.normalize(W_geometry, p=2, dim=0)
        d_positive = ((geometry_features[:, :, None] - W_hat[None, :, :]) ** 2).sum(dim=1)
        relative = d_negative - d_positive

        positive = (targets[:, known_mask] == 1)
        open_hinge = torch.clamp(
            self.open_margin - d_negative[:, known_mask], min=0.0)
        reciprocal = (
            open_hinge[positive].mean() if positive.any()
            else geometry_features.new_zeros(()))

        pseudo_mask = (targets[:, heldout] == 1).any(dim=1)
        normal_mask = ~pseudo_mask
        k = min(self.top_k, known_count)
        known_scores = torch.topk(relative[:, known_mask], k=k, dim=1).values.mean(dim=1)
        if pseudo_mask.any() and normal_mask.any():
            pseudo = F.softplus(
                self.pseudo_margin - known_scores[normal_mask].mean()
                + known_scores[pseudo_mask].mean())
        else:
            pseudo = geometry_features.new_zeros(())

        total = cls + reg + self.alpha * reciprocal + self.pseudo_weight * pseudo
        return {
            "total": total,
            "L_cls": cls,
            "L_reg_W": reg,
            "L_coupling": features.new_zeros(()),
            "L_reciprocal": self.alpha * reciprocal,
            "L_pseudo": self.pseudo_weight * pseudo,
            "L_label_rank": features.new_zeros(()),
            "pseudo_fraction": pseudo_mask.float().mean(),
        }

    def fit(self, train_loader, num_epochs):
        started = time.time()
        self._pre_warmup(train_loader)
        self.timing["pre_warmup_s"] = time.time() - started
        started = time.time()
        self._warmup(train_loader)
        self.timing["warmup_s"] = time.time() - started

        groups = []
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.backbone_lr,
                           "weight_decay": self.weight_decay})
        head_geometry_params = list(self.head.parameters())
        if self.pseudo_variant != "B0":
            head_geometry_params += list(self.recip_bank.parameters())
        groups.append({
            "params": head_geometry_params,
            "lr": self.lr, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups)
        history = {"loss": [], "components": []}
        started = time.time()
        schedule = self._holdout_schedule(train_loader, num_epochs)
        for epoch in range(num_epochs):
            self.train()
            heldout, known_mask = schedule[epoch]
            totals = {key: 0.0 for key in [
                "total", "L_cls", "L_reg_W", "L_coupling", "L_reciprocal",
                "L_pseudo", "L_label_rank", "pseudo_fraction"]}
            batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                features = self._forward_encoder(x_batch)
                logits = self.head(features)
                losses = self._losses(
                    features, logits, y_batch, heldout, known_mask)
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
                for key in totals:
                    totals[key] += float(losses[key].detach().item())
                batches += 1
            averaged = {key: value / max(1, batches) for key, value in totals.items()}
            history["loss"].append(averaged["total"])
            history["components"].append(averaged)
        self.timing["main_training_s"] = time.time() - started
        return history

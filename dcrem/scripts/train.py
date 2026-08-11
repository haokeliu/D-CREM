#!/usr/bin/env python3
"""D-CREM training entry point.

Launches D-CREM training on tabular or image datasets with the full
pipeline: warm-up → mode A/B training → evaluation.

Usage:
  # Tabular data, mode B
  python dcrem/scripts/train.py --dataset enron --mode B --epochs 50

  # Image data (pre-extracted features), mode A
  python dcrem/scripts/train.py --dataset voc2007 --feature resnet50 --mode A

  # Identity encoder smoke test (equivalence check)
  python dcrem/scripts/train.py --dataset enron --encoder identity --linear-kernel
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def _json_safe(value):
    """Recursively replace non-finite numeric values with JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _code_revision():
    """Return the current Git revision and dirty state without failing a run."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, encoding="utf-8", errors="replace").strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            text=True, encoding="utf-8", errors="replace").strip())
        return {"git_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_revision": None, "git_dirty": None}


def build_encoder(encoder_type, input_dim, **kwargs):
    """Build encoder by name."""
    from dcrem.models.encoder import (
        TabularMLP, ResNet50Encoder, CLIPViTEncoder, IdentityEncoder,
    )
    if encoder_type == "mlp":
        return TabularMLP(input_dim,
                          output_dim=kwargs.get("output_dim", 128))
    elif encoder_type == "resnet50":
        return ResNet50Encoder(
            frozen=kwargs.get("frozen", True),
            output_dim=kwargs.get("output_dim", 2048))
    elif encoder_type == "clip":
        return CLIPViTEncoder(frozen=kwargs.get("frozen", True))
    elif encoder_type == "identity":
        return IdentityEncoder(input_dim)
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")


def build_model(encoder, num_classes, config):
    """Build heads, reciprocal bank, margins, correlation module, calibrator."""
    from dcrem.models.heads import ClassifierHead, ReciprocalBank, MarginVector
    from dcrem.models.correlation import CorrelationModule, StaticCorrelationModule
    from dcrem.models.calibrator import OpenSetCalibrator

    d = encoder.output_dim
    head = ClassifierHead(d, num_classes)
    recip_bank = ReciprocalBank(d, num_classes)
    margins = MarginVector(num_classes)

    correlation_mode = config.get("correlation_mode", "learned")
    corr_mod = None
    if correlation_mode == "learned":
        corr_mod = CorrelationModule(
            num_classes,
            embed_dim=config.get("label_embed_dim", 128),
        )
    elif correlation_mode == "static_train":
        train_target = config.get("static_train_target")
        if train_target is None:
            raise ValueError("static_train correlation requires train_target")
        corr_mod = StaticCorrelationModule(train_target)
    elif correlation_mode != "none":
        raise ValueError(f"Unknown correlation mode: {correlation_mode}")

    calibrator = None
    if config.get("use_calibrator", False):
        calibrator = OpenSetCalibrator(num_classes)

    return head, recip_bank, margins, corr_mod, calibrator


def _distances_for(trainer, X, device):
    from dcrem.models.calibrator import OpenSetCalibrator

    tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    feats = trainer._forward_encoder(tensor)
    if hasattr(trainer, "reciprocal_score_values"):
        return trainer.reciprocal_score_values(feats).cpu().numpy()
    return OpenSetCalibrator.compute_distances(
        feats, _trainer_reciprocal_parameters(trainer)).cpu().numpy()


def _forward_inputs(trainer, inputs, device):
    """Encode either an in-memory feature matrix or an image DataLoader."""
    if isinstance(inputs, torch.utils.data.DataLoader):
        feature_batches, logit_batches = [], []
        for x_batch, _ in inputs:
            x_batch = x_batch.to(device, non_blocking=True)
            feats = trainer._forward_encoder(x_batch)
            feature_batches.append(feats.cpu())
            logit_batches.append(trainer.head(feats).cpu())
        return torch.cat(feature_batches), torch.cat(logit_batches)
    tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    feats = trainer._forward_encoder(tensor)
    return feats, trainer.head(feats)


def _trainer_reciprocal_parameters(trainer):
    if hasattr(trainer, "get_reciprocal_parameters"):
        return trainer.get_reciprocal_parameters()
    return trainer.recip_bank()


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _select_and_evaluate_score(val_values, val_labels, test_values, test_labels):
    """Select K for one predeclared score on validation and evaluate test once."""
    from dcrem.eval import evaluate_fixed_k, k_search_osr

    val_result, k_detail = k_search_osr(val_values, val_labels)
    selected_k = int(val_result["best_K"])
    test_result = evaluate_fixed_k(test_values, test_labels, selected_k)
    return test_result, {
        "selected_K": selected_k,
        "validation_metrics": val_result,
        "validation_k_search": {
            str(k): [float(v[0]), float(v[1])] for k, v in k_detail.items()
        },
    }


def evaluate(trainer, val_X, val_osr_labels, test_X, test_Y_QN,
             test_osr_labels, device):
    """Evaluate reciprocal and logit scores with independently locked K."""
    from dcrem.eval import compute_mll_metrics, evaluate_fixed_k, k_search_osr

    trainer.eval()
    with torch.no_grad():
        val_feats, val_logits = _forward_inputs(trainer, val_X, device)
        test_feats, test_logits = _forward_inputs(trainer, test_X, device)
        val_feats = val_feats.to(device)
        val_logits = val_logits.to(device)
        test_feats = test_feats.to(device)
        test_logits = test_logits.to(device)

        # Closed-set
        outputs_QN = test_logits.cpu().numpy().T
        mll = compute_mll_metrics(outputs_QN, test_Y_QN)

        from dcrem.models.calibrator import OpenSetCalibrator
        P = _trainer_reciprocal_parameters(trainer)
        if hasattr(trainer, "reciprocal_score_values"):
            val_distances = trainer.reciprocal_score_values(
                val_feats).cpu().numpy()
            test_distances = trainer.reciprocal_score_values(
                test_feats).cpu().numpy()
        else:
            val_distances = OpenSetCalibrator.compute_distances(
                val_feats, P).cpu().numpy()
            test_distances = OpenSetCalibrator.compute_distances(
                test_feats, P).cpu().numpy()

        reciprocal_metrics, reciprocal_selection = _select_and_evaluate_score(
            val_distances, val_osr_labels, test_distances, test_osr_labels)
        logit_metrics, logit_selection = _select_and_evaluate_score(
            val_logits.cpu().numpy(), val_osr_labels,
            test_logits.cpu().numpy(), test_osr_labels)

    requested_primary = getattr(trainer, "primary_score", "reciprocal")
    if requested_primary == "logit":
        primary_metrics, primary_selection = logit_metrics, logit_selection
    else:
        primary_metrics, primary_selection = (
            reciprocal_metrics, reciprocal_selection)
    selection = {
        **primary_selection,
        "primary_score": requested_primary,
        "score_evaluations": {
            "reciprocal": {
                "test_metrics": reciprocal_metrics,
                **reciprocal_selection,
            },
            "logit": {
                "test_metrics": logit_metrics,
                **logit_selection,
            },
        },
    }
    return {**mll, **primary_metrics}, selection


def evaluate_validation_only(trainer, val_X, val_Y_QN, val_osr_labels,
                             device):
    """Evaluate development candidates without reading or scoring the test fold."""
    from dcrem.eval import compute_mll_metrics, k_search_osr
    from dcrem.models.calibrator import OpenSetCalibrator

    trainer.eval()
    with torch.no_grad():
        val_tensor = torch.as_tensor(val_X, dtype=torch.float32, device=device)
        val_feats = trainer._forward_encoder(val_tensor)
        val_logits = trainer.head(val_feats)
        P = _trainer_reciprocal_parameters(trainer)
        if hasattr(trainer, "reciprocal_score_values"):
            val_distances = trainer.reciprocal_score_values(
                val_feats).cpu().numpy()
        else:
            val_distances = OpenSetCalibrator.compute_distances(
                val_feats, P).cpu().numpy()
        reciprocal, reciprocal_detail = k_search_osr(
            val_distances, val_osr_labels)
        logit, logit_detail = k_search_osr(
            val_logits.cpu().numpy(), val_osr_labels)
        mll = compute_mll_metrics(val_logits.cpu().numpy().T, val_Y_QN)
        relative = relative_detail = relative_values = None
        if hasattr(trainer, "score_values"):
            relative_values = trainer.score_values(val_feats).cpu().numpy()
            relative, relative_detail = k_search_osr(
                relative_values, val_osr_labels)

    requested_primary = getattr(
        trainer, "primary_score",
        getattr(
            trainer, "development_primary_score",
            "relative" if relative is not None else "reciprocal"))
    if requested_primary == "logit":
        primary = logit
        primary_detail = logit_detail
    elif requested_primary == "relative" and relative is not None:
        primary = relative
        primary_detail = relative_detail
    else:
        primary = reciprocal
        primary_detail = reciprocal_detail
    score_evaluations = {
        "reciprocal": {
            "validation_metrics": reciprocal,
            "selected_K": int(reciprocal["best_K"]),
            "validation_k_search": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in reciprocal_detail.items()
            },
        },
        "logit": {
            "validation_metrics": logit,
            "selected_K": int(logit["best_K"]),
            "validation_k_search": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in logit_detail.items()
            },
        },
    }
    if relative is not None:
        score_evaluations["relative"] = {
            "validation_metrics": relative,
            "selected_K": int(relative["best_K"]),
            "validation_k_search": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in relative_detail.items()
            },
        }
    selection = {
        "primary_score": requested_primary,
        "selected_K": int(primary["best_K"]),
        "validation_metrics": primary,
        "validation_k_search": {
            str(k): [float(v[0]), float(v[1])]
            for k, v in primary_detail.items()
        },
        "score_evaluations": score_evaluations,
        "score_geometry": {
            "reciprocal_vs_logit_per_label_pearson": _safe_correlation(
                val_distances, val_logits.cpu().numpy()),
            "relative_vs_logit_per_label_pearson": (
                _safe_correlation(relative_values, val_logits.cpu().numpy())
                if relative_values is not None else None),
        },
    }
    return {**mll, **primary}, selection


def collect_training_diagnostics(trainer, train_loader):
    """Collect one deterministic post-training batch of scale/geometry data."""
    import torch.nn.functional as F
    from dcrem.losses.open_space import reciprocal_distances

    trainer.eval()
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(trainer.device)
    y_batch = y_batch.to(trainer.device)
    raw = trainer.encoder(x_batch)
    feats = trainer._forward_encoder(x_batch)
    logits = trainer.head(feats)
    diagnostics = {
        "batch_size": int(x_batch.shape[0]),
        "raw_feature_norm": {
            "mean": float(raw.norm(dim=1).mean().item()),
            "std": float(raw.norm(dim=1).std(unbiased=False).item()),
        },
        "feature_norm": {
            "mean": float(feats.norm(dim=1).mean().item()),
            "std": float(feats.norm(dim=1).std(unbiased=False).item()),
        },
        "W_fro": float(trainer.head.W.norm(p="fro").item()),
    }
    if hasattr(trainer, "get_reciprocal_parameters"):
        P = trainer.get_reciprocal_parameters()
        if P.ndim == 3:
            mean_P = P.mean(dim=2)
            cosine = F.cosine_similarity(
                trainer.head.W[:, :, None], P, dim=0)
            prototype_gaps = (
                trainer.head.W[:, :, None] - P).norm(dim=0)
            pairwise = []
            if P.shape[2] > 1:
                for label_index in range(P.shape[1]):
                    pairwise.append(torch.pdist(P[:, label_index, :].T))
            diagnostics.update({
                "P_fro": float(P.norm().item()),
                "W_P_gap_fro": float(
                    (trainer.head.W - mean_P).norm(p="fro").item()),
                "W_P_column_cosine_mean": float(cosine.mean().item()),
                "W_P_column_cosine_min": float(cosine.min().item()),
                "reciprocal_prototypes": int(P.shape[2]),
                "W_P_prototype_gap_mean": float(
                    prototype_gaps.mean().item()),
                "prototype_pairwise_distance_mean": (
                    float(torch.cat(pairwise).mean().item())
                    if pairwise else 0.0),
            })
        else:
            cosine = F.cosine_similarity(trainer.head.W.T, P.T, dim=1)
            diagnostics.update({
                "P_fro": float(P.norm(p="fro").item()),
                "W_P_gap_fro": float(
                    (trainer.head.W - P).norm(p="fro").item()),
                "W_P_column_cosine_mean": float(cosine.mean().item()),
                "W_P_column_cosine_min": float(cosine.min().item()),
            })
    if not hasattr(trainer, "_compute_losses"):
        return diagnostics

    loss_dict = trainer._compute_losses(feats, logits, y_batch)
    diagnostics["loss_components"] = {
        key: float(value.detach().item()) for key, value in loss_dict.items()
    }
    W = trainer.head.W
    P = _trainer_reciprocal_parameters(trainer)
    R = trainer.margins()
    with torch.no_grad():
        cosine = F.cosine_similarity(W.T, P.T, dim=1)
        d2 = reciprocal_distances(feats, P)
        if trainer.radius_free_open:
            active = trainer.open_margin - d2 > 0
        else:
            active = trainer.open_margin + (R * R).unsqueeze(0) - d2 > 0
        positive = y_batch == 1
        active_positive = active & positive
        diagnostics.update({
            "P_fro": float(P.norm(p="fro").item()),
            "W_P_gap_fro": float((W - P).norm(p="fro").item()),
            "W_P_column_cosine_mean": float(cosine.mean().item()),
            "W_P_column_cosine_min": float(cosine.min().item()),
            "R": {
                "mean": float(R.mean().item()),
                "min": float(R.min().item()),
                "max": float(R.max().item()),
            },
            "positive_hinge_active_fraction": float(
                active_positive.sum().item() / max(1, positive.sum().item())),
        })

    groups = {
        "encoder": [p for p in trainer.encoder.parameters() if p.requires_grad],
        "W": [trainer.head.W],
        "P": list(trainer.recip_bank.parameters()),
        "R": list(trainer.margins.parameters()),
    }
    differentiable_groups = {
        name: [p for p in params if p.requires_grad]
        for name, params in groups.items()
    }
    flat_params = [
        p for values in differentiable_groups.values() for p in values]
    gradient_norms = {}
    for loss_name in ["L_cls", "L_reg_W", "L_corr", "L_coupling",
                      "L_open", "L_unif", "L_div"]:
        loss = loss_dict[loss_name]
        if not loss.requires_grad:
            gradient_norms[loss_name] = {name: 0.0 for name in groups}
            continue
        grads = torch.autograd.grad(
            loss, flat_params, retain_graph=True, allow_unused=True)
        offset = 0
        per_group = {}
        for name, params in differentiable_groups.items():
            selected = grads[offset:offset + len(params)]
            offset += len(params)
            squared = sum(
                float(g.detach().pow(2).sum().item())
                for g in selected if g is not None)
            per_group[name] = squared ** 0.5
        gradient_norms[loss_name] = per_group
    diagnostics["gradient_norms"] = gradient_norms
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="D-CREM training")
    # Data
    parser.add_argument("--dataset", type=str, default="enron",
                        help="Dataset name (enron, slashdot, bibtex, voc2007, coco2014)")
    parser.add_argument("--feature", type=str, default=None,
                        help="Feature type for image datasets (pca, resnet50, clip)")
    parser.add_argument(
        "--raw-images", action="store_true",
        help="Load raw VOC2007 images for end-to-end backbone training")
    parser.add_argument("--known-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-standardize", action="store_true",
                        help="Disable train-fitted feature standardisation")
    parser.add_argument("--non-deterministic", action="store_true",
                        help="Allow nondeterministic PyTorch/CUDA kernels")

    # Model
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "resnet50", "clip", "identity"])
    correlation_group = parser.add_mutually_exclusive_group()
    correlation_group.add_argument("--no-correlation", action="store_true",
                                   help="Disable label-correlation regularisation")
    correlation_group.add_argument(
        "--static-correlation", action="store_true",
        help="Use a frozen co-occurrence Laplacian built from the training fold")
    parser.add_argument("--use-calibrator", action="store_true",
                        help="Enable learnable OSR calibrator")
    parser.add_argument("--linear-kernel", action="store_true",
                        help="Use linear kernel (for identity encoder equivalence test)")
    parser.add_argument("--no-l2norm", action="store_true",
                        help="Skip L2 normalization (ablation A1)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder parameters (ablation A2)")
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze the pretrained visual backbone but train its projection")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warm-up initialization (ablation A7)")
    parser.add_argument(
        "--classifier-induced-reciprocal", action="store_true",
        help="Use P:=W exactly; no free reciprocal parameters or coupling loss")
    parser.add_argument(
        "--primary-score", choices=["reciprocal", "logit"],
        default="reciprocal",
        help="Predeclared OSR score used for the reported primary metrics")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Tag for output directory (e.g. 'ablation_A1')")
    parser.add_argument(
        "--v3-objective", action="store_true",
        help="Validation-only residual reciprocal + episodic pseudo-unknown objective")
    parser.add_argument(
        "--development-only", action="store_true",
        help="Save validation-only diagnostics under analysis_cache; never score test")

    # Training
    parser.add_argument("--mode", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--embedding-dim", type=int, default=128,
        help="Output dimension of the tabular MLP representation")
    parser.add_argument(
        "--block-interval", type=int, default=10,
        help="Epoch interval between exact W/b updates in Mode B")
    parser.add_argument(
        "--skip-summary-refresh", action="store_true",
        help=argparse.SUPPRESS)

    # Hyperparameters
    parser.add_argument("--lamda1", type=float, default=1.0)
    parser.add_argument("--lamda2", type=float, default=0.1)
    parser.add_argument("--lamda3", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--open-reduction", choices=["sum", "mean"],
                        default="sum")
    parser.add_argument("--radius-free-open", action="store_true")
    parser.add_argument("--open-margin", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma-div", type=float, default=0.01)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--pseudo-weight", type=float, default=0.1)
    parser.add_argument("--pseudo-margin", type=float, default=0.1)
    parser.add_argument("--label-rank-weight", type=float, default=0.0)
    parser.add_argument("--label-rank-margin", type=float, default=0.1)
    parser.add_argument("--label-rank-hard-fraction", type=float, default=1.0)
    parser.add_argument("--reciprocal-prototypes", type=int, default=1)
    parser.add_argument("--hard-prototype-init", action="store_true")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--pseudo-target-fraction", type=float, default=0.3)
    parser.add_argument("--pseudo-top-k", type=int, default=3)
    parser.add_argument(
        "--pseudo-variant", choices=["legacy", "B0", "B1", "B2", "B3"],
        default="legacy",
        help="Validation-only pseudo-unknown ablation variant")
    parser.add_argument(
        "--development-primary-score", choices=["reciprocal", "relative"],
        default="relative")

    args = parser.parse_args()
    if args.static_correlation and args.lamda2 <= 0:
        parser.error("--static-correlation requires --lamda2 > 0")
    if args.v3_objective and args.mode != "A":
        parser.error("--v3-objective requires --mode A")
    if args.v3_objective and not args.development_only:
        parser.error("v3 is development-only until its validation gate passes")
    if args.reciprocal_prototypes > 1 and not args.v3_objective:
        parser.error("multiple reciprocal prototypes require --v3-objective")
    if args.hard_prototype_init and not args.v3_objective:
        parser.error("hard prototype initialization requires --v3-objective")
    if args.embedding_dim <= 0:
        parser.error("--embedding-dim must be positive")
    if args.block_interval <= 0:
        parser.error("--block-interval must be positive")
    if args.raw_images and (args.dataset != "voc2007" or args.encoder != "resnet50"):
        parser.error("--raw-images currently requires --dataset voc2007 --encoder resnet50")

    # Seed before constructing loaders or model parameters.
    from dcrem.reproducibility import seed_everything
    reproducibility = seed_everything(
        args.seed, deterministic=not args.non_deterministic)

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    reproducibility.update({
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
    })

    # ── Data ──
    IMAGE_DATASETS = {"voc2007", "coco2014"}
    if args.raw_images:
        from dcrem.data.image import get_voc2007_raw_protocol
        protocol = get_voc2007_raw_protocol(
            known_ratio=args.known_ratio, seed=args.seed,
            batch_size=args.batch_size, train_ratio=args.train_ratio,
            val_ratio=args.val_ratio)
    elif args.dataset in IMAGE_DATASETS:
        ft = args.feature or "resnet50"
        from dcrem.data.image import get_image_protocol
        protocol = get_image_protocol(
            args.dataset, ft, known_ratio=args.known_ratio, seed=args.seed,
            standardize=not args.no_standardize, batch_size=args.batch_size,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    else:
        from dcrem.data.tabular import get_tabular_protocol
        protocol = get_tabular_protocol(
            args.dataset, known_ratio=args.known_ratio, seed=args.seed,
            standardize=not args.no_standardize, batch_size=args.batch_size,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_loader = protocol["train_loader"]
    input_dim = protocol["input_dim"]
    num_classes = protocol["num_classes"]
    print(f"Data: input_dim={input_dim}, num_classes={num_classes}, "
          f"train_batches={len(train_loader)}")

    # ── Model ──
    correlation_mode = (
        "static_train" if args.static_correlation else
        "none" if args.no_correlation else "learned"
    )
    encoder = build_encoder(
        args.encoder, input_dim,
        frozen=(args.freeze_backbone or args.encoder == "clip"),
        output_dim=args.embedding_dim)
    if args.v3_objective:
        from dcrem.models.heads import (
            ClassifierHead, MultiReciprocalBank, ReciprocalBank,
            ResidualReciprocalBank)
        head = ClassifierHead(encoder.output_dim, num_classes)
        if args.pseudo_variant == "legacy":
            recip_bank = ResidualReciprocalBank(
                encoder.output_dim, num_classes,
                residual_scale=args.residual_scale)
        elif args.hard_prototype_init or args.reciprocal_prototypes > 1:
            recip_bank = MultiReciprocalBank(
                encoder.output_dim, num_classes,
                num_prototypes=args.reciprocal_prototypes)
        else:
            recip_bank = ReciprocalBank(encoder.output_dim, num_classes)
        margins = corr_mod = calibrator = None
        correlation_mode = "not_applicable_v3"
    else:
        head, recip_bank, margins, corr_mod, calibrator = \
            build_model(encoder, num_classes, {
                "correlation_mode": correlation_mode,
                "static_train_target": (
                    protocol["split"]["train_target"]
                    if correlation_mode == "static_train" else None),
                "use_calibrator": args.use_calibrator,
            })
    print(f"Encoder: {args.encoder} (output_dim={encoder.output_dim})")
    print(f"Params: encoder={sum(p.numel() for p in encoder.parameters()):,}, "
          f"head={head.W.numel() + head.b.numel():,}, "
          f"P={sum(p.numel() for p in recip_bank.parameters()) if recip_bank is not None else 0:,}, "
          f"R={margins.raw_R.numel() if margins is not None else 0:,}")

    # ── Trainer config ──
    trainer_config = {
        "lamda1": args.lamda1, "lamda2": args.lamda2, "lamda3": args.lamda3,
        "alpha": args.alpha, "beta": args.beta,
        "open_reduction": args.open_reduction,
        "radius_free_open": args.radius_free_open,
        "open_margin": args.open_margin,
        "gamma": args.gamma_div,           # L_div weight
        "tau": 2.0, "theta_div": 0.9,
        "lr": args.lr, "backbone_lr": args.backbone_lr,
        "weight_decay": 1e-4,
        "T_sylvester": args.block_interval, "T_warmup": 5,
        "pre_warmup_epochs": (
            0 if args.no_warmup else 10 if args.encoder == "mlp" else 0),
        "no_l2norm": args.no_l2norm,
        "freeze_encoder": args.freeze_encoder,
        "no_warmup": args.no_warmup,
        "classifier_induced_reciprocal": args.classifier_induced_reciprocal,
        "primary_score": args.primary_score,
        "residual_scale": args.residual_scale,
        "pseudo_weight": args.pseudo_weight,
        "pseudo_margin": args.pseudo_margin,
        "label_rank_weight": args.label_rank_weight,
        "label_rank_margin": args.label_rank_margin,
        "label_rank_hard_fraction": args.label_rank_hard_fraction,
        "reciprocal_prototypes": args.reciprocal_prototypes,
        "holdout_fraction": args.holdout_fraction,
        "pseudo_target_fraction": args.pseudo_target_fraction,
        "pseudo_top_k": args.pseudo_top_k,
        "pseudo_variant": args.pseudo_variant,
        "development_primary_score": args.development_primary_score,
        "seed": args.seed,
    }
    trainer_config["lamda2"] = (
        args.lamda2 if correlation_mode not in (
            "none", "not_applicable_v3")
        else 0.0)

    # ── Trainer ──
    if args.v3_objective:
        from dcrem.optim.v3_trainer import DCREMV3Trainer
        trainer = DCREMV3Trainer(
            encoder, head, recip_bank, trainer_config, device)
    else:
        from dcrem.optim.trainer import DCREMTrainer
        trainer = DCREMTrainer(
            encoder, head, recip_bank, margins,
            corr_mod=corr_mod, calibrator=calibrator,
            config=trainer_config,
        )
        trainer.to(device)

    # ── Train ──
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    if args.v3_objective:
        history = trainer.fit(train_loader, args.epochs)
    elif args.mode == "B":
        history = trainer.fit_mode_B(train_loader, args.epochs, log_every=5)
    else:
        history = trainer.fit_mode_A(train_loader, args.epochs, log_every=5)
    t_train = time.time() - t0
    print(f"Training time: {t_train:.1f}s")

    # ── Evaluate ──
    t0 = time.time()
    if args.development_only:
        metrics, selection = evaluate_validation_only(
            trainer, protocol["val_X"], protocol["val_Y_QN"],
            protocol["val_osr_labels"], device)
    else:
        metrics, selection = evaluate(
            trainer,
            protocol["val_X"], protocol["val_osr_labels"],
            protocol["test_X"], protocol["test_Y_QN"],
            protocol["test_osr_labels"], device)
    t_eval = time.time() - t0
    max_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        if device.type == "cuda" else None)
    post_training = collect_training_diagnostics(trainer, train_loader)

    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: N/A")

    # ── Save ──
    setting = f"protocol_v2_mode{args.mode}_r{args.known_ratio}"
    if args.run_tag:
        setting = f"protocol_v2_{args.run_tag}_r{args.known_ratio}"
    if args.development_only:
        run_dir = os.path.join(
            REPO_ROOT, "results", "analysis_cache_protocol_v2",
            "development", args.dataset, setting)
    else:
        run_dir = os.path.join(REPO_ROOT, "results", "dcrem", args.dataset, setting)
    os.makedirs(run_dir, exist_ok=True)
    if not args.v3_objective:
        training_objective = (
            "classifier_induced_hyperspherical_core"
            if args.classifier_induced_reciprocal else "full")
    elif args.reciprocal_prototypes > 1:
        training_objective = "v3_multimodal_reciprocal_prototypes"
    elif args.hard_prototype_init:
        training_objective = "v3_single_hard_negative_prototype_init"
    elif args.label_rank_weight > 0:
        training_objective = (
            "v3_frozen_classifier_hard_negative_reciprocal_ranking"
            if args.label_rank_hard_fraction < 1
            else "v3_label_conditional_reciprocal_ranking")
    else:
        training_objective = (
            "v3_residual_pseudo_unknown" if args.pseudo_variant == "legacy"
            else f"v3_pseudo_gate_{args.pseudo_variant.lower()}")
    record = _json_safe({
        "metrics": {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
        "config": {
            **vars(args),
            "correlation_mode": correlation_mode,
            "training_objective": training_objective,
            "effective_lamda2": trainer_config["lamda2"],
            "static_c_source": (
                "train_target_only" if correlation_mode == "static_train" else None),
        },
        "protocol": {
            "version": 2,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "preprocessing_fit": "train_only",
            "evaluation_scope": (
                "validation_only" if args.development_only else "formal_test"),
            "test_accessed": not args.development_only,
            **selection,
        },
        "reproducibility": {**reproducibility, **_code_revision()},
        "diagnostics": {"history": history, "post_training": post_training},
        "time": {
            "train_s": round(t_train, 3), "eval_s": round(t_eval, 3),
            "max_cuda_memory_mb": (
                round(max_cuda_memory_mb, 3)
                if max_cuda_memory_mb is not None else None),
            **{key: round(value, 3) for key, value in trainer.timing.items()},
        },
        "timestamp": datetime.now().isoformat(),
    })
    path = os.path.join(run_dir, f"seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str, allow_nan=False)
    print(f"\nSaved: {path}")

    if not args.development_only and not args.skip_summary_refresh:
        from scripts.build_results_report import write_report
        print(f"Result summary: {write_report()}")


if __name__ == "__main__":
    main()

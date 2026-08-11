#!/usr/bin/env python3
"""Generate all Phase 3 analysis figures and tables from D-CREM results.

Produces:
  Fig_convergence  — loss curves for tabular + image, Mode A + Mode B
  Fig_coupling     — ||W-P||_F evolution during training
  Fig_R_dist       — R_k histogram across datasets
  Fig_tsne         — test features + reciprocal points t-SNE
  Fig_CD           — Friedman critical difference diagram
  Table_case       — case study: correct rejections + false accepts
  Table_Rk         — R_k per-label statistics
  Table_time       — efficiency comparison (already exists, just regenerate)

The script re-runs short training sessions with history capture, then
generates all plots from the captured data + existing result JSONs.

Usage:
  python scripts/run_analysis.py --all
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FIGS_DIR = os.path.join(REPO_ROOT, "results", "figures")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

TABULAR_DATASETS = ["enron", "slashdot", "bibtex",
                    "yahoo-recreation", "yahoo-arts", "yahoo-education"]
RATIOS = [0.3, 0.5, 0.7]
SEEDS = list(range(10))

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: collect results from saved JSONs
# ═══════════════════════════════════════════════════════════════════════════════

def load_seeds(result_dir, seeds=SEEDS):
    """Load all seed result JSONs, sorted by seed."""
    runs = []
    for s in seeds:
        path = os.path.join(result_dir, f"seed{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs.append(json.load(f))
    return sorted(runs, key=lambda x: x.get("seed", 0))


def mean_std_metric(runs, metric):
    vals = [r["metrics"].get(metric, float("nan")) for r in runs]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return 0, 0
    return np.mean(vals), np.std(vals)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fig_convergence + Fig_coupling: requires re-running training with history
# ═══════════════════════════════════════════════════════════════════════════════

def run_training_with_history(dataset, encoder, mode, known_ratio, seed,
                              feature=None, epochs=50, batch_size=64):
    """Run a single D-CREM training with history capture, return history dict."""
    import torch
    from dcrem.reproducibility import seed_everything
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if dataset in ("voc2007", "coco2014"):
        ft = feature or "clip"
        from dcrem.data.image import get_image_protocol
        protocol = get_image_protocol(
            dataset, ft, known_ratio=known_ratio, seed=seed,
            batch_size=batch_size)
    else:
        from dcrem.data.tabular import get_tabular_protocol
        protocol = get_tabular_protocol(
            dataset, known_ratio=known_ratio, seed=seed,
            batch_size=batch_size)
    train_loader = protocol["train_loader"]
    test_X = protocol["test_X"]
    test_Y_QN = protocol["test_Y_QN"]
    osr_labels = protocol["test_osr_labels"]
    input_dim = protocol["input_dim"]
    num_classes = protocol["num_classes"]

    # Build model
    from dcrem.models.encoder import TabularMLP, IdentityEncoder
    from dcrem.models.heads import ClassifierHead, ReciprocalBank, MarginVector
    from dcrem.models.correlation import CorrelationModule

    if encoder == "identity":
        enc = IdentityEncoder(input_dim)
    else:
        enc = TabularMLP(input_dim)

    head = ClassifierHead(enc.output_dim, num_classes)
    recip = ReciprocalBank(enc.output_dim, num_classes)
    margins = MarginVector(num_classes)
    corr = CorrelationModule(num_classes, embed_dim=128)

    config = {
        "lamda1": 1.0, "lamda2": 0.1, "lamda3": 10.0,
        "alpha": 1.0, "beta": 0.1, "gamma": 0.01,
        "tau": 2.0, "theta_div": 0.9,
        "lr": 1e-4, "backbone_lr": 1e-5,
        "weight_decay": 1e-4, "T_sylvester": 10, "T_warmup": 5,
        "pre_warmup_epochs": 10 if encoder == "mlp" else 0,
        "no_l2norm": False, "freeze_encoder": False, "no_warmup": False,
    }

    from dcrem.optim.trainer import DCREMTrainer
    trainer = DCREMTrainer(enc, head, recip, margins, corr_mod=corr, config=config)
    trainer.to(device)

    print(f"  Training {dataset} encoder={encoder} mode={mode} r={known_ratio} s={seed}...")
    t0 = time.time()

    if mode == "B":
        history = trainer.fit_mode_B(train_loader, epochs, log_every=5)
    else:
        history = trainer.fit_mode_A(train_loader, epochs, log_every=5)

    t_train = time.time() - t0
    print(f"  Done in {t_train:.1f}s — {len(history.get('loss',[]))} epochs")

    # Also compute final metrics and capture R_k values
    trainer.eval()
    with torch.no_grad():
        X_test = torch.as_tensor(test_X, dtype=torch.float32).to(device)
        feats_t = trainer._forward_encoder(X_test)
        feats = feats_t.cpu().numpy()
        P_t = trainer.recip_bank()
        P = P_t.cpu().numpy()
        if hasattr(trainer.margins, 'raw_R'):
            R_k = torch.nn.functional.softplus(trainer.margins.raw_R).detach().cpu().numpy()
        else:
            R_k = trainer.margins().detach().cpu().numpy()
        W_np = trainer.head.W.detach().cpu().numpy()
        logits = trainer.head(feats_t).cpu().numpy()
        from dcrem.models.calibrator import OpenSetCalibrator
        distances = OpenSetCalibrator.compute_distances(feats_t, P_t).cpu().numpy()

        # Select K on validation and apply the locked K to test.
        osr_labels_flat = np.asarray(osr_labels).ravel()
        X_val = torch.as_tensor(
            protocol["val_X"], dtype=torch.float32, device=device)
        val_feats = trainer._forward_encoder(X_val)
        val_distances = OpenSetCalibrator.compute_distances(
            val_feats, P_t).cpu().numpy()
        from dcrem.eval import evaluate_fixed_k, k_search_osr, top_k_scores
        val_res, _ = k_search_osr(
            val_distances, protocol["val_osr_labels"])
        best_k = int(val_res["best_K"])
        osr_res = evaluate_fixed_k(distances, osr_labels_flat, best_k)
        osr_score = top_k_scores(distances, best_k)

    # Save analysis cache
    cache = {
        "history": history,
        "W": W_np.tolist(),
        "P": P.tolist(),
        "R_k": R_k.tolist(),
        "features": feats.tolist(),
        "logits": logits.tolist(),
        "distances": distances.tolist(),
        "osr_labels": osr_labels_flat.tolist(),
        "osr_score": osr_score.tolist(),
        "macroAUC": osr_res.get("macroAUC", 0),
        "AUROC": osr_res.get("AUROC", 0),
        "AUPR": osr_res.get("AUPR", 0),
        "best_K": best_k,
        "test_Y": test_Y_QN.tolist() if hasattr(test_Y_QN, 'tolist') else (
            test_Y_QN if isinstance(test_Y_QN, list) else list(test_Y_QN)),
        "num_classes": num_classes,
        "config": {k: str(v) for k, v in config.items()},
    }

    cache_dir = os.path.join(REPO_ROOT, "results", "analysis_cache_protocol_v2")
    os.makedirs(cache_dir, exist_ok=True)
    ftag = f"_{feature}" if feature else ""
    cache_path = os.path.join(cache_dir,
        f"{dataset}{ftag}_{encoder}_mode{mode}_r{known_ratio}_s{seed}.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"  Cached: {cache_path}")

    return cache


def plot_convergence(caches, output_dir):
    """Plot loss convergence from cached training histories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping convergence plot")
        return

    panels = [
        ("Mode B: Tabular (slashdot, r=0.5)", [c for c in caches if "slashdot" in c["name"] and "modeB" in c["name"]]),
        ("Mode B: Image (VOC2007 CLIP, r=0.5)", [c for c in caches if "voc2007" in c["name"] and "clip" in c["name"] and "modeB" in c["name"]]),
    ]
    panels = [(title, subset) for title, subset in panels if subset]
    if not panels:
        print("  No convergence histories available, skipping convergence plot")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    axes = np.atleast_1d(axes)

    for ax, (title, caches_subset) in zip(axes, panels):
        for cache in caches_subset:
            data = cache["data"]
            hist = data.get("history", {})
            losses = hist.get("loss", [])
            if losses:
                ax.plot(losses, alpha=0.7, label=cache["name"])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_coupling(caches, output_dir):
    """Plot ||W-P||_F evolution from cached training histories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping coupling plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for c in caches:
        data = c["data"]
        hist = data.get("history", {})
        components = hist.get("components", [])
        wp_gaps = [comp.get("wp_gap", 0) for comp in components if "wp_gap" in comp]
        if wp_gaps and len(wp_gaps) > 2:
            ax.plot(wp_gaps, label=c["name"], alpha=0.8)

    ax.set_title("||W - P||_F Evolution During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||W - P||_F")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_coupling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Fig_R_dist: R_k distribution across datasets
# ═══════════════════════════════════════════════════════════════════════════════

def plot_R_distribution(caches, output_dir):
    """Plot R_k histogram from cached analysis data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping R distribution plot")
        return

    # Collect R_k from multiple cached runs
    all_R_vals = {}
    for c in caches:
        name = c["name"]
        data = c["data"]
        R_k = np.array(data.get("R_k", []))
        if len(R_k) > 0:
            all_R_vals[name] = R_k

    panels = [
        ("Tabular", ["slashdot", "enron", "bibtex"]),
        ("Image", ["voc2007"]),
    ]
    panels = [
        (title_suffix, datasets_subset)
        for title_suffix, datasets_subset in panels
        if any(any(dataset in name for dataset in datasets_subset)
               for name in all_R_vals)
    ]
    if not panels:
        print("  No R_k values available, skipping R_k distribution plot")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_R_vals)))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for ax, (title_suffix, datasets_subset) in zip(axes, panels):
        for i, (name, R_k) in enumerate(all_R_vals.items()):
            if any(d in name for d in datasets_subset):
                ax.hist(R_k, bins=15, alpha=0.5, label=name, color=colors[i % len(colors)])
        ax.set_title(f"R_k Distribution — {title_suffix}")
        ax.set_xlabel("R_k value")
        ax.set_ylabel("Count")
        if ax.patches:
            ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_R_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fig_tsne: test features + reciprocal points
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne(cache_data, output_dir, name_suffix=""):
    """t-SNE of test features + reciprocal points, colored by OSR label."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  sklearn/matplotlib not available, skipping t-SNE")
        return

    feats = np.array(cache_data["features"])
    P = np.array(cache_data["P"])  # (d, q)
    osr = np.array(cache_data["osr_labels"])  # 1=含未知标签, 0=纯已知

    # Subsample for speed
    n_test = feats.shape[0]
    n_sample = min(500, n_test)
    idx = np.random.RandomState(42).choice(n_test, n_sample, replace=False)
    feats_sample = feats[idx]
    osr_sample = osr[idx]

    # Combine features + P points
    combined = np.vstack([feats_sample, P.T])
    n_combined = combined.shape[0]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_combined - 1))
    embedded = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot test samples
    for label, color, name in [(0, "#2ecc71", "Known only"), (1, "#e74c3c", "Has unknown")]:
        mask = osr_sample == label
        if mask.any():
            ax.scatter(embedded[:n_sample][mask, 0], embedded[:n_sample][mask, 1],
                      c=color, label=name, alpha=0.5, s=15)

    # Plot reciprocal points
    ax.scatter(embedded[n_sample:, 0], embedded[n_sample:, 1],
              c="#3498db", marker="X", s=80, edgecolors="black", linewidths=0.5,
              label="Reciprocal points p_k")

    ax.set_title(f"t-SNE: Test Features + Reciprocal Points{name_suffix}")
    ax.legend(fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    path = os.path.join(output_dir, f"Fig_tsne{name_suffix.replace(' ','_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fig_CD: Friedman Critical Difference diagram
# ═══════════════════════════════════════════════════════════════════════════════

def build_cd_diagram(output_dir):
    """Build a Protocol-v2 Friedman average-rank table and plot."""
    methods = ["OC-SVM", "IFOREST", "SLAN", "CREM",
               "D-CREM Mode A", "D-CREM Mode B"]
    rankings = defaultdict(list)
    rank_rows = []

    for ds in TABULAR_DATASETS:
        for r in RATIOS:
            method_aurocs = {}
            for method in methods:
                if method in ("OC-SVM", "IFOREST", "SLAN"):
                    baseline_dir = {
                        "OC-SVM": "ocsvm",
                        "IFOREST": "iforest",
                        "SLAN": "slan",
                    }[method]
                    result_dir = os.path.join(
                        REPO_ROOT, "results", "baselines_v2", baseline_dir,
                        ds, f"known_ratio={r}")
                elif method == "CREM":
                    result_dir = os.path.join(REPO_ROOT, "results", "crem_v2", ds,
                                              f"known_ratio={r}")
                else:
                    mode = "A" if method.endswith("A") else "B"
                    result_dir = os.path.join(
                        REPO_ROOT, "results", "dcrem", ds,
                        f"protocol_v2_mode{mode}_r{r}")

                runs = load_seeds(result_dir, SEEDS)
                if runs:
                    aurocs = [r["metrics"].get("AUROC") for r in runs
                              if r["metrics"].get("AUROC") is not None]
                    if aurocs:
                        method_aurocs[method] = np.mean(aurocs)

            if len(method_aurocs) != len(methods):
                continue
            sorted_m = sorted(method_aurocs.items(), key=lambda x: x[1], reverse=True)
            rank_row = {}
            for rank, (method, _) in enumerate(sorted_m, 1):
                rankings[method].append(rank)
                rank_row[method] = rank
            rank_rows.append(rank_row)

    # Table
    lines = [
        "## Friedman CD Rankings",
        "",
        "Average rank (lower = better) across 6 datasets x 3 ratios = 18 settings.",
        "",
        "| Method | Avg Rank | #Settings |",
        "|---|---|---|",
    ]
    for method in methods:
        if rankings[method]:
            avg_r = np.mean(rankings[method])
            lines.append(f"| {method} | {avg_r:.2f} | {len(rankings[method])} |")

    if len(rank_rows) >= 2:
        from scipy.stats import friedmanchisquare
        statistic, p_value = friedmanchisquare(
            *[[row[method] for row in rank_rows] for method in methods])
        lines.extend([
            "",
            f"Friedman chi-square = {statistic:.4f}, p = {p_value:.4g} "
            f"(N = {len(rank_rows)} settings, k = {len(methods)} methods).",
        ])

    table = "\n".join(lines)
    with open(os.path.join(output_dir, "Fig_CD_rankings.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Saved: Fig_CD_rankings.md")

    # Try to make a visual CD plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        avg_ranks = {}
        for m in methods:
            if rankings[m]:
                avg_ranks[m] = np.mean(rankings[m])

        sorted_methods = sorted(avg_ranks.items(), key=lambda x: x[1])
        names = [m for m, _ in sorted_methods]
        ranks = [r for _, r in sorted_methods]

        fig, ax = plt.subplots(figsize=(10, 3))
        y_pos = range(len(names))
        ax.barh(y_pos, ranks, color=plt.cm.RdYlGn_r((np.array(ranks) - min(ranks)) /
                (max(ranks) - min(ranks) + 0.01)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title("Friedman Ranking — Multi-label OSR Methods")
        ax.invert_yaxis()

        for i, (rank, name) in enumerate(zip(ranks, names)):
            ax.text(rank + 0.02, i, f"{rank:.2f}", va="center")

        fig.tight_layout()
        path = os.path.join(output_dir, "Fig_CD.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  CD plot skipped: {e}")

    return table


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Table_case: case study
# ═══════════════════════════════════════════════════════════════════════════════

def build_case_study(cache_data, output_dir):
    """Build threshold-free examples at the extremes of knownness score."""
    distances = np.array(cache_data["distances"])  # (N, q)
    osr_labels = np.array(cache_data["osr_labels"])
    osr_score = np.array(cache_data["osr_score"])

    # Lower top-K distance means more likely unknown.  Without a validation-
    # selected threshold these are examples, not "correct/false" decisions.
    unknown_idx = np.where(osr_labels == 1)[0]
    known_idx = np.where(osr_labels == 0)[0]

    unknown_examples = unknown_idx[np.argsort(osr_score[unknown_idx])][:5]
    hard_known_examples = known_idx[np.argsort(osr_score[known_idx])][:5]

    lines = [
        "## Table_case: Case Study",
        "",
        "### True-unknown examples with lowest knownness score",
        "",
        "| Sample | OSR Score | Min Distance | Top-3 Nearest Labels (distance) |",
        "|---|---|---|---|",
    ]
    for i, idx in enumerate(unknown_examples):
        score = osr_score[idx]
        dists = distances[idx]
        top3 = np.argsort(dists)[:3]
        top3_str = ", ".join(f"L{k}({dists[k]:.3f})" for k in top3)
        lines.append(f"| {i+1} | {score:.4f} | {dists.min():.4f} | {top3_str} |")

    lines.append("")
    lines.append("### Hard known examples with lowest knownness score")
    lines.append("")
    lines.append("| Sample | OSR Score | Min Distance | Top-3 Nearest Labels (distance) |")
    lines.append("|---|---|---|---|")
    for i, idx in enumerate(hard_known_examples):
        score = osr_score[idx]
        dists = distances[idx]
        top3 = np.argsort(dists)[:3]
        top3_str = ", ".join(f"L{k}({dists[k]:.3f})" for k in top3)
        lines.append(f"| {i+1} | {score:.4f} | {dists.min():.4f} | {top3_str} |")

    table = "\n".join(lines)
    with open(os.path.join(output_dir, "Table_case.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Saved: Table_case.md")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Table_Rk: R_k per-label statistics
# ═══════════════════════════════════════════════════════════════════════════════

def build_rk_table(caches, output_dir):
    """Build R_k statistics table from multiple runs."""
    lines = [
        "## Table_Rk: R_k Distribution Statistics",
        "",
        "R_k controls the 'known region' boundary for each label.",
        "Larger R_k → known samples must be farther from p_k → larger margin for unknown detection.",
        "",
        "| Dataset | #Labels | R_k Mean | R_k Std | R_k Min | R_k Max |",
        "|---|---|---|---|---|---|",
    ]

    for c in caches:
        name = c["name"]
        data = c["data"]
        R_k = np.array(data.get("R_k", []))
        if len(R_k) == 0:
            continue
        lines.append(f"| {name} | {len(R_k)} | {R_k.mean():.4f} | {R_k.std():.4f} "
                     f"| {R_k.min():.4f} | {R_k.max():.4f} |")

    table = "\n".join(lines)
    with open(os.path.join(output_dir, "Table_Rk.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Saved: Table_Rk.md")


# ═══════════════════════════════════════════════════════════════════════════════
# Main: orchestrate everything
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures and tables")
    parser.add_argument("--all", action="store_true",
                        help="Run training captures + all plots/tables")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate plots from cached data (skip re-training)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip re-training, use existing caches if available")
    args = parser.parse_args()

    cache_dir = os.path.join(REPO_ROOT, "results", "analysis_cache_protocol_v2")
    os.makedirs(cache_dir, exist_ok=True)

    # ── Training runs for history capture ──
    training_specs = [
        # (dataset, encoder, mode, ratio, seed, feature)
        ("slashdot", "mlp", "B", 0.5, 0, None),
        ("voc2007", "identity", "B", 0.5, 0, "clip"),
        ("voc2007", "identity", "B", 0.5, 0, "resnet50"),
    ]

    caches = []
    for ds, enc, mode, r, s, feat in training_specs:
        ftag = f"_{feat}" if feat else ""
        cache_path = os.path.join(cache_dir,
            f"{ds}{ftag}_{enc}_mode{mode}_r{r}_s{s}.json")

        if os.path.exists(cache_path) and (args.plots_only or args.skip_training):
            print(f"Loading cached: {cache_path}")
            with open(cache_path) as f:
                data = json.load(f)
        elif args.plots_only:
            print(f"No cache for {ds} {enc}, skipping (--plots-only)")
            continue
        else:
            print(f"\n{'='*60}")
            print(f"Training: {ds} {enc} mode={mode} r={r} s={s}")
            print("=" * 60)
            data = run_training_with_history(ds, enc, mode, r, s,
                                             feature=feat, epochs=50)
        ftag_name = f"_{feat}" if feat else ""
        caches.append({"name": f"{ds}{ftag_name}_{enc}_mode{mode}_r{r}", "data": data})

    if not caches:
        print("No cached data available. Run without --plots-only first.")
        return

    # ── Generate plots ──
    print(f"\n{'='*60}")
    print("Generating figures and tables...")
    print("=" * 60)

    print("\n[Fig_convergence] Loss curves")
    plot_convergence(caches, FIGS_DIR)

    print("\n[Fig_coupling] ||W-P||_F evolution")
    plot_coupling(caches[:2], FIGS_DIR)  # tabular + clip image

    print("\n[Fig_R_dist] R_k histogram")
    plot_R_distribution(caches, FIGS_DIR)

    print("\n[Fig_tsne] t-SNE visualization")
    for c in caches:
        name_suffix = f" ({c['name']})"
        plot_tsne(c["data"], FIGS_DIR, name_suffix)

    print("\n[Fig_CD] Friedman CD diagram")
    build_cd_diagram(TABLES_DIR)

    print("\n[Table_case] Case study")
    # Use slashdot cache for case study
    slashdot_cache = next((c for c in caches if "slashdot" in c["name"]), caches[0])
    build_case_study(slashdot_cache["data"], TABLES_DIR)

    print("\n[Table_Rk] R_k statistics")
    build_rk_table(caches, TABLES_DIR)

    print(f"\n{'='*60}")
    print(f"All done! Figures: {FIGS_DIR}/  Tables: {TABLES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

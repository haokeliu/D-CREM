#!/usr/bin/env python3
"""Phase 3: D-CREM main experiments, baselines, ablation, and analysis.

Usage:
  # Main experiments — D-CREM on all tabular datasets
  python scripts/run_phase3.py main --mode B --ratios 0.3,0.5,0.7 --seeds 0-9

  # Single dataset quick test
  python scripts/run_phase3.py main --dataset enron --mode B --ratios 0.5 --seeds 0-2

  # Extreme protocol
  python scripts/run_phase3.py main --mode B --ratios 0.2 --seeds 0-9

  # Baselines (OC-SVM, IFOREST, SLAN, MUENL-F)
  python scripts/run_phase3.py baselines --ratios 0.3,0.5,0.7 --seeds 0-9

  # Ablation experiments
  python scripts/run_phase3.py ablation --mode A --datasets enron,slashdot,bibtex \
    --ratios 0.3,0.5,0.7 --ablations full,N1,E1,S1,U1 --seeds 0-9

  # Confirmatory Mode-B ablation
  python scripts/run_phase3.py ablation --mode B --datasets enron,slashdot,bibtex \
    --ratios 0.3,0.5,0.7 --ablations full,N1,E1,S1,U1 --seeds 0-9

  # Pre-registered Mode-B sensitivity matrix
  python scripts/run_phase3.py sensitivity --datasets enron,slashdot,bibtex \
    --ratios 0.3,0.5,0.7 --seeds 0-4

  # Image experiments (VOC2007)
  python scripts/run_phase3.py image --dataset voc2007 --feature clip --mode B --seeds 0-9

  # Refresh the structured result summary
  python scripts/run_phase3.py summarize

  # All-in-one (main + baselines + ablation + summarize)
  python scripts/run_phase3.py all --seeds 0-9
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DCREM_TRAIN = os.path.join(REPO_ROOT, "dcrem", "scripts", "train.py")
SLAN_RUN = os.path.join(REPO_ROOT, "scripts", "run_slan.py")
MUENL_F_RUN = os.path.join(REPO_ROOT, "scripts", "run_muenl_f.py")
RESULTS_DCREM = os.path.join(REPO_ROOT, "results", "dcrem")
RESULTS_BASELINES = os.path.join(REPO_ROOT, "results", "baselines_v2")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")

TABULAR_DATASETS = ["enron", "slashdot", "bibtex",
                    "yahoo-recreation", "yahoo-arts", "yahoo-education"]
IMAGE_DATASETS = ["voc2007", "coco2014"]
ALL_DATASETS = TABULAR_DATASETS + IMAGE_DATASETS

# ── Ablation configs ─────────────────────────────────────────────────────
# Paper-core D-CREM excludes mechanisms that did not show stable AUROC gains.
# The implementation remains available for reproducing superseded diagnostics.
PAPER_CORE_FLAGS = [
    "--no-correlation", "--lamda2", "0",
    "--alpha", "0", "--lamda3", "0",
    "--gamma-div", "0",
    "--no-warmup",
    "--classifier-induced-reciprocal",
    "--primary-score", "reciprocal",
]
PAPER_CORE_TAG = "paper_core"

# Each ablation overrides one component of the paper-core configuration.
ABLATION_CONFIGS = {
    "full": {
        "desc": "完整 paper-core 模型",
        "flags": [],
    },
    "N1": {  # No L2 normalization
        "desc": "去掉 L2 归一化",
        "flags": ["--no-l2norm"],
    },
    "E1": {  # Frozen encoder
        "desc": "冻结 backbone",
        "flags": ["--freeze-encoder"],
    },
    "S1": {  # Replace reciprocal distance with raw classifier logits
        "desc": "使用 classifier logits 替代 reciprocal distance 评分",
        "flags": ["--primary-score", "logit"],
    },
    "U1": {  # No uniformity loss
        "desc": "β=0 去掉 L_unif",
        "flags": ["--beta", "0"],
    },
}

ABLATION_DATASETS = ["enron", "slashdot", "bibtex"]

# One-factor-at-a-time Mode-B robustness matrix. REF is shared by all four
# axes so it is run once rather than duplicated for every sensitivity curve.
SENSITIVITY_CONFIGS = {
    "REF": {"factor": "reference", "value": None, "flags": []},
    "T1": {"factor": "block_interval", "value": 1,
           "flags": ["--block-interval", "1"]},
    "T5": {"factor": "block_interval", "value": 5,
           "flags": ["--block-interval", "5"]},
    "T25": {"factor": "block_interval", "value": 25,
            "flags": ["--block-interval", "25"]},
    "T50": {"factor": "block_interval", "value": 50,
            "flags": ["--block-interval", "50"]},
    "L01": {"factor": "lamda1", "value": 0.1,
            "flags": ["--lamda1", "0.1"]},
    "L03": {"factor": "lamda1", "value": 0.3,
            "flags": ["--lamda1", "0.3"]},
    "L3": {"factor": "lamda1", "value": 3.0,
           "flags": ["--lamda1", "3"]},
    "L10": {"factor": "lamda1", "value": 10.0,
            "flags": ["--lamda1", "10"]},
    "B0": {"factor": "beta", "value": 0.0,
           "flags": ["--beta", "0"]},
    "B003": {"factor": "beta", "value": 0.03,
             "flags": ["--beta", "0.03"]},
    "B03": {"factor": "beta", "value": 0.3,
            "flags": ["--beta", "0.3"]},
    "B1": {"factor": "beta", "value": 1.0,
           "flags": ["--beta", "1"]},
    "D64": {"factor": "embedding_dim", "value": 64,
            "flags": ["--embedding-dim", "64"]},
    "D256": {"factor": "embedding_dim", "value": 256,
             "flags": ["--embedding-dim", "256"]},
}

# Validation-only additive development ladder.  These configurations are
# diagnostics, not paper results, and are saved outside the formal result tree.
LITE_DEVELOPMENT_CONFIGS = {
    "F0": {
        "desc": "Legacy full objective",
        "flags": [],
    },
    "M1": {
        "desc": "Full objective with label-mean open loss",
        "flags": ["--open-reduction", "mean"],
    },
    "L1": {
        "desc": "Lite: coupling + radius-free mean open loss",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open",
        ],
    },
    "L2": {
        "desc": "Lite L1 + uniformity",
        "flags": [
            "--no-correlation", "--lamda2", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open",
        ],
    },
    "L3": {
        "desc": "Lite: coupling + learned-radius mean open loss",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
        ],
    },
    "G1": {
        "desc": "Grid: margin2 alpha.1 coupling1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.1", "--lamda3", "1",
        ],
    },
    "G2": {
        "desc": "Grid: margin3 alpha.1 coupling1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "3",
            "--alpha", "0.1", "--lamda3", "1",
        ],
    },
    "G3": {
        "desc": "Grid: margin2 alpha.05 coupling.1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.05", "--lamda3", "0.1",
        ],
    },
    "G4": {
        "desc": "Grid: margin3 alpha.05 coupling.1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "3",
            "--alpha", "0.05", "--lamda3", "0.1",
        ],
    },
    "G5": {
        "desc": "Grid: margin2 alpha.1 coupling.1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.1", "--lamda3", "0.1",
        ],
    },
    "G6": {
        "desc": "Grid: margin3 alpha.1 coupling.1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "3",
            "--alpha", "0.1", "--lamda3", "0.1",
        ],
    },
    "H1": {
        "desc": "Focused: margin2 alpha.05 coupling1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.05", "--lamda3", "1",
        ],
    },
    "H2": {
        "desc": "Focused: margin2 alpha.02 coupling1",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.02", "--lamda3", "1",
        ],
    },
    "H3": {
        "desc": "Focused: margin2 alpha.1 coupling10",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.1", "--lamda3", "10",
        ],
    },
    "H4": {
        "desc": "Focused: margin2 alpha.05 coupling10",
        "flags": [
            "--no-correlation", "--lamda2", "0", "--beta", "0",
            "--gamma-div", "0", "--open-reduction", "mean",
            "--radius-free-open", "--open-margin", "2",
            "--alpha", "0.05", "--lamda3", "10",
        ],
    },
    "V31": {
        "desc": "v3 tangent residual rho.5 + episodic pseudo-unknown",
        "flags": [
            "--v3-objective", "--residual-scale", "0.5",
            "--alpha", "0.1", "--pseudo-weight", "0.1",
            "--open-margin", "2", "--holdout-fraction", "0.2",
        ],
    },
    "V32": {
        "desc": "v3 tangent residual rho1 + episodic pseudo-unknown",
        "flags": [
            "--v3-objective", "--residual-scale", "1.0",
            "--alpha", "0.1", "--pseudo-weight", "0.1",
            "--open-margin", "2", "--holdout-fraction", "0.2",
        ],
    },
    "V33": {
        "desc": "v3 rho.5 with weaker reciprocal and stronger pseudo loss",
        "flags": [
            "--v3-objective", "--residual-scale", "0.5",
            "--alpha", "0.05", "--pseudo-weight", "0.2",
            "--open-margin", "2", "--holdout-fraction", "0.2",
        ],
    },
    "V34": {
        "desc": "v3 balanced pseudo-only rho.5",
        "flags": [
            "--v3-objective", "--residual-scale", "0.5",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--holdout-fraction", "0.2", "--pseudo-target-fraction", "0.3",
        ],
    },
    "V35": {
        "desc": "v3 balanced pseudo-only rho1",
        "flags": [
            "--v3-objective", "--residual-scale", "1.0",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--holdout-fraction", "0.2", "--pseudo-target-fraction", "0.3",
        ],
    },
    "V36": {
        "desc": "v3 balanced pseudo-only rho.5 stronger ranking",
        "flags": [
            "--v3-objective", "--residual-scale", "0.5",
            "--alpha", "0", "--pseudo-weight", "1.0",
            "--holdout-fraction", "0.2", "--pseudo-target-fraction", "0.3",
        ],
    },
    "V37": {
        "desc": "v3 balanced pseudo-only rho.5 wide holdout target.3",
        "flags": [
            "--v3-objective", "--residual-scale", "0.5",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.3",
        ],
    },
    "V38": {
        "desc": "v3 balanced pseudo-only rho.5 wide holdout target.5",
        "flags": [
            "--v3-objective", "--residual-scale", "0.5",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.5",
        ],
    },
    "B0": {
        "desc": "Pseudo gate baseline: classifier geometry P=W",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B0",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
        ],
    },
    "B1": {
        "desc": "Pseudo gate: free P + classifier coupling, no pseudo loss",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--lamda3", "10",
        ],
    },
    "B2": {
        "desc": "Pseudo gate: B1 + pairwise pseudo-unknown ranking",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B2",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--pseudo-margin", "0.1", "--lamda3", "10",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.5",
        ],
    },
    "B3": {
        "desc": "Pseudo gate: B2 + visible-label-only episodic classification",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B3",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--pseudo-margin", "0.1", "--lamda3", "10",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.5",
        ],
    },
    "C1": {
        "desc": "Pseudo weight grid: pairwise ranking .1 / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B2",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--pseudo-margin", "0.1", "--lamda3", "1",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.5",
        ],
    },
    "C2": {
        "desc": "Pseudo weight grid: pairwise ranking .5 / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B2",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0.5",
            "--pseudo-margin", "0.1", "--lamda3", "1",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.5",
        ],
    },
    "C3": {
        "desc": "Pseudo weight grid: pairwise ranking .1 / coupling .1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B2",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0.1",
            "--pseudo-margin", "0.1", "--lamda3", "0.1",
            "--holdout-fraction", "0.8", "--pseudo-target-fraction", "0.5",
        ],
    },
    "D1": {
        "desc": "Label-conditional reciprocal ranking .1 / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--lamda3", "1",
        ],
    },
    "D2": {
        "desc": "Label-conditional reciprocal ranking .5 / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.5", "--label-rank-margin", "0.1",
            "--lamda3", "1",
        ],
    },
    "D3": {
        "desc": "Label-conditional reciprocal ranking .1 / coupling .1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--lamda3", "0.1",
        ],
    },
    "R1": {
        "desc": "Frozen-W hard negatives: top 25%, ranking .1 / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.25", "--lamda3", "1",
        ],
    },
    "R2": {
        "desc": "Frozen-W hard negatives: top 10%, ranking .1 / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.1", "--lamda3", "1",
        ],
    },
    "R3": {
        "desc": "Frozen-W hard negatives: top 25%, ranking .1 / coupling .1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.25", "--lamda3", "0.1",
        ],
    },
    "Q0": {
        "desc": "One reciprocal prototype initialized from hard negatives",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--hard-prototype-init", "--reciprocal-prototypes", "1",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.25", "--lamda3", "0.1",
        ],
    },
    "Q1": {
        "desc": "Two hard-negative reciprocal prototypes / coupling .1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--hard-prototype-init", "--reciprocal-prototypes", "2",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.25", "--lamda3", "0.1",
        ],
    },
    "Q2": {
        "desc": "Three hard-negative reciprocal prototypes / coupling .1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--hard-prototype-init", "--reciprocal-prototypes", "3",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.25", "--lamda3", "0.1",
        ],
    },
    "Q3": {
        "desc": "Two hard-negative reciprocal prototypes / coupling 1",
        "flags": [
            "--v3-objective", "--pseudo-variant", "B1",
            "--development-primary-score", "reciprocal",
            "--hard-prototype-init", "--reciprocal-prototypes", "2",
            "--alpha", "0", "--pseudo-weight", "0",
            "--label-rank-weight", "0.1", "--label-rank-margin", "0.1",
            "--label-rank-hard-fraction", "0.25", "--lamda3", "1",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def parse_seeds(seed_str):
    if "-" in seed_str:
        lo, hi = seed_str.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in seed_str.split(",")]


def parse_ratios(ratio_str):
    return [float(r) for r in ratio_str.split(",")]


def parse_datasets(dataset_str):
    """Parse dataset argument: 'all' → TABULAR_DATASETS, 'a,b,c' → ['a','b','c']."""
    if dataset_str == "all":
        return list(TABULAR_DATASETS)
    return [d.strip() for d in dataset_str.split(",")]


def run_cmd(cmd, desc=""):
    """Run a command, print output on failure."""
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s): {' '.join(cmd)}")
        print(f"  STDERR: {result.stderr[-500:]}")
        return False
    return True


def dcrem_exists(dataset, mode, ratio, seed, extra_dir=None):
    """Check if a D-CREM result already exists."""
    if extra_dir:
        run_dir = os.path.join(RESULTS_DCREM, dataset, extra_dir,
                               f"seed{seed}.json")
    else:
        run_dir = os.path.join(RESULTS_DCREM, dataset,
                               f"protocol_v2_mode{mode}_r{ratio}",
                               f"seed{seed}.json")
    return os.path.exists(run_dir)


def load_dcrem_result(dataset, mode, ratio, seed, extra_dir=None):
    """Load a saved D-CREM result JSON."""
    if extra_dir:
        path = os.path.join(RESULTS_DCREM, dataset, extra_dir,
                           f"seed{seed}.json")
    else:
        path = os.path.join(RESULTS_DCREM, dataset,
                           f"protocol_v2_mode{mode}_r{ratio}",
                           f"seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# D-CREM runner
# ═══════════════════════════════════════════════════════════════════════════

def run_dcrem_single(dataset, mode, ratio, seed, extra_flags=None,
                     ablation_id=None, run_tag=None, feature_type=None,
                     epochs=100, batch_size=128,
                     python_bin=None, development_only=False,
                     defer_summary=False):
    """Run a single D-CREM experiment. Returns True on success."""
    py = python_bin or sys.executable

    # Determine encoder
    if dataset in IMAGE_DATASETS:
        # ImageDataset supplies pre-extracted 2-D features, not image tensors.
        encoder = "identity"
    else:
        encoder = "mlp"

    cmd = [
        py, DCREM_TRAIN,
        "--dataset", dataset,
        "--encoder", encoder,
        "--mode", mode,
        "--known-ratio", str(ratio),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if feature_type:
        cmd.extend(["--feature", feature_type])

    if extra_flags:
        cmd.extend(extra_flags)
    if development_only:
        cmd.append("--development-only")
    if defer_summary:
        cmd.append("--skip-summary-refresh")

    # Determine output directory for existence check
    extra_dir = None
    effective_run_tag = run_tag
    if ablation_id:
        effective_run_tag = f"ablation_mode{mode}_core_{ablation_id}"
    if effective_run_tag:
        cmd.extend(["--run-tag", effective_run_tag])
        extra_dir = f"protocol_v2_{effective_run_tag}_r{ratio}"

    if development_only:
        development_path = os.path.join(
            REPO_ROOT, "results", "analysis_cache_protocol_v2",
            "development", dataset, extra_dir, f"seed{seed}.json")
        if os.path.exists(development_path):
            return True
    elif dcrem_exists(dataset, mode, ratio, seed, extra_dir):
        return True  # skip, already done

    desc = f"{dataset} mode={mode} r={ratio} seed={seed}"
    if ablation_id:
        desc += f" [{ablation_id}]"
    print(f"  [{desc}]", end=" ", flush=True)

    success = run_cmd(cmd, desc)
    if success:
        print("OK")
    return success


def run_dcrem_batch(datasets, mode, ratios, seeds, extra_flags=None,
                    ablation_id=None, run_tag=None, epochs=100, batch_size=128,
                    python_bin=None):
    """Run D-CREM over a batch of configurations."""
    total = len(datasets) * len(ratios) * len(seeds)
    done = 0
    t_start = time.time()

    for ds in datasets:
        for r in ratios:
            for seed in seeds:
                success = run_dcrem_single(
                    ds, mode, r, seed,
                    extra_flags=extra_flags,
                    ablation_id=ablation_id,
                    run_tag=run_tag,
                    epochs=epochs, batch_size=batch_size,
                    python_bin=python_bin,
                )
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = done / elapsed * 60
                    remaining = (total - done) / max(rate, 0.01)
                    print(f"  [{done}/{total}] {elapsed/60:.1f}min elapsed, "
                          f"~{remaining:.1f}min remaining")

    elapsed = time.time() - t_start
    print(f"\nDone: {done} runs in {elapsed/60:.1f} min.")


# ═══════════════════════════════════════════════════════════════════════════
# Baseline runners (OC-SVM, IFOREST, SLAN, MUENL-F)
# ═══════════════════════════════════════════════════════════════════════════

def run_sklearn_baseline(dataset, ratio, seed, method_name, model_factory):
    """Run an sklearn baseline (OC-SVM or IFOREST) and save results."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from crem.data import get_dataset

    save_dir = os.path.join(RESULTS_BASELINES, method_name, dataset,
                            f"known_ratio={ratio}")
    save_path = os.path.join(save_dir, f"seed{seed}.json")
    if os.path.exists(save_path):
        return  # already done

    data = get_dataset(dataset, known_ratio=ratio, seed=seed, standardize=True)
    X_train = data["train_data"]       # (N_train, d)
    X_test = data["test_data"]          # (N_test, d)
    Y_train = data["train_target"]      # (Q, N_train, ±1)
    osr_labels = data["osr_labels"].ravel()
    q = Y_train.shape[0]

    # Multi-label → per-label anomaly detection → aggregate
    t0 = time.time()
    try:
        anomaly_scores = np.zeros((X_test.shape[0], q), dtype=np.float64)
        for k in range(q):
            pos_mask = Y_train[k, :] == 1
            if pos_mask.sum() < 2:
                anomaly_scores[:, k] = 0.0
                continue
            X_pos = X_train[pos_mask]
            model = model_factory(seed)
            model.fit(X_pos)
            anomaly_scores[:, k] = model.decision_function(X_test)

        # Aggregate: min normalized score → open-set score
        # Normalize per-label scores to [0,1] range before aggregation
        osr_scores = np.zeros(X_test.shape[0], dtype=np.float64)
        for i in range(X_test.shape[0]):
            osr_scores[i] = anomaly_scores[i, :].min()

        # Compute metrics
        auroc = float(roc_auc_score(osr_labels, -osr_scores))
        aupr = float(average_precision_score(osr_labels, -osr_scores))
        t_total = time.time() - t0

        os.makedirs(save_dir, exist_ok=True)
        record = {
            "metrics": {
                "AUROC": auroc,
                "AUPR": aupr,
                "macroAUC": None,
            },
            "time": {"total_s": round(t_total, 3)},
            "known_ratio": ratio,
            "seed": seed,
            "dataset": dataset,
            "method": method_name,
            "protocol": {
                "version": 2,
                "train_ratio": 0.4,
                "val_ratio": 0.1,
                "test_ratio": 0.5,
                "preprocessing_fit": "train_only",
                "standardized": True,
            },
            "timestamp": datetime.now().isoformat(),
        }
        with open(save_path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    except Exception as e:
        print(f"  ERROR {method_name} {dataset} r={ratio} s={seed}: {e}")
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"error": str(e), "seed": seed, "known_ratio": ratio},
                      f, indent=2)


def run_image_sklearn_baseline(dataset, feature, ratio, seed,
                               method_name, model_factory):
    """Run a fixed sklearn baseline on one frozen-feature image split."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from dcrem.data.image import get_image_protocol

    setting = f"feature={feature}_known_ratio={ratio}"
    save_dir = os.path.join(RESULTS_BASELINES, method_name, dataset, setting)
    save_path = os.path.join(save_dir, f"seed{seed}.json")
    if os.path.exists(save_path):
        return True
    protocol = get_image_protocol(
        dataset, feature, known_ratio=ratio, seed=seed,
        standardize=True, batch_size=128, train_ratio=0.4, val_ratio=0.1)
    X_train = protocol["split"]["train_data"]
    Y_train = protocol["split"]["train_target"]
    X_test = protocol["test_X"]
    labels = np.asarray(protocol["test_osr_labels"]).ravel()
    started = time.time()
    per_label = np.zeros((X_test.shape[0], Y_train.shape[0]), dtype=np.float64)
    for k in range(Y_train.shape[0]):
        positives = Y_train[k] == 1
        if positives.sum() < 2:
            continue
        model = model_factory(seed)
        model.fit(X_train[positives])
        per_label[:, k] = model.decision_function(X_test)
    known_score = per_label.min(axis=1)
    record = {
        "metrics": {
            "AUROC": float(roc_auc_score(labels, -known_score)),
            "AUPR": float(average_precision_score(labels, -known_score)),
            "macroAUC": None,
        },
        "dataset": dataset, "feature": feature, "method": method_name,
        "known_ratio": ratio, "seed": seed,
        "time": {"total_s": round(time.time() - started, 3)},
        "protocol": {
            "version": 2, "train_ratio": 0.4, "val_ratio": 0.1,
            "test_ratio": 0.5, "preprocessing_fit": "train_only",
            "standardized": True, "hyperparameters": "predeclared_fixed",
        },
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    return True


def run_slan_baseline(dataset, ratio, seed, tau_grid="0.8", python_bin=None):
    """Run SLAN in an isolated process so its O(n^2) matrices are released."""
    save_path = os.path.join(
        RESULTS_BASELINES, "slan", dataset, f"known_ratio={ratio}",
        f"seed{seed}.json")
    if os.path.exists(save_path):
        return True
    py = python_bin or sys.executable
    return run_cmd([
        py, SLAN_RUN, "--dataset", dataset, "--known-ratio", str(ratio),
        "--seed", str(seed), "--tau-grid", tau_grid,
    ])


def run_muenl_f_baseline(
        dataset, ratio, seed, radius_grid="1.0", python_bin=None):
    """Run MUENL-F in an isolated process to release each fitted forest."""
    save_path = os.path.join(
        RESULTS_BASELINES, "muenl_f", dataset, f"known_ratio={ratio}",
        f"seed{seed}.json")
    if os.path.exists(save_path):
        return True
    py = python_bin or sys.executable
    return run_cmd([
        py, MUENL_F_RUN, "--dataset", dataset, "--known-ratio", str(ratio),
        "--seed", str(seed), "--radius-grid", radius_grid,
    ])


def run_all_baselines(datasets, ratios, seeds,
                      methods=("ocsvm", "iforest", "slan", "muenl_f"),
                      tau_grid="0.8", radius_grid="1.0", python_bin=None):
    """Run selected Protocol-v2 baselines on tabular datasets."""
    methods = tuple(methods)
    allowed = {"ocsvm", "iforest", "slan", "muenl_f"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown baseline methods: {sorted(unknown)}")

    baselines = {}
    if "ocsvm" in methods:
        from sklearn.svm import OneClassSVM
        baselines["ocsvm"] = lambda seed: OneClassSVM(
            kernel="rbf", nu=0.1, gamma="scale")
    if "iforest" in methods:
        from sklearn.ensemble import IsolationForest
        baselines["iforest"] = lambda seed: IsolationForest(
            n_estimators=100, contamination=0.1, random_state=seed)

    total = len(datasets) * len(ratios) * len(seeds) * len(methods)
    done = 0

    for ds in datasets:
        for r in ratios:
            for seed in seeds:
                for name in methods:
                    desc = f"{name} {ds} r={r} s={seed}"
                    print(f"  [{done+1}/{total}] {desc}", end=" ", flush=True)
                    if name == "slan":
                        success = run_slan_baseline(
                            ds, r, seed, tau_grid=tau_grid,
                            python_bin=python_bin)
                    elif name == "muenl_f":
                        success = run_muenl_f_baseline(
                            ds, r, seed, radius_grid=radius_grid,
                            python_bin=python_bin)
                    else:
                        run_sklearn_baseline(
                            ds, r, seed, name, baselines[name])
                        success = True
                    print("OK" if success else "FAILED")
                    done += 1

    print(f"\nAll baselines done: {done} runs.")
    from scripts.build_results_report import write_report
    print(f"Result summary: {write_report()}")


# ═══════════════════════════════════════════════════════════════════════════
# CREM baseline runner (for any missing known_ratios)
# ═══════════════════════════════════════════════════════════════════════════

def run_crem_baseline(dataset, ratio, seed, python_bin=None):
    """Run CREM baseline if not already done."""
    run_dir = os.path.join(REPO_ROOT, "results", "crem_v2", dataset,
                           f"known_ratio={ratio}")
    save_path = os.path.join(run_dir, f"seed{seed}.json")
    if os.path.exists(save_path):
        return True

    py = python_bin or sys.executable
    crem_script = os.path.join(REPO_ROOT, "run_crem.py")
    cmd = [py, crem_script, "--dataset", dataset,
           "--known_ratio", str(ratio), "--seed", str(seed),
           "--quiet"]
    print(f"  CREM {dataset} r={ratio} s={seed}", end=" ", flush=True)
    success = run_cmd(cmd)
    if success:
        print("OK")
    return success


# ═══════════════════════════════════════════════════════════════════════════
# Image experiments
# ═══════════════════════════════════════════════════════════════════════════

def run_image_dcrem(dataset, feature_type, mode, ratios, seeds, python_bin=None):
    """Run D-CREM on image features."""
    py = python_bin or sys.executable

    encoder_map = {"pca": "identity", "resnet50": "identity", "clip": "identity"}

    for r in ratios:
        for seed in seeds:
            run_tag = f"{PAPER_CORE_TAG}_mode{mode}_{feature_type}_identity"
            extra_dir = f"protocol_v2_{run_tag}_r{r}"
            if dcrem_exists(dataset, mode, r, seed, extra_dir):
                print(f"  [skip] {dataset} {feature_type} mode={mode} r={r} s={seed}")
                continue

            cmd = [
                py, DCREM_TRAIN,
                "--dataset", dataset,
                "--feature", feature_type,
                "--encoder", encoder_map.get(feature_type, "resnet50"),
                "--run-tag", run_tag,
                "--mode", mode,
                "--known-ratio", str(r),
                "--seed", str(seed),
                "--epochs", "100",
            ] + PAPER_CORE_FLAGS
            print(f"  {dataset}/{feature_type} mode={mode} r={r} s={seed}",
                  end=" ", flush=True)
            if run_cmd(cmd):
                print("OK")

    from scripts.build_results_report import write_report
    print(f"Result summary: {write_report()}")


def run_image_e2e_voc2007(ratios, seeds, epochs=30, batch_size=32,
                          arms=("e2e", "frozen"), python_bin=None):
    """Run paired raw-image ResNet-50 paper-core experiments on VOC2007."""
    py = python_bin or sys.executable
    for arm in arms:
        if arm not in {"e2e", "frozen"}:
            raise ValueError(f"unknown image-e2e arm: {arm}")
        tag = f"paper_core_image_{arm}_resnet50_modeB"
        extra_flags = ["--freeze-backbone"] if arm == "frozen" else []
        for ratio in ratios:
            for seed in seeds:
                extra_dir = f"protocol_v2_{tag}_r{ratio}"
                if dcrem_exists("voc2007", "B", ratio, seed, extra_dir):
                    print(f"  [skip] voc2007 {arm} r={ratio} s={seed}")
                    continue
                cmd = [
                    py, DCREM_TRAIN,
                    "--dataset", "voc2007",
                    "--raw-images",
                    "--encoder", "resnet50",
                    "--embedding-dim", "128",
                    "--run-tag", tag,
                    "--mode", "B",
                    "--known-ratio", str(ratio),
                    "--seed", str(seed),
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--backbone-lr", "1e-5",
                    "--block-interval", "10",
                    "--skip-summary-refresh",
                ] + PAPER_CORE_FLAGS + extra_flags
                print(f"  voc2007/raw {arm} r={ratio} s={seed}",
                      end=" ", flush=True)
                if run_cmd(cmd):
                    print("OK")
    from scripts.build_results_report import write_report
    print(f"Result summary: {write_report()}")
    from scripts.analyze_protocol_v2_supplements import (
        build_voc_e2e, voc_e2e_complete)
    if voc_e2e_complete():
        print(f"Image paired summary: {build_voc_e2e()}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary table generation
# ═══════════════════════════════════════════════════════════════════════════

def collect_metrics(result_dir, dataset, setting_prefix, seeds):
    """Collect metrics across seeds for one setting."""
    metrics_list = []
    for seed in seeds:
        path = os.path.join(result_dir, dataset, setting_prefix,
                           f"seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            if "metrics" in rec:
                metrics_list.append(rec["metrics"])
    if not metrics_list:
        return None

    keys = ["AUROC", "AUPR", "macroAUC", "AveragePrecision"]
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and m[k] is not None
                and not (isinstance(m[k], float) and np.isnan(m[k]))]
        if vals:
            result[k] = (np.mean(vals), np.std(vals))
        else:
            result[k] = None
    result["n"] = len(metrics_list)
    return result


def build_main_table(datasets, ratios, method_dirs, seeds, table_name):
    """Build a markdown table for main results."""
    lines = [
        f"## {table_name}",
        "",
        "| Dataset | Ratio | " + " | ".join(method_dirs.keys()) + " |",
        "|" + "---|" * (len(method_dirs) + 2) + "|",
    ]

    for ds in datasets:
        for r in ratios:
            cells = []
            for method, result_base in method_dirs.items():
                result_dir, setting_fmt = result_base
                setting = setting_fmt.format(dataset=ds, ratio=r)
                metrics = collect_metrics(result_dir, ds, setting, seeds)
                if metrics and metrics.get("AUROC"):
                    m, s = metrics["AUROC"]
                    cells.append(f"{m:.4f}±{s:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {ds} | {r} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_ablation_table(datasets, ratio, seeds):
    """Build ablation summary table."""
    base_dir = os.path.join(RESULTS_DCREM, ABLATION_DATASETS[0])
    base_setting = f"protocol_v2_modeB_r{ratio}"

    # Collect base result
    base_metrics = {}
    base_path = os.path.join(RESULTS_DCREM, ABLATION_DATASETS[0],
                             base_setting)
    for seed in seeds:
        path = os.path.join(base_path, f"seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            if "metrics" in rec:
                for k, v in rec["metrics"].items():
                    base_metrics.setdefault(k, []).append(v)

    base_means = {}
    for k, vals in base_metrics.items():
        vals_clean = [v for v in vals if v is not None
                      and not (isinstance(v, float) and np.isnan(v))]
        if vals_clean:
            base_means[k] = (np.mean(vals_clean), np.std(vals_clean))

    lines = [
        "## Table_ablation: Ablation study",
        "",
        f"Dataset: {', '.join(datasets)}, known_ratio={ratio}",
        "",
        "| Variant | AUROC | AUPR | macroAUC | Δ AUROC vs Base |",
        "|---|---|---|---|---|",
    ]

    # Base row
    if base_means.get("AUROC"):
        lines.append(
            f"| Base (D-CREM mode B) | "
            f"{base_means['AUROC'][0]:.4f}±{base_means['AUROC'][1]:.4f} | "
            f"{base_means.get('AUPR', ('—',0))[0]:.4f}±{base_means.get('AUPR', ('—',0))[1]:.4f} | "
            f"{base_means.get('macroAUC', ('—',0))[0]:.4f}±{base_means.get('macroAUC', ('—',0))[1]:.4f} | "
            f"— |")

    for ab_id, ab_cfg in ABLATION_CONFIGS.items():
        ab_setting = f"protocol_v2_ablation_{ab_id}_r{ratio}"
        metrics_list = []
        for seed in seeds:
            path = os.path.join(RESULTS_DCREM, ABLATION_DATASETS[0],
                               ab_setting, f"seed{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    rec = json.load(f)
                if "metrics" in rec:
                    metrics_list.append(rec["metrics"])

        if not metrics_list:
            lines.append(f"| {ab_id}: {ab_cfg['desc']} | — | — | — | — |")
            continue

        means = {}
        for k in ["AUROC", "AUPR", "macroAUC"]:
            vals = [m[k] for m in metrics_list if k in m and m[k] is not None
                    and not (isinstance(m[k], float) and np.isnan(m[k]))]
            if vals:
                means[k] = (np.mean(vals), np.std(vals))
            else:
                means[k] = None

        delta = ""
        if means.get("AUROC") and base_means.get("AUROC"):
            d = means["AUROC"][0] - base_means["AUROC"][0]
            delta = f"{d:+.4f}"

        lines.append(
            f"| {ab_id}: {ab_cfg['desc']} | "
            f"{means['AUROC'][0]:.4f}±{means['AUROC'][1]:.4f} | "
            f"{means['AUPR'][0]:.4f}±{means['AUPR'][1]:.4f} | "
            f"{means['macroAUC'][0]:.4f}±{means['macroAUC'][1]:.4f} | "
            f"{delta} |")

    return "\n".join(lines)


def summarize_phase3(ratios, seeds):
    """Refresh the machine-readable Protocol-v2 result summary."""
    from scripts.build_results_report import write_report

    path = write_report()
    print(f"Summary: {path}")
    return [path]


# ═══════════════════════════════════════════════════════════════════════════
# Analysis experiments (figures + tables)
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(dataset, mode, ratio, seed, python_bin=None):
    """Run analysis experiments for a trained model:
    - Loss convergence history
    - ||W-P||_F evolution
    - R_k distribution
    - t-SNE visualization
    """
    py = python_bin or sys.executable
    analysis_script = os.path.join(REPO_ROOT, "scripts", "run_analysis.py")

    # run_analysis.py owns a fixed, paper-facing capture matrix.  Keep the
    # historical arguments in this wrapper for CLI compatibility, but do not
    # forward unsupported options to it.
    cmd = [py, analysis_script, "--all"]
    print("  Analysis: protocol-v2 standard capture matrix")
    subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: D-CREM main experiments, baselines & ablation")

    sub = parser.add_subparsers(dest="command", help="Experiment type")

    # ── main: D-CREM main experiments ──
    pm = sub.add_parser("main", help="Run D-CREM main experiments (tabular)")
    pm.add_argument("--dataset", type=str, default="all",
                    help="Dataset (default: all tabular)")
    pm.add_argument("--mode", type=str, default="B", choices=["A", "B"])
    pm.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pm.add_argument("--seeds", type=str, default="0-9")
    pm.add_argument("--epochs", type=int, default=50)
    pm.add_argument("--batch-size", type=int, default=128)
    pm.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── baselines: sklearn baselines ──
    pb = sub.add_parser(
        "baselines", help="Run baselines (OC-SVM, IFOREST, SLAN, MUENL-F)")
    pb.add_argument("--dataset", type=str, default="all")
    pb.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pb.add_argument("--seeds", type=str, default="0-9")
    pb.add_argument("--methods", type=str,
                    default="ocsvm,iforest,slan,muenl_f",
                    help="Comma-separated baseline names")
    pb.add_argument("--tau-grid", type=str, default="0.8",
                    help="SLAN validation candidates, e.g. 0.6,0.7,0.8,0.9")
    pb.add_argument(
        "--radius-grid", type=str, default="1.0",
        help="MUENL-F validation candidates, e.g. 0.75,1.0,1.25")
    pb.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── ablation ──
    pa = sub.add_parser("ablation", help="Run ablation experiments")
    pa.add_argument("--ablations", type=str,
                    default="full,N1,E1,S1,U1",
                    help="Comma-separated ablation IDs")
    pa.add_argument("--mode", type=str, default="A", choices=["A", "B"])
    pa.add_argument("--datasets", type=str,
                    default="enron,slashdot,bibtex")
    pa.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pa.add_argument("--seeds", type=str, default="0-9")
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--python", type=str, default=None, dest="python_bin")

    psens = sub.add_parser(
        "sensitivity", help="Run the pre-registered Mode-B robustness matrix")
    psens.add_argument("--configs", type=str,
                       default=",".join(SENSITIVITY_CONFIGS),
                       help="Comma-separated sensitivity configuration IDs")
    psens.add_argument("--datasets", type=str,
                       default="enron,slashdot,bibtex")
    psens.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    psens.add_argument("--seeds", type=str, default="0-4")
    psens.add_argument("--epochs", type=int, default=50)
    psens.add_argument("--batch-size", type=int, default=128)
    psens.add_argument("--python", type=str, default=None, dest="python_bin")

    pd = sub.add_parser(
        "develop-lite",
        help="Run validation-only additive candidate screening")
    pd.add_argument("--datasets", type=str,
                    default="enron,bibtex,yahoo-education")
    pd.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pd.add_argument("--seeds", type=str, default="0-4")
    pd.add_argument("--candidates", type=str, default="F0,M1,L1,L2,L3")
    pd.add_argument("--epochs", type=int, default=50)
    pd.add_argument("--batch-size", type=int, default=128)
    pd.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── image ──
    pi = sub.add_parser("image", help="Run D-CREM on image datasets")
    pi.add_argument("--dataset", type=str, default="voc2007",
                    choices=["voc2007", "coco2014"])
    pi.add_argument("--feature", type=str, default="clip",
                    choices=["pca", "resnet50", "clip"])
    pi.add_argument("--mode", type=str, default="B", choices=["A", "B"])
    pi.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pi.add_argument("--seeds", type=str, default="0-9")
    pi.add_argument("--python", type=str, default=None, dest="python_bin")

    pie = sub.add_parser(
        "image-e2e", help="Run paired raw-image VOC2007 ResNet-50 experiments")
    pie.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pie.add_argument("--seeds", type=str, default="0-4")
    pie.add_argument("--arms", type=str, default="e2e,frozen")
    pie.add_argument("--epochs", type=int, default=30)
    pie.add_argument("--batch-size", type=int, default=32)
    pie.add_argument("--python", type=str, default=None, dest="python_bin")

    pic = sub.add_parser("image-controls", help="Run VOC frozen-feature controls")
    pic.add_argument("--dataset", type=str, default="voc2007",
                     choices=["voc2007"])
    pic.add_argument("--features", type=str, default="clip,resnet50")
    pic.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    pic.add_argument("--seeds", type=str, default="0-9")
    pic.add_argument("--epochs", type=int, default=50)
    pic.add_argument("--batch-size", type=int, default=128)
    pic.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── summarize ──
    ps = sub.add_parser("summarize", help="Refresh structured Protocol-v2 results")
    ps.add_argument("--ratios", type=str, default="0.3,0.5,0.7")
    ps.add_argument("--seeds", type=str, default="0-9")

    # ── analysis ──
    pn = sub.add_parser("analysis", help="Run analysis experiments")
    pn.add_argument("--dataset", type=str, default="enron")
    pn.add_argument("--mode", type=str, default="B")
    pn.add_argument("--ratio", type=float, default=0.5)
    pn.add_argument("--seed", type=int, default=0)
    pn.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── all: main + baselines + ablation + summarize ──
    pall = sub.add_parser("all", help="Run everything (main + baselines + ablation + summarize)")
    pall.add_argument("--seeds", type=str, default="0-9")
    pall.add_argument("--python", type=str, default=None, dest="python_bin")
    pall.add_argument("--skip-baselines", action="store_true")
    pall.add_argument("--skip-ablation", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    seeds = parse_seeds(getattr(args, "seeds", "0-9"))
    py = getattr(args, "python_bin", None)

    if args.command == "main":
        ratios = parse_ratios(args.ratios)
        datasets = parse_datasets(args.dataset)
        print(f"Phase 3 — Main D-CREM experiments")
        print(f"  Datasets: {datasets}")
        print(f"  Mode: {args.mode}")
        print(f"  Ratios: {ratios}")
        print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
        print(f"  Total: {len(datasets) * len(ratios) * len(seeds)} runs")
        print()
        run_dcrem_batch(datasets, args.mode, ratios, seeds,
                        extra_flags=PAPER_CORE_FLAGS,
                        run_tag=f"{PAPER_CORE_TAG}_mode{args.mode}",
                        epochs=args.epochs, batch_size=args.batch_size,
                        python_bin=py)

    elif args.command == "develop-lite":
        ratios = parse_ratios(args.ratios)
        datasets = parse_datasets(args.datasets)
        candidates = [value.strip() for value in args.candidates.split(",")]
        unknown = set(candidates) - set(LITE_DEVELOPMENT_CONFIGS)
        if unknown:
            raise ValueError(f"Unknown Lite development candidates: {sorted(unknown)}")
        total = len(candidates) * len(datasets) * len(ratios) * len(seeds)
        print("Phase 3 — validation-only Lite development")
        print(f"  Candidates: {candidates}")
        print(f"  Datasets: {datasets}; ratios: {ratios}; seeds: {seeds}")
        print(f"  Total: {total} development runs; test folds are not scored")
        done = 0
        started = time.time()
        for candidate in candidates:
            cfg = LITE_DEVELOPMENT_CONFIGS[candidate]
            print(f"\n--- {candidate}: {cfg['desc']} ---")
            for ds in datasets:
                for ratio in ratios:
                    for seed in seeds:
                        ok = run_dcrem_single(
                            ds, "A", ratio, seed,
                            extra_flags=cfg["flags"],
                            run_tag=f"development_lite_{candidate}",
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            python_bin=py,
                            development_only=True,
                        )
                        if not ok:
                            raise RuntimeError(
                                f"Development run failed: {candidate} {ds} "
                                f"r={ratio} seed={seed}")
                        done += 1
                        if done % 10 == 0:
                            elapsed = time.time() - started
                            eta = (total - done) / max(done / max(elapsed, 1e-9), 1e-9)
                            print(f"  [{done}/{total}] ETA {eta/60:.1f}min")

    elif args.command == "baselines":
        ratios = parse_ratios(args.ratios)
        datasets = parse_datasets(args.dataset)
        methods = [value.strip() for value in args.methods.split(",")]
        print(f"Phase 3 — Baselines")
        print(f"  Datasets: {datasets}")
        print(f"  Methods: {methods}")
        print(f"  Ratios: {ratios}")
        print(f"  Seeds: {seeds[0]}..{seeds[-1]}")
        print()
        run_all_baselines(
            datasets, ratios, seeds, methods=methods,
            tau_grid=args.tau_grid, radius_grid=args.radius_grid,
            python_bin=py)

    elif args.command == "ablation":
        ab_ids = [a.strip() for a in args.ablations.split(",")]
        unknown = set(ab_ids) - set(ABLATION_CONFIGS)
        if unknown:
            raise ValueError(f"Unknown ablations: {sorted(unknown)}")
        ratios = parse_ratios(args.ratios)
        datasets = parse_datasets(args.datasets)
        print(f"Phase 3 — Mode-{args.mode} ablation experiments")
        print(f"  Ablations: {ab_ids}")
        print(f"  Datasets: {datasets}")
        print(f"  Ratios: {ratios}")
        print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
        print()

        invalid_datasets = set(datasets) - set(ABLATION_DATASETS)
        if invalid_datasets:
            raise ValueError(f"Invalid formal ablation datasets: {sorted(invalid_datasets)}")
        total = len(ab_ids) * len(datasets) * len(ratios) * len(seeds)
        done = 0
        t_start = time.time()

        for ab_id in ab_ids:
            ab_cfg = ABLATION_CONFIGS[ab_id]
            print(f"\n--- {ab_id}: {ab_cfg['desc']} ---")
            for ds in datasets:
                for ratio in ratios:
                    for seed in seeds:
                        ok = run_dcrem_single(
                            ds, args.mode, ratio, seed,
                            extra_flags=PAPER_CORE_FLAGS + ab_cfg["flags"],
                            ablation_id=ab_id,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            python_bin=py,
                            defer_summary=True,
                        )
                        if not ok:
                            raise RuntimeError(
                                f"Ablation run failed: mode={args.mode} {ab_id} "
                                f"{ds} r={ratio} seed={seed}")
                        done += 1
                        if done % 20 == 0:
                            elapsed = time.time() - t_start
                            rate = done / max(elapsed, 1e-9)
                            eta = (total - done) / max(rate, 1e-9)
                            print(f"  [{done}/{total}] {elapsed/60:.1f}min, "
                                  f"ETA {eta/60:.1f}min")

        elapsed = time.time() - t_start
        print(f"\nAblations done: {done} runs in {elapsed/60:.1f}min.")
        from scripts.build_results_report import write_report
        print(f"Result summary: {write_report()}")

    elif args.command == "sensitivity":
        config_ids = [value.strip() for value in args.configs.split(",")]
        unknown = set(config_ids) - set(SENSITIVITY_CONFIGS)
        if unknown:
            raise ValueError(f"Unknown sensitivity configs: {sorted(unknown)}")
        ratios = parse_ratios(args.ratios)
        datasets = parse_datasets(args.datasets)
        invalid_datasets = set(datasets) - set(ABLATION_DATASETS)
        if invalid_datasets:
            raise ValueError(
                f"Invalid sensitivity datasets: {sorted(invalid_datasets)}")
        total = len(config_ids) * len(datasets) * len(ratios) * len(seeds)
        print("Phase 3 — Mode-B sensitivity experiments")
        print(f"  Configs: {config_ids}")
        print(f"  Datasets: {datasets}")
        print(f"  Ratios: {ratios}")
        print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
        print(f"  Total: {total} runs")
        done = 0
        started = time.time()
        for config_id in config_ids:
            config = SENSITIVITY_CONFIGS[config_id]
            print(f"\n--- {config_id}: {config['factor']}={config['value']} ---")
            for dataset in datasets:
                for ratio in ratios:
                    for seed in seeds:
                        ok = run_dcrem_single(
                            dataset, "B", ratio, seed,
                            extra_flags=PAPER_CORE_FLAGS + config["flags"],
                            run_tag=f"sensitivity_modeB_core_{config_id}",
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            python_bin=py,
                            defer_summary=True,
                        )
                        if not ok:
                            raise RuntimeError(
                                f"Sensitivity run failed: {config_id} "
                                f"{dataset} r={ratio} seed={seed}")
                        done += 1
                        if done % 20 == 0:
                            elapsed = time.time() - started
                            eta = (total - done) / max(
                                done / max(elapsed, 1e-9), 1e-9)
                            print(f"  [{done}/{total}] {elapsed/60:.1f}min, "
                                  f"ETA {eta/60:.1f}min")
        elapsed = time.time() - started
        print(f"\nSensitivity done: {done} runs in {elapsed/60:.1f}min.")
        from scripts.build_results_report import write_report
        print(f"Result summary: {write_report()}")

    elif args.command == "image":
        ratios = parse_ratios(args.ratios)
        print(f"Phase 3 — Image experiments")
        print(f"  Dataset: {args.dataset}, Feature: {args.feature}")
        print(f"  Mode: {args.mode}, Ratios: {ratios}")
        print(f"  Seeds: {seeds[0]}..{seeds[-1]}")
        print()
        run_image_dcrem(args.dataset, args.feature, args.mode,
                        ratios, seeds, python_bin=py)

    elif args.command == "image-e2e":
        ratios = parse_ratios(args.ratios)
        arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
        print("Phase 3 — VOC2007 raw-image end-to-end experiment")
        print(f"  Arms: {arms}, Ratios: {ratios}, Seeds: {seeds}")
        run_image_e2e_voc2007(
            ratios, seeds, epochs=args.epochs, batch_size=args.batch_size,
            arms=arms, python_bin=py)

    elif args.command == "image-controls":
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM

        features = [value.strip() for value in args.features.split(",")]
        ratios = parse_ratios(args.ratios)
        factories = {
            "ocsvm": lambda seed: OneClassSVM(
                kernel="rbf", nu=0.1, gamma="scale"),
            "iforest": lambda seed: IsolationForest(
                n_estimators=100, contamination=0.1, random_state=seed),
        }
        total = len(features) * len(ratios) * len(seeds) * 2
        done = 0
        started = time.time()
        for feature in features:
            for ratio in ratios:
                for seed in seeds:
                    for method, factory in factories.items():
                        run_image_sklearn_baseline(
                            args.dataset, feature, ratio, seed, method, factory)
                        done += 1
                    if done % 15 == 0:
                        elapsed = time.time() - started
                        eta = (total - done) / max(done / max(elapsed, 1e-9), 1e-9)
                        print(f"  [{done}/{total}] ETA {eta/60:.1f}min")

    elif args.command == "summarize":
        ratios = parse_ratios(args.ratios)
        print("Phase 3 — Consolidated result refresh")
        paths = summarize_phase3(ratios, seeds)
        print(f"\nUpdated {len(paths)} document.")

    elif args.command == "analysis":
        run_analysis(args.dataset, args.mode, args.ratio, args.seed,
                     python_bin=py)

    elif args.command == "all":
        ratios = [0.3, 0.5, 0.7]
        print("=" * 60)
        print("Phase 3 — Full experiment suite")
        print("=" * 60)
        print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
        print()

        # 1. Main D-CREM
        print("[1/4] Main D-CREM experiments (Mode B)...")
        run_dcrem_batch(TABULAR_DATASETS, "B", ratios, seeds,
                        extra_flags=PAPER_CORE_FLAGS,
                        run_tag=f"{PAPER_CORE_TAG}_modeB",
                        epochs=50, python_bin=py)

        # 2. Baselines
        if not args.skip_baselines:
            print("\n[2/4] baselines...")
            run_all_baselines(
                TABULAR_DATASETS, ratios, seeds, python_bin=py)

        # 3. Ablation
        if not args.skip_ablation:
            print("\n[3/4] Ablation experiments...")
            for ab_id, ab_cfg in ABLATION_CONFIGS.items():
                print(f"\n  {ab_id}: {ab_cfg['desc']}")
                for ds in ABLATION_DATASETS:
                    for ratio in ratios:
                        for seed in seeds:
                            run_dcrem_single(
                                ds, "A", ratio, seed,
                                extra_flags=PAPER_CORE_FLAGS + ab_cfg["flags"],
                                ablation_id=ab_id,
                                epochs=50, batch_size=128, python_bin=py,
                            )

        # 4. Summarize
        print("\n[4/4] Refreshing consolidated results...")
        paths = summarize_phase3(ratios, seeds)
        print(f"\nUpdated {len(paths)} document.")
        for p in paths:
            print(f"  {p}")

        print("\nPhase 3 complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the official-paper SLAN baseline under CREM Protocol v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from baselines.slan import SLAN, SLANParams
from crem.data import get_dataset
from dcrem.eval.mll_metrics import compute_mll_metrics
from dcrem.eval.osr_metrics import compute_osr_metrics


def _tau_candidates(text: str):
    values = [float(value) for value in text.split(",")]
    if not values or any(value < 0 or value > 1 for value in values):
        raise argparse.ArgumentTypeError("tau grid must contain values in [0,1]")
    return values


def run(dataset: str, known_ratio: float, seed: int, tau_grid,
        force: bool = False, quick: bool = False):
    tau_grid = [float(value) for value in tau_grid]
    result_family = "smoke" if quick else "baselines_v2"
    save_dir = os.path.join(
        REPO_ROOT, "results", result_family, "slan", dataset,
        f"known_ratio={known_ratio}")
    save_path = os.path.join(save_dir, f"seed{seed}.json")
    if os.path.exists(save_path) and not force:
        print(f"Skip existing: {save_path}")
        return save_path

    split = get_dataset(
        dataset, known_ratio=known_ratio, seed=seed, standardize=True)
    params = SLANParams()
    if quick:
        params = SLANParams(
            outer_iterations=2, z_iterations=2, f_iterations=2,
            admm_iterations=2)
    model = SLAN(params)

    started = time.time()
    model.fit(split["train_data"], split["train_target"])
    train_seconds = time.time() - started

    select_started = time.time()
    selected_tau, validation = model.select_tau(
        split["val_data"], split["val_osr_labels"].ravel(), tau_grid)
    validation_seconds = time.time() - select_started

    test_started = time.time()
    outputs, known_scores, diagnostics = model.decision_function(
        split["test_data"], tau=selected_tau)
    test_seconds = time.time() - test_started
    metrics = {
        **compute_osr_metrics(known_scores, split["osr_labels"].ravel()),
        **compute_mll_metrics(outputs, split["test_target"]),
    }

    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    record = {
        "metrics": {
            key: finite_or_none(value)
            for key, value in metrics.items()
        },
        "config": params.to_dict(),
        "selection": {
            "selected_tau": selected_tau,
            "tau_grid": list(tau_grid),
            "validation_metrics": {
                str(tau): {name: finite_or_none(value)
                           for name, value in values.items()}
                for tau, values in validation.items()
            },
            "selection_fold": "validation",
        },
        "time": {
            "train_s": round(train_seconds, 3),
            "validation_s": round(validation_seconds, 3),
            "test_s": round(test_seconds, 3),
            "total_s": round(train_seconds + validation_seconds + test_seconds, 3),
        },
        "known_ratio": known_ratio,
        "seed": seed,
        "dataset": dataset,
        "method": "slan",
        "paper_eligible": not quick,
        "quick_smoke": quick,
        "protocol": {
            "version": 2,
            "train_ratio": 0.4,
            "val_ratio": 0.1,
            "test_ratio": 0.5,
            "preprocessing_fit": "train_only",
            "standardized": True,
            "osr_label_convention": "1=contains_unknown",
        },
        "diagnostics": {
            "iterations": int(len(model.loss_history_)),
            "final_loss": float(model.loss_history_[-1]),
            "usable_classes": int(np.sum(diagnostics["class_used"])),
        },
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(f"Saved: {save_path}")
    print(f"AUROC={metrics['AUROC']:.4f} AUPR={metrics['AUPR']:.4f} "
          f"tau={selected_tau:g} total={record['time']['total_s']:.1f}s")
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="enron")
    parser.add_argument("--known-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-grid", type=_tau_candidates, default=[0.8],
                        help="validation-only candidates, e.g. 0.6,0.7,0.8,0.9")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="two-iteration smoke test; not a paper result")
    args = parser.parse_args()
    run(args.dataset, args.known_ratio, args.seed, args.tau_grid,
        force=args.force, quick=args.quick)


if __name__ == "__main__":
    main()

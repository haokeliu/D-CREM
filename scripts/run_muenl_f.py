#!/usr/bin/env python3
"""Run the official-algorithm MuENL-F baseline under CREM Protocol v2."""

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

from baselines.muenl_f import MuENLF, MuENLFParams
from crem.data import get_dataset
from dcrem.eval.mll_metrics import compute_mll_metrics
from dcrem.eval.osr_metrics import compute_osr_metrics


def _radius_candidates(text: str):
    values = [float(value) for value in text.split(",")]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("radius grid must be nonnegative")
    return values


def run(dataset: str, known_ratio: float, seed: int, radius_grid,
        force: bool = False, quick: bool = False):
    radius_grid = [float(value) for value in radius_grid]
    result_family = "smoke" if quick else "baselines_v2"
    save_dir = os.path.join(
        REPO_ROOT, "results", result_family, "muenl_f", dataset,
        f"known_ratio={known_ratio}")
    save_path = os.path.join(save_dir, f"seed{seed}.json")
    if os.path.exists(save_path) and not force:
        print(f"Skip existing: {save_path}")
        return save_path

    split = get_dataset(
        dataset, known_ratio=known_ratio, seed=seed, standardize=True)
    params = MuENLFParams(random_state=seed)
    if quick:
        params = MuENLFParams(
            classifier_sweeps=1,
            classifier_inner_iterations=2,
            psi=min(32, split["train_data"].shape[0]),
            num_trees=3,
            max_height=3,
            split_retries=3,
            random_state=seed,
        )
    model = MuENLF(params)

    started = time.time()
    model.fit(split["train_data"], split["train_target"])
    train_seconds = time.time() - started

    select_started = time.time()
    selected_ratio, validation = model.select_radius_ratio(
        split["val_data"], split["val_osr_labels"].ravel(), radius_grid)
    validation_seconds = time.time() - select_started

    test_started = time.time()
    outputs, known_scores = model.decision_function(
        split["test_data"], radius_ratio=selected_ratio)
    test_seconds = time.time() - test_started
    metrics = {
        **compute_osr_metrics(known_scores, split["osr_labels"].ravel()),
        **compute_mll_metrics(outputs, split["test_target"]),
    }

    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    record = {
        "metrics": {key: finite_or_none(value) for key, value in metrics.items()},
        "config": params.to_dict(),
        "selection": {
            "selected_radius_ratio": selected_ratio,
            "radius_grid": radius_grid,
            "validation_metrics": {
                str(ratio): {
                    name: finite_or_none(value) for name, value in values.items()
                }
                for ratio, values in validation.items()
            },
            "selection_fold": "validation",
        },
        "time": {
            "train_s": round(train_seconds, 3),
            "validation_s": round(validation_seconds, 3),
            "test_s": round(test_seconds, 3),
            "total_s": round(
                train_seconds + validation_seconds + test_seconds, 3),
        },
        "known_ratio": known_ratio,
        "seed": seed,
        "dataset": dataset,
        "method": "muenl_f",
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
            "classifier_sweeps": len(model.classifier.loss_history_),
            "classifier_final_loss": finite_or_none(
                model.classifier.loss_history_[-1]),
            "num_trees": len(model.forest.trees_),
        },
        "implementation": {
            "source": "official MuENL MATLAB package",
            "scope": "static PLR + MuENLForest + MuENLDetect (MuENL-F)",
            "third_party_source_redistributed": False,
        },
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(f"Saved: {save_path}")
    print(
        f"AUROC={metrics['AUROC']:.4f} AUPR={metrics['AUPR']:.4f} "
        f"radius={selected_ratio:g} total={record['time']['total_s']:.1f}s")
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="enron")
    parser.add_argument("--known-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--radius-grid", type=_radius_candidates, default=[1.0],
        help="validation-only radius multipliers, e.g. 0.75,1.0,1.25")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="three-tree smoke test; stored outside paper-facing results")
    args = parser.parse_args()
    run(args.dataset, args.known_ratio, args.seed, args.radius_grid,
        force=args.force, quick=args.quick)


if __name__ == "__main__":
    main()

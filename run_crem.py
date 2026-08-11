#!/usr/bin/env python3
"""Unified CREM experiment entry point.

Usage:
  python run_crem.py --dataset enron --known_ratio 0.5 --seed 0
  python run_crem.py --dataset enron --known_ratio 0.5 --seeds 0-9
  python run_crem.py --dataset enron --known_ratio 0.5 --seeds 0,1,2
  python run_crem.py --dataset enron --known_ratio 0.5 --seed 0 --standardize
  python run_crem.py --dataset enron --known_ratio 0.3 --seed 0 --c-mode identity
  python run_crem.py --dataset enron --known_ratio 0.5 --seed 0 --param '{"gamma":0.10}'
  python run_crem.py --list
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crem import crem_train, crem_validate_and_test, kernelization
from crem.config import (
    ALL_DATASETS, get_params, get_dataset_specs,
)
from crem.data import build_all_caches, build_full_cache, get_dataset
from crem.label_correlation import get_C_matrix
from crem.logger import load_runs, make_setting_name, mean_std, save_run


def parse_seeds(seed_str):
    """Parse '0-9' or '0,1,2,3' into list of ints."""
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_single(dataset, known_ratio, seed, standardize=False,
               param_override=None, c_mode="full", method="crem_v2",
               output_dir=None, verbose=True, use_dataset_best=False,
               train_ratio=0.4, val_ratio=0.1):
    """Run a single CREM experiment and log results.

    Returns the path to the saved JSON.
    """
    np.random.seed(seed)
    # ── Params ──
    nominal, effective = get_params(
        dataset, param_override, use_dataset_best=use_dataset_best)

    # ── Data ──
    data = get_dataset(dataset, known_ratio=known_ratio, seed=seed,
                       standardize=standardize, train_ratio=train_ratio,
                       val_ratio=val_ratio)

    q = data["train_target"].shape[0]
    n_tr, d = data["train_data"].shape
    n_val = data["val_data"].shape[0]
    n_te = data["test_data"].shape[0]
    osr_pct = 100 * float(data["osr_labels"].sum()) / n_te

    if verbose:
        print(f"Dataset: {dataset}  known_ratio={known_ratio}  seed={seed}")
        print(f"  Train: {n_tr}, Validation: {n_val}, Test: {n_te}, Features: {d}, "
              f"Known labels: {q}, OSR%: {osr_pct:.1f}")
        if c_mode != "full":
            print(f"  C-mode: {c_mode}")
        if standardize:
            print(f"  Standardized: yes")

    # ── Kernel ──
    gamma = effective["gamma"]
    train_Kernel = kernelization(data["train_data"], data["train_data"],
                                 "RBF", (gamma,))
    val_Kernel = kernelization(data["val_data"], data["train_data"],
                               "RBF", (gamma,))
    test_Kernel = kernelization(data["test_data"], data["train_data"],
                                "RBF", (gamma,))

    # ── C matrix override (M2) ──
    C_override = None
    if c_mode != "full":
        label_names = [str(n) for n in data.get("known_label_names", [])]
        if not label_names:
            label_names = None
        C_raw = get_C_matrix(c_mode, data["train_target"],
                             label_names=label_names, seed=seed)
        if C_raw is not None:
            # Convert to Laplacian form (same as train.py does internally)
            C_override = np.diag(C_raw @ np.ones(q)) - C_raw

    # ── Train ──
    t0 = time.time()
    model = crem_train(data["train_target"], train_Kernel, nominal,
                       verbose=verbose, C_override=C_override)
    t_train = time.time() - t0

    # ── Test ──
    t0 = time.time()
    result, selection = crem_validate_and_test(
        data["train_target"], data["val_target"], data["test_target"],
        data["val_osr_labels"], data["osr_labels"], model,
        train_Kernel, val_Kernel, test_Kernel, nominal, verbose=verbose)
    t_test = time.time() - t0

    # ── Log ──
    if verbose:
        print(f"\n  AUROC={result['AUROC']:.4f}  AUPR={result['AUPR']:.4f}  "
              f"macroAUC={result['macroAUC']:.4f}  best_K={result.get('best_K','N/A')}")
        print(f"  Time: train={t_train:.1f}s  test={t_test:.1f}s")

    extra = {
        "protocol_version": 2,
        "preprocessing_fit": "train_only",
        "best_K": result.get("best_K"),
        "n_train": n_tr, "n_validation": n_val, "n_test": n_te,
        "train_ratio": train_ratio, "val_ratio": val_ratio,
        "q_known": q, "osr_pct": float(osr_pct),
        "validation": selection,
    }

    path = save_run(method, dataset, known_ratio, seed,
                    result, nominal, effective, t_train, t_test,
                    extra=extra, c_mode=c_mode if c_mode != "full" else None)

    return path


def main():
    parser = argparse.ArgumentParser(
        description="CREM: Multi-label Open-set Recognition — Unified Runner")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (enron, slashdot, bibtex, ...)")
    parser.add_argument("--known_ratio", type=float, default=0.5,
                        help="Fraction of labels treated as known (default: 0.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single random seed")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Multiple seeds: '0-9' or '0,1,2,3'")
    parser.add_argument("--method", type=str, default="crem_v2",
                        help="Method name for results directory (default: crem_v2)")
    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override results root directory")
    parser.add_argument("--standardize", action="store_true",
                        help="Apply feature standardisation (zero mean, unit var)")
    parser.add_argument("--param", type=str, default=None,
                        help='JSON string to override hyperparams, e.g. \'{"gamma":0.10}\'')
    parser.add_argument(
        "--legacy-dataset-params", action="store_true",
        help="Use old per-dataset test-selected parameters (legacy reproduction only)")
    parser.add_argument("--c-mode", type=str, default="full",
                        choices=["full", "identity", "random", "semantic"],
                        help="Label correlation variant for M2 (default: full)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force rebuild of data caches")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--build-cache", action="store_true",
                        help="Build all data caches and exit")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-run output")

    args = parser.parse_args()

    # ── Special actions ──
    if args.list:
        print("Available datasets:")
        for ds in ALL_DATASETS:
            N, d, L, lc = get_dataset_specs(ds)
            print(f"  {ds:20s}  N={N:5d}  d={d:4d}  L={L:2d}  LCard={lc:.3f}")
        return

    if args.build_cache:
        build_all_caches()
        return

    if args.dataset is None:
        parser.error("--dataset is required (or use --list / --build-cache)")

    # ── Param override ──
    param_override = None
    if args.param:
        param_override = json.loads(args.param)

    # ── Output dir override ──
    if args.output_dir:
        from crem import logger
        logger.RESULTS_DIR = args.output_dir
        logger.TABLES_DIR = os.path.join(args.output_dir, "tables")

    # ── Resolve seeds ──
    if args.seeds is not None:
        seeds = parse_seeds(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [0]

    # ── Ensure cache exists ──
    if args.no_cache:
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache",
            f"{args.dataset}_protocol_v2.mat")
        if os.path.exists(cache_path):
            os.remove(cache_path)
    build_full_cache(args.dataset)

    # ── Run ──
    for seed in seeds:
        run_single(
            dataset=args.dataset,
            known_ratio=args.known_ratio,
            seed=seed,
            standardize=args.standardize,
            param_override=param_override,
            c_mode=args.c_mode,
            method=args.method,
            use_dataset_best=args.legacy_dataset_params,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            verbose=not args.quiet,
        )

    # ── Quick summary if multiple seeds ──
    if len(seeds) > 1:
        runs = load_runs(args.method, args.dataset, args.known_ratio,
                         c_mode=args.c_mode if args.c_mode != "full" else None)
        print(f"\n=== Summary: {args.dataset} known_ratio={args.known_ratio} "
              f"({len(runs)} seeds) ===")
        for metric in ["AUROC", "AUPR", "macroAUC"]:
            m, s = mean_std(runs, metric)
            print(f"  {metric}: {m:.4f} ± {s:.4f}")

    from scripts.build_results_report import write_report
    print(f"\nResult summary: {write_report()}")


if __name__ == "__main__":
    main()

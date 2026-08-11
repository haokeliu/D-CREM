#!/usr/bin/env python3
"""Phase 1 destructive / motivation experiments (M1, M2, M3).

Usage:
  # M1: extreme known_ratio
  python scripts/run_phase1.py m1 --dataset enron --seeds 0-9
  python scripts/run_phase1.py m1 --all --seeds 0-9

  # M2: label correlation ablation (3 or 4 variants depending on dataset)
  python scripts/run_phase1.py m2 --dataset enron --seeds 0-9
  python scripts/run_phase1.py m2 --all --seeds 0-9

  # One-shot: both M1 + M2 for a dataset
  python scripts/run_phase1.py all --dataset enron --seeds 0-9
  python scripts/run_phase1.py all --all --seeds 0-9   # all datasets
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crem.config import ALL_DATASETS
RUN_CREM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "run_crem.py")

M1_RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7]
M2_RATIOS = [0.2, 0.5]
# semantic C only makes sense for bibtex (label names are parseable words);
# other datasets have coded labels like C.C5, A.A3 or TF-IDF token indices.
C_MODES_ALL = ["full", "identity", "random", "semantic"]
C_MODES_NO_SEMANTIC = ["full", "identity", "random"]


def _c_modes_for(dataset):
    """Return the appropriate c_mode list for a dataset."""
    if dataset == "bibtex":
        return C_MODES_ALL
    return C_MODES_NO_SEMANTIC


def run_m1(datasets, seeds, python_bin=None):
    """M1: extreme known_ratio experiment."""
    py = python_bin or sys.executable
    total = len(datasets) * len(M1_RATIOS) * len(seeds)
    i = 0
    t_start = time.time()

    for ds in datasets:
        for r in M1_RATIOS:
            seeds_str = ",".join(str(s) for s in seeds)
            cmd = [py, RUN_CREM, "--dataset", ds,
                   "--known_ratio", str(r),
                   "--seeds", seeds_str,
                   "--standardize",
                   "--quiet"]
            print(f"\n[M1] {ds} known_ratio={r}  [{i}/{total}]")
            subprocess.run(cmd, check=True)
            i += len(seeds)

    elapsed = time.time() - t_start
    print(f"\n[M1] Done. {total} runs in {elapsed/60:.1f} min.")


def run_m2(datasets, seeds, python_bin=None):
    """M2: label correlation ablation."""
    py = python_bin or sys.executable
    # Count total runs
    total = 0
    for ds in datasets:
        c_modes = _c_modes_for(ds)
        total += len(c_modes) * len(M2_RATIOS) * len(seeds)

    i = 0
    t_start = time.time()

    for ds in datasets:
        c_modes = _c_modes_for(ds)
        for r in M2_RATIOS:
            for cm in c_modes:
                seeds_str = ",".join(str(s) for s in seeds)
                cmd = [py, RUN_CREM, "--dataset", ds,
                       "--known_ratio", str(r),
                       "--seeds", seeds_str,
                       "--c-mode", cm,
                       "--standardize",
                       "--quiet"]
                print(f"\n[M2] {ds} known_ratio={r} c_mode={cm}  [{i}/{total}]")
                subprocess.run(cmd, check=True)
                i += len(seeds)

    elapsed = time.time() - t_start
    print(f"\n[M2] Done. {total} runs in {elapsed/60:.1f} min.")


def run_m3(datasets, seeds, python_bin=None):
    """M3: feature quality ceiling test (NOT YET IMPLEMENTED).

    Requires VOC2007 / MS-COCO image datasets + pre-trained feature extractors.
    """
    print("\n[M3] NOT YET IMPLEMENTED.")
    print("  M3 requires image datasets (VOC2007, MS-COCO 2014) and pre-trained")
    print("  feature extractors (ResNet-50, CLIP ViT-B/32).")
    print("  Please download the datasets and run the feature extraction pipeline,")
    print("  then re-run this command.")
    print("  See docs/protocol-v2.zh-CN.md for protocol details.")


def summarize_phase1(datasets, seeds):
    """Refresh the machine-readable Protocol-v2 result summary."""
    from scripts.build_results_report import write_report

    path = write_report()
    print(f"\nProtocol-v2 summary updated: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 destructive / motivation experiments")
    sub = parser.add_subparsers(dest="command", help="Experiment to run")

    # ── M1 ──
    p1 = sub.add_parser("m1", help="M1: extreme known_ratio sweep")
    p1.add_argument("--dataset", type=str, default=None)
    p1.add_argument("--all", action="store_true", dest="all_datasets")
    p1.add_argument("--seeds", type=str, default="0-9")
    p1.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── M2 ──
    p2 = sub.add_parser("m2", help="M2: label correlation ablation")
    p2.add_argument("--dataset", type=str, default=None)
    p2.add_argument("--all", action="store_true", dest="all_datasets")
    p2.add_argument("--seeds", type=str, default="0-9")
    p2.add_argument("--python", type=str, default=None, dest="python_bin")

    # ── M3 ──
    sub.add_parser("m3", help="M3: feature quality ceiling test (placeholder)")

    # ── ALL (M1+M2) ──
    pa = sub.add_parser("all", help="Run M1 + M2")
    pa.add_argument("--dataset", type=str, default=None)
    pa.add_argument("--all", action="store_true", dest="all_datasets")
    pa.add_argument("--seeds", type=str, default="0-9")
    pa.add_argument("--python", type=str, default=None, dest="python_bin")
    pa.add_argument("--skip-m1", action="store_true")
    pa.add_argument("--skip-m2", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # ── Resolve datasets ──
    if hasattr(args, "all_datasets") and args.all_datasets:
        datasets = list(ALL_DATASETS)
    elif hasattr(args, "dataset") and args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ["enron"]  # default: enron first for verification

    # Parse seeds
    seed_str = getattr(args, "seeds", "0-9")
    if "-" in seed_str:
        lo, hi = seed_str.split("-", 1)
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in seed_str.split(",")]

    py = getattr(args, "python_bin", None)

    print(f"Phase 1 [{args.command}]")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print()

    if args.command == "m1":
        run_m1(datasets, seeds, python_bin=py)
        summarize_phase1(datasets, seeds)
    elif args.command == "m2":
        run_m2(datasets, seeds, python_bin=py)
        summarize_phase1(datasets, seeds)
    elif args.command == "m3":
        run_m3(datasets, seeds, python_bin=py)
    elif args.command == "all":
        if not args.skip_m1:
            run_m1(datasets, seeds, python_bin=py)
        if not args.skip_m2:
            run_m2(datasets, seeds, python_bin=py)
        summarize_phase1(datasets, seeds)


if __name__ == "__main__":
    main()

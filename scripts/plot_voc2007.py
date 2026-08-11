#!/usr/bin/env python3
"""VOC2007 image comparison charts — Protocol v2 results.

Layout: 2×2 grid
  Row 1: AUROC  (left=CLIP, right=ResNet50)
  Row 2: AUPR   (left=CLIP, right=ResNet50)

Within each panel: grouped bars — 3 ratios × 3 methods (CREM / D-CREM A / D-CREM B).
Error bars = mean ± 1 std across seeds.

Output: results/figures/Fig_image_AUROC.png, .svg + Fig_image_AUPR.png, .svg
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
FIGS_DIR = os.path.join(RESULTS_ROOT, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = ["clip", "resnet50"]
RATIOS = [0.3, 0.5, 0.7]
METRICS = ["AUROC", "AUPR"]

# (method_label, feature, source_dir, setting_template)
SOURCES = [
    ("CREM",     "clip",     "crem_v2", "protocol_v2_m3b_clip_known_ratio={ratio}"),
    ("CREM",     "resnet50", "crem_v2", "protocol_v2_m3b_resnet50_known_ratio={ratio}"),
    ("D-CREM A", "clip",     "dcrem",   "protocol_v2_modeA_clip_identity_r{ratio}"),
    ("D-CREM A", "resnet50", "dcrem",   "protocol_v2_modeA_resnet50_identity_r{ratio}"),
    ("D-CREM B", "clip",     "dcrem",   "protocol_v2_modeB_clip_identity_r{ratio}"),
    ("D-CREM B", "resnet50", "dcrem",   "protocol_v2_modeB_resnet50_identity_r{ratio}"),
]

METHOD_ORDER = ["CREM", "D-CREM A", "D-CREM B"]

# Categorical palette — 3 fixed hues (dataviz validated for light mode)
C_PALETTE = {
    "CREM":     "#5B8CB8",
    "D-CREM A": "#E07B4C",
    "D-CREM B": "#5DAE5D",
}
BAR_EDGE = "#333333"
BAR_W = 0.22
GROUP_PAD = 0.06

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


def load_all():
    """Return dict: (method, feature, ratio) -> list of metric dicts."""
    data = defaultdict(list)
    for method, feat, src, tmpl in SOURCES:
        for ratio in RATIOS:
            rundir = os.path.join(
                RESULTS_ROOT, src, "voc2007", tmpl.format(ratio=ratio))
            if not os.path.isdir(rundir):
                continue
            for fn in sorted(os.listdir(rundir)):
                if not (fn.startswith("seed") and fn.endswith(".json")):
                    continue
                with open(os.path.join(rundir, fn), encoding="utf-8") as fh:
                    rec = json.load(fh)
                metrics = rec.get("metrics", {})
                if "AUROC" in metrics:
                    data[(method, feat, ratio)].append(metrics)
    return data


def ms(arr):
    a = np.array([v for v in arr if v is not None], dtype=np.float64)
    if len(a) == 0:
        return None, None, 0
    return float(np.mean(a)), float(np.std(a, ddof=1)), len(a)


def one_panel(ax, data, feat, metric):
    """Draw grouped bars for one feature × metric panel."""
    n_ratio = len(RATIOS)
    n_method = len(METHOD_ORDER)
    total_groups = n_ratio * n_method

    xs = np.arange(n_ratio) * (n_method * BAR_W + GROUP_PAD)
    all_bars = []

    for j, method in enumerate(METHOD_ORDER):
        hh, ee = [], []
        for ratio in RATIOS:
            recs = data.get((method, feat, ratio), [])
            vals = [r.get(metric) for r in recs]
            m, s, _ = ms(vals)
            hh.append(m if m is not None else 0)
            ee.append(s if s is not None else 0)

        offset = (j - (n_method - 1) / 2) * BAR_W
        bars = ax.bar(
            xs + offset, hh, BAR_W, color=C_PALETTE[method],
            edgecolor=BAR_EDGE, linewidth=0.5,
            yerr=ee, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        all_bars.extend(bars)

        # direct labels on bars
        for xpos, h in zip(xs + offset, hh):
            if h > 0:
                ax.text(xpos, h + 0.008, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=6.5, color="#555555")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"r = {r}" for r in RATIOS])
    ax.set_title(f"{feat.upper()}  ·  {metric}", fontweight="normal", pad=6)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # auto y-range
    all_h = [b.get_height() for b in all_bars if b.get_height() > 0]
    if all_h:
        ax.set_ylim(max(0, min(all_h) * 0.8), max(all_h) * 1.15)


def build_figure(data):
    """Build one figure with both AUROC and AUPR in a 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, metric in enumerate(METRICS):
        for col, feat in enumerate(FEATURES):
            one_panel(axes[row][col], data, feat, metric)

    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_PALETTE[m],
                              edgecolor=BAR_EDGE, linewidth=0.5)
               for m in METHOD_ORDER]
    fig.legend(handles, METHOD_ORDER, loc="lower center", ncol=3,
               frameon=True, fancybox=False, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, -0.01), fontsize=9)

    fig.suptitle("VOC2007  —  Frozen Image Features  (mean ± std, N=10 seeds)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


def main():
    data = load_all()
    total = sum(len(v) for v in data.values())
    print(f"Loaded {total} records across {len(data)} (method, feature, ratio) keys")

    # completeness check
    for method in METHOD_ORDER:
        for feat in FEATURES:
            for ratio in RATIOS:
                n = len(data.get((method, feat, ratio), []))
                status = "OK" if n == 10 else f"MISSING ({n}/10)"
                if n < 10:
                    print(f"  WARN: {method}/{feat} r={ratio}: {status}")

    fig = build_figure(data)
    for ext in ("png", "svg"):
        path = os.path.join(FIGS_DIR, f"Fig_image_VOC2007.{ext}")
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)

    # refresh the machine-readable Protocol-v2 summary
    from scripts.build_results_report import write_report
    print(f"Result summary: {write_report()}")

    print(f"\nAll outputs in {FIGS_DIR}/")


if __name__ == "__main__":
    main()

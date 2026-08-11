"""Unified experiment logger: JSON persistence + auto markdown summarisation.

Each run is saved to:
  results/{method}/{dataset}/{setting}/seed{seed}.json

with fields: metrics, config (nominal + effective), time, timestamp, extra.

The module also provides table builders that read back the JSON files and
produce markdown tables (mean ± std) and paired t-test p-values.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

from .config import RESULTS_DIR, get_paper_auroc


# ── Metric order for display ────────────────────────────────────────────
METRIC_ORDER = [
    "AUROC", "AUPR", "macroAUC", "AveragePrecision",
    "RankingLoss", "Coverage", "OneError",
]

METRIC_LATEX = {
    "AUROC": "AUROC",
    "AUPR": "AUPR",
    "macroAUC": "MacroAUC",
    "AveragePrecision": "AvgPrec",
    "RankingLoss": "RankLoss",
    "Coverage": "Coverage",
    "OneError": "OneErr",
}


# ═══════════════════════════════════════════════════════════════════════════
# Single-run persistence
# ═══════════════════════════════════════════════════════════════════════════

def make_setting_name(known_ratio, c_mode=None):
    """Encode experiment setting as a directory name."""
    base = f"known_ratio={known_ratio}"
    if c_mode and c_mode != "full":
        base += f"_c={c_mode}"
    return base


def save_run(method, dataset, known_ratio, seed, metrics, config_nominal,
             config_effective, duration_train, duration_test, extra=None,
             c_mode=None):
    """Persist one experimental run as JSON.

    Parameters
    ----------
    method : str           e.g. 'crem'
    dataset : str          e.g. 'enron'
    known_ratio : float    e.g. 0.5
    seed : int
    metrics : dict         {AUROC, AUPR, macroAUC, AveragePrecision, ...}
    config_nominal : dict  hyperparams as passed to crem_train
    config_effective : dict  hyperparams after swap (actual values used)
    duration_train : float seconds
    duration_test : float  seconds
    extra : dict | None    optional extra data (K-search detail, loss curve, …)
    c_mode : str | None    'full' / 'identity' / 'random' / 'semantic'
    """
    setting = make_setting_name(known_ratio, c_mode)
    run_dir = os.path.join(RESULTS_DIR, method, dataset, setting)
    os.makedirs(run_dir, exist_ok=True)

    record = {
        "metrics": {
            k: (float(metrics[k]) if np.isfinite(metrics[k]) else None)
            for k in METRIC_ORDER if k in metrics
        },
        "config": {
            "nominal": {k: v for k, v in config_nominal.items()
                        if not isinstance(v, (np.ndarray,))},
            "effective": {k: v for k, v in config_effective.items()
                          if not isinstance(v, (np.ndarray,))},
        },
        "time": {
            "train_s": round(float(duration_train), 3),
            "test_s": round(float(duration_test), 3),
            "total_s": round(float(duration_train + duration_test), 3),
        },
        "timestamp": datetime.now().isoformat(),
        "known_ratio": known_ratio,
        "seed": seed,
        "dataset": dataset,
        "method": method,
    }
    if c_mode:
        record["c_mode"] = c_mode
    if extra:
        record["extra"] = _sanitise_extra(extra)

    path = os.path.join(run_dir, f"seed{seed}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str, allow_nan=False)

    return path


def _sanitise_extra(extra):
    """Convert numpy values in extra dict to plain Python for JSON."""
    if isinstance(extra, dict):
        return {k: _sanitise_extra(v) for k, v in extra.items()}
    if isinstance(extra, (list, tuple)):
        return [_sanitise_extra(v) for v in extra]
    if isinstance(extra, (np.integer,)):
        return int(extra)
    if isinstance(extra, (np.floating,)):
        return float(extra) if np.isfinite(extra) else None
    if isinstance(extra, float):
        return extra if np.isfinite(extra) else None
    if isinstance(extra, np.ndarray):
        return extra.tolist()
    return extra


# ═══════════════════════════════════════════════════════════════════════════
# Bulk loading
# ═══════════════════════════════════════════════════════════════════════════

def load_runs(method, dataset, known_ratio, c_mode=None):
    """Load all seed JSON files for one (method, dataset, setting) combo.

    Returns list of dicts, sorted by seed.
    """
    setting = make_setting_name(known_ratio, c_mode)
    run_dir = os.path.join(RESULTS_DIR, method, dataset, setting)
    if not os.path.isdir(run_dir):
        return []
    runs = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("seed") and fname.endswith(".json"):
            with open(os.path.join(run_dir, fname)) as f:
                runs.append(json.load(f))
    runs.sort(key=lambda r: r.get("seed", 0))
    return runs


def load_all_runs(method, dataset):
    """Load all runs for a given method+dataset across all settings.

    Returns dict: setting_name -> list of run dicts.
    """
    base = os.path.join(RESULTS_DIR, method, dataset)
    if not os.path.isdir(base):
        return {}
    result = {}
    for setting in sorted(os.listdir(base)):
        setting_dir = os.path.join(base, setting)
        if not os.path.isdir(setting_dir):
            continue
        runs = []
        for fname in sorted(os.listdir(setting_dir)):
            if fname.startswith("seed") and fname.endswith(".json"):
                with open(os.path.join(setting_dir, fname)) as f:
                    runs.append(json.load(f))
        if runs:
            runs.sort(key=lambda r: r.get("seed", 0))
            result[setting] = runs
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_metric(runs, metric):
    """Extract vector of a metric across runs."""
    return np.array([r["metrics"].get(metric, np.nan) for r in runs])


def mean_std(runs, metric):
    """Return (mean, std) for a metric across runs."""
    vals = _extract_metric(runs, metric)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def paired_ttest(runs_a, runs_b, metric):
    """Paired t-test p-value between two groups (must be same length)."""
    a = _extract_metric(runs_a, metric)
    b = _extract_metric(runs_b, metric)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return float("nan")
    t_stat, p = stats.ttest_rel(a[mask], b[mask])
    return float(p)


# ═══════════════════════════════════════════════════════════════════════════
# Markdown table builders
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_mean_std(mean_val, std_val, decimals=4):
    """Format as 'mean ± std'."""
    if np.isnan(mean_val):
        return "N/A"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def _fmt_pvalue(p):
    """Format p-value with significance stars."""
    if np.isnan(p):
        return "N/A"
    stars = ""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    return f"{p:.4f}{stars}"


def build_p0_table(method, datasets, known_ratio=0.5):
    """Build Phase 0 reproduction-check table (mean ± std vs paper).

    Returns markdown string.
    """
    lines = [
        "## Table_P0: Reproduction check (mean ± std across seeds)",
        "",
        f"| Dataset | AUROC (ours) | AUROC (paper) | Δ | AUPR (ours) | macroAUC (ours) |",
        "|---|---|---|---|---|---|",
    ]
    for ds in datasets:
        runs = load_runs(method, ds, known_ratio)
        if not runs:
            lines.append(f"| {ds} | (no data) | | | | |")
            continue
        auroc_m, auroc_s = mean_std(runs, "AUROC")
        aupr_m, aupr_s = mean_std(runs, "AUPR")
        mauc_m, mauc_s = mean_std(runs, "macroAUC")
        paper_auroc = get_paper_auroc(ds)
        delta = (auroc_m - paper_auroc) if (paper_auroc and not np.isnan(auroc_m)) else float("nan")
        delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        paper_s = f"{paper_auroc:.3f}" if paper_auroc else "N/A"
        lines.append(
            f"| {ds} | {_fmt_mean_std(auroc_m, auroc_s)} | {paper_s} | {delta_s} | "
            f"{_fmt_mean_std(aupr_m, aupr_s)} | {_fmt_mean_std(mauc_m, mauc_s)} |"
        )
    return "\n".join(lines)


def build_m1_table(method, datasets, ratios, metric="AUROC"):
    """Build M1 Table: known_ratio × dataset matrix.

    Returns markdown string.
    """
    header = "| Dataset | " + " | ".join(f"r={r}" for r in ratios) + " |"
    sep = "|---" * (len(ratios) + 1) + "|"
    lines = [
        f"## Table_M1: {metric} across known_ratios (mean ± std)",
        "",
        header,
        sep,
    ]
    for ds in datasets:
        cells = [ds]
        for r in ratios:
            runs = load_runs(method, ds, r)
            if not runs:
                cells.append("N/A")
            else:
                m, s = mean_std(runs, metric)
                cells.append(_fmt_mean_std(m, s, 4))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_m2_table(method, datasets, ratios, c_modes, metric="AUROC"):
    """Build M2 Table: c_mode × ratio × dataset comparison with p-values.

    Returns markdown string.
    """
    lines = [
        f"## Table_M2: Label correlation ablation — {metric} (mean ± std)",
        "",
    ]

    for r in ratios:
        lines.append(f"### known_ratio = {r}")
        lines.append("")
        header = "| Dataset | " + " | ".join(c_modes) + " |"
        sep = "|---" * (len(c_modes) + 1) + "|"
        lines.append(header)
        lines.append(sep)

        for ds in datasets:
            cells = [ds]
            run_groups = {}
            for cm in c_modes:
                run_groups[cm] = load_runs(method, ds, r, c_mode=cm)

            # Check if all modes have data
            if all(len(g) == 0 for g in run_groups.values()):
                cells.extend(["N/A"] * len(c_modes))
            else:
                for cm in c_modes:
                    runs = run_groups[cm]
                    if not runs:
                        cells.append("N/A")
                    else:
                        m, s = mean_std(runs, metric)
                        cells.append(_fmt_mean_std(m, s, 4))
            lines.append("| " + " | ".join(cells) + " |")

            # p-value row for THIS dataset (full vs each variant)
            runs_full = run_groups.get("full", [])
            p_cells = [f"  p (full vs)"]
            for cm in c_modes:
                if cm == "full":
                    p_cells.append("—")
                else:
                    runs_cm = run_groups.get(cm, [])
                    if len(runs_full) < 2 or len(runs_cm) < 2:
                        p_cells.append("N/A")
                    else:
                        p = paired_ttest(runs_full, runs_cm, metric)
                        p_cells.append(_fmt_pvalue(p))
            lines.append("| " + " | ".join(p_cells) + " |")

        lines.append("")

    return "\n".join(lines)

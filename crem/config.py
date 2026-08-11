"""Centralised hyperparameter configuration for CREM experiments.

Provides paper defaults, legacy per-dataset hyperparameters, and paper
reference values.  All functions return BOTH nominal (MATLAB input naming)
and effective (post-swap) parameter dictionaries so the logger can record
what actually takes effect.

The swap quirk (preserved from MATLAB): inside crem_train(),
  lamda1_eff = param["lamda3"]   # coupling W-P
  lamda2_eff = param["lamda2"]   # label correlation graph regularisation
  lamda3_eff = param["lamda1"]   # ridge penalty on W
  alpha is hard-coded to 1 regardless of param["alpha"]
"""

import os

# ── Paper nominal defaults (CREM Table 1, main.py line 41) ──────────────
DEFAULT_PARAMS_NOMINAL = {
    "lamda1": 1,
    "lamda2": 0.1,
    "lamda3": 10,
    "alpha": 1,
    "gamma": 0.05,
    "K": 3,  # fallback when K-search is disabled
}

# ── Per-dataset best hyperparameters (nominal, from grid search) ────────
# Source: crem-hyperparameters.md memory, 2026-07-29
# These are NOMINAL values (paper naming); effective values after swap
# are computed by get_params().
PER_DATASET_BEST_NOMINAL = {
    "enron": {
        "lamda1": 1, "lamda2": 0.1, "lamda3": 10, "alpha": 1, "gamma": 0.10,
    },
    "slashdot": {
        "lamda1": 1, "lamda2": 0.01, "lamda3": 100, "alpha": 1, "gamma": 0.50,
    },
    "yahoo-recreation": {
        "lamda1": 1, "lamda2": 0.01, "lamda3": 100, "alpha": 1, "gamma": 0.03,
    },
    "yahoo-arts": {
        "lamda1": 1, "lamda2": 0.01, "lamda3": 10, "alpha": 1, "gamma": 0.20,
    },
    "yahoo-education": {
        "lamda1": 1, "lamda2": 0.1, "lamda3": 100, "alpha": 1, "gamma": 0.50,
    },
    "bibtex": {
        "lamda1": 1, "lamda2": 0.1, "lamda3": 10, "alpha": 1, "gamma": 0.05,
    },
}

# ── Paper Table 1 reference values (AUROC mean) ─────────────────────────
PAPER_TABLE1_AUROC = {
    "enron": 0.592,
    "slashdot": 0.562,
    "yahoo-recreation": 0.591,
    "yahoo-arts": 0.628,
    "yahoo-education": 0.599,
    "bibtex": 0.567,
}

# ── Dataset preprocessing targets (Paper Table 2) ───────────────────────
DATASET_SPECS = {
    "enron":            {"N": 1702, "d": 1001, "L": 24, "LCard": 3.124},
    "slashdot":         {"N": 3659, "d": 1079, "L": 14, "LCard": 1.173},
    "yahoo-recreation": {"N": 5000, "d": 606,  "L": 15, "LCard": 1.361},
    "yahoo-arts":       {"N": 5000, "d": 462,  "L": 14, "LCard": 1.512},
    "yahoo-education":  {"N": 5000, "d": 550,  "L": 11, "LCard": 1.374},
    "bibtex":           {"N": 7395, "d": 1835, "L": 27, "LCard": 0.954},
}

# ── All available datasets (order matters for tables) ───────────────────
ALL_DATASETS = [
    "enron", "slashdot", "yahoo-recreation", "yahoo-arts",
    "yahoo-education", "bibtex",
]

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Source data: env var override, else the repo-bundled datasets_raw/
SOURCE_DIR = os.environ.get(
    "CREM_DATA_DIR",
    os.path.join(BASE_DIR, "datasets_raw"),
)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


def effective_from_nominal(nominal):
    """Resolve the single MATLAB-compatible parameter swap.

    ``crem_train`` accepts *nominal* parameters and performs this conversion
    exactly once.  Keeping the conversion here prevents callers from passing
    an already-swapped dictionary back into the trainer.
    """
    return {
        "lamda1": nominal["lamda3"],  # effective coupling W-P
        "lamda2": nominal["lamda2"],  # label correlation
        "lamda3": nominal["lamda1"],  # effective ridge penalty on W
        "alpha": 1,                    # hard-coded in the MATLAB code
        "gamma": nominal.get("gamma", DEFAULT_PARAMS_NOMINAL["gamma"]),
        "K": nominal.get("K", 3),
    }


def get_params(dataset=None, param_override=None, use_dataset_best=False):
    """Return (nominal, effective) parameter dicts for a dataset.

    Priority: param_override > selected base configuration.

    Paper defaults are used unless ``use_dataset_best=True``.  The legacy
    per-dataset settings were selected on test AUROC and therefore must not be
    used by the leakage-free train/validation/test protocol.

    The *effective* dict reflects the swap inside crem_train():
      lamda1_eff ← param["lamda3"]
      lamda2_eff ← param["lamda2"]
      lamda3_eff ← param["lamda1"]
      alpha_eff  ← 1 (hard-coded)
    """
    if use_dataset_best and dataset and dataset in PER_DATASET_BEST_NOMINAL:
        nominal = dict(PER_DATASET_BEST_NOMINAL[dataset])
    else:
        nominal = dict(DEFAULT_PARAMS_NOMINAL)

    if param_override:
        nominal.update(param_override)

    effective = effective_from_nominal(nominal)

    return nominal, effective


def get_dataset_specs(name):
    """Return (N, d, L, LCard) targets for a dataset."""
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_SPECS)}")
    s = DATASET_SPECS[name]
    return s["N"], s["d"], s["L"], s["LCard"]


def get_paper_auroc(name):
    """Return the paper-reported AUROC mean for a dataset, or None."""
    return PAPER_TABLE1_AUROC.get(name)

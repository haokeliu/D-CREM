#!/usr/bin/env python3
"""Aggregate the pre-registered Protocol-v2 supplementary experiments."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
DATASETS = [
    "enron", "slashdot", "bibtex", "yahoo-recreation", "yahoo-arts",
    "yahoo-education",
]
ABLATION_DATASETS = ["enron", "slashdot", "bibtex"]
RATIOS = [0.3, 0.5, 0.7]
SEEDS = list(range(10))
ABLATIONS = ["N1", "E1", "S1", "U1"]
SENSITIVITY_SEEDS = list(range(5))
IMAGE_E2E_SEEDS = list(range(5))


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def percentile_ci(values, rng, n_boot=10_000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def sign_flip_p(values):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        signs = np.array([1.0 if mask & (1 << i) else -1.0
                          for i in range(len(values))])
        exceed += abs((values * signs).mean()) >= observed - 1e-15
    return float(exceed / total)


def paired_stats(values, rng):
    values = np.asarray(values, dtype=float)
    try:
        p_w = float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        p_w = 1.0
    return {
        "n": int(len(values)),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci95": percentile_ci(values, rng),
        "wilcoxon_p": p_w,
        "exact_sign_flip_p": sign_flip_p(values),
        "paired_deltas": values.tolist(),
    }


def holm(items, p_key, output_key):
    ordered = sorted(items, key=lambda item: item[p_key])
    running = 0.0
    m = len(ordered)
    for rank, item in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * item[p_key])
        running = max(running, adjusted)
        item[output_key] = running


def hierarchical_ci(setting_values, rng, n_boot=10_000):
    """Equal-setting bootstrap, resampling seeds inside each selected setting."""
    keys = list(setting_values)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        means = []
        for index in chosen:
            values = np.asarray(setting_values[keys[index]], dtype=float)
            means.append(rng.choice(values, size=len(values), replace=True).mean())
        draws[b] = np.mean(means)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def ablation_path(dataset, variant, ratio, seed, mode="A"):
    return (RESULTS / "dcrem" / dataset /
            f"protocol_v2_ablation_mode{mode}_core_{variant}_r{ratio}" /
            f"seed{seed}.json")


def build_ablation(mode="A"):
    rng = np.random.default_rng(20260809)
    output = {"protocol_version": 2, "primary_metric": "AUROC", "metrics": {}}
    for metric in ("AUROC", "AUPR", "macroAUC"):
        rows = []
        across = {variant: {} for variant in ABLATIONS}
        for dataset in ABLATION_DATASETS:
            for ratio in RATIOS:
                family = []
                for variant in ABLATIONS:
                    deltas = []
                    for seed in SEEDS:
                        full = load(ablation_path(
                            dataset, "full", ratio, seed, mode=mode))
                        ablated = load(ablation_path(
                            dataset, variant, ratio, seed, mode=mode))
                        a = full["metrics"].get(metric)
                        b = ablated["metrics"].get(metric)
                        if finite(a) and finite(b):
                            deltas.append(float(a - b))
                    if len(deltas) != 10:
                        raise RuntimeError(
                            f"Incomplete {metric}: {variant}/{dataset}/{ratio}: {len(deltas)}")
                    row = {"variant": variant, "dataset": dataset, "ratio": ratio,
                           **paired_stats(deltas, rng)}
                    family.append(row)
                    rows.append(row)
                    across[variant][f"{dataset}|{ratio}"] = deltas
                holm(family, "exact_sign_flip_p", "holm_sign_flip_p")
                holm(family, "wilcoxon_p", "holm_wilcoxon_p")
        aggregate = {}
        for variant, settings in across.items():
            setting_means = [float(np.mean(values)) for values in settings.values()]
            aggregate[variant] = {
                "settings": len(settings),
                "equal_setting_mean_delta": float(np.mean(setting_means)),
                "hierarchical_ci95": hierarchical_ci(settings, rng),
                "positive_settings": int(sum(value > 0 for value in setting_means)),
            }
        output["metrics"][metric] = {"setting_rows": rows, "aggregate": aggregate}
    output["mode"] = mode
    path = TABLES / f"ablation_mode{mode}_core_effects.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sensitivity_path(dataset, config_id, ratio, seed):
    return (RESULTS / "dcrem" / dataset /
            f"protocol_v2_sensitivity_modeB_core_{config_id}_r{ratio}" /
            f"seed{seed}.json")


def sensitivity_complete(config_ids):
    return all(
        sensitivity_path(dataset, config_id, ratio, seed).exists()
        for config_id in config_ids
        for dataset in ABLATION_DATASETS
        for ratio in RATIOS
        for seed in SENSITIVITY_SEEDS)


def mode_b_ablation_complete():
    return all(
        ablation_path(dataset, variant, ratio, seed, mode="B").exists()
        for variant in ["full", *ABLATIONS]
        for dataset in ABLATION_DATASETS
        for ratio in RATIOS
        for seed in SEEDS)


def build_sensitivity():
    from scripts.run_phase3 import SENSITIVITY_CONFIGS

    rng = np.random.default_rng(20260814)
    config_ids = list(SENSITIVITY_CONFIGS)
    output = {
        "protocol_version": 2,
        "mode": "B",
        "design": "pre_registered_one_factor_at_a_time",
        "reference": {
            "config_id": "REF", "block_interval": 10,
            "lamda1": 1.0, "beta": 0.1, "embedding_dim": 128,
        },
        "configs": SENSITIVITY_CONFIGS,
        "seeds": SENSITIVITY_SEEDS,
        "metrics": {},
    }

    def metric_value(record, metric):
        if metric == "validation_AUROC":
            return record["protocol"]["validation_metrics"]["AUROC"]
        if metric == "train_s":
            return record["time"]["train_s"]
        return record["metrics"][metric]

    for metric in ("AUROC", "AUPR", "macroAUC", "validation_AUROC", "train_s"):
        rows = []
        effects = {config_id: {} for config_id in config_ids if config_id != "REF"}
        for dataset in ABLATION_DATASETS:
            for ratio in RATIOS:
                ref_records = [load(sensitivity_path(
                    dataset, "REF", ratio, seed)) for seed in SENSITIVITY_SEEDS]
                ref_values = [metric_value(record, metric) for record in ref_records]
                for config_id in config_ids:
                    records = [load(sensitivity_path(
                        dataset, config_id, ratio, seed))
                        for seed in SENSITIVITY_SEEDS]
                    values = [metric_value(record, metric) for record in records]
                    deltas = [float(value - ref) for value, ref in zip(values, ref_values)]
                    row = {
                        "config_id": config_id,
                        "factor": SENSITIVITY_CONFIGS[config_id]["factor"],
                        "value": SENSITIVITY_CONFIGS[config_id]["value"],
                        "dataset": dataset,
                        "ratio": ratio,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "sample_std": float(np.std(values, ddof=1)),
                        "mean_delta_vs_ref": float(np.mean(deltas)),
                    }
                    rows.append(row)
                    if config_id != "REF":
                        effects[config_id][f"{dataset}|{ratio}"] = deltas

                    expected = SENSITIVITY_CONFIGS[config_id]
                    expected_values = {
                        "block_interval": 10,
                        "lamda1": 1.0,
                        "beta": 0.1,
                        "embedding_dim": 128,
                    }
                    if expected["factor"] != "reference":
                        expected_values[expected["factor"]] = expected["value"]
                    for record in records:
                        config = record["config"]
                        if config["mode"] != "B":
                            raise RuntimeError(f"Non-Mode-B sensitivity record: {config_id}")
                        for key, expected_value in expected_values.items():
                            if config[key] != expected_value:
                                raise RuntimeError(
                                    f"{key} mismatch: {config_id}; "
                                    f"expected {expected_value}, got {config[key]}")

        aggregate = {}
        for config_id, settings in effects.items():
            setting_means = [float(np.mean(values)) for values in settings.values()]
            aggregate[config_id] = {
                "settings": len(settings),
                "equal_setting_mean_delta_vs_ref": float(np.mean(setting_means)),
                "hierarchical_ci95": hierarchical_ci(settings, rng),
                "positive_settings": int(sum(value > 0 for value in setting_means)),
            }
        output["metrics"][metric] = {"setting_rows": rows, "aggregate": aggregate}

    path = TABLES / "modeB_core_sensitivity.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def main_record(method, dataset, ratio, seed):
    if method in ("D-CREM A", "D-CREM B"):
        mode = method[-1]
        path = (RESULTS / "dcrem" / dataset /
                f"protocol_v2_paper_core_mode{mode}_r{ratio}" / f"seed{seed}.json")
    elif method == "CREM":
        path = RESULTS / "crem_v2" / dataset / f"known_ratio={ratio}" / f"seed{seed}.json"
    else:
        folder = {"OC-SVM": "ocsvm", "IFOREST": "iforest", "SLAN": "slan",
                  "MUENL-F": "muenl_f"}[method]
        path = RESULTS / "baselines_v2" / folder / dataset / f"known_ratio={ratio}" / f"seed{seed}.json"
    return load(path)


def paper_core_record(mode, dataset, ratio, seed):
    path = (RESULTS / "dcrem" / dataset /
            f"protocol_v2_paper_core_mode{mode}_r{ratio}" / f"seed{seed}.json")
    return load(path)


def build_paper_core_mode_comparison():
    """Summarize paired Mode A/B quality, runtime, and convergence histories."""
    rows = []
    paired_deltas = []
    mode_effects = {dataset: {} for dataset in DATASETS}
    mode_values = {mode: {"AUROC": [], "train_s": []} for mode in ("A", "B")}
    histories = {mode: [] for mode in ("A", "B")}

    for dataset in DATASETS:
        for ratio in RATIOS:
            records = {
                mode: [paper_core_record(mode, dataset, ratio, seed)
                       for seed in SEEDS]
                for mode in ("A", "B")
            }
            row = {"dataset": dataset, "ratio": ratio, "n": len(SEEDS)}
            for mode in ("A", "B"):
                aurocs = [record["metrics"]["AUROC"] for record in records[mode]]
                train_times = [record["time"]["train_s"] for record in records[mode]]
                row[f"mode_{mode}_auroc_mean"] = float(np.mean(aurocs))
                row[f"mode_{mode}_auroc_sample_std"] = float(np.std(aurocs, ddof=1))
                row[f"mode_{mode}_train_s_mean"] = float(np.mean(train_times))
                mode_values[mode]["AUROC"].extend(aurocs)
                mode_values[mode]["train_s"].extend(train_times)
                histories[mode].extend(
                    record["diagnostics"]["history"]["loss"] for record in records[mode])
            deltas = [
                records["A"][index]["metrics"]["AUROC"] -
                records["B"][index]["metrics"]["AUROC"]
                for index in range(len(SEEDS))
            ]
            row["paired_A_minus_B_auroc_mean"] = float(np.mean(deltas))
            mode_effects[dataset][ratio] = deltas
            paired_deltas.extend(deltas)
            rows.append(row)

    aggregate = {}
    for mode in ("A", "B"):
        loss = np.asarray(histories[mode], dtype=float)
        aggregate[mode] = {
            "runs": int(len(mode_values[mode]["AUROC"])),
            "auroc_mean": float(np.mean(mode_values[mode]["AUROC"])),
            "auroc_sample_std": float(np.std(mode_values[mode]["AUROC"], ddof=1)),
            "train_s_mean": float(np.mean(mode_values[mode]["train_s"])),
            "train_s_sample_std": float(np.std(mode_values[mode]["train_s"], ddof=1)),
            "mean_loss_by_epoch": np.mean(loss, axis=0).tolist(),
            "sample_std_loss_by_epoch": np.std(loss, axis=0, ddof=1).tolist(),
        }
    paired_array = np.asarray(paired_deltas, dtype=float)
    aggregate["paired_A_minus_B_auroc"] = {
        "n": int(len(paired_array)),
        "mean_delta": float(np.mean(paired_array)),
        "median_delta": float(np.median(paired_array)),
        "ci95": percentile_ci(paired_array, np.random.default_rng(20260812)),
        "wilcoxon_p": float(wilcoxon(paired_array, alternative="two-sided").pvalue),
        "dataset_first_hierarchical_ci95": dataset_first_ci(
            mode_effects, np.random.default_rng(20260813)),
    }
    output = {
        "protocol_version": 2,
        "primary_metric": "AUROC",
        "paper_core": "classifier_induced_reciprocal",
        "rows": rows,
        "aggregate": aggregate,
    }
    path = TABLES / "paper_core_mode_comparison.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def dataset_first_ci(effects, rng, n_boot=10_000):
    datasets = list(effects)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in chosen_datasets:
            ratio_map = effects[dataset]
            chosen_ratios = rng.choice(list(ratio_map), size=len(ratio_map), replace=True)
            ratio_means = []
            for ratio in chosen_ratios:
                values = np.asarray(ratio_map[float(ratio)], dtype=float)
                ratio_means.append(rng.choice(values, size=len(values), replace=True).mean())
            dataset_means.append(np.mean(ratio_means))
        draws[b] = np.mean(dataset_means)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def build_main_stats():
    rng = np.random.default_rng(20260811)
    comparators = ["CREM", "OC-SVM", "IFOREST", "SLAN", "MUENL-F"]
    output = {"protocol_version": 2, "primary_metric": "AUROC", "methods": {}}
    for method in ("D-CREM A", "D-CREM B"):
        method_output = {"comparisons": {}}
        for comparator in comparators:
            effects = {dataset: {} for dataset in DATASETS}
            for dataset in DATASETS:
                for ratio in RATIOS:
                    effects[dataset][ratio] = [
                        main_record(method, dataset, ratio, seed)["metrics"]["AUROC"] -
                        main_record(comparator, dataset, ratio, seed)["metrics"]["AUROC"]
                        for seed in SEEDS
                    ]
            dataset_effects = {
                dataset: float(np.mean([value for values in ratios.values()
                                        for value in values]))
                for dataset, ratios in effects.items()
            }
            method_output["comparisons"][comparator] = {
                "dataset_effects": dataset_effects,
                "equal_dataset_mean_delta": float(np.mean(list(dataset_effects.values()))),
                "dataset_first_hierarchical_ci95": dataset_first_ci(effects, rng),
                "positive_datasets": int(sum(value > 0 for value in dataset_effects.values())),
            }
        output["methods"][method] = method_output
    path = TABLES / "main_hierarchical_stats.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_voc():
    rows = []
    specs = []
    for feature in ("clip", "resnet50"):
        specs.extend([
            ("D-CREM A", RESULTS / "dcrem" / "voc2007",
             f"protocol_v2_modeA_{feature}_identity_r{{ratio}}"),
            ("D-CREM B", RESULTS / "dcrem" / "voc2007",
             f"protocol_v2_modeB_{feature}_identity_r{{ratio}}"),
            ("OC-SVM", RESULTS / "baselines_v2" / "ocsvm" / "voc2007",
             f"feature={feature}_known_ratio={{ratio}}"),
            ("IFOREST", RESULTS / "baselines_v2" / "iforest" / "voc2007",
             f"feature={feature}_known_ratio={{ratio}}"),
        ])
        for method, base, setting_template in specs[-4:]:
            for ratio in RATIOS:
                records = [load(base / setting_template.format(ratio=ratio) / f"seed{s}.json")
                           for s in SEEDS]
                for metric in ("AUROC", "AUPR"):
                    values = [record["metrics"][metric] for record in records]
                    rows.append({"feature": feature, "method": method, "ratio": ratio,
                                 "metric": metric, "n": 10,
                                 "mean": float(np.mean(values)),
                                 "sample_std": float(np.std(values, ddof=1))})
    path = TABLES / "voc2007_controls.json"
    path.write_text(json.dumps({"protocol_version": 2, "rows": rows},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def voc_e2e_complete():
    base = RESULTS / "dcrem" / "voc2007"
    return all(
        (base / f"protocol_v2_paper_core_image_{arm}_resnet50_modeB_r{ratio}" /
         f"seed{seed}.json").exists()
        for arm in ("e2e", "frozen")
        for ratio in RATIOS for seed in IMAGE_E2E_SEEDS)


def build_voc_e2e():
    """Aggregate paired raw-image ResNet-50 end-to-end and frozen controls."""
    base = RESULTS / "dcrem" / "voc2007"
    records = {arm: {} for arm in ("e2e", "frozen")}
    rows = []
    for arm in records:
        for ratio in RATIOS:
            setting = base / f"protocol_v2_paper_core_image_{arm}_resnet50_modeB_r{ratio}"
            current = [load(setting / f"seed{seed}.json")
                       for seed in IMAGE_E2E_SEEDS]
            records[arm][ratio] = current
            for metric in ("AUROC", "AUPR", "macroAUC"):
                values = np.asarray([item["metrics"][metric] for item in current])
                rows.append({
                    "arm": arm, "ratio": ratio, "metric": metric,
                    "n": len(values), "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                })

    rng = np.random.default_rng(20260810)
    paired = {}
    for metric in ("AUROC", "AUPR", "macroAUC"):
        by_setting = {
            ("voc2007", ratio): [
                records["e2e"][ratio][index]["metrics"][metric] -
                records["frozen"][ratio][index]["metrics"][metric]
                for index in range(len(IMAGE_E2E_SEEDS))]
            for ratio in RATIOS
        }
        paired[metric] = {
            "direction": "e2e_minus_frozen_backbone",
            "equal_ratio_mean_delta": float(np.mean([
                value for values in by_setting.values() for value in values])),
            "hierarchical_ci95": hierarchical_ci(by_setting, rng),
            "positive_ratio_settings": int(sum(
                np.mean(values) > 0 for values in by_setting.values())),
            "setting_deltas": {
                str(key[1]): values for key, values in by_setting.items()
            },
        }
    resource = {}
    for arm in records:
        flat = [item for ratio in RATIOS for item in records[arm][ratio]]
        resource[arm] = {
            "mean_train_s": float(np.mean([item["time"]["train_s"] for item in flat])),
            "mean_max_cuda_memory_mb": float(np.mean([
                item["time"]["max_cuda_memory_mb"] for item in flat])),
        }
    output = {
        "protocol_version": 2,
        "design": "paired_raw_voc2007_resnet50_modeB",
        "runs": 30,
        "rows": rows,
        "paired_effects": paired,
        "resources": resource,
    }
    path = TABLES / "voc2007_e2e_resnet50.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = [build_ablation(), build_main_stats(),
             build_paper_core_mode_comparison()]
    if mode_b_ablation_complete():
        paths.append(build_ablation("B"))
    from scripts.run_phase3 import SENSITIVITY_CONFIGS
    if sensitivity_complete(SENSITIVITY_CONFIGS):
        paths.append(build_sensitivity())
    if voc_e2e_complete():
        paths.append(build_voc_e2e())
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

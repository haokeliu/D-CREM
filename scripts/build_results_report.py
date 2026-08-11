#!/usr/bin/env python3
"""Summarize Protocol-v2 results without modifying experiment documentation.

Only JSON records under the v2 result roots and carrying protocol version 2
are accepted. Legacy directories are intentionally invisible. A structured
JSON summary is always written; Markdown is generated only when requested.
"""

from __future__ import annotations

import json
import math
import os
import argparse
from collections import defaultdict
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
SUMMARY_PATH = os.path.join(RESULTS_ROOT, "tables", "results_summary.json")

TABULAR_DATASETS = [
    "enron", "slashdot", "bibtex", "yahoo-recreation", "yahoo-arts",
    "yahoo-education",
]
MAIN_RATIOS = [0.3, 0.5, 0.7]
MAIN_SEEDS = 10

# These records obey Protocol v2 mechanically, but their old single-setting
# ablation design is superseded: A3 disabled correlation rather than holding a
# train-fold static C and is identical to A9.  Keep the JSON for diagnosis but
# never surface it as current paper-facing evidence.
SUPERSEDED_ABLATION_SETTINGS = {
    f"protocol_v2_ablation_{ablation}_r0.5"
    for ablation in ("A1", "A2", "A3", "A4", "A6", "A7", "A9")
}


def _is_superseded_dcrem_setting(setting):
    """Exclude pre-core paper runs while retaining their JSON for audit."""
    if setting in SUPERSEDED_ABLATION_SETTINGS:
        return True
    if setting.startswith("protocol_v2_ablation_modeA_"):
        return not setting.startswith("protocol_v2_ablation_modeA_core_")
    if setting.startswith(("protocol_v2_modeA_r", "protocol_v2_modeB_r")):
        return True
    if (setting.startswith("protocol_v2_mode") and
            ("_clip_identity_" in setting or "_resnet50_identity_" in setting)):
        return True
    return False


def _protocol_version(record):
    protocol = record.get("protocol", {})
    if isinstance(protocol, dict) and protocol.get("version") is not None:
        return protocol.get("version")
    extra = record.get("extra", {})
    if isinstance(extra, dict):
        return extra.get("protocol_version")
    return None


def _load_record(path):
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if _protocol_version(record) != 2 or not isinstance(record.get("metrics"), dict):
        return None
    return record


def _iter_json(root):
    if not os.path.isdir(root):
        return
    for directory, _, files in os.walk(root):
        for filename in sorted(files):
            if filename.endswith(".json"):
                yield os.path.join(directory, filename)


def collect_groups():
    groups = defaultdict(list)
    invalid = []
    superseded = []

    roots = [
        ("CREM", os.path.join(RESULTS_ROOT, "crem_v2")),
        ("D-CREM", os.path.join(RESULTS_ROOT, "dcrem")),
        ("baseline", os.path.join(RESULTS_ROOT, "baselines_v2")),
    ]

    for family, root in roots:
        for path in _iter_json(root) or []:
            relative = os.path.relpath(path, root).split(os.sep)
            record = _load_record(path)
            if record is None:
                invalid.append(os.path.relpath(path, REPO_ROOT))
                continue

            if family == "CREM" and len(relative) >= 3:
                method, dataset, setting = "CREM", relative[0], relative[1]
            elif family == "D-CREM" and len(relative) >= 3:
                dataset, setting = relative[0], relative[1]
                if not setting.startswith("protocol_v2_"):
                    invalid.append(os.path.relpath(path, REPO_ROOT))
                    continue
                if _is_superseded_dcrem_setting(setting):
                    superseded.append(os.path.relpath(path, REPO_ROOT))
                    continue
                method = "D-CREM"
            elif family == "baseline" and len(relative) >= 4:
                method = {"ocsvm": "OC-SVM", "iforest": "IFOREST",
                          "slan": "SLAN", "muenl_f": "MUENL-F"}.get(
                    relative[0], relative[0].upper())
                dataset, setting = relative[1], relative[2]
            else:
                invalid.append(os.path.relpath(path, REPO_ROOT))
                continue
            groups[(method, dataset, setting)].append(record)

    for records in groups.values():
        records.sort(key=lambda item: item.get("seed", -1))
    return groups, invalid, superseded


def _values(records, metric):
    values = []
    for record in records:
        value = record.get("metrics", {}).get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def _mean_std(records, metric):
    values = _values(records, metric)
    if not values:
        return "—"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.4f}"
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f"{mean:.4f} ± {math.sqrt(variance):.4f}"


def _count(groups, method, dataset, setting):
    return len(groups.get((method, dataset, setting), []))


def build_report(groups, invalid, superseded):
    valid_runs = sum(len(records) for records in groups.values())
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"> 自动生成时间：{generated}",
        ">",
        "> 这里只接收带 Protocol v2 标记的 JSON；`results/` 是可再生中间产物。",
        "",
        "### 当前结果状态",
        "",
        f"- 可纳入当前论文汇总的 Protocol v2 runs：**{valid_runs}**",
        f"- 被拒绝或损坏的 v2 路径：**{len(invalid)}**",
        f"- 设计废弃、仅保留诊断的旧消融 JSON：**{len(superseded)}**",
    ]
    if valid_runs == 0:
        lines.extend([
            "- 结论：**尚无可用于论文的正式实验结果。**",
            "- 下一步：先执行第 4 节验证，再按第 5–9 节运行实验。",
        ])

    lines.extend([
        "",
        "### 主实验完成度",
        "",
        "每格为 `已有 seed 数 / 目标 seed 数`。CREM bibtex 如最终只跑 5 seeds，",
        "需在论文表注中单独声明；这里仍以 10 作为完整矩阵目标。",
        "",
        "| Dataset | Ratio | CREM | D-CREM A | D-CREM B | OC-SVM | IFOREST | SLAN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in TABULAR_DATASETS:
        for ratio in MAIN_RATIOS:
            setting = f"known_ratio={ratio}"
            row = [
                dataset,
                str(ratio),
                f"{_count(groups, 'CREM', dataset, setting)}/{MAIN_SEEDS}",
                f"{_count(groups, 'D-CREM', dataset, f'protocol_v2_paper_core_modeA_r{ratio}')}/{MAIN_SEEDS}",
                f"{_count(groups, 'D-CREM', dataset, f'protocol_v2_paper_core_modeB_r{ratio}')}/{MAIN_SEEDS}",
                f"{_count(groups, 'OC-SVM', dataset, setting)}/{MAIN_SEEDS}",
                f"{_count(groups, 'IFOREST', dataset, setting)}/{MAIN_SEEDS}",
                f"{_count(groups, 'SLAN', dataset, setting)}/{MAIN_SEEDS}",
            ]
            lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### 已有聚合结果",
        "",
    ])
    if not groups:
        lines.append("暂无。")
    else:
        lines.extend([
            "| Method | Dataset | Setting | Seeds | AUROC | AUPR | macroAUC |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for (method, dataset, setting), records in sorted(groups.items()):
            lines.append(
                f"| {method} | {dataset} | `{setting}` | {len(records)} | "
                f"{_mean_std(records, 'AUROC')} | {_mean_std(records, 'AUPR')} | "
                f"{_mean_std(records, 'macroAUC')} |")

    if invalid:
        lines.extend([
            "",
            "### 被拒绝的文件",
            "",
            *[f"- `{path}`" for path in invalid],
        ])
    return "\n".join(lines) + "\n"


def build_summary(groups, invalid, superseded):
    """Return a machine-readable aggregate using the same accepted records."""
    aggregate = []
    for (method, dataset, setting), records in sorted(groups.items()):
        row = {
            "method": method,
            "dataset": dataset,
            "setting": setting,
            "seeds": len(records),
        }
        for metric in ("AUROC", "AUPR", "macroAUC"):
            values = _values(records, metric)
            if values:
                mean = sum(values) / len(values)
                variance = (
                    sum((value - mean) ** 2 for value in values) /
                    (len(values) - 1)
                    if len(values) > 1 else 0.0
                )
                row[metric] = {
                    "mean": mean,
                    "sample_std": math.sqrt(variance),
                    "n": len(values),
                }
        aggregate.append(row)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_version": 2,
        "valid_runs": sum(len(records) for records in groups.values()),
        "invalid_files": invalid,
        "superseded_ablation_files": superseded,
        "groups": aggregate,
    }


def write_report(markdown_path=None):
    """Write JSON summary and optionally a human-readable Markdown report.

    The return value remains a path for compatibility with experiment runners.
    """
    groups, invalid, superseded = collect_groups()
    summary = build_summary(groups, invalid, superseded)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if markdown_path:
        output_path = os.path.abspath(markdown_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(build_report(groups, invalid, superseded))
        return output_path
    return SUMMARY_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown", metavar="PATH",
        help="optionally write the full human-readable Markdown report",
    )
    args = parser.parse_args()
    groups, invalid, superseded = collect_groups()
    valid_runs = sum(len(records) for records in groups.values())
    print(
        f"Protocol v2: {valid_runs} valid runs, {len(invalid)} invalid, "
        f"{len(superseded)} superseded ablation records."
    )
    path = write_report(args.markdown)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

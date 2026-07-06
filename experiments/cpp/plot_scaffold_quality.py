#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Analyze pre-optimize scaffold-neighbor quality as a recall proxy."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
DISPLAY_NAMES = {
    "Wiki-1M": "Wiki-1M",
    "OpenAI-2M": "OpenAI-2M",
    "YFCC-10M": "YFCC-10M (uint8)",
}
COLORS = {
    "Wiki-1M": "#0072B2",
    "OpenAI-2M": "#D55E00",
    "YFCC-10M": "#009E73",
}
MARKERS = {
    "Wiki-1M": "o",
    "OpenAI-2M": "s",
    "YFCC-10M": "^",
}
INTEGER_FIELDS = {
    "parts",
    "rows",
    "queries",
    "graph_degree",
    "intermediate_graph_degree",
    "itopk",
    "scaffold_repeats",
    "scaffold_neighbors_per_leaf",
    "scaffold_first_repeat_neighbors_per_leaf",
    "scaffold_seed",
    "quality_sample_rows",
    "preopt_graph_degree_cap",
    "missing_candidates",
}
FLOAT_FIELDS = {
    "oracle_partition_build_ms_excluded",
    "merge_api_e2e_ms_instrumented",
    "quality_measurement_ms",
    "merge_api_ms_excluding_quality_measurement",
    "search_ms",
    "recall",
    "qps",
    "unique_scaffold_degree_mean",
    "preopt_candidate_rank_mean",
    "preopt_best_candidate_rank_mean",
    "preopt_top4_candidate_rank_mean",
    "preopt_fraction_rank_le_16",
    "preopt_fraction_rank_le_32",
    "preopt_fraction_rank_le_64",
    "preopt_fraction_rank_le_output_degree",
}
CORRELATION_METRICS = (
    ("all-edge mean rank", "preopt_candidate_rank_mean", -1),
    ("best-edge mean rank", "preopt_best_candidate_rank_mean", -1),
    ("best-four mean rank", "preopt_top4_candidate_rank_mean", -1),
    ("fraction rank <= 16", "preopt_fraction_rank_le_16", 1),
    ("fraction rank <= 32", "preopt_fraction_rank_le_32", 1),
    ("fraction rank <= 64", "preopt_fraction_rank_le_64", 1),
    ("count rank <= 16", "preopt_count_rank_le_16", 1),
    ("count rank <= 32", "preopt_count_rank_le_32", 1),
    ("count rank <= 64", "preopt_count_rank_le_64", 1),
    ("unique scaffold degree", "unique_scaffold_degree_mean", 1),
)
PLOT_METRICS = (
    ("preopt_candidate_rank_mean", "All scaffold edges: mean rank", True),
    ("preopt_best_candidate_rank_mean", "Best scaffold edge: mean rank", True),
    ("preopt_count_rank_le_16", "Scaffold edges in top 16 per row", False),
)


def normalize_svg(path):
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def parse_args():
    root = Path(__file__).resolve().parent
    results = root / "merge_api_results"
    plots = results / "plots"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat-input",
        type=Path,
        default=results / "scaffold_rank_quality.csv",
    )
    parser.add_argument(
        "--seed-input",
        type=Path,
        default=results / "scaffold_seed_quality.csv",
    )
    parser.add_argument(
        "--variant-input",
        type=Path,
        default=results / "scaffold_degree_quality.csv",
    )
    parser.add_argument(
        "--confirmation-input",
        type=Path,
        default=results / "scaffold_k8_confirmation.csv",
    )
    parser.add_argument(
        "--serial-cap-input",
        type=Path,
        default=results / "scaffold_k8_cap72.csv",
    )
    parser.add_argument(
        "--cap-input",
        type=Path,
        default=results / "scaffold_k8_cap72_warp.csv",
    )
    parser.add_argument(
        "--mixed-input",
        type=Path,
        default=results / "scaffold_mixed_width_quality.csv",
    )
    parser.add_argument(
        "--cap-width-input",
        type=Path,
        default=results / "scaffold_cap_width_quality.csv",
    )
    parser.add_argument(
        "--cap64-confirmation-input",
        type=Path,
        default=results / "scaffold_cap64_confirmation.csv",
    )
    parser.add_argument(
        "--cap64-repeat-input",
        type=Path,
        default=results / "scaffold_cap64_repeat_quality.csv",
    )
    parser.add_argument(
        "--derived-output",
        type=Path,
        default=results / "scaffold_quality_derived.csv",
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=results / "scaffold_quality_correlations.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=plots / "scaffold_quality_proxy.png",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=plots / "scaffold_quality_proxy.svg",
    )
    parser.add_argument(
        "--degree-summary-output",
        type=Path,
        default=results / "scaffold_degree_quality_summary.csv",
    )
    parser.add_argument(
        "--tradeoff-output",
        type=Path,
        default=plots / "scaffold_degree_tradeoff.png",
    )
    parser.add_argument(
        "--tradeoff-svg",
        type=Path,
        default=plots / "scaffold_degree_tradeoff.svg",
    )
    parser.add_argument(
        "--transfer-summary-output",
        type=Path,
        default=results / "scaffold_k8_transfer_summary.csv",
    )
    parser.add_argument(
        "--transfer-output",
        type=Path,
        default=plots / "scaffold_k8_transfer.png",
    )
    parser.add_argument(
        "--transfer-svg",
        type=Path,
        default=plots / "scaffold_k8_transfer.svg",
    )
    parser.add_argument(
        "--cap-summary-output",
        type=Path,
        default=results / "scaffold_k8_cap72_summary.csv",
    )
    parser.add_argument(
        "--cap-output",
        type=Path,
        default=plots / "scaffold_k8_cap72_transfer.png",
    )
    parser.add_argument(
        "--cap-svg",
        type=Path,
        default=plots / "scaffold_k8_cap72_transfer.svg",
    )
    parser.add_argument(
        "--mixed-summary-output",
        type=Path,
        default=results / "scaffold_mixed_width_summary.csv",
    )
    parser.add_argument(
        "--mixed-output",
        type=Path,
        default=plots / "scaffold_mixed_width_transfer.png",
    )
    parser.add_argument(
        "--mixed-svg",
        type=Path,
        default=plots / "scaffold_mixed_width_transfer.svg",
    )
    parser.add_argument(
        "--cap-kernel-summary-output",
        type=Path,
        default=results / "scaffold_cap_kernel_summary.csv",
    )
    parser.add_argument(
        "--cap-width-summary-output",
        type=Path,
        default=results / "scaffold_cap_width_summary.csv",
    )
    parser.add_argument(
        "--cap-width-output",
        type=Path,
        default=plots / "scaffold_cap_width_tradeoff.png",
    )
    parser.add_argument(
        "--cap-width-svg",
        type=Path,
        default=plots / "scaffold_cap_width_tradeoff.svg",
    )
    parser.add_argument(
        "--cap64-summary-output",
        type=Path,
        default=results / "scaffold_cap64_transfer_summary.csv",
    )
    parser.add_argument(
        "--cap64-output",
        type=Path,
        default=plots / "scaffold_cap64_transfer.png",
    )
    parser.add_argument(
        "--cap64-svg",
        type=Path,
        default=plots / "scaffold_cap64_transfer.svg",
    )
    parser.add_argument(
        "--cap64-repeat-summary-output",
        type=Path,
        default=results / "scaffold_cap64_repeat_summary.csv",
    )
    parser.add_argument(
        "--efficiency-summary-output",
        type=Path,
        default=results / "scaffold_efficiency_summary.csv",
    )
    parser.add_argument(
        "--efficiency-output",
        type=Path,
        default=plots / "scaffold_efficiency_frontier.png",
    )
    parser.add_argument(
        "--efficiency-svg",
        type=Path,
        default=plots / "scaffold_efficiency_frontier.svg",
    )
    return parser.parse_args()


def load_rows(path, source, required):
    if not path.exists():
        if required:
            raise RuntimeError(f"missing input: {path}")
        return []
    rows = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required_fields = {
            "dataset",
            "parts",
            "scaffold_repeats",
            "quality_sample_rows",
            "recall",
            "unique_scaffold_degree_mean",
            "preopt_candidate_rank_mean",
            "preopt_best_candidate_rank_mean",
            "preopt_top4_candidate_rank_mean",
            "preopt_fraction_rank_le_16",
            "preopt_fraction_rank_le_32",
            "preopt_fraction_rank_le_64",
            "missing_candidates",
        }
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path} lacks fields: {sorted(missing)}")
        for raw in reader:
            row = dict(raw)
            row["source"] = source
            row.setdefault("scaffold_neighbors_per_leaf", "4")
            row.setdefault(
                "scaffold_first_repeat_neighbors_per_leaf",
                row["scaffold_neighbors_per_leaf"],
            )
            row.setdefault("scaffold_seed", "1234")
            row.setdefault("preopt_graph_degree_cap", "0")
            for field in INTEGER_FIELDS:
                if field in row:
                    row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                if field in row:
                    row[field] = float(row[field])
            if row["missing_candidates"] != 0:
                raise RuntimeError(
                    f"{path}: missing scaffold candidates for "
                    f"{row['dataset']} parts={row['parts']}"
                )
            degree = row["unique_scaffold_degree_mean"]
            for threshold in (16, 32, 64):
                row[f"preopt_count_rank_le_{threshold}"] = (
                    degree * row[f"preopt_fraction_rank_le_{threshold}"]
                )
            row["merge_seconds"] = (
                row["merge_api_ms_excluding_quality_measurement"] / 1000.0
            )
            first_neighbors = row[
                "scaffold_first_repeat_neighbors_per_leaf"
            ]
            later_neighbors = row["scaffold_neighbors_per_leaf"]
            if first_neighbors == later_neighbors:
                row["configuration"] = (
                    f"k{later_neighbors}-r{row['scaffold_repeats']}"
                )
            else:
                row["configuration"] = (
                    f"k{first_neighbors}+"
                    f"{row['scaffold_repeats'] - 1}xk{later_neighbors}"
                )
            if row["preopt_graph_degree_cap"]:
                row["configuration"] += (
                    f"-cap{row['preopt_graph_degree_cap']}"
                )
            rows.append(row)
    return rows


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1)
        index = end
    return ranks


def pearson(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left, right):
    return pearson(rankdata(left), rankdata(right))


def correlation_rows(rows, scope, group_fields, varying_field):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for key, samples in sorted(groups.items()):
        if len(samples) < 3:
            continue
        if len({sample[varying_field] for sample in samples}) < 3:
            continue
        metadata = dict(zip(group_fields, key))
        recall = np.asarray([sample["recall"] for sample in samples])
        for display, field, direction in CORRELATION_METRICS:
            quality = direction * np.asarray([sample[field] for sample in samples])
            output.append(
                {
                    "scope": scope,
                    "dataset": metadata.get("dataset", ""),
                    "parts": metadata.get("parts", ""),
                    "scaffold_repeats": metadata.get("scaffold_repeats", ""),
                    "scaffold_neighbors_per_leaf": metadata.get(
                        "scaffold_neighbors_per_leaf", ""
                    ),
                    "scaffold_seed": metadata.get("scaffold_seed", ""),
                    "varying": varying_field,
                    "samples": len(samples),
                    "metric": display,
                    "metric_field": field,
                    "quality_direction": direction,
                    "pearson": pearson(quality, recall),
                    "spearman": spearman(quality, recall),
                }
            )
    return output


def add_aggregate_correlations(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scope"], row["metric"], row["metric_field"])].append(row)
    output = list(rows)
    for (scope, metric, field), samples in sorted(grouped.items()):
        output.append(
            {
                "scope": f"{scope}-mean",
                "dataset": "mean",
                "parts": "",
                "scaffold_repeats": "",
                "scaffold_neighbors_per_leaf": "",
                "scaffold_seed": "",
                "varying": samples[0]["varying"],
                "samples": sum(sample["samples"] for sample in samples),
                "metric": metric,
                "metric_field": field,
                "quality_direction": samples[0]["quality_direction"],
                "pearson": float(np.nanmean([sample["pearson"] for sample in samples])),
                "spearman": float(
                    np.nanmean([sample["spearman"] for sample in samples])
                ),
            }
        )
    return output


def write_correlations(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scope",
        "dataset",
        "parts",
        "scaffold_repeats",
        "scaffold_neighbors_per_leaf",
        "scaffold_seed",
        "varying",
        "samples",
        "metric",
        "metric_field",
        "quality_direction",
        "pearson",
        "spearman",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in ("pearson", "spearman"):
                output[field] = f"{row[field]:.9f}"
            writer.writerow(output)


def write_derived(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "source",
        "dataset",
        "parts",
        "implementation",
        "scaffold_repeats",
        "scaffold_neighbors_per_leaf",
        "scaffold_first_repeat_neighbors_per_leaf",
        "scaffold_seed",
        "preopt_graph_degree_cap",
        "configuration",
        "merge_seconds",
        "recall",
        "unique_scaffold_degree_mean",
        "preopt_candidate_rank_mean",
        "preopt_best_candidate_rank_mean",
        "preopt_top4_candidate_rank_mean",
        "preopt_count_rank_le_16",
        "preopt_count_rank_le_32",
        "preopt_count_rank_le_64",
        "preopt_fraction_rank_le_16",
        "preopt_fraction_rank_le_32",
        "preopt_fraction_rank_le_64",
        "quality_sample_rows",
        "missing_candidates",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extra
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_repeat_rows(rows):
    if not rows:
        return []
    max_parts = max(row["parts"] for row in rows)
    return [
        row
        for row in rows
        if row["parts"] == max_parts
        and row["scaffold_neighbors_per_leaf"] == 4
        and row["scaffold_first_repeat_neighbors_per_leaf"] == 4
        and row["preopt_graph_degree_cap"] == 0
        and row["scaffold_seed"] == 1234
    ]


def select_seed_rows(rows):
    if not rows:
        return []
    max_parts = max(row["parts"] for row in rows)
    return [
        row
        for row in rows
        if row["parts"] == max_parts
        and row["scaffold_neighbors_per_leaf"] == 4
        and row["scaffold_first_repeat_neighbors_per_leaf"] == 4
        and row["preopt_graph_degree_cap"] == 0
        and row["scaffold_repeats"] == 2
    ]


def mean_metric_correlation(correlations, scope, field, statistic):
    values = [
        row[statistic]
        for row in correlations
        if row["scope"] == scope and row["metric_field"] == field
    ]
    return float(np.nanmean(values)) if values else math.nan


def plot_proxy(path, svg_path, repeat_rows, seed_rows, correlations):
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))

    for column, (field, label, invert) in enumerate(PLOT_METRICS):
        axis = axes[0, column]
        for dataset in DATASETS:
            samples = sorted(
                (row for row in repeat_rows if row["dataset"] == dataset),
                key=lambda row: row["scaffold_repeats"],
            )
            if not samples:
                continue
            x = [row[field] for row in samples]
            y = [row["recall"] for row in samples]
            axis.plot(
                x,
                y,
                color=COLORS[dataset],
                marker=MARKERS[dataset],
                linewidth=2.0,
                markersize=6,
                label=DISPLAY_NAMES[dataset],
            )
            for row in samples:
                axis.annotate(
                    f"r{row['scaffold_repeats']}",
                    (row[field], row["recall"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=COLORS[dataset],
                )
        if invert:
            axis.invert_xaxis()
        corr = mean_metric_correlation(
            correlations, "repeat-sweep", field, "pearson"
        )
        axis.set_title(f"{label}\nmean Pearson r = {corr:.3f}")
        axis.set_xlabel(label)
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Recall@12")
    axes[0, 0].legend(frameon=False, fontsize=9)

    for column, (field, label, invert) in enumerate(PLOT_METRICS):
        axis = axes[1, column]
        if seed_rows:
            for dataset in DATASETS:
                samples = [
                    row for row in seed_rows if row["dataset"] == dataset
                ]
                if not samples:
                    continue
                x = np.asarray([row[field] for row in samples])
                y = np.asarray([row["recall"] for row in samples])
                axis.scatter(
                    x,
                    y,
                    color=COLORS[dataset],
                    marker=MARKERS[dataset],
                    s=35,
                    alpha=0.72,
                    label=DISPLAY_NAMES[dataset],
                )
                if np.ptp(x) > 0:
                    fit = np.polyfit(x, y, 1)
                    xx = np.linspace(x.min(), x.max(), 100)
                    axis.plot(xx, np.polyval(fit, xx), color=COLORS[dataset], lw=1.5)
            corr = mean_metric_correlation(
                correlations, "seed-sweep", field, "pearson"
            )
            axis.set_title(f"Fixed k4-r2 seed sweep\nmean Pearson r = {corr:.3f}")
        else:
            axis.text(
                0.5,
                0.5,
                "Fixed-budget seed sweep not present",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#666666",
            )
            axis.set_title("Fixed k4-r2 seed sweep")
        if invert:
            axis.invert_xaxis()
        axis.set_xlabel(label)
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_ylabel("Recall@12")

    fanin = max(row["parts"] for row in repeat_rows)
    fig.suptitle(
        f"Pre-optimize scaffold quality as a recall proxy ({fanin}-way merge)",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Ranks are exact within the distance-sorted merge candidate list immediately "
        "before CAGRA optimize; lower rank is better.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.045, 0.995, 0.94), h_pad=2.2, w_pad=1.6)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)


def dedupe_configuration_rows(rows):
    selected = {}
    for row in rows:
        if row["scaffold_seed"] != 1234:
            continue
        key = (
            row["dataset"],
            row["parts"],
            row["scaffold_neighbors_per_leaf"],
            row["scaffold_first_repeat_neighbors_per_leaf"],
            row["scaffold_repeats"],
            row["preopt_graph_degree_cap"],
        )
        selected[key] = row
    return list(selected.values())


def relative_ratio(numerator, denominator):
    return numerator / denominator if denominator else math.nan


def write_degree_summary(path, repeat_rows, variant_rows):
    rows = dedupe_configuration_rows(repeat_rows + variant_rows)
    lookup = {
        (
            row["dataset"],
            row["parts"],
            row["scaffold_neighbors_per_leaf"],
            row["scaffold_repeats"],
        ): row
        for row in rows
    }
    fields = (
        "dataset",
        "parts",
        "scaffold_neighbors_per_leaf",
        "scaffold_first_repeat_neighbors_per_leaf",
        "scaffold_repeats",
        "candidate_budget",
        "source",
        "merge_seconds",
        "recall",
        "preopt_count_rank_le_16",
        "preopt_best_candidate_rank_mean",
        "merge_slowdown_vs_k4_same_repeats_x",
        "recall_delta_vs_k4_same_repeats",
        "count16_delta_vs_k4_same_repeats",
        "merge_speedup_vs_k4_same_budget_x",
        "recall_delta_vs_k4_same_budget",
    )
    output = []
    for row in sorted(
        rows,
        key=lambda item: (
            DATASETS.index(item["dataset"]),
            item["parts"],
            item["scaffold_neighbors_per_leaf"],
            item["scaffold_repeats"],
        ),
    ):
        dataset = row["dataset"]
        parts = row["parts"]
        neighbors = row["scaffold_neighbors_per_leaf"]
        repeats = row["scaffold_repeats"]
        first_neighbors = row["scaffold_first_repeat_neighbors_per_leaf"]
        budget = first_neighbors + (repeats - 1) * neighbors
        same_repeats = lookup.get((dataset, parts, 4, repeats))
        same_budget = (
            lookup.get((dataset, parts, 4, budget // 4))
            if budget % 4 == 0
            else None
        )
        result = {
            "dataset": dataset,
            "parts": parts,
            "scaffold_neighbors_per_leaf": neighbors,
            "scaffold_first_repeat_neighbors_per_leaf": first_neighbors,
            "scaffold_repeats": repeats,
            "candidate_budget": budget,
            "source": row["source"],
            "merge_seconds": row["merge_seconds"],
            "recall": row["recall"],
            "preopt_count_rank_le_16": row["preopt_count_rank_le_16"],
            "preopt_best_candidate_rank_mean": row[
                "preopt_best_candidate_rank_mean"
            ],
            "merge_slowdown_vs_k4_same_repeats_x": math.nan,
            "recall_delta_vs_k4_same_repeats": math.nan,
            "count16_delta_vs_k4_same_repeats": math.nan,
            "merge_speedup_vs_k4_same_budget_x": math.nan,
            "recall_delta_vs_k4_same_budget": math.nan,
        }
        if same_repeats is not None:
            result["merge_slowdown_vs_k4_same_repeats_x"] = relative_ratio(
                row["merge_seconds"], same_repeats["merge_seconds"]
            )
            result["recall_delta_vs_k4_same_repeats"] = (
                row["recall"] - same_repeats["recall"]
            )
            result["count16_delta_vs_k4_same_repeats"] = (
                row["preopt_count_rank_le_16"]
                - same_repeats["preopt_count_rank_le_16"]
            )
        if same_budget is not None:
            result["merge_speedup_vs_k4_same_budget_x"] = relative_ratio(
                same_budget["merge_seconds"], row["merge_seconds"]
            )
            result["recall_delta_vs_k4_same_budget"] = (
                row["recall"] - same_budget["recall"]
            )
        output.append(result)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in output:
            formatted = dict(row)
            for field in fields:
                if isinstance(formatted[field], float):
                    formatted[field] = (
                        "" if math.isnan(formatted[field]) else f"{formatted[field]:.9f}"
                    )
            writer.writerow(formatted)


def pareto_rows(rows, time_field="merge_seconds", recall_field="recall"):
    output = []
    for row in rows:
        dominated = any(
            other[time_field] <= row[time_field]
            and other[recall_field] >= row[recall_field]
            and (
                other[time_field] < row[time_field]
                or other[recall_field] > row[recall_field]
            )
            for other in rows
        )
        if not dominated:
            output.append(row)
    return output


def plot_degree_tradeoff(path, svg_path, repeat_rows, variant_rows):
    rows = dedupe_configuration_rows(repeat_rows + variant_rows)
    if not variant_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    max_parts = max(row["parts"] for row in variant_rows)
    rows = [row for row in rows if row["parts"] == max_parts]
    degree_colors = {
        1: "#882255",
        2: "#44AA99",
        4: "#0072B2",
        8: "#DDCC77",
        16: "#EE7733",
        32: "#CC3311",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=False)
    for axis, dataset in zip(axes, DATASETS):
        samples = [row for row in rows if row["dataset"] == dataset]
        for neighbors in sorted({row["scaffold_neighbors_per_leaf"] for row in samples}):
            degree_rows = sorted(
                (
                    row
                    for row in samples
                    if row["scaffold_neighbors_per_leaf"] == neighbors
                ),
                key=lambda row: row["scaffold_repeats"],
            )
            axis.plot(
                [row["merge_seconds"] for row in degree_rows],
                [row["recall"] for row in degree_rows],
                color=degree_colors[neighbors],
                marker="o",
                linewidth=2.0,
                markersize=6,
                label=f"k={neighbors} per tree",
            )
            for row in degree_rows:
                axis.annotate(
                    f"r{row['scaffold_repeats']}",
                    (row["merge_seconds"], row["recall"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=degree_colors[neighbors],
                )
        frontier = pareto_rows(samples)
        axis.scatter(
            [row["merge_seconds"] for row in frontier],
            [row["recall"] for row in frontier],
            s=105,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.2,
            zorder=4,
            label="measured Pareto frontier",
        )
        defaults = [
            row
            for row in samples
            if row["scaffold_neighbors_per_leaf"] == 4
            and row["scaffold_repeats"] == 2
        ]
        if defaults:
            axis.scatter(
                defaults[0]["merge_seconds"],
                defaults[0]["recall"],
                marker="*",
                s=150,
                color="#111827",
                zorder=5,
                label="uncapped k4-r2 reference",
            )
        axis.set_xscale("log")
        axis.set_title(DISPLAY_NAMES[dataset], fontsize=12.5, fontweight="bold")
        axis.set_xlabel("Merge time (s, log scale)")
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall@12")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"Leaf neighbors vs independent trees ({max_parts}-way merge)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        "k changes neighbors selected from each already-computed leaf matrix; "
        "r changes independent pivot trees. Hollow circles are nondominated measurements.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.995, 0.84), w_pad=1.8)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)


def build_transfer_rows(rows, baseline_selector, variant_selector):
    selected = dedupe_configuration_rows(rows)
    baseline_lookup = {
        (row["dataset"], row["parts"]): row
        for row in selected
        if baseline_selector(row)
    }
    variant_lookup = {
        (row["dataset"], row["parts"]): row
        for row in selected
        if variant_selector(row)
    }
    output = []
    for dataset in DATASETS:
        parts_values = sorted(
            parts
            for candidate_dataset, parts in baseline_lookup
            if candidate_dataset == dataset
            and (dataset, parts) in variant_lookup
        )
        for parts in parts_values:
            baseline = baseline_lookup[(dataset, parts)]
            variant = variant_lookup[(dataset, parts)]
            output.append(
                {
                    "dataset": dataset,
                    "parts": parts,
                    "baseline_source": baseline["source"],
                    "variant_source": variant["source"],
                    "baseline_configuration": baseline["configuration"],
                    "variant_configuration": variant["configuration"],
                    "baseline_merge_seconds": baseline["merge_seconds"],
                    "variant_merge_seconds": variant["merge_seconds"],
                    "merge_slowdown_x": variant["merge_seconds"]
                    / baseline["merge_seconds"],
                    "baseline_recall": baseline["recall"],
                    "variant_recall": variant["recall"],
                    "recall_delta": variant["recall"] - baseline["recall"],
                    "baseline_count_rank_le_16": baseline[
                        "preopt_count_rank_le_16"
                    ],
                    "variant_count_rank_le_16": variant[
                        "preopt_count_rank_le_16"
                    ],
                    "count_rank_le_16_delta": variant[
                        "preopt_count_rank_le_16"
                    ]
                    - baseline["preopt_count_rank_le_16"],
                }
            )
    return output


def write_transfer_summary(path, rows):
    fields = (
        "dataset",
        "parts",
        "baseline_source",
        "variant_source",
        "baseline_configuration",
        "variant_configuration",
        "baseline_merge_seconds",
        "variant_merge_seconds",
        "merge_slowdown_x",
        "baseline_recall",
        "variant_recall",
        "recall_delta",
        "baseline_count_rank_le_16",
        "variant_count_rank_le_16",
        "count_rank_le_16_delta",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in fields:
                if isinstance(output[field], float):
                    output[field] = f"{output[field]:.9f}"
            writer.writerow(output)


def plot_k8_transfer(
    path,
    svg_path,
    rows,
    title="k8 vs k4 at two independent trees: transfer across fan-in",
    note="k8 reuses each tree's existing leaf distance matrix.",
):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("recall_delta", "Recall@12 gain", 0.0),
        ("merge_slowdown_x", "Merge-time ratio", 1.0),
        (
            "count_rank_le_16_delta",
            "Additional scaffold edges in top 16 / row",
            0.0,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    for axis, (field, label, reference) in zip(axes, metrics):
        for dataset in DATASETS:
            samples = sorted(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: row["parts"],
            )
            if not samples:
                continue
            axis.plot(
                [row["parts"] for row in samples],
                [row[field] for row in samples],
                color=COLORS[dataset],
                marker=MARKERS[dataset],
                linewidth=2.0,
                markersize=6,
                label=DISPLAY_NAMES[dataset],
            )
        axis.axhline(reference, color="#6b7280", linestyle="--", linewidth=1.2)
        axis.set_xscale("log", base=2)
        parts = sorted({row["parts"] for row in rows})
        axis.set_xticks(parts, [str(part) for part in parts])
        axis.set_xlabel("Number of input graphs")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        note,
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.995, 0.93), w_pad=1.8)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)



def build_cap_kernel_rows(serial_rows, optimized_rows):
    def select(rows, neighbors):
        return {
            (row["dataset"], row["parts"]): row
            for row in dedupe_configuration_rows(rows)
            if row["scaffold_repeats"] == 2
            and row["scaffold_neighbors_per_leaf"] == neighbors
            and row["scaffold_first_repeat_neighbors_per_leaf"] == neighbors
            and row["preopt_graph_degree_cap"] == 72
        }

    serial_baseline = select(serial_rows, 4)
    serial_variant = select(serial_rows, 8)
    warp_baseline = select(optimized_rows, 4)
    warp_variant = select(optimized_rows, 8)
    common = (
        set(serial_baseline)
        & set(serial_variant)
        & set(warp_baseline)
        & set(warp_variant)
    )
    output = []
    for dataset in DATASETS:
        for parts in sorted(
            part
            for candidate_dataset, part in common
            if candidate_dataset == dataset
        ):
            key = (dataset, parts)
            old_base = serial_baseline[key]
            old_variant = serial_variant[key]
            new_base = warp_baseline[key]
            new_variant = warp_variant[key]
            serial_excess = (
                old_variant["merge_seconds"] - old_base["merge_seconds"]
            )
            warp_excess = (
                new_variant["merge_seconds"] - new_base["merge_seconds"]
            )
            output.append(
                {
                    "dataset": dataset,
                    "parts": parts,
                    "serial_baseline_seconds": old_base["merge_seconds"],
                    "serial_variant_seconds": old_variant["merge_seconds"],
                    "serial_premium_x": old_variant["merge_seconds"]
                    / old_base["merge_seconds"],
                    "warp_baseline_seconds": new_base["merge_seconds"],
                    "warp_variant_seconds": new_variant["merge_seconds"],
                    "warp_premium_x": new_variant["merge_seconds"]
                    / new_base["merge_seconds"],
                    "serial_excess_seconds": serial_excess,
                    "warp_excess_seconds": warp_excess,
                    "excess_cost_reduction_x": serial_excess / warp_excess
                    if warp_excess > 0
                    else math.nan,
                    "serial_recall_gain": old_variant["recall"]
                    - old_base["recall"],
                    "warp_recall_gain": new_variant["recall"]
                    - new_base["recall"],
                    "variant_recall_delta_between_kernels": new_variant[
                        "recall"
                    ]
                    - old_variant["recall"],
                }
            )
    return output


def write_cap_kernel_summary(path, rows):
    fields = (
        "dataset",
        "parts",
        "serial_baseline_seconds",
        "serial_variant_seconds",
        "serial_premium_x",
        "warp_baseline_seconds",
        "warp_variant_seconds",
        "warp_premium_x",
        "serial_excess_seconds",
        "warp_excess_seconds",
        "excess_cost_reduction_x",
        "serial_recall_gain",
        "warp_recall_gain",
        "variant_recall_delta_between_kernels",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if isinstance(row[field], float)
                        and math.isnan(row[field])
                        else f"{row[field]:.9f}"
                        if isinstance(row[field], float)
                        else row[field]
                    )
                    for field in fields
                }
            )


def build_cap_width_rows(rows):
    selected = [
        row
        for row in dedupe_configuration_rows(rows)
        if row["scaffold_repeats"] == 2
        and row["scaffold_first_repeat_neighbors_per_leaf"]
        == row["scaffold_neighbors_per_leaf"]
        and row["scaffold_neighbors_per_leaf"] in (4, 8)
    ]
    if not selected:
        return []
    max_parts = max(row["parts"] for row in selected)
    selected = [row for row in selected if row["parts"] == max_parts]
    output = []
    for dataset in DATASETS:
        samples = [row for row in selected if row["dataset"] == dataset]
        baselines = [
            row
            for row in samples
            if row["scaffold_neighbors_per_leaf"] == 4
            and row["preopt_graph_degree_cap"] == 0
        ]
        if not baselines:
            continue
        baseline = baselines[0]
        dataset_rows = []
        for row in samples:
            neighbors = row["scaffold_neighbors_per_leaf"]
            cap = row["preopt_graph_degree_cap"]
            candidate_degree = 64 + 2 * neighbors
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "parts": max_parts,
                    "scaffold_neighbors_per_leaf": neighbors,
                    "preopt_graph_degree_cap": cap,
                    "retained_degree": cap if cap else candidate_degree,
                    "configuration": f"k{neighbors}-"
                    + (f"cap{cap}" if cap else "uncapped"),
                    "merge_seconds": row["merge_seconds"],
                    "merge_slowdown_vs_k4_uncapped_x": row["merge_seconds"]
                    / baseline["merge_seconds"],
                    "recall": row["recall"],
                    "recall_delta_vs_k4_uncapped": row["recall"]
                    - baseline["recall"],
                    "preopt_count_rank_le_16": row[
                        "preopt_count_rank_le_16"
                    ],
                    "pareto": False,
                }
            )
        frontier_ids = {id(row) for row in pareto_rows(dataset_rows)}
        for row in dataset_rows:
            row["pareto"] = id(row) in frontier_ids
        output.extend(dataset_rows)
    return sorted(
        output,
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            row["scaffold_neighbors_per_leaf"],
            row["retained_degree"],
            row["preopt_graph_degree_cap"],
        ),
    )


def write_cap_width_summary(path, rows):
    fields = (
        "dataset",
        "parts",
        "scaffold_neighbors_per_leaf",
        "preopt_graph_degree_cap",
        "retained_degree",
        "configuration",
        "merge_seconds",
        "merge_slowdown_vs_k4_uncapped_x",
        "recall",
        "recall_delta_vs_k4_uncapped",
        "preopt_count_rank_le_16",
        "pareto",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in fields:
                if isinstance(output[field], float):
                    output[field] = f"{output[field]:.9f}"
            writer.writerow(output)


def plot_cap_width(path, svg_path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {4: "#0072B2", 8: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    for axis, dataset in zip(axes, DATASETS):
        samples = [row for row in rows if row["dataset"] == dataset]
        for neighbors in (4, 8):
            degree_rows = sorted(
                (
                    row
                    for row in samples
                    if row["scaffold_neighbors_per_leaf"] == neighbors
                ),
                key=lambda row: (
                    row["retained_degree"],
                    row["preopt_graph_degree_cap"],
                ),
            )
            if not degree_rows:
                continue
            axis.plot(
                [row["merge_seconds"] for row in degree_rows],
                [row["recall"] for row in degree_rows],
                color=colors[neighbors],
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=f"k{neighbors} scaffold",
            )
            for row in degree_rows:
                cap = row["preopt_graph_degree_cap"]
                axis.annotate(
                    f"cap {cap}" if cap else "full",
                    (row["merge_seconds"], row["recall"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.2,
                    color=colors[neighbors],
                )
        frontier = [row for row in samples if row["pareto"]]
        axis.scatter(
            [row["merge_seconds"] for row in frontier],
            [row["recall"] for row in frontier],
            s=125,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.2,
            zorder=5,
            label="measured Pareto frontier",
        )
        defaults = [
            row
            for row in samples
            if row["scaffold_neighbors_per_leaf"] == 4
            and row["preopt_graph_degree_cap"] == 0
        ]
        if defaults:
            axis.scatter(
                defaults[0]["merge_seconds"],
                defaults[0]["recall"],
                marker="*",
                s=145,
                color="#111827",
                zorder=6,
                label="uncapped k4-r2 reference",
            )
        axis.set_title(DISPLAY_NAMES[dataset], fontsize=12.5, fontweight="bold")
        axis.set_xlabel("Merge time (s)")
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall@12")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fanin = max(row["parts"] for row in rows)
    fig.suptitle(
        f"Ranked unique candidate-cap sweep ({fanin}-way merge)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        "Each cap keeps the nearest unique IDs after exact candidate sorting; all points share "
        "the same input graphs within a dataset.",
        ha="center",
        fontsize=9.3,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.995, 0.83), w_pad=1.8)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)


def build_cap64_transfer_rows(trial_sets):
    trials = []
    for source, source_rows in trial_sets:
        selected = [
            row
            for row in dedupe_configuration_rows(source_rows)
            if row["scaffold_repeats"] == 2
            and row["scaffold_first_repeat_neighbors_per_leaf"]
            == row["scaffold_neighbors_per_leaf"]
            and row["scaffold_neighbors_per_leaf"] in (4, 8)
            and row["preopt_graph_degree_cap"] in (0, 64)
        ]
        lookup = {
            (
                row["dataset"],
                row["parts"],
                row["scaffold_neighbors_per_leaf"],
                row["preopt_graph_degree_cap"],
            ): row
            for row in selected
        }
        for dataset in DATASETS:
            parts_values = sorted(
                {
                    parts
                    for candidate_dataset, parts, neighbors, cap in lookup
                    if candidate_dataset == dataset
                    and neighbors == 4
                    and cap == 0
                }
            )
            for parts in parts_values:
                baseline = lookup[(dataset, parts, 4, 0)]
                for strategy, neighbors in (
                    ("k4-cap64", 4),
                    ("k8-cap64", 8),
                ):
                    variant = lookup.get((dataset, parts, neighbors, 64))
                    if variant is None:
                        continue
                    trials.append(
                        {
                            "source": source,
                            "dataset": dataset,
                            "parts": parts,
                            "strategy": strategy,
                            "merge_ratio": variant["merge_seconds"]
                            / baseline["merge_seconds"],
                            "recall_delta": variant["recall"]
                            - baseline["recall"],
                            "qps_ratio": variant["qps"] / baseline["qps"],
                            "top16_delta": variant[
                                "preopt_count_rank_le_16"
                            ]
                            - baseline["preopt_count_rank_le_16"],
                        }
                    )

    grouped = defaultdict(list)
    for row in trials:
        grouped[(row["dataset"], row["parts"], row["strategy"])].append(
            row
        )
    output = []
    for (dataset, parts, strategy), rows in grouped.items():
        result = {
            "dataset": dataset,
            "parts": parts,
            "strategy": strategy,
            "trials": len(rows),
            "sources": ";".join(sorted(row["source"] for row in rows)),
        }
        for metric in (
            "merge_ratio",
            "recall_delta",
            "qps_ratio",
            "top16_delta",
        ):
            values = [row[metric] for row in rows]
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        output.append(result)
    return sorted(
        output,
        key=lambda row: (
            row["strategy"],
            DATASETS.index(row["dataset"]),
            row["parts"],
        ),
    )


def write_cap64_transfer_summary(path, rows):
    fields = (
        "dataset",
        "parts",
        "strategy",
        "trials",
        "sources",
        "merge_ratio_mean",
        "merge_ratio_min",
        "merge_ratio_max",
        "recall_delta_mean",
        "recall_delta_min",
        "recall_delta_max",
        "qps_ratio_mean",
        "qps_ratio_min",
        "qps_ratio_max",
        "top16_delta_mean",
        "top16_delta_min",
        "top16_delta_max",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in fields:
                if isinstance(output[field], float):
                    output[field] = f"{output[field]:.9f}"
            writer.writerow(output)


def plot_cap64_transfer(path, svg_path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    strategies = ("k4-cap64", "k8-cap64")
    metrics = (
        ("recall_delta", "Recall@12 gain", 0.0),
        ("merge_ratio", "Merge-time ratio", 1.0),
        ("qps_ratio", "Query-throughput ratio", 1.0),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0), sharex="col")
    for row_index, strategy in enumerate(strategies):
        for column, (metric, label, reference) in enumerate(metrics):
            axis = axes[row_index, column]
            for dataset in DATASETS:
                samples = sorted(
                    (
                        row
                        for row in rows
                        if row["strategy"] == strategy
                        and row["dataset"] == dataset
                    ),
                    key=lambda row: row["parts"],
                )
                if not samples:
                    continue
                x = [row["parts"] for row in samples]
                mean = [row[f"{metric}_mean"] for row in samples]
                low = [row[f"{metric}_min"] for row in samples]
                high = [row[f"{metric}_max"] for row in samples]
                axis.plot(
                    x,
                    mean,
                    color=COLORS[dataset],
                    marker=MARKERS[dataset],
                    linewidth=2.0,
                    markersize=5.5,
                    label=DISPLAY_NAMES[dataset],
                )
                axis.fill_between(
                    x, low, high, color=COLORS[dataset], alpha=0.14
                )
            axis.axhline(
                reference, color="#6b7280", linestyle="--", linewidth=1.2
            )
            axis.set_xscale("log", base=2)
            parts = sorted({row["parts"] for row in rows})
            axis.set_xticks(parts, [str(part) for part in parts])
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(label, fontsize=12, fontweight="bold")
            if row_index == 1:
                axis.set_xlabel("Number of input graphs")
        axes[row_index, 0].text(
            -0.31,
            0.5,
            strategy,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Cap64 transfer with independent input-graph builds",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Lines are means; bands span the independent trials. Ratios and recall deltas use the "
        "same-process uncapped k4-r2 control.",
        ha="center",
        fontsize=9.3,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.05, 0.995, 0.90), h_pad=1.8, w_pad=1.5)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)


def build_efficiency_rows(repeat_rows, variant_rows, cap_rows, mixed_rows, cap_repeat_rows):
    uniform = [
        row
        for row in dedupe_configuration_rows(repeat_rows + variant_rows + cap_repeat_rows)
        if row["preopt_graph_degree_cap"] == 0
        and row["scaffold_first_repeat_neighbors_per_leaf"]
        == row["scaffold_neighbors_per_leaf"]
    ]
    special = [
        row
        for row in dedupe_configuration_rows(cap_rows + cap_repeat_rows + mixed_rows)
        if (
            row["preopt_graph_degree_cap"] == 64
            and row["scaffold_neighbors_per_leaf"] in (4, 8)
            and row["scaffold_first_repeat_neighbors_per_leaf"]
            == row["scaffold_neighbors_per_leaf"]
        )
        or (
            row["preopt_graph_degree_cap"] == 0
            and {
                row["scaffold_neighbors_per_leaf"],
                row["scaffold_first_repeat_neighbors_per_leaf"],
            }
            == {4, 8}
            and row["scaffold_repeats"] == 2
        )
    ]
    matched_control_rows = dedupe_configuration_rows(
        cap_rows + cap_repeat_rows + mixed_rows
    )
    matched_baselines = {
        (row["source"], row["dataset"], row["parts"]): row
        for row in matched_control_rows
        if row["scaffold_neighbors_per_leaf"] == 4
        and row["scaffold_first_repeat_neighbors_per_leaf"] == 4
        and row["scaffold_repeats"] == 2
        and row["preopt_graph_degree_cap"] == 0
    }
    matched_controls = {
        (
            row["source"],
            row["dataset"],
            row["parts"],
            row["scaffold_neighbors_per_leaf"],
            row["scaffold_first_repeat_neighbors_per_leaf"],
            row["scaffold_repeats"],
        ): row
        for row in matched_control_rows
        if row["preopt_graph_degree_cap"] == 0
        and row["scaffold_first_repeat_neighbors_per_leaf"]
        == row["scaffold_neighbors_per_leaf"]
    }
    candidates = uniform + special
    if not candidates:
        return []
    max_parts = max(row["parts"] for row in candidates)
    candidates = [row for row in candidates if row["parts"] == max_parts]
    output = []
    for dataset in DATASETS:
        samples = [row for row in candidates if row["dataset"] == dataset]
        baselines = [
            row
            for row in samples
            if row["scaffold_neighbors_per_leaf"] == 4
            and row["scaffold_first_repeat_neighbors_per_leaf"] == 4
            and row["scaffold_repeats"] == 2
            and row["preopt_graph_degree_cap"] == 0
        ]
        if not baselines:
            continue
        baseline = baselines[0]
        uniform_lookup = {
            (
                row["scaffold_neighbors_per_leaf"],
                row["scaffold_repeats"],
            ): row
            for row in samples
            if row["preopt_graph_degree_cap"] == 0
            and row["scaffold_first_repeat_neighbors_per_leaf"]
            == row["scaffold_neighbors_per_leaf"]
        }
        dataset_rows = []
        for row in samples:
            neighbors = row["scaffold_neighbors_per_leaf"]
            first_neighbors = row["scaffold_first_repeat_neighbors_per_leaf"]
            repeats = row["scaffold_repeats"]
            cap = row["preopt_graph_degree_cap"]
            if cap:
                strategy = f"k{neighbors}-r{repeats} + unique cap{cap}"
                family = f"capped-k{neighbors}"
            elif first_neighbors > neighbors:
                strategy = "k8+k4 mixed (wide first)"
                family = "mixed-first"
            elif first_neighbors < neighbors:
                strategy = "k4+k8 mixed (wide second)"
                family = "mixed-second"
            else:
                strategy = f"k{neighbors}-r{repeats}"
                family = "uniform"
            if family.startswith("capped-"):
                matched_control = matched_controls.get(
                    (
                        row["source"],
                        row["dataset"],
                        row["parts"],
                        neighbors,
                        first_neighbors,
                        repeats,
                    )
                )
                transport_anchor = uniform_lookup.get(
                    (neighbors, repeats), matched_control or baseline
                )
                comparison_baseline = matched_control or transport_anchor
            elif family != "uniform":
                comparison_baseline = matched_baselines.get(
                    (row["source"], row["dataset"], row["parts"]), baseline
                )
                transport_anchor = baseline
            else:
                comparison_baseline = row
                transport_anchor = row
            normalized_merge_seconds = transport_anchor["merge_seconds"] * (
                row["merge_seconds"] / comparison_baseline["merge_seconds"]
            )
            normalized_recall = transport_anchor["recall"] + (
                row["recall"] - comparison_baseline["recall"]
            )
            extra_seconds = normalized_merge_seconds - baseline["merge_seconds"]
            recall_gain = normalized_recall - baseline["recall"]
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "parts": max_parts,
                    "strategy": strategy,
                    "family": family,
                    "scaffold_repeats": repeats,
                    "scaffold_neighbors_per_leaf": neighbors,
                    "scaffold_first_repeat_neighbors_per_leaf": first_neighbors,
                    "preopt_graph_degree_cap": cap,
                    "candidate_budget": first_neighbors
                    + (repeats - 1) * neighbors,
                    "comparison_baseline_source": comparison_baseline[
                        "source"
                    ],
                    "merge_seconds": row["merge_seconds"],
                    "recall": row["recall"],
                    "normalized_merge_seconds": normalized_merge_seconds,
                    "normalized_recall": normalized_recall,
                    "preopt_count_rank_le_16": row[
                        "preopt_count_rank_le_16"
                    ],
                    "merge_slowdown_vs_k4_r2_x": normalized_merge_seconds
                    / baseline["merge_seconds"],
                    "extra_merge_seconds_vs_k4_r2": extra_seconds,
                    "recall_delta_vs_k4_r2": recall_gain,
                    "top16_delta_vs_k4_r2": row["preopt_count_rank_le_16"]
                    - baseline["preopt_count_rank_le_16"],
                    "recall_gain_per_extra_second": recall_gain / extra_seconds
                    if extra_seconds > 0
                    else math.nan,
                    "pareto": False,
                }
            )
        frontier_ids = {
            id(row)
            for row in pareto_rows(
                dataset_rows,
                time_field="normalized_merge_seconds",
                recall_field="normalized_recall",
            )
        }
        for row in dataset_rows:
            row["pareto"] = id(row) in frontier_ids
        output.extend(dataset_rows)
    return sorted(
        output,
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            row["merge_seconds"],
            row["strategy"],
        ),
    )


def write_efficiency_summary(path, rows):
    fields = (
        "dataset",
        "parts",
        "strategy",
        "family",
        "scaffold_repeats",
        "scaffold_neighbors_per_leaf",
        "scaffold_first_repeat_neighbors_per_leaf",
        "preopt_graph_degree_cap",
        "candidate_budget",
        "comparison_baseline_source",
        "merge_seconds",
        "recall",
        "normalized_merge_seconds",
        "normalized_recall",
        "preopt_count_rank_le_16",
        "merge_slowdown_vs_k4_r2_x",
        "extra_merge_seconds_vs_k4_r2",
        "recall_delta_vs_k4_r2",
        "top16_delta_vs_k4_r2",
        "recall_gain_per_extra_second",
        "pareto",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in fields:
                if isinstance(output[field], float):
                    output[field] = (
                        "" if math.isnan(output[field]) else f"{output[field]:.9f}"
                    )
            writer.writerow(output)


def plot_efficiency(path, svg_path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    degree_colors = {
        1: "#882255",
        2: "#44AA99",
        4: "#0072B2",
        8: "#DDCC77",
        16: "#EE7733",
        32: "#CC3311",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    for axis, dataset in zip(axes, DATASETS):
        samples = [row for row in rows if row["dataset"] == dataset]
        for neighbors in (1, 2, 4, 8, 16, 32):
            uniform = sorted(
                (
                    row
                    for row in samples
                    if row["family"] == "uniform"
                    and row["scaffold_neighbors_per_leaf"] == neighbors
                ),
                key=lambda row: row["scaffold_repeats"],
            )
            if not uniform:
                continue
            axis.plot(
                [row["normalized_merge_seconds"] for row in uniform],
                [row["normalized_recall"] for row in uniform],
                color=degree_colors[neighbors],
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                label=f"uniform k{neighbors}",
            )
            for row in uniform:
                axis.annotate(
                    f"r{row['scaffold_repeats']}",
                    (row["normalized_merge_seconds"], row["normalized_recall"]),
                    xytext=(3, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=degree_colors[neighbors],
                )
        for family, marker, color, label in (
            ("mixed-first", "P", "#CC79A7", "mixed k8 then k4"),
            ("mixed-second", "D", "#56B4E9", "mixed k4 then k8"),
            ("capped-k4", "*", "#6B7280", "k4 + unique cap64"),
            ("capped-k8", "X", "#111827", "k8 + unique cap64"),
        ):
            points = sorted(
                [row for row in samples if row["family"] == family],
                key=lambda row: row["scaffold_repeats"],
            )
            if not points:
                continue
            if family.startswith("capped-"):
                axis.plot(
                    [row["normalized_merge_seconds"] for row in points],
                    [row["normalized_recall"] for row in points],
                    color=color,
                    linestyle="--",
                    marker=marker,
                    linewidth=1.8,
                    markersize=8,
                    zorder=5,
                    label=label,
                )
                for row in points:
                    axis.annotate(
                        f"r{row['scaffold_repeats']}",
                        (row["normalized_merge_seconds"], row["normalized_recall"]),
                        xytext=(
                            (-12, -11) if family == "capped-k4" else (4, -11)
                        ),
                        ha="right" if family == "capped-k4" else "left",
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )
            else:
                axis.scatter(
                    [row["normalized_merge_seconds"] for row in points],
                    [row["normalized_recall"] for row in points],
                    marker=marker,
                    s=95,
                    color=color,
                    zorder=5,
                    label=label,
                )
        frontier = [row for row in samples if row["pareto"]]
        axis.scatter(
            [row["normalized_merge_seconds"] for row in frontier],
            [row["normalized_recall"] for row in frontier],
            s=130,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.2,
            zorder=6,
            label="measured Pareto frontier",
        )
        axis.set_xscale("log")
        axis.set_title(DISPLAY_NAMES[dataset], fontsize=12.5, fontweight="bold")
        axis.set_xlabel("Merge time (s, matched-control normalized; log scale)")
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall@12")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fanin = max(row["parts"] for row in rows)
    fig.suptitle(
        f"Scaffold recall / merge-time frontier ({fanin}-way merge)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        "Cap64 series use same-process uncapped controls; hollow circles are nondominated. "
        "Mixed schedules reuse a k8 leaf head without another tree.",
        ha="center",
        fontsize=9.3,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.995, 0.82), w_pad=1.8)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)

def main():
    args = parse_args()
    repeat_rows = load_rows(args.repeat_input, "repeat", required=True)
    seed_rows = load_rows(args.seed_input, "seed", required=False)
    variant_rows = load_rows(args.variant_input, "variant", required=False)
    confirmation_rows = load_rows(
        args.confirmation_input, "confirmation", required=False
    )
    serial_cap_rows = load_rows(
        args.serial_cap_input, "cap72-serial", required=False
    )
    cap_rows = load_rows(args.cap_input, "cap72-warp", required=False)
    mixed_rows = load_rows(args.mixed_input, "mixed", required=False)
    cap_width_rows = load_rows(
        args.cap_width_input, "cap-width", required=False
    )
    cap64_confirmation_rows = load_rows(
        args.cap64_confirmation_input, "cap64-confirmation", required=False
    )
    cap64_repeat_rows = load_rows(
        args.cap64_repeat_input, "cap64-repeat", required=False
    )
    all_rows = (
        repeat_rows
        + seed_rows
        + variant_rows
        + confirmation_rows
        + serial_cap_rows
        + cap_rows
        + mixed_rows
        + cap_width_rows
        + cap64_confirmation_rows
        + cap64_repeat_rows
    )
    write_derived(args.derived_output, all_rows)
    write_cap64_repeat_summary(
        args.cap64_repeat_summary_output, cap64_repeat_rows
    )

    correlations = correlation_rows(
        repeat_rows,
        "repeat-sweep",
        (
            "dataset",
            "parts",
            "scaffold_neighbors_per_leaf",
            "scaffold_first_repeat_neighbors_per_leaf",
            "scaffold_seed",
        ),
        "scaffold_repeats",
    )
    correlations += correlation_rows(
        seed_rows,
        "seed-sweep",
        (
            "dataset",
            "parts",
            "scaffold_repeats",
            "scaffold_neighbors_per_leaf",
            "scaffold_first_repeat_neighbors_per_leaf",
        ),
        "scaffold_seed",
    )
    correlations = add_aggregate_correlations(correlations)

    def two_tree(row, neighbors, first_neighbors, cap):
        return (
            row["scaffold_repeats"] == 2
            and row["scaffold_neighbors_per_leaf"] == neighbors
            and row["scaffold_first_repeat_neighbors_per_leaf"]
            == first_neighbors
            and row["preopt_graph_degree_cap"] == cap
        )

    transfer_rows = build_transfer_rows(
        repeat_rows + variant_rows + confirmation_rows,
        lambda row: two_tree(row, 4, 4, 0),
        lambda row: two_tree(row, 8, 8, 0),
    )
    write_transfer_summary(args.transfer_summary_output, transfer_rows)
    plot_k8_transfer(
        args.transfer_output,
        args.transfer_svg,
        transfer_rows,
        note=(
            "k8 widens candidate selection from each existing leaf matrix; "
            "2/8-way points use matched oracle inputs."
        ),
    )

    cap_transfer_rows = build_transfer_rows(
        cap_rows,
        lambda row: two_tree(row, 4, 4, 72),
        lambda row: two_tree(row, 8, 8, 72),
    )
    write_transfer_summary(args.cap_summary_output, cap_transfer_rows)
    plot_k8_transfer(
        args.cap_output,
        args.cap_svg,
        cap_transfer_rows,
        title="Ranked k8-cap72 vs k4: transfer across fan-in",
        note=(
            "Both variants use two trees and a 72-wide optimizer input; "
            "a warp retains nearest unique candidates after full sorting."
        ),
    )

    mixed_first_rows = build_transfer_rows(
        mixed_rows,
        lambda row: two_tree(row, 4, 4, 0),
        lambda row: two_tree(row, 4, 8, 0),
    )
    mixed_second_rows = build_transfer_rows(
        mixed_rows,
        lambda row: two_tree(row, 4, 4, 0),
        lambda row: two_tree(row, 8, 4, 0),
    )
    write_transfer_summary(
        args.mixed_summary_output, mixed_first_rows + mixed_second_rows
    )
    plot_k8_transfer(
        args.mixed_output,
        args.mixed_svg,
        mixed_first_rows,
        title="Mixed k8+k4 vs uniform k4+k4: transfer across fan-in",
        note=(
            "Only the first tree widens its existing leaf selection to k8; "
            "the raw summary also records the reverse k4+k8 order."
        ),
    )

    cap_kernel_rows = build_cap_kernel_rows(serial_cap_rows, cap_rows)
    write_cap_kernel_summary(args.cap_kernel_summary_output, cap_kernel_rows)
    cap_width_summary = build_cap_width_rows(cap_width_rows)
    write_cap_width_summary(args.cap_width_summary_output, cap_width_summary)
    plot_cap_width(
        args.cap_width_output, args.cap_width_svg, cap_width_summary
    )
    cap64_transfer_rows = build_cap64_transfer_rows(
        (
            ("cap-width", cap_width_rows),
            ("cap64-confirmation", cap64_confirmation_rows),
        )
    )
    write_cap64_transfer_summary(
        args.cap64_summary_output, cap64_transfer_rows
    )
    plot_cap64_transfer(
        args.cap64_output, args.cap64_svg, cap64_transfer_rows
    )
    efficiency_rows = build_efficiency_rows(
        repeat_rows, variant_rows, cap_width_rows, mixed_rows, cap64_repeat_rows
    )
    write_efficiency_summary(args.efficiency_summary_output, efficiency_rows)
    plot_efficiency(args.efficiency_output, args.efficiency_svg, efficiency_rows)

    write_correlations(args.correlation_output, correlations)
    plot_proxy(
        args.output,
        args.svg,
        select_repeat_rows(repeat_rows),
        select_seed_rows(seed_rows),
        correlations,
    )
    write_degree_summary(
        args.degree_summary_output, repeat_rows, variant_rows
    )
    plot_degree_tradeoff(
        args.tradeoff_output, args.tradeoff_svg, repeat_rows, variant_rows
    )


def write_cap64_repeat_summary(path, rows):
    def config_key(row):
        return (
            row["dataset"],
            row["parts"],
            row["scaffold_neighbors_per_leaf"],
            row["scaffold_first_repeat_neighbors_per_leaf"],
            row["scaffold_repeats"],
            row["scaffold_seed"],
        )

    controls = {
        config_key(row): row
        for row in rows
        if row["preopt_graph_degree_cap"] == 0
        and row["scaffold_first_repeat_neighbors_per_leaf"]
        == row["scaffold_neighbors_per_leaf"]
    }
    output = []
    for row in rows:
        if row["preopt_graph_degree_cap"] != 64:
            continue
        control = controls.get(config_key(row))
        if control is None:
            continue
        output.append(
            {
                "dataset": row["dataset"],
                "parts": row["parts"],
                "scaffold_neighbors_per_leaf": row[
                    "scaffold_neighbors_per_leaf"
                ],
                "scaffold_repeats": row["scaffold_repeats"],
                "candidate_budget": row["scaffold_repeats"]
                * row["scaffold_neighbors_per_leaf"],
                "uncapped_merge_seconds": control["merge_seconds"],
                "cap64_merge_seconds": row["merge_seconds"],
                "merge_time_ratio": row["merge_seconds"]
                / control["merge_seconds"],
                "merge_time_reduction_pct": 100.0
                * (1.0 - row["merge_seconds"] / control["merge_seconds"]),
                "uncapped_recall": control["recall"],
                "cap64_recall": row["recall"],
                "recall_delta": row["recall"] - control["recall"],
                "uncapped_qps": control["qps"],
                "cap64_qps": row["qps"],
                "qps_ratio": row["qps"] / control["qps"],
            }
        )
    output.sort(
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            row["scaffold_neighbors_per_leaf"],
            row["scaffold_repeats"],
        )
    )
    fields = (
        "dataset",
        "parts",
        "scaffold_neighbors_per_leaf",
        "scaffold_repeats",
        "candidate_budget",
        "uncapped_merge_seconds",
        "cap64_merge_seconds",
        "merge_time_ratio",
        "merge_time_reduction_pct",
        "uncapped_recall",
        "cap64_recall",
        "recall_delta",
        "uncapped_qps",
        "cap64_qps",
        "qps_ratio",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in output:
            writer.writerow(
                {
                    field: f"{row[field]:.9f}"
                    if isinstance(row[field], float)
                    else row[field]
                    for field in fields
                }
            )


if __name__ == "__main__":
    main()

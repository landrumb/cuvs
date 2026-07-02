# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot the 8-way CAGRA k=4 merge leaf-size sweep."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
LEAF_SIZES = (64, 128, 256, 512, 1024)
MARKERS = {"Wiki-1M": "o", "OpenAI-2M": "s", "YFCC-10M": "^"}
PRODUCTION_LEAF_SIZES = (64, 128, 256)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "merge_api_results" / "leaf_size_8way.csv",
    )
    parser.add_argument(
        "--production-input",
        type=Path,
        default=root / "merge_api_results" / "leaf_size_production_8way.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "merge_api_results" / "plots" / "k4_leaf_size_8way.png",
    )
    return parser.parse_args()


def load_results(path):
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["parts"]) != 8:
                continue
            key = (row["dataset"], int(row["leaf_size"]))
            groups[key].append(
                (float(row["merge_api_e2e_ms"]), float(row["recall"]))
            )

    expected = {(dataset, leaf) for dataset in DATASETS for leaf in LEAF_SIZES}
    if set(groups) != expected:
        raise RuntimeError(
            f"coverage mismatch: missing={sorted(expected - set(groups))} "
            f"extra={sorted(set(groups) - expected)}"
        )
    wrong_counts = {key: len(values) for key, values in groups.items() if len(values) != 2}
    if wrong_counts:
        raise RuntimeError(f"expected two runs per point: {wrong_counts}")
    return groups


def load_production(path):
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["parts"]) != 8:
                continue
            key = (row["dataset"], int(row["leaf_size"]))
            groups[key].append(
                (float(row["merge_api_e2e_ms"]), float(row["recall"]))
            )

    expected = {
        (dataset, leaf) for dataset in DATASETS for leaf in PRODUCTION_LEAF_SIZES
    }
    if set(groups) != expected:
        raise RuntimeError(
            f"production coverage mismatch: missing={sorted(expected - set(groups))} "
            f"extra={sorted(set(groups) - expected)}"
        )
    expected_counts = {key: 2 for key in expected}
    expected_counts[("YFCC-10M", 128)] = 1
    wrong_counts = {
        key: len(values)
        for key, values in groups.items()
        if len(values) != expected_counts[key]
    }
    if wrong_counts:
        raise RuntimeError(f"unexpected production repeat counts: {wrong_counts}")
    return groups


def mean_and_range(values):
    mean = statistics.mean(values)
    return mean, mean - min(values), max(values) - mean


def main():
    args = parse_args()
    groups = load_results(args.input)
    production = load_production(args.production_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, (build_ax, recall_ax) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    for dataset in DATASETS:
        build_samples = {
            leaf: [value[0] for value in groups[(dataset, leaf)]]
            for leaf in LEAF_SIZES
        }
        recall_samples = {
            leaf: [value[1] for value in groups[(dataset, leaf)]]
            for leaf in LEAF_SIZES
        }
        baseline_build = statistics.mean(build_samples[64])
        baseline_recall = statistics.mean(recall_samples[64])

        build_stats = [mean_and_range(build_samples[leaf]) for leaf in LEAF_SIZES]
        build_means = [value[0] / baseline_build for value in build_stats]
        build_errors = [
            [value[1] / baseline_build for value in build_stats],
            [value[2] / baseline_build for value in build_stats],
        ]

        recall_stats = [mean_and_range(recall_samples[leaf]) for leaf in LEAF_SIZES]
        recall_means = [
            100.0 * (value[0] - baseline_recall) for value in recall_stats
        ]
        recall_errors = [
            [100.0 * value[1] for value in recall_stats],
            [100.0 * value[2] for value in recall_stats],
        ]

        build_handle = build_ax.errorbar(
            LEAF_SIZES,
            build_means,
            yerr=build_errors,
            marker=MARKERS[dataset],
            linewidth=2,
            capsize=3,
            label=dataset,
        )
        recall_handle = recall_ax.errorbar(
            LEAF_SIZES,
            recall_means,
            yerr=recall_errors,
            marker=MARKERS[dataset],
            linewidth=2,
            capsize=3,
            label=dataset,
        )

        production_build_samples = {
            leaf: [value[0] for value in production[(dataset, leaf)]]
            for leaf in PRODUCTION_LEAF_SIZES
        }
        production_recall_samples = {
            leaf: [value[1] for value in production[(dataset, leaf)]]
            for leaf in PRODUCTION_LEAF_SIZES
        }
        production_build_stats = [
            mean_and_range(production_build_samples[leaf])
            for leaf in PRODUCTION_LEAF_SIZES
        ]
        production_build_means = [
            value[0] / baseline_build for value in production_build_stats
        ]
        production_build_errors = [
            [value[1] / baseline_build for value in production_build_stats],
            [value[2] / baseline_build for value in production_build_stats],
        ]
        production_recall_stats = [
            mean_and_range(production_recall_samples[leaf])
            for leaf in PRODUCTION_LEAF_SIZES
        ]
        production_recall_means = [
            100.0 * (value[0] - baseline_recall)
            for value in production_recall_stats
        ]
        production_recall_errors = [
            [100.0 * value[1] for value in production_recall_stats],
            [100.0 * value[2] for value in production_recall_stats],
        ]
        production_label = (
            "production GEMM (default=256)" if dataset == DATASETS[0] else None
        )
        production_color = build_handle[0].get_color()
        build_ax.errorbar(
            PRODUCTION_LEAF_SIZES,
            production_build_means,
            yerr=production_build_errors,
            marker="*",
            markersize=12,
            linestyle="--",
            linewidth=1.8,
            capsize=3,
            color=production_color,
            markeredgecolor="#111827",
            markeredgewidth=0.8,
            zorder=6,
            label=production_label,
        )
        recall_ax.errorbar(
            PRODUCTION_LEAF_SIZES,
            production_recall_means,
            yerr=production_recall_errors,
            marker="*",
            markersize=12,
            linestyle="--",
            linewidth=1.8,
            capsize=3,
            color=production_color,
            markeredgecolor="#111827",
            markeredgewidth=0.8,
            zorder=6,
            label=production_label,
        )
        label_offsets = {
            "Wiki-1M": (0, 8),
            "OpenAI-2M": (0, -17),
            "YFCC-10M": (0, 8),
        }
        for index, leaf in enumerate(PRODUCTION_LEAF_SIZES):
            if leaf == 128:
                continue
            build_ax.annotate(
                f"{production_build_stats[index][0] / 1000.0:.3f}s",
                (leaf, production_build_means[index]),
                xytext=label_offsets[dataset],
                textcoords="offset points",
                fontsize=7.5,
                ha="center",
            )

    for axis in (build_ax, recall_ax):
        axis.set_xscale("log", base=2)
        axis.set_xticks(LEAF_SIZES, labels=[str(value) for value in LEAF_SIZES])
        axis.axvline(256, color="0.45", linestyle=":", linewidth=1.2)
        axis.grid(True, which="major", alpha=0.28)
        axis.set_xlabel("pivot-tree leaf size")

    build_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    build_ax.set_ylabel("merge build time / legacy direct-L2 leaf-64 time")
    build_ax.set_title("Merge build-time multiplier")
    build_ax.legend(frameon=False)

    recall_ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    recall_ax.set_ylabel("Recall@12 change vs legacy direct-L2 leaf 64 (percentage points)")
    recall_ax.set_title("Recall change")
    fig.suptitle(
        "8-way CAGRA k=4 merge: legacy direct L2 vs production GEMM"
    )
    fig.text(
        0.5,
        0.015,
        "Solid lines are the two-run direct-L2 sweep; dashed stars are FP16/FP32 (float) or int8/int32 "
        "(YFCC) production. Dotted vertical line marks the default 256. "
        "Oracular partition construction is excluded.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    svg_output = args.output.with_suffix(".svg")
    fig.savefig(args.output, dpi=180)
    fig.savefig(svg_output)
    svg_output.write_text(
        "\n".join(line.rstrip() for line in svg_output.read_text().splitlines()) + "\n"
    )
    print(f"wrote {args.output}")
    print(f"wrote {svg_output}")


if __name__ == "__main__":
    main()

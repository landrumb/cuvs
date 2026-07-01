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


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "merge_api_results" / "leaf_size_8way.csv",
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


def mean_and_range(values):
    mean = statistics.mean(values)
    return mean, mean - min(values), max(values) - mean


def main():
    args = parse_args()
    groups = load_results(args.input)
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

        build_ax.errorbar(
            LEAF_SIZES,
            build_means,
            yerr=build_errors,
            marker=MARKERS[dataset],
            linewidth=2,
            capsize=3,
            label=dataset,
        )
        recall_ax.errorbar(
            LEAF_SIZES,
            recall_means,
            yerr=recall_errors,
            marker=MARKERS[dataset],
            linewidth=2,
            capsize=3,
            label=dataset,
        )

    for axis in (build_ax, recall_ax):
        axis.set_xscale("log", base=2)
        axis.set_xticks(LEAF_SIZES, labels=[str(value) for value in LEAF_SIZES])
        axis.axvline(256, color="0.45", linestyle=":", linewidth=1.2)
        axis.grid(True, which="major", alpha=0.28)
        axis.set_xlabel("pivot-tree leaf size")

    build_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    build_ax.set_ylabel("merge build time / leaf-64 time")
    build_ax.set_title("Merge build-time multiplier")
    build_ax.legend(frameon=False)

    recall_ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    recall_ax.set_ylabel("Recall@12 change vs leaf 64 (percentage points)")
    recall_ax.set_title("Recall change")
    fig.suptitle("8-way CAGRA k=4 merge: effect of larger pivot-tree leaves")
    fig.text(
        0.5,
        0.015,
        "Mean of two full-query runs; merge timing excludes oracular partition-graph construction.",
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

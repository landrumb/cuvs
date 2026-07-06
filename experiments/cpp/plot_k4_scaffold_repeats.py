# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot build time and recall for the 8-way scaffold repeat sweep."""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
DISPLAY_NAMES = {
    "Wiki-1M": "Wiki-1M",
    "OpenAI-2M": "OpenAI-2M",
    "YFCC-10M": "YFCC-10M (uint8)",
}
REPEATS = tuple(range(1, 9))
DEFAULT_REPEATS = 8
IMPLEMENTATION_RE = re.compile(r"^sweep-r(?P<run>[12])-repeat(?P<repeats>[1-8])$")
BUILD_COLOR = "#172033"
RECALL_COLOR = "#00a8d9"
DEFAULT_COLOR = "#16c8ff"


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "merge_api_results" / "scaffold_repeat_sweep_8way.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            root
            / "merge_api_results"
            / "scaffold_repeat_sweep_8way_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "merge_api_results"
            / "plots"
            / "k4_scaffold_repeat_sweep_8way.png"
        ),
    )
    return parser.parse_args()


def load_results(path):
    groups = defaultdict(list)
    seen_runs = defaultdict(set)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["parts"]) != 8:
                continue
            match = IMPLEMENTATION_RE.fullmatch(row["implementation"])
            if not match:
                raise RuntimeError(
                    f"unexpected implementation label: {row['implementation']}"
                )
            dataset = row["dataset"]
            repeats = int(match.group("repeats"))
            run = int(match.group("run"))
            key = (dataset, repeats)
            if run in seen_runs[key]:
                raise RuntimeError(f"duplicate run {run} for {key}")
            seen_runs[key].add(run)
            groups[key].append(
                {
                    "merge_ms": float(row["merge_api_e2e_ms"]),
                    "search_ms": float(row["search_ms"]),
                    "recall": float(row["recall"]),
                    "qps": float(row["qps"]),
                }
            )

    expected = {(dataset, repeats) for dataset in DATASETS for repeats in REPEATS}
    actual = set(groups)
    if actual != expected:
        raise RuntimeError(
            f"coverage mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    wrong_counts = {key: len(values) for key, values in groups.items() if len(values) != 2}
    if wrong_counts:
        raise RuntimeError(f"expected two runs per point: {wrong_counts}")
    wrong_runs = {key: runs for key, runs in seen_runs.items() if runs != {1, 2}}
    if wrong_runs:
        raise RuntimeError(f"expected run labels 1 and 2: {wrong_runs}")
    return groups


def mean_and_range(values):
    mean = statistics.mean(values)
    return mean, mean - min(values), max(values) - mean


def summarize(groups):
    summary = {}
    for dataset in DATASETS:
        baseline_merge = statistics.mean(
            sample["merge_ms"] for sample in groups[(dataset, 1)]
        )
        baseline_recall = statistics.mean(
            sample["recall"] for sample in groups[(dataset, 1)]
        )
        previous_merge = None
        previous_recall = None
        for repeats in REPEATS:
            samples = groups[(dataset, repeats)]
            merge_stats = mean_and_range([sample["merge_ms"] for sample in samples])
            recall_stats = mean_and_range([sample["recall"] for sample in samples])
            search_mean = statistics.mean(sample["search_ms"] for sample in samples)
            qps_mean = statistics.mean(sample["qps"] for sample in samples)
            merge_mean = merge_stats[0]
            recall_mean = recall_stats[0]
            summary[(dataset, repeats)] = {
                "runs": len(samples),
                "scaffold_degree": 4 * repeats,
                "merge_ms": merge_mean,
                "merge_low": merge_stats[1],
                "merge_high": merge_stats[2],
                "merge_delta_vs_repeat1_pct": 100.0
                * (merge_mean / baseline_merge - 1.0),
                "merge_delta_vs_previous_ms": (
                    0.0 if previous_merge is None else merge_mean - previous_merge
                ),
                "search_ms": search_mean,
                "recall": recall_mean,
                "recall_low": recall_stats[1],
                "recall_high": recall_stats[2],
                "recall_delta_vs_repeat1": recall_mean - baseline_recall,
                "recall_delta_vs_previous": (
                    0.0 if previous_recall is None else recall_mean - previous_recall
                ),
                "qps": qps_mean,
            }
            previous_merge = merge_mean
            previous_recall = recall_mean
    return summary


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "dataset",
        "parts",
        "scaffold_repeats",
        "runs",
        "scaffold_degree",
        "merge_api_e2e_ms",
        "merge_min_ms",
        "merge_max_ms",
        "merge_delta_vs_repeat1_pct",
        "merge_delta_vs_previous_ms",
        "search_ms",
        "recall",
        "recall_min",
        "recall_max",
        "recall_delta_vs_repeat1",
        "recall_delta_vs_previous",
        "qps",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for dataset in DATASETS:
            for repeats in REPEATS:
                row = summary[(dataset, repeats)]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "parts": 8,
                        "scaffold_repeats": repeats,
                        "runs": row["runs"],
                        "scaffold_degree": row["scaffold_degree"],
                        "merge_api_e2e_ms": f"{row['merge_ms']:.6f}",
                        "merge_min_ms": f"{row['merge_ms'] - row['merge_low']:.6f}",
                        "merge_max_ms": f"{row['merge_ms'] + row['merge_high']:.6f}",
                        "merge_delta_vs_repeat1_pct": (
                            f"{row['merge_delta_vs_repeat1_pct']:.6f}"
                        ),
                        "merge_delta_vs_previous_ms": (
                            f"{row['merge_delta_vs_previous_ms']:.6f}"
                        ),
                        "search_ms": f"{row['search_ms']:.6f}",
                        "recall": f"{row['recall']:.9f}",
                        "recall_min": f"{row['recall'] - row['recall_low']:.9f}",
                        "recall_max": f"{row['recall'] + row['recall_high']:.9f}",
                        "recall_delta_vs_repeat1": (
                            f"{row['recall_delta_vs_repeat1']:.9f}"
                        ),
                        "recall_delta_vs_previous": (
                            f"{row['recall_delta_vs_previous']:.9f}"
                        ),
                        "qps": f"{row['qps']:.6f}",
                    }
                )


def plot(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.7), sharex="col")

    for column, dataset in enumerate(DATASETS):
        build_ax = axes[0, column]
        recall_ax = axes[1, column]
        rows = [summary[(dataset, repeats)] for repeats in REPEATS]

        build_means = [row["merge_ms"] / 1000.0 for row in rows]
        build_errors = [
            [row["merge_low"] / 1000.0 for row in rows],
            [row["merge_high"] / 1000.0 for row in rows],
        ]
        recall_means = [row["recall"] for row in rows]
        recall_errors = [
            [row["recall_low"] for row in rows],
            [row["recall_high"] for row in rows],
        ]

        build_ax.errorbar(
            REPEATS,
            build_means,
            yerr=build_errors,
            color=BUILD_COLOR,
            marker="o",
            linewidth=2.2,
            capsize=3,
        )
        recall_ax.errorbar(
            REPEATS,
            recall_means,
            yerr=recall_errors,
            color=RECALL_COLOR,
            marker="o",
            linewidth=2.2,
            capsize=3,
        )

        for axis in (build_ax, recall_ax):
            axis.axvspan(
                DEFAULT_REPEATS - 0.14,
                DEFAULT_REPEATS + 0.14,
                color=DEFAULT_COLOR,
                alpha=0.11,
                linewidth=0,
            )
            axis.axvline(DEFAULT_REPEATS, color=DEFAULT_COLOR, linestyle=":", linewidth=1.6)
            axis.set_xticks(REPEATS)
            axis.set_xlim(0.72, 8.28)
            axis.grid(True, alpha=0.25)

        build_ax.scatter(
            [DEFAULT_REPEATS],
            [build_means[DEFAULT_REPEATS - 1]],
            marker="*",
            s=150,
            color=DEFAULT_COLOR,
            edgecolor=BUILD_COLOR,
            linewidth=0.8,
            zorder=5,
        )
        recall_ax.scatter(
            [DEFAULT_REPEATS],
            [recall_means[DEFAULT_REPEATS - 1]],
            marker="*",
            s=150,
            color=DEFAULT_COLOR,
            edgecolor=BUILD_COLOR,
            linewidth=0.8,
            zorder=5,
        )
        build_ax.set_title(DISPLAY_NAMES[dataset], fontsize=12, fontweight="bold")
        recall_ax.set_xlabel("independent scaffold repeats")
        recall_ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        build_ax.annotate(
            f"r8 uncapped: {build_means[DEFAULT_REPEATS - 1]:.3f} s",
            (DEFAULT_REPEATS, build_means[DEFAULT_REPEATS - 1]),
            xytext=(9, 8),
            textcoords="offset points",
            fontsize=8,
            color="0.25",
        )
        recall_ax.annotate(
            f"{recall_means[DEFAULT_REPEATS - 1]:.4f}",
            (DEFAULT_REPEATS, recall_means[DEFAULT_REPEATS - 1]),
            xytext=(9, 7),
            textcoords="offset points",
            fontsize=8,
            color="0.25",
        )
        recall_ax.annotate(
            f"{recall_means[0]:.4f}",
            (1, recall_means[0]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            color="0.25",
        )

    axes[0, 0].set_ylabel("merge build time (s)")
    axes[1, 0].set_ylabel("Recall@12")
    fig.suptitle(
        "Fastener 8-way merge: uncapped independent scaffold repeat sweep",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Means and min–max bars from two matched full-query sweeps; dotted cyan band marks the "
        "current repeat count of 8 (these historical points are uncapped). Oracular input-index "
        "construction is excluded.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), h_pad=2.0, w_pad=2.1)
    fig.savefig(path, dpi=180)
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    return svg_path


def main():
    args = parse_args()
    groups = load_results(args.input)
    summary = summarize(groups)
    write_summary(args.summary, summary)
    svg_path = plot(args.output, summary)
    print(f"wrote {args.summary}")
    print(f"wrote {args.output}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()

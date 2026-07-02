# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot actual pivot-tree leaf-size distributions for the 8-way merge runs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
COLORS = {"Wiki-1M": "#4c78a8", "OpenAI-2M": "#f58518", "YFCC-10M": "#54a24b"}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--histogram-input",
        type=Path,
        default=root / "merge_api_results" / "leaf_size_histogram_8way.csv",
    )
    parser.add_argument(
        "--summary-input",
        type=Path,
        default=root / "merge_api_results" / "leaf_size_distribution_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "merge_api_results" / "plots" / "k4_leaf_size_histograms_8way.png",
    )
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    svg = output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(f"wrote {output}")
    print(f"wrote {svg}")


def main():
    args = parse_args()
    histogram = read_rows(args.histogram_input)
    summary = {row["dataset"]: row for row in read_rows(args.summary_input)}
    if set(summary) != set(DATASETS):
        raise RuntimeError(f"summary coverage mismatch: {set(summary)}")

    by_dataset = {
        dataset: sorted(
            (row for row in histogram if row["dataset"] == dataset),
            key=lambda row: int(row["size"]),
        )
        for dataset in DATASETS
    }
    for dataset, rows in by_dataset.items():
        if [int(row["size"]) for row in rows] != list(range(1, 129)):
            raise RuntimeError(f"histogram coverage mismatch for {dataset}")

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        rows = by_dataset[dataset]
        sizes = [int(row["size"]) for row in rows]
        percentages = [float(row["percent_of_leaves"]) for row in rows]
        stats = summary[dataset]
        mean = float(stats["mean"])
        median = float(stats["median"])
        ax.bar(sizes, percentages, width=1.0, color=COLORS[dataset], alpha=0.88)
        ax.axvline(mean, color="#b91c1c", linewidth=1.6, label=f"mean {mean:.1f}")
        ax.axvline(
            median,
            color="#111827",
            linewidth=1.5,
            linestyle="--",
            label=f"median {median:g}",
        )
        text = (
            f"leaves: {int(stats['leaves']):,}\n"
            f"mean / median: {mean:.1f} / {median:g}\n"
            f"p10-p90: {float(stats['p10']):g}-{float(stats['p90']):g}\n"
            f"p95 / max: {float(stats['p95']):g} / {int(stats['max'])}\n"
            f"exactly 128: {float(stats['full_128_pct']):.3f}%"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
        )
        ax.set_title(dataset)
        ax.set_xlim(0.5, 128.5)
        ax.set_xticks((1, 16, 32, 48, 64, 80, 96, 112, 128))
        ax.set_xlabel("actual points in leaf")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=8)

    axes[0].set_ylabel("leaves at size (%)")
    fig.suptitle("Actual pivot-tree leaf sizes in 8-way production merges (configured maximum 128)")
    fig.text(
        0.5,
        0.015,
        "Exact size frequencies from the deterministic pivot tree; each distribution sums to the full dataset row count.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

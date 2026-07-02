# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Overlay pivot-tree leaf-size distributions for three split policies."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
METHODS = ("baseline", "score-median", "balanced-retry")
METHOD_LABELS = {
    "baseline": "baseline nearest pivot",
    "score-median": "sort d(x,a) - d(x,b) + midpoint",
    "balanced-retry": "balanced two-pass pivots",
}
METHOD_SHORT_LABELS = {
    "baseline": "baseline",
    "score-median": "midpoint",
    "balanced-retry": "two-pass",
}
COLORS = {
    "baseline": "#4c78a8",
    "score-median": "#f58518",
    "balanced-retry": "#54a24b",
}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--histogram-input",
        type=Path,
        default=root
        / "merge_api_results"
        / "leaf_size_pivot_variants_histogram_8way.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=root
        / "merge_api_results"
        / "leaf_size_pivot_variants_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "merge_api_results"
        / "plots"
        / "k4_leaf_size_pivot_variants_8way.png",
    )
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def grouped_rows(rows):
    groups = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        if key in groups and any(
            existing["size"] == row["size"] for existing in groups[key]
        ):
            raise RuntimeError(f"duplicate histogram bin for {key}: {row['size']}")
        groups.setdefault(key, []).append(row)

    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if set(groups) != expected:
        raise RuntimeError(
            f"histogram coverage mismatch: expected {expected}, got {set(groups)}"
        )

    for key, group in groups.items():
        group.sort(key=lambda row: int(row["size"]))
        sizes = [int(row["size"]) for row in group]
        if sizes != list(range(1, 129)):
            raise RuntimeError(f"size-bin coverage mismatch for {key}")
        rows_values = {int(row["rows"]) for row in group}
        maxima = {int(row["max_leaf_size"]) for row in group}
        if len(rows_values) != 1 or maxima != {128}:
            raise RuntimeError(f"inconsistent metadata for {key}")
        covered = sum(int(row["covered_points"]) for row in group)
        if covered != next(iter(rows_values)):
            raise RuntimeError(f"point coverage mismatch for {key}")
    return groups


def calculate_stats(group):
    sizes = np.array([int(row["size"]) for row in group], dtype=np.int64)
    counts = np.array([int(row["count"]) for row in group], dtype=np.int64)
    values = np.repeat(sizes, counts)
    rows = int(group[0]["rows"])
    if values.size != counts.sum() or int(values.sum()) != rows:
        raise RuntimeError("invalid histogram totals")
    quantiles = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
    return {
        "rows": rows,
        "leaves": int(values.size),
        "min": int(values.min()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "mean": float(values.mean()),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": int(values.max()),
        "full_128_leaves": int(counts[-1]),
        "full_128_pct": 100.0 * float(counts[-1]) / float(values.size),
    }


def write_summary(path, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "method",
        "rows",
        "leaves",
        "min",
        "p10",
        "p25",
        "median",
        "mean",
        "p75",
        "p90",
        "p95",
        "max",
        "full_128_leaves",
        "full_128_pct",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "method": method,
                        **stats[(dataset, method)],
                    }
                )
    print(f"wrote {path}")


def draw_distribution(axis, group, method):
    sizes = [int(row["size"]) for row in group]
    percentages = [float(row["percent_of_leaves"]) for row in group]
    color = COLORS[method]
    axis.bar(
        sizes,
        percentages,
        width=1.0,
        color=color,
        edgecolor=color,
        linewidth=0.45,
        alpha=0.27,
        label=METHOD_LABELS[method],
    )
    axis.step(
        sizes,
        percentages,
        where="mid",
        color=color,
        linewidth=1.0,
        alpha=0.95,
    )


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    svg = output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(f"wrote {output}")
    print(f"wrote {svg}")


def main():
    args = parse_args()
    groups = grouped_rows(read_rows(args.histogram_input))
    stats = {key: calculate_stats(group) for key, group in groups.items()}
    write_summary(args.summary_output, stats)

    non_midpoint_peak = max(
        float(row["percent_of_leaves"])
        for (dataset, method), group in groups.items()
        if method != "score-median"
        for row in group
    )
    zoom_limit = 1.15 * non_midpoint_peak

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.4, 8.2),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": (1.15, 1.0)},
    )
    for column, dataset in enumerate(DATASETS):
        full_axis = axes[0, column]
        zoom_axis = axes[1, column]
        for method in METHODS:
            group = groups[(dataset, method)]
            draw_distribution(full_axis, group, method)
            draw_distribution(zoom_axis, group, method)

        full_axis.set_title(dataset, fontsize=12)
        full_axis.set_xlim(0.5, 128.5)
        full_axis.set_ylim(bottom=0)
        full_axis.grid(axis="y", alpha=0.22)
        full_axis.set_axisbelow(True)

        lines = []
        for method in METHODS:
            item = stats[(dataset, method)]
            lines.append(
                f"{METHOD_SHORT_LABELS[method]:8s}  "
                f"{item['leaves']:>7,} leaves  "
                f"mean {item['mean']:>5.1f}  "
                f"median {item['median']:g}"
            )
        full_axis.text(
            0.025,
            0.965,
            "\n".join(lines),
            transform=full_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "#d1d5db"},
        )

        zoom_axis.set_xlim(0.5, 128.5)
        zoom_axis.set_ylim(0, zoom_limit)
        zoom_axis.set_xticks((1, 16, 32, 48, 64, 80, 96, 112, 128))
        zoom_axis.set_xlabel("actual points in leaf")
        zoom_axis.grid(axis="y", alpha=0.25)
        zoom_axis.set_axisbelow(True)
        zoom_axis.text(
            0.985,
            0.95,
            f"zoom: 0–{zoom_limit:.1f}%\nmidpoint peaks clipped",
            transform=zoom_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.35",
        )

    axes[0, 0].set_ylabel("leaves at size (%) — full scale")
    axes[1, 0].set_ylabel("leaves at size (%) — zoom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "Pivot-tree leaf sizes by split policy (configured maximum 128)",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Transparent overlays use exact deterministic frequencies (seed 1234); "
        "each distribution covers the full dataset.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_figure(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

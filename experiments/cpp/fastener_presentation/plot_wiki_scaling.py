#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Validate, plot, and tabulate the Wiki-10M Fastener scaling experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROWS = (1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000)
PARTS = (2, 8, 128)
IMPLEMENTATIONS = ("rebuild", "k4-scaffold-cap64")
COLORS = {2: "#0072B2", 8: "#E69F00", 128: "#009E73"}
MARKERS = {2: "o", 8: "s", 128: "^"}
LINESTYLES = {"rebuild": "--", "k4-scaffold-cap64": "-"}
NAMES = {"rebuild": "Rebuild", "k4-scaffold-cap64": "Fastener"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "data" / "wiki_scaling.csv")
    parser.add_argument("--plot-dir", type=Path, default=root / "plots")
    parser.add_argument("--table", type=Path, default=root / "tables" / "wiki_scaling.md")
    return parser.parse_args()


def load(path: Path) -> dict[tuple[int, int, str], dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing result file: {path}")
    output: dict[tuple[int, int, str], dict[str, str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "Wiki-10M-prefix":
                raise RuntimeError(f"unexpected dataset: {row['dataset']}")
            key = (int(row["rows"]), int(row["parts"]), row["implementation"])
            if key in output:
                raise RuntimeError(f"duplicate result: {key}")
            output[key] = row
    expected = {
        (rows, parts, implementation)
        for rows in ROWS
        for parts in PARTS
        for implementation in IMPLEMENTATIONS
    }
    if set(output) != expected:
        raise RuntimeError(
            f"coverage mismatch: missing={sorted(expected - set(output))} "
            f"extra={sorted(set(output) - expected)}"
        )
    return output


def merge_seconds(row: dict[str, str]) -> float:
    return float(row["merge_api_e2e_ms"]) / 1000.0


def normalize_svg(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def plot(rows: dict[tuple[int, int, str], dict[str, str]], plot_dir: Path) -> None:
    fig, (time_axis, recall_axis, delta_axis) = plt.subplots(1, 3, figsize=(16, 9))
    x = [value / 1_000_000 for value in ROWS]
    recalls: list[float] = []
    for parts in PARTS:
        for implementation in IMPLEMENTATIONS:
            samples = [rows[(count, parts, implementation)] for count in ROWS]
            time_axis.plot(
                x,
                [merge_seconds(sample) for sample in samples],
                color=COLORS[parts],
                marker=MARKERS[parts],
                linestyle=LINESTYLES[implementation],
                linewidth=2.2,
                markersize=7,
            )
            values = [float(sample["recall"]) for sample in samples]
            recalls.extend(values)
            recall_axis.plot(
                x,
                values,
                color=COLORS[parts],
                marker=MARKERS[parts],
                linestyle=LINESTYLES[implementation],
                linewidth=2.2,
                markersize=7,
            )
        deltas = [
            100.0
            * (
                float(rows[(count, parts, "k4-scaffold-cap64")]["recall"])
                - float(rows[(count, parts, "rebuild")]["recall"])
            )
            for count in ROWS
        ]
        delta_axis.plot(
            x,
            deltas,
            color=COLORS[parts],
            marker=MARKERS[parts],
            linewidth=2.2,
            markersize=7,
        )

    for axis in (time_axis, recall_axis, delta_axis):
        axis.set_xticks(x, labels=[f"{value:g}M" for value in x])
        axis.set_xlabel("Wiki dataset size")
        axis.grid(True, alpha=0.25)
    time_axis.set_yscale("log")
    time_axis.set_ylabel("Merge/build time (s, log scale)")
    time_axis.set_title("Build-time scaling")
    recall_axis.set_ylabel("Recall@12")
    recall_axis.set_title("Recall scaling")
    span = max(max(recalls) - min(recalls), 0.01)
    recall_axis.set_ylim(max(0, min(recalls) - 0.1 * span), min(1, max(recalls) + 0.1 * span))
    delta_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.4)
    delta_axis.set_ylabel("Recall@12 delta (percentage points)")
    delta_axis.set_title("Fastener recall minus rebuild")

    handles = [
        Line2D([0], [0], color=COLORS[parts], marker=MARKERS[parts], label=f"{parts} graphs")
        for parts in PARTS
    ]
    handles.extend(
        Line2D([0], [0], color="#333333", linestyle=LINESTYLES[implementation], label=NAMES[implementation])
        for implementation in IMPLEMENTATIONS
    )
    fig.suptitle("Fastener scaling on Wiki-10M prefixes", fontsize=17, weight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.84, wspace=0.27)
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / "wiki_scaling.png"
    svg = plot_dir / "wiki_scaling.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    normalize_svg(svg)
    print(f"wrote {png}")
    print(f"wrote {svg}")




def plot_swapped(
    rows: dict[tuple[int, int, str], dict[str, str]], plot_dir: Path
) -> None:
    fig, (time_axis, recall_axis, delta_axis) = plt.subplots(1, 3, figsize=(16, 9))
    x = list(range(len(PARTS)))
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
    markers = ("o", "s", "^", "D", "v", "P")
    recalls: list[float] = []
    for index, count in enumerate(ROWS):
        color = palette[index]
        marker = markers[index]
        for implementation in IMPLEMENTATIONS:
            samples = [rows[(count, parts, implementation)] for parts in PARTS]
            time_axis.plot(
                x,
                [merge_seconds(sample) for sample in samples],
                color=color,
                marker=marker,
                linestyle=LINESTYLES[implementation],
                linewidth=2.1,
                markersize=6,
            )
            values = [float(sample["recall"]) for sample in samples]
            recalls.extend(values)
            recall_axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                linestyle=LINESTYLES[implementation],
                linewidth=2.1,
                markersize=6,
            )
        deltas = [
            100.0
            * (
                float(rows[(count, parts, "k4-scaffold-cap64")]["recall"])
                - float(rows[(count, parts, "rebuild")]["recall"])
            )
            for parts in PARTS
        ]
        delta_axis.plot(
            x,
            deltas,
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=6,
        )

    for axis in (time_axis, recall_axis, delta_axis):
        axis.set_xticks(x, labels=[str(parts) for parts in PARTS])
        axis.set_xlabel("Input graph fan-in")
        axis.grid(True, alpha=0.25)
    time_axis.set_yscale("log")
    time_axis.set_ylabel("Merge/build time (s, log scale)")
    time_axis.set_title("Build time by fan-in")
    recall_axis.set_ylabel("Recall@12")
    recall_axis.set_title("Recall by fan-in")
    span = max(max(recalls) - min(recalls), 0.01)
    recall_axis.set_ylim(
        max(0, min(recalls) - 0.1 * span),
        min(1, max(recalls) + 0.1 * span),
    )
    delta_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.4)
    delta_axis.set_ylabel("Recall@12 delta (percentage points)")
    delta_axis.set_title("Fastener recall minus rebuild")

    handles = [
        Line2D(
            [0],
            [0],
            color=palette[index],
            marker=markers[index],
            linewidth=2.1,
            label=f"{count / 1_000_000:g}M rows",
        )
        for index, count in enumerate(ROWS)
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=LINESTYLES[implementation],
            label=NAMES[implementation],
        )
        for implementation in IMPLEMENTATIONS
    )
    fig.suptitle(
        "Fastener Wiki scaling, fan-in on x-axis",
        fontsize=17,
        weight="bold",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.10,
        top=0.80,
        wspace=0.27,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / "wiki_scaling_swapped.png"
    svg = plot_dir / "wiki_scaling_swapped.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    normalize_svg(svg)
    print(f"wrote {png}")
    print(f"wrote {svg}")


def write_table(rows: dict[tuple[int, int, str], dict[str, str]], path: Path) -> None:
    lines = [
        "# Wiki-10M scaling",
        "",
        "Prefixes are taken from `wiki_all_10M/base.10M.fbin`; every size uses its exact "
        "query ground truth. Input partition construction is excluded. Fastener uses the "
        "production defaults (8 repeats, leaf size 256, k=4, pre-optimize cap 64).",
        "",
        "| rows | graphs | rebuild time (s) | Fastener time (s) | speedup | rebuild Recall@12 | Fastener Recall@12 | recall delta (pp) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for count in ROWS:
        for parts in PARTS:
            rebuild = rows[(count, parts, "rebuild")]
            fastener = rows[(count, parts, "k4-scaffold-cap64")]
            rebuild_time = merge_seconds(rebuild)
            fastener_time = merge_seconds(fastener)
            rebuild_recall = float(rebuild["recall"])
            fastener_recall = float(fastener["recall"])
            lines.append(
                f"| {count:,} | {parts} | {rebuild_time:.3f} | {fastener_time:.3f} | "
                f"{rebuild_time / fastener_time:.2f}x | {rebuild_recall:.6f} | "
                f"{fastener_recall:.6f} | {100 * (fastener_recall - rebuild_recall):+.3f} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    rows = load(args.input)
    plot(rows, args.plot_dir)
    plot_swapped(rows, args.plot_dir)
    write_table(rows, args.table)


if __name__ == "__main__":
    main()

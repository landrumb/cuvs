#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Validate, plot, and tabulate the Fastener presentation parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
PARTS = (2, 8, 128)
COLORS = {2: "#0072B2", 8: "#E69F00", 128: "#009E73"}
MARKERS = {2: "o", 8: "s", 128: "^"}


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    values: tuple[int, ...]
    default: int
    file_name: str | None
    plot_stem: str


FAMILIES = (
    Family(
        key="repeats",
        label="Pivot-tree + leaf-kNN repeats",
        values=(1, 2, 4, 8, 16, 32),
        default=8,
        file_name="repeats.csv",
        plot_stem="parameter_repeats",
    ),
    Family(
        key="neighbors",
        label="Neighbors connected per leaf (k)",
        values=(1, 2, 4, 8, 16),
        default=4,
        file_name="neighbors_per_leaf.csv",
        plot_stem="parameter_neighbors_per_leaf",
    ),
    Family(
        key="leaf_size",
        label="Pivot-tree leaf size",
        values=(64, 128, 256, 512),
        default=256,
        file_name=None,
        plot_stem="parameter_leaf_size",
    ),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--plot-dir", type=Path, default=root / "plots")
    parser.add_argument(
        "--table", type=Path, default=root / "tables" / "parameter_sweeps.md"
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing result file: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty result file: {path}")
    return rows


def index_rows(rows: list[dict[str, str]], source: Path) -> dict[tuple, dict]:
    result: dict[tuple, dict] = {}
    for row in rows:
        key = (row["dataset"], int(row["parts"]), row["implementation"])
        if key in result:
            raise RuntimeError(f"duplicate row in {source}: {key}")
        result[key] = row
    return result


def load_baselines(data_dir: Path) -> dict[tuple[str, int], dict]:
    path = data_dir / "rebuild_baseline.csv"
    rows = index_rows(load_csv(path), path)
    output: dict[tuple[str, int], dict] = {}
    for dataset in DATASETS:
        for parts in PARTS:
            key = (dataset, parts, "rebuild")
            if key not in rows:
                raise RuntimeError(f"missing rebuild baseline: {key}")
            output[(dataset, parts)] = rows[key]
    expected = {(dataset, parts, "rebuild") for dataset in DATASETS for parts in PARTS}
    extra = set(rows) - expected
    if extra:
        raise RuntimeError(f"unexpected rebuild rows: {sorted(extra)}")
    return output


def implementation_for(family: Family, value: int) -> str:
    if family.key == "repeats":
        return f"k4-scaffold-repeat{value}-cap64"
    if family.key == "neighbors":
        if value == 4:
            return "k4-scaffold-cap64"
        return f"k4-scaffold-k{value}-cap64"
    if family.key == "leaf_size":
        return "k4-scaffold-cap64"
    raise AssertionError(family.key)


def load_family(data_dir: Path, family: Family) -> dict[tuple[str, int, int], dict]:
    output: dict[tuple[str, int, int], dict] = {}
    if family.file_name is not None:
        path = data_dir / family.file_name
        indexed = index_rows(load_csv(path), path)
        for dataset in DATASETS:
            for parts in PARTS:
                for value in family.values:
                    raw_key = (dataset, parts, implementation_for(family, value))
                    if raw_key not in indexed:
                        raise RuntimeError(f"missing {family.key} result: {raw_key}")
                    output[(dataset, parts, value)] = indexed[raw_key]
        expected = {
            (dataset, parts, implementation_for(family, value))
            for dataset in DATASETS
            for parts in PARTS
            for value in family.values
        }
        extra = set(indexed) - expected
        if extra:
            raise RuntimeError(f"unexpected {family.key} rows: {sorted(extra)}")
        return output

    for value in family.values:
        path = data_dir / f"leaf_size_{value}.csv"
        indexed = index_rows(load_csv(path), path)
        for dataset in DATASETS:
            for parts in PARTS:
                raw_key = (dataset, parts, implementation_for(family, value))
                if raw_key not in indexed:
                    raise RuntimeError(f"missing leaf-size result: value={value} {raw_key}")
                output[(dataset, parts, value)] = indexed[raw_key]
        expected = {
            (dataset, parts, implementation_for(family, value))
            for dataset in DATASETS
            for parts in PARTS
        }
        extra = set(indexed) - expected
        if extra:
            raise RuntimeError(f"unexpected leaf-size-{value} rows: {sorted(extra)}")
    return output


def value(row: dict, field: str) -> float:
    return float(row[field])


def merge_seconds(row: dict) -> float:
    return value(row, "merge_api_e2e_ms") / 1000.0


def normalize_svg(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def set_recall_limits(axis, recall_values: list[float]) -> None:
    low = min(recall_values)
    high = max(recall_values)
    span = max(high - low, 0.01)
    axis.set_ylim(max(0.0, low - span * 0.12), min(1.0, high + span * 0.12))


def plot_family(
    plot_dir: Path,
    family: Family,
    rows: dict[tuple[str, int, int], dict],
    baselines: dict[tuple[str, int], dict],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    for column, dataset in enumerate(DATASETS):
        time_axis = axes[0, column]
        recall_axis = axes[1, column]
        recall_values: list[float] = []
        for parts in PARTS:
            samples = [rows[(dataset, parts, setting)] for setting in family.values]
            times = [merge_seconds(sample) for sample in samples]
            recalls = [value(sample, "recall") for sample in samples]
            recall_values.extend(recalls)
            time_axis.plot(
                family.values,
                times,
                color=COLORS[parts],
                marker=MARKERS[parts],
                linewidth=2.1,
                markersize=5.5,
                label=f"{parts} graphs",
            )
            recall_axis.plot(
                family.values,
                recalls,
                color=COLORS[parts],
                marker=MARKERS[parts],
                linewidth=2.1,
                markersize=5.5,
            )

        rebuild_times = [merge_seconds(baselines[(dataset, parts)]) for parts in PARTS]
        rebuild_recalls = [value(baselines[(dataset, parts)], "recall") for parts in PARTS]
        recall_values.extend(rebuild_recalls)
        time_mean = statistics.mean(rebuild_times)
        recall_mean = statistics.mean(rebuild_recalls)
        time_axis.axhline(time_mean, color="#333333", linestyle="--", linewidth=1.8)
        recall_axis.axhline(recall_mean, color="#333333", linestyle="--", linewidth=1.8)
        if max(rebuild_times) > min(rebuild_times):
            time_axis.axhspan(
                min(rebuild_times), max(rebuild_times), color="#333333", alpha=0.08
            )
        if max(rebuild_recalls) > min(rebuild_recalls):
            recall_axis.axhspan(
                min(rebuild_recalls), max(rebuild_recalls), color="#333333", alpha=0.08
            )

        for axis in (time_axis, recall_axis):
            axis.axvline(family.default, color="#CC79A7", linestyle=":", linewidth=2.0)
            axis.set_xscale("log", base=2)
            axis.set_xticks(family.values, labels=[str(item) for item in family.values])
            axis.grid(True, which="major", alpha=0.25)
        time_axis.set_yscale("log")
        time_axis.set_title(dataset, fontsize=13, weight="bold")
        time_axis.set_ylabel("Merge/build time (s)" if column == 0 else "")
        recall_axis.set_ylabel("Recall@12" if column == 0 else "")
        recall_axis.set_xlabel(family.label)
        set_recall_limits(recall_axis, recall_values)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[parts],
            marker=MARKERS[parts],
            linewidth=2.1,
            label=f"Fastener: {parts} graphs",
        )
        for parts in PARTS
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="--", label="Rebuild baseline"),
            Line2D([0], [0], color="#CC79A7", linestyle=":", label="Production default"),
        ]
    )
    fig.suptitle(f"Fastener parameter sweep: {family.label}", fontsize=16, weight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.84, wspace=0.20, hspace=0.16)
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"{family.plot_stem}.png"
    svg = plot_dir / f"{family.plot_stem}.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    normalize_svg(svg)
    print(f"wrote {png}")
    print(f"wrote {svg}")




def plot_family_swapped(
    plot_dir: Path,
    family: Family,
    rows: dict[tuple[str, int, int], dict],
    baselines: dict[tuple[str, int], dict],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    x = list(range(len(PARTS)))
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
    markers = ("o", "s", "^", "D", "v", "P")
    for column, dataset in enumerate(DATASETS):
        time_axis = axes[0, column]
        recall_axis = axes[1, column]
        recall_values: list[float] = []
        for index, setting in enumerate(family.values):
            samples = [rows[(dataset, parts, setting)] for parts in PARTS]
            times = [merge_seconds(sample) for sample in samples]
            recalls = [value(sample, "recall") for sample in samples]
            recall_values.extend(recalls)
            is_default = setting == family.default
            label = f"{setting}" + (" (default)" if is_default else "")
            style = {
                "color": palette[index],
                "marker": markers[index],
                "linewidth": 3.0 if is_default else 1.9,
                "markersize": 7 if is_default else 5.5,
                "label": label,
                "zorder": 4 if is_default else 2,
            }
            time_axis.plot(x, times, **style)
            recall_axis.plot(x, recalls, **style)

        rebuild = [baselines[(dataset, parts)] for parts in PARTS]
        rebuild_times = [merge_seconds(sample) for sample in rebuild]
        rebuild_recalls = [value(sample, "recall") for sample in rebuild]
        recall_values.extend(rebuild_recalls)
        time_axis.plot(
            x,
            rebuild_times,
            color="#222222",
            marker="X",
            linestyle="--",
            linewidth=2.2,
            markersize=6.5,
            label="Rebuild",
        )
        recall_axis.plot(
            x,
            rebuild_recalls,
            color="#222222",
            marker="X",
            linestyle="--",
            linewidth=2.2,
            markersize=6.5,
            label="Rebuild",
        )

        for axis in (time_axis, recall_axis):
            axis.set_xticks(x, labels=[str(parts) for parts in PARTS])
            axis.grid(True, alpha=0.25)
        time_axis.set_yscale("log")
        time_axis.set_title(dataset, fontsize=13, weight="bold")
        time_axis.set_ylabel("Merge/build time (s)" if column == 0 else "")
        recall_axis.set_ylabel("Recall@12" if column == 0 else "")
        recall_axis.set_xlabel("Input graph fan-in")
        set_recall_limits(recall_axis, recall_values)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"Fastener parameter sweep, fan-in on x-axis: {family.label}",
        fontsize=16,
        weight="bold",
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.81,
        wspace=0.20,
        hspace=0.16,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / f"{family.plot_stem}_swapped.png"
    svg = plot_dir / f"{family.plot_stem}_swapped.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    normalize_svg(svg)
    print(f"wrote {png}")
    print(f"wrote {svg}")


def markdown_table(
    family: Family,
    rows: dict[tuple[str, int, int], dict],
    baselines: dict[tuple[str, int], dict],
) -> list[str]:
    lines = [
        f"## {family.label}",
        "",
        "| dataset | graphs | value | Fastener time (s) | Fastener Recall@12 | rebuild time (s) | rebuild Recall@12 | speedup | recall delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for parts in PARTS:
            baseline = baselines[(dataset, parts)]
            rebuild_time = merge_seconds(baseline)
            rebuild_recall = value(baseline, "recall")
            for setting in family.values:
                sample = rows[(dataset, parts, setting)]
                fastener_time = merge_seconds(sample)
                recall = value(sample, "recall")
                lines.append(
                    f"| {dataset} | {parts} | {setting} | {fastener_time:.4f} | "
                    f"{recall:.6f} | {rebuild_time:.4f} | {rebuild_recall:.6f} | "
                    f"{rebuild_time / fastener_time:.2f}x | {recall - rebuild_recall:+.6f} |"
                )
    lines.append("")
    return lines


def write_tables(
    path: Path,
    family_rows: dict[str, dict[tuple[str, int, int], dict]],
    baselines: dict[tuple[str, int], dict],
) -> None:
    lines = [
        "# Fastener parameter sweeps",
        "",
        "All times are end-to-end merge/build time after input partition graphs exist. "
        "Search uses the full query set, Recall@12, graph degree 64, intermediate degree 128, "
        "and itopk 160. Rebuild rows physically reconstruct CAGRA on the full dataset.",
        "",
        "Production defaults are 8 repeats, leaf size 256, and k=4. Measurements were made "
        "on one NVIDIA H100 PCIe 80 GB. Input graph construction is excluded.",
        "",
    ]
    for family in FAMILIES:
        lines.extend(markdown_table(family, family_rows[family.key], baselines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    baselines = load_baselines(args.data_dir)
    family_rows = {
        family.key: load_family(args.data_dir, family) for family in FAMILIES
    }
    for family in FAMILIES:
        plot_family(args.plot_dir, family, family_rows[family.key], baselines)
        plot_family_swapped(args.plot_dir, family, family_rows[family.key], baselines)
    write_tables(args.table, family_rows, baselines)


if __name__ == "__main__":
    main()

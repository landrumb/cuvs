#!/usr/bin/env python3
"""Plot Fastener and binary-tree recall relative to matched rebuild recall."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "merge_api_results" / "fanin_2_128_summary.csv"
OUTPUT_DIR = ROOT / "merge_api_results" / "plots"
DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
PARTS = (2, 4, 8)
METHODS = (
    ("Fastener (8 repeats)", "fastener_repeat8_recall", "#2563eb"),
    ("Binary-tree cross-query", "binary_cross_query_recall", "#d97706"),
)


def rows() -> dict[tuple[str, int], dict[str, str]]:
    with INPUT.open(newline="") as handle:
        result = {(row["dataset"], int(row["parts"])): row for row in csv.DictReader(handle)}
    expected = {(dataset, parts) for dataset in DATASETS for parts in PARTS}
    missing = expected - set(result)
    if missing:
        raise RuntimeError(f"missing measurements: {sorted(missing)}")
    return result


def main() -> None:
    values = rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    x = np.arange(len(PARTS))
    width = 0.34
    relative_values: list[float] = []
    for dataset in DATASETS:
        for _, field, _ in METHODS:
            for parts in PARTS:
                row = values[(dataset, parts)]
                relative_values.append(100 * float(row[field]) / float(row["physical_rebuild_recall"]))
    lower = min(96.0, np.floor(min(relative_values) * 10) / 10 - 0.1)

    for axis, dataset in zip(axes, DATASETS):
        for idx, (label, field, color) in enumerate(METHODS):
            relative = [100 * float(values[(dataset, parts)][field]) /
                        float(values[(dataset, parts)]["physical_rebuild_recall"])
                        for parts in PARTS]
            bars = axis.bar(x + (idx - 0.5) * width, relative, width, label=label,
                            color=color, edgecolor="white", linewidth=0.8)
            for bar, value in zip(bars, relative):
                axis.annotate(f"{value:.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        axis.axhline(100, color="#111827", linestyle=(0, (4, 3)), linewidth=1.3, label="Rebuild (100%)")
        axis.set_title(dataset, fontsize=13, weight="bold")
        axis.set_xticks(x, [f"{parts}-way" for parts in PARTS])
        axis.set_xlabel("Input partitions")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(lower, 100.65)
    axes[0].set_ylabel("Recall relative to matched rebuild (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=3, frameon=False)
    fig.suptitle("Fastener and binary-tree cross-query recall relative to rebuild\n"
                 "Recall@12 / matched physical-rebuild Recall@12; CAGRA search with itopk=160",
                 fontsize=16, y=0.975)
    # Keep the plotting axes below the shared title and legend.
    # Constrained layout cannot account reliably for a figure-level legend here.
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.79, wspace=0.10)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"fastener_tree_relative_to_rebuild_recall.{suffix}", dpi=200 if suffix == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()

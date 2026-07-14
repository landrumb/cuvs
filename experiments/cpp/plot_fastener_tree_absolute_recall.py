#!/usr/bin/env python3
"""Plot absolute Fastener and binary-tree cross-query Recall@12."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "merge_api_results" / "fanin_2_128_summary.csv"
OUT = ROOT / "merge_api_results" / "plots"
DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
PARTS = (2, 4, 8)
METHODS = (
    ("Fastener (8 repeats)", "fastener_repeat8_recall", "#2563eb"),
    ("Binary-tree cross-query", "binary_cross_query_recall", "#d97706"),
)


def load() -> dict[tuple[str, int], dict[str, str]]:
    with INPUT.open(newline="") as handle:
        data = {(r["dataset"], int(r["parts"])): r for r in csv.DictReader(handle)}
    expected = {(dataset, parts) for dataset in DATASETS for parts in PARTS}
    if expected - set(data):
        raise RuntimeError(f"missing measurements: {sorted(expected - set(data))}")
    return data


def main() -> None:
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    x = np.arange(len(PARTS))
    width = 0.25
    recalls = [float(row[field]) for row in data.values() for _, field, _ in METHODS]
    ymin = max(0.0, min(recalls) - 0.035)
    for axis, dataset in zip(axes, DATASETS):
        for i, (label, field, color) in enumerate(METHODS):
            values = [float(data[(dataset, parts)][field]) for parts in PARTS]
            bars = axis.bar(x + (i - 1) * width, values, width, label=label, color=color,
                            edgecolor="white", linewidth=0.8)
            for bar, value in zip(bars, values):
                axis.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        rebuild_recall = float(data[(dataset, PARTS[0])]["physical_rebuild_recall"])
        axis.axhline(rebuild_recall, color="#111827", linestyle=(0, (4, 3)), linewidth=1.4, label="Rebuild")
        axis.set_title(dataset, fontsize=13, weight="bold")
        axis.set_xticks(x, [f"{parts}-way" for parts in PARTS])
        axis.set_xlabel("Input partitions")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(ymin, 1.006)
    axes[0].set_ylabel("Absolute Recall@12")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=3, frameon=False)
    fig.suptitle("Absolute recall of Fastener and binary-tree cross-query versus rebuild\n"
                 "CAGRA search with itopk=160", fontsize=16, y=0.975)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.79, wspace=0.10)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"fastener_tree_absolute_recall.{suffix}", dpi=200 if suffix == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()

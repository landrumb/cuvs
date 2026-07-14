#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot the 8-way Fastener clustering Pareto comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ["Wiki-1M", "OpenAI-2M", "YFCC-10M"]
PIVOT_REPEATS = [1, 2, 4, 8, 16, 32]
FLAT_ITERATIONS = [1, 2, 5, 10, 20]
TARGET_CLUSTER_SIZE = 256
TREE_ITERATIONS = 5


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    return parser.parse_args()


def require_one(frame: pd.DataFrame, dataset: str, implementation: str) -> pd.Series:
    rows = frame[
        (frame.dataset == dataset)
        & (frame.parts == 8)
        & (frame.implementation == implementation)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"{dataset} {implementation}: expected one row, found {len(rows)}"
        )
    return rows.iloc[0]


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(
        ["merge_api_e2e_ms", "recall"], ascending=[True, False]
    )
    keep: set[int] = set()
    best_recall = float("-inf")
    for index, row in ordered.iterrows():
        if row.recall > best_recall:
            keep.add(index)
            best_recall = row.recall
    return frame.index.to_series().map(lambda index: index in keep)


def load_data(root: Path) -> pd.DataFrame:
    kmeans = pd.read_csv(root / "data" / "kmeans_merge_pareto.csv")
    repeats = pd.read_csv(root / "data" / "repeats.csv")
    rebuild = pd.read_csv(root / "data" / "rebuild_baseline.csv")
    records: list[dict[str, object]] = []

    for dataset in DATASETS:
        for repeat in PIVOT_REPEATS:
            implementation = f"k4-scaffold-repeat{repeat}-cap64"
            row = require_one(repeats, dataset, implementation)
            records.append(
                {
                    **row.to_dict(),
                    "family": "Pivot tree",
                    "configuration": f"{repeat} repeat{'s' if repeat != 1 else ''}",
                    "order": repeat,
                }
            )
        for iterations in FLAT_ITERATIONS:
            implementation = (
                f"flat-kmeans-target{TARGET_CLUSTER_SIZE}-iter{iterations}-k4-cap64"
            )
            row = require_one(kmeans, dataset, implementation)
            records.append(
                {
                    **row.to_dict(),
                    "family": "Flat balanced k-means",
                    "configuration": f"{iterations} iteration{'s' if iterations != 1 else ''}",
                    "order": iterations,
                }
            )
        for branching in [2, 5]:
            implementation = (
                f"kmeans-tree-b{branching}-leaf256-iter{TREE_ITERATIONS}-k4-cap64"
            )
            row = require_one(kmeans, dataset, implementation)
            records.append(
                {
                    **row.to_dict(),
                    "family": f"Lloyd k-means tree, k={branching}",
                    "configuration": (
                        f"k={branching}, {TREE_ITERATIONS} iterations/level"
                    ),
                    "order": branching,
                }
            )
        row = require_one(rebuild, dataset, "rebuild")
        records.append(
            {
                **row.to_dict(),
                "family": "Rebuild",
                "configuration": "Rebuild",
                "order": 0,
            }
        )

    data = pd.DataFrame(records)
    if data[["merge_api_e2e_ms", "recall"]].isna().any().any():
        raise RuntimeError("Pareto inputs contain missing merge time or recall")
    data["pareto"] = False
    for dataset in DATASETS:
        mask = data.dataset == dataset
        data.loc[mask, "pareto"] = pareto_mask(data.loc[mask]).values
    return data


def plot(root: Path, data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    styles = {
        "Pivot tree": ("#4C78A8", "o", "-"),
        "Flat balanced k-means": ("#F58518", "s", "-"),
        "Lloyd k-means tree, k=2": ("#54A24B", "^", "None"),
        "Lloyd k-means tree, k=5": ("#B279A2", "v", "None"),
        "Rebuild": ("#E45756", "*", "None"),
    }
    for axis, dataset in zip(axes, DATASETS):
        rows = data[data.dataset == dataset]
        frontier = rows[rows.pareto].sort_values("merge_api_e2e_ms")
        axis.plot(
            frontier.merge_api_e2e_ms / 1000.0,
            frontier.recall,
            color="#222222",
            linestyle=":",
            linewidth=2.0,
            label="Overall Pareto frontier",
            zorder=1,
        )
        for family, (color, marker, linestyle) in styles.items():
            family_rows = rows[rows.family == family].sort_values("order")
            axis.plot(
                family_rows.merge_api_e2e_ms / 1000.0,
                family_rows.recall,
                color=color,
                marker=marker,
                markersize=11 if family == "Rebuild" else 7,
                linestyle=linestyle,
                linewidth=2.0,
                label=family,
                zorder=2,
            )
            for row in family_rows.itertuples():
                if family == "Pivot tree":
                    label = f"{int(row.order)}x"
                elif family == "Flat balanced k-means":
                    label = f"{int(row.order)}i"
                elif family.startswith("Lloyd"):
                    label = f"k={int(row.order)}"
                else:
                    continue
                axis.annotate(
                    label,
                    (row.merge_api_e2e_ms / 1000.0, row.recall),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_title(dataset)
        axis.set_xscale("log")
        axis.set_xlabel("Merge time (s, log scale)")
        axis.set_ylabel("Index recall@12")
        axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "8-way Fastener merge: clustering time–recall Pareto comparison "
        "(up and to the left is better)",
        y=0.94,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output = root / "plots" / "kmeans_merge_pareto"
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def write_table(root: Path, data: pd.DataFrame) -> None:
    lines = [
        "# 8-way clustering Pareto comparison",
        "",
        "Merge time includes clustering, exact cross-input top-4 neighbor "
        "construction within each cluster, graph append/sort/cap, and CAGRA "
        "optimization. Recall is query recall@12 for the merged index.",
        "",
        "| Dataset | Family | Configuration | Merge time (s) | Recall@12 | Pareto |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for row in data.sort_values(
        ["dataset", "family", "merge_api_e2e_ms"]
    ).itertuples():
        lines.append(
            f"| {row.dataset} | {row.family} | {row.configuration} | "
            f"{row.merge_api_e2e_ms / 1000:.3f} | {row.recall:.6f} | "
            f"{'yes' if row.pareto else ''} |"
        )
    (root / "tables" / "kmeans_merge_pareto.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    args = parse_args()
    data = load_data(args.root)
    plot(args.root, data)
    write_table(args.root, data)
    print("wrote k-means merge Pareto plot and table")


if __name__ == "__main__":
    main()

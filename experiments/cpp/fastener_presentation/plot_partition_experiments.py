#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot Fastener partition-quality, implicit-kNN, and merge-variant experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

DATASETS = ["Wiki-1M", "OpenAI-2M", "YFCC-10M"]
PARTS = [2, 8, 128]
REPEATS = [1, 2, 4, 8, 16, 32]
PIVOT_METHODS = ["pivot-binary", "pivot-ternary"]
COLORS = {2: "#4C78A8", 8: "#F58518", 128: "#54A24B"}
METHOD_COLORS = {
    "Fastener binary": "#4C78A8",
    "Fastener ternary": "#F58518",
    "Native NN-descent k32": "#B279A2",
    "Rebuild": "#E45756",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def require_rows(frame: pd.DataFrame, mask: pd.Series, expected: int, description: str) -> None:
    count = int(mask.sum())
    if count != expected:
        raise RuntimeError(f"{description}: expected {expected} rows, found {count}")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_path = root / "data" / "partition_quality.csv"
    flat_path = root / "data" / "kmeans_flat_clustering.csv"
    variants_path = root / "data" / "partition_variants.csv"
    repeats_path = root / "data" / "neighbors_per_leaf.csv"
    rebuild_path = root / "data" / "rebuild_baseline.csv"
    for path in [quality_path, flat_path, variants_path, repeats_path, rebuild_path]:
        if not path.exists():
            raise RuntimeError(f"missing input: {path}")

    quality = pd.read_csv(quality_path)
    for dataset in DATASETS:
        partition = quality[
            (quality.dataset == dataset) & (quality.record_type == "partition")
        ]
        require_rows(partition, partition.index == partition.index, 12, f"{dataset} partition")
        implicit = quality[
            (quality.dataset == dataset) & (quality.record_type == "implicit-scaffold")
        ]
        require_rows(implicit, implicit.index == implicit.index, 36, f"{dataset} scaffold")
        native = quality[
            (quality.dataset == dataset) & (quality.record_type == "implicit-native")
        ]
        require_rows(native, native.index == native.index, 12, f"{dataset} native")
    flat = pd.read_csv(flat_path)
    for dataset in DATASETS:
        mask = flat.dataset == dataset
        require_rows(flat, mask, 1, f"{dataset} flat k-means")
    expected_flat = {
        (dataset, "kmeans-balanced-flat-iter20") for dataset in DATASETS
    }
    actual_flat = set(zip(flat.dataset, flat.method))
    if actual_flat != expected_flat:
        raise RuntimeError("flat k-means coverage mismatch")
    for row in flat.itertuples():
        expected_clusters = (int(row.rows) + 255) // 256
        if int(row.leaf_count) != expected_clusters:
            raise RuntimeError(
                f"{row.dataset}: expected {expected_clusters} clusters, "
                f"found {int(row.leaf_count)}"
            )

    variants = pd.read_csv(variants_path)
    binary = pd.read_csv(repeats_path)
    binary = binary[binary.implementation == "k4-scaffold-cap64"].copy()
    rebuild = pd.read_csv(rebuild_path)
    merged = pd.concat([variants, binary, rebuild], ignore_index=True)
    names = {
        "k4-scaffold-cap64": "Fastener binary",
        "ternary-scaffold": "Fastener ternary",
        "native-knn-k32": "Native NN-descent k32",
        "rebuild": "Rebuild",
    }
    merged["method"] = merged.implementation.map(names)
    if merged.method.isna().any():
        bad = sorted(merged.loc[merged.method.isna(), "implementation"].unique())
        raise RuntimeError(f"unexpected merged implementation labels: {bad}")
    for dataset in DATASETS:
        for parts in PARTS:
            mask = (merged.dataset == dataset) & (merged.parts == parts)
            require_rows(merged, mask, 4, f"{dataset} fan-in {parts} merged variants")
    return quality, merged, flat


def plot_partition_retention(root: Path, quality: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    for column, dataset in enumerate(DATASETS):
        data = quality[
            (quality.dataset == dataset) & (quality.record_type == "partition")
        ]
        for method, marker, linestyle in [
            ("pivot-binary", "o", "-"),
            ("pivot-ternary", "s", "--"),
        ]:
            rows = data[data.method == method].sort_values("repeats")
            label = method.replace("pivot-", "").title()
            axes[0, column].plot(
                rows.repeats,
                rows.cumulative_ms / 1000.0,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axes[1, column].plot(
                rows.repeats,
                rows.nn_same_partition_rate,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Cumulative partition time (s)")
        axes[1, column].set_ylabel("Exact 12-NN retained in a leaf")
        axes[1, column].set_xlabel("Independent pivot-tree repeats")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks(REPEATS, labels=REPEATS)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Binary versus ternary pivot-tree partition quality", fontsize=16)
    fig.tight_layout()
    save_figure(fig, root / "plots" / "partition_retention")



def plot_kmeans_flat_clustering(
    root: Path, quality: pd.DataFrame, flat: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    pivot_styles = {
        "pivot-binary": ("Binary pivot tree", "#4C78A8", "o", "-"),
        "pivot-ternary": ("Ternary pivot tree", "#F58518", "s", "--"),
    }
    for column, dataset in enumerate(DATASETS):
        pivot = quality[
            (quality.dataset == dataset)
            & (quality.record_type == "partition")
        ]
        for method, (label, color, marker, linestyle) in pivot_styles.items():
            rows = pivot[pivot.method == method].sort_values("repeats")
            axes[0, column].plot(
                rows.repeats,
                rows.cumulative_ms / 1000.0,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axes[1, column].plot(
                rows.repeats,
                rows.nn_same_partition_rate,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )

        kmeans = flat[flat.dataset == dataset].iloc[0]
        axes[0, column].axhline(
            kmeans.cumulative_ms / 1000.0,
            color="#B279A2",
            linestyle="-.",
            linewidth=2.4,
            label="Flat balanced k-means",
        )
        axes[1, column].axhline(
            kmeans.nn_same_partition_rate,
            color="#B279A2",
            linestyle="-.",
            linewidth=2.4,
            label="Flat balanced k-means",
        )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Cumulative construction time (s)")
        axes[1, column].set_ylabel("Exact 12-NN retained in one cluster")
        axes[1, column].set_xlabel("Independent pivot-tree repeats")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks(REPEATS, labels=REPEATS)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Pivot trees versus flat balanced k-means (k = ceil(n / 256))",
        y=0.94,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, root / "plots" / "kmeans_flat_clustering")


def plot_implicit_scaffold(root: Path, quality: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    for column, dataset in enumerate(DATASETS):
        data = quality[
            (quality.dataset == dataset) & (quality.record_type == "implicit-scaffold")
        ]
        for method, linestyle in [("pivot-binary", "-"), ("pivot-ternary", "--")]:
            for parts in PARTS:
                rows = data[(data.method == method) & (data.parts == parts)].sort_values(
                    "repeats"
                )
                label = f"{method.replace('pivot-', '').title()}, {parts}-way"
                axes[0, column].plot(
                    rows.repeats,
                    rows.cumulative_ms / 1000.0,
                    color=COLORS[parts],
                    linestyle=linestyle,
                    marker="o" if method == "pivot-binary" else "s",
                    label=label,
                )
                axes[1, column].plot(
                    rows.repeats,
                    rows.implicit_cross_knn_recall,
                    color=COLORS[parts],
                    linestyle=linestyle,
                    marker="o" if method == "pivot-binary" else "s",
                    label=label,
                )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Scaffold construction time (s)")
        axes[1, column].set_ylabel("Cross-partition exact 12-NN recall")
        axes[1, column].set_xlabel("Independent pivot-tree repeats")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks(REPEATS, labels=REPEATS)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.suptitle("Implicit Fastener scaffold as a cross-partition kNN generator", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, root / "plots" / "implicit_scaffold_knn")




def plot_implicit_scaffold_swapped(root: Path, quality: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    x = np.arange(len(PARTS))
    repeats = [1, 2, 4, 8, 16, 32]
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
    methods = {
        "pivot-binary": ("Binary", "-", "o"),
        "pivot-ternary": ("Ternary", "--", "s"),
    }
    for column, dataset in enumerate(DATASETS):
        data = quality[
            (quality.dataset == dataset)
            & (quality.record_type == "implicit-scaffold")
        ]
        for repeat_index, repeat in enumerate(repeats):
            color = palette[repeat_index]
            width = 2.8 if repeat == 8 else 1.7
            for method, (_, linestyle, marker) in methods.items():
                rows = (
                    data[(data.method == method) & (data.repeats == repeat)]
                    .set_index("parts")
                    .loc[PARTS]
                )
                axes[0, column].plot(
                    x,
                    rows.cumulative_ms / 1000.0,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=width,
                )
                axes[1, column].plot(
                    x,
                    rows.implicit_cross_knn_recall,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=width,
                )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Scaffold construction time (s)")
        axes[1, column].set_ylabel("Cross-partition exact 12-NN recall")
        axes[1, column].set_xlabel("Input graph fan-in")
        axes[1, column].set_xticks(x, labels=PARTS)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)

    handles = [
        Line2D(
            [0],
            [0],
            color=palette[index],
            linewidth=2.8 if repeat == 8 else 1.7,
            label=f"{repeat} repeats" + (" (default)" if repeat == 8 else ""),
        )
        for index, repeat in enumerate(repeats)
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=linestyle,
            marker=marker,
            label=label,
        )
        for label, linestyle, marker in methods.values()
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Implicit Fastener scaffold, fan-in on x-axis",
        y=0.88,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    save_figure(fig, root / "plots" / "implicit_scaffold_knn_swapped")


def plot_native_comparison(root: Path, quality: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    styles = {
        "pivot-binary": ("Binary pivot repeats", "#4C78A8", "o", "-"),
        "pivot-ternary": ("Ternary pivot repeats", "#F58518", "s", "--"),
        "native-nn-descent": ("Native NN-descent", "#B279A2", "^", "-."),
    }
    for column, dataset in enumerate(DATASETS):
        pivot = quality[
            (quality.dataset == dataset)
            & (quality.record_type == "implicit-scaffold")
            & (quality.parts == 8)
        ].copy()
        pivot["candidate_degree"] = pivot.repeats * pivot.neighbors_per_leaf
        native = quality[
            (quality.dataset == dataset)
            & (quality.record_type == "implicit-native")
            & (quality.parts == 8)
        ].copy()
        native["candidate_degree"] = native.neighbors_per_leaf
        data = pd.concat([pivot, native], ignore_index=True)
        for method, (label, color, marker, linestyle) in styles.items():
            rows = data[data.method == method].sort_values("candidate_degree")
            axes[0, column].plot(
                rows.candidate_degree,
                rows.cumulative_ms / 1000.0,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            axes[1, column].plot(
                rows.candidate_degree,
                rows.implicit_cross_knn_recall,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Graph construction time (s)")
        axes[1, column].set_ylabel("Cross-partition exact 12-NN recall")
        axes[1, column].set_xlabel("Candidate graph degree")
        axes[1, column].set_xscale("log", base=2)
        axes[1, column].set_xticks([4, 8, 12, 16, 32, 64, 128])
        axes[1, column].get_xaxis().set_major_formatter(ScalarFormatter())
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Pivot-tree scaffold versus native cuVS NN-descent", fontsize=16)
    fig.tight_layout()
    save_figure(fig, root / "plots" / "native_vs_pivot_knn")


def plot_merged_variants(root: Path, merged: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    x = np.arange(len(PARTS))
    for column, dataset in enumerate(DATASETS):
        data = merged[merged.dataset == dataset]
        for method in METHOD_COLORS:
            rows = data[data.method == method].set_index("parts").loc[PARTS]
            axes[0, column].plot(
                x,
                rows.merge_api_e2e_ms / 1000.0,
                marker="o",
                color=METHOD_COLORS[method],
                label=method,
            )
            axes[1, column].plot(
                x,
                rows.recall,
                marker="o",
                color=METHOD_COLORS[method],
                label=method,
            )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Merge time (s)")
        axes[1, column].set_ylabel("Query recall@12")
        axes[1, column].set_xlabel("Input graph fan-in")
        axes[1, column].set_xticks(x, labels=PARTS)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Fastener partition variants versus rebuild", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, root / "plots" / "partition_variant_merge")


def write_tables(
    root: Path, quality: pd.DataFrame, merged: pd.DataFrame, flat: pd.DataFrame
) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fastener partition experiment tables",
        "",
        "Times are end-to-end benchmark measurements on the experiment host. "
        "Partition CAGRA build time is excluded from merge time, consistently across variants.",
        "",
        "## Pivot partition retention at the default eight repeats",
        "",
        "| Dataset | Tree | Partition time (s) | Exact 12-NN same-leaf rate |",
        "|---|---:|---:|---:|",
    ]
    rows = quality[
        (quality.record_type == "partition") & (quality.repeats == 8)
    ].sort_values(["dataset", "method"])
    for row in rows.itertuples():
        lines.append(
            f"| {row.dataset} | {row.method} | {row.cumulative_ms / 1000:.3f} | "
            f"{row.nn_same_partition_rate:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Flat balanced k-means with k = ceil(n / 256)",
            "",
            "This is one direct, non-hierarchical cuVS balanced k-means fit with "
            "20 iterations and squared L2 distance.",
            "",
            "| Dataset | Rows | Clusters (k) | Iterations | Time (s) | Min size | Mean size | Max size | Exact 12-NN same-cluster rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in flat.sort_values("dataset").itertuples():
        iterations = int(row.method.rsplit("iter", 1)[1])
        lines.append(
            f"| {row.dataset} | {int(row.rows):,} | {int(row.leaf_count):,} | "
            f"{iterations} | {row.cumulative_ms / 1000:.3f} | "
            f"{int(row.min_leaf_size)} | {row.mean_leaf_size:.3f} | "
            f"{int(row.max_leaf_size)} | {row.nn_same_partition_rate:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Native NN-descent all-neighbors quality",
            "",
            "| Dataset | k | Build time (s) | Exact 12-NN recall |",
            "|---|---:|---:|---:|",
        ]
    )
    native = quality[
        (quality.record_type == "implicit-native") & (quality.parts == 0)
    ].sort_values(["dataset", "neighbors_per_leaf"])
    for row in native.itertuples():
        lines.append(
            f"| {row.dataset} | {int(row.neighbors_per_leaf)} | "
            f"{row.cumulative_ms / 1000:.3f} | {row.implicit_knn_recall:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Merged-index results",
            "",
            "| Dataset | Fan-in | Method | Merge time (s) | Recall@12 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in merged.sort_values(["dataset", "parts", "method"]).itertuples():
        lines.append(
            f"| {row.dataset} | {int(row.parts)} | {row.method} | "
            f"{row.merge_api_e2e_ms / 1000:.3f} | {row.recall:.6f} |"
        )
    (tables / "partition_experiments.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    quality, merged, flat = load_inputs(args.root)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    plot_partition_retention(args.root, quality)
    plot_kmeans_flat_clustering(args.root, quality, flat)
    plot_implicit_scaffold(args.root, quality)
    plot_implicit_scaffold_swapped(args.root, quality)
    plot_native_comparison(args.root, quality)
    plot_merged_variants(args.root, merged)
    write_tables(args.root, quality, merged, flat)
    print(f"wrote partition plots and tables under {args.root}")


if __name__ == "__main__":
    main()

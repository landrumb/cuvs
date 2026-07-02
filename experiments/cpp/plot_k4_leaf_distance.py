# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot 8-way CAGRA k=4 leaf-distance and origin-diversity experiments."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
DISTANCE_VARIANTS = (
    ("direct", "Direct L2"),
    ("symmetric", "Symmetric L2"),
    ("norm_dot", "Norm + dot"),
    ("production", "Production FP32 / int8 GEMM"),
    ("mixed", "FP16 input, FP32 accumulate (1 GiB)"),
)
ORIGIN_VARIANTS = (
    ("direct-l2", "Unconstrained"),
    ("direct-l2-min2-origins", "At least 2"),
    ("direct-l2-min3-origins", "At least 3"),
    ("direct-l2-distinct-origin", "Strict 4"),
)
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distance-input",
        type=Path,
        default=root / "merge_api_results" / "leaf_distance_8way.csv",
    )
    parser.add_argument(
        "--precision-input",
        type=Path,
        default=root / "merge_api_results" / "leaf_gemm_precision_summary.csv",
    )
    parser.add_argument(
        "--origin-input",
        type=Path,
        default=root / "merge_api_results" / "origin_diversity_8way.csv",
    )
    parser.add_argument(
        "--profile-input",
        type=Path,
        default=root / "merge_api_results" / "leaf_distance_profile_8way_wiki.csv",
    )
    parser.add_argument(
        "--scratch-input",
        type=Path,
        default=root / "merge_api_results" / "results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "merge_api_results" / "plots",
    )
    parser.add_argument(
        "--skip-origin",
        action="store_true",
        help="Do not regenerate the origin-diversity figure.",
    )
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def distance_key(row):
    implementation = row["implementation"]
    if implementation == "direct-l2":
        return "direct"
    if implementation == "symmetric-l2":
        return "symmetric"
    if implementation in ("symmetric-norm-dot", "symmetric-norm-dot-dp4a"):
        return "norm_dot"
    if implementation == "gemm-production":
        return "production"
    return None


def selected_distance_rows(rows, precision_rows):
    selected = {}
    for row in rows:
        key = distance_key(row)
        if key is not None:
            pair = (row["dataset"], key)
            if pair in selected:
                raise RuntimeError(f"duplicate selected distance row: {pair}")
            selected[pair] = row

    for row in precision_rows:
        if (
            row["dataset"] in ("Wiki-1M", "OpenAI-2M")
            and row["implementation"] == "gemm-fp16-fp32-1g"
        ):
            selected[(row["dataset"], "mixed")] = row

    expected = {
        (dataset, key)
        for dataset in DATASETS
        for key, _ in DISTANCE_VARIANTS
        if key != "mixed" or dataset != "YFCC-10M"
    }
    if set(selected) != expected:
        raise RuntimeError(
            f"distance coverage mismatch: missing={sorted(expected - set(selected))} "
            f"extra={sorted(set(selected) - expected)}"
        )
    return selected

def format_recall(recall):
    return f"{100.0 * recall:.3g}"


def save_figure(fig, output):
    fig.savefig(output, dpi=180)
    svg = output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(f"wrote {output}")
    print(f"wrote {svg}")


def plot_distance(rows, precision_rows, scratch_rows, output):
    selected = selected_distance_rows(rows, precision_rows)
    scratch = {
        row["dataset"]: row
        for row in scratch_rows
        if int(row["parts"]) == 8 and row["implementation"] == "rebuild"
    }
    if set(scratch) != set(DATASETS):
        raise RuntimeError(f"scratch CAGRA coverage mismatch: {set(scratch)}")

    x = np.arange(len(DATASETS), dtype=float)
    width = 0.15
    offsets = (np.arange(len(DISTANCE_VARIANTS)) - 2.0) * width
    fig, (build_ax, qps_ax, recall_ax, workspace_ax) = plt.subplots(
        1, 4, figsize=(22.0, 5.5)
    )

    for variant_index, ((key, label), color) in enumerate(zip(DISTANCE_VARIANTS, COLORS)):
        positions = []
        build_values = []
        qps_values = []
        recall_deltas = []
        recalls = []
        for dataset_index, dataset in enumerate(DATASETS):
            pair = (dataset, key)
            if pair not in selected:
                continue
            baseline = scratch[dataset]
            row = selected[pair]
            positions.append(x[dataset_index] + offsets[variant_index])
            build_values.append(
                float(row["merge_api_e2e_ms"]) / float(baseline["merge_api_e2e_ms"])
            )
            qps_values.append(float(row["qps"]) / float(baseline["qps"]))
            recall = float(row["recall"])
            recalls.append(recall)
            recall_deltas.append(100.0 * (recall - float(baseline["recall"])))

        build_ax.bar(positions, build_values, width, color=color, label=label)
        qps_ax.bar(positions, qps_values, width, color=color, label=label)
        recall_bars = recall_ax.bar(
            positions, recall_deltas, width, color=color, label=label
        )
        for bar, recall in zip(recall_bars, recalls):
            delta = bar.get_height()
            recall_ax.annotate(
                format_recall(recall),
                (bar.get_x() + bar.get_width() / 2, delta),
                xytext=(0, 4 if delta >= 0 else -5),
                textcoords="offset points",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=7.2,
                rotation=90,
            )

    for axis in (build_ax, qps_ax, recall_ax):
        axis.set_xticks(x, DATASETS)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    build_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    build_ax.set_ylabel("merge build time / scratch-CAGRA time")
    build_ax.set_title("End-to-end merge build time")
    build_ax.legend(frameon=False, fontsize=8)
    qps_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    qps_ax.set_ylabel("search QPS / scratch-CAGRA QPS")
    qps_ax.set_title("Search throughput")
    recall_ax.margins(y=0.15)
    recall_ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    recall_ax.set_ylabel("Recall@12 change vs scratch CAGRA (percentage points)")
    recall_ax.set_title("Recall change (labels are actual recall, %)")

    precision = {
        (row["dataset"], row["implementation"]): row for row in precision_rows
    }
    workspace_datasets = ("Wiki-1M", "OpenAI-2M")
    workspace_x = np.arange(len(workspace_datasets), dtype=float)
    workspace_width = 0.34
    workspace_methods = (
        ("gemm-fp32-control", "FP32, 2 GiB cap", "#b279a2"),
        ("gemm-fp16-fp32-1g", "FP16/FP32, 1 GiB cap", "#e45756"),
    )
    for method_index, (implementation, label, color) in enumerate(workspace_methods):
        values = []
        per_leaf_kib = []
        for dataset in workspace_datasets:
            row = precision[(dataset, implementation)]
            values.append(float(row["allocated_workspace_bytes"]) / (1024.0**3))
            per_leaf_kib.append(float(row["total_bytes_per_leaf"]) / 1024.0)
        positions = workspace_x + (method_index - 0.5) * workspace_width
        bars = workspace_ax.bar(
            positions, values, workspace_width, color=color, label=label
        )
        for bar, kib in zip(bars, per_leaf_kib):
            workspace_ax.annotate(
                f"{kib:.0f} KiB/leaf",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    workspace_ax.set_xticks(workspace_x, workspace_datasets)
    workspace_ax.set_ylabel("allocated temporary workspace (GiB)")
    workspace_ax.set_title("Bounded GEMM workspace")
    workspace_ax.set_ylim(0, 2.45)
    workspace_ax.grid(axis="y", alpha=0.25)
    workspace_ax.set_axisbelow(True)
    workspace_ax.legend(frameon=False, fontsize=8)

    fig.suptitle("8-way CAGRA k=4 merge: leaf-distance precision and workspace")
    fig.text(
        0.5,
        0.012,
        "Dashed lines are scratch-CAGRA baselines; full datasets/query sets; oracular partition builds excluded. "
        "Mixed precision uses FP16 inputs with FP32 accumulation and Gram output.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_figure(fig, output)
    plt.close(fig)

def plot_origin(rows, output):
    indexed = {(row["dataset"], row["implementation"]): row for row in rows}
    expected = {(dataset, key) for dataset in DATASETS for key, _ in ORIGIN_VARIANTS}
    if set(indexed) != expected:
        raise RuntimeError(
            f"origin coverage mismatch: missing={sorted(expected - set(indexed))} "
            f"extra={sorted(set(indexed) - expected)}"
        )
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.19
    offsets = (np.arange(len(ORIGIN_VARIANTS)) - 1.5) * width
    fig, (build_ax, recall_ax) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for variant_index, ((key, label), color) in enumerate(zip(ORIGIN_VARIANTS, COLORS)):
        build_values = []
        recall_deltas = []
        recalls = []
        for dataset in DATASETS:
            baseline = indexed[(dataset, "direct-l2")]
            row = indexed[(dataset, key)]
            build_values.append(
                float(row["merge_api_e2e_ms"]) / float(baseline["merge_api_e2e_ms"])
            )
            recall = float(row["recall"])
            recalls.append(recall)
            recall_deltas.append(100.0 * (recall - float(baseline["recall"])))
        positions = x + offsets[variant_index]
        build_ax.bar(positions, build_values, width, color=color, label=label)
        recall_bars = recall_ax.bar(positions, recall_deltas, width, color=color, label=label)
        for bar, recall in zip(recall_bars, recalls):
            delta = bar.get_height()
            recall_ax.annotate(
                format_recall(recall),
                (bar.get_x() + bar.get_width() / 2, delta),
                xytext=(0, 4 if delta >= 0 else -5),
                textcoords="offset points",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=7.5,
                rotation=90,
            )
    for axis in (build_ax, recall_ax):
        axis.set_xticks(x, DATASETS)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    build_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    build_ax.set_ylabel("merge build time / unconstrained time")
    build_ax.set_title("End-to-end merge build time")
    build_ax.legend(frameon=False, ncol=2, fontsize=9)
    recall_ax.margins(y=0.25)
    recall_ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    recall_ax.set_ylabel("Recall@12 change vs unconstrained (percentage points)")
    recall_ax.set_title("Recall change (labels are actual recall, %)")
    fig.suptitle("8-way CAGRA k=4 merge: scaffold origin diversity")
    fig.text(
        0.5,
        0.012,
        "One full-query run per policy; small deltas include partition-index rebuild variation.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, output)
    plt.close(fig)


def plot_profile(rows, output):
    by_implementation = {}
    wall_times = {}
    measurements = {}
    for row in rows:
        by_implementation.setdefault(row["implementation"], []).append(
            (row["operation"], float(row["gpu_time_ms"]))
        )
        wall_times[row["implementation"]] = float(row["merge_wall_ms"])
        measurements[row["implementation"]] = row.get("measurement", "")
    expected = {"direct-l2", "gemm-fp32-events", "gemm-fp16-events"}
    if set(by_implementation) != expected:
        raise RuntimeError(f"profile implementations must be {expected}")

    order = ("direct-l2", "gemm-fp32-events", "gemm-fp16-events")
    labels = ("Direct L2", "FP32 GEMM", "FP16 input / FP32 accumulate")
    y_positions = [2, 1, 0]
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    segment_colors = {
        "direct leaf L2": "#4c78a8",
        "gather leaf vectors": "#f58518",
        "batched GEMM": "#54a24b",
        "Gram top-k selection": "#b279a2",
    }
    totals = {}
    for y, implementation in zip(y_positions, order):
        left = 0.0
        total = sum(value for _, value in by_implementation[implementation])
        totals[implementation] = total
        for operation, value in by_implementation[implementation]:
            ax.barh(
                y,
                value,
                left=left,
                color=segment_colors[operation],
                label=operation,
            )
            if value >= 1.0:
                ax.text(
                    left + value / 2,
                    y,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            left += value
        ax.text(total + 2.5, y, f"{total:.1f} ms", va="center", fontsize=10)

    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="lower right", fontsize=9)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("leaf-neighbor GPU time (ms)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    speedup = totals["gemm-fp32-events"] / totals["gemm-fp16-events"]
    ax.set_title(
        f"Wiki-1M 8-way leaf stage: mixed precision is {speedup:.2f}x faster than FP32"
    )
    fig.text(
        0.5,
        0.035,
        f"Profiled merge wall time: {wall_times['direct-l2']:.1f} ms direct, "
        f"{wall_times['gemm-fp32-events']:.1f} ms FP32, "
        f"{wall_times['gemm-fp16-events']:.1f} ms mixed.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.text(
        0.5,
        0.012,
        "Direct L2 is from the retained Nsight capture; both GEMM rows use identical opt-in CUDA-event timing.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_figure(fig, output)
    plt.close(fig)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_distance(
        read_rows(args.distance_input),
        read_rows(args.precision_input),
        read_rows(args.scratch_input),
        args.output_dir / "k4_leaf_distance_8way.png",
    )
    if not args.skip_origin:
        plot_origin(
            read_rows(args.origin_input),
            args.output_dir / "k4_origin_diversity_8way.png",
        )
    plot_profile(read_rows(args.profile_input), args.output_dir / "k4_leaf_distance_profile.png")


if __name__ == "__main__":
    main()

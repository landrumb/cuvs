# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Validate, summarize, and plot the 2--128 graph merge fan-in sweep."""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
DISPLAY_NAMES = {
    "Wiki-1M": "Wiki-1M",
    "OpenAI-2M": "OpenAI-2M",
    "YFCC-10M": "YFCC-10M (uint8)",
}
PARTS = (2, 4, 8, 16, 32, 64, 128)
REBUILD_PARTS = (2, 4, 8)
FASTENER_IMPLEMENTATIONS = (
    ("k4-scaffold", 2, "fastener"),
    ("k4-scaffold-repeat8", 8, "fastener_repeat8"),
    ("k4-scaffold-repeat16", 16, "fastener_repeat16"),
    ("k4-scaffold-repeat32", 32, "fastener_repeat32"),
)
IMPLEMENTATIONS = tuple(item[0] for item in FASTENER_IMPLEMENTATIONS) + (
    "binary-cross-query",
)
IMPLEMENTATION_NAMES = {
    "k4-scaffold": "Fastener (2 repeats)",
    "k4-scaffold-repeat8": "Fastener (8 repeats)",
    "k4-scaffold-repeat16": "Fastener (16 repeats)",
    "k4-scaffold-repeat32": "Fastener (32 repeats)",
    "binary-cross-query": "Binary tree cross-query merge",
}
COLORS = {
    "k4-scaffold": "#9ecae1",
    "k4-scaffold-repeat8": "#6baed6",
    "k4-scaffold-repeat16": "#3182bd",
    "k4-scaffold-repeat32": "#08519c",
    "binary-cross-query": "#e76f51",
}
MARKERS = {
    "k4-scaffold": "o",
    "k4-scaffold-repeat8": "^",
    "k4-scaffold-repeat16": "D",
    "k4-scaffold-repeat32": "v",
    "binary-cross-query": "s",
}


def normalize_svg(path):
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "merge_api_results" / "fanin_2_128_raw.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "merge_api_results" / "fanin_2_128_summary.csv",
    )
    parser.add_argument(
        "--rebuild-input",
        type=Path,
        default=root / "merge_api_results" / "results.csv",
        help="Direct in-memory physical-rebuild measurements at 2/4/8-way fan-in.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "merge_api_results" / "plots" / "fastener_fanin_2_128.png",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=root / "merge_api_results" / "plots" / "fastener_fanin_2_128.svg",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=(
            root
            / "merge_api_results"
            / "plots"
            / "fastener_fanin_2_128_log.png"
        ),
    )
    parser.add_argument(
        "--log-svg",
        type=Path,
        default=(
            root
            / "merge_api_results"
            / "plots"
            / "fastener_fanin_2_128_log.svg"
        ),
    )
    return parser.parse_args()


def load_results(path):
    results = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], int(row["parts"]), row["implementation"])
            if key in results:
                raise RuntimeError(f"duplicate result: {key}")
            results[key] = {
                "dtype": row["dtype"],
                "rows": int(row["rows"]),
                "queries": int(row["queries"]),
                "oracle_build_ms": float(row["oracle_partition_build_ms_excluded"]),
                "merge_ms": float(row["merge_api_e2e_ms"]),
                "search_ms": float(row["search_ms"]),
                "recall": float(row["recall"]),
                "qps": float(row["qps"]),
            }

    expected = {
        (dataset, parts, implementation)
        for dataset in DATASETS
        for parts in PARTS
        for implementation in IMPLEMENTATIONS
    }
    actual = set(results)
    if actual != expected:
        raise RuntimeError(
            f"coverage mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return results


def load_rebuild_results(path):
    samples = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["implementation"] != "rebuild":
                continue
            dataset = row["dataset"]
            parts = int(row["parts"])
            if dataset not in DATASETS or parts not in REBUILD_PARTS:
                continue
            key = (dataset, parts)
            if key in samples:
                raise RuntimeError(f"duplicate rebuild result: {key}")
            samples[key] = {
                "merge_ms": float(row["merge_api_e2e_ms"]),
                "recall": float(row["recall"]),
                "qps": float(row["qps"]),
            }

    expected = {
        (dataset, parts) for dataset in DATASETS for parts in REBUILD_PARTS
    }
    if set(samples) != expected:
        raise RuntimeError(
            f"rebuild coverage mismatch: missing={sorted(expected - set(samples))} "
            f"extra={sorted(set(samples) - expected)}"
        )
    return {
        dataset: {
            metric: statistics.mean(
                samples[(dataset, parts)][metric] for parts in REBUILD_PARTS
            )
            for metric in ("merge_ms", "recall", "qps")
        }
        for dataset in DATASETS
    }


def summarize(results, rebuild_results):
    summary = {}
    for dataset in DATASETS:
        for parts in PARTS:
            repeat2 = results[(dataset, parts, "k4-scaffold")]
            baseline = results[(dataset, parts, "binary-cross-query")]
            row = {
                "dtype": repeat2["dtype"],
                "rows": repeat2["rows"],
                "queries": repeat2["queries"],
                "physical_rebuild_merge_ms": rebuild_results[dataset]["merge_ms"],
                "physical_rebuild_recall": rebuild_results[dataset]["recall"],
                "physical_rebuild_qps": rebuild_results[dataset]["qps"],
                "baseline_oracle_build_ms": baseline["oracle_build_ms"],
                "baseline_merge_ms": baseline["merge_ms"],
                "baseline_search_ms": baseline["search_ms"],
                "baseline_recall": baseline["recall"],
                "baseline_qps": baseline["qps"],
            }
            for implementation, _repeats, prefix in FASTENER_IMPLEMENTATIONS:
                values = results[(dataset, parts, implementation)]
                row[f"{prefix}_oracle_build_ms"] = values["oracle_build_ms"]
                row[f"{prefix}_merge_ms"] = values["merge_ms"]
                row[f"{prefix}_merge_speedup_vs_binary_x"] = (
                    baseline["merge_ms"] / values["merge_ms"]
                )
                row[f"{prefix}_merge_slowdown_vs_repeat2_x"] = (
                    values["merge_ms"] / repeat2["merge_ms"]
                )
                row[f"{prefix}_search_ms"] = values["search_ms"]
                row[f"{prefix}_recall"] = values["recall"]
                row[f"{prefix}_recall_delta_vs_binary"] = (
                    values["recall"] - baseline["recall"]
                )
                row[f"{prefix}_recall_delta_vs_repeat2"] = (
                    values["recall"] - repeat2["recall"]
                )
                row[f"{prefix}_qps"] = values["qps"]
                row[f"{prefix}_qps_delta_vs_binary_pct"] = (
                    100.0 * (values["qps"] / baseline["qps"] - 1.0)
                )
            summary[(dataset, parts)] = row
    return summary


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "parts",
        "dtype",
        "rows",
        "queries",
        "physical_rebuild_merge_ms",
        "physical_rebuild_recall",
        "physical_rebuild_qps",
    ]
    variant_fields = (
        "oracle_partition_build_ms_excluded",
        "merge_ms",
        "merge_speedup_vs_binary_x",
        "merge_slowdown_vs_repeat2_x",
        "search_ms",
        "recall",
        "recall_delta_vs_binary",
        "recall_delta_vs_repeat2",
        "qps",
        "qps_delta_vs_binary_pct",
    )
    for _implementation, _repeats, prefix in FASTENER_IMPLEMENTATIONS:
        fieldnames.extend(f"{prefix}_{field}" for field in variant_fields)
    fieldnames.extend(
        (
            "baseline_oracle_partition_build_ms_excluded",
            "binary_cross_query_merge_ms",
            "binary_cross_query_search_ms",
            "binary_cross_query_recall",
            "binary_cross_query_qps",
        )
    )

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for dataset in DATASETS:
            for parts in PARTS:
                values = summary[(dataset, parts)]
                output = {
                    "dataset": dataset,
                    "parts": parts,
                    "dtype": values["dtype"],
                    "rows": values["rows"],
                    "queries": values["queries"],
                    "physical_rebuild_merge_ms": (
                        f"{values['physical_rebuild_merge_ms']:.6f}"
                    ),
                    "physical_rebuild_recall": (
                        f"{values['physical_rebuild_recall']:.9f}"
                    ),
                    "physical_rebuild_qps": f"{values['physical_rebuild_qps']:.6f}",
                }
                for _implementation, _repeats, prefix in FASTENER_IMPLEMENTATIONS:
                    output.update(
                        {
                            f"{prefix}_oracle_partition_build_ms_excluded": (
                                f"{values[f'{prefix}_oracle_build_ms']:.6f}"
                            ),
                            f"{prefix}_merge_ms": (
                                f"{values[f'{prefix}_merge_ms']:.6f}"
                            ),
                            f"{prefix}_merge_speedup_vs_binary_x": (
                                f"{values[f'{prefix}_merge_speedup_vs_binary_x']:.6f}"
                            ),
                            f"{prefix}_merge_slowdown_vs_repeat2_x": (
                                f"{values[f'{prefix}_merge_slowdown_vs_repeat2_x']:.6f}"
                            ),
                            f"{prefix}_search_ms": (
                                f"{values[f'{prefix}_search_ms']:.6f}"
                            ),
                            f"{prefix}_recall": (
                                f"{values[f'{prefix}_recall']:.9f}"
                            ),
                            f"{prefix}_recall_delta_vs_binary": (
                                f"{values[f'{prefix}_recall_delta_vs_binary']:.9f}"
                            ),
                            f"{prefix}_recall_delta_vs_repeat2": (
                                f"{values[f'{prefix}_recall_delta_vs_repeat2']:.9f}"
                            ),
                            f"{prefix}_qps": f"{values[f'{prefix}_qps']:.6f}",
                            f"{prefix}_qps_delta_vs_binary_pct": (
                                f"{values[f'{prefix}_qps_delta_vs_binary_pct']:.6f}"
                            ),
                        }
                    )
                output.update(
                    {
                        "baseline_oracle_partition_build_ms_excluded": (
                            f"{values['baseline_oracle_build_ms']:.6f}"
                        ),
                        "binary_cross_query_merge_ms": (
                            f"{values['baseline_merge_ms']:.6f}"
                        ),
                        "binary_cross_query_search_ms": (
                            f"{values['baseline_search_ms']:.6f}"
                        ),
                        "binary_cross_query_recall": (
                            f"{values['baseline_recall']:.9f}"
                        ),
                        "binary_cross_query_qps": f"{values['baseline_qps']:.6f}",
                    }
                )
                writer.writerow(output)


def add_series(axis, results, dataset, metric, scale=1.0):
    for implementation in IMPLEMENTATIONS:
        values = [
            results[(dataset, parts, implementation)][metric] * scale
            for parts in PARTS
        ]
        axis.plot(
            PARTS,
            values,
            color=COLORS[implementation],
            marker=MARKERS[implementation],
            markersize=5.5,
            linewidth=2.1,
            label=IMPLEMENTATION_NAMES[implementation],
        )


def plot(path, svg_path, results, rebuild_results, log_merge_time=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15.3, 10.2), sharex="col")

    for column, dataset in enumerate(DATASETS):
        merge_ax = axes[0, column]
        recall_ax = axes[1, column]
        qps_ax = axes[2, column]

        add_series(merge_ax, results, dataset, "merge_ms", scale=0.001)
        add_series(recall_ax, results, dataset, "recall")
        add_series(qps_ax, results, dataset, "qps", scale=0.001)
        rebuild = rebuild_results[dataset]
        for axis, value in (
            (merge_ax, rebuild["merge_ms"] / 1000.0),
            (recall_ax, rebuild["recall"]),
            (qps_ax, rebuild["qps"] / 1000.0),
        ):
            axis.axhline(
                value,
                color="#6b7280",
                linestyle="--",
                linewidth=1.8,
                label="Physical rebuild (mean of 2/4/8-way)",
            )

        if log_merge_time:
            merge_ax.set_yscale("log")
        else:
            merge_ax.set_ylim(bottom=0.0)
        merge_ax.set_title(DISPLAY_NAMES[dataset], fontsize=13, fontweight="bold")
        recall_ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        qps_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}k"))

        recall_values = [
            results[(dataset, parts, implementation)]["recall"]
            for parts in PARTS
            for implementation in IMPLEMENTATIONS
        ]
        recall_values.append(rebuild["recall"])
        recall_span = max(recall_values) - min(recall_values)
        recall_pad = max(0.008, recall_span * 0.10)
        recall_ax.set_ylim(min(recall_values) - recall_pad, max(recall_values) + recall_pad)

        qps_values = [
            results[(dataset, parts, implementation)]["qps"] / 1000.0
            for parts in PARTS
            for implementation in IMPLEMENTATIONS
        ]
        qps_values.append(rebuild["qps"] / 1000.0)
        qps_span = max(qps_values) - min(qps_values)
        qps_pad = max(max(qps_values) * 0.01, qps_span * 0.18)
        qps_ax.set_ylim(min(qps_values) - qps_pad, max(qps_values) + qps_pad)

        for axis in (merge_ax, recall_ax, qps_ax):
            axis.set_xscale("log", base=2)
            axis.set_xticks(PARTS, [str(parts) for parts in PARTS])
            axis.grid(True, which="major", alpha=0.26)
            axis.spines[["top", "right"]].set_visible(False)
        qps_ax.set_xlabel("Number of input graphs")

    merge_axis_label = (
        "Merge build time (s, log scale)"
        if log_merge_time
        else "Merge build time (s)"
    )
    axes[0, 0].set_ylabel(merge_axis_label)
    axes[1, 0].set_ylabel("Recall@12")
    axes[2, 0].set_ylabel("Query throughput (kQPS)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Fastener & cross-query scaling vs subgraph count",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "H100 PCIe; graph degree 64; itopk 160; Fastener repeats: 2, 8, 16, 32. "
        "Rebuild hlines are physical-rebuild means at 2/4/8-way; leaf builds are excluded.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.035, 0.045, 0.995, 0.935), h_pad=2.0, w_pad=1.8)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    normalize_svg(svg_path)
    plt.close(fig)


def main():
    args = parse_args()
    results = load_results(args.input)
    rebuild_results = load_rebuild_results(args.rebuild_input)
    summary = summarize(results, rebuild_results)
    write_summary(args.summary, summary)
    plot(args.output, args.svg, results, rebuild_results)
    plot(
        args.log_output,
        args.log_svg,
        results,
        rebuild_results,
        log_merge_time=True,
    )


if __name__ == "__main__":
    main()

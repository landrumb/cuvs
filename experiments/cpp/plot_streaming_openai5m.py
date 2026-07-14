# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Validate, aggregate, and plot the OpenAI-5M streaming-build experiment."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIGS = (
    ("naive-host", 1),
    ("prefetch-device", 1),
    ("prefetch-device", 2),
    ("prefetch-device", 4),
    ("prefetch-device", 8),
)
NETWORK_ONLY_CONFIGS = (("prefetch-single-build", 8),)
CONFIGS_BY_MODE = {
    "local": CONFIGS,
    "network": CONFIGS[:1] + NETWORK_ONLY_CONFIGS + CONFIGS[1:],
}
MODES = ("local", "network")
RUNS = {1, 2}
DISPLAY_LABELS = {
    ("naive-host", 1): "Naive\nhost",
    ("prefetch-single-build", 8): "Transfer pipeline\n+ single build",
    ("prefetch-device", 1): "Prefetch\n1 chunk",
    ("prefetch-device", 2): "Pipeline\n2 chunks",
    ("prefetch-device", 4): "Pipeline\n4 chunks",
    ("prefetch-device", 8): "Pipeline\n8 chunks",
}
NUMERIC_FIELDS = (
    "prepare_ms_excluded",
    "device_alloc_ms",
    "download_wall_ms",
    "download_achieved_mib_s",
    "disk_read_active_ms",
    "h2d_active_ms",
    "load_wall_sum_ms",
    "build_sum_ms",
    "build_wait_ms",
    "merge_ms",
    "total_ms",
    "serial_work_ms",
    "overlap_saved_ms",
    "load_build_overlap_ms",
    "download_build_overlap_ms",
    "download_load_overlap_ms",
    "allocation_download_overlap_ms",
)
PART_FLOAT_FIELDS = (
    "download_start_ms",
    "download_end_ms",
    "load_start_ms",
    "load_end_ms",
    "load_wall_ms",
    "disk_read_active_ms",
    "h2d_active_ms",
    "build_wait_ms",
    "build_start_ms",
    "build_end_ms",
    "build_ms",
)
COLORS = {
    "local": "#172033",
    "network": "#00a8d9",
    "load": "#16c8ff",
    "build": "#29384f",
    "merge": "#ff9f1c",
    "allocation": "#b8c2cc",
    "download": "#8b95a5",
    "total": "#d62828",
    "overlap": "#00a8d9",
}


def parse_args():
    root = Path(__file__).resolve().parent
    results = root / "merge_api_results"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-input",
        type=Path,
        default=results / "streaming_openai5m_summary_20260709.csv",
    )
    parser.add_argument(
        "--parts-input",
        type=Path,
        default=results / "streaming_openai5m_parts_20260709.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=results / "streaming_openai5m_aggregated_20260709.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=results / "plots",
    )
    parser.add_argument("--timeline-run", type=int, default=2)
    return parser.parse_args()


def mean(values):
    return statistics.mean(values)


def load_summary(path):
    groups = defaultdict(list)
    seen_runs = defaultdict(set)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for line, row in enumerate(reader, start=2):
            key = (row["mode"], row["build_path"], int(row["parts"]))
            run = int(row["run"])
            if key not in {
                (mode, *config)
                for mode in MODES
                for config in CONFIGS_BY_MODE[mode]
            }:
                raise RuntimeError(f"unexpected configuration on line {line}: {key}")
            if run in seen_runs[key]:
                raise RuntimeError(f"duplicate run {run} for {key}")
            seen_runs[key].add(run)
            if row["dataset"] != "openai_5m":
                raise RuntimeError(f"unexpected dataset on line {line}")
            if (
                int(row["rows"]) != 5_000_000
                or int(row["dim"]) != 1_536
                or int(row["data_bytes"]) != 30_720_000_000
            ):
                raise RuntimeError(f"unexpected dataset shape on line {line}")
            if (
                int(row["graph_degree"]) != 64
                or int(row["intermediate_graph_degree"]) != 128
            ):
                raise RuntimeError(f"non-default graph degree on line {line}")
            if int(row["valid"]) != 1:
                raise RuntimeError(f"invalid final index on line {line}")
            if row["mode"] == "network":
                if abs(float(row["network_target_s"]) - 36.067) > 1e-9:
                    raise RuntimeError(f"unexpected network target on line {line}")
                if abs(float(row["download_wall_ms"]) - 36_067.0) > 20.0:
                    raise RuntimeError(f"network pacing missed target on line {line}")
            parsed = dict(row)
            parsed["run"] = run
            for field in NUMERIC_FIELDS:
                parsed[field] = float(row[field])
            groups[key].append(parsed)

    expected = {
                (mode, *config)
                for mode in MODES
                for config in CONFIGS_BY_MODE[mode]
            }
    actual = set(groups)
    if actual != expected:
        raise RuntimeError(
            f"coverage mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    wrong_runs = {key: runs for key, runs in seen_runs.items() if runs != RUNS}
    if wrong_runs:
        raise RuntimeError(f"expected runs 1 and 2 for every point: {wrong_runs}")
    return groups


def load_parts(path, summary_groups):
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for line, row in enumerate(reader, start=2):
            key = (
                row["mode"],
                row["build_path"],
                int(row["parts"]),
                int(row["run"]),
            )
            summary_key = key[:3]
            if summary_key not in summary_groups:
                raise RuntimeError(f"unexpected part configuration on line {line}: {key}")
            parsed = dict(row)
            for field in ("parts", "run", "part", "row_offset", "rows", "payload_bytes"):
                parsed[field] = int(row[field])
            for field in PART_FLOAT_FIELDS:
                parsed[field] = float(row[field])
            groups[key].append(parsed)

    expected = {
        (mode, build_path, parts, run)
        for mode, build_path, parts in summary_groups
        for run in RUNS
    }
    actual = set(groups)
    if actual != expected:
        raise RuntimeError(
            f"part coverage mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    for key, rows in groups.items():
        parts = key[2]
        ids = {row["part"] for row in rows}
        if len(rows) != parts or ids != set(range(parts)):
            raise RuntimeError(f"malformed part rows for {key}")
        if sum(row["rows"] for row in rows) != 5_000_000:
            raise RuntimeError(f"part row sum mismatch for {key}")
        rows.sort(key=lambda row: row["part"])
    return groups


def summarize(groups):
    result = {}
    for mode in MODES:
        naive_total = mean(
            row["total_ms"] for row in groups[(mode, "naive-host", 1)]
        )
        prefetch_total = mean(
            row["total_ms"] for row in groups[(mode, "prefetch-device", 1)]
        )
        for build_path, parts in CONFIGS_BY_MODE[mode]:
            samples = groups[(mode, build_path, parts)]
            row = {
                "mode": mode,
                "build_path": build_path,
                "parts": parts,
                "runs": len(samples),
            }
            for field in NUMERIC_FIELDS:
                values = [sample[field] for sample in samples]
                row[field] = mean(values)
                row[f"{field}_min"] = min(values)
                row[f"{field}_max"] = max(values)
            total = row["total_ms"]
            row["savings_vs_naive_ms"] = naive_total - total
            row["reduction_vs_naive_pct"] = 100.0 * (naive_total - total) / naive_total
            row["speedup_vs_naive_x"] = naive_total / total
            row["savings_vs_prefetch1_ms"] = prefetch_total - total
            row["reduction_vs_prefetch1_pct"] = (
                100.0 * (prefetch_total - total) / prefetch_total
            )
            result[(mode, build_path, parts)] = row
    return result


def write_aggregate(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "mode",
        "build_path",
        "parts",
        "runs",
        "total_mean_s",
        "total_min_s",
        "total_max_s",
        "savings_vs_naive_s",
        "reduction_vs_naive_pct",
        "speedup_vs_naive_x",
        "savings_vs_prefetch1_s",
        "reduction_vs_prefetch1_pct",
        "prepare_excluded_mean_s",
        "device_alloc_mean_ms",
        "download_mean_s",
        "download_effective_mean_mib_s",
        "disk_read_active_mean_s",
        "h2d_active_mean_s",
        "load_wall_sum_mean_s",
        "build_sum_mean_s",
        "build_wait_mean_s",
        "merge_mean_s",
        "serial_work_mean_s",
        "overlap_saved_mean_s",
        "load_build_overlap_mean_s",
        "download_build_overlap_mean_s",
        "download_load_overlap_mean_s",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for mode in MODES:
            for build_path, parts in CONFIGS_BY_MODE[mode]:
                row = summary[(mode, build_path, parts)]
                writer.writerow(
                    {
                        "mode": mode,
                        "build_path": build_path,
                        "parts": parts,
                        "runs": row["runs"],
                        "total_mean_s": f"{row['total_ms'] / 1000.0:.6f}",
                        "total_min_s": f"{row['total_ms_min'] / 1000.0:.6f}",
                        "total_max_s": f"{row['total_ms_max'] / 1000.0:.6f}",
                        "savings_vs_naive_s": (
                            f"{row['savings_vs_naive_ms'] / 1000.0:.6f}"
                        ),
                        "reduction_vs_naive_pct": (
                            f"{row['reduction_vs_naive_pct']:.6f}"
                        ),
                        "speedup_vs_naive_x": f"{row['speedup_vs_naive_x']:.6f}",
                        "savings_vs_prefetch1_s": (
                            f"{row['savings_vs_prefetch1_ms'] / 1000.0:.6f}"
                        ),
                        "reduction_vs_prefetch1_pct": (
                            f"{row['reduction_vs_prefetch1_pct']:.6f}"
                        ),
                        "prepare_excluded_mean_s": (
                            f"{row['prepare_ms_excluded'] / 1000.0:.6f}"
                        ),
                        "device_alloc_mean_ms": f"{row['device_alloc_ms']:.6f}",
                        "download_mean_s": (
                            f"{row['download_wall_ms'] / 1000.0:.6f}"
                        ),
                        "download_effective_mean_mib_s": (
                            f"{row['download_achieved_mib_s']:.6f}"
                        ),
                        "disk_read_active_mean_s": (
                            f"{row['disk_read_active_ms'] / 1000.0:.6f}"
                        ),
                        "h2d_active_mean_s": (
                            f"{row['h2d_active_ms'] / 1000.0:.6f}"
                        ),
                        "load_wall_sum_mean_s": (
                            f"{row['load_wall_sum_ms'] / 1000.0:.6f}"
                        ),
                        "build_sum_mean_s": f"{row['build_sum_ms'] / 1000.0:.6f}",
                        "build_wait_mean_s": f"{row['build_wait_ms'] / 1000.0:.6f}",
                        "merge_mean_s": f"{row['merge_ms'] / 1000.0:.6f}",
                        "serial_work_mean_s": (
                            f"{row['serial_work_ms'] / 1000.0:.6f}"
                        ),
                        "overlap_saved_mean_s": (
                            f"{row['overlap_saved_ms'] / 1000.0:.6f}"
                        ),
                        "load_build_overlap_mean_s": (
                            f"{row['load_build_overlap_ms'] / 1000.0:.6f}"
                        ),
                        "download_build_overlap_mean_s": (
                            f"{row['download_build_overlap_ms'] / 1000.0:.6f}"
                        ),
                        "download_load_overlap_mean_s": (
                            f"{row['download_load_overlap_ms'] / 1000.0:.6f}"
                        ),
                    }
                )


def plot_total(path, summary, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    configs = tuple(
        config
        for config in CONFIGS_BY_MODE[mode]
        if config != ("prefetch-device", 1)
    )
    x = list(range(len(configs)))
    labels = [DISPLAY_LABELS[config] for config in configs]
    rows = [summary[(mode, *config)] for config in configs]
    values = [row["total_ms"] / 1000.0 for row in rows]
    errors = [
        [
            (row["total_ms"] - row["total_ms_min"]) / 1000.0
            for row in rows
        ],
        [
            (row["total_ms_max"] - row["total_ms"]) / 1000.0
            for row in rows
        ],
    ]
    bars = axis.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        width=0.68,
        color=(
            [
                "#8b95a5"
                if config == ("naive-host", 1)
                else "#29384f"
                if config == ("prefetch-single-build", 8)
                else COLORS[mode]
                for config in configs
            ]
            if mode == "network"
            else COLORS[mode]
        ),
        edgecolor="white",
        linewidth=0.8,
    )
    baseline = values[0]
    axis.axhline(
        baseline,
        color="#6c757d",
        linewidth=1.3,
        linestyle="--",
        label=f"Naive mean: {baseline:.1f} s",
    )
    for index, (bar, value, row) in enumerate(zip(bars, values, rows)):
        text = f"{value:.1f} s"
        if index > 0:
            text += f"\n-{row['reduction_vs_naive_pct']:.1f}%"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.8,
            text,
            ha="center",
            va="bottom",
            fontsize=11,
        )
    title = (
        "Cold-local OpenAI-5M CAGRA construction"
        if mode == "local"
        else "OpenAI-5M CAGRA construction with 36.067 s paced download"
    )
    axis.set_title(title, fontsize=16)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 110 if mode == "local" else 132)
    axis.set_ylabel("Valid final index wall time (s)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_local_overlap(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary[("local", *config)] for config in CONFIGS]
    labels = [DISPLAY_LABELS[config] for config in CONFIGS]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    bottom = [0.0] * len(rows)
    components = (
        ("device_alloc_ms", "Device allocation", COLORS["allocation"]),
        ("load_wall_sum_ms", "Load wall sum", COLORS["load"]),
        ("build_sum_ms", "CAGRA build sum", COLORS["build"]),
        ("merge_ms", "Fastener merge", COLORS["merge"]),
    )
    for field, label, color in components:
        values = [row[field] / 1000.0 for row in rows]
        axes[0].bar(x, values, bottom=bottom, label=label, color=color, width=0.68)
        bottom = [old + value for old, value in zip(bottom, values)]
    totals = [row["total_ms"] / 1000.0 for row in rows]
    axes[0].scatter(
        x,
        totals,
        color=COLORS["total"],
        marker="D",
        s=55,
        zorder=4,
        label="Measured wall time",
    )
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Active-stage sums; stacked work overlaps on the critical path")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, frameon=False, loc="upper left")

    overlap = [row["overlap_saved_ms"] / 1000.0 for row in rows]
    load_build = [row["load_build_overlap_ms"] / 1000.0 for row in rows]
    axes[1].bar(
        x,
        overlap,
        width=0.58,
        color=COLORS["overlap"],
        label="Serial work minus wall time",
    )
    axes[1].scatter(
        x,
        load_build,
        color=COLORS["total"],
        marker="o",
        zorder=4,
        label="Measured load/build intersection",
    )
    axes[1].axhline(0.0, color="#6c757d", linewidth=0.8)
    axes[1].set_ylabel("Overlap (s)")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Cold-local OpenAI-5M pipeline accounting", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_timeline_mode(path, summary_groups, parts_groups, run, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    legend_handles = {}
    key = (mode, "prefetch-device", 8, run)
    for row in parts_groups[key]:
        y = row["part"]
        intervals = (
            (
                "Download",
                row["download_start_ms"],
                row["download_end_ms"],
                COLORS["download"],
            ),
            (
                "Disk + H2D",
                row["load_start_ms"],
                row["load_end_ms"],
                COLORS["load"],
            ),
            (
                "CAGRA build",
                row["build_start_ms"],
                row["build_end_ms"],
                COLORS["build"],
            ),
        )
        for label, start, end, color in intervals:
            if end <= start:
                continue
            handle = axis.barh(
                y,
                (end - start) / 1000.0,
                left=start / 1000.0,
                height=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
            legend_handles.setdefault(label, handle)

    run_summary = next(
        row
        for row in summary_groups[(mode, "prefetch-device", 8)]
        if row["run"] == run
    )
    merge_end = run_summary["total_ms"]
    merge_start = merge_end - run_summary["merge_ms"]
    handle = axis.barh(
        8,
        run_summary["merge_ms"] / 1000.0,
        left=merge_start / 1000.0,
        height=0.62,
        color=COLORS["merge"],
        edgecolor="white",
        linewidth=0.4,
        label="Fastener merge",
    )
    legend_handles.setdefault("Fastener merge", handle)
    axis.axvline(
        run_summary["total_ms"] / 1000.0,
        color=COLORS["total"],
        linewidth=1.3,
        linestyle="--",
    )
    axis.text(
        run_summary["total_ms"] / 1000.0 - 0.5,
        8.0,
        f"{run_summary['total_ms'] / 1000.0:.2f} s",
        ha="right",
        va="center",
        color="black",
        fontsize=10,
    )
    axis.set_yticks(
        range(9),
        [f"Part {part}" for part in range(8)] + ["Merge"],
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlabel("Time from measured boundary (s)")
    axis.set_xlim(left=0.0)
    title = (
        "Cold-local eight-chunk load/build pipeline"
        if mode == "local"
        else "Eight-chunk pipeline with 36.067 s paced download"
    )
    axis.set_title(title, fontsize=16)
    legend_order = (
        "Download",
        "Disk + H2D",
        "CAGRA build",
        "Fastener merge",
    )
    visible = [label for label in legend_order if label in legend_handles]
    axis.legend(
        [legend_handles[label] for label in visible],
        visible,
        ncol=len(visible),
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_network_speedup(path, summary_groups, parts_groups, run):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 3.5, 3.5]},
    )
    naive_axis, single_axis, pipeline_axis = axes
    colors = {
        "Download": COLORS["download"],
        "Load / transfer": COLORS["load"],
        "CAGRA build": COLORS["build"],
        "Fastener merge": COLORS["merge"],
    }

    naive_part = parts_groups[("network", "naive-host", 1, run)][0]
    naive_intervals = (
        (
            "Download",
            naive_part["download_start_ms"],
            naive_part["download_end_ms"],
        ),
        (
            "Load / transfer",
            naive_part["load_start_ms"],
            naive_part["load_end_ms"],
        ),
        (
            "CAGRA build",
            naive_part["build_start_ms"],
            naive_part["build_end_ms"],
        ),
    )
    legend_handles = {}
    for label, start, end in naive_intervals:
        handle = naive_axis.barh(
            0,
            (end - start) / 1000.0,
            left=start / 1000.0,
            height=0.55,
            color=colors[label],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        legend_handles[label] = handle
    naive_summary = next(
        row
        for row in summary_groups[("network", "naive-host", 1)]
        if row["run"] == run
    )
    naive_axis.axvline(
        naive_summary["total_ms"] / 1000.0,
        color=COLORS["total"],
        linewidth=1.3,
        linestyle="--",
    )
    naive_axis.text(
        naive_summary["total_ms"] / 1000.0 - 0.7,
        0,
        f"{naive_summary['total_ms'] / 1000.0:.2f} s",
        ha="right",
        va="center",
        color="white",
        fontsize=10,
    )
    naive_axis.set_yticks((0,), ("Full dataset",))
    naive_axis.set_title("Serialized full-dataset construction")
    naive_axis.grid(axis="x", alpha=0.25)

    single_parts = parts_groups[
        ("network", "prefetch-single-build", 8, run)
    ]
    for part in single_parts:
        y = part["part"]
        for label, start, end in (
            (
                "Download",
                part["download_start_ms"],
                part["download_end_ms"],
            ),
            (
                "Load / transfer",
                part["load_start_ms"],
                part["load_end_ms"],
            ),
        ):
            single_axis.barh(
                y,
                (end - start) / 1000.0,
                left=start / 1000.0,
                height=0.58,
                color=colors[label],
                edgecolor="white",
                linewidth=0.4,
            )
    single_build = next(part for part in single_parts if part["build_end_ms"] > 0.0)
    single_axis.barh(
        8,
        (single_build["build_end_ms"] - single_build["build_start_ms"])
        / 1000.0,
        left=single_build["build_start_ms"] / 1000.0,
        height=0.58,
        color=colors["CAGRA build"],
        edgecolor="white",
        linewidth=0.4,
    )
    single_summary = next(
        row
        for row in summary_groups[("network", "prefetch-single-build", 8)]
        if row["run"] == run
    )
    single_axis.axvline(
        single_summary["total_ms"] / 1000.0,
        color=COLORS["total"],
        linewidth=1.3,
        linestyle="--",
    )
    single_axis.text(
        single_summary["total_ms"] / 1000.0 - 0.7,
        7.55,
        f"{single_summary['total_ms'] / 1000.0:.2f} s",
        ha="right",
        va="bottom",
        color="black",
        fontsize=10,
    )
    single_axis.set_yticks(
        range(9),
        [f"Part {part}" for part in range(8)] + ["Full build"],
    )
    single_axis.invert_yaxis()
    single_axis.set_title("Eight-chunk transfer pipeline, then one CAGRA build")
    single_axis.grid(axis="x", alpha=0.25)

    pipeline_parts = parts_groups[
        ("network", "prefetch-device", 8, run)
    ]
    for part in pipeline_parts:
        y = part["part"]
        intervals = (
            (
                "Download",
                part["download_start_ms"],
                part["download_end_ms"],
            ),
            (
                "Load / transfer",
                part["load_start_ms"],
                part["load_end_ms"],
            ),
            (
                "CAGRA build",
                part["build_start_ms"],
                part["build_end_ms"],
            ),
        )
        for label, start, end in intervals:
            pipeline_axis.barh(
                y,
                (end - start) / 1000.0,
                left=start / 1000.0,
                height=0.58,
                color=colors[label],
                edgecolor="white",
                linewidth=0.4,
            )
    pipeline_summary = next(
        row
        for row in summary_groups[("network", "prefetch-device", 8)]
        if row["run"] == run
    )
    merge_end = pipeline_summary["total_ms"]
    merge_start = merge_end - pipeline_summary["merge_ms"]
    handle = pipeline_axis.barh(
        8,
        pipeline_summary["merge_ms"] / 1000.0,
        left=merge_start / 1000.0,
        height=0.58,
        color=colors["Fastener merge"],
        edgecolor="white",
        linewidth=0.4,
        label="Fastener merge",
    )
    legend_handles["Fastener merge"] = handle
    pipeline_axis.axvline(
        pipeline_summary["total_ms"] / 1000.0,
        color=COLORS["total"],
        linewidth=1.3,
        linestyle="--",
    )
    pipeline_axis.text(
        pipeline_summary["total_ms"] / 1000.0 - 0.7,
        7.55,
        f"{pipeline_summary['total_ms'] / 1000.0:.2f} s",
        ha="right",
        va="bottom",
        color="black",
        fontsize=10,
    )
    pipeline_axis.set_yticks(
        range(9),
        [f"Part {part}" for part in range(8)] + ["Merge"],
    )
    pipeline_axis.invert_yaxis()
    pipeline_axis.set_title("Eight-chunk pipelined construction")
    pipeline_axis.set_xlabel("Time from download start (s)")
    pipeline_axis.grid(axis="x", alpha=0.25)
    pipeline_axis.set_xlim(
        0.0,
        naive_summary["total_ms"] / 1000.0 * 1.03,
    )
    legend_order = (
        "Download",
        "Load / transfer",
        "CAGRA build",
        "Fastener merge",
    )
    pipeline_axis.legend(
        [legend_handles[label] for label in legend_order],
        legend_order,
        ncol=4,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#d0d0d0",
        loc="upper right",
    )
    fig.suptitle(
        "OpenAI-5M construction strategies",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    summary_groups = load_summary(args.summary_input)
    parts_groups = load_parts(args.parts_input, summary_groups)
    summary = summarize(summary_groups)
    write_aggregate(args.aggregate_output, summary)
    for mode in MODES:
        plot_total(
            args.plots_dir
            / f"streaming_openai5m_total_{mode}_20260709.png",
            summary,
            mode,
        )
        plot_timeline_mode(
            args.plots_dir
            / f"streaming_openai5m_pipeline_{mode}_20260709.png",
            summary_groups,
            parts_groups,
            args.timeline_run,
            mode,
        )
    plot_network_speedup(
        args.plots_dir
        / "streaming_openai5m_network_speedup_20260709.png",
        summary_groups,
        parts_groups,
        args.timeline_run,
    )
    plot_local_overlap(
        args.plots_dir / "streaming_openai5m_local_overlap_20260709.png",
        summary,
    )
    print(f"validated {sum(len(rows) for rows in summary_groups.values())} runs")
    print(f"wrote {args.aggregate_output}")
    print(f"wrote plots under {args.plots_dir}")


if __name__ == "__main__":
    main()

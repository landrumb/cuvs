#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "pandas",
# ]
# ///
"""Plot CAGRA build-compare recall/runtime CSVs.

Example:
    uv run experiments/cpp/plot_cagra_build_compare.py \
        experiments/cpp/closest.csv experiments/cpp/random.csv \
        -o build_time_vs_recall.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_FRACTION_COLORS = {
    0.0: "#440154",
    0.25: "#3b528b",
    0.5: "#21918c",
    0.75: "#5ec962",
    1.0: "#fde725",
}

SELECTION_STYLES = {
    "closest": "-",
    "default": "-",
    "random": "--",
}

DEFAULT_TITLE = "Build Time vs Recall by IVF Fraction"
DEFAULT_XLABEL = "Build Time (s)"
NNDESCENT_TITLE = "NN-Descent Time vs Recall by IVF Fraction"
NNDESCENT_XLABEL = "NN-Descent Time (s)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a build-time vs recall plot from CSVs written by "
            "experiments/cpp/src/cagra_build_compare.cu."
        )
    )
    parser.add_argument(
        "csv",
        nargs="+",
        type=Path,
        help="One or more CAGRA_BUILD_COMPARE CSV output files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cagra_build_compare_recall.png"),
        help="Output image path. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--term-thresh",
        type=float,
        default=None,
        help=(
            "Only plot rows with this termination threshold. If omitted and "
            "multiple thresholds are present, the smallest value is used."
        ),
    )
    parser.add_argument(
        "--time-col",
        default="build_s",
        help="CSV column to use for the x-axis. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--nndescent-time",
        action="store_true",
        help="Plot NN-descent time (nndescent_s) instead of total build time.",
    )
    parser.add_argument(
        "--recall-col",
        default="recall",
        help="CSV column to use for the y-axis. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Point label column. Defaults to max_iters or max_it.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Plot title. Defaults to %(default)r.",
    )
    parser.add_argument(
        "--xlabel",
        default=DEFAULT_XLABEL,
        help="X-axis label. Defaults to %(default)r.",
    )
    parser.add_argument(
        "--ylabel",
        default="Recall@12",
        help="Y-axis label. Defaults to %(default)r.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.75,
        help="Lower y-axis bound. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Optional upper y-axis bound. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(10.0, 6.0),
        help="Figure size in inches. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not annotate each point with its iteration count.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively after writing it.",
    )
    args = parser.parse_args()
    if args.nndescent_time:
        if args.time_col == "build_s":
            args.time_col = "nndescent_s"
        if args.xlabel == DEFAULT_XLABEL:
            args.xlabel = NNDESCENT_XLABEL
        if args.title == DEFAULT_TITLE:
            args.title = NNDESCENT_TITLE
    return args


def load_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep=None, engine="python")
        frame.columns = [column.strip() for column in frame.columns]
        frame["source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one CSV is required")
    return pd.concat(frames, ignore_index=True)


def normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "max_it": "max_iters",
        "ivf_frac": "ivf_frac",
        "ivf_sel": "ivf_sel",
    }
    results = results.rename(columns=rename_map).copy()

    if "succeeded" in results.columns:
        results = results[results["succeeded"].astype(str).str.lower() != "0"]

    for column in (
        "max_iters",
        "term_thresh",
        "ivf_frac",
        "build_s",
        "nndescent_s",
        "recall",
    ):
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    if "ivf_sel" not in results.columns:
        results["ivf_sel"] = "closest"

    results["ivf_sel"] = results["ivf_sel"].astype(str).str.strip().str.lower()
    return results


def filter_threshold(results: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if "term_thresh" not in results.columns:
        return results

    thresholds = sorted(results["term_thresh"].dropna().unique())
    if not thresholds:
        return results

    selected = threshold if threshold is not None else thresholds[0]
    return results[results["term_thresh"].sub(selected).abs() < 1e-12]


def require_columns(results: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in results.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")


def color_for_fraction(fraction: float) -> str:
    for known_fraction, color in DEFAULT_FRACTION_COLORS.items():
        if abs(fraction - known_fraction) < 1e-9:
            return color
    return plt.cm.viridis(float(fraction))


def plot_results(results: pd.DataFrame, args: argparse.Namespace) -> None:
    label_col = args.label_col
    if label_col is None:
        label_col = "max_iters" if "max_iters" in results.columns else "max_it"

    require_columns(
        results,
        [args.time_col, args.recall_col, "ivf_frac", "ivf_sel", label_col],
    )

    results = results.dropna(
        subset=[args.time_col, args.recall_col, "ivf_frac", label_col]
    )
    if results.empty:
        raise ValueError("no plottable rows remain after filtering")

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    groups = results.groupby(["ivf_sel", "ivf_frac"], sort=True)
    for (selection, fraction), group in groups:
        group = group.sort_values(args.time_col)
        color = color_for_fraction(float(fraction))
        linestyle = SELECTION_STYLES.get(str(selection), "-.")

        ax.plot(
            group[args.time_col],
            group[args.recall_col],
            marker="o",
            linestyle=linestyle,
            color=color,
            linewidth=1.6,
            markersize=4.0,
        )

        if not args.no_labels:
            for _, row in group.iterrows():
                ax.annotate(
                    f"{int(row[label_col])}",
                    (row[args.time_col], row[args.recall_col]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color="black",
                )

    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    ax.grid(False)

    fractions = sorted(results["ivf_frac"].dropna().unique())
    fraction_handles = [
        Line2D(
            [0],
            [0],
            color=color_for_fraction(float(fraction)),
            linewidth=2,
            label=f"{fraction:.2f}",
        )
        for fraction in fractions
    ]
    selection_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=SELECTION_STYLES.get(selection, "-."),
            linewidth=2,
            label="Default (closest)" if selection == "closest" else selection.title(),
        )
        for selection in sorted(results["ivf_sel"].dropna().unique())
    ]

    legend1 = ax.legend(
        handles=fraction_handles,
        title="IVF FRACTION",
        loc="center right",
        bbox_to_anchor=(1.0, 0.36),
    )
    legend1.get_title().set_fontweight("bold")
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=selection_handles,
        title="SELECTION",
        loc="center right",
        bbox_to_anchor=(1.0, 0.12),
    )
    legend2.get_title().set_fontweight("bold")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = normalize_results(load_results(args.csv))
    results = filter_threshold(results, args.term_thresh)
    plot_results(results, args)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

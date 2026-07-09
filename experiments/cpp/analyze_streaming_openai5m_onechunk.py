# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Analyze the balanced one-chunk naive-versus-prefetch robustness study."""

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PATHS = ("naive-host", "prefetch-device")
FIELDS = (
    "disk_read_active_ms",
    "h2d_active_ms",
    "load_wall_sum_ms",
    "build_sum_ms",
    "total_ms",
)


def parse_args():
    root = Path(__file__).resolve().parent
    results = root / "merge_api_results"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-input",
        type=Path,
        default=results / "streaming_openai5m_summary_20260709.csv",
    )
    parser.add_argument(
        "--additional-input",
        type=Path,
        default=(
            results
            / "streaming_openai5m_onechunk_robustness_summary_20260709.csv"
        ),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=results / "streaming_openai5m_onechunk_pairs_20260709.csv",
    )
    parser.add_argument(
        "--statistics-output",
        type=Path,
        default=results / "streaming_openai5m_onechunk_statistics_20260709.csv",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=(
            results
            / "plots"
            / "streaming_openai5m_onechunk_robustness_20260709.png"
        ),
    )
    return parser.parse_args()


def load_pairs(paths):
    by_run = {}
    for path in paths:
        with path.open(newline="") as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                if (
                    row["mode"] != "local"
                    or int(row["parts"]) != 1
                    or row["build_path"] not in PATHS
                ):
                    continue
                if (
                    row["dataset"] != "openai_5m"
                    or int(row["rows"]) != 5_000_000
                    or int(row["dim"]) != 1_536
                    or int(row["valid"]) != 1
                ):
                    raise RuntimeError(f"invalid retained row at {path}:{line}")
                run = int(row["run"])
                if row["build_path"] in by_run.setdefault(run, {}):
                    raise RuntimeError(
                        f"duplicate {row['build_path']} row for run {run}"
                    )
                parsed = {"run": run}
                for field in FIELDS:
                    parsed[field] = float(row[field]) / 1000.0
                by_run[run][row["build_path"]] = parsed

    if set(by_run) != set(range(1, 11)):
        raise RuntimeError(f"expected runs 1-10, found {sorted(by_run)}")
    for run, pair in by_run.items():
        if set(pair) != set(PATHS):
            raise RuntimeError(f"incomplete pair for run {run}: {sorted(pair)}")
    return by_run


def mean(pair_map, build_path, field):
    return statistics.mean(
        pair_map[run][build_path][field] for run in range(1, 11)
    )


def calculate(pair_map):
    naive = [
        pair_map[run]["naive-host"]["total_ms"] for run in range(1, 11)
    ]
    prefetch = [
        pair_map[run]["prefetch-device"]["total_ms"]
        for run in range(1, 11)
    ]
    differences = [left - right for left, right in zip(naive, prefetch)]
    n = len(differences)
    difference_mean = statistics.mean(differences)
    difference_sd = statistics.stdev(differences)
    difference_se = difference_sd / math.sqrt(n)
    confidence_low, confidence_high = stats.t.interval(
        0.95,
        n - 1,
        loc=difference_mean,
        scale=difference_se,
    )
    paired_t = stats.ttest_rel(naive, prefetch)
    wilcoxon = stats.wilcoxon(
        differences,
        alternative="two-sided",
        method="exact",
    )
    naive_first = differences[0::2]
    prefetch_first = differences[1::2]
    order_test = stats.ttest_ind(
        naive_first,
        prefetch_first,
        equal_var=False,
    )
    confirmation = differences[2:]
    confirmation_mean = statistics.mean(confirmation)
    confirmation_sd = statistics.stdev(confirmation)
    confirmation_ci = stats.t.interval(
        0.95,
        len(confirmation) - 1,
        loc=confirmation_mean,
        scale=confirmation_sd / math.sqrt(len(confirmation)),
    )
    confirmation_t = stats.ttest_1samp(confirmation, 0.0)
    confirmation_wilcoxon = stats.wilcoxon(
        confirmation,
        method="exact",
    )
    result = {
        "pairs": n,
        "naive_mean_s": statistics.mean(naive),
        "naive_sd_s": statistics.stdev(naive),
        "prefetch_mean_s": statistics.mean(prefetch),
        "prefetch_sd_s": statistics.stdev(prefetch),
        "mean_savings_s": difference_mean,
        "mean_savings_pct": (
            100.0 * difference_mean / statistics.mean(naive)
        ),
        "savings_sd_s": difference_sd,
        "savings_se_s": difference_se,
        "savings_ci95_low_s": confidence_low,
        "savings_ci95_high_s": confidence_high,
        "paired_t": paired_t.statistic,
        "paired_t_df": paired_t.df,
        "paired_t_p_two_sided": paired_t.pvalue,
        "cohen_dz": difference_mean / difference_sd,
        "wilcoxon_statistic": wilcoxon.statistic,
        "wilcoxon_p_exact_two_sided": wilcoxon.pvalue,
        "pairs_prefetch_faster": sum(value > 0.0 for value in differences),
        "pairs_naive_faster": sum(value < 0.0 for value in differences),
        "naive_first_mean_savings_s": statistics.mean(naive_first),
        "prefetch_first_mean_savings_s": statistics.mean(prefetch_first),
        "order_welch_t": order_test.statistic,
        "order_welch_df": order_test.df,
        "order_welch_p_two_sided": order_test.pvalue,
        "confirmation_pairs": len(confirmation),
        "confirmation_mean_savings_s": confirmation_mean,
        "confirmation_savings_sd_s": confirmation_sd,
        "confirmation_ci95_low_s": confirmation_ci[0],
        "confirmation_ci95_high_s": confirmation_ci[1],
        "confirmation_t": confirmation_t.statistic,
        "confirmation_t_df": confirmation_t.df,
        "confirmation_t_p_two_sided": confirmation_t.pvalue,
        "confirmation_wilcoxon_statistic": confirmation_wilcoxon.statistic,
        "confirmation_wilcoxon_p_exact_two_sided": confirmation_wilcoxon.pvalue,
        "confirmation_pairs_prefetch_faster": sum(value > 0 for value in confirmation),
    }
    for field in FIELDS:
        naive_mean = mean(pair_map, "naive-host", field)
        prefetch_mean = mean(pair_map, "prefetch-device", field)
        result[f"naive_{field.removesuffix('_ms')}_mean_s"] = naive_mean
        result[f"prefetch_{field.removesuffix('_ms')}_mean_s"] = prefetch_mean
        result[f"savings_{field.removesuffix('_ms')}_mean_s"] = (
            naive_mean - prefetch_mean
        )
    return result, naive, prefetch, differences


def write_pairs(path, pair_map):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "run",
        "order",
        "naive_total_s",
        "prefetch_total_s",
        "naive_minus_prefetch_s",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in range(1, 11):
            naive = pair_map[run]["naive-host"]["total_ms"]
            prefetch = pair_map[run]["prefetch-device"]["total_ms"]
            writer.writerow(
                {
                    "run": run,
                    "order": (
                        "naive-first" if run % 2 == 1 else "prefetch-first"
                    ),
                    "naive_total_s": f"{naive:.6f}",
                    "prefetch_total_s": f"{prefetch:.6f}",
                    "naive_minus_prefetch_s": f"{naive - prefetch:.6f}",
                }
            )


def write_statistics(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(result),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                field: (
                    value
                    if isinstance(value, int)
                    else f"{float(value):.9f}"
                )
                for field, value in result.items()
            }
        )


def plot(path, result, naive, prefetch, differences):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    runs = list(range(1, 11))
    positions = (0, 1)
    for run, naive_value, prefetch_value in zip(runs, naive, prefetch):
        color = "#00a8d9" if naive_value > prefetch_value else "#d62828"
        axes[0].plot(
            positions,
            (naive_value, prefetch_value),
            color=color,
            alpha=0.65,
            marker="o",
            linewidth=1.5,
        )
    axes[0].set_xticks(positions, ("Naive host", "Device prefetch"))
    axes[0].set_ylabel("Valid final index wall time (s)")
    axes[0].set_title("Paired construction times")
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#00a8d9" if value > 0.0 else "#d62828" for value in differences]
    axes[1].bar(runs, differences, color=colors, width=0.72)
    axes[1].axhline(0.0, color="#6c757d", linewidth=0.9)
    axes[1].axhspan(
        result["savings_ci95_low_s"],
        result["savings_ci95_high_s"],
        color="#00a8d9",
        alpha=0.14,
        label="95% CI for mean",
    )
    axes[1].axhline(
        result["mean_savings_s"],
        color="#172033",
        linewidth=1.6,
        linestyle="--",
        label=f"Mean: {result['mean_savings_s']:.2f} s",
    )
    axes[1].set_xticks(runs)
    axes[1].set_xlabel("Balanced pair")
    axes[1].set_ylabel("Naive minus prefetch (s)")
    axes[1].set_title("Paired prefetch savings")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Cold-local one-chunk prefetch robustness",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    pair_map = load_pairs((args.initial_input, args.additional_input))
    result, naive, prefetch, differences = calculate(pair_map)
    write_pairs(args.pairs_output, pair_map)
    write_statistics(args.statistics_output, result)
    plot(args.plot_output, result, naive, prefetch, differences)
    print(
        f"mean savings {result['mean_savings_s']:.6f} s; "
        f"95% CI [{result['savings_ci95_low_s']:.6f}, "
        f"{result['savings_ci95_high_s']:.6f}]; "
        f"paired t p={result['paired_t_p_two_sided']:.9f}; "
        f"exact Wilcoxon p={result['wilcoxon_p_exact_two_sided']:.9f}"
    )


if __name__ == "__main__":
    main()

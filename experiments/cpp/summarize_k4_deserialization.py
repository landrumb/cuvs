# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Summarize deserialization-inclusive Fastener and rebuild merge timings."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("Wiki-1M", "OpenAI-2M", "YFCC-10M")
PARTS = (2, 4, 8)
IMPLEMENTATIONS = ("fastener-repeat2", "rebuild")
METRICS = (
    "deserialize_ms",
    "merge_api_ms",
    "other_overhead_ms",
    "total_overhead_ms",
    "end_to_end_ms",
)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "merge_api_results" / "deserialization_merge_breakdown.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "merge_api_results"
            / "deserialization_merge_breakdown_summary.csv"
        ),
    )
    return parser.parse_args()


def load(path):
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], int(row["parts"]), row["implementation"])
            pieces = (
                float(row["deserialize_ms"])
                + float(row["merge_api_ms"])
                + float(row["other_overhead_ms"])
            )
            if abs(pieces - float(row["end_to_end_ms"])) > 0.002:
                raise RuntimeError(f"timing accounting mismatch for {key}: {row}")
            groups[key].append(row)

    expected = {
        (dataset, parts, implementation)
        for dataset in DATASETS
        for parts in PARTS
        for implementation in IMPLEMENTATIONS
    }
    if set(groups) != expected:
        raise RuntimeError(
            f"coverage mismatch: missing={sorted(expected - set(groups))} "
            f"extra={sorted(set(groups) - expected)}"
        )
    if any(len(rows) != 2 for rows in groups.values()):
        raise RuntimeError("expected exactly two timed runs per case and implementation")
    return groups


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def summarize(groups):
    for dataset in DATASETS:
        for parts in PARTS:
            fastener = groups[(dataset, parts, "fastener-repeat2")]
            rebuild = groups[(dataset, parts, "rebuild")]
            serialized_bytes = {int(row["serialized_bytes"]) for row in fastener + rebuild}
            if len(serialized_bytes) != 1:
                raise RuntimeError(f"serialized byte mismatch for {(dataset, parts)}")
            serialized_bytes = serialized_bytes.pop()

            values = {}
            for implementation, rows in (("fastener", fastener), ("rebuild", rebuild)):
                for metric in METRICS:
                    values[f"{implementation}_{metric}"] = mean(rows, metric)

            fastener_total = values["fastener_end_to_end_ms"]
            rebuild_total = values["rebuild_end_to_end_ms"]
            yield {
                "dataset": dataset,
                "parts": parts,
                "runs": len(fastener),
                "serialized_bytes": serialized_bytes,
                "serialized_gib": serialized_bytes / (1024**3),
                **values,
                "fastener_deserialize_pct": 100.0
                * values["fastener_deserialize_ms"]
                / fastener_total,
                "fastener_merge_api_pct": 100.0
                * values["fastener_merge_api_ms"]
                / fastener_total,
                "rebuild_deserialize_pct": 100.0
                * values["rebuild_deserialize_ms"]
                / rebuild_total,
                "rebuild_merge_api_pct": 100.0
                * values["rebuild_merge_api_ms"]
                / rebuild_total,
                "merge_api_speedup": values["rebuild_merge_api_ms"]
                / values["fastener_merge_api_ms"],
                "end_to_end_speedup": rebuild_total / fastener_total,
                "fastener_end_to_end_reduction_pct": 100.0
                * (1.0 - fastener_total / rebuild_total),
            }


def write(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def main():
    args = parse_args()
    groups = load(args.input)
    write(args.output, summarize(groups))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

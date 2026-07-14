#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Validate generated exact neighbor IDs against an existing exact reference."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_ibin(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        rows, cols = struct.unpack("<II", stream.read(8))
    values = np.memmap(path, mode="r", dtype="<i4", offset=8, shape=(rows, cols))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()

    generated = read_ibin(args.generated)
    reference = read_ibin(args.reference)
    if generated.shape[0] != reference.shape[0]:
        raise RuntimeError("generated/reference row count mismatch")
    if generated.shape[1] < args.k or reference.shape[1] < args.k:
        raise RuntimeError("generated/reference ground truth has fewer than k columns")

    generated = generated[:, : args.k]
    reference = reference[:, : args.k]
    exact_positions = float(np.mean(generated == reference))
    overlap_counts = np.empty(generated.shape[0], dtype=np.int32)
    for row in range(generated.shape[0]):
        overlap_counts[row] = len(set(generated[row].tolist()) & set(reference[row].tolist()))
    set_recall = float(np.mean(overlap_counts / args.k))
    exact_set_rows = float(np.mean(overlap_counts == args.k))
    report = {
        "generated": str(args.generated),
        "reference": str(args.reference),
        "rows": int(generated.shape[0]),
        "k": args.k,
        "exact_position_fraction": exact_positions,
        "mean_set_recall": set_recall,
        "exact_set_row_fraction": exact_set_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if set_recall < 0.999:
        raise RuntimeError(f"exact GT validation recall too low: {set_recall:.6f}")


if __name__ == "__main__":
    main()

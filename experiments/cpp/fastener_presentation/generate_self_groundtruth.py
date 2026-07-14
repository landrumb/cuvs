#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Generate exact sampled self-kNN ground truth for partition-quality studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import cupy as cp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def matrix_metadata(path: Path) -> tuple[int, int, np.dtype]:
    with path.open("rb") as stream:
        rows, dim = struct.unpack("<II", stream.read(8))
    payload = path.stat().st_size - 8
    count = rows * dim
    if payload == count:
        dtype = np.dtype("uint8")
    elif payload == count * 4:
        dtype = np.dtype("float32")
    else:
        raise RuntimeError(f"unsupported matrix payload: {path}")
    return rows, dim, dtype


def write_ibin(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<i4", order="C")
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", *values.shape))
        stream.write(values.tobytes(order="C"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rows, dim, dtype = matrix_metadata(args.dataset)
    if args.sample_size < 1 or args.sample_size > rows:
        raise RuntimeError("sample size must be between 1 and dataset rows")
    if args.k < 1 or args.k >= rows:
        raise RuntimeError("k must be positive and smaller than dataset rows")
    if args.batch_size < 1:
        raise RuntimeError("batch size must be positive")

    handle = cp.cuda.device.get_cublas_handle()
    cp.cuda.cublas.setMathMode(handle, cp.cuda.cublas.CUBLAS_DEFAULT_MATH)

    host = np.memmap(args.dataset, mode="r", dtype=dtype, offset=8, shape=(rows, dim))
    started = time.perf_counter()
    dataset = cp.asarray(host, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    upload_seconds = time.perf_counter() - started
    norms = cp.einsum("ij,ij->i", dataset, dataset)

    rng = np.random.default_rng(args.seed)
    sample_ids = np.sort(rng.choice(rows, size=args.sample_size, replace=False)).astype(np.int32)
    neighbors = np.empty((args.sample_size, args.k), dtype=np.int32)
    distances = np.empty((args.sample_size, args.k), dtype=np.float32)

    search_started = time.perf_counter()
    for start in range(0, args.sample_size, args.batch_size):
        end = min(start + args.batch_size, args.sample_size)
        batch_ids_host = sample_ids[start:end]
        batch_ids = cp.asarray(batch_ids_host)
        queries = dataset[batch_ids]
        batch_distances = norms[batch_ids, None] + norms[None, :] - 2.0 * (queries @ dataset.T)
        cp.maximum(batch_distances, 0.0, out=batch_distances)
        batch_distances[cp.arange(end - start), batch_ids] = cp.inf
        candidates = cp.argpartition(batch_distances, args.k - 1, axis=1)[:, : args.k]
        candidate_distances = cp.take_along_axis(batch_distances, candidates, axis=1)
        order = cp.argsort(candidate_distances, axis=1)
        candidates = cp.take_along_axis(candidates, order, axis=1)
        candidate_distances = cp.take_along_axis(candidate_distances, order, axis=1)
        neighbors[start:end] = cp.asnumpy(candidates).astype(np.int32, copy=False)
        distances[start:end] = cp.asnumpy(candidate_distances)
        elapsed = time.perf_counter() - search_started
        print(f"{args.label}: exact self-kNN {end}/{args.sample_size} ({elapsed:.1f}s)", flush=True)

    search_seconds = time.perf_counter() - search_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / f"{args.label}_sample_ids.ibin"
    neighbors_path = args.output_dir / f"{args.label}_self_gt_k{args.k}.ibin"
    distances_path = args.output_dir / f"{args.label}_self_gt_k{args.k}_distances.fbin"
    metadata_path = args.output_dir / f"{args.label}_self_gt_k{args.k}.json"
    write_ibin(ids_path, sample_ids[:, None])
    write_ibin(neighbors_path, neighbors)
    with distances_path.open("wb") as stream:
        stream.write(struct.pack("<II", *distances.shape))
        stream.write(np.asarray(distances, dtype="<f4", order="C").tobytes(order="C"))
    metadata = {
        "dataset": str(args.dataset), "dataset_rows": rows, "dimension": dim,
        "source_dtype": dtype.name, "distance_dtype": "float32",
        "sample_size": args.sample_size, "k": args.k, "seed": args.seed,
        "batch_size": args.batch_size, "self_edges_excluded": True,
        "method": "exact exhaustive squared-L2 GPU search",
        "tf32_enabled": False,
        "upload_seconds": upload_seconds, "search_seconds": search_seconds,
        "sample_ids": ids_path.name, "neighbors": neighbors_path.name,
        "distances": distances_path.name,
        "sample_ids_sha256": sha256(ids_path),
        "neighbors_sha256": sha256(neighbors_path),
        "distances_sha256": sha256(distances_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()

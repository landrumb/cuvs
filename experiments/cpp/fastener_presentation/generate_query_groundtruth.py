#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Generate exact query kNN ground truth for a prefix of a binary dataset."""

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
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
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


def write_matrix(path: Path, values: np.ndarray, dtype: str) -> None:
    values = np.asarray(values, dtype=dtype, order="C")
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
    file_rows, dim, dataset_dtype = matrix_metadata(args.dataset)
    query_rows, query_dim, query_dtype = matrix_metadata(args.queries)
    if args.rows < 1 or args.rows > file_rows:
        raise RuntimeError("--rows must select a non-empty dataset prefix")
    if query_dim != dim:
        raise RuntimeError("dataset/query dimension mismatch")
    if args.k < 1 or args.k > args.rows:
        raise RuntimeError("k must be between 1 and selected rows")
    if args.batch_size < 1:
        raise RuntimeError("batch size must be positive")

    handle = cp.cuda.device.get_cublas_handle()
    cp.cuda.cublas.setMathMode(handle, cp.cuda.cublas.CUBLAS_DEFAULT_MATH)

    dataset_host = np.memmap(
        args.dataset, mode="r", dtype=dataset_dtype, offset=8, shape=(file_rows, dim)
    )[: args.rows]
    queries_host = np.memmap(
        args.queries, mode="r", dtype=query_dtype, offset=8, shape=(query_rows, dim)
    )
    upload_started = time.perf_counter()
    dataset = cp.asarray(dataset_host, dtype=cp.float32)
    queries = cp.asarray(queries_host, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    upload_seconds = time.perf_counter() - upload_started
    dataset_norms = cp.einsum("ij,ij->i", dataset, dataset)
    query_norms = cp.einsum("ij,ij->i", queries, queries)

    neighbors = np.empty((query_rows, args.k), dtype=np.int32)
    distances = np.empty((query_rows, args.k), dtype=np.float32)
    search_started = time.perf_counter()
    for start in range(0, query_rows, args.batch_size):
        end = min(start + args.batch_size, query_rows)
        batch_distances = (
            query_norms[start:end, None]
            + dataset_norms[None, :]
            - 2.0 * (queries[start:end] @ dataset.T)
        )
        cp.maximum(batch_distances, 0.0, out=batch_distances)
        candidates = cp.argpartition(batch_distances, args.k - 1, axis=1)[:, : args.k]
        candidate_distances = cp.take_along_axis(batch_distances, candidates, axis=1)
        order = cp.argsort(candidate_distances, axis=1)
        candidates = cp.take_along_axis(candidates, order, axis=1)
        candidate_distances = cp.take_along_axis(candidate_distances, order, axis=1)
        neighbors[start:end] = cp.asnumpy(candidates).astype(np.int32, copy=False)
        distances[start:end] = cp.asnumpy(candidate_distances)
        if end == query_rows or end % 512 == 0:
            elapsed = time.perf_counter() - search_started
            print(f"rows={args.rows}: exact query kNN {end}/{query_rows} ({elapsed:.1f}s)", flush=True)

    search_seconds = time.perf_counter() - search_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    distances_path = args.output.with_name(args.output.stem + "_distances.fbin")
    metadata_path = args.output.with_suffix(".json")
    write_matrix(args.output, neighbors, "<i4")
    write_matrix(distances_path, distances, "<f4")
    metadata = {
        "dataset": str(args.dataset), "selected_rows": args.rows,
        "dataset_file_rows": file_rows, "queries": str(args.queries),
        "query_rows": query_rows, "dimension": dim,
        "dataset_dtype": dataset_dtype.name, "query_dtype": query_dtype.name,
        "distance_dtype": "float32", "k": args.k, "batch_size": args.batch_size,
        "method": "exact exhaustive squared-L2 GPU search",
        "tf32_enabled": False,
        "upload_seconds": upload_seconds, "search_seconds": search_seconds,
        "neighbors": args.output.name, "distances": distances_path.name,
        "neighbors_sha256": sha256(args.output),
        "distances_sha256": sha256(distances_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()

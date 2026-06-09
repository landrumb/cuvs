# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Plot CAGRA build time: NN-Descent vs IVF-PQ across dataset sizes."""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else (
    "/raid/blandrum/local_datasets/wiki_all_10M/build_algo_compare.csv"
)
out_path = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/coder/cuvs/experiments/cpp/build_algo_compare.png"
)

# best (min over runs) per (rows, algo)
best = defaultdict(lambda: float("inf"))
with open(csv_path) as f:
    for row in csv.DictReader(f):
        ms = float(row["build_ms"])
        if ms != ms:  # NaN
            continue
        key = (int(row["rows"]), row["algo"])
        best[key] = min(best[key], ms)

rows = sorted({r for (r, _) in best})
ivf = [best.get((r, "ivf_pq"), float("nan")) for r in rows]
nnd = [best.get((r, "nn_descent"), float("nan")) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(rows, ivf, "o-", label="IVF-PQ")
ax1.plot(rows, nnd, "s-", label="NN-Descent")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("dataset rows")
ax1.set_ylabel("CAGRA build time (ms)")
ax1.set_title("CAGRA graph-build time vs dataset size\nwiki_all_10M, 768-d, L2, deg 64/128")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

ratio = [n / i if (i == i and n == n and i > 0) else float("nan")
         for i, n in zip(ivf, nnd)]
ax2.plot(rows, ratio, "d-", color="purple")
ax2.axhline(1.0, color="k", ls="--", lw=1, label="parity (nnd == ivf-pq)")
ax2.set_xscale("log")
ax2.set_xlabel("dataset rows")
ax2.set_ylabel("NN-Descent / IVF-PQ build time")
ax2.set_title("Speed ratio (>1 means NN-Descent slower)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

fig.tight_layout()
fig.savefig(out_path, dpi=120)
print("wrote", out_path)
for r, i, n, q in zip(rows, ivf, nnd, ratio):
    print(f"{r:>10}  ivf_pq={i:>10.1f}ms  nn_descent={n:>10.1f}ms  nnd/ivf={q:.3f}")

# Fastener merge Nsight Compute findings

Date: 2026-07-09

Branch: `cagra-fastener-device-memory`

Commit: `5c4cb166` (`Use CUDA VMM for CAGRA merge datasets`)

## Summary

The production-shaped float32 merge is primarily limited by two passes over the dataset:

1. pivot-tree side assignment is **46.96%** of measured kernel time; and
2. graph distance calculation/sort is **14.73%**.

There is no general occupancy problem. The important distinction is:

- pivot assignment has 62.5% theoretical occupancy, 61.4% achieved occupancy, only 6.3% SM issue
  activity, and 95.0% of not-issued PC samples attributed to long-scoreboard stalls;
- the distance/sort kernel is also latency/bandwidth limited: 79.6% DRAM throughput, 23.6% L2 hit
  rate, and 91.2% long-scoreboard stalls;
- leaf gathering, leaf GEMM, and optimizer pruning are productive. In particular, the cuBLAS GEMM
  achieves 87.6% SM throughput despite only 12.5% occupancy, so raising its occupancy is not a useful
  goal by itself.

The highest-value work is therefore to improve or avoid the random dataset traffic in pivot
assignment, then reduce the amount of exact distance traffic sent through `kern_sort`. Leaf-gather
index arithmetic is a reasonable second-tier target. The GEMM, optimizer prune, stable scatter,
repeat union, and append kernels do not currently need attention.

## Workload and method

### Hardware and software

| item | value |
| --- | --- |
| GPU | NVIDIA H100 PCIe, 80 GB, compute capability 9.0, 114 SMs |
| Driver | 580.126.20 |
| CUDA compiler | 13.2.78 |
| Nsight Compute | 2026.2.1.0, build 38283040 |
| Build | Release, `-O3 -lineinfo`, `sm_90a` |

The Ubuntu-provided Nsight Compute 2022.4 could connect to the process but could not prepare this
CUDA 13.2 kernel for profiling. The report uses the user-local NVIDIA 2026.2.1 Arm/SBSA package;
no system packages were changed.

### Benchmark shape

| parameter | value |
| --- | --- |
| dataset | Wiki-1M, 1,000,000 x 768 float32 |
| fan-in | 8 equal parts |
| implementation | k4 Fastener scaffold |
| input/intermediate/output graph degree | 64 / 64 / 64 |
| repeats | 8 |
| neighbors per repeat | 4 |
| pre-optimizer unique cap | 64 |

`CAGRA_MERGE_API_BENCH --profile-merge` brackets only the merge with `cudaProfilerStart()` and
`cudaProfilerStop()`. Partition-index construction, query search, and recall calculation are not in
the profiled range.

Collection had two stages:

1. collect `gpu__time_duration.sum` for every launch in the merge and aggregate by demangled kernel
   name; and
2. collect the `detailed` set for one representative hot launch from each of the five kernel families
   covering 95.7% of total kernel time.

Nsight Compute controls GPU clocks and replay adds substantial wall-clock overhead. Absolute
`merge_ms` from a profiled run is therefore not a benchmark result. Duration shares and the per-kernel
counters below are the useful profiler outputs.

### Unprofiled sanity run

| merge | search | Recall@12 | QPS |
| ---: | ---: | ---: | ---: |
| 1,078.13 ms | 143.215 ms | 0.992042 | 69,825 |

This is a single sanity run, not a variance study. It is consistent with the approximately 1.05 s
8-way Wiki-1M runs already documented for this branch.

## Kernel-time breakdown

The duration sweep observed 943 launches with 22 unique demangled kernel names and 948.465 ms of
summed kernel duration.

| kernel family | launches | cumulative time | share | largest launch |
| --- | ---: | ---: | ---: | ---: |
| `pivot_assign_sides_kernel<float>` | 324 | 445.366 ms | 46.96% | 2.898 ms |
| cuBLAS FP32 leaf Gram GEMM, 256x64x8 tile | 32 | 172.513 ms | 18.19% | 5.412 ms |
| `kern_sort<float, uint32_t, 4>` | 1 | 139.670 ms | 14.73% | 139.670 ms |
| `gather_float_leaf_vectors_kernel` | 40 | 92.616 ms | 9.76% | 2.871 ms |
| optimizer `kern_fused_prune<uint32_t, 4>` | 4 | 57.523 ms | 6.06% | 15.490 ms |
| `leaf_gram_knn_kernel<float, 4>` | 40 | 9.085 ms | 0.96% | 0.272 ms |
| optimizer reverse-graph construction | 64 | 7.385 ms | 0.78% | 0.148 ms |
| optimizer graph merge | 4 | 5.425 ms | 0.57% | 1.427 ms |
| cap/dedup | 1 | 5.375 ms | 0.57% | 5.375 ms |
| stable scatter | 324 | 4.613 ms | 0.49% | 0.015 ms |
| everything else | 109 | 8.894 ms | 0.94% | - |

The first five rows account for **95.7%** of measured kernel time.

## Detailed hotspot counters

Each column below is from a representative large launch, not an average over all launches of that
name. `SM busy` is `sm__throughput`, while `DRAM busy` is the device DRAM throughput relative to peak.

| metric | pivot assignment | leaf gather | leaf GEMM | distance/sort | optimizer prune |
| --- | ---: | ---: | ---: | ---: | ---: |
| representative duration | 2.820 ms | 2.856 ms | 5.404 ms | 141.476 ms | 14.690 ms |
| grid / block | 3,907 / 128 | 1,048,576 / 256 | 8,192 / 128 | 125,000 / 256 | 65,536 / 128 |
| registers/thread | 44 | 32 | 255 | 48 | 32 |
| shared memory/block | 1.5 KiB | 1.0 KiB | 31.8 KiB | 1.0 KiB | 3.0 KiB |
| theoretical occupancy | 62.50% | 100% | 12.50% | 62.50% | 100% |
| achieved occupancy | 61.44% | 91.02% | 12.44% | 58.57% | 98.14% |
| waves/SM | 3.43 | 1,149.75 | 35.93 | 219.30 | 35.93 |
| SM busy | 6.32% | 83.82% | 87.64% | 30.00% | 97.14% |
| DRAM busy | 53.73% | 39.70% | 19.59% | 79.58% | 5.06% |
| DRAM read | 1,094 GB/s | 253 GB/s | 298 GB/s | 1,620 GB/s | 99 GB/s |
| DRAM write | 1 GB/s | 556 GB/s | 101 GB/s | 3 GB/s | 5 GB/s |
| L1 sector hit rate | 48.54% | 29.00% | 19.22% | 42.88% | 18.17% |
| L2 sector hit rate | 38.32% | 69.33% | 84.51% | 23.62% | 57.79% |
| issue active | 6.32% | 83.82% | 87.64% | 30.00% | 97.14% |

Not-issued warp stall sampling:

| kernel | dominant sampled stall reasons |
| --- | --- |
| pivot assignment | long scoreboard 95.0%, barrier 4.0% |
| leaf gather | long scoreboard 37.4%, math-pipe throttle 27.5%, wait 20.5% |
| leaf GEMM | short scoreboard 21.7%, long scoreboard 20.6%, wait 16.5%, barrier 16.2% |
| distance/sort | long scoreboard 91.2%, wait 5.3% |
| optimizer prune | wait 44.8%, short scoreboard 26.0%, long scoreboard 12.0% |

PC sampling is statistical, but these results are decisive for the two large latency-bound kernels.

### Reproduction outline

```bash
cmake -S experiments/cpp -B experiments/cpp/build-ncu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90a \
  -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build experiments/cpp/build-ncu --target CAGRA_MERGE_API_BENCH -j16

ncu --metrics gpu__time_duration.sum --profile-from-start off \
  -o wiki1m_8way_timing \
  experiments/cpp/build-ncu/CAGRA_MERGE_API_BENCH \
  --dataset /home/coder/cuvs/datasets/wiki_all_1M/base.1M.fbin \
  --queries /home/coder/cuvs/datasets/wiki_all_1M/queries.fbin \
  --groundtruth /home/coder/cuvs/datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin \
  --output-csv /tmp/wiki1m_ncu.csv --label wiki1m-ncu \
  --implementation k4-scaffold --parts 8 --graph-degree 64 \
  --intermediate-graph-degree 64 --scaffold-repeats 8 --profile-merge
```

The detailed reports use `--set detailed`, `--kernel-name regex:<name>`, and `--launch-count 1` with
the same benchmark arguments. This checkout reused an existing local `cuvs` CMake package during
configuration; a fresh environment may need its normal dependency configuration.

## Sections that could use love

### 1. Pivot assignment: highest priority

Relevant source:

- `cpp/src/neighbors/detail/cagra/cagra_merge_scaffold.cuh:259` (`l2_distance_pair`)
- `cpp/src/neighbors/detail/cagra/cagra_merge_scaffold.cuh:315` (`pivot_assign_sides_kernel`)

This family is almost half of all kernel time. It is not compute-bound: only 6.3% of SM issue capacity
is active. It is latency-bound on loads from the row-major dataset. Nsight Compute attributes 95.0%
of not-issued samples to long scoreboard and flags **96,000,000 excessive global sectors**, 47.5% of
the total theoretical sectors for the representative launch.

The `float4` path at lines 281-300 gives each thread one point. At a fixed loop iteration, adjacent
warp lanes therefore access the same four dimensions from unrelated 768-element rows. The accesses
are individually aligned but not coalesced across the warp.

Recommended experiments, in order:

1. Prototype a warp-per-point mapping. Each lane handles dimensions of one point and the warp reduces
   the two distances. A `float4` per lane would cover 128 consecutive dimensions per iteration and
   turn the point and pivot reads into coalesced transactions. Keep one block per bounded pivot chunk
   and reduce the four/eight warp results into `chunk_left_counts`.
2. If the mapping is promising, use `__launch_bounds__` or refactor address/index temporaries to see
   whether the current 44 registers/thread can drop enough to move theoretical occupancy from 62.5%
   to 75%. Do not force a register cap if spills erase the gain.
3. Measure adaptive work reduction rather than another simple block-size sweep: stop building a row's
   scaffold once it has enough unique cross-origin candidates, or evaluate a quality-preserving
   low-dimensional routing sketch for pivot assignment. Pivot and leaf work both scale almost linearly
   with the eight repeats, so avoided repeats are much more valuable than shaving the tiny scatter.

The branch notes already report that 64- and 256-thread launches did not beat 128 threads. The new
evidence says to change the warp/data mapping, not just the block size.

### 2. Exact distance/sort: high priority

Relevant source:

- `cpp/src/neighbors/detail/cagra/cagra_merge.cuh:435-466`
- `cpp/src/neighbors/detail/cagra/graph_core.cuh:74-173`

The single `kern_sort` launch is 14.7% of all kernel time. The bitonic sort is not the main concern;
the same kernel first computes exact 768-D distances for every candidate at lines 92-155. It reaches
79.6% of DRAM peak, reads 1.62 TB/s, has only a 23.6% L2 hit rate, and spends 91.2% of not-issued
samples on long scoreboard.

Recommended experiments:

1. Instrument the number of unique IDs in each 96-entry `base(64) + scaffold(32)` row before exact
   distances. If overlap is meaningful, deduplicate/compact IDs before `kern_sort` so duplicates do
   not trigger a full 768-D distance calculation. The current cap/dedup occurs after distance sort.
2. Consider a Fastener-specific distance/top-k path that computes exact distance only for compacted
   candidates and writes the 64 retained unique IDs directly. This can also remove the separate 5.4 ms
   cap/dedup launch, though avoiding candidate-vector reads is the real win.
3. As a smaller experiment, cache each source vector once per warp/block. The current loop reloads the
   source row for every candidate. Most of those reloads should hit L1, so this is lower leverage than
   reducing random destination reads, but a 24 KiB/block shared cache for eight 768-D source rows would
   not be the current occupancy limiter (registers already limit the kernel to five blocks/SM).
4. If unique compaction is insufficient, evaluate a cheap shortlist using a quality-validated sketch
   before exact distance. This changes ranking semantics and must be gated on Recall@12 and connectivity.

### 3. Leaf gather: medium priority

Relevant source:

- `cpp/src/neighbors/detail/cagra/cagra_merge_scaffold.cuh:464-489`

Gather is 9.8% of kernel time, but it is already healthy at 91% achieved occupancy and 83.8% SM issue
activity. Nsight Compute identifies ALU as the most utilized pipeline. The hot loop reconstructs
`d`, `local_row`, and `local_leaf` with runtime 64-bit division/modulo at lines 477-480 for every
element.

Recommended experiments:

1. Use a multidimensional launch where blocks/warps map explicitly to `(leaf, local_row)` and threads
   map to contiguous dimensions. That removes most division/modulo work while preserving contiguous
   writes.
2. Only after that, consider a custom indirect-input Gram kernel that gathers into shared memory and
   computes the leaf Gram matrix without writing and rereading the 2 GiB bounded workspace. This could
   attack both gather traffic and GEMM setup, but replacing the efficient cuBLAS kernel is a much larger
   project.

### 4. Repeat-level algorithmic work: quality-gated opportunity

Pivot assignment, gather, and GEMM together account for 74.9% of kernel time and all repeat eight
times. The existing quality work selected eight repeats for a reason, so simply lowering the default
is not supported by this profile. A better target is per-row or per-partition early completion based
on unique cross-origin coverage, followed by the same recall/connectivity suite used to choose r8.

## Sections that do not need attention yet

- **cuBLAS leaf GEMM:** 18.2% of time, but 87.6% SM throughput and 84.5% L2 hit rate. Its 12.5%
  occupancy is register-limited inside a vendor kernel and is sufficient to saturate FP32 execution.
  Do not optimize for occupancy alone. TF32 or another lower-precision path is only worth considering
  with explicit quality/determinism acceptance.
- **Optimizer fused prune:** 6.1% of time, 98.1% achieved occupancy, 97.1% SM issue activity. This is
  already highly utilized.
- **Stable scatter:** 324 launches but only 0.49% cumulative time. Launch fusion would not move the
  end-to-end number unless it also removes a pivot-level dataset pass.
- **Leaf top-k, reverse graph, graph merge, repeat union, padding, and append:** individually below 1%
  or already efficient enough that they are not first-order targets.
- **L2 compression suggestions from Nsight Compute:** the embedding and graph data are dense and the
  detailed reports observed 0% successful compression. The generic estimated speedups should not be
  treated as achievable for this data.

## Limitations

- This profile covers the production float32 Wiki-1M shape on one H100. The uint8/int8 GEMM and gather
  paths, and the float16 direct-leaf fallback, can have different bottlenecks.
- Kernel replay changes profiled wall time. Only the unprofiled sanity run is quoted as merge latency.
- Nsight Compute profiles kernels, not CUDA VMM mapping calls or copy-engine transfers. The VMM dataset
  consolidation is part of the unprofiled merge time but not the 948.465 ms kernel sum.
- The SIFT-1M harness was attempted but its oracle partition CAGRA build failed before the merge because
  the upstream builder rejected invalid/duplicate neighbor output. It was excluded rather than treated
  as a Fastener result.
- Existing unrelated worktree changes in `experiments/cpp/CMakeLists.txt` and streaming-build artifacts
  were left untouched.

## Artifacts

- [All-launch duration report](wiki1m_8way_timing.ncu-rep)
- [All-launch raw CSV](wiki1m_8way_timing_raw.csv)
- [Pivot detailed report](wiki1m_pivot_detailed.ncu-rep)
- [Gather detailed report](wiki1m_gather_detailed.ncu-rep)
- [Leaf GEMM detailed report](wiki1m_leaf_gemm_detailed.ncu-rep)
- [Distance/sort detailed report](wiki1m_sort_detailed.ncu-rep)
- [Optimizer prune detailed report](wiki1m_prune_detailed.ncu-rep)
- `wiki1m_*_detailed_raw.csv`: raw aggregate metrics
- `wiki1m_*_details.txt`: section/rule exports
- `wiki1m_{pivot,gather,sort}_source.txt`: CUDA/SASS source correlation

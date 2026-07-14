# Fastener presentation experiments

This directory is the single artifact root for the experiments requested in
/home/coder/Fastener Presentation TODO.md.

## Reproducibility

- Source worktree: /home/coder/cuvs-fastener-device-memory
- Source base commit: 0aaab512c2210809c4d65de9829a1e8c5b55b812
- GPU: NVIDIA H100 PCIe, 81,559 MiB
- Driver: 580.126.20
- CUDA compiler: 13.2 (V13.2.78)
- Experiment date: 2026-07-09 UTC
- Distance metric: squared L2
- Query metric: recall@12
- CAGRA graph degree: 64
- CAGRA intermediate graph degree: 128
- Search itopk_size: 160
- Fastener production default: binary pivot tree, leaf size 256, k=4 per
  repeat, eight repeats, candidates sorted and capped to degree 64 before
  CAGRA optimization.
- Unless a plot says otherwise, CAGRA partition-index construction time is
  excluded from merge time. The raw CSV retains that excluded time in
  oracle_partition_build_ms_excluded.

The benchmark datasets are:

| Label | Base vectors | Queries | Existing query ground truth |
|---|---|---|---|
| Wiki-1M | /raid/blandrum/local_datasets/wiki_all_1M/base.1M.fbin | /raid/blandrum/local_datasets/wiki_all_1M/queries.fbin | /raid/blandrum/local_datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin |
| OpenAI-2M | /raid/blandrum/openai/openai_base.bin | /raid/blandrum/openai/openai_query.bin | /raid/blandrum/openai/openai-2M.GT |
| YFCC-10M | /raid/blandrum/yfcc/base.10M.u8bin | /raid/blandrum/yfcc/query.public.100K.u8bin | /raid/blandrum/yfcc/unfiltered.GT.public.ibin |

## Artifact layout

- data/: raw benchmark CSV files.
- groundtruth/: generated exact query and sampled self-kNN ground truth,
  distances, checksums, and JSON metadata.
- logs/: complete command output for long-running cases.
- plots/: presentation figures in both PNG and SVG.
- tables/: Markdown endmatter tables.
- generate_query_groundtruth.py: exhaustive prefix query ground truth.
- generate_self_groundtruth.py: exhaustive sampled self-kNN ground truth.
- validate_query_groundtruth.py: exact-generator audit against Wiki-1M.
- run_parameter_sweeps.sh: repeats, leaf-size, and leaf-k sweeps.
- run_wiki_groundtruth.sh: validated Wiki 2M/4M/6M query ground truth.
- run_wiki_scaling.sh: Wiki prefix scaling matrix.
- run_partition_quality.sh: binary/ternary partition retention, implicit
  scaffold quality, and native NN-descent quality.
- run_kmeans_flat_clustering.sh: direct non-hierarchical balanced k-means with
  k = ceil(n / 256).
- run_partition_variants.sh: binary/ternary/native merged-index comparison.
- run_kmeans_merge_pareto.sh: flat balanced and hierarchical Lloyd k-means
  scaffolds at eight-way fan-in.
- plot_parameter_sweeps.py, plot_wiki_scaling.py, and
  plot_partition_experiments.py: strict-coverage plot/table generators.
- plot_kmeans_merge_pareto.py: strict-coverage Pareto plot/table generator.

Plot files ending in `_swapped` are alternate encodings of every figure
where fan-in was originally a line series over a parameter x-axis. They put
the 2/8/128 input-graph fan-in on the x-axis and use the parameter values as
line series; the original figures remain unchanged.

## Ground-truth methodology

The prefix and self-kNN generators perform exhaustive GPU matrix
multiplication over every selected base vector, form squared-L2 distances,
and select the exact smallest 12 values. Self edges are set to infinity for
self-kNN. TF32 is explicitly disabled and the cuBLAS handle uses default
math. Each output JSON records source dimensions, dtypes, timings, seeds,
and SHA-256 hashes.

Before producing the new Wiki 2M, 4M, and 6M files, the same generator is run
on the first 1M rows and compared to the existing Wiki-1M reference. The
validation report is groundtruth/wiki_prefix_1M_validation.json; its mean
top-12 set recall must be at least 0.999 or the runner stops.
The recorded audit reached 0.999975 mean set recall, with exact top-12 sets
for 99.97% of query rows.

## Experiment design

### Parameter sweeps

Each sweep covers Wiki-1M, OpenAI-2M, and YFCC-10M at fan-ins 2, 8, and 128,
with a rebuild baseline:

- repeats: 1, 2, 4, 8, 16, 32 (default 8);
- terminal leaf size: 64, 128, 256, 512 (default 256);
- exact leaf neighbors per repeat: 1, 2, 4, 8, 16 (default 4).

The plots use dataset columns, merge-time panels above recall panels, one
line per fan-in, and a vertical default marker.

### Wiki scaling

The scaling experiment uses prefixes of the full Wiki-10M base at exactly
1M, 2M, 4M, 6M, 8M, and 10M rows. Every size runs fan-ins 2, 8, and 128 for
both rebuild and production-default Fastener. The 2M/4M/6M points use the new
validated exact ground truth.

The third panel reports Fastener recall minus the matched rebuild recall in
absolute percentage points; zero is parity.

### Partition and kNN quality

A deterministic 4,096-row sample per dataset receives exhaustive self-12NN
ground truth. For independent repeat counts 1, 2, 4, 8, 16, and 32, the
quality benchmark records:

- exact self-12NN same-leaf retention and partition time;
- implicit scaffold self-12NN recall;
- implicit cross-partition self-12NN recall for fan-ins 2, 8, and 128;
- unique implicit candidate degree;
- binary and ternary pivot-tree variants.

Native cuVS NN-descent is measured at graph degrees 12, 32, and 64, including
build time, full self-12NN recall, and cross-partition recall. The k=32 native
graph and the ternary tree are also substituted into Fastener and compared
with production binary Fastener and rebuild for merge time and query recall.

The flat-clustering baseline is a direct, non-hierarchical cuVS balanced
k-means fit with k = ceil(n / 256), 20 iterations, and squared-L2 distance.
It is run once on every dataset. The benchmark reports end-to-end fit time,
cluster-size statistics, and the fraction of exact self-12NN pairs assigned
to the same cluster, using the same exhaustive sampled ground truth as the
pivot trees.

### 8-way clustering Pareto comparison

The merge Pareto experiment replaces the pivot tree itself, rather than merely
measuring partition retention. Flat balanced k-means uses
k = ceil(n / 256) with 1, 2, 5, 10, or 20 iterations. Ordinary Lloyd k-means
trees use branching factors 2 and 5, five centroid-update iterations at every
level, no cluster-size balancing, and stop when every leaf has at most the
configured Fastener leaf size of 256.

Every clustering produces exact top-4 cross-input nearest neighbors within
each cluster. Those edges are appended to the eight input CAGRA graphs and
pass through the same distance sort, degree-64 cap, CAGRA optimization,
search, and recall@12 measurement as pivot-tree Fastener. The plot includes
pivot-tree repeats 1, 2, 4, 8, 16, and 32 plus rebuild, and highlights the
overall nondominated time-recall frontier.


## Commands

The plot scripts require Matplotlib, NumPy, and pandas.

From the worktree root:

    bash experiments/cpp/fastener_presentation/run_wiki_groundtruth.sh
    bash experiments/cpp/fastener_presentation/run_wiki_scaling.sh
    bash experiments/cpp/fastener_presentation/run_parameter_sweeps.sh
    bash experiments/cpp/fastener_presentation/run_partition_quality.sh
    bash experiments/cpp/fastener_presentation/run_partition_variants.sh
    bash experiments/cpp/fastener_presentation/run_kmeans_flat_clustering.sh
    bash experiments/cpp/fastener_presentation/run_kmeans_merge_pareto.sh

    python3 experiments/cpp/fastener_presentation/plot_wiki_scaling.py
    python3 experiments/cpp/fastener_presentation/plot_parameter_sweeps.py
    python3 experiments/cpp/fastener_presentation/plot_partition_experiments.py
    python3 experiments/cpp/fastener_presentation/plot_kmeans_merge_pareto.py

All runners are resumable at complete-case or complete-batch boundaries and
refuse ambiguous partial batches.

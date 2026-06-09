# CAGRA Split Merge Construction Experiments

## Dataset

Source split artifacts: `/raid/blandrum/split-wiki`

Current artifact manifest:

- Rows: 1,000,000 total, split into two 500,000-row partitions
- Dim: 768
- Queries: 10,000
- Partition build: CAGRA IVF-PQ, graph degree 64, intermediate graph degree 128
- Existing query-time partition merge recall@12 vs brute force: 0.981892
- Existing query-time partition search+merge: 152.545641 ms

## Harness

New binary: `experiments/cpp/build-nosccache/CAGRA_SPLIT_MERGE_CONSTRUCTION`

The harness loads the split partition indexes, builds an initial merged graph from the partition CAGRA graphs, injects candidates according to the selected variant, and then either:

- runs seeded NN-Descent, sorts the resulting kNN graph by real distance, optimizes with CAGRA, and searches the merged index;
- with `--skip-nnd`, sorts and optimizes the mixed seed graph directly; or
- with `--skip-nnd --skip-optimize`, constructs the searchable merged index directly from the raw mixed seed graph.

CSV output: `/raid/blandrum/split-wiki/merge_construction_results.csv`

For k-way experiments, the harness now discovers sorted `part_*` directories under `--split-dir`. `--part-start` and `--part-count` restrict evaluation to a contiguous subset of parts, and `--groundtruth <neighbors.ibin>` maps source-ID ground truth through `original_ids.ibin` for full-dataset evaluations without running brute force.

Figure: ![Append-k recall and build time](cagra_append_k_recall_build.png)

The build-time subplot uses solid lines for measured end-to-end variant build time and dotted lines for the same rows after subtracting partition graph load.

Important timing note: `variant_build_ms` includes partition index deserialization / graph loading (`load_graph_ms`). A production merge API receiving already-live indexes should compare against `variant_build_ms - load_graph_ms`.

## Full-Size Results

Scratch baseline over the combined 1M-row dataset:

| Label | Build ms | Search ms | Recall@12 | Notes |
|---|---:|---:|---:|---|
| `full_scratch_ivfpq` | 3218.448 | 91.759 | 0.974558 | Public CAGRA build, IVF-PQ graph build |

Selected merge-construction variants:

| Label | Strategy | Insert | Sample | Candidates | Candidate itopk | Build ms | Build ms excl. load | Search ms | Recall@12 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `full_seed_partition_only` | none + seeded NN-Descent | append | 1.00 | 0 | n/a | 57966.563 | 55826.697 | 94.010 | 0.982858 | Quality high, NN-Descent dominates at 54.8 s |
| `full_seed_partition_only_iter5` | none + seeded NN-Descent | append | 1.00 | 0 | n/a | 26069.747 | 23803.210 | 97.997 | 0.919367 | Lower iterations too low recall |
| `full_random_append32_iter5` | random-global + seeded NN-Descent | append | 1.00 | 32 | n/a | 30511.640 | 28284.547 | 99.247 | 0.736817 | Random candidates did not help low-iter NN-Descent |
| `full_nnd_query_append2_canditopk2_iter5` | query opposite index + seeded NN-Descent | append | 1.00 | 2 | 2 | 29454.267 | 27272.807 | 63.568 | 0.951317 | Query candidates help vs random but still below scratch and slow |
| `full_nnd_query_append2_canditopk2_iter1` | query opposite index + seeded NN-Descent | append | 1.00 | 2 | 2 | 23242.754 | 19245.984 | 64.673 | 0.863100 | Minimal NN-Descent damages the direct seed graph |
| `full_nnd_query_append2_canditopk2_iter2` | query opposite index + seeded NN-Descent | append | 1.00 | 2 | 2 | 23248.985 | 20968.159 | 64.565 | 0.880533 | Still far below direct optimize |
| `full_nnd_query_append2_canditopk2_iter3` | query opposite index + seeded NN-Descent | append | 1.00 | 2 | 2 | 25276.628 | 23018.565 | 64.358 | 0.901983 | Improves slowly, still below scratch/direct |
| `full_direct_partition_only` | none, direct optimize | append | 1.00 | 0 | n/a | 2844.602 | 627.699 | 92.715 | 0.489600 | Fast but disconnected across partitions |
| `full_direct_random_append32` | random-global, direct optimize | append | 1.00 | 32 | n/a | 4978.102 | 2754.907 | 99.081 | 0.501500 | Random global edges insufficient |
| `full_direct_query_append1_canditopk1` | query opposite index, direct optimize | append | 1.00 | 1 | 1 | 6024.142 | 3781.620 | 60.546 | 0.971725 | Just below scratch recall |
| `full_direct_query_append2_canditopk2` | query opposite index, direct optimize | append | 1.00 | 2 | 2 | 6171.050 | 3936.804 | 60.295 | 0.975108 | Smallest tested candidate count above scratch recall |
| `full_direct_query_all_append2_canditopk2` | query every other index, direct optimize | append | 1.00 | 2 | 2 | 6291.284 | 4007.985 | 60.265 | 0.975142 | Same strategy as query-one for two partitions |
| `full_direct_query_append4_canditopk4` | query opposite index, direct optimize | append | 1.00 | 4 | 4 | 6432.502 | 4204.649 | 60.009 | 0.977417 | Better recall, modest extra build time |
| `full_direct_query_append8_canditopk8` | query opposite index, direct optimize | append | 1.00 | 8 | 8 | 7052.322 | 4794.978 | 60.071 | 0.980325 | Close to query-time split-merge recall |
| `full_direct_query_append32` | query opposite index, direct optimize | append | 1.00 | 32 | 64 | 13076.654 | 10840.823 | 60.279 | 0.981642 | Highest direct recall, candidate search expensive |
| `full_direct_query_append32_sample10` | query opposite index, direct optimize | append | 0.10 | 32 | 64 | 5871.210 | 3672.896 | 64.640 | 0.873067 | Sampling 10% loses too much recall |
| `full_direct_query_append32_sample50` | query opposite index, direct optimize | append | 0.50 | 32 | 64 | 9215.222 | 6942.650 | 62.537 | 0.959975 | Still below scratch recall |
| `full_direct_query_append32_sample75` | query opposite index, direct optimize | append | 0.75 | 32 | 64 | 11220.628 | 8938.967 | 61.418 | 0.974125 | Nearly scratch recall, still slower |
| `full_direct_query_replace8_random` | query opposite index, direct optimize | replace | 1.00 | 8 | 64 | 14478.177 | 12322.101 | 59.883 | 0.979292 | Fixed degree, but replacement slot selection is costly |
| `full_direct_query_replace8_farthest_sample10` | query opposite index, direct optimize | replace | 0.10 | 8 | 64 | 54774.543 | 52586.737 | 60.187 | 0.966167 | CPU distance ranking dominates |
| `full_direct_query_replace8_nearest_sample10` | query opposite index, direct optimize | replace | 0.10 | 8 | 64 | 55520.264 | 53296.153 | 60.325 | 0.965692 | No quality win; also very expensive |

## One-Point Search Recall

The recall numbers in this log were evaluated with CAGRA's default single-point search initialization: `search_width = 1` and `num_random_samplings = 1`. Candidate-generation searches also used those defaults unless otherwise noted; the `candidate_itopk` sweep only changes `itopk_size`.

Focused promising rows under this caveat:

| Label | Build ms | Build ms excl. load | Search ms | Recall@12 |
|---|---:|---:|---:|---:|
| `full_scratch_ivfpq` | 3218.448 | 3218.448 | 91.759 | 0.974558 |
| `full_direct_query_append2_canditopk2` | 6171.050 | 3936.804 | 60.295 | 0.975108 |
| `full_direct_query_all_append2_canditopk2` | 6291.284 | 4007.985 | 60.265 | 0.975142 |
| `full_direct_query_append2_canditopk8` | 6643.911 | 4422.506 | 60.113 | 0.976617 |
| `full_direct_query_append4_canditopk4` | 6432.502 | 4204.649 | 60.009 | 0.977417 |
| `full_direct_query_append4_canditopk8` | 6724.300 | 4507.140 | 59.943 | 0.978083 |
| `full_direct_query_append8_canditopk8` | 7052.322 | 4794.978 | 60.071 | 0.980325 |
| `full_direct_query_append8_canditopk16` | 7775.009 | 5515.512 | 59.911 | 0.980533 |
| `full_direct_query_append32` | 13076.654 | 10840.823 | 60.279 | 0.981642 |
| `full_noopt_query_append8_canditopk8` | 6572.475 | 4314.049 | 65.048 | 0.976608 |
| `full_noopt_query_append16_canditopk16` | 7819.357 | 5503.502 | 71.327 | 0.980758 |
| `full_noopt_query_append32_canditopk32` | 9842.454 | 7622.091 | 84.193 | 0.983792 |
| `full_direct_query_replace8_random` | 14478.177 | 12322.101 | 59.883 | 0.979292 |
| `full_seed_partition_only` | 57966.563 | 55826.697 | 94.010 | 0.982858 |


## Candidate Beam Sweep

These runs use direct optimize (`--skip-nnd`) with query candidates and full-row append. `Candidate itopk` is the CAGRA search beam used while querying the other partition index to generate candidates.

| Appended candidates | Candidate itopk | Build ms | Candidate ms | Recall@12 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 6024.142 | 3069.753 | 0.971725 |
| 2 | 2 | 6171.050 | 3159.222 | 0.975108 |
| 2 | 4 | 6400.927 | 3366.378 | 0.976317 |
| 2 | 8 | 6643.911 | 3663.194 | 0.976617 |
| 2 | 16 | 7316.540 | 4346.267 | 0.976642 |
| 2 | 64 | 11210.872 | 8233.866 | 0.976167 |
| 4 | 4 | 6432.502 | 3342.929 | 0.977417 |
| 4 | 8 | 6724.300 | 3649.936 | 0.978083 |
| 4 | 16 | 7651.286 | 4472.028 | 0.978333 |
| 4 | 64 | 11346.589 | 8189.255 | 0.977783 |
| 8 | 8 | 7052.322 | 3700.404 | 0.980325 |
| 8 | 16 | 7775.009 | 4418.915 | 0.980533 |
| 16 | 16 | 8350.410 | 4484.880 | 0.980933 |
| 32 | 32 | 10652.506 | 5803.232 | 0.981275 |
| 8 | 64 | 11560.185 | 8221.669 | 0.979933 |

Beam-width takeaway: increasing candidate search beam helps a little up to about 8-16, but recall saturates quickly and larger beams mostly increase candidate search time. Returning more appended candidates has a larger effect than widening the beam after a modest value.

## No-Optimize Append Sweep

These runs add `--skip-optimize` on top of `--skip-nnd`, so the mixed seed graph is searched directly. In append mode that raw graph has degree `64 + append_k`; the optimized runs sort the mixed graph and then CAGRA-prune it back to `graph_degree = 64`. That degree difference is important when interpreting high-k no-optimize recall.

| Appended candidates | Optimized build ms | Optimized search ms | Optimized recall@12 | No-opt build ms | No-opt search ms | No-opt recall@12 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2844.602 | 92.715 | 0.489600 | 2419.519 | 91.723 | 0.489625 |
| 1 | 6024.142 | 60.546 | 0.971725 | 5501.478 | 59.766 | 0.965483 |
| 2 | 6171.050 | 60.295 | 0.975108 | 5858.704 | 60.604 | 0.969192 |
| 4 | 6432.502 | 60.009 | 0.977417 | 6027.949 | 62.076 | 0.973375 |
| 8 | 7052.322 | 60.071 | 0.980325 | 6572.475 | 65.048 | 0.976608 |
| 16 | 8350.410 | 60.033 | 0.980933 | 7819.357 | 71.327 | 0.980758 |
| 32 | 10652.506 | 60.211 | 0.981275 | 9842.454 | 84.193 | 0.983792 |

No-optimize shows that the cross-query append itself carries most of the quality: append1 reaches `0.965483` recall@12, append8 is already above scratch IVF-PQ (`0.976608` vs `0.974558`), and append16 is effectively tied with optimized append16. At low k, optimize is still worth about `0.004` to `0.006` absolute recall for roughly `0.3` to `0.5 s` extra build time. At k32, no-optimize has the highest recall in this sweep (`0.983792`), but it is searching a degree-96 raw graph and search time rises to `84.193 ms`; the optimized degree-64 rows stay near `60 ms` search time.

## Multi-Way wiki_all_10M Results

10-way split artifacts live at `/raid/blandrum/split-wiki-10m-10way`. The split is seeded random k-way over `wiki_all_10M`, with ten 1M-row parts, graph degree 64, intermediate graph degree 128, and IVF-PQ-built CAGRA graphs for each part. Result rows are copied to `experiments/cpp/merge_construction_results_10way.csv`.

Full 10-way rows below use `--groundtruth /raid/blandrum/local_datasets/wiki_all_10M/groundtruth.10M.neighbors.ibin`, so `bf_ms` is ground-truth loading/mapping time rather than brute-force search time.

| Label | Parts | Strategy | Append k | Build ms | Candidate ms | Search ms | Recall@12 | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `full10_noopt_append0` | 10 | no optimize, no cross candidates | 0 | 32496.823 | 0.011 | 97.008 | 0.097592 | Disconnected 10-way floor, about one partition's worth of neighbors |
| `full10_noopt_query_one_append2_canditopk2` | 10 | no optimize, ring query-one | 2 | 53864.023 | 29973.908 | 65.732 | 0.575333 | One target partition per source is not enough, but repairs much of the disconnected floor |
| `full10_noopt_query_all_append2_canditopk2` | 10 | no optimize, query all other parts | 2 | 145437.853 | 121098.490 | 66.259 | 0.760125 | All-to-all cross candidates help but append2 under-doses multi-way connectivity |
| `full10_noopt_query_all_append8_canditopk8` | 10 | no optimize, query all other parts | 8 | 207260.762 | 180505.268 | 71.539 | 0.883258 | Higher raw degree improves recall; candidate generation dominates |
| `full10_noopt_query_all_append16_canditopk16` | 10 | no optimize, query all other parts | 16 | 321798.768 | 290036.318 | 78.367 | 0.914700 | Best full 10-way no-opt row so far, but expensive and searches degree 80 |

Direct optimize over all ten 1M parts ran out of memory during `sort_knn_graph` while trying to allocate another `30.7 GB`. Following up on that with smaller full-subgraph subsets, these rows use brute-force ground truth over only the selected subset. That makes them fair tests of physical merge quality for a smaller fan-in, not directly comparable to the full 10-way ground-truth rows above.

| Label | Parts | Rows | Append k | Build ms | Candidate ms | Search ms | Recall@12 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sub10m_p2_direct_partition_only` | 2 | 2M | 0 | 6239.668 | 0.012 | 96.619 | 0.484942 | Two disconnected 1M graphs; optimize alone does not invent cross edges |
| `sub10m_p2_direct_query_all_append2_canditopk2` | 2 | 2M | 2 | 13047.890 | 6645.077 | 62.936 | 0.949942 | Strong repair, but below the earlier 1M 50/50 append2 result |
| `sub10m_p3_direct_query_all_append2_canditopk2` | 3 | 3M | 2 | 21916.257 | 12794.442 | 63.560 | 0.934550 | Recall drops as fan-in grows; append2 is under-dosed |
| `sub10m_p3_direct_query_all_append8_canditopk8` | 3 | 3M | 8 | 27466.584 | 17308.827 | 63.144 | 0.951658 | More cross candidates recover quality for 3-way merge |
| `sub10m_p4_direct_query_all_append8_canditopk8` | 4 | 4M | 8 | 44157.728 | 30624.291 | 63.554 | 0.945133 | Still fits; recall starts trending down |
| `sub10m_p5_direct_query_all_append8_canditopk8` | 5 | 5M | 8 | 61588.528 | 45877.280 | 63.802 | 0.938825 | Candidate generation dominates build time |
| `sub10m_p7_direct_query_all_append8_canditopk8` | 7 | 7M | 8 | 111262.136 | 89584.769 | 64.186 | 0.928450 | Large fan-in all-to-all append is expensive and lower quality |
| `sub10m_p8_direct_query_all_append8_canditopk8` | 8 | 8M | 8 | 142698.867 | 117408.442 | 64.438 | 0.922150 | Largest direct-optimize subset that fit in this run |

A 9-part direct-optimize append8 run OOMed during `sort_knn_graph` while trying to allocate `27.6 GB`; the 10-part partition-only direct-optimize run OOMed on a `30.7 GB` allocation. This makes small-fan-in physical compaction a practical way to evaluate optimize-based methods, but it also argues against a single all-at-once 10-way optimize in this harness.

Multi-way interpretation: the simple append recipe does transfer, but not cleanly. On 10-way random splits, the number of missing cross-partition routes grows with fan-in, and append2 is not enough. Increasing append k helps substantially, but all-to-all candidate generation scales like `parts * (parts - 1)` and becomes the dominant cost. The sister-repo `lowcross_reprune` and balanced-compaction ideas are directly relevant here: rather than querying every row against every other part, we need boundary/interior dosing, tree-style pairwise or small-fan-in compaction, and likely a cheaper cross-candidate routing policy.

## Findings So Far

- Seeded NN-Descent is not competitive with the current IVF-PQ scratch baseline on this dataset. It can produce high recall (`0.982858`), but the 20-iteration run spends `54.8 s` in NN-Descent alone. One to three iterations actively degrade the strong direct query-append seed graph (`0.863100`, `0.880533`, `0.901983` recall@12), and five iterations is still below scratch even with query-generated candidates (`0.951317`).
- Query-generated cross-partition appends are doing the useful merge work. Without NN-Descent or optimize, append1 already reaches `0.965483` recall@12, append8 reaches `0.976608`, and append32 reaches `0.983792`. The append32 no-optimize row is the highest recall measured here, but it keeps a degree-96 graph and searches in `84.193 ms`.
- Direct optimize remains the better balanced path when the output should be a normal degree-64 CAGRA graph. Querying the opposite partition index for every row and appending 2 to 8 candidates produces scratch-comparable or better recall on 2-way 1M, with merged-index search around `60 ms` versus `91.8 ms` for the scratch IVF-PQ index and `152.5 ms` for query-time split merge. On 10M k-way subsets, direct optimize is feasible up to 8 selected 1M parts in this harness, but recall falls from `0.951658` at 3 parts append8 to `0.922150` at 8 parts append8.
- No-optimize is a plausible alternative only if the larger raw degree is acceptable. It saves the sort/optimize work and can improve high-k recall, but search time climbs with append k (`65.0 ms` at k8, `71.3 ms` at k16, `84.2 ms` at k32 on 2-way 1M; `78.4 ms` at k16 on full 10-way 10M). Full 10-way no-optimize append16 reaches `0.914700` recall@12, but spends `290 s` in all-to-all candidate generation.
- Multi-way all-at-once merging is harder than the 2-way case. Query-all append2 reaches only `0.760125` recall@12 on the full 10-way 10M split without optimize, and direct optimize over all 10 parts OOMs in the current harness. Smaller fan-in experiments show that append8 plus optimize is viable and reasonably strong, which points toward tree-style compaction instead of a single 10-way optimize pass.
- Random global candidates are ineffective for direct optimization and low-iteration NN-Descent on this split. Recall stays near `0.50` direct and `0.74` with five NN-Descent iterations.
- Sampling rows hurts direct-query append quality. At 75% sampling, append32 nearly matches scratch (`0.974125` vs `0.974558`), but it is still slower than scratch. Lower samples are clearly below scratch recall.
- Append is better than replacement in the current prototype. Random replacement can reach good recall, but per-row replacement slot selection is slower than append. Farthest/nearest replacement computed on CPU is not viable as implemented.
- The best measured quality/time tradeoff for a degree-64 output is still `full_direct_query_append2_canditopk2`: recall@12 `0.975108`, build `6.17 s` including graph load, or `3.94 s` excluding graph load. That is close to but still slower than scratch IVF-PQ build (`3.22 s`).

## Next Things To Try

- Avoid deserialization in the timing path by running the merge experiment on live indexes, or subtract `load_graph_ms` consistently when comparing merge API costs.
- Optimize candidate generation: separate candidate search params, lower `itopk`, larger or direct device batches, and maybe a specialized all-points cross-index search path that avoids per-batch host gathers.
- Try appending 2 to 4 query candidates plus a cheap graph connectivity post-pass instead of NN-Descent.
- For multi-way merges, prioritize a tree or tournament compaction schedule with fan-in no larger than 4-8 parts, since direct optimize fits up to 8 1M parts here but OOMs at 9-10 parts.
- Port the sister-repo low-cross idea into this harness: light cross-search all rows, classify boundary rows by cross/within distance ratio, then run richer cross-search only for boundary rows before optimize.
- Separate the no-optimize effect from the larger-degree effect by pruning the raw `64 + append_k` graph back to 64 neighbors without full CAGRA optimize, and by comparing optimized output degrees 80 and 96 against the no-optimize k16/k32 rows.
- If seeded NN-Descent remains in scope, add an early-stop / low-iteration mode designed for already-good seeds; the stock iteration cost is too high here.

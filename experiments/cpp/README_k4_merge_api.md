# CAGRA k=4 scaffold merge API

This worktree changes the physical `cuvs::neighbors::cagra::merge()` implementation for eligible
merges. Unfiltered `L2Expanded` merges of at least two attached, uncompressed CAGRA indexes now:

1. concatenate the input datasets on the device;
2. preserve each input graph as a disconnected, offset-adjusted base graph;
3. build one deterministic pivot tree (seed 1234) with leaf size 128;
4. connect every point to its four nearest leaf points originating in another input graph;
5. append those four scaffold neighbors to the base graph;
6. distance-sort the dense graph in place;
7. run the existing CAGRA graph optimizer to the requested `index_params.graph_degree`; and
8. return an index that owns both its aligned dataset and device graph.

The scaffold uses stable binary scatter rather than a global radix sort at every pivot level.
Per-chunk counts preserve the checkpoint's active-first, stable ordering while transferring only
compact count and offset arrays. Float32 and uint8 distance loops use aligned four-element loads,
and the pivot kernel computes both pivot distances while loading each point once. Uint8 data stays
native byte storage throughout partition construction, scaffold construction, optimization, and
search.

The optimizer writes directly to its final device matrix. The returned index takes ownership of
that matrix, avoiding optimizer host writeback and the subsequent host-to-device graph copy. The
merge also uses an in-place device sort, avoiding scratch copies of an already-device-resident
dataset and graph.

The rebuild implementation remains the fallback for filters, metrics other than `L2Expanded`
(subject to existing rebuild support), compressed indexes, fewer than two inputs, unsupported
degree combinations, and device allocation failure. The public merge function signature is
unchanged.

## Validation

- Release instantiations compile with warnings-as-errors for float32, float16, int8, and uint8.
- A leaf-64 fixed-input comparison found zero differences across 131,072 float32 and 131,072
  uint8 scaffold entries versus the checkpoint global-sort path.
- Runtime smoke tests at the current leaf-128 default for all four datatypes verify graph bounds,
  cross-origin edges, owned dataset and graph lifetimes, graph-only output, and successful CAGRA
  search.
- The leaf-64 checkpoint passed the existing float32 and uint8 merge suites. The leaf-64 optimized
  objects additionally pass seven focused upstream cases covering L2 device input, L2 host input,
  and the InnerProduct rebuild fallback.
- All nine leaf-64 dataset/fan-in benchmarks and all 30 leaf-size sweep runs completed with the
  complete query and ground-truth sets.

## Benchmark boundary

`CAGRA_MERGE_API_BENCH` invokes the public `cagra::merge()` call. Input partition indexes are
constructed contiguously in-process so concatenation preserves source IDs. Their construction is
oracular and excluded, as are dataset/query reads and query transfer. `merge_api_e2e_ms` includes
merged-dataset allocation/copy, scaffold construction, base-graph append, distance sort, CAGRA
optimization, and final index attachment.

Search uses CAGRA with k=12, `itopk_size=160`, one warmup, and the median of three repetitions.
Every dataset uses its complete query set and provided ground truth. Measurements were made on an
NVIDIA H100 PCIe 80 GB. Raw retained results are in
[merge_api_results/results.csv](merge_api_results/results.csv); all optimization trials, including
rejected variants, are in
[merge_api_results/optimization_exploration.csv](merge_api_results/optimization_exploration.csv).

## Leaf-64 optimization results versus scratch construction

Parentheses give the optimized leaf-64 k=4 change relative to scratch CAGRA at the same dataset
and fan-in.

| dataset | fan-in | scratch merge | optimized leaf-64 merge | scratch Recall@12 | leaf-64 Recall@12 | scratch QPS | leaf-64 QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 2 | 3.633 s | 0.362 s (10.0x faster) | 0.992233 | 0.990825 (-0.001408) | 71,238 | 68,298 (-4.13%) |
| Wiki-1M | 4 | 3.673 s | 0.389 s (9.44x faster) | 0.992125 | 0.984325 (-0.007800) | 71,137 | 67,664 (-4.88%) |
| Wiki-1M | 8 | 3.567 s | 0.399 s (8.95x faster) | 0.992350 | 0.969192 (-0.023158) | 71,230 | 67,008 (-5.93%) |
| OpenAI-2M | 2 | 12.742 s | 1.730 s (7.37x faster) | 0.967746 | 0.956200 (-0.011546) | 34,357 | 33,431 (-2.70%) |
| OpenAI-2M | 4 | 12.596 s | 1.858 s (6.78x faster) | 0.968196 | 0.939079 (-0.029117) | 34,291 | 33,078 (-3.54%) |
| OpenAI-2M | 8 | 12.719 s | 1.920 s (6.63x faster) | 0.967888 | 0.903054 (-0.064834) | 34,289 | 32,759 (-4.46%) |
| YFCC-10M (uint8) | 2 | 23.759 s | 1.831 s (13.0x faster) | 0.988858 | 0.984067 (-0.004791) | 350,026 | 348,848 (-0.34%) |
| YFCC-10M (uint8) | 4 | 23.616 s | 1.841 s (12.8x faster) | 0.988560 | 0.973110 (-0.015450) | 350,951 | 351,690 (+0.21%) |
| YFCC-10M (uint8) | 8 | 23.681 s | 1.847 s (12.8x faster) | 0.988664 | 0.944563 (-0.044101) | 350,825 | 346,738 (-1.16%) |

## Leaf-64 improvement over the checkpoint

The optimized leaf-64 row gives its actual value; the percentage reduction and recall/QPS
changes are relative to commit `db4260f6`.

| dataset | fan-in | checkpoint merge | optimized leaf-64 merge | checkpoint Recall@12 | leaf-64 Recall@12 | checkpoint QPS | leaf-64 QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 2 | 0.812 s | 0.362 s (-55.35%) | 0.990458 | 0.990825 (+0.000367) | 68,823 | 68,298 (-0.76%) |
| Wiki-1M | 4 | 0.867 s | 0.389 s (-55.13%) | 0.985033 | 0.984325 (-0.000708) | 67,910 | 67,664 (-0.36%) |
| Wiki-1M | 8 | 0.864 s | 0.399 s (-53.87%) | 0.969400 | 0.969192 (-0.000208) | 66,774 | 67,008 (+0.35%) |
| OpenAI-2M | 2 | 2.975 s | 1.730 s (-41.86%) | 0.955571 | 0.956200 (+0.000629) | 33,427 | 33,431 (+0.01%) |
| OpenAI-2M | 4 | 3.151 s | 1.858 s (-41.05%) | 0.938821 | 0.939079 (+0.000258) | 33,049 | 33,078 (+0.09%) |
| OpenAI-2M | 8 | 3.238 s | 1.920 s (-40.72%) | 0.903867 | 0.903054 (-0.000813) | 32,710 | 32,759 (+0.15%) |
| YFCC-10M (uint8) | 2 | 6.892 s | 1.831 s (-73.43%) | 0.984146 | 0.984067 (-0.000079) | 350,068 | 348,848 (-0.35%) |
| YFCC-10M (uint8) | 4 | 6.738 s | 1.841 s (-72.68%) | 0.972943 | 0.973110 (+0.000167) | 346,943 | 351,690 (+1.37%) |
| YFCC-10M (uint8) | 8 | 6.851 s | 1.847 s (-73.04%) | 0.944492 | 0.944563 (+0.000071) | 347,100 | 346,738 (-0.10%) |

Every merge-time result improved. Recall deltas range from -0.000813 to +0.000629 and QPS changes
from -0.76% to +1.37%, consistent with the variation from rebuilding the oracular partition
indexes between runs.

## Optimization exploration

Two-way merge time at each retained stage; all stages use leaf size 64:

| stage | Wiki-1M | OpenAI-2M | YFCC-10M |
| --- | ---: | ---: | ---: |
| fresh checkpoint rerun | 0.822 s | 2.973 s | 6.577 s |
| remove full ID-array host copies | 0.787 s | 2.911 s | 5.311 s |
| replace global sort with stable scatter | 0.736 s | 2.753 s | 3.781 s |
| aligned four-element distance loads | 0.669 s | 2.470 s | 3.428 s |
| fuse both pivot distances | 0.583 s | 2.169 s | 3.420 s |
| device optimizer output and owned graph | 0.377 s | 1.769 s | 1.922 s |
| compact scatter offsets | 0.370 s | 1.761 s | 1.856 s |
| compact uint32 chunk bounds | 0.368 s | 1.758 s | 1.844 s |
| in-place device graph sort | 0.362 s | 1.730 s | 1.831 s |

Rejected experiments:

- RMM/Thrust policy plus reusable chunk storage was 1.5–2.6% slower across all datasets.
- A 20-level pivot cap was faster but reduced Recall@12 by 0.0036–0.0074; 24/32 levels also
  introduced quality uncertainty, so the exact 64-level path remains.
- Native uint8 DP4A increased YFCC time from 3.420 s to 3.471 s.
- Shared pivot caching slowed both float datasets despite a small YFCC improvement.
- 64- and 256-thread pivot launches did not beat 128 threads across all datasets.
- Active-only pivot descriptors were within noise and slightly slower on YFCC.

## Larger-leaf sweep at 8-way fan-in

Eight-way fan-in has the largest recall loss, so the leaf-size experiment prioritizes that case.
The sweep covers leaf sizes 64, 128, 256, 512, and 1024 on all three datasets. Each point is the
arithmetic mean of two independent runs using the complete query and ground-truth sets (10,000
Wiki queries, 20,000 OpenAI queries, and 100,000 YFCC queries). The timing boundary is the same
merge-only `merge_api_e2e_ms` boundary described above; construction of the eight input partition
graphs remains oracular and excluded.

Parentheses give the change from the leaf-64 mean for the same dataset. Actual build time and
Recall@12 are shown first.

| dataset | leaf size | merge build time | Recall@12 |
| --- | ---: | ---: | ---: |
| Wiki-1M | 64 | 0.400 s (baseline) | 0.969054 (baseline) |
| Wiki-1M | 128 | 0.477 s (+19.09%) | 0.974100 (+0.005046) |
| Wiki-1M | 256 | 0.652 s (+62.97%) | 0.976913 (+0.007859) |
| Wiki-1M | 512 | 1.021 s (+155.24%) | 0.974617 (+0.005563) |
| Wiki-1M | 1024 | 1.741 s (+335.07%) | 0.963480 (-0.005574) |
| OpenAI-2M | 64 | 1.917 s (baseline) | 0.904121 (baseline) |
| OpenAI-2M | 128 | 2.375 s (+23.92%) | 0.913361 (+0.009240) |
| OpenAI-2M | 256 | 3.346 s (+74.53%) | 0.920098 (+0.015977) |
| OpenAI-2M | 512 | 5.330 s (+178.06%) | 0.920044 (+0.015923) |
| OpenAI-2M | 1024 | 9.379 s (+389.27%) | 0.906021 (+0.001900) |
| YFCC-10M (uint8) | 64 | 1.851 s (baseline) | 0.944352 (baseline) |
| YFCC-10M (uint8) | 128 | 1.835 s (-0.88%) | 0.950997 (+0.006644) |
| YFCC-10M (uint8) | 256 | 1.890 s (+2.09%) | 0.955341 (+0.010989) |
| YFCC-10M (uint8) | 512 | 2.203 s (+18.98%) | 0.954311 (+0.009959) |
| YFCC-10M (uint8) | 1024 | 2.748 s (+48.44%) | 0.942174 (-0.002178) |

![Eight-way leaf-size build-time and recall tradeoff](merge_api_results/plots/k4_leaf_size_8way.png)

The dashed horizontal lines are the leaf-64 baselines, the dotted vertical line marks leaf 256,
and error bars span the two runs. Raw measurements are in
[merge_api_results/leaf_size_8way.csv](merge_api_results/leaf_size_8way.csv); the figure is
reproducible with [plot_k4_leaf_size.py](plot_k4_leaf_size.py).

The tradeoff is consistent across datasets:

- Leaf 128 is the balanced setting. Recall improves by 0.005046 on Wiki, 0.009240 on OpenAI, and
  0.006644 on YFCC. The float32 merge cost rises by 19.09–23.92%, while native-uint8 YFCC is
  0.88% faster than leaf 64.
- Leaf 256 gives the highest mean recall on every dataset: +0.007859 on Wiki, +0.015977 on OpenAI,
  and +0.010989 on YFCC. Its merge-time cost is +62.97%, +74.53%, and only +2.09% respectively.
- Leaf 512 adds substantial work without improving on leaf 256. Leaf 1024 is worse still: it costs
  4.35x on Wiki and 4.89x on OpenAI, and recall falls below leaf 64 on Wiki and YFCC.
- The likely cost mechanism is that doubling the leaf removes roughly one pivot level but doubles
  the per-point leaf candidate comparisons. This favors low-dimensional native uint8 YFCC more
  than the 768/1536-dimensional float datasets. The post-256 recall reversal suggests that forcing
  some farther cross-origin bridges at smaller leaves helps navigation after CAGRA optimization;
  this interpretation is an inference from the measured curve.

Based on this sweep, the production default is now 128. Leaf 256 remains a recall-oriented
alternative; sizes above 256 are dominated.

## Nsight Systems profile

The final leaf-64 two-way Wiki capture measures 368.887 ms profiled versus 855.152 ms at the
leaf-64 checkpoint. The ordinary leaf-64 benchmark improves from 811.624 ms to 362.392 ms
(2.24x).

| item | checkpoint | optimized |
| --- | ---: | ---: |
| pivot assignment kernels | 195.174 ms | 63.479 ms |
| leaf cross-origin k-NN | 56.405 ms | 42.194 ms |
| distance sort | 103.708 ms | 104.138 ms |
| CAGRA optimize NVTX range | 312.464 ms | 111.255 ms |
| `cudaFree` API time | 199.656 ms | 3.294 ms |
| device-to-host data | 912.000 MB | 4.051 MB |
| host-to-device data | 800.668 MB | 32.668 MB |
| device-to-device data | 6,688.000 MB | 3,107.157 MB |

The remaining dominant kernels are distance sort (104.138 ms), optimizer pruning (93.929 ms),
pivot assignment (63.479 ms), and leaf k-NN (42.194 ms). See the
[profile report](merge_api_results/nsys/README.md), the
[checkpoint capture](merge_api_results/nsys/wiki1m_2way_k4_2026.nsys-rep), and the
[optimized capture](merge_api_results/nsys/wiki1m_2way_k4_optimized.nsys-rep).

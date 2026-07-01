# CAGRA k=4 scaffold merge API

This worktree changes the existing physical cuvs::neighbors::cagra::merge() implementation for
eligible merges. Unfiltered L2 merges of at least two attached, uncompressed CAGRA indexes now:

1. concatenate the input datasets on the device;
2. preserve each input graph as a disconnected, offset-adjusted base graph;
3. build one deterministic pivot tree (seed 1234) over the merged vectors with leaf size 64;
4. connect every point to its four nearest leaf points originating in another input graph;
5. append those four scaffold neighbors to the base graph;
6. distance-sort the dense graph;
7. run the existing CAGRA graph optimizer to the requested index_params.graph_degree; and
8. return an index with an owned/aligned dataset and owned device graph.

The scaffold distance kernels are templated on the input type. YFCC therefore remains uint8_t
through partition construction, scaffold construction, graph optimization, and search; it is not
converted to float storage.

The prior rebuild implementation remains the fallback for filters, non-L2 metrics, compressed
indexes, fewer than two inputs, unsupported degree combinations, and device allocation failure.
The public function signature is unchanged.

## Validation

- Strict release instantiation builds passed with warnings-as-errors for float, half, int8_t, and uint8_t.
- A runtime smoke test merged two indexes for each datatype, checked graph bounds and cross-origin
  edges, verified dataset ownership, and completed a CAGRA search.
- Existing upstream merge gtests passed for float32 and uint8, including device-backed and
  host-backed L2 inputs with a 1-D (alignment-sensitive) dataset.
- Existing InnerProduct merge gtests passed through the rebuild fallback.

## Benchmark boundary

CAGRA_MERGE_API_BENCH invokes the same public cagra::merge() call in both binaries. The baseline
binary resolves the current rebuild implementation from the unmodified cuVS build; the comparison
binary resolves the implementation in this worktree.

Input partition indexes are constructed contiguously in-process so concatenation preserves source
IDs. Their construction is oracular and excluded. Dataset/query file reads, query transfer, and
partition-index construction are also excluded. merge_api_e2e_ms times the complete public merge
call and includes merged-dataset allocation/copy, graph work, and final index attachment. For the
k=4 implementation that means scaffold construction, base-graph append, distance sort, and CAGRA
optimization. For rebuild it means the current scratch graph construction.

Search uses CAGRA with k=12, itopk_size=160, a warmup, and the median of three timed repetitions.
Every dataset uses its complete query set and provided ground truth. Measurements were made on an
NVIDIA H100 PCIe 80 GB. Raw results are in
[merge_api_results/results.csv](merge_api_results/results.csv).

## Results

The k=4 row gives its actual merge time, Recall@12, and QPS; its speedup or change relative to the
paired rebuild run is in parentheses.

| dataset | fan-in | rebuild merge | k=4 merge | rebuild Recall@12 | k=4 Recall@12 | rebuild QPS | k=4 QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 2 | 3.633 s | 0.812 s (4.48x faster) | 0.992233 | 0.990458 (-0.001775) | 71,238 | 68,823 (-3.39%) |
| Wiki-1M | 4 | 3.673 s | 0.867 s (4.24x faster) | 0.992125 | 0.985033 (-0.007092) | 71,137 | 67,910 (-4.54%) |
| Wiki-1M | 8 | 3.567 s | 0.864 s (4.13x faster) | 0.992350 | 0.969400 (-0.022950) | 71,230 | 66,774 (-6.26%) |
| OpenAI-2M | 2 | 12.742 s | 2.975 s (4.28x faster) | 0.967746 | 0.955571 (-0.012175) | 34,357 | 33,427 (-2.71%) |
| OpenAI-2M | 4 | 12.596 s | 3.151 s (4.00x faster) | 0.968196 | 0.938821 (-0.029375) | 34,291 | 33,049 (-3.62%) |
| OpenAI-2M | 8 | 12.719 s | 3.238 s (3.93x faster) | 0.967888 | 0.903867 (-0.064021) | 34,289 | 32,710 (-4.60%) |
| YFCC-10M (uint8) | 2 | 23.759 s | 6.892 s (3.45x faster) | 0.988858 | 0.984146 (-0.004712) | 350,026 | 350,068 (+0.01%) |
| YFCC-10M (uint8) | 4 | 23.616 s | 6.738 s (3.50x faster) | 0.988560 | 0.972943 (-0.015617) | 350,951 | 346,943 (-1.14%) |
| YFCC-10M (uint8) | 8 | 23.681 s | 6.851 s (3.46x faster) | 0.988664 | 0.944492 (-0.044172) | 350,825 | 347,100 (-1.06%) |

The construction benefit is stable with fan-in: 3.45-4.48x. The quality cost is not stable. At
2-way the recall loss is 0.001775 on Wiki, 0.012175 on OpenAI, and 0.004712 on YFCC. At 8-way it
grows to 0.022950, 0.064021, and 0.044172, respectively. The current one-pass k=4 scaffold is
therefore a strong low-latency merge for small fan-in, but it is not a recall-equivalent replacement
for rebuild at eight inputs.

## Nsight Systems profile

A merge-only Nsight Systems capture of the two-way Wiki-1M k=4 run measured 855.152 ms versus
811.624 ms without profiling. The capture excludes the oracular partition builds and search. Pivot
assignment (195.174 ms), distance sorting (103.708 ms), optimize pruning (92.967 ms), and leaf
cross-origin k-NN (56.405 ms) account for 95.3% of GPU kernel time. The optimizer's complete NVTX
range is 312.464 ms. Device-to-host and host-to-device transfers consume another 217.506 ms of GPU
memory-operation time, largely around the repeated pivot levels. See the
[profile report](merge_api_results/nsys/README.md) and the
[interactive Nsight report](merge_api_results/nsys/wiki1m_2way_k4_2026.nsys-rep).

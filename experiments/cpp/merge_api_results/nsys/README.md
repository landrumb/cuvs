# Wiki-1M k=4 merge profile

Nsight Systems 2026.3.1 profiled the public `cuvs::neighbors::cagra::merge()` call for the
two-way Wiki-1M k=4 scaffold merge on an NVIDIA H100 PCIe 80 GB. CUDA profiler start/stop calls
bound the capture immediately around the merge, after the two input indexes were constructed and
synchronized. Partition graph construction, file I/O, query transfer, and search are excluded.
The merge itself still includes merged-dataset allocation/copy, scaffold construction, base-graph
append, distance sorting, CAGRA optimization, and final index attachment.

The profiled merge took 855.152 ms. The ordinary benchmark took 811.624 ms, so capture overhead
was 5.36%. Recall@12 was 0.990508 and QPS was 68,881, consistent with the ordinary run's 0.990458
and 68,823 QPS.

## GPU time

GPU kernels accounted for 470.554 ms and CUDA memory operations for 225.393 ms. These categories
sum to 695.947 ms of GPU activity; they should not be added to CUDA API time because API calls can
overlap device work.

| operation | time | share of kernel time | calls |
| --- | ---: | ---: | ---: |
| pivot assignment | 195.174 ms | 41.5% | 50 |
| distance sort | 103.708 ms | 22.0% | 1 |
| optimize fused prune | 92.967 ms | 19.8% | 4 |
| leaf cross-origin k-NN | 56.405 ms | 12.0% | 1 |
| optimize graph merge | 6.308 ms | 1.3% | 4 |
| optimize reverse graph | 5.977 ms | 1.3% | 64 |
| copy partition graphs | 5.970 ms | 1.3% | 2 |
| append scaffold | 0.241 ms | 0.1% | 1 |

The first four kernels consume 448.254 ms, or 95.3% of all kernel time. Pure graph assembly is
small: copying the two input graphs plus appending the scaffold takes about 6.21 ms on the GPU.

| memory operation | time | data moved | calls |
| --- | ---: | ---: | ---: |
| device to host | 155.233 ms | 912.000 MB | 111 |
| host to device | 62.273 ms | 800.668 MB | 123 |
| device to device | 7.545 ms | 6,688.000 MB | 5 |
| memset | 0.342 ms | 32.950 MB | 301 |

The repeated pivot levels are the clearest optimization target: 50 pivot-assignment kernels and
50 `sort_by_key` ranges accompany substantial device/host traffic. Keeping pivot partitioning and
range discovery on the GPU should remove much of that transfer and synchronization cost.

## Host/API and NVTX observations

The CAGRA optimizer NVTX range is 312.464 ms: prune is 150.516 ms, reverse-graph construction is
114.634 ms, and combine is 46.660 ms. CUDA allocation churn is also visible: 123 `cudaMalloc`
calls and 121 `cudaFree` calls, with `cudaFree` occupying 199.656 ms of CUDA API time. CUDA API
time is not an additive breakdown of merge wall time.

## Artifacts and reproduction

- `wiki1m_2way_k4_2026.nsys-rep`: interactive Nsight Systems report.
- `wiki1m_2way_k4_2026.sqlite`: exported event database.
- `wiki1m_2way_k4_profile_*.csv`: kernel, memory, CUDA API, and NVTX summaries.
- `wiki1m_2way_k4_profile_result.csv`: benchmark output for the captured run.

The capture command was:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --stats=true \
  --output wiki1m_2way_k4_2026 \
  CAGRA_MERGE_API_BENCH_K4_PROFILE \
  --dataset /raid/blandrum/local_datasets/wiki_all_1M/base.1M.fbin \
  --queries /raid/blandrum/local_datasets/wiki_all_1M/queries.fbin \
  --groundtruth /raid/blandrum/local_datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin \
  --output-csv wiki1m_2way_k4_profile_result.csv \
  --label Wiki-1M \
  --implementation k4-scaffold-nsys-2026 \
  --parts 2 \
  --profile-merge
```

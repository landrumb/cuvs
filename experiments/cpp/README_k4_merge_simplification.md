# CAGRA k=4 merge simplification ledger

This file tracks review-surface reductions for the k=4 scaffold merge and records the benchmark
evidence used to keep or reject each change. The comparison base is commit `cf06948a` unless a row
names another base.

## Measurement conventions

- **Physical LOC:** `wc -l` for the affected implementation files.
- **Top-level declarations:** simple source scan of namespace-scope functions, kernels, and structs.
  This is a review-surface proxy, not a formal C++ complexity score.
- **Decision points:** source occurrences of `if`, `for`, `while`, `switch`, and `catch`.
- **Preprocessor branches:** `#if`, `#ifdef`, `#ifndef`, `#elif`, and `#else`.
- **Build time:** public `cagra::merge()` `merge_api_e2e_ms`; oracle partition construction,
  dataset reads, and query transfer are excluded.
- **Recall:** Recall@12 with the complete query and ground-truth set.
- Performance comparisons use 8-way fan-in and two fresh runs per policy unless noted otherwise.

## Starting review surface

| file / state | physical LOC | top-level declarations | GPU kernels | preprocessor branches | decision points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cagra_merge_scaffold.cuh` at `cf06948a` | 1,358 | 32 | 13 | 14 | 126 |
| experiment-expanded working tree | 1,591 | 39 | 15 | 21 | 146 |
| `cagra_merge.cuh` at `cf06948a` | 296 | — | 0 | — | — |

After S01, the PR adds 1,358 lines for the scaffold header and 122 net lines to `cagra_merge.cuh`
relative to the pre-k=4 parent `e9b9e755`. Reducing that production review surface is the primary
objective.

## Final retained result

| production review metric | `cf06948a` | retained | delta |
| --- | ---: | ---: | ---: |
| scaffold header physical LOC | 1,358 | 928 | **-430 (-31.7%)** |
| source GPU-kernel definitions | 13 | 9 | **-4** |
| preprocessor branches | 14 | 0 | **-14** |
| decision points | 126 | 84 | **-42** |

Relative to the pre-k=4 parent, the retained production addition is 928 scaffold lines plus the
existing 122 net wrapper lines, or 1,050 lines total. Including the removed experiment-only policy
and harness code, the working review surface is 911 lines smaller than the expanded starting state.
All final float32, float16, int8, and uint8 release instantiations compile with warnings as errors.

## Change ledger

### S01 — Remove dominated pivot-policy variants from production — kept

The score-midpoint and balanced-retry policies were useful experiments but not suitable production
options. Their implementation added compile-time branches, helper functions, and two kernels to the
main scaffold header. The dedicated tree-generation executable duplicated another 229 lines.

Review-surface change:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| production scaffold header LOC | 1,591 | 1,358 | **-233** |
| histogram-generator C++ LOC | 229 | 0 | **-229** |
| experiment CMake LOC | 19 | 0 | **-19** |
| top-level declarations in scaffold header | 39 | 32 | **-7** |
| GPU kernels in scaffold header | 15 | 13 | **-2** |
| preprocessor branches in scaffold header | 21 | 14 | **-7** |
| decision points in scaffold header | 146 | 126 | **-20** |
| total implementation / harness LOC | — | — | **-481** |

Performance evidence (two-run means, leaf size 128):

| dataset | policy | build vs baseline | Recall@12 delta |
| --- | --- | ---: | ---: |
| Wiki-1M | score-midpoint | -11.44% | -0.000205 |
| Wiki-1M | balanced retry | +9.10% | -0.000600 |
| OpenAI-2M | score-midpoint | -3.19% | -0.004392 |
| OpenAI-2M | balanced retry | +11.63% | -0.001109 |
| YFCC-10M | score-midpoint | -6.67% | -0.002331 |
| YFCC-10M | balanced retry | +11.91% | -0.000408 |

Decision: retain the nearest-pivot baseline. Score-midpoint exchanges material recall for speed, and
balanced retry is dominated on both build time and recall. The raw CSVs, summary, transparent
histogram overlay, plotting script, and README conclusions remain as evidence; only the production
and generator machinery was removed.

### S02 — Remove dormant production CSV and profiling hooks — kept

Three default-off environment-variable hooks wrote leaf sizes, workspace details, and CUDA-event
timings from the production header. They were used only while developing the experiments, had no
callers elsewhere in the repository, and put file I/O plus profiling state into the normal merge
implementation.

Review-surface change:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 1,358 | 1,223 | **-135** |
| top-level declarations | 32 | 29 | **-3** |
| GPU kernels | 13 | 13 | 0 |
| preprocessor branches | 14 | 14 | 0 |
| decision points | 126 | 117 | **-9** |
| standard-library includes | 9 | 7 | **-2** |

Performance evidence (two-run means, current leaf-size 256 default):

| dataset | baseline build | without hooks | build delta | baseline Recall@12 | without hooks | recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 317.772 ms | 317.790 ms | +0.006% | 0.977209 | 0.976829 | -0.000380 |
| OpenAI-2M | 1,489.779 ms | 1,492.530 ms | +0.185% | 0.920258 | 0.920075 | -0.000183 |
| YFCC-10M | 1,584.513 ms | 1,584.212 ms | -0.019% | 0.955760 | 0.955516 | -0.000244 |

All four datatype instantiations compile with warnings as errors. Alternating baseline/candidate
ordering keeps all build-time changes within 0.2%; the small recall movement is within the
fresh-oracle partition variation already observed in retained runs. Decision: keep the deletion.
Raw results are in
[simplification_instrumentation_cleanup_8way.csv](merge_api_results/simplification_instrumentation_cleanup_8way.csv).
Cumulative implementation/harness reduction through S02 is 616 lines.

### S03 — Remove opt-in FP16 experiment branches — kept

The production configuration uses FP32 input, accumulation, and Gram output. The header nevertheless
carried an alternate FP16 gather kernel, an FP16 Gram top-k kernel, and nested macro branches for
two mixed-precision experiments. Those experiments are retained in the results documentation, but
their code no longer expands the production review surface.

Review-surface change from S02:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 1,223 | 1,042 | **-181** |
| top-level declarations | 29 | 27 | **-2** |
| GPU kernels | 13 | 11 | **-2** |
| preprocessor branches | 14 | 4 | **-10** |
| decision points | 117 | 101 | **-16** |
| CUDA-specific includes | 1 | 0 | **-1** |

All four datatype instantiations compile with warnings as errors. For the retained float32 and uint8
paths, both the embedded CUDA fatbinary and primary host text section are byte-identical to S02.
One full-query smoke per dataset provides an end-to-end check:

| dataset | S02 build mean | no-FP16 smoke | build delta | S02 Recall@12 mean | no-FP16 smoke | recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 317.790 ms | 317.640 ms | -0.047% | 0.976829 | 0.976975 | +0.000146 |
| OpenAI-2M | 1,492.530 ms | 1,486.449 ms | -0.407% | 0.920075 | 0.920663 | +0.000588 |
| YFCC-10M | 1,584.212 ms | 1,598.044 ms | +0.873% | 0.955516 | 0.955462 | -0.000054 |

With identical retained machine code, the observed runtime movement is measurement/fresh-oracle
variation rather than an implementation effect. Decision: keep the deletion. Raw smoke results are
in [simplification_remove_fp16_8way.csv](merge_api_results/simplification_remove_fp16_8way.csv).
Cumulative implementation/harness reduction through S03 is 797 lines; the production scaffold
header itself is 316 lines smaller than the `cf06948a` baseline.

### S04 — Remove the duplicate maximum-cluster alias — kept

`k_max_cluster` was exactly `k_cluster_size`; it had no independent bound or behavior. Replacing
its 32 uses with the canonical name removes one declaration and one concept from shared-memory
sizes, Gram strides, launch parameters, and workspace calculations.

Review-surface change from S03: 1 production line and 1 named constant removed; branch, kernel, and
decision-point counts are unchanged. Float32 and uint8 compile cleanly, and their CUDA fatbins plus
primary host text sections are byte-identical before and after the edit. Build time and recall
therefore have no implementation-level change. Decision: keep the deletion.

Cumulative implementation/harness reduction through S04 is 798 lines; the production scaffold
header is 317 lines smaller than the `cf06948a` baseline.

### S05 — Pass host-only policy through runtime build parameters — kept

Five values that do not shape device arrays, shared memory, or launch dimensions now live in a
`build_params` value passed to the scaffold builder: pivot seed, pivot-tree level cap, active and
inactive descriptor chunk sizes, and leaf-GEMM workspace bytes. Positive chunk sizes and level caps
are validated once at entry. The production caller keeps the same defaults.

Review-surface change from S04:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 1,041 | 1,045 | +4 |
| namespace compile-time policy constants | 5 | 0 | **-5** |
| runtime parameter fields | 0 | 5 | +5 |
| workspace preprocessor branches | 2 | 0 | **-2** |
| runtime parameter structs | 0 | 1 | +1 |
| GPU kernels / decision points | 11 / 101 | 11 / 101 | 0 |

The four datatype instantiations compile with warnings as errors. Alternating paired benchmarks show
that dereferencing the small host parameter object has no significant cost:

| dataset | compile-time build | runtime build | build delta | compile-time Recall@12 | runtime Recall@12 | recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 317.796 ms | 317.513 ms | -0.089% | 0.976925 | 0.977284 | +0.000359 |
| OpenAI-2M | 1,490.788 ms | 1,491.639 ms | +0.057% | 0.920446 | 0.920250 | -0.000197 |
| YFCC-10M | 1,583.266 ms | 1,588.272 ms | +0.316% | 0.955468 | 0.955625 | +0.000157 |

Decision: keep the runtime parameters. Raw paired runs are in
[simplification_runtime_params_8way.csv](merge_api_results/simplification_runtime_params_8way.csv).
The four-line net addition changes the cumulative implementation/harness reduction to 794 lines;
the production scaffold remains 313 lines smaller than the `cf06948a` baseline.

### S06 — Unify float and integer Gram top-k kernels — kept

The FP32 and int32 Gram consumers duplicated leaf loading, origin filtering, four-neighbor
selection, and graph writes. One `leaf_gram_knn_kernel<GramT>` now owns that shared logic. Two
compile-time type branches retain the distinct float distance/finiteness behavior and exact int32
distance formula; template argument deduction keeps both launches identical at the call site.

Review-surface change from S05:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 1,045 | 991 | **-54** |
| source GPU-kernel definitions | 11 | 10 | **-1** |
| top-level declarations | 28 | 27 | **-1** |
| decision points | 101 | 89 | **-12** |
| generated Gram top-k specializations | 2 | 2 | 0 |

Float32 and uint8 compile with warnings as errors. Alternating paired runs show no performance or
quality regression:

| dataset | separate build | unified build | build delta | separate Recall@12 | unified Recall@12 | recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 317.775 ms | 317.195 ms | -0.183% | 0.976888 | 0.977146 | +0.000259 |
| OpenAI-2M | 1,491.591 ms | 1,492.436 ms | +0.057% | 0.920228 | 0.920617 | +0.000389 |
| YFCC-10M | 1,588.750 ms | 1,584.150 ms | -0.290% | 0.955625 | 0.955513 | -0.000112 |

Decision: keep the unified template. Raw paired runs are in
[simplification_unified_topk_8way.csv](merge_api_results/simplification_unified_topk_8way.csv).
Cumulative implementation/harness reduction through S06 is 848 lines; the production scaffold
header is 367 lines smaller than the `cf06948a` baseline.

### R01 — Factor wrapper input inspection — rejected

A prototype inspected datasets once and passed dimension, row count, offsets, stride, and maximum
input degree to both rebuild and scaffold paths. It removed two validation loops and the separate
eligibility scan, but it broadened the PR into stable fallback code:

| item | before | prototype | delta |
| --- | ---: | ---: | ---: |
| wrapper physical LOC | 296 | 288 | -8 |
| wrapper decision points | 21 | 14 | -7 |
| new top-level declarations | 0 | 2 | +2 |
| lines touched versus pre-k=4 parent | 126 | 164 | **+38** |

The eight-line reduction is not worth 38 additional reviewed lines and a new metadata abstraction in
the rebuild fallback. The prototype was reverted before compile/performance testing; no production
behavior changed and the S06 benchmark state remains current.

### S07 — Remove the final experiment macro and float-max alias — kept

Leaf size still must be compile-time because it sizes shared arrays, Gram strides, and launch
dimensions, but production has one retained value: 256. Removing the experiment macro makes that
choice explicit. The one-use handwritten float maximum was replaced with
`std::numeric_limits<float>::max()`.

Review-surface change from S06:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 991 | 985 | **-6** |
| preprocessor branches | 2 | 0 | **-2** |
| leaf-size definitions | 2 conditional definitions | 1 fixed definition | **-1** |
| redundant numeric aliases | 1 | 0 | **-1** |

Float32 and uint8 compile cleanly. Their CUDA fatbins and primary host text sections are
byte-identical to S06, so build time and recall have no implementation-level change. Decision: keep
the cleanup. Cumulative implementation/harness reduction is 854 lines; the production scaffold is
373 lines smaller than the `cf06948a` baseline.

### S08 — Consolidate float and uint8 vectorized distance branches — kept

Both pivot-distance helpers had separate `float4` and `uchar4` branches with identical alignment
checks, four-lane loops, and accumulation. A compile-time `vector_type` alias now selects the lane
type; explicit conversion to float works for both specializations. Half and int8 scalar fallbacks
are unchanged.

Review-surface change from S07:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 985 | 938 | **-47** |
| decision points | 89 | 85 | **-4** |
| duplicated vectorized branches | 4 | 2 | **-2** |

Float32 and uint8 compile cleanly, and both their CUDA fatbins and primary host text sections are
byte-identical to S07. Build time and recall therefore have no implementation-level change.
Decision: keep the consolidation. Cumulative implementation/harness reduction is 901 lines; the
production scaffold is 420 lines smaller than the `cf06948a` baseline.

### R02 — Unify float and integer leaf-gather kernels — rejected

A constrained `gather_leaf_vectors_kernel<InputT, OutputT>` prototype shared indexing, padding,
and output writes across float, uint8, and int8. It reduced the scaffold from 938 to 920 lines,
source kernel definitions from 10 to 9, and decision points from 85 to 83.

Paired results:

| dataset | separate build | unified build | build delta | separate Recall@12 | unified Recall@12 | recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 317.948 ms | 316.856 ms | -0.344% | 0.977104 | 0.976925 | -0.000179 |
| OpenAI-2M | 1,491.543 ms | 1,493.691 ms | +0.144% | 0.920338 | 0.920244 | -0.000094 |
| YFCC-10M | 1,584.862 ms | 1,595.307 ms | **+0.659%** | 0.955633 | 0.955464 | -0.000169 |

The repeated YFCC penalty is not worth an 18-line reduction, particularly because the additional
input-dimension launch parameter exists only to accommodate the shared abstraction. The prototype
was reverted. Raw runs are in
[simplification_unified_gather_8way.csv](merge_api_results/simplification_unified_gather_8way.csv).

### S09 — Fuse input-graph copy and scaffold append — kept

Each partition-copy launch already owns a disjoint global row range. It now writes the four
scaffold neighbors for that row immediately after copying or padding the input graph, eliminating a
separate full-graph append kernel and launch while preserving column order.

Review-surface change from S08:

| item | before | after | delta |
| --- | ---: | ---: | ---: |
| scaffold header LOC | 938 | 928 | **-10** |
| source GPU-kernel definitions | 10 | 9 | **-1** |
| append-stage launches | partitions + 1 | partitions | **-1** |
| decision points | 85 | 84 | **-1** |

Float32 and uint8 compile cleanly. Alternating paired runs:

| dataset | separate build | fused build | build delta | separate Recall@12 | fused Recall@12 | recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-1M | 318.612 ms | 318.184 ms | -0.134% | 0.977442 | 0.976838 | -0.000604 |
| OpenAI-2M | 1,490.076 ms | 1,493.227 ms | +0.211% | 0.920357 | 0.920302 | -0.000055 |
| YFCC-10M | 1,590.292 ms | 1,588.181 ms | -0.133% | 0.955532 | 0.955647 | +0.000115 |

There is no consistent timing penalty, and recall remains in the established fresh-oracle band.
Decision: keep the fusion. Raw runs are in
[simplification_fused_append_8way.csv](merge_api_results/simplification_fused_append_8way.csv).
Cumulative implementation/harness reduction is 911 lines; the production scaffold is 430 lines
smaller than the `cf06948a` baseline.

## Constant-to-runtime audit

Prefer runtime values when they do not alter kernel resource shape, static storage, or unrolling.

| constant | disposition | rationale |
| --- | --- | --- |
| `k_degree` | keep compile time | sizes per-thread top-k arrays and unrolled loops |
| `k_cluster_size` | keep compile time, fixed at 256 (S07) | sizes shared arrays, Gram strides, and launch dimensions |
| `k_max_cluster` | removed (S04) | duplicated `k_cluster_size`; retained code sections are identical |
| `k_pivot_block_size` | keep compile time | sizes the shared reduction array and launch |
| `k_seed` | runtime `build_params::seed` (S05) | host pivot selection only |
| `k_max_pivot_tree_levels` | runtime `build_params::max_pivot_tree_levels` (S05) | host loop bound only |
| `k_leaf_gemm_workspace_bytes` | runtime `build_params::leaf_gemm_workspace_bytes` (S05) | host allocation / batching policy only |
| `k_pivot_assign_chunk` | runtime `build_params::pivot_assign_chunk` (S05) | host descriptor construction only |
| `k_leaf_assign_chunk` | runtime `build_params::leaf_assign_chunk` (S05) | host descriptor construction only |

## Candidate queue

| candidate | expected simplification | risk / required evidence | status |
| --- | --- | --- | --- |
| Remove production-only CSV/profiling hooks | eliminate file I/O and CUDA-event scaffolding | four-type compile + paired benchmarks | kept (S02) |
| Remove opt-in FP16 experiment branches | collapse float GEMM path to the retained FP32 production path | four-type compile + binary equivalence + smokes | kept (S03) |
| Remove `k_max_cluster` alias | one fewer concept across kernels and workspace math | float/uint8 compile + binary equivalence | kept (S04) |
| Pass host-only policy at runtime | remove five compile-time policies and workspace macro | four-type compile + paired benchmarks | kept (S05) |
| Unify float/int Gram top-k kernels | remove duplicated selection logic | float/uint8 compile + paired benchmarks | kept (S06) |
| Remove leaf-size macro and float-max alias | expose one production leaf value and remove final preprocessor branches | float/uint8 compile + binary equivalence | kept (S07) |
| Consolidate float/uint8 distance branches | remove duplicated aligned vector loops | float/uint8 compile + binary equivalence | kept (S08) |
| Fuse graph copy and scaffold append | remove one kernel and launch | float/uint8 compile + paired benchmarks | kept (S09) |
| Unify float/integer leaf gathers | remove duplicated gather indexing | YFCC build +0.659% for 18 lines | rejected (R02) |
| Factor shared wrapper input inspection | remove duplicate validation and eligibility scans | increased PR touched lines by 38 | rejected (R01) |

## Retained evidence

- Raw performance runs:
  [pivot_tree_variants_8way.csv](merge_api_results/pivot_tree_variants_8way.csv)
- Performance summary:
  [pivot_tree_variants_8way_summary.csv](merge_api_results/pivot_tree_variants_8way_summary.csv)
- Leaf-size summary:
  [leaf_size_pivot_variants_summary.csv](merge_api_results/leaf_size_pivot_variants_summary.csv)
- Transparent histogram overlay:
  [k4_leaf_size_pivot_variants_8way.png](merge_api_results/plots/k4_leaf_size_pivot_variants_8way.png)
- Instrumentation-cleanup paired runs:
  [simplification_instrumentation_cleanup_8way.csv](merge_api_results/simplification_instrumentation_cleanup_8way.csv)
- No-FP16 full-query smokes:
  [simplification_remove_fp16_8way.csv](merge_api_results/simplification_remove_fp16_8way.csv)
- Runtime-parameter paired runs:
  [simplification_runtime_params_8way.csv](merge_api_results/simplification_runtime_params_8way.csv)
- Unified top-k paired runs:
  [simplification_unified_topk_8way.csv](merge_api_results/simplification_unified_topk_8way.csv)
- Rejected unified-gather paired runs:
  [simplification_unified_gather_8way.csv](merge_api_results/simplification_unified_gather_8way.csv)
- Fused-append paired runs:
  [simplification_fused_append_8way.csv](merge_api_results/simplification_fused_append_8way.csv)

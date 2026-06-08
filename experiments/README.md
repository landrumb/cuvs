# cuVS experiments

Scratch space for standalone C++ binaries used to profile / benchmark cuVS
internals. Built as a self-contained CMake project (like `examples/cpp`) that
links against `libcuvs`.

## Building

By default the build reuses an already-built `libcuvs` from `../cpp/build`:

```bash
cd experiments
./build.sh                       # NATIVE arch, reuses ../cpp/build
./build.sh --gpu-arch=90-real    # specific arch
./build.sh clean                 # remove cpp/build
```

If you have not built libcuvs yet, build it first:

```bash
cd ..; ./build.sh libcuvs
```

To build cuvs from source via CPM instead of reusing `../cpp/build`, set
`CPM_cuvs_SOURCE=/path/to/cuvs` (or edit `CUVS_REPO_REL` in `build.sh`).

Binaries land in `experiments/cpp/build/`.

## `CAGRA_SPLIT_SEARCH`

Builds two CAGRA indexes on contiguous halves of a dataset, searches both halves independently, merges the two result sets by distance with the right-half row offset applied, and reports recall against exact GPU flat search. Outputs are written under `--output-dir` with merge-compatible serialized indexes in `left/cagra.index` and `right/cagra.index`, plus `manifest.json` for reproduction metadata. Defaults use `--build-algo ivf_pq --graph-degree 16 --intermediate-graph-degree 32`; pass `--build-algo auto` or `--build-algo nn_descent` to test those paths explicitly.

```bash
./cpp/build/CAGRA_SPLIT_SEARCH --output-dir split_out \
  /raid/blandrum/local_datasets/sift-128-euclidean/base.fbin \
  /raid/blandrum/local_datasets/sift-128-euclidean/query.fbin
```

## `CAGRA_BUILD_PROFILE`

Loads a dataset with the integrated binary loader (`read_bin_dataset` in
`cpp/src/common.cuh`) and times CAGRA graph construction, with an optional
untimed warmup so a profiler can capture steady-state behavior.

```
./cpp/build/CAGRA_BUILD_PROFILE <data file> <datatype> \
    [graph_degree] [intermediate_graph_degree] [build_algo] \
    [warmup_iters] [timed_iters] [max_N]
```

- **data file** — binary: `[uint32 N][uint32 dim][N*dim values]` (the `.bin`
  format consumed by the examples, e.g. `sift_base.bin`).
- **datatype** — `float`, `int8`, or `uint8`.
- **build_algo** — `auto` (heuristic, default), `ivf_pq`, or `nn_descent`.

Example — profile with Nsight Systems:

```bash
nsys profile -o cagra_build \
  ./cpp/build/CAGRA_BUILD_PROFILE /data/sift_base.bin float 64 128 auto 1 5
```

CAGRA's build path is annotated with NVTX ranges. To see them, build `libcuvs`
with `-DCUVS_NVTX=ON` (the experiment project already enables NVTX for its own
code).

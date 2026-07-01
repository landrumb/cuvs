/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuvs::neighbors::cagra::detail::merge_scaffold {

inline constexpr int k_degree                = 4;
inline constexpr int k_cluster_size          = 64;
inline constexpr int k_max_cluster           = 64;
inline constexpr int k_pivot_assign_chunk    = 256;
inline constexpr int k_leaf_assign_chunk     = 1024;
inline constexpr int k_max_pivot_tree_levels = 64;
inline constexpr uint64_t k_seed             = 1234;
inline constexpr float k_inf                 = 3.4028234663852886e38f;

struct host_range {
  int64_t start = 0;
  int64_t end   = 0;
};

struct pivot_chunk {
  int64_t start     = 0;
  int64_t end       = 0;
  uint32_t key_base = 0;
  uint32_t pivot_a  = 0;
  uint32_t pivot_b  = 0;
  uint8_t active    = 0;
};

__host__ __device__ inline uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

template <typename T>
std::vector<T> copy_to_host(raft::resources const& res, rmm::device_uvector<T> const& input)
{
  std::vector<T> output(input.size());
  raft::copy(output.data(), input.data(), input.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return output;
}

inline std::vector<host_range> scan_key_ranges(std::vector<uint32_t> const& keys)
{
  std::vector<host_range> ranges;
  if (keys.empty()) { return ranges; }
  int64_t start = 0;
  uint32_t key  = keys.front();
  for (int64_t i = 1; i < static_cast<int64_t>(keys.size()); ++i) {
    if (keys[static_cast<size_t>(i)] != key) {
      ranges.push_back({start, i});
      start = i;
      key   = keys[static_cast<size_t>(i)];
    }
  }
  ranges.push_back({start, static_cast<int64_t>(keys.size())});
  return ranges;
}

inline std::vector<host_range> split_large_ranges(std::vector<host_range> const& ranges)
{
  std::vector<host_range> output;
  output.reserve(ranges.size());
  for (auto const& range : ranges) {
    for (int64_t start = range.start; start < range.end; start += k_cluster_size) {
      output.push_back({start, std::min<int64_t>(range.end, start + k_cluster_size)});
    }
  }
  return output;
}

inline int make_pivot_chunks(std::vector<host_range> const& ranges,
                             std::vector<uint32_t> const& ids,
                             int level,
                             std::vector<pivot_chunk>& chunks)
{
  int active_count = 0;
  for (auto const& range : ranges) {
    if (range.end - range.start > k_cluster_size) { ++active_count; }
  }

  chunks.clear();
  int active_index = 0;
  int leaf_index   = 0;
  for (size_t r = 0; r < ranges.size(); ++r) {
    auto const& range = ranges[r];
    int64_t n         = range.end - range.start;
    bool active       = n > k_cluster_size;
    uint32_t key_base = 0;
    uint32_t pivot_a  = 0;
    uint32_t pivot_b  = 0;
    int chunk_size    = active ? k_pivot_assign_chunk : k_leaf_assign_chunk;

    if (active) {
      key_base      = static_cast<uint32_t>(2 * active_index++);
      uint64_t h    = splitmix64(k_seed ^ (uint64_t(level) * 0x94d049bb133111ebull) ^
                              (uint64_t(r) * 0x9e3779b97f4a7c15ull));
      int64_t off_a = static_cast<int64_t>(h % static_cast<uint64_t>(n));
      int64_t off_b = static_cast<int64_t>(splitmix64(h) % static_cast<uint64_t>(n));
      if (off_a == off_b) { off_b = (off_b + 1) % n; }
      pivot_a = ids[static_cast<size_t>(range.start + off_a)];
      pivot_b = ids[static_cast<size_t>(range.start + off_b)];
    } else {
      key_base = static_cast<uint32_t>(2 * active_count + leaf_index++);
    }

    for (int64_t start = range.start; start < range.end; start += chunk_size) {
      chunks.push_back({start,
                        std::min<int64_t>(range.end, start + chunk_size),
                        key_base,
                        pivot_a,
                        pivot_b,
                        static_cast<uint8_t>(active ? 1 : 0)});
    }
  }
  return active_count;
}

template <typename T>
__device__ float l2_distance(T const* dataset, int64_t dim, uint32_t a, uint32_t b)
{
  float acc   = 0.0f;
  T const* pa = dataset + static_cast<int64_t>(a) * dim;
  T const* pb = dataset + static_cast<int64_t>(b) * dim;
  for (int64_t d = 0; d < dim; ++d) {
    float diff = static_cast<float>(pa[d]) - static_cast<float>(pb[d]);
    acc += diff * diff;
  }
  return acc;
}

template <typename T>
__global__ void pivot_assign_keys_kernel(T const* dataset,
                                         int64_t dim,
                                         uint32_t const* ids,
                                         pivot_chunk const* chunks,
                                         int64_t chunk_count,
                                         uint32_t* keys)
{
  int64_t chunk_idx = blockIdx.x;
  if (chunk_idx >= chunk_count) { return; }
  pivot_chunk chunk = chunks[chunk_idx];
  for (int64_t pos = chunk.start + threadIdx.x; pos < chunk.end; pos += blockDim.x) {
    if (!chunk.active) {
      keys[pos] = chunk.key_base;
      continue;
    }
    uint32_t id = ids[pos];
    float da    = l2_distance(dataset, dim, id, chunk.pivot_a);
    float db    = l2_distance(dataset, dim, id, chunk.pivot_b);
    keys[pos]   = chunk.key_base + (db < da ? 1u : 0u);
  }
}

static __global__ void initialize_partition_kernel(uint32_t* origins,
                                                   uint32_t* fallback,
                                                   int64_t start,
                                                   int64_t rows,
                                                   uint32_t origin,
                                                   uint32_t other_start)
{
  int64_t local_row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_row >= rows) { return; }
  int64_t row   = start + local_row;
  origins[row]  = origin;
  fallback[row] = other_start;
}

template <typename T>
__global__ void leaf_cross_knn_kernel(T const* dataset,
                                      int64_t dim,
                                      uint32_t const* sorted_ids,
                                      uint32_t const* origins,
                                      uint32_t const* leaf_starts,
                                      uint32_t const* leaf_ends,
                                      int64_t leaf_count,
                                      uint32_t* graph,
                                      uint8_t* degrees)
{
  int64_t leaf = blockIdx.x;
  if (leaf >= leaf_count) { return; }
  uint32_t start = leaf_starts[leaf];
  uint32_t end   = leaf_ends[leaf];
  int leaf_n     = static_cast<int>(end - start);
  if (leaf_n <= 1 || leaf_n > k_max_cluster) { return; }

  __shared__ uint32_t ids[k_max_cluster];
  for (int i = threadIdx.x; i < leaf_n; i += blockDim.x) {
    ids[i] = sorted_ids[start + i];
  }
  __syncthreads();

  if (threadIdx.x < leaf_n) {
    int u = threadIdx.x;
    float top_d[k_degree];
    uint8_t top_v[k_degree];
#pragma unroll
    for (int t = 0; t < k_degree; ++t) {
      top_d[t] = k_inf;
      top_v[t] = std::numeric_limits<uint8_t>::max();
    }

    for (int v = 0; v < leaf_n; ++v) {
      if (u == v || origins[ids[u]] == origins[ids[v]]) { continue; }
      float distance = l2_distance(dataset, dim, ids[u], ids[v]);
      int worst      = 0;
#pragma unroll
      for (int t = 1; t < k_degree; ++t) {
        if (top_d[t] > top_d[worst]) { worst = t; }
      }
      if (distance < top_d[worst]) {
        top_d[worst] = distance;
        top_v[worst] = static_cast<uint8_t>(v);
      }
    }

    int selected = 0;
#pragma unroll
    for (int t = 0; t < k_degree; ++t) {
      if (isfinite(top_d[t]) && top_v[t] != std::numeric_limits<uint8_t>::max()) {
        graph[static_cast<int64_t>(ids[u]) * k_degree + selected] = ids[top_v[t]];
        ++selected;
      }
    }
    degrees[ids[u]] = static_cast<uint8_t>(selected);
  }
}

static __global__ void pad_scaffold_kernel(uint32_t* graph,
                                           uint8_t* degrees,
                                           uint32_t const* fallback,
                                           int64_t rows)
{
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) { return; }
  int degree   = degrees[row];
  int64_t base = row * k_degree;
  if (degree == 0) {
    graph[base] = fallback[row];
    degree      = 1;
  }
  for (int j = degree; j < k_degree; ++j) {
    graph[base + j] = graph[base + (j % degree)];
  }
}

template <typename T>
auto build(raft::resources const& res,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
           std::vector<int64_t> const& offsets) -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream  = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);
  RAFT_EXPECTS(offsets.size() >= 3, "k=4 scaffold merge requires at least two input indices");
  RAFT_EXPECTS(rows <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
               "k=4 scaffold merge requires the merged row count to fit in uint32_t");

  auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, rows, k_degree);
  rmm::device_uvector<uint8_t> degrees(rows, stream);
  rmm::device_uvector<uint32_t> origins(rows, stream);
  rmm::device_uvector<uint32_t> fallback(rows, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(degrees.data(), 0, degrees.size() * sizeof(uint8_t), stream));

  for (size_t part = 0; part + 1 < offsets.size(); ++part) {
    int64_t part_rows    = offsets[part + 1] - offsets[part];
    uint32_t other_start = static_cast<uint32_t>(offsets[(part + 1) % (offsets.size() - 1)]);
    int blocks           = static_cast<int>((part_rows + 255) / 256);
    initialize_partition_kernel<<<blocks, 256, 0, stream>>>(origins.data(),
                                                            fallback.data(),
                                                            offsets[part],
                                                            part_rows,
                                                            static_cast<uint32_t>(part),
                                                            other_start);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  rmm::device_uvector<uint32_t> keys(rows, stream);
  rmm::device_uvector<uint32_t> ids(rows, stream);
  thrust::sequence(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(ids.data()),
                   thrust::device_pointer_cast(ids.data() + ids.size()),
                   uint32_t{0});
  raft::resource::sync_stream(res);

  std::vector<uint32_t> ids_host(static_cast<size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) {
    ids_host[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
  }
  std::vector<uint32_t> keys_host;
  std::vector<host_range> ranges{{0, rows}};
  std::vector<pivot_chunk> chunks;

  for (int level = 0; level < k_max_pivot_tree_levels; ++level) {
    int active_count = make_pivot_chunks(ranges, ids_host, level, chunks);
    if (active_count == 0) { break; }

    rmm::device_uvector<pivot_chunk> device_chunks(chunks.size(), stream);
    raft::copy(device_chunks.data(), chunks.data(), chunks.size(), stream);
    pivot_assign_keys_kernel<<<static_cast<int>(chunks.size()), 128, 0, stream>>>(
      dataset.data_handle(),
      dataset.extent(1),
      ids.data(),
      device_chunks.data(),
      static_cast<int64_t>(chunks.size()),
      keys.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        thrust::device_pointer_cast(keys.data()),
                        thrust::device_pointer_cast(keys.data() + keys.size()),
                        thrust::device_pointer_cast(ids.data()));
    keys_host = copy_to_host(res, keys);
    ids_host  = copy_to_host(res, ids);
    ranges    = scan_key_ranges(keys_host);
  }

  auto leaves = split_large_ranges(ranges);
  std::vector<uint32_t> starts_host(leaves.size());
  std::vector<uint32_t> ends_host(leaves.size());
  for (size_t i = 0; i < leaves.size(); ++i) {
    starts_host[i] = static_cast<uint32_t>(leaves[i].start);
    ends_host[i]   = static_cast<uint32_t>(leaves[i].end);
  }
  rmm::device_uvector<uint32_t> starts(starts_host.size(), stream);
  rmm::device_uvector<uint32_t> ends(ends_host.size(), stream);
  raft::copy(starts.data(), starts_host.data(), starts.size(), stream);
  raft::copy(ends.data(), ends_host.data(), ends.size(), stream);
  leaf_cross_knn_kernel<<<static_cast<int>(leaves.size()), 64, 0, stream>>>(dataset.data_handle(),
                                                                            dataset.extent(1),
                                                                            ids.data(),
                                                                            origins.data(),
                                                                            starts.data(),
                                                                            ends.data(),
                                                                            leaves.size(),
                                                                            graph.data_handle(),
                                                                            degrees.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  int blocks = static_cast<int>((rows + 255) / 256);
  pad_scaffold_kernel<<<blocks, 256, 0, stream>>>(
    graph.data_handle(), degrees.data(), fallback.data(), rows);
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  return graph;
}

static __global__ void copy_partition_graph_kernel(uint32_t const* source,
                                                   int64_t source_rows,
                                                   int64_t source_degree,
                                                   uint32_t* destination,
                                                   int64_t destination_degree,
                                                   int64_t base_degree,
                                                   uint32_t offset)
{
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= source_rows) { return; }
  int64_t source_base      = row * source_degree;
  int64_t destination_base = (row + offset) * destination_degree;
  for (int64_t j = 0; j < base_degree; ++j) {
    destination[destination_base + j] = source[source_base + (j % source_degree)] + offset;
  }
}

static __global__ void append_scaffold_kernel(uint32_t const* scaffold,
                                              uint32_t* graph,
                                              int64_t rows,
                                              int64_t graph_degree,
                                              int64_t base_degree)
{
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) { return; }
#pragma unroll
  for (int j = 0; j < k_degree; ++j) {
    graph[row * graph_degree + base_degree + j] = scaffold[row * k_degree + j];
  }
}

template <typename T, typename IdxT>
auto append_to_input_graphs(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
  std::vector<int64_t> const& offsets,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> scaffold)
  -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream         = raft::resource::get_cuda_stream(res);
  int64_t base_degree = 0;
  for (auto const* index : indices) {
    base_degree = std::max<int64_t>(base_degree, index->graph_degree());
  }
  int64_t graph_degree = base_degree + k_degree;
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, scaffold.extent(0), graph_degree);

  for (size_t part = 0; part < indices.size(); ++part) {
    auto source = indices[part]->graph();
    RAFT_EXPECTS(source.extent(1) > 0, "Input CAGRA graphs must have nonzero degree");
    int blocks = static_cast<int>((source.extent(0) + 255) / 256);
    copy_partition_graph_kernel<<<blocks, 256, 0, stream>>>(source.data_handle(),
                                                            source.extent(0),
                                                            source.extent(1),
                                                            graph.data_handle(),
                                                            graph_degree,
                                                            base_degree,
                                                            static_cast<uint32_t>(offsets[part]));
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  int blocks = static_cast<int>((scaffold.extent(0) + 255) / 256);
  append_scaffold_kernel<<<blocks, 256, 0, stream>>>(
    scaffold.data_handle(), graph.data_handle(), scaffold.extent(0), graph_degree, base_degree);
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  return graph;
}

}  // namespace cuvs::neighbors::cagra::detail::merge_scaffold

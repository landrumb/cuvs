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
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra::detail::merge_scaffold {

inline constexpr int k_degree       = 4;
inline constexpr int k_cluster_size = 256;
static_assert(k_cluster_size > 0 && k_cluster_size <= 1024);
inline constexpr int k_pivot_block_size = 128;

struct build_params {
  int pivot_assign_chunk           = 256;
  int leaf_assign_chunk            = 1024;
  int max_pivot_tree_levels        = 64;
  uint64_t seed                    = 1234;
  size_t leaf_gemm_workspace_bytes = size_t{2} * 1024 * 1024 * 1024;
};

struct host_range {
  int64_t start = 0;
  int64_t end   = 0;
};

struct pivot_chunk {
  uint32_t start       = 0;
  uint32_t end         = 0;
  uint32_t pivot_a     = 0;
  uint32_t pivot_b     = 0;
  uint32_t range_index = 0;
  uint8_t active       = 0;
};

struct scatter_offset {
  uint32_t left  = 0;
  uint32_t right = 0;
};

__host__ __device__ inline uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
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
                             int level,
                             build_params const& params,
                             std::vector<pivot_chunk>& chunks)
{
  int active_count = 0;
  for (auto const& range : ranges) {
    if (range.end - range.start > k_cluster_size) { ++active_count; }
  }

  chunks.clear();
  for (size_t r = 0; r < ranges.size(); ++r) {
    auto const& range = ranges[r];
    int64_t n         = range.end - range.start;
    bool active       = n > k_cluster_size;
    uint32_t pivot_a  = 0;
    uint32_t pivot_b  = 0;
    int chunk_size    = active ? params.pivot_assign_chunk : params.leaf_assign_chunk;

    if (active) {
      uint64_t h    = splitmix64(params.seed ^ (uint64_t(level) * 0x94d049bb133111ebull) ^
                              (uint64_t(r) * 0x9e3779b97f4a7c15ull));
      int64_t off_a = static_cast<int64_t>(h % static_cast<uint64_t>(n));
      int64_t off_b = static_cast<int64_t>(splitmix64(h) % static_cast<uint64_t>(n));
      if (off_a == off_b) { off_b = (off_b + 1) % n; }
      pivot_a = static_cast<uint32_t>(range.start + off_a);
      pivot_b = static_cast<uint32_t>(range.start + off_b);
    }

    for (int64_t chunk_start = range.start; chunk_start < range.end; chunk_start += chunk_size) {
      chunks.push_back(
        {static_cast<uint32_t>(chunk_start),
         static_cast<uint32_t>(std::min<int64_t>(range.end, chunk_start + chunk_size)),
         pivot_a,
         pivot_b,
         static_cast<uint32_t>(r),
         static_cast<uint8_t>(active ? 1 : 0)});
    }
  }
  return active_count;
}

inline auto prepare_scatter(std::vector<host_range> const& ranges,
                            std::vector<uint32_t> const& chunk_left_counts,
                            std::vector<pivot_chunk> const& chunks,
                            std::vector<scatter_offset>& scatter_offsets) -> std::vector<host_range>
{
  RAFT_EXPECTS(chunk_left_counts.size() == chunks.size(), "Pivot chunk count mismatch");

  // Match the old stable sort exactly: split active ranges first, in range order, then carry
  // completed leaves after them. Prefixes within each chunk preserve the input ID order.
  scatter_offsets.resize(chunks.size());
  std::vector<uint32_t> range_left_counts(ranges.size(), 0);
  int64_t active_rows = 0;
  for (size_t r = 0; r < ranges.size(); ++r) {
    if (ranges[r].end - ranges[r].start > k_cluster_size) {
      active_rows += ranges[r].end - ranges[r].start;
    }
  }
  for (size_t i = 0; i < chunks.size(); ++i) {
    if (chunks[i].active) { range_left_counts[chunks[i].range_index] += chunk_left_counts[i]; }
  }

  std::vector<int64_t> left_bases(ranges.size(), 0);
  std::vector<int64_t> right_bases(ranges.size(), 0);
  std::vector<int64_t> leaf_bases(ranges.size(), 0);
  std::vector<host_range> active_ranges;
  std::vector<host_range> leaf_ranges;
  active_ranges.reserve(ranges.size() * 2);
  leaf_ranges.reserve(ranges.size());

  int64_t active_cursor = 0;
  int64_t leaf_cursor   = active_rows;
  for (size_t r = 0; r < ranges.size(); ++r) {
    auto const& range = ranges[r];
    int64_t n         = range.end - range.start;
    if (n > k_cluster_size) {
      int64_t left   = range_left_counts[r];
      int64_t right  = n - left;
      left_bases[r]  = active_cursor;
      right_bases[r] = active_cursor + left;
      if (left > 0) { active_ranges.push_back({active_cursor, active_cursor + left}); }
      if (right > 0) {
        active_ranges.push_back({active_cursor + left, active_cursor + left + right});
      }
      active_cursor += n;
    } else {
      leaf_bases[r] = leaf_cursor;
      leaf_ranges.push_back({leaf_cursor, leaf_cursor + n});
      leaf_cursor += n;
    }
  }

  std::vector<uint32_t> left_prefix(ranges.size(), 0);
  std::vector<uint32_t> right_prefix(ranges.size(), 0);
  for (size_t i = 0; i < chunks.size(); ++i) {
    auto const& chunk = chunks[i];
    auto& output      = scatter_offsets[i];
    size_t r          = chunk.range_index;
    if (chunk.active) {
      uint32_t left  = chunk_left_counts[i];
      uint32_t right = static_cast<uint32_t>(chunk.end - chunk.start) - left;
      output.left    = static_cast<uint32_t>(left_bases[r] + left_prefix[r]);
      output.right   = static_cast<uint32_t>(right_bases[r] + right_prefix[r]);
      left_prefix[r] += left;
      right_prefix[r] += right;
    } else {
      output.left  = static_cast<uint32_t>(leaf_bases[r] + chunk.start - ranges[r].start);
      output.right = output.left;
    }
  }

  active_ranges.insert(active_ranges.end(), leaf_ranges.begin(), leaf_ranges.end());
  return active_ranges;
}

template <typename T>
__device__ float l2_distance(T const* dataset, int64_t dim, uint32_t a, uint32_t b)
{
  float acc   = 0.0f;
  T const* pa = dataset + static_cast<int64_t>(a) * dim;
  T const* pb = dataset + static_cast<int64_t>(b) * dim;
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t>) {
    using vector_type = std::conditional_t<std::is_same_v<T, float>, float4, uchar4>;
    if (dim % 4 == 0 && reinterpret_cast<uintptr_t>(pa) % alignof(vector_type) == 0 &&
        reinterpret_cast<uintptr_t>(pb) % alignof(vector_type) == 0) {
      auto pa4 = reinterpret_cast<vector_type const*>(pa);
      auto pb4 = reinterpret_cast<vector_type const*>(pb);
      for (int64_t d = 0; d < dim / 4; ++d) {
        vector_type va = pa4[d];
        vector_type vb = pb4[d];
        float diff     = static_cast<float>(va.x) - static_cast<float>(vb.x);
        acc += diff * diff;
        diff = static_cast<float>(va.y) - static_cast<float>(vb.y);
        acc += diff * diff;
        diff = static_cast<float>(va.z) - static_cast<float>(vb.z);
        acc += diff * diff;
        diff = static_cast<float>(va.w) - static_cast<float>(vb.w);
        acc += diff * diff;
      }
      return acc;
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    float diff = static_cast<float>(pa[d]) - static_cast<float>(pb[d]);
    acc += diff * diff;
  }
  return acc;
}

template <typename T>
__device__ void l2_distance_pair(T const* dataset,
                                 int64_t dim,
                                 uint32_t point,
                                 uint32_t pivot_a,
                                 uint32_t pivot_b,
                                 float& distance_a,
                                 float& distance_b)
{
  distance_a        = 0.0f;
  distance_b        = 0.0f;
  T const* point_p  = dataset + static_cast<int64_t>(point) * dim;
  T const* pivot_ap = dataset + static_cast<int64_t>(pivot_a) * dim;
  T const* pivot_bp = dataset + static_cast<int64_t>(pivot_b) * dim;
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t>) {
    using vector_type = std::conditional_t<std::is_same_v<T, float>, float4, uchar4>;
    if (dim % 4 == 0 && reinterpret_cast<uintptr_t>(point_p) % alignof(vector_type) == 0 &&
        reinterpret_cast<uintptr_t>(pivot_ap) % alignof(vector_type) == 0 &&
        reinterpret_cast<uintptr_t>(pivot_bp) % alignof(vector_type) == 0) {
      auto point4   = reinterpret_cast<vector_type const*>(point_p);
      auto pivot_a4 = reinterpret_cast<vector_type const*>(pivot_ap);
      auto pivot_b4 = reinterpret_cast<vector_type const*>(pivot_bp);
      for (int64_t d = 0; d < dim / 4; ++d) {
        vector_type value = point4[d];
        vector_type va    = pivot_a4[d];
        vector_type vb    = pivot_b4[d];
        float diff_a      = static_cast<float>(value.x) - static_cast<float>(va.x);
        float diff_b      = static_cast<float>(value.x) - static_cast<float>(vb.x);
        distance_a += diff_a * diff_a;
        distance_b += diff_b * diff_b;
        diff_a = static_cast<float>(value.y) - static_cast<float>(va.y);
        diff_b = static_cast<float>(value.y) - static_cast<float>(vb.y);
        distance_a += diff_a * diff_a;
        distance_b += diff_b * diff_b;
        diff_a = static_cast<float>(value.z) - static_cast<float>(va.z);
        diff_b = static_cast<float>(value.z) - static_cast<float>(vb.z);
        distance_a += diff_a * diff_a;
        distance_b += diff_b * diff_b;
        diff_a = static_cast<float>(value.w) - static_cast<float>(va.w);
        diff_b = static_cast<float>(value.w) - static_cast<float>(vb.w);
        distance_a += diff_a * diff_a;
        distance_b += diff_b * diff_b;
      }
      return;
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    float value  = static_cast<float>(point_p[d]);
    float diff_a = value - static_cast<float>(pivot_ap[d]);
    float diff_b = value - static_cast<float>(pivot_bp[d]);
    distance_a += diff_a * diff_a;
    distance_b += diff_b * diff_b;
  }
}

template <typename T>
__global__ void pivot_assign_sides_kernel(T const* dataset,
                                          int64_t dim,
                                          uint32_t const* ids,
                                          pivot_chunk const* chunks,
                                          int64_t chunk_count,
                                          uint8_t* sides,
                                          uint32_t* chunk_left_counts)
{
  int64_t chunk_idx = blockIdx.x;
  if (chunk_idx >= chunk_count) { return; }
  pivot_chunk chunk   = chunks[chunk_idx];
  uint32_t local_left = 0;
  if (chunk.active) {
    uint32_t pivot_a = ids[chunk.pivot_a];
    uint32_t pivot_b = ids[chunk.pivot_b];
    for (int64_t pos = chunk.start + threadIdx.x; pos < chunk.end; pos += blockDim.x) {
      uint32_t id = ids[pos];
      float da;
      float db;
      l2_distance_pair(dataset, dim, id, pivot_a, pivot_b, da, db);
      uint8_t side = static_cast<uint8_t>(db < da ? 1 : 0);
      sides[pos]   = side;
      local_left += side == 0;
    }
  }

  __shared__ uint32_t reduction[k_pivot_block_size];
  reduction[threadIdx.x] = local_left;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) { reduction[threadIdx.x] += reduction[threadIdx.x + offset]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { chunk_left_counts[chunk_idx] = reduction[0]; }
}

static __global__ void stable_scatter_kernel(uint32_t const* input_ids,
                                             uint8_t const* sides,
                                             pivot_chunk const* chunks,
                                             scatter_offset const* scatter_offsets,
                                             int64_t chunk_count,
                                             uint32_t* output_ids)
{
  int64_t chunk_idx = blockIdx.x;
  if (chunk_idx >= chunk_count) { return; }
  pivot_chunk chunk     = chunks[chunk_idx];
  scatter_offset output = scatter_offsets[chunk_idx];
  if (!chunk.active) {
    for (int64_t pos = chunk.start + threadIdx.x; pos < chunk.end; pos += blockDim.x) {
      output_ids[output.left + pos - chunk.start] = input_ids[pos];
    }
    return;
  }

  int64_t pos    = chunk.start + threadIdx.x;
  bool valid     = pos < chunk.end;
  bool goes_left = valid && sides[pos] == 0;
  __shared__ uint32_t left_prefix[256];
  left_prefix[threadIdx.x] = goes_left ? 1u : 0u;
  __syncthreads();
  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    uint32_t add = threadIdx.x >= offset ? left_prefix[threadIdx.x - offset] : 0;
    __syncthreads();
    if (threadIdx.x >= offset) { left_prefix[threadIdx.x] += add; }
    __syncthreads();
  }

  if (valid) {
    uint32_t left_before    = left_prefix[threadIdx.x] - (goes_left ? 1u : 0u);
    uint32_t right_before   = threadIdx.x - left_before;
    uint32_t destination    = goes_left ? output.left + left_before : output.right + right_before;
    output_ids[destination] = input_ids[pos];
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
  if (leaf_n <= 1 || leaf_n > k_cluster_size) { return; }

  __shared__ uint32_t ids[k_cluster_size];
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
      top_d[t] = std::numeric_limits<float>::max();
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

static __global__ void gather_float_leaf_vectors_kernel(float const* dataset,
                                                        int64_t dim,
                                                        uint32_t const* sorted_ids,
                                                        uint32_t const* leaf_starts,
                                                        uint32_t const* leaf_ends,
                                                        int64_t leaf_offset,
                                                        int64_t leaf_count,
                                                        float* leaf_vectors)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = leaf_count * k_cluster_size * dim;
  for (; linear < total; linear += stride) {
    int64_t d          = linear % dim;
    int64_t local_row  = (linear / dim) % k_cluster_size;
    int64_t local_leaf = linear / (dim * k_cluster_size);
    int64_t leaf       = leaf_offset + local_leaf;
    int64_t leaf_n     = static_cast<int64_t>(leaf_ends[leaf] - leaf_starts[leaf]);
    float value        = 0.0f;
    if (local_row < leaf_n) {
      uint32_t point = sorted_ids[leaf_starts[leaf] + local_row];
      value          = dataset[static_cast<int64_t>(point) * dim + d];
    }
    leaf_vectors[linear] = value;
  }
}

template <typename T>
__global__ void gather_integer_leaf_vectors_kernel(T const* dataset,
                                                   int64_t input_dim,
                                                   int64_t output_dim,
                                                   uint32_t const* sorted_ids,
                                                   uint32_t const* leaf_starts,
                                                   uint32_t const* leaf_ends,
                                                   int64_t leaf_offset,
                                                   int64_t leaf_count,
                                                   int8_t* leaf_vectors)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = leaf_count * k_cluster_size * output_dim;
  for (; linear < total; linear += stride) {
    int64_t d          = linear % output_dim;
    int64_t local_row  = (linear / output_dim) % k_cluster_size;
    int64_t local_leaf = linear / (output_dim * k_cluster_size);
    int64_t leaf       = leaf_offset + local_leaf;
    int8_t value       = 0;
    int64_t leaf_n     = static_cast<int64_t>(leaf_ends[leaf] - leaf_starts[leaf]);
    if (local_row < leaf_n && d < input_dim) {
      uint32_t point = sorted_ids[leaf_starts[leaf] + local_row];
      auto input     = dataset[static_cast<int64_t>(point) * input_dim + d];
      if constexpr (std::is_same_v<T, uint8_t>) {
        // Subtracting the same constant from every coordinate preserves pairwise L2 distance.
        value = static_cast<int8_t>(static_cast<int>(input) - 128);
      } else {
        value = input;
      }
    }
    leaf_vectors[linear] = value;
  }
}

template <typename GramT>
static __global__ void leaf_gram_knn_kernel(GramT const* gram,
                                            uint32_t const* sorted_ids,
                                            uint32_t const* origins,
                                            uint32_t const* leaf_starts,
                                            uint32_t const* leaf_ends,
                                            int64_t leaf_offset,
                                            int64_t leaf_count,
                                            uint32_t* graph,
                                            uint8_t* degrees)
{
  static_assert(std::is_same_v<GramT, float> || std::is_same_v<GramT, int32_t>);
  int64_t local_leaf = blockIdx.x;
  if (local_leaf >= leaf_count) { return; }
  int64_t leaf   = leaf_offset + local_leaf;
  uint32_t start = leaf_starts[leaf];
  int leaf_n     = static_cast<int>(leaf_ends[leaf] - start);
  if (leaf_n <= 1 || leaf_n > k_cluster_size) { return; }

  __shared__ uint32_t ids[k_cluster_size];
  __shared__ uint32_t leaf_origins[k_cluster_size];
  for (int i = threadIdx.x; i < leaf_n; i += blockDim.x) {
    ids[i]          = sorted_ids[start + i];
    leaf_origins[i] = origins[ids[i]];
  }
  __syncthreads();

  int u = threadIdx.x;
  if (u >= leaf_n) { return; }
  GramT top_d[k_degree];
  uint8_t top_v[k_degree];
#pragma unroll
  for (int t = 0; t < k_degree; ++t) {
    top_d[t] = std::numeric_limits<GramT>::max();
    top_v[t] = std::numeric_limits<uint8_t>::max();
  }

  int64_t gram_base = local_leaf * k_cluster_size * k_cluster_size;
  GramT norm_u      = gram[gram_base + u * k_cluster_size + u];
  for (int v = 0; v < leaf_n; ++v) {
    if (u == v || leaf_origins[u] == leaf_origins[v]) { continue; }
    GramT norm_v = gram[gram_base + v * k_cluster_size + v];
    GramT dot    = gram[gram_base + v * k_cluster_size + u];
    GramT distance;
    if constexpr (std::is_same_v<GramT, float>) {
      distance = fmaxf(0.0f, fmaf(-2.0f, dot, norm_u + norm_v));
    } else {
      distance = norm_u + norm_v - 2 * dot;
    }
    int worst = 0;
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
    bool valid = top_v[t] != std::numeric_limits<uint8_t>::max();
    if constexpr (std::is_same_v<GramT, float>) { valid = valid && isfinite(top_d[t]); }
    if (valid) {
      graph[static_cast<int64_t>(ids[u]) * k_degree + selected] = ids[top_v[t]];
      ++selected;
    }
  }
  degrees[ids[u]] = static_cast<uint8_t>(selected);
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
           std::vector<int64_t> const& offsets,
           build_params const& params = {}) -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream  = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);
  RAFT_EXPECTS(offsets.size() >= 3, "k=4 scaffold merge requires at least two input indices");
  RAFT_EXPECTS(rows <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
               "k=4 scaffold merge requires the merged row count to fit in uint32_t");
  RAFT_EXPECTS(params.pivot_assign_chunk > 0 && params.leaf_assign_chunk > 0 &&
                 params.max_pivot_tree_levels > 0,
               "k=4 scaffold merge runtime chunk sizes and level cap must be positive");

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

  rmm::device_uvector<uint32_t> ids(rows, stream);
  rmm::device_uvector<uint32_t> next_ids(rows, stream);
  rmm::device_uvector<uint8_t> sides(rows, stream);
  thrust::sequence(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(ids.data()),
                   thrust::device_pointer_cast(ids.data() + ids.size()),
                   uint32_t{0});
  raft::resource::sync_stream(res);

  std::vector<host_range> ranges{{0, rows}};
  std::vector<pivot_chunk> chunks;
  std::vector<uint32_t> chunk_left_counts;
  std::vector<scatter_offset> scatter_offsets;
  rmm::device_uvector<pivot_chunk> device_chunks(0, stream);
  rmm::device_uvector<uint32_t> device_left_counts(0, stream);
  rmm::device_uvector<scatter_offset> device_scatter_offsets(0, stream);

  for (int level = 0; level < params.max_pivot_tree_levels; ++level) {
    int active_count = make_pivot_chunks(ranges, level, params, chunks);
    if (active_count == 0) { break; }

    device_chunks.resize(chunks.size(), stream);
    device_left_counts.resize(chunks.size(), stream);
    raft::copy(device_chunks.data(), chunks.data(), chunks.size(), stream);
    pivot_assign_sides_kernel<<<static_cast<int>(chunks.size()), k_pivot_block_size, 0, stream>>>(
      dataset.data_handle(),
      dataset.extent(1),
      ids.data(),
      device_chunks.data(),
      static_cast<int64_t>(chunks.size()),
      sides.data(),
      device_left_counts.data());
    RAFT_CUDA_TRY(cudaGetLastError());

    chunk_left_counts.resize(chunks.size());
    raft::copy(
      chunk_left_counts.data(), device_left_counts.data(), device_left_counts.size(), stream);
    raft::resource::sync_stream(res);
    ranges = prepare_scatter(ranges, chunk_left_counts, chunks, scatter_offsets);

    device_scatter_offsets.resize(scatter_offsets.size(), stream);
    raft::copy(
      device_scatter_offsets.data(), scatter_offsets.data(), scatter_offsets.size(), stream);
    stable_scatter_kernel<<<static_cast<int>(chunks.size()), 256, 0, stream>>>(
      ids.data(),
      sides.data(),
      device_chunks.data(),
      device_scatter_offsets.data(),
      static_cast<int64_t>(chunks.size()),
      next_ids.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    std::swap(ids, next_ids);
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

  // A full Gram matrix does twice the arithmetic of a triangular distance matrix, but batched
  // cuBLAS is substantially faster than the direct leaf kernel. Process bounded leaf batches so
  // the temporary gathered vectors and Gram matrices never exceed the workspace cap.
  bool used_gemm = false;
  if constexpr (std::is_same_v<T, float>) {
    int64_t input_dimension = dataset.extent(1);
    if (input_dimension <= std::numeric_limits<int>::max()) {
      int dimension                   = static_cast<int>(input_dimension);
      size_t vector_elements_per_leaf = static_cast<size_t>(k_cluster_size) * dimension;
      size_t gram_elements_per_leaf   = static_cast<size_t>(k_cluster_size) * k_cluster_size;
      size_t vector_bytes_per_leaf    = vector_elements_per_leaf * sizeof(float);
      size_t gram_bytes_per_leaf      = gram_elements_per_leaf * sizeof(float);
      size_t bytes_per_leaf           = vector_bytes_per_leaf + gram_bytes_per_leaf;
      if (bytes_per_leaf <= params.leaf_gemm_workspace_bytes) {
        size_t batch_capacity = std::max<size_t>(
          1, std::min<size_t>(leaves.size(), params.leaf_gemm_workspace_bytes / bytes_per_leaf));
        rmm::device_uvector<float> leaf_vectors(batch_capacity * vector_elements_per_leaf, stream);
        rmm::device_uvector<float> gram(batch_capacity * gram_elements_per_leaf, stream);
        float alpha             = 1.0f;
        float beta              = 0.0f;
        long long vector_stride = static_cast<long long>(vector_elements_per_leaf);
        long long gram_stride   = static_cast<long long>(gram_elements_per_leaf);
        auto cublas_handle      = raft::resource::get_cublas_handle(res);
        RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
        for (size_t leaf_offset = 0; leaf_offset < leaves.size(); leaf_offset += batch_capacity) {
          size_t batch_size    = std::min(batch_capacity, leaves.size() - leaf_offset);
          int64_t gather_items = static_cast<int64_t>(batch_size * vector_elements_per_leaf);
          int gather_blocks =
            static_cast<int>(std::min<int64_t>((gather_items + 255) / 256, 1048576));
          gather_float_leaf_vectors_kernel<<<gather_blocks, 256, 0, stream>>>(
            dataset.data_handle(),
            dimension,
            ids.data(),
            starts.data(),
            ends.data(),
            static_cast<int64_t>(leaf_offset),
            static_cast<int64_t>(batch_size),
            leaf_vectors.data());
          RAFT_CUDA_TRY(cudaGetLastError());
          RAFT_CUBLAS_TRY(cublasGemmStridedBatchedEx(cublas_handle,
                                                     CUBLAS_OP_T,
                                                     CUBLAS_OP_N,
                                                     k_cluster_size,
                                                     k_cluster_size,
                                                     dimension,
                                                     &alpha,
                                                     leaf_vectors.data(),
                                                     CUDA_R_32F,
                                                     dimension,
                                                     vector_stride,
                                                     leaf_vectors.data(),
                                                     CUDA_R_32F,
                                                     dimension,
                                                     vector_stride,
                                                     &beta,
                                                     gram.data(),
                                                     CUDA_R_32F,
                                                     k_cluster_size,
                                                     gram_stride,
                                                     static_cast<int>(batch_size),
                                                     CUBLAS_COMPUTE_32F,
                                                     CUBLAS_GEMM_DEFAULT));
          leaf_gram_knn_kernel<<<static_cast<int>(batch_size), k_cluster_size, 0, stream>>>(
            gram.data(),
            ids.data(),
            origins.data(),
            starts.data(),
            ends.data(),
            static_cast<int64_t>(leaf_offset),
            static_cast<int64_t>(batch_size),
            graph.data_handle(),
            degrees.data());
          RAFT_CUDA_TRY(cudaGetLastError());
        }
        used_gemm = true;
      }
    }
  } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
    constexpr int64_t max_safe_dimension = std::numeric_limits<int32_t>::max() / (255 * 255);
    int64_t input_dimension              = dataset.extent(1);
    if (input_dimension <= max_safe_dimension) {
      int64_t padded_dimension        = (input_dimension + 3) & ~int64_t{3};
      size_t vector_elements_per_leaf = static_cast<size_t>(k_cluster_size) * padded_dimension;
      size_t gram_elements_per_leaf   = static_cast<size_t>(k_cluster_size) * k_cluster_size;
      size_t vector_bytes_per_leaf    = vector_elements_per_leaf * sizeof(int8_t);
      size_t gram_bytes_per_leaf      = gram_elements_per_leaf * sizeof(int32_t);
      size_t bytes_per_leaf           = vector_bytes_per_leaf + gram_bytes_per_leaf;
      if (bytes_per_leaf <= params.leaf_gemm_workspace_bytes) {
        size_t batch_capacity = std::max<size_t>(
          1, std::min<size_t>(leaves.size(), params.leaf_gemm_workspace_bytes / bytes_per_leaf));
        rmm::device_uvector<int8_t> leaf_vectors(batch_capacity * vector_elements_per_leaf, stream);
        rmm::device_uvector<int32_t> gram(batch_capacity * gram_elements_per_leaf, stream);
        int32_t alpha           = 1;
        int32_t beta            = 0;
        int dimension           = static_cast<int>(padded_dimension);
        long long vector_stride = static_cast<long long>(vector_elements_per_leaf);
        long long gram_stride   = static_cast<long long>(gram_elements_per_leaf);
        auto cublas_handle      = raft::resource::get_cublas_handle(res);
        RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
        for (size_t leaf_offset = 0; leaf_offset < leaves.size(); leaf_offset += batch_capacity) {
          size_t batch_size    = std::min(batch_capacity, leaves.size() - leaf_offset);
          int64_t gather_items = static_cast<int64_t>(batch_size * vector_elements_per_leaf);
          int gather_blocks =
            static_cast<int>(std::min<int64_t>((gather_items + 255) / 256, 1048576));
          gather_integer_leaf_vectors_kernel<<<gather_blocks, 256, 0, stream>>>(
            dataset.data_handle(),
            input_dimension,
            padded_dimension,
            ids.data(),
            starts.data(),
            ends.data(),
            static_cast<int64_t>(leaf_offset),
            static_cast<int64_t>(batch_size),
            leaf_vectors.data());
          RAFT_CUDA_TRY(cudaGetLastError());
          RAFT_CUBLAS_TRY(cublasGemmStridedBatchedEx(cublas_handle,
                                                     CUBLAS_OP_T,
                                                     CUBLAS_OP_N,
                                                     k_cluster_size,
                                                     k_cluster_size,
                                                     dimension,
                                                     &alpha,
                                                     leaf_vectors.data(),
                                                     CUDA_R_8I,
                                                     dimension,
                                                     vector_stride,
                                                     leaf_vectors.data(),
                                                     CUDA_R_8I,
                                                     dimension,
                                                     vector_stride,
                                                     &beta,
                                                     gram.data(),
                                                     CUDA_R_32I,
                                                     k_cluster_size,
                                                     gram_stride,
                                                     static_cast<int>(batch_size),
                                                     CUBLAS_COMPUTE_32I,
                                                     CUBLAS_GEMM_DEFAULT));
          leaf_gram_knn_kernel<<<static_cast<int>(batch_size), k_cluster_size, 0, stream>>>(
            gram.data(),
            ids.data(),
            origins.data(),
            starts.data(),
            ends.data(),
            static_cast<int64_t>(leaf_offset),
            static_cast<int64_t>(batch_size),
            graph.data_handle(),
            degrees.data());
          RAFT_CUDA_TRY(cudaGetLastError());
        }
        used_gemm = true;
      }
    }
  }

  if (!used_gemm) {
    leaf_cross_knn_kernel<<<static_cast<int>(leaves.size()), k_cluster_size, 0, stream>>>(
      dataset.data_handle(),
      dataset.extent(1),
      ids.data(),
      origins.data(),
      starts.data(),
      ends.data(),
      leaves.size(),
      graph.data_handle(),
      degrees.data());
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  int blocks = static_cast<int>((rows + 255) / 256);
  pad_scaffold_kernel<<<blocks, 256, 0, stream>>>(
    graph.data_handle(), degrees.data(), fallback.data(), rows);
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  return graph;
}

static __global__ void copy_partition_with_scaffold_kernel(uint32_t const* source,
                                                           uint32_t const* scaffold,
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
  int64_t global_row       = row + offset;
  int64_t destination_base = global_row * destination_degree;
  for (int64_t j = 0; j < base_degree; ++j) {
    destination[destination_base + j] = source[source_base + (j % source_degree)] + offset;
  }
#pragma unroll
  for (int j = 0; j < k_degree; ++j) {
    destination[destination_base + base_degree + j] = scaffold[global_row * k_degree + j];
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
    copy_partition_with_scaffold_kernel<<<blocks, 256, 0, stream>>>(
      source.data_handle(),
      scaffold.data_handle(),
      source.extent(0),
      source.extent(1),
      graph.data_handle(),
      graph_degree,
      base_degree,
      static_cast<uint32_t>(offsets[part]));
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  raft::resource::sync_stream(res);
  return graph;
}

}  // namespace cuvs::neighbors::cagra::detail::merge_scaffold

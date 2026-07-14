/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cluster/kmeans_balanced_build_clusters_impl.cuh>
#include <neighbors/detail/cagra/cagra_merge_scaffold.cuh>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace fastener_experiment::kmeans_scaffold {

using cuvs::neighbors::cagra::detail::merge_scaffold::host_range;
inline constexpr int k_max_branching = 5;
inline constexpr int k_tree_chunk_size = 256;

struct cluster_layout {
  rmm::device_uvector<uint32_t> ids;
  std::vector<host_range> clusters;

  cluster_layout(rmm::device_uvector<uint32_t> &&input_ids,
                 std::vector<host_range> &&input_clusters)
      : ids(std::move(input_ids)), clusters(std::move(input_clusters)) {}
};

template <typename T>
auto flat_balanced(
    raft::resources const &res,
    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
    int target_cluster_size, int iterations) -> cluster_layout {
  auto stream = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);
  int64_t clusters = (rows + target_cluster_size - 1) / target_cluster_size;
  cuvs::cluster::kmeans::balanced_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;
  params.n_iters = iterations;

  auto centroids = raft::make_device_matrix<float, int64_t>(res, clusters,
                                                            dataset.extent(1));
  auto labels = raft::make_device_vector<uint32_t, int64_t>(res, rows);
  auto cluster_size =
      raft::make_device_vector<uint32_t, int64_t>(res, clusters);
  if constexpr (std::is_same_v<T, float>) {
    cuvs::cluster::kmeans_balanced::helpers::build_clusters(
        res, params, raft::make_const_mdspan(dataset), centroids.view(),
        labels.view(), cluster_size.view(), raft::identity_op{},
        std::optional<raft::device_vector_view<const float>>{});
  } else {
    cuvs::cluster::kmeans_balanced::helpers::build_clusters(
        res, params, raft::make_const_mdspan(dataset), centroids.view(),
        labels.view(), cluster_size.view(), raft::cast_op<float>{},
        std::optional<raft::device_vector_view<const float>>{});
  }

  rmm::device_uvector<uint32_t> ids(rows, stream);
  thrust::sequence(
      thrust::cuda::par.on(stream), thrust::device_pointer_cast(ids.data()),
      thrust::device_pointer_cast(ids.data() + ids.size()), uint32_t{0});
  thrust::sort_by_key(thrust::cuda::par.on(stream), labels.data_handle(),
                      labels.data_handle() + rows, ids.data());

  std::vector<uint32_t> sizes(clusters);
  raft::copy(sizes.data(), cluster_size.data_handle(), sizes.size(), stream);
  raft::resource::sync_stream(res);
  std::vector<host_range> ranges;
  ranges.reserve(clusters);
  int64_t start = 0;
  for (uint32_t size : sizes) {
    if (size > 0) {
      ranges.push_back({start, start + size});
    }
    start += size;
  }
  RAFT_EXPECTS(start == rows,
               "Flat k-means cluster sizes do not sum to the dataset size");
  return cluster_layout(std::move(ids), std::move(ranges));
}

struct tree_chunk {
  uint32_t start;
  uint32_t end;
  uint32_t active_parent;
};

struct branch_counts {
  uint32_t value[k_max_branching];
};

struct branch_offsets {
  uint32_t value[k_max_branching];
};

template <typename T>
__global__ void
initialize_tree_centroids_kernel(T const *dataset, int64_t dim,
                                 uint32_t const *ids, host_range const *parents,
                                 int active_parents, int branching,
                                 float *centroids) {
  int centroid = blockIdx.x;
  if (centroid >= active_parents * branching) {
    return;
  }
  int parent = centroid / branching;
  int child = centroid % branching;
  auto range = parents[parent];
  int64_t n = range.end - range.start;
  int64_t position = range.start + ((child + 1) * n) / (branching + 1);
  uint32_t row = ids[position];
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    centroids[static_cast<int64_t>(centroid) * dim + d] =
        static_cast<float>(dataset[static_cast<int64_t>(row) * dim + d]);
  }
}

template <typename T>
__global__ void
assign_tree_children_kernel(T const *dataset, int64_t dim, uint32_t const *ids,
                            tree_chunk const *chunks, int64_t chunk_count,
                            float const *centroids, int branching,
                            uint8_t *assignments, branch_counts *counts) {
  int64_t chunk_index = blockIdx.x;
  if (chunk_index >= chunk_count) {
    return;
  }
  auto chunk = chunks[chunk_index];
  uint32_t local[k_max_branching] = {};
  int64_t position = chunk.start + threadIdx.x;
  if (position < chunk.end) {
    uint32_t row = ids[position];
    int preferred = static_cast<int>(
        cuvs::neighbors::cagra::detail::merge_scaffold::splitmix64(row) %
        static_cast<uint64_t>(branching));
    int best = 0;
    float best_dst = std::numeric_limits<float>::max();
    for (int child = 0; child < branching; ++child) {
      float distance = 0.0f;
      auto center =
          centroids +
          (static_cast<int64_t>(chunk.active_parent) * branching + child) * dim;
      auto point = dataset + static_cast<int64_t>(row) * dim;
      for (int64_t d = 0; d < dim; ++d) {
        float diff = static_cast<float>(point[d]) - center[d];
        distance += diff * diff;
      }
      if (distance < best_dst || (distance == best_dst && child == preferred)) {
        best = child;
        best_dst = distance;
      }
    }
    assignments[position] = static_cast<uint8_t>(best);
    local[best] = 1;
  }

  __shared__ uint32_t reductions[k_max_branching][k_tree_chunk_size];
  for (int child = 0; child < branching; ++child) {
    reductions[child][threadIdx.x] = local[child];
  }
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      for (int child = 0; child < branching; ++child) {
        reductions[child][threadIdx.x] +=
            reductions[child][threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int child = 0; child < branching; ++child) {
      counts[chunk_index].value[child] = reductions[child][0];
    }
  }
}

__global__ void scatter_tree_children_kernel(uint32_t const *input_ids,
                                             uint8_t const *assignments,
                                             tree_chunk const *chunks,
                                             branch_offsets const *offsets,
                                             int64_t chunk_count, int branching,
                                             uint32_t *output_ids) {
  int64_t chunk_index = blockIdx.x;
  if (chunk_index >= chunk_count) {
    return;
  }
  auto chunk = chunks[chunk_index];
  int64_t position = chunk.start + threadIdx.x;
  bool valid = position < chunk.end;
  int assignment = valid ? assignments[position] : 0;
  __shared__ uint32_t prefixes[k_max_branching][k_tree_chunk_size];
  for (int child = 0; child < branching; ++child) {
    prefixes[child][threadIdx.x] = valid && assignment == child ? 1u : 0u;
  }
  __syncthreads();
  for (int child = 0; child < branching; ++child) {
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
      uint32_t add =
          threadIdx.x >= offset ? prefixes[child][threadIdx.x - offset] : 0;
      __syncthreads();
      if (threadIdx.x >= offset) {
        prefixes[child][threadIdx.x] += add;
      }
      __syncthreads();
    }
  }
  if (valid) {
    uint32_t before = prefixes[assignment][threadIdx.x] - 1;
    output_ids[offsets[chunk_index].value[assignment] + before] =
        input_ids[position];
  }
}

template <typename T>
__global__ void
recompute_tree_centroids_kernel(T const *dataset, int64_t dim,
                                uint32_t const *ids, host_range const *children,
                                int child_count, float *centroids) {
  int child = blockIdx.x;
  int64_t d = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (child >= child_count || d >= dim) {
    return;
  }
  auto range = children[child];
  if (range.end == range.start) {
    return;
  }
  float sum = 0.0f;
  for (int64_t position = range.start; position < range.end; ++position) {
    uint32_t row = ids[position];
    sum += static_cast<float>(dataset[static_cast<int64_t>(row) * dim + d]);
  }
  centroids[static_cast<int64_t>(child) * dim + d] =
      sum / static_cast<float>(range.end - range.start);
}

inline auto prepare_tree_scatter(std::vector<host_range> const &parents,
                                 std::vector<tree_chunk> const &chunks,
                                 std::vector<branch_counts> const &chunk_counts,
                                 int branching,
                                 std::vector<branch_offsets> &offsets)
    -> std::pair<std::vector<host_range>,
                 std::vector<std::array<uint32_t, k_max_branching>>> {
  std::vector<std::array<uint32_t, k_max_branching>> totals(parents.size());
  for (auto &value : totals) {
    value.fill(0);
  }
  for (size_t i = 0; i < chunks.size(); ++i) {
    for (int child = 0; child < branching; ++child) {
      totals[chunks[i].active_parent][child] += chunk_counts[i].value[child];
    }
  }
  std::vector<host_range> child_ranges(parents.size() * branching);
  for (size_t parent = 0; parent < parents.size(); ++parent) {
    int64_t cursor = parents[parent].start;
    for (int child = 0; child < branching; ++child) {
      child_ranges[parent * branching + child] = {
          cursor, cursor + totals[parent][child]};
      cursor += totals[parent][child];
    }
    RAFT_EXPECTS(cursor == parents[parent].end,
                 "Tree k-means assignment lost rows");
  }

  offsets.resize(chunks.size());
  std::vector<std::array<uint32_t, k_max_branching>> prefixes(parents.size());
  for (auto &value : prefixes) {
    value.fill(0);
  }
  for (size_t i = 0; i < chunks.size(); ++i) {
    auto parent = chunks[i].active_parent;
    for (int child = 0; child < branching; ++child) {
      offsets[i].value[child] =
          static_cast<uint32_t>(child_ranges[parent * branching + child].start +
                                prefixes[parent][child]);
      prefixes[parent][child] += chunk_counts[i].value[child];
    }
  }
  return {std::move(child_ranges), std::move(totals)};
}

template <typename T>
auto lloyd_tree(
    raft::resources const &res,
    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
    int branching, int leaf_size, int iterations) -> cluster_layout {
  RAFT_EXPECTS(branching == 2 || branching == 5,
               "Tree k-means branching must be two or five");
  RAFT_EXPECTS(leaf_size > 0 && iterations > 0,
               "Tree k-means leaf size and iterations must be positive");
  auto stream = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);
  rmm::device_uvector<uint32_t> ids(rows, stream);
  rmm::device_uvector<uint32_t> next_ids(rows, stream);
  rmm::device_uvector<uint8_t> assignments(rows, stream);
  thrust::sequence(
      thrust::cuda::par.on(stream), thrust::device_pointer_cast(ids.data()),
      thrust::device_pointer_cast(ids.data() + ids.size()), uint32_t{0});
  std::vector<host_range> ranges{{0, rows}};

  for (int level = 0; level < 64; ++level) {
    std::vector<host_range> active;
    std::vector<int> range_to_active(ranges.size(), -1);
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (ranges[i].end - ranges[i].start > leaf_size) {
        range_to_active[i] = static_cast<int>(active.size());
        active.push_back(ranges[i]);
      }
    }
    if (active.empty()) {
      return cluster_layout(std::move(ids), std::move(ranges));
    }

    std::vector<tree_chunk> chunks;
    for (size_t parent = 0; parent < active.size(); ++parent) {
      for (int64_t start = active[parent].start; start < active[parent].end;
           start += k_tree_chunk_size) {
        chunks.push_back({static_cast<uint32_t>(start),
                          static_cast<uint32_t>(std::min<int64_t>(
                              active[parent].end, start + k_tree_chunk_size)),
                          static_cast<uint32_t>(parent)});
      }
    }
    rmm::device_uvector<host_range> device_parents(active.size(), stream);
    rmm::device_uvector<tree_chunk> device_chunks(chunks.size(), stream);
    rmm::device_uvector<branch_counts> device_counts(chunks.size(), stream);
    rmm::device_uvector<branch_offsets> device_offsets(chunks.size(), stream);
    rmm::device_uvector<host_range> device_children(active.size() * branching,
                                                    stream);
    auto centroids = raft::make_device_matrix<float, int64_t>(
        res, static_cast<int64_t>(active.size()) * branching,
        dataset.extent(1));
    raft::copy(device_parents.data(), active.data(), active.size(), stream);
    raft::copy(device_chunks.data(), chunks.data(), chunks.size(), stream);
    initialize_tree_centroids_kernel<<<
        static_cast<int>(active.size()) * branching, 256, 0, stream>>>(
        dataset.data_handle(), dataset.extent(1), ids.data(),
        device_parents.data(), active.size(), branching,
        centroids.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());

    std::vector<host_range> child_ranges;
    std::vector<std::array<uint32_t, k_max_branching>> child_totals;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      assign_tree_children_kernel<<<static_cast<int>(chunks.size()),
                                    k_tree_chunk_size, 0, stream>>>(
          dataset.data_handle(), dataset.extent(1), ids.data(),
          device_chunks.data(), chunks.size(), centroids.data_handle(),
          branching, assignments.data(), device_counts.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      std::vector<branch_counts> counts(chunks.size());
      raft::copy(counts.data(), device_counts.data(), counts.size(), stream);
      raft::resource::sync_stream(res);
      std::vector<branch_offsets> offsets;
      std::tie(child_ranges, child_totals) =
          prepare_tree_scatter(active, chunks, counts, branching, offsets);
      raft::copy(device_offsets.data(), offsets.data(), offsets.size(), stream);
      raft::copy(device_children.data(), child_ranges.data(),
                 child_ranges.size(), stream);
      raft::copy(next_ids.data(), ids.data(), ids.size(), stream);
      scatter_tree_children_kernel<<<static_cast<int>(chunks.size()),
                                     k_tree_chunk_size, 0, stream>>>(
          ids.data(), assignments.data(), device_chunks.data(),
          device_offsets.data(), chunks.size(), branching, next_ids.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      std::swap(ids, next_ids);
      dim3 centroid_grid(
          static_cast<unsigned>(child_ranges.size()),
          static_cast<unsigned>((dataset.extent(1) + 127) / 128));
      recompute_tree_centroids_kernel<<<centroid_grid, 128, 0, stream>>>(
          dataset.data_handle(), dataset.extent(1), ids.data(),
          device_children.data(), child_ranges.size(), centroids.data_handle());
      RAFT_CUDA_TRY(cudaGetLastError());
    }

    std::vector<host_range> next_ranges;
    next_ranges.reserve(ranges.size() + active.size() * (branching - 1));
    for (size_t i = 0; i < ranges.size(); ++i) {
      int active_index = range_to_active[i];
      if (active_index < 0) {
        next_ranges.push_back(ranges[i]);
      } else {
        for (int child = 0; child < branching; ++child) {
          auto range = child_ranges[active_index * branching + child];
          if (range.end > range.start) {
            next_ranges.push_back(range);
          }
        }
      }
    }
    RAFT_EXPECTS(next_ranges.size() > ranges.size(),
                 "Tree k-means failed to subdivide an oversized cluster");
    ranges = std::move(next_ranges);
  }
  RAFT_FAIL("Tree k-means exceeded the 64-level safety limit");
}

struct query_chunk {
  uint32_t query_start;
  uint32_t query_end;
  uint32_t cluster_start;
  uint32_t cluster_end;
};

template <typename T, int Degree>
__global__ void
cluster_cross_knn_kernel(T const *dataset, int64_t dim, uint32_t const *ids,
                         uint32_t const *origins, query_chunk const *chunks,
                         int64_t chunk_count, uint32_t *graph,
                         uint8_t *degrees) {
  int64_t chunk_index = blockIdx.x;
  if (chunk_index >= chunk_count) {
    return;
  }
  auto chunk = chunks[chunk_index];
  uint32_t position = chunk.query_start + threadIdx.x;
  if (position >= chunk.query_end) {
    return;
  }
  uint32_t row = ids[position];
  float top_distance[Degree];
  uint32_t top_id[Degree];
#pragma unroll
  for (int i = 0; i < Degree; ++i) {
    top_distance[i] = std::numeric_limits<float>::max();
    top_id[i] = std::numeric_limits<uint32_t>::max();
  }
  for (uint32_t candidate_position = chunk.cluster_start;
       candidate_position < chunk.cluster_end; ++candidate_position) {
    uint32_t candidate = ids[candidate_position];
    if (candidate == row || origins[candidate] == origins[row]) {
      continue;
    }
    float distance =
        cuvs::neighbors::cagra::detail::merge_scaffold::l2_distance(
            dataset, dim, row, candidate);
    int worst = 0;
#pragma unroll
    for (int i = 1; i < Degree; ++i) {
      if (top_distance[i] > top_distance[worst]) {
        worst = i;
      }
    }
    if (distance < top_distance[worst]) {
      top_distance[worst] = distance;
      top_id[worst] = candidate;
    }
  }
  int selected = 0;
#pragma unroll
  for (int i = 0; i < Degree; ++i) {
    if (top_id[i] != std::numeric_limits<uint32_t>::max()) {
      graph[static_cast<int64_t>(row) * Degree + selected++] = top_id[i];
    }
  }
  degrees[row] = static_cast<uint8_t>(selected);
}

template <typename T, int Degree>
auto build_graph_impl(
    raft::resources const &res,
    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
    std::vector<int64_t> const &partition_offsets, cluster_layout const &layout)
    -> raft::device_matrix<uint32_t, int64_t> {
  auto stream = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, rows, Degree);
  rmm::device_uvector<uint8_t> degrees(rows, stream);
  rmm::device_uvector<uint32_t> origins(rows, stream);
  rmm::device_uvector<uint32_t> fallback(rows, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(degrees.data(), 0, degrees.size(), stream));
  for (size_t part = 0; part + 1 < partition_offsets.size(); ++part) {
    int64_t part_rows = partition_offsets[part + 1] - partition_offsets[part];
    uint32_t other = static_cast<uint32_t>(
        partition_offsets[(part + 1) % (partition_offsets.size() - 1)]);
    int blocks = static_cast<int>((part_rows + 255) / 256);
    cuvs::neighbors::cagra::detail::merge_scaffold::
        initialize_partition_kernel<<<blocks, 256, 0, stream>>>(
            origins.data(), fallback.data(), partition_offsets[part], part_rows,
            static_cast<uint32_t>(part), other);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  std::vector<host_range> small_clusters;
  std::vector<query_chunk> large_chunks;
  for (auto const &cluster : layout.clusters) {
    if (cluster.end - cluster.start <=
        cuvs::neighbors::cagra::detail::merge_scaffold::k_cluster_size) {
      small_clusters.push_back(cluster);
    } else {
      for (int64_t start = cluster.start; start < cluster.end;
           start += k_tree_chunk_size) {
        large_chunks.push_back({static_cast<uint32_t>(start),
                                static_cast<uint32_t>(std::min<int64_t>(
                                    cluster.end, start + k_tree_chunk_size)),
                                static_cast<uint32_t>(cluster.start),
                                static_cast<uint32_t>(cluster.end)});
      }
    }
  }

  if (!small_clusters.empty()) {
    std::vector<uint32_t> starts_host(small_clusters.size());
    std::vector<uint32_t> ends_host(small_clusters.size());
    for (size_t i = 0; i < small_clusters.size(); ++i) {
      starts_host[i] = static_cast<uint32_t>(small_clusters[i].start);
      ends_host[i] = static_cast<uint32_t>(small_clusters[i].end);
    }
    rmm::device_uvector<uint32_t> starts(starts_host.size(), stream);
    rmm::device_uvector<uint32_t> ends(ends_host.size(), stream);
    raft::copy(starts.data(), starts_host.data(), starts.size(), stream);
    raft::copy(ends.data(), ends_host.data(), ends.size(), stream);

    constexpr size_t workspace_bytes = size_t{2} * 1024 * 1024 * 1024;
    constexpr int leaf_size =
        cuvs::neighbors::cagra::detail::merge_scaffold::k_cluster_size;
    bool used_gemm = false;
    if constexpr (std::is_same_v<T, float>) {
      int64_t input_dimension = dataset.extent(1);
      if (input_dimension <= std::numeric_limits<int>::max()) {
        int dimension = static_cast<int>(input_dimension);
        size_t vector_elements = static_cast<size_t>(leaf_size) * dimension;
        size_t gram_elements = static_cast<size_t>(leaf_size) * leaf_size;
        size_t bytes_per_cluster =
            vector_elements * sizeof(float) + gram_elements * sizeof(float);
        if (bytes_per_cluster <= workspace_bytes) {
          size_t batch_capacity = std::max<size_t>(
              1, std::min<size_t>(small_clusters.size(),
                                  workspace_bytes / bytes_per_cluster));
          rmm::device_uvector<float> vectors(batch_capacity * vector_elements,
                                             stream);
          rmm::device_uvector<float> gram(batch_capacity * gram_elements,
                                          stream);
          float alpha = 1.0f;
          float beta = 0.0f;
          long long vector_stride = static_cast<long long>(vector_elements);
          long long gram_stride = static_cast<long long>(gram_elements);
          auto handle = raft::resource::get_cublas_handle(res);
          RAFT_CUBLAS_TRY(
              cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
          for (size_t offset = 0; offset < small_clusters.size();
               offset += batch_capacity) {
            size_t batch =
                std::min(batch_capacity, small_clusters.size() - offset);
            int64_t items = static_cast<int64_t>(batch * vector_elements);
            int gather_blocks = static_cast<int>(
                std::min<int64_t>((items + 255) / 256, 1048576));
            cuvs::neighbors::cagra::detail::merge_scaffold::
                gather_float_leaf_vectors_kernel<<<gather_blocks, 256, 0,
                                                   stream>>>(
                    dataset.data_handle(), dimension, layout.ids.data(),
                    starts.data(), ends.data(), static_cast<int64_t>(offset),
                    static_cast<int64_t>(batch), vectors.data());
            RAFT_CUDA_TRY(cudaGetLastError());
            RAFT_CUBLAS_TRY(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, leaf_size, leaf_size,
                dimension, &alpha, vectors.data(), CUDA_R_32F, dimension,
                vector_stride, vectors.data(), CUDA_R_32F, dimension,
                vector_stride, &beta, gram.data(), CUDA_R_32F, leaf_size,
                gram_stride, static_cast<int>(batch), CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT));
            cuvs::neighbors::cagra::detail::merge_scaffold::
                leaf_gram_knn_kernel<float, Degree>
                <<<static_cast<int>(batch), leaf_size, 0, stream>>>(
                    gram.data(), layout.ids.data(), origins.data(),
                    starts.data(), ends.data(), static_cast<int64_t>(offset),
                    static_cast<int64_t>(batch), graph.data_handle(),
                    degrees.data());
            RAFT_CUDA_TRY(cudaGetLastError());
          }
          used_gemm = true;
        }
      }
    } else if constexpr (std::is_same_v<T, uint8_t> ||
                         std::is_same_v<T, int8_t>) {
      constexpr int64_t max_safe_dimension =
          std::numeric_limits<int32_t>::max() / (255 * 255);
      int64_t input_dimension = dataset.extent(1);
      if (input_dimension <= max_safe_dimension) {
        int64_t padded_dimension = (input_dimension + 3) & ~int64_t{3};
        size_t vector_elements =
            static_cast<size_t>(leaf_size) * padded_dimension;
        size_t gram_elements = static_cast<size_t>(leaf_size) * leaf_size;
        size_t bytes_per_cluster =
            vector_elements * sizeof(int8_t) + gram_elements * sizeof(int32_t);
        if (bytes_per_cluster <= workspace_bytes) {
          size_t batch_capacity = std::max<size_t>(
              1, std::min<size_t>(small_clusters.size(),
                                  workspace_bytes / bytes_per_cluster));
          rmm::device_uvector<int8_t> vectors(batch_capacity * vector_elements,
                                              stream);
          rmm::device_uvector<int32_t> gram(batch_capacity * gram_elements,
                                            stream);
          int32_t alpha = 1;
          int32_t beta = 0;
          int dimension = static_cast<int>(padded_dimension);
          long long vector_stride = static_cast<long long>(vector_elements);
          long long gram_stride = static_cast<long long>(gram_elements);
          auto handle = raft::resource::get_cublas_handle(res);
          RAFT_CUBLAS_TRY(
              cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
          for (size_t offset = 0; offset < small_clusters.size();
               offset += batch_capacity) {
            size_t batch =
                std::min(batch_capacity, small_clusters.size() - offset);
            int64_t items = static_cast<int64_t>(batch * vector_elements);
            int gather_blocks = static_cast<int>(
                std::min<int64_t>((items + 255) / 256, 1048576));
            cuvs::neighbors::cagra::detail::merge_scaffold::
                gather_integer_leaf_vectors_kernel<<<gather_blocks, 256, 0,
                                                     stream>>>(
                    dataset.data_handle(), input_dimension, padded_dimension,
                    layout.ids.data(), starts.data(), ends.data(),
                    static_cast<int64_t>(offset), static_cast<int64_t>(batch),
                    vectors.data());
            RAFT_CUDA_TRY(cudaGetLastError());
            RAFT_CUBLAS_TRY(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, leaf_size, leaf_size,
                dimension, &alpha, vectors.data(), CUDA_R_8I, dimension,
                vector_stride, vectors.data(), CUDA_R_8I, dimension,
                vector_stride, &beta, gram.data(), CUDA_R_32I, leaf_size,
                gram_stride, static_cast<int>(batch), CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT));
            cuvs::neighbors::cagra::detail::merge_scaffold::
                leaf_gram_knn_kernel<int32_t, Degree>
                <<<static_cast<int>(batch), leaf_size, 0, stream>>>(
                    gram.data(), layout.ids.data(), origins.data(),
                    starts.data(), ends.data(), static_cast<int64_t>(offset),
                    static_cast<int64_t>(batch), graph.data_handle(),
                    degrees.data());
            RAFT_CUDA_TRY(cudaGetLastError());
          }
          used_gemm = true;
        }
      }
    }
    if (!used_gemm) {
      cuvs::neighbors::cagra::detail::merge_scaffold::leaf_cross_knn_kernel<
          T, Degree>
          <<<static_cast<int>(small_clusters.size()), leaf_size, 0, stream>>>(
              dataset.data_handle(), dataset.extent(1), layout.ids.data(),
              origins.data(), starts.data(), ends.data(), small_clusters.size(),
              graph.data_handle(), degrees.data());
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  }

  if (!large_chunks.empty()) {
    rmm::device_uvector<query_chunk> device_chunks(large_chunks.size(), stream);
    raft::copy(device_chunks.data(), large_chunks.data(), large_chunks.size(),
               stream);
    cluster_cross_knn_kernel<T, Degree>
        <<<static_cast<int>(large_chunks.size()), k_tree_chunk_size, 0,
           stream>>>(dataset.data_handle(), dataset.extent(1),
                     layout.ids.data(), origins.data(), device_chunks.data(),
                     large_chunks.size(), graph.data_handle(), degrees.data());
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  int blocks = static_cast<int>((rows + 255) / 256);
  cuvs::neighbors::cagra::detail::merge_scaffold::
      pad_scaffold_kernel<<<blocks, 256, 0, stream>>>(
          graph.data_handle(), degrees.data(), fallback.data(), rows, Degree);
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  return graph;
}

template <typename T>
auto build_graph(
    raft::resources const &res,
    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
    std::vector<int64_t> const &partition_offsets, cluster_layout const &layout,
    int neighbors) -> raft::device_matrix<uint32_t, int64_t> {
  switch (neighbors) {
  case 1:
    return build_graph_impl<T, 1>(res, dataset, partition_offsets, layout);
  case 2:
    return build_graph_impl<T, 2>(res, dataset, partition_offsets, layout);
  case 4:
    return build_graph_impl<T, 4>(res, dataset, partition_offsets, layout);
  case 8:
    return build_graph_impl<T, 8>(res, dataset, partition_offsets, layout);
  case 16:
    return build_graph_impl<T, 16>(res, dataset, partition_offsets, layout);
  case 32:
    return build_graph_impl<T, 32>(res, dataset, partition_offsets, layout);
  default:
    RAFT_FAIL("K-means scaffold degree must be one of 1, 2, 4, 8, 16, or 32");
  }
}

} // namespace fastener_experiment::kmeans_scaffold

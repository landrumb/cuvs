/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cagra_binary_cross_query_common.cuh"

#include <raft/core/copy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/matrix/init.cuh>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra::detail::binary_cross_query {

template <typename T>
__global__ void
fill_merged_dataset_kernel(T *merged_dataset, const T *dataset_a,
                           const T *dataset_b, int64_t n_a, int64_t n_b,
                           int64_t dim, int64_t stride_a, int64_t stride_b) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto size = (n_a + n_b) * dim;
  if (idx >= size) {
    return;
  }

  auto row = idx / dim;
  auto col = idx % dim;
  if (row < n_a) {
    merged_dataset[idx] = dataset_a[row * stride_a + col];
  } else {
    auto local_row = row - n_a;
    merged_dataset[idx] = dataset_b[local_row * stride_b + col];
  }
}

static __global__ void fill_within_neighbors_kernel(uint32_t *within_all,
                                                    const uint32_t *graph_a,
                                                    const uint32_t *graph_b,
                                                    int64_t n_a, int64_t n_b,
                                                    uint32_t degree) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = (n_a + n_b) * int64_t(degree);
  if (idx >= total) {
    return;
  }

  auto row = idx / degree;
  auto col = idx % degree;
  if (row < n_a) {
    within_all[idx] = graph_a[row * degree + col];
  } else {
    auto local_row = row - n_a;
    within_all[idx] = graph_b[local_row * degree + col] + uint32_t(n_a);
  }
}

template <typename T>
__global__ void
compute_within_distances_kernel(const T *dataset, const uint32_t *within_all,
                                float *within_dists, int64_t n_total,
                                int64_t dim, uint32_t degree,
                                cuvs::distance::DistanceType metric) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = n_total * int64_t(degree);
  if (idx >= total) {
    return;
  }

  auto row = idx / degree;
  auto nbr = within_all[idx];
  if (nbr == uint32_t(row)) {
    within_dists[idx] = std::numeric_limits<float>::infinity();
    return;
  }

  float dist = 0.0f;
  float norm2_dst = 0.0f;
  auto row_offset = row * dim;
  auto nbr_offset = int64_t(nbr) * dim;

  for (int64_t d = 0; d < dim; ++d) {
    auto src = static_cast<float>(dataset[row_offset + d]);
    auto dst = static_cast<float>(dataset[nbr_offset + d]);
    if (metric == cuvs::distance::DistanceType::L2Expanded) {
      auto diff = src - dst;
      dist += diff * diff;
    } else {
      dist -= src * dst;
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        norm2_dst += dst * dst;
      }
    }
  }

  if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    dist /= sqrtf(fmaxf(norm2_dst, 1e-20f));
  }

  within_dists[idx] = dist;
}

static __global__ void add_offset_kernel(uint32_t *indices, int64_t size,
                                         uint32_t offset) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  indices[idx] += offset;
}

static __global__ void
compute_ratio_kernel(const float *within_dists, const float *light_dists_a,
                     const float *light_dists_b, float *ratio, int64_t n_a,
                     int64_t n_total, uint32_t degree, uint32_t k_cross_light) {
  auto row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_total) {
    return;
  }

  float within_farthest = 0.0f;
  auto row_offset = row * degree;
  for (uint32_t col = 0; col < degree; ++col) {
    within_farthest = fmaxf(within_farthest, within_dists[row_offset + col]);
  }

  auto cross_nearest = row < n_a ? light_dists_a[row * k_cross_light]
                                 : light_dists_b[(row - n_a) * k_cross_light];
  ratio[row] = cross_nearest / fmaxf(within_farthest, 1e-10f);
}

static __global__ void mark_boundary_kernel(const int64_t *boundary_rows,
                                            int64_t count,
                                            uint8_t *is_boundary) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  is_boundary[boundary_rows[idx]] = 1;
}

static __global__ void seed_cross_from_light_kernel(
    uint32_t *cross_nbrs, float *cross_dists, const uint32_t *light_nbrs_a,
    const uint32_t *light_nbrs_b, const float *light_dists_a,
    const float *light_dists_b, int64_t n_a, int64_t n_b,
    uint32_t k_cross_light, uint32_t k_cross_full) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = (n_a + n_b) * int64_t(k_cross_light);
  if (idx >= total) {
    return;
  }

  auto row = idx / k_cross_light;
  auto col = idx % k_cross_light;
  auto dst = row * k_cross_full + col;
  if (row < n_a) {
    auto src = row * k_cross_light + col;
    cross_nbrs[dst] = light_nbrs_a[src];
    cross_dists[dst] = light_dists_a[src];
  } else {
    auto local_row = row - n_a;
    auto src = local_row * k_cross_light + col;
    cross_nbrs[dst] = light_nbrs_b[src];
    cross_dists[dst] = light_dists_b[src];
  }
}

static __global__ void scatter_boundary_results_kernel(
    const int64_t *boundary_rows, int64_t boundary_count,
    const uint32_t *boundary_nbrs, const float *boundary_dists,
    uint32_t *cross_nbrs, float *cross_dists, uint32_t k_cross_full) {
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = boundary_count * int64_t(k_cross_full);
  if (idx >= total) {
    return;
  }

  auto boundary_idx = idx / k_cross_full;
  auto col = idx % k_cross_full;
  auto row = boundary_rows[boundary_idx];
  auto dst = row * k_cross_full + col;
  cross_nbrs[dst] = boundary_nbrs[idx];
  cross_dists[dst] = boundary_dists[idx];
}

static __global__ void fill_flat_candidates_kernel(
    int64_t *flat_rows, uint32_t *flat_neighbors, float *flat_dists,
    const uint32_t *within_all, const float *within_dists,
    const uint32_t *cross_nbrs, const float *cross_dists, int64_t n_total,
    uint32_t degree, uint32_t sentinel) {
  auto cand_width = uint32_t(2 * degree);
  auto idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = n_total * int64_t(cand_width);
  if (idx >= total) {
    return;
  }

  auto row = idx / cand_width;
  auto col = idx % cand_width;

  flat_rows[idx] = row;

  uint32_t nbr;
  float dist;
  if (col < degree) {
    nbr = within_all[row * degree + col];
    dist = within_dists[row * degree + col];
  } else {
    auto cross_col = col - degree;
    nbr = cross_nbrs[row * degree + cross_col];
    dist = cross_dists[row * degree + cross_col];
  }

  if (nbr == uint32_t(row) || dist == std::numeric_limits<float>::infinity()) {
    flat_neighbors[idx] = sentinel;
    flat_dists[idx] = std::numeric_limits<float>::infinity();
  } else {
    flat_neighbors[idx] = nbr;
    flat_dists[idx] = dist;
  }
}

static __global__ void build_overgraph_kernel(const int64_t *row_offsets,
                                              const uint32_t *unique_neighbors,
                                              uint32_t *over_graph,
                                              int64_t n_total,
                                              uint32_t overdegree,
                                              uint32_t sentinel) {
  auto row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_total) {
    return;
  }

  auto start = row_offsets[row];
  auto end = row_offsets[row + 1];
  uint32_t first_valid = sentinel;
  uint32_t written = 0;

  for (int64_t i = start; i < end && written < overdegree; ++i) {
    auto nbr = unique_neighbors[i];
    if (nbr == sentinel || nbr == uint32_t(row)) {
      continue;
    }
    if (first_valid == sentinel) {
      first_valid = nbr;
    }
    over_graph[row * overdegree + written] = nbr;
    ++written;
  }

  if (first_valid == sentinel) {
    first_valid = row == 0 ? 1u : 0u;
  }
  for (; written < overdegree; ++written) {
    over_graph[row * overdegree + written] = first_valid;
  }
}

template <class T, class IdxT>
void run_cross_search_to_device(
    raft::resources const &handle, const cagra::index<T, IdxT> &target_index,
    raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
    uint32_t k, uint32_t global_offset,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
    raft::device_matrix_view<float, int64_t, raft::row_major> distances) {
  cagra::search_params search_params;
  search_params.itopk_size = search_itopk_size(k);
  search_params.max_queries = 50000;

  cagra::search(handle, search_params, target_index, queries, neighbors,
                distances);
  if (global_offset == 0) {
    return;
  }

  auto total = neighbors.extent(0) * neighbors.extent(1);
  auto threads = 256;
  auto blocks = (total + threads - 1) / threads;
  add_offset_kernel<<<blocks, threads, 0,
                      raft::resource::get_cuda_stream(handle)>>>(
      neighbors.data_handle(), total, global_offset);
}

template <class T, class IdxT>
::cuvs::neighbors::cagra::index<T, IdxT>
merge_pair(raft::resources const &handle, const cagra::index_params &params,
           std::vector<cuvs::neighbors::cagra::index<T, IdxT> *> &indices) {
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type = typename cagra_index_t::dataset_index_type;

  RAFT_EXPECTS(
      indices.size() == 2,
      "binary cross-query baseline requires exactly two inputs per merge");
  RAFT_EXPECTS(
      !params.compression.has_value(),
      "binary cross-query baseline does not support compressed output");
  RAFT_EXPECTS(params.graph_degree > 0,
               "binary cross-query baseline requires graph_degree > 0");
  RAFT_EXPECTS(
      params.metric == cuvs::distance::DistanceType::L2Expanded ||
          params.metric == cuvs::distance::DistanceType::InnerProduct ||
          params.metric == cuvs::distance::DistanceType::CosineExpanded,
      "binary cross-query baseline supports L2, inner product, and cosine");

  auto *index_a = indices[0];
  auto *index_b = indices[1];

  RAFT_EXPECTS(index_a != nullptr && index_b != nullptr,
               "Null pointer detected in 'indices'. Ensure all elements are "
               "valid before usage.");

  auto *strided_a =
      dynamic_cast<const strided_dataset<T, ds_idx_type> *>(&index_a->data());
  auto *strided_b =
      dynamic_cast<const strided_dataset<T, ds_idx_type> *>(&index_b->data());
  RAFT_EXPECTS(
      strided_a != nullptr && strided_b != nullptr,
      "binary cross-query baseline requires attached, uncompressed datasets");
  RAFT_EXPECTS(index_a->size() > 0 && index_b->size() > 0,
               "binary cross-query baseline requires non-empty inputs");
  RAFT_EXPECTS(index_a->dim() == index_b->dim(),
               "binary cross-query baseline requires matching dimensions");
  RAFT_EXPECTS(index_a->graph_degree() == params.graph_degree &&
                   index_b->graph_degree() == params.graph_degree,
               "binary cross-query baseline requires input degree to match "
               "output degree");

  auto dim = int64_t(index_a->dim());
  auto n_a = int64_t(index_a->size());
  auto n_b = int64_t(index_b->size());
  auto n_total = n_a + n_b;
  auto degree = uint32_t(params.graph_degree);

  auto merged_dataset =
      raft::make_device_matrix<T, int64_t>(handle, n_total, dim);
  {
    auto total = n_total * dim;
    auto threads = 256;
    auto blocks = (total + threads - 1) / threads;
    fill_merged_dataset_kernel<<<blocks, threads, 0,
                                 raft::resource::get_cuda_stream(handle)>>>(
        merged_dataset.data_handle(), strided_a->view().data_handle(),
        strided_b->view().data_handle(), n_a, n_b, dim,
        int64_t(strided_a->stride()), int64_t(strided_b->stride()));
  }
  auto merged_dataset_view =
      raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
          merged_dataset.data_handle(), n_total, dim);
  auto queries_a =
      raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
          merged_dataset.data_handle(), n_a, dim);
  auto queries_b =
      raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
          merged_dataset.data_handle() + n_a * dim, n_b, dim);
  raft::resource::sync_stream(handle);

  auto k_cross_light = std::min<uint32_t>(k_cross_interior, degree);
  auto k_cross_full = degree;
  auto overdegree = int64_t(k_overdegree_mult * degree);
  auto sentinel = std::numeric_limits<uint32_t>::max();
  auto inf = std::numeric_limits<float>::infinity();

  // Step 1: run a cheap cross-search for every node. This is only used to
  // estimate how close each node is to the seam before deciding which rows
  // deserve a full search.
  auto light_nbrs_a = raft::make_device_matrix<uint32_t, int64_t>(
      handle, n_a, int64_t(k_cross_light));
  auto light_nbrs_b = raft::make_device_matrix<uint32_t, int64_t>(
      handle, n_b, int64_t(k_cross_light));
  auto light_dists_a = raft::make_device_matrix<float, int64_t>(
      handle, n_a, int64_t(k_cross_light));
  auto light_dists_b = raft::make_device_matrix<float, int64_t>(
      handle, n_b, int64_t(k_cross_light));

  run_cross_search_to_device(handle, *index_b, queries_a, k_cross_light,
                             uint32_t(n_a), light_nbrs_a.view(),
                             light_dists_a.view());
  run_cross_search_to_device(handle, *index_a, queries_b, k_cross_light, 0,
                             light_nbrs_b.view(), light_dists_b.view());

  auto within_all = raft::make_device_matrix<uint32_t, int64_t>(
      handle, n_total, int64_t(degree));
  auto within_dists = raft::make_device_matrix<float, int64_t>(handle, n_total,
                                                               int64_t(degree));

  // Step 2: compute exact distances for the existing within-segment graph
  // edges. The boundary score compares the nearest cross-segment distance
  // against the farthest current within-segment edge for each node.
  {
    auto total = n_total * int64_t(degree);
    auto threads = 256;
    auto blocks = (total + threads - 1) / threads;
    fill_within_neighbors_kernel<<<blocks, threads, 0,
                                   raft::resource::get_cuda_stream(handle)>>>(
        within_all.data_handle(), index_a->graph().data_handle(),
        index_b->graph().data_handle(), n_a, n_b, degree);
    compute_within_distances_kernel<<<
        blocks, threads, 0, raft::resource::get_cuda_stream(handle)>>>(
        merged_dataset.data_handle(), within_all.data_handle(),
        within_dists.data_handle(), n_total, dim, degree, params.metric);
  }

  auto ratio = raft::make_device_vector<float, int64_t>(handle, n_total);
  {
    auto threads = 256;
    auto blocks = (n_total + threads - 1) / threads;
    compute_ratio_kernel<<<blocks, threads, 0,
                           raft::resource::get_cuda_stream(handle)>>>(
        within_dists.data_handle(), light_dists_a.data_handle(),
        light_dists_b.data_handle(), ratio.data_handle(), n_a, n_total, degree,
        k_cross_light);
  }
  raft::resource::sync_stream(handle);

  auto boundary_target =
      std::max<int64_t>(1, static_cast<int64_t>(k_boundary_fraction * n_total));
  boundary_target = std::min(boundary_target, n_total - 1);

  // Step 3: classify the lowest-ratio rows as seam/boundary nodes. Those rows
  // get a full cross-search budget; interior rows keep only the cheap cross
  // candidates.
  auto boundary_order =
      raft::make_device_vector<int64_t, int64_t>(handle, n_total);
  auto stream = raft::resource::get_cuda_stream(handle);
  auto exec = thrust::cuda::par.on(stream);
  auto ratio_ptr = thrust::device_pointer_cast(ratio.data_handle());
  auto order_ptr = thrust::device_pointer_cast(boundary_order.data_handle());
  thrust::sequence(exec, order_ptr, order_ptr + n_total, int64_t{0});
  thrust::sort_by_key(exec, ratio_ptr, ratio_ptr + n_total, order_ptr);

  auto boundary_rows =
      raft::make_device_vector<int64_t, int64_t>(handle, boundary_target);
  thrust::copy_n(exec, order_ptr, boundary_target,
                 thrust::device_pointer_cast(boundary_rows.data_handle()));

  auto boundary_rows_ptr =
      thrust::device_pointer_cast(boundary_rows.data_handle());
  auto count_a = static_cast<int64_t>(thrust::count_if(
      exec, boundary_rows_ptr, boundary_rows_ptr + boundary_target,
      [n_a] __device__(int64_t row) { return row < n_a; }));
  auto count_b = boundary_target - count_a;

  auto boundary_rows_a =
      raft::make_device_vector<int64_t, int64_t>(handle, count_a);
  auto boundary_rows_b =
      raft::make_device_vector<int64_t, int64_t>(handle, count_b);
  thrust::copy_if(exec, boundary_rows_ptr, boundary_rows_ptr + boundary_target,
                  thrust::device_pointer_cast(boundary_rows_a.data_handle()),
                  [n_a] __device__(int64_t row) { return row < n_a; });
  thrust::copy_if(exec, boundary_rows_ptr, boundary_rows_ptr + boundary_target,
                  thrust::device_pointer_cast(boundary_rows_b.data_handle()),
                  [n_a] __device__(int64_t row) { return row >= n_a; });

  auto cross_nbrs = raft::make_device_matrix<uint32_t, int64_t>(
      handle, n_total, int64_t(k_cross_full));
  auto cross_dists = raft::make_device_matrix<float, int64_t>(
      handle, n_total, int64_t(k_cross_full));
  raft::matrix::fill(
      handle,
      raft::make_device_vector_view<uint32_t, int64_t>(
          cross_nbrs.data_handle(), static_cast<int64_t>(cross_nbrs.size())),
      sentinel);
  raft::matrix::fill(
      handle,
      raft::make_device_vector_view<float, int64_t>(
          cross_dists.data_handle(), static_cast<int64_t>(cross_dists.size())),
      inf);

  // Seed every row with cross-segment candidates. Interior rows reuse the light
  // search results, while boundary rows are overwritten below with a
  // full-degree search.
  {
    auto total = n_total * int64_t(k_cross_light);
    auto threads = 256;
    auto blocks = (total + threads - 1) / threads;
    seed_cross_from_light_kernel<<<blocks, threads, 0, stream>>>(
        cross_nbrs.data_handle(), cross_dists.data_handle(),
        light_nbrs_a.data_handle(), light_nbrs_b.data_handle(),
        light_dists_a.data_handle(), light_dists_b.data_handle(), n_a, n_b,
        k_cross_light, k_cross_full);
  }

  if (count_a > 0) {
    auto boundary_queries_a =
        gather_rows(handle, merged_dataset_view,
                    raft::make_device_vector_view<const int64_t, int64_t>(
                        boundary_rows_a.data_handle(), count_a));

    auto boundary_nbrs_a = raft::make_device_matrix<uint32_t, int64_t>(
        handle, count_a, int64_t(k_cross_full));
    auto boundary_dists_a = raft::make_device_matrix<float, int64_t>(
        handle, count_a, int64_t(k_cross_full));
    run_cross_search_to_device(
        handle, *index_b,
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
            boundary_queries_a.view().data_handle(),
            boundary_queries_a.view().extent(0),
            boundary_queries_a.view().extent(1)),
        k_cross_full, uint32_t(n_a), boundary_nbrs_a.view(),
        boundary_dists_a.view());
    auto total = count_a * int64_t(k_cross_full);
    auto threads = 256;
    auto blocks = (total + threads - 1) / threads;
    scatter_boundary_results_kernel<<<blocks, threads, 0, stream>>>(
        boundary_rows_a.data_handle(), count_a, boundary_nbrs_a.data_handle(),
        boundary_dists_a.data_handle(), cross_nbrs.data_handle(),
        cross_dists.data_handle(), k_cross_full);
  }

  if (count_b > 0) {
    auto boundary_queries_b =
        gather_rows(handle, merged_dataset_view,
                    raft::make_device_vector_view<const int64_t, int64_t>(
                        boundary_rows_b.data_handle(), count_b));
    auto boundary_nbrs_b = raft::make_device_matrix<uint32_t, int64_t>(
        handle, count_b, int64_t(k_cross_full));
    auto boundary_dists_b = raft::make_device_matrix<float, int64_t>(
        handle, count_b, int64_t(k_cross_full));
    run_cross_search_to_device(
        handle, *index_a,
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
            boundary_queries_b.view().data_handle(),
            boundary_queries_b.view().extent(0),
            boundary_queries_b.view().extent(1)),
        k_cross_full, 0, boundary_nbrs_b.view(), boundary_dists_b.view());
    auto total = count_b * int64_t(k_cross_full);
    auto threads = 256;
    auto blocks = (total + threads - 1) / threads;
    scatter_boundary_results_kernel<<<blocks, threads, 0, stream>>>(
        boundary_rows_b.data_handle(), count_b, boundary_nbrs_b.data_handle(),
        boundary_dists_b.data_handle(), cross_nbrs.data_handle(),
        cross_dists.data_handle(), k_cross_full);
  }
  raft::resource::sync_stream(handle);

  // Step 4: combine within-graph and cross-graph candidates, sort them by
  // distance, drop duplicates, and keep an overdegree pool for CAGRA's
  // topology-aware prune.
  auto cand_width = int64_t(2 * degree);
  auto flat_size = n_total * cand_width;
  auto flat_rows =
      raft::make_device_vector<int64_t, int64_t>(handle, flat_size);
  auto flat_nbrs =
      raft::make_device_vector<uint32_t, int64_t>(handle, flat_size);
  auto flat_dists = raft::make_device_vector<float, int64_t>(handle, flat_size);
  {
    auto threads = 256;
    auto blocks = (flat_size + threads - 1) / threads;
    fill_flat_candidates_kernel<<<blocks, threads, 0, stream>>>(
        flat_rows.data_handle(), flat_nbrs.data_handle(),
        flat_dists.data_handle(), within_all.data_handle(),
        within_dists.data_handle(), cross_nbrs.data_handle(),
        cross_dists.data_handle(), n_total, degree, sentinel);
  }

  auto flat_rows_ptr = thrust::device_pointer_cast(flat_rows.data_handle());
  auto flat_nbrs_ptr = thrust::device_pointer_cast(flat_nbrs.data_handle());
  auto flat_dists_ptr = thrust::device_pointer_cast(flat_dists.data_handle());
  auto flat_key_begin = thrust::make_zip_iterator(
      thrust::make_tuple(flat_rows_ptr, flat_nbrs_ptr));
  thrust::sort_by_key(exec, flat_key_begin, flat_key_begin + flat_size,
                      flat_dists_ptr);

  auto unique_rows =
      raft::make_device_vector<int64_t, int64_t>(handle, flat_size);
  auto unique_nbrs =
      raft::make_device_vector<uint32_t, int64_t>(handle, flat_size);
  auto unique_dists =
      raft::make_device_vector<float, int64_t>(handle, flat_size);
  auto unique_rows_ptr = thrust::device_pointer_cast(unique_rows.data_handle());
  auto unique_nbrs_ptr = thrust::device_pointer_cast(unique_nbrs.data_handle());
  auto unique_dists_ptr =
      thrust::device_pointer_cast(unique_dists.data_handle());
  auto unique_key_begin = thrust::make_zip_iterator(
      thrust::make_tuple(unique_rows_ptr, unique_nbrs_ptr));
  auto reduce_end = thrust::reduce_by_key(
      exec, flat_key_begin, flat_key_begin + flat_size, flat_dists_ptr,
      unique_key_begin, unique_dists_ptr,
      [] __device__(const auto &lhs, const auto &rhs) {
        return thrust::get<0>(lhs) == thrust::get<0>(rhs) &&
               thrust::get<1>(lhs) == thrust::get<1>(rhs);
      },
      thrust::minimum<float>());
  auto unique_count = reduce_end.first - unique_key_begin;

  auto sort_key_begin = thrust::make_zip_iterator(
      thrust::make_tuple(unique_rows_ptr, unique_dists_ptr));
  thrust::sort_by_key(exec, sort_key_begin, sort_key_begin + unique_count,
                      unique_nbrs_ptr);

  auto row_offsets =
      raft::make_device_vector<int64_t, int64_t>(handle, n_total + 1);
  thrust::lower_bound(exec, unique_rows_ptr, unique_rows_ptr + unique_count,
                      thrust::counting_iterator<int64_t>(0),
                      thrust::counting_iterator<int64_t>(n_total + 1),
                      thrust::device_pointer_cast(row_offsets.data_handle()));

  auto over_graph_dev =
      raft::make_device_matrix<uint32_t, int64_t>(handle, n_total, overdegree);
  {
    auto threads = 256;
    auto blocks = (n_total + threads - 1) / threads;
    build_overgraph_kernel<<<blocks, threads, 0, stream>>>(
        row_offsets.data_handle(), unique_nbrs.data_handle(),
        over_graph_dev.data_handle(), n_total, uint32_t(overdegree), sentinel);
  }
  raft::resource::sync_stream(handle);

  auto over_graph =
      raft::make_host_matrix<uint32_t, int64_t>(n_total, overdegree);
  raft::copy(handle, over_graph.view(), over_graph_dev.view());
  raft::resource::sync_stream(handle);

  auto optimized_graph =
      raft::make_host_matrix<uint32_t, int64_t>(n_total, int64_t(degree));
  cagra::helpers::optimize(handle, over_graph.view(), optimized_graph.view());

  // Step 5: package the optimized graph as a searchable index, attaching the
  // merged dataset only if the caller wants the output index to be immediately
  // searchable.
  cagra_index_t merged_index(handle, params.metric);
  merged_index.update_graph(
      handle,
      raft::make_host_matrix_view<const uint32_t, int64_t, raft::row_major>(
          optimized_graph.data_handle(), optimized_graph.extent(0),
          optimized_graph.extent(1)));
  if (params.attach_dataset_on_build) {
    using matrix_t = decltype(merged_dataset);
    using layout_t = typename matrix_t::layout_type;
    using container_policy_t = typename matrix_t::container_policy_type;
    using owning_t = owning_dataset<T, int64_t, layout_t, container_policy_t>;
    auto out_layout = raft::make_strided_layout(
        merged_dataset.view().extents(), cuda::std::array<int64_t, 2>{dim, 1});
    merged_index.update_dataset(
        handle, owning_t{std::move(merged_dataset), out_layout});
  }

  return merged_index;
}

/**
 * Run the historical 2-ary cross-query merge as a balanced tournament.
 *
 * Each level owns only its inputs and outputs. Replacing current_level at the
 * end of a level releases the previous intermediate indexes before the next
 * level starts, instead of retaining the whole tournament.
 */
template <class T, class IdxT>
::cuvs::neighbors::cagra::index<T, IdxT>
merge_tree(raft::resources const &handle, cagra::index_params const &params,
           std::vector<cuvs::neighbors::cagra::index<T, IdxT>> current_level) {
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  RAFT_EXPECTS(current_level.size() >= 2,
               "binary cross-query tournament requires at least two inputs");

  while (current_level.size() > 1) {
    std::vector<cagra_index_t> next_level;
    next_level.reserve((current_level.size() + 1) / 2);
    for (size_t i = 0; i + 1 < current_level.size(); i += 2) {
      std::vector<cagra_index_t *> pair{&current_level[i],
                                        &current_level[i + 1]};
      next_level.push_back(merge_pair(handle, params, pair));
    }
    if (current_level.size() % 2 != 0) {
      next_level.push_back(std::move(current_level.back()));
    }
    current_level = std::move(next_level);
  }

  return std::move(current_level.front());
}

} // namespace cuvs::neighbors::cagra::detail::binary_cross_query

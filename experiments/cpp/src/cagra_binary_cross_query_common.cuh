/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail::binary_cross_query {

// Constants used by the native balanced-compaction implementation in the
// earlier merge experiments.
constexpr uint32_t k_cross_interior = 8;
constexpr float k_boundary_fraction = 0.5f;
constexpr uint32_t k_overdegree_mult = 2;

inline auto search_itopk_size(uint32_t k) -> size_t {
  return std::max<size_t>(64, ((static_cast<size_t>(k) + 31) / 32) * 32 + 32);
}

template <class DatasetView>
auto gather_rows(raft::resources const &handle, DatasetView const &dataset,
                 raft::device_vector_view<const int64_t, int64_t> row_ids) {
  using T = std::remove_const_t<typename DatasetView::value_type>;
  auto gathered = raft::make_device_matrix<T, int64_t>(
      handle, row_ids.extent(0), dataset.extent(1));
  if (row_ids.extent(0) == 0) {
    return gathered;
  }

  raft::matrix::copy_rows(handle, raft::make_const_mdspan(dataset),
                          gathered.view(), row_ids);
  raft::resource::sync_stream(handle);
  return gathered;
}

} // namespace cuvs::neighbors::cagra::detail::binary_cross_query

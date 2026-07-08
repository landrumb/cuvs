/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cagra_merge_scaffold.cuh"
#include "cagra_merge_vmm.cuh"
#include "graph_core.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

/**
 * A strided dataset view that shares ownership of a contiguous device matrix.
 *
 * Fastener uses this to keep incrementally migrated inputs attached to slices of the merged VMM
 * dataset while the merge is in progress. This keeps the inputs valid for exception handling
 * without retaining duplicate device allocations.
 */
template <typename T>
class shared_contiguous_dataset final : public strided_dataset<T, int64_t> {
 public:
  using storage_type = contiguous_dataset_storage<T>;
  using view_type    = typename strided_dataset<T, int64_t>::view_type;

  shared_contiguous_dataset(std::shared_ptr<storage_type> storage,
                            int64_t row_offset,
                            int64_t n_rows,
                            int64_t dim) noexcept
    : storage_(std::move(storage)), row_offset_(row_offset), n_rows_(n_rows), dim_(dim)
  {
  }

  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }

  [[nodiscard]] auto view() const noexcept -> view_type final
  {
    auto* data = storage_->data_handle();
    return raft::make_device_strided_matrix_view<const T, int64_t>(
      data == nullptr ? nullptr : data + row_offset_ * dim_, n_rows_, dim_, dim_);
  }

 private:
  std::shared_ptr<storage_type> storage_;
  int64_t row_offset_;
  int64_t n_rows_;
  int64_t dim_;
};

template <typename T, typename IdxT>
void copy_input_datasets(raft::resources const& handle,
                         std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
                         std::vector<int64_t> const& offsets,
                         int64_t dim,
                         T* destination)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  for (std::size_t i = 0; i < indices.size(); ++i) {
    auto const* source_dataset =
      dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&indices[i]->data());
    auto source = source_dataset->view();
    raft::copy_matrix(destination + offsets[i] * dim,
                      static_cast<std::size_t>(dim),
                      source.data_handle(),
                      static_cast<std::size_t>(source_dataset->stride()),
                      static_cast<std::size_t>(dim),
                      static_cast<std::size_t>(source.extent(0)),
                      raft::resource::get_cuda_stream(handle));
  }
}

/**
 * Consolidate input datasets using the legacy direct device-to-device copy.
 *
 * This allocates the final contiguous matrix while all source allocations remain live, so owning
 * inputs temporarily occupy two full logical dataset copies. It is the compatibility fallback for
 * configurations where incremental CUDA VMM consolidation is unavailable.
 */
template <typename T, typename IdxT>
auto consolidate_datasets_via_device_copy(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
  std::vector<int64_t> const& offsets,
  int64_t dim) -> std::shared_ptr<contiguous_dataset_storage<T>>
{
  auto storage = std::make_shared<device_matrix_dataset_storage<T>>(handle, offsets.back(), dim);
  copy_input_datasets(handle, indices, offsets, dim, storage->data_handle());
  raft::resource::sync_stream(handle);
  return storage;
}

/**
 * Incrementally populate one contiguous CUDA virtual address range.
 *
 * Each iteration commits only the physical pages needed for the next input, performs a
 * device-to-device copy into the final logical offset, synchronizes that copy, and replaces the
 * input's owning dataset with a shared view. Destroying the old dataset returns its physical pages
 * before the next mapping is created, so peak temporary device storage is one input plus VMM
 * granularity rather than a second full dataset.
 */
template <typename T, typename IdxT>
auto consolidate_owned_datasets_via_vmm(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
  std::vector<int64_t> const& offsets,
  int64_t dim) -> std::shared_ptr<contiguous_dataset_storage<T>>
{
  using cagra_index_t       = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type         = typename cagra_index_t::dataset_index_type;
  using storage_type        = vmm_dataset_storage<T>;
  using shared_dataset_type = shared_contiguous_dataset<T>;

  auto storage =
    std::make_shared<storage_type>(offsets.back(), dim, static_cast<std::size_t>(indices.size()));
  std::vector<std::unique_ptr<shared_dataset_type>> shared_input_datasets;
  shared_input_datasets.reserve(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    shared_input_datasets.emplace_back(
      std::make_unique<shared_dataset_type>(storage, offsets[i], offsets[i + 1] - offsets[i], dim));
  }

  for (std::size_t i = 0; i < indices.size(); ++i) {
    auto const* source_dataset =
      dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&indices[i]->data());
    auto source = source_dataset->view();

    storage->map_until(checked_dataset_bytes<T>(offsets[i + 1], dim));
    raft::copy_matrix(storage->data_handle() + offsets[i] * dim,
                      static_cast<std::size_t>(dim),
                      source.data_handle(),
                      static_cast<std::size_t>(source_dataset->stride()),
                      static_cast<std::size_t>(dim),
                      static_cast<std::size_t>(source.extent(0)),
                      raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);

    // Assignment destroys the old owning dataset only after its D2D copy is complete. The new
    // shared slice keeps all mappings already populated by earlier iterations alive.
    indices[i]->update_dataset(handle, std::move(shared_input_datasets[i]));
  }

  RAFT_EXPECTS(storage->fully_mapped(), "CUDA VMM dataset was not fully mapped");
  return storage;
}

template <typename T, typename IdxT>
auto consolidate_owned_datasets(raft::resources const& handle,
                                std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                                std::vector<int64_t> const& offsets,
                                int64_t dim) -> std::shared_ptr<contiguous_dataset_storage<T>>
{
  bool const can_use_vmm = offsets.back() > 0 && dim > 0 && cuda_vmm_supported() &&
                           current_device_resource_releases_to_cuda();
  if (can_use_vmm) {
    try {
      auto storage = consolidate_owned_datasets_via_vmm(handle, indices, offsets, dim);
      RAFT_LOG_DEBUG(
        "cagra::merge: consolidated owning datasets with incremental CUDA VMM mappings");
      return storage;
    } catch (cuda_vmm_error const& error) {
      if (error.result() != CUDA_ERROR_OUT_OF_MEMORY &&
          error.result() != CUDA_ERROR_NOT_SUPPORTED) {
        throw;
      }
      // Inputs migrated before a failed mapping remain valid shared VMM slices. The direct-copy
      // fallback can concatenate those slices together with any untouched inputs.
      RAFT_LOG_WARN(
        "cagra::merge: CUDA VMM consolidation unavailable (%s); using direct device copy",
        error.what());
    }
  } else {
    RAFT_LOG_DEBUG(
      "cagra::merge: CUDA VMM disabled by device support or the active RMM resource; using direct "
      "device copy");
  }
  return consolidate_datasets_via_device_copy(handle, indices, offsets, dim);
}

template <typename T, typename IdxT>
void release_input_datasets(raft::resources const& handle,
                            std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                            uint32_t dim)
{
  using ds_idx_type = typename cuvs::neighbors::cagra::index<T, IdxT>::dataset_index_type;

  // Complete every potentially throwing host allocation before modifying an input index.
  std::vector<std::unique_ptr<cuvs::neighbors::empty_dataset<ds_idx_type>>> empty_datasets;
  empty_datasets.reserve(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    empty_datasets.emplace_back(std::make_unique<cuvs::neighbors::empty_dataset<ds_idx_type>>(dim));
  }
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i]->update_dataset(handle, std::move(empty_datasets[i]));
  }
  raft::resource::sync_stream(handle);
}

template <bool ApplyRowFilter = true, class T, class IdxT>
index<T, IdxT> merge_rebuild(raft::resources const& handle,
                             const cagra::index_params& params,
                             std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                             const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  int64_t stride               = -1;

  if constexpr (ApplyRowFilter) {
    RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
                 "Bitmap filter isn't supported inside cagra::merge");
  }

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
        strided_dset != nullptr) {
      if (dim == 0) {
        dim    = index->dim();
        stride = strided_dset->stride();
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (dynamic_cast<const cuvs::neighbors::empty_dataset<int64_t>*>(&index->data()) !=
               nullptr) {
      RAFT_FAIL(
        "cagra::merge only supports an index to which the dataset is attached. Please check if the "
        "index was built with index_param.attach_dataset_on_build = true, or if a dataset was "
        "attached after the build.");
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dataset index");
    }
  }

  IdxT offset = 0;

  auto merge_dataset = [&](T* dst) {
    for (cagra_index_t* index : indices) {
      auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
      raft::copy_matrix(dst + offset * dim,
                        dim,
                        strided_dset->view().data_handle(),
                        static_cast<size_t>(stride),
                        dim,
                        static_cast<size_t>(strided_dset->n_rows()),
                        raft::resource::get_cuda_stream(handle));

      offset += IdxT(index->data().n_rows());
    }
  };

  try {
    auto updated_dataset =
      raft::make_device_matrix<T, int64_t>(handle, int64_t(new_dataset_size), int64_t(dim));

    merge_dataset(updated_dataset.data_handle());

    if constexpr (ApplyRowFilter) {
      if (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset) {
        auto actual_filter =
          dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
            row_filter);
        auto filtered_row_count = actual_filter.view().count(handle);

        // Convert the filter to a CSR matrix (so that we can pass indices to raft::copy_rows)
        auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
          handle, 1, new_dataset_size);
        indices_csr.initialize_sparsity(filtered_row_count);

        actual_filter.view().to_csr(handle, indices_csr);

        // Get the indices array from the csr matrix. Note that this returns a raft::span object
        // and we need to pass as device_vector_view, which is a 1D mdspan (instead of a span)
        // so we need to translate here (and adjust to be const)
        auto indices      = indices_csr.structure_view().get_indices();
        auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
          indices.data(), static_cast<int64_t>(indices.size()));

        auto filtered_dataset =
          raft::make_device_matrix<T, int64_t>(handle, filtered_row_count, dim);
        raft::matrix::copy_rows(handle,
                                raft::make_const_mdspan(updated_dataset.view()),
                                filtered_dataset.view(),
                                indices_view);

        auto merged_index =
          cagra::build(handle, params, raft::make_const_mdspan(filtered_dataset.view()));
        if (!merged_index.data().is_owning() && params.attach_dataset_on_build) {
          using matrix_t           = decltype(updated_dataset);
          using layout_t           = typename matrix_t::layout_type;
          using container_policy_t = typename matrix_t::container_policy_type;
          using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
          auto out_layout          = raft::make_strided_layout(filtered_dataset.view().extents(),
                                                      cuda::std::array<int64_t, 2>{stride, 1});

          merged_index.update_dataset(handle, owning_t{std::move(filtered_dataset), out_layout});
        }
        RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
        return merged_index;
      }
    }

    auto merged_index =
      cagra::build(handle, params, raft::make_const_mdspan(updated_dataset.view()));
    if (!merged_index.data().is_owning() && params.attach_dataset_on_build) {
      using matrix_t           = decltype(updated_dataset);
      using layout_t           = typename matrix_t::layout_type;
      using container_policy_t = typename matrix_t::container_policy_type;
      using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
      auto out_layout          = raft::make_strided_layout(updated_dataset.view().extents(),
                                                  cuda::std::array<int64_t, 2>{stride, 1});

      merged_index.update_dataset(handle, owning_t{std::move(updated_dataset), out_layout});
    }
    RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
    return merged_index;
  } catch (std::bad_alloc& e) {
    // We don't currently support the cpu memory fallback with filtered merge, since the
    // 'raft::matrix::copy_rows' only supports gpu memory
    if constexpr (ApplyRowFilter) {
      RAFT_EXPECTS(row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None,
                   "Filtered merge isn't available on cpu memory");
    }

    RAFT_LOG_DEBUG("cagra::merge: using host memory for merged dataset");

    auto updated_dataset =
      raft::make_host_matrix<T, std::int64_t>(std::int64_t(new_dataset_size), std::int64_t(dim));

    merge_dataset(updated_dataset.data_handle());

    auto merged_index =
      cagra::build(handle, params, raft::make_const_mdspan(updated_dataset.view()));
    if (!merged_index.data().is_owning() && params.attach_dataset_on_build) {
      using matrix_t           = decltype(updated_dataset);
      using layout_t           = typename matrix_t::layout_type;
      using container_policy_t = typename matrix_t::container_policy_type;
      using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
      auto out_layout          = raft::make_strided_layout(updated_dataset.view().extents(),
                                                  cuda::std::array<int64_t, 2>{stride, 1});
      merged_index.update_dataset(handle, owning_t{std::move(updated_dataset), out_layout});
    }
    return merged_index;
  }
}

template <class T, class IdxT>
index<T, IdxT> merge_with_k4_scaffold(raft::resources const& handle,
                                      const cagra::index_params& params,
                                      std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                                      merge_scaffold::build_params const& scaffold_params = {})
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  std::vector<int64_t> offsets;
  offsets.reserve(indices.size() + 1);
  offsets.push_back(0);
  bool all_datasets_owning = true;

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    auto const* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
    if (strided_dset == nullptr) {
      if (dynamic_cast<const cuvs::neighbors::empty_dataset<int64_t>*>(&index->data()) != nullptr) {
        RAFT_FAIL(
          "cagra::merge only supports an index to which the dataset is attached. Please check if "
          "the index was built with index_param.attach_dataset_on_build = true, or if a dataset "
          "was attached after the build.");
      }
      RAFT_FAIL("cagra::merge only supports an uncompressed dataset index");
    }
    if (dim == 0) {
      dim = index->dim();
    } else {
      RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
    }
    new_dataset_size += index->size();
    all_datasets_owning = all_datasets_owning && index->data().is_owning();
    offsets.push_back(static_cast<int64_t>(new_dataset_size));
  }

  std::shared_ptr<contiguous_dataset_storage<T>> updated_dataset;
  if (all_datasets_owning) {
    updated_dataset =
      consolidate_owned_datasets(handle, indices, offsets, static_cast<int64_t>(dim));
  } else {
    // A non-owning input allocation cannot be released by merge. Retain the legacy direct-copy
    // behavior for this case because the caller controls the source allocation's lifetime.
    updated_dataset =
      consolidate_datasets_via_device_copy(handle, indices, offsets, static_cast<int64_t>(dim));
  }

  auto stream          = raft::resource::get_cuda_stream(handle);
  bool measure_quality = scaffold_params.quality_stats_output != nullptr;
  rmm::device_uvector<uint8_t> scaffold_degrees(measure_quality ? new_dataset_size : 0, stream);
  auto scaffold     = merge_scaffold::build<T>(handle,
                                           raft::make_const_mdspan(updated_dataset->view()),
                                           offsets,
                                           scaffold_params,
                                           measure_quality ? scaffold_degrees.data() : nullptr);
  auto merged_graph = merge_scaffold::append_to_input_graphs<T, IdxT>(
    handle, indices, offsets, raft::make_const_mdspan(scaffold.view()));

  RAFT_EXPECTS(static_cast<int64_t>(params.graph_degree) <= merged_graph.extent(1),
               "Requested output graph degree exceeds input graph degree plus the scaffold");
  cagra::detail::graph::sort_knn_graph_device_inplace(
    handle, params.metric, raft::make_const_mdspan(updated_dataset->view()), merged_graph.view());

  if (measure_quality) {
    auto quality_start = std::chrono::steady_clock::now();
    merge_scaffold::measure_preopt_quality(handle,
                                           raft::make_const_mdspan(scaffold.view()),
                                           scaffold_degrees.data(),
                                           raft::make_const_mdspan(merged_graph.view()),
                                           params.graph_degree,
                                           scaffold_params.quality_sample_rows,
                                           *scaffold_params.quality_stats_output);
    auto quality_end = std::chrono::steady_clock::now();
    scaffold_params.quality_stats_output->measurement_ms =
      std::chrono::duration<double, std::milli>(quality_end - quality_start).count();
  }
  int64_t preopt_graph_degree_cap = scaffold_params.preopt_graph_degree_cap;
  if (preopt_graph_degree_cap == merge_scaffold::k_cap_to_output_graph_degree) {
    preopt_graph_degree_cap = params.graph_degree;
  }
  if (preopt_graph_degree_cap > 0) {
    RAFT_EXPECTS(preopt_graph_degree_cap >= static_cast<int64_t>(params.graph_degree) &&
                   preopt_graph_degree_cap <= merged_graph.extent(1),
                 "Pre-optimize graph degree cap must be between output and candidate degree");
    if (preopt_graph_degree_cap < merged_graph.extent(1)) {
      merged_graph = merge_scaffold::cap_sorted_graph(
        handle, raft::make_const_mdspan(merged_graph.view()), preopt_graph_degree_cap);
    }
  }

  auto optimized_graph = raft::make_device_matrix<uint32_t, int64_t>(
    handle, int64_t(new_dataset_size), int64_t(params.graph_degree));
  cagra::detail::graph::optimize(
    handle, merged_graph.view(), optimized_graph.view(), params.guarantee_connectivity);

  index<T, IdxT> merged_index(handle, params.metric);
  merged_index.update_graph(handle, std::move(optimized_graph));
  if (params.attach_dataset_on_build) {
    merged_index.update_dataset(handle,
                                shared_contiguous_dataset<T>{std::move(updated_dataset),
                                                             0,
                                                             static_cast<int64_t>(new_dataset_size),
                                                             static_cast<int64_t>(dim)});
  }

  // A successful Fastener merge consumes owning attached datasets. Leaving the inputs graph-only
  // ensures they cannot keep an earlier combined allocation alive during a later merge tree.
  if (all_datasets_owning) { release_input_datasets(handle, indices, static_cast<uint32_t>(dim)); }
  return merged_index;
}

template <class T, class IdxT>
index<T, IdxT> merge(raft::resources const& handle,
                     const cagra::index_params& params,
                     std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                     const cuvs::neighbors::filtering::base_filter& row_filter)
{
  bool l2_metric              = params.metric == cuvs::distance::DistanceType::L2Expanded;
  bool graph_degree_supported = false;
  if (!indices.empty()) {
    std::size_t max_input_degree = 0;
    for (auto const* index : indices) {
      if (index != nullptr) {
        max_input_degree = std::max<std::size_t>(max_input_degree, index->graph_degree());
      }
    }
    graph_degree_supported =
      params.graph_degree > 0 &&
      params.graph_degree <=
        max_input_degree + merge_scaffold::k_degree * merge_scaffold::k_default_repeats;
  }

  bool use_scaffold =
    row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None &&
    indices.size() >= 2 && l2_metric && !params.compression.has_value() && graph_degree_supported;
  if (!use_scaffold) { return merge_rebuild(handle, params, indices, row_filter); }

  return merge_with_k4_scaffold(handle, params, indices);
}

}  // namespace cuvs::neighbors::cagra::detail

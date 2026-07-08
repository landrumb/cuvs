/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/cagra/cagra_merge.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

using index_type = index<float, uint32_t>;

class current_device_resource_guard {
 public:
  template <typename Resource>
  explicit current_device_resource_guard(Resource resource)
    : original_(rmm::mr::set_current_device_resource(std::move(resource)))
  {
  }

  current_device_resource_guard(current_device_resource_guard const&)                    = delete;
  auto operator=(current_device_resource_guard const&) -> current_device_resource_guard& = delete;

  ~current_device_resource_guard() noexcept
  {
    rmm::mr::set_current_device_resource(std::move(original_));
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> original_;
};

auto make_dataset(raft::resources const& res, int64_t rows, int64_t dim, int64_t row_offset)
  -> raft::host_matrix<float, int64_t>
{
  auto dataset = raft::make_host_matrix<float, int64_t>(res, rows, dim);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      dataset(i, j) = static_cast<float>((i + row_offset) * dim + j) / 128.0f;
    }
  }
  return dataset;
}

auto make_ring_graph(raft::resources const& res, int64_t rows, int64_t degree)
  -> raft::host_matrix<uint32_t, int64_t>
{
  auto graph = raft::make_host_matrix<uint32_t, int64_t>(res, rows, degree);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < degree; ++j) {
      graph(i, j) = static_cast<uint32_t>((i + j + 1) % rows);
    }
  }
  return graph;
}

auto make_index(raft::resources const& res,
                raft::host_matrix<float, int64_t> const& dataset,
                raft::host_matrix<uint32_t, int64_t> const& graph) -> index_type
{
  return index_type(res,
                    cuvs::distance::DistanceType::L2Expanded,
                    raft::make_const_mdspan(dataset.view()),
                    raft::make_const_mdspan(graph.view()));
}

void expect_dataset_eq(raft::resources const& res,
                       raft::device_matrix_view<const float, int64_t, raft::layout_stride> actual,
                       raft::host_matrix<float, int64_t> const& expected)
{
  ASSERT_EQ(actual.extent(0), expected.extent(0));
  ASSERT_EQ(actual.extent(1), expected.extent(1));

  auto actual_host =
    raft::make_host_matrix<float, int64_t>(res, actual.extent(0), actual.extent(1));
  raft::copy(actual_host.data_handle(),
             actual.data_handle(),
             actual.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (int64_t i = 0; i < actual.extent(0); ++i) {
    for (int64_t j = 0; j < actual.extent(1); ++j) {
      EXPECT_EQ(actual_host(i, j), expected(i, j));
    }
  }
}

TEST(CagraMergeDatasetMemory, ConsolidationFallsBackToDirectDeviceCopy)
{
  auto statistics =
    rmm::mr::statistics_resource_adaptor{rmm::mr::get_current_device_resource_ref()};
  current_device_resource_guard resource_guard{statistics};

  {
    raft::resources res;
    constexpr int64_t rows0  = 128;
    constexpr int64_t rows1  = 192;
    constexpr int64_t dim    = 16;
    constexpr int64_t degree = 4;

    auto dataset0 = make_dataset(res, rows0, dim, 0);
    auto dataset1 = make_dataset(res, rows1, dim, rows0);
    auto graph0   = make_ring_graph(res, rows0, degree);
    auto graph1   = make_ring_graph(res, rows1, degree);
    auto index0   = make_index(res, dataset0, graph0);
    auto index1   = make_index(res, dataset1, graph1);

    ASSERT_TRUE(index0.data().is_owning());
    ASSERT_TRUE(index1.data().is_owning());
    std::vector<index_type*> indices{&index0, &index1};
    std::vector<int64_t> offsets{0, rows0, rows0 + rows1};

    statistics.push_counters();
    auto storage              = detail::consolidate_owned_datasets(res, indices, offsets, dim);
    auto [bytes, allocations] = statistics.pop_counters();

    auto const combined_bytes = detail::checked_dataset_bytes<float>(rows0 + rows1, dim);
    EXPECT_EQ(bytes.value, combined_bytes);
    EXPECT_EQ(bytes.peak, combined_bytes);
    EXPECT_EQ(allocations.value, 1);
    EXPECT_EQ(allocations.peak, 1);
    EXPECT_FALSE(storage->is_vmm());
    EXPECT_EQ(storage->extent(0), rows0 + rows1);
    EXPECT_EQ(storage->extent(1), dim);
    EXPECT_NE(index0.dataset().data_handle(), storage->data_handle());
    EXPECT_NE(index1.dataset().data_handle(), storage->data_handle() + rows0 * dim);

    auto expected = make_dataset(res, rows0 + rows1, dim, 0);
    expect_dataset_eq(res, storage->view(), expected);
  }
}

TEST(CagraMergeDatasetMemory, VmmConsolidationUsesAtMostOneInputScratch)
{
  if (!detail::cuda_vmm_supported()) { GTEST_SKIP() << "CUDA VMM is not supported"; }

  raft::resources res;
  constexpr int64_t rows0  = 16384;
  constexpr int64_t rows1  = 32768;
  constexpr int64_t dim    = 16;
  constexpr int64_t degree = 4;

  auto dataset0 = make_dataset(res, rows0, dim, 0);
  auto dataset1 = make_dataset(res, rows1, dim, rows0);
  auto graph0   = make_ring_graph(res, rows0, degree);
  auto graph1   = make_ring_graph(res, rows1, degree);
  auto index0   = make_index(res, dataset0, graph0);
  auto index1   = make_index(res, dataset1, graph1);

  std::vector<index_type*> indices{&index0, &index1};
  std::vector<int64_t> offsets{0, rows0, rows0 + rows1};
  auto storage            = detail::consolidate_owned_datasets_via_vmm(res, indices, offsets, dim);
  auto const* vmm_storage = dynamic_cast<detail::vmm_dataset_storage<float> const*>(storage.get());

  ASSERT_NE(vmm_storage, nullptr);
  ASSERT_TRUE(storage->is_vmm());
  ASSERT_TRUE(vmm_storage->fully_mapped());
  EXPECT_EQ(storage->extent(0), rows0 + rows1);
  EXPECT_EQ(storage->extent(1), dim);
  EXPECT_EQ(index0.dataset().data_handle(), storage->data_handle());
  EXPECT_EQ(index1.dataset().data_handle(), storage->data_handle() + rows0 * dim);

  auto const bytes0      = detail::checked_dataset_bytes<float>(rows0, dim);
  auto const bytes1      = detail::checked_dataset_bytes<float>(rows1, dim);
  auto const total_bytes = bytes0 + bytes1;
  EXPECT_GE(vmm_storage->physical_bytes(), total_bytes);
  EXPECT_LT(vmm_storage->physical_bytes() - total_bytes, vmm_storage->granularity());
  EXPECT_LE(vmm_storage->max_incremental_scratch_bytes(),
            std::max(bytes0, bytes1) + vmm_storage->granularity());

  expect_dataset_eq(res, index0.dataset(), dataset0);
  expect_dataset_eq(res, index1.dataset(), dataset1);
}

TEST(CagraMergeDatasetMemory, FastenerDirectFallbackConsumesOwningInputDatasets)
{
  auto statistics =
    rmm::mr::statistics_resource_adaptor{rmm::mr::get_current_device_resource_ref()};
  current_device_resource_guard resource_guard{statistics};

  raft::resources res;
  constexpr int64_t rows0  = 256;
  constexpr int64_t rows1  = 384;
  constexpr int64_t dim    = 16;
  constexpr int64_t degree = 4;

  auto dataset0 = make_dataset(res, rows0, dim, 0);
  auto dataset1 = make_dataset(res, rows1, dim, rows0);
  auto graph0   = make_ring_graph(res, rows0, degree);
  auto graph1   = make_ring_graph(res, rows1, degree);
  auto index0   = make_index(res, dataset0, graph0);
  auto index1   = make_index(res, dataset1, graph1);

  auto expected = raft::make_host_matrix<float, int64_t>(res, rows0 + rows1, dim);
  for (int64_t i = 0; i < rows0 + rows1; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      expected(i, j) = i < rows0 ? dataset0(i, j) : dataset1(i - rows0, j);
    }
  }

  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = degree;
  params.intermediate_graph_degree = degree;
  params.attach_dataset_on_build   = true;
  params.guarantee_connectivity    = false;
  std::vector<index_type*> indices{&index0, &index1};

  auto merged = detail::merge_with_k4_scaffold(res, params, indices);
  ASSERT_EQ(merged.size(), rows0 + rows1);
  ASSERT_TRUE(merged.data().is_owning());
  expect_dataset_eq(res, merged.dataset(), expected);

  EXPECT_EQ(index0.dataset().extent(0), 0);
  EXPECT_EQ(index1.dataset().extent(0), 0);
  EXPECT_EQ(index0.dataset().extent(1), dim);
  EXPECT_EQ(index1.dataset().extent(1), dim);
  EXPECT_EQ(index0.size(), rows0);
  EXPECT_EQ(index1.size(), rows1);
  EXPECT_EQ(index0.dim(), dim);
  EXPECT_EQ(index1.dim(), dim);
}

TEST(CagraMergeDatasetMemory, VmmOutputCanBeConsumedByNextMergeLevel)
{
  if (!detail::cuda_vmm_supported() || !detail::current_device_resource_releases_to_cuda()) {
    GTEST_SKIP() << "Incremental VMM path is unavailable";
  }

  raft::resources res;
  constexpr int64_t rows0  = 256;
  constexpr int64_t rows1  = 384;
  constexpr int64_t rows2  = 512;
  constexpr int64_t dim    = 16;
  constexpr int64_t degree = 4;

  auto dataset0 = make_dataset(res, rows0, dim, 0);
  auto dataset1 = make_dataset(res, rows1, dim, rows0);
  auto dataset2 = make_dataset(res, rows2, dim, rows0 + rows1);
  auto graph0   = make_ring_graph(res, rows0, degree);
  auto graph1   = make_ring_graph(res, rows1, degree);
  auto graph2   = make_ring_graph(res, rows2, degree);
  auto index0   = make_index(res, dataset0, graph0);
  auto index1   = make_index(res, dataset1, graph1);
  auto index2   = make_index(res, dataset2, graph2);

  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = degree;
  params.intermediate_graph_degree = degree;
  params.attach_dataset_on_build   = true;
  params.guarantee_connectivity    = false;

  std::vector<index_type*> first_level{&index0, &index1};
  auto merged01 = detail::merge_with_k4_scaffold(res, params, first_level);
  ASSERT_EQ(merged01.size(), rows0 + rows1);
  ASSERT_EQ(index0.dataset().extent(0), 0);
  ASSERT_EQ(index1.dataset().extent(0), 0);

  std::vector<index_type*> second_level{&merged01, &index2};
  auto merged = detail::merge_with_k4_scaffold(res, params, second_level);
  ASSERT_EQ(merged.size(), rows0 + rows1 + rows2);
  EXPECT_EQ(merged01.dataset().extent(0), 0);
  EXPECT_EQ(index2.dataset().extent(0), 0);

  auto expected = make_dataset(res, rows0 + rows1 + rows2, dim, 0);
  expect_dataset_eq(res, merged.dataset(), expected);
}

}  // namespace
}  // namespace cuvs::neighbors::cagra

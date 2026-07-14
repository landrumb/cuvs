/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cluster/detail/kmeans_common.cuh>
#include <distance/fused_distance_nn.cuh>

#include <raft/linalg/norm.cuh>
#include <raft/matrix/init.cuh>

#include <limits>

namespace cuvs::cluster::kmeans::detail {

/**
 * Experiment-local instantiation for direct high-k balanced k-means.
 *
 * The shared cuVS library keeps this internal symbol hidden. The benchmark only
 * uses squared L2 and requires int64 indexing, so instantiate the fused path
 * without pulling every pairwise-distance implementation into the executable.
 */
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
    raft::resources const &handle,
    raft::device_matrix_view<const DataT, IndexT> X,
    raft::device_matrix_view<const DataT, IndexT> centroids,
    raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>
        min_cluster_and_distance,
    raft::device_vector_view<const DataT, IndexT> l2_norm_x,
    rmm::device_uvector<DataT> &centroid_norm_buffer,
    cuvs::distance::DistanceType metric, int, int,
    rmm::device_uvector<char> &workspace) {
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded,
               "k-means experiment helper only supports squared L2");
  auto stream = raft::resource::get_cuda_stream(handle);
  auto n_samples = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  centroid_norm_buffer.resize(n_clusters, stream);
  auto centroid_norms = raft::make_device_vector_view<DataT, IndexT>(
      centroid_norm_buffer.data(), n_clusters);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle, centroids, centroid_norms);

  raft::KeyValuePair<IndexT, DataT> initial_value(
      0, std::numeric_limits<DataT>::max());
  raft::matrix::fill(handle, min_cluster_and_distance, initial_value);
  workspace.resize(sizeof(int) * n_samples, stream);

  cuvs::distance::fusedDistanceNNMinReduce<
      DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
      min_cluster_and_distance.data_handle(), X.data_handle(),
      centroids.data_handle(), l2_norm_x.data_handle(),
      centroid_norms.data_handle(), n_samples, n_clusters, n_features,
      workspace.data(), false, false, true, metric, 0.0f, stream);
}

template void minClusterAndDistanceCompute<float, int64_t>(
    raft::resources const &, raft::device_matrix_view<const float, int64_t>,
    raft::device_matrix_view<const float, int64_t>,
    raft::device_vector_view<raft::KeyValuePair<int64_t, float>, int64_t>,
    raft::device_vector_view<const float, int64_t>,
    rmm::device_uvector<float> &, cuvs::distance::DistanceType, int, int,
    rmm::device_uvector<char> &);

} // namespace cuvs::cluster::kmeans::detail

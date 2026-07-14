/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cluster/kmeans_balanced_build_clusters_impl.cuh>
#include <cuvs/neighbors/nn_descent.hpp>
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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace scaffold = cuvs::neighbors::cagra::detail::merge_scaffold;
using clock_type = std::chrono::steady_clock;
constexpr std::size_t io_chunk_bytes = 1ull << 30;

struct options {
  std::string dataset;
  std::string sample_ids;
  std::string self_groundtruth;
  std::string output_csv;
  std::string label;
  std::vector<int> repeats{1, 2, 4, 8, 16, 32};
  std::vector<int> parts{2, 8, 128};
  std::vector<int> native_degrees{12, 32, 64};
  bool kmeans_flat_only = false;
  int kmeans_iterations = 20;
  uint64_t seed = 1234;
};

template <typename T> struct bin_matrix {
  uint32_t rows = 0;
  uint32_t dim = 0;
  std::vector<T> data;
};

std::vector<int> parse_list(std::string const &text) {
  std::vector<int> values;
  std::size_t start = 0;
  while (start < text.size()) {
    auto end = text.find(',', start);
    auto token = text.substr(start, end - start);
    if (token.empty()) {
      throw std::runtime_error("Invalid comma-separated list");
    }
    values.push_back(std::stoi(token));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return values;
}

options parse_args(int argc, char **argv) {
  options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      return std::string(argv[++i]);
    };
    if (arg == "--dataset") {
      opts.dataset = value();
    } else if (arg == "--sample-ids") {
      opts.sample_ids = value();
    } else if (arg == "--self-groundtruth") {
      opts.self_groundtruth = value();
    } else if (arg == "--output-csv") {
      opts.output_csv = value();
    } else if (arg == "--label") {
      opts.label = value();
    } else if (arg == "--repeats") {
      opts.repeats = parse_list(value());
    } else if (arg == "--parts") {
      opts.parts = parse_list(value());
    } else if (arg == "--native-degrees") {
      opts.native_degrees = parse_list(value());
    } else if (arg == "--kmeans-flat-only") {
      opts.kmeans_flat_only = true;
    } else if (arg == "--kmeans-iterations") {
      opts.kmeans_iterations = std::stoi(value());
    } else if (arg == "--seed") {
      opts.seed = std::stoull(value());
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (opts.dataset.empty() || opts.sample_ids.empty() ||
      opts.self_groundtruth.empty() || opts.output_csv.empty() ||
      opts.label.empty()) {
    throw std::runtime_error("dataset, sample IDs, self ground truth, output "
                             "CSV, and label required");
  }
  if (opts.repeats.empty() || opts.parts.empty() ||
      !std::is_sorted(opts.repeats.begin(), opts.repeats.end()) ||
      opts.repeats.front() < 1 || opts.repeats.back() > 32) {
    throw std::runtime_error("repeat checkpoints must be sorted in [1, 32]");
  }
  for (int part_count : opts.parts) {
    if (part_count < 2) {
      throw std::runtime_error("parts must be at least two");
    }
  }
  for (int degree : opts.native_degrees) {
    if (degree < 12 || degree > 128) {
      throw std::runtime_error("native degrees must be in [12, 128]");
    }
  }
  if (opts.kmeans_iterations < 1 || opts.kmeans_iterations > 100) {
    throw std::runtime_error("k-means iterations must be in [1, 100]");
  }
  return opts;
}

void read_exact(std::istream &input, char *destination, std::streamsize bytes,
                std::string const &path) {
  input.read(destination, bytes);
  if (input.gcount() != bytes) {
    throw std::runtime_error("Unexpected EOF: " + path);
  }
}

void read_large(std::istream &input, char *destination, std::size_t bytes,
                std::string const &path) {
  for (std::size_t done = 0; done < bytes;) {
    auto chunk =
        static_cast<std::streamsize>(std::min(io_chunk_bytes, bytes - done));
    read_exact(input, destination + done, chunk, path);
    done += static_cast<std::size_t>(chunk);
  }
}

template <typename T> bin_matrix<T> read_bin(std::string const &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Could not open " + path);
  }
  bin_matrix<T> matrix;
  read_exact(input, reinterpret_cast<char *>(&matrix.rows), sizeof(matrix.rows),
             path);
  read_exact(input, reinterpret_cast<char *>(&matrix.dim), sizeof(matrix.dim),
             path);
  auto count = static_cast<std::size_t>(matrix.rows) * matrix.dim;
  auto expected = 2 * sizeof(uint32_t) + count * sizeof(T);
  if (std::filesystem::file_size(path) != expected) {
    throw std::runtime_error("Unexpected payload size: " + path);
  }
  matrix.data.resize(count);
  read_large(input, reinterpret_cast<char *>(matrix.data.data()),
             count * sizeof(T), path);
  return matrix;
}

bool is_uint8_matrix(std::string const &path) {
  std::ifstream input(path, std::ios::binary);
  uint32_t rows = 0, dim = 0;
  read_exact(input, reinterpret_cast<char *>(&rows), sizeof(rows), path);
  read_exact(input, reinterpret_cast<char *>(&dim), sizeof(dim), path);
  auto payload = std::filesystem::file_size(path) - 2 * sizeof(uint32_t);
  auto count = static_cast<uint64_t>(rows) * dim;
  if (payload == count) {
    return true;
  }
  if (payload == count * sizeof(float)) {
    return false;
  }
  throw std::runtime_error("Dataset is neither uint8 nor float32");
}

double elapsed_ms(clock_type::time_point start, clock_type::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static __global__ void assign_leaf_ids_kernel(uint32_t const *sorted_ids,
                                              uint32_t const *starts,
                                              uint32_t const *ends,
                                              int64_t leaf_count,
                                              uint32_t *leaf_ids) {
  int64_t leaf = blockIdx.x;
  if (leaf >= leaf_count) {
    return;
  }
  for (uint32_t pos = starts[leaf] + threadIdx.x; pos < ends[leaf];
       pos += blockDim.x) {
    leaf_ids[sorted_ids[pos]] = static_cast<uint32_t>(leaf);
  }
}

static __global__ void gather_leaf_labels_kernel(uint32_t const *sample_ids,
                                                 int32_t const *groundtruth,
                                                 int64_t samples, int64_t k,
                                                 uint32_t const *leaf_ids,
                                                 uint32_t *output) {
  int64_t sample = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample >= samples) {
    return;
  }
  output[sample * (k + 1)] = leaf_ids[sample_ids[sample]];
  for (int64_t j = 0; j < k; ++j) {
    output[sample * (k + 1) + j + 1] = leaf_ids[groundtruth[sample * k + j]];
  }
}

static __global__ void
gather_sample_graph_kernel(uint32_t const *sample_ids, int64_t samples,
                           uint32_t const *graph, int64_t graph_stride,
                           uint8_t const *degrees, uint32_t *output_graph,
                           uint8_t *output_degrees) {
  int64_t sample = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample >= samples) {
    return;
  }
  uint32_t row = sample_ids[sample];
  output_degrees[sample] = degrees[row];
  for (int64_t j = 0; j < graph_stride; ++j) {
    output_graph[sample * graph_stride + j] =
        graph[static_cast<int64_t>(row) * graph_stride + j];
  }
}

template <typename T>
auto partition_once(raft::resources const &res,
                    raft::device_matrix_view<const T, int64_t> dataset,
                    scaffold::build_params const &params)
    -> std::pair<rmm::device_uvector<uint32_t>,
                 std::vector<scaffold::host_range>> {
  auto stream = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);
  rmm::device_uvector<uint32_t> ids(rows, stream);
  rmm::device_uvector<uint32_t> next_ids(rows, stream);
  rmm::device_uvector<uint8_t> sides(rows, stream);
  thrust::sequence(
      thrust::cuda::par.on(stream), thrust::device_pointer_cast(ids.data()),
      thrust::device_pointer_cast(ids.data() + ids.size()), uint32_t{0});
  raft::resource::sync_stream(res);

  std::vector<scaffold::host_range> ranges{{0, rows}};
  std::vector<scaffold::pivot_chunk> chunks;
  std::vector<uint32_t> chunk_left_counts;
  std::vector<scaffold::scatter_offset> scatter_offsets;
  rmm::device_uvector<scaffold::pivot_chunk> device_chunks(0, stream);
  rmm::device_uvector<uint32_t> device_left_counts(0, stream);
  rmm::device_uvector<scaffold::scatter_offset> device_scatter_offsets(0,
                                                                       stream);
  std::vector<uint3> chunk_ternary_counts;
  std::vector<scaffold::ternary_scatter_offset> ternary_scatter_offsets;
  rmm::device_uvector<uint3> device_ternary_counts(0, stream);
  rmm::device_uvector<scaffold::ternary_scatter_offset>
      device_ternary_scatter_offsets(0, stream);
  for (int level = 0; level < params.max_pivot_tree_levels; ++level) {
    int active_count =
        scaffold::make_pivot_chunks(ranges, level, params, chunks);
    if (active_count == 0) {
      break;
    }
    device_chunks.resize(chunks.size(), stream);
    raft::copy(device_chunks.data(), chunks.data(), chunks.size(), stream);
    if (params.pivot_arity == 2) {
      device_left_counts.resize(chunks.size(), stream);
      scaffold::pivot_assign_sides_kernel<<<static_cast<int>(chunks.size()),
                                            scaffold::k_pivot_block_size, 0,
                                            stream>>>(
          dataset.data_handle(), dataset.extent(1), ids.data(),
          device_chunks.data(), chunks.size(), sides.data(),
          device_left_counts.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      chunk_left_counts.resize(chunks.size());
      raft::copy(chunk_left_counts.data(), device_left_counts.data(),
                 chunks.size(), stream);
      raft::resource::sync_stream(res);
      ranges = scaffold::prepare_scatter(ranges, chunk_left_counts, chunks,
                                         scatter_offsets);
      device_scatter_offsets.resize(scatter_offsets.size(), stream);
      raft::copy(device_scatter_offsets.data(), scatter_offsets.data(),
                 scatter_offsets.size(), stream);
      scaffold::stable_scatter_kernel<<<static_cast<int>(chunks.size()), 256, 0,
                                        stream>>>(
          ids.data(), sides.data(), device_chunks.data(),
          device_scatter_offsets.data(), chunks.size(), next_ids.data());
      RAFT_CUDA_TRY(cudaGetLastError());
    } else {
      device_ternary_counts.resize(chunks.size(), stream);
      scaffold::pivot_assign_ternary_kernel<<<static_cast<int>(chunks.size()),
                                              scaffold::k_pivot_block_size, 0,
                                              stream>>>(
          dataset.data_handle(), dataset.extent(1), ids.data(),
          device_chunks.data(), chunks.size(), sides.data(),
          device_ternary_counts.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      chunk_ternary_counts.resize(chunks.size());
      raft::copy(chunk_ternary_counts.data(), device_ternary_counts.data(),
                 chunks.size(), stream);
      raft::resource::sync_stream(res);
      ranges = scaffold::prepare_ternary_scatter(
          ranges, chunk_ternary_counts, chunks, ternary_scatter_offsets);
      device_ternary_scatter_offsets.resize(ternary_scatter_offsets.size(),
                                            stream);
      raft::copy(device_ternary_scatter_offsets.data(),
                 ternary_scatter_offsets.data(), ternary_scatter_offsets.size(),
                 stream);
      scaffold::stable_ternary_scatter_kernel<<<static_cast<int>(chunks.size()),
                                                256, 0, stream>>>(
          ids.data(), sides.data(), device_chunks.data(),
          device_ternary_scatter_offsets.data(), chunks.size(),
          next_ids.data());
      RAFT_CUDA_TRY(cudaGetLastError());
    }
    std::swap(ids, next_ids);
  }

  auto leaves = scaffold::split_large_ranges(ranges);
  std::vector<uint32_t> starts_host(leaves.size()), ends_host(leaves.size());
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    starts_host[i] = static_cast<uint32_t>(leaves[i].start);
    ends_host[i] = static_cast<uint32_t>(leaves[i].end);
  }
  rmm::device_uvector<uint32_t> starts(leaves.size(), stream),
      ends(leaves.size(), stream);
  rmm::device_uvector<uint32_t> leaf_ids(rows, stream);
  raft::copy(starts.data(), starts_host.data(), starts.size(), stream);
  raft::copy(ends.data(), ends_host.data(), ends.size(), stream);
  assign_leaf_ids_kernel<<<static_cast<int>(leaves.size()), 256, 0, stream>>>(
      ids.data(), starts.data(), ends.data(), leaves.size(), leaf_ids.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  return {std::move(leaf_ids), std::move(leaves)};
}

std::vector<int64_t> make_offsets(int64_t rows, int parts) {
  std::vector<int64_t> offsets{0};
  int64_t base = rows / parts;
  int64_t rem = rows % parts;
  for (int part = 0; part < parts; ++part) {
    offsets.push_back(offsets.back() + base + (part < rem ? 1 : 0));
  }
  return offsets;
}

int origin_of(uint32_t id, std::vector<int64_t> const &offsets) {
  return static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), id) -
                          offsets.begin() - 1);
}

void write_header_if_needed(std::ofstream &output, bool exists) {
  if (!exists) {
    output << "dataset,record_type,parts,method,dtype,rows,dim,leaf_size,"
              "repeats,neighbors_per_leaf,"
              "sample_rows,k,seed,incremental_ms,cumulative_ms,leaf_count,min_"
              "leaf_size,mean_leaf_size,"
              "max_leaf_size,nn_same_partition_rate,implicit_knn_recall,"
              "implicit_cross_knn_recall,"
              "implicit_unique_degree_mean\n";
  }
}

template <typename T> int run(options const &opts) {
  auto dataset = read_bin<T>(opts.dataset);
  auto sample_matrix = read_bin<int32_t>(opts.sample_ids);
  auto truth = read_bin<int32_t>(opts.self_groundtruth);
  if (sample_matrix.dim != 1 || truth.rows != sample_matrix.rows ||
      truth.dim != 12) {
    throw std::runtime_error(
        "Expected sample IDs [n,1] and self ground truth [n,12]");
  }
  std::vector<uint32_t> sample_ids(sample_matrix.rows);
  for (std::size_t i = 0; i < sample_ids.size(); ++i) {
    if (sample_matrix.data[i] < 0 ||
        static_cast<uint32_t>(sample_matrix.data[i]) >= dataset.rows) {
      throw std::runtime_error("Sample ID out of range");
    }
    sample_ids[i] = static_cast<uint32_t>(sample_matrix.data[i]);
  }

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto device_dataset =
      raft::make_device_matrix<T, int64_t>(res, dataset.rows, dataset.dim);
  raft::copy(device_dataset.data_handle(), dataset.data.data(),
             dataset.data.size(), stream);
  rmm::device_uvector<uint32_t> device_sample_ids(sample_ids.size(), stream);
  rmm::device_uvector<int32_t> device_truth(truth.data.size(), stream);
  raft::copy(device_sample_ids.data(), sample_ids.data(), sample_ids.size(),
             stream);
  raft::copy(device_truth.data(), truth.data.data(), truth.data.size(), stream);
  raft::resource::sync_stream(res);

  bool exists = std::filesystem::exists(opts.output_csv);
  std::ofstream output(opts.output_csv, std::ios::app);
  if (!output) {
    throw std::runtime_error("Could not write output CSV");
  }
  write_header_if_needed(output, exists);
  auto dtype = std::is_same_v<T, uint8_t> ? "uint8" : "float32";
  int64_t samples = sample_ids.size();
  int64_t k = truth.dim;
  if (opts.kmeans_flat_only) {
    int64_t n_clusters =
        (static_cast<int64_t>(dataset.rows) + scaffold::k_cluster_size - 1) /
        scaffold::k_cluster_size;
    cuvs::cluster::kmeans::balanced_params params;
    params.metric = cuvs::distance::DistanceType::L2Expanded;
    params.n_iters = opts.kmeans_iterations;

    auto started = clock_type::now();
    auto centroids =
        raft::make_device_matrix<float, int64_t>(res, n_clusters, dataset.dim);
    auto labels =
        raft::make_device_vector<uint32_t, int64_t>(res, dataset.rows);
    auto cluster_sizes =
        raft::make_device_vector<uint32_t, int64_t>(res, n_clusters);
    if constexpr (std::is_same_v<T, float>) {
      cuvs::cluster::kmeans_balanced::helpers::build_clusters(
          res, params, raft::make_const_mdspan(device_dataset.view()),
          centroids.view(), labels.view(), cluster_sizes.view(),
          raft::identity_op{},
          std::optional<raft::device_vector_view<const float>>{});
    } else {
      cuvs::cluster::kmeans_balanced::helpers::build_clusters(
          res, params, raft::make_const_mdspan(device_dataset.view()),
          centroids.view(), labels.view(), cluster_sizes.view(),
          raft::cast_op<float>{},
          std::optional<raft::device_vector_view<const float>>{});
    }
    raft::resource::sync_stream(res);
    double build_ms = elapsed_ms(started, clock_type::now());

    rmm::device_uvector<uint32_t> gathered(samples * (k + 1), stream);
    int blocks = static_cast<int>((samples + 255) / 256);
    gather_leaf_labels_kernel<<<blocks, 256, 0, stream>>>(
        device_sample_ids.data(), device_truth.data(), samples, k,
        labels.data_handle(), gathered.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    std::vector<uint32_t> gathered_host(gathered.size());
    std::vector<uint32_t> cluster_sizes_host(n_clusters);
    raft::copy(gathered_host.data(), gathered.data(), gathered.size(), stream);
    raft::copy(cluster_sizes_host.data(), cluster_sizes.data_handle(),
               cluster_sizes_host.size(), stream);
    raft::resource::sync_stream(res);

    uint64_t matches = 0;
    for (int64_t sample = 0; sample < samples; ++sample) {
      uint32_t sample_label = gathered_host[sample * (k + 1)];
      for (int64_t j = 0; j < k; ++j) {
        matches += gathered_host[sample * (k + 1) + j + 1] == sample_label;
      }
    }
    uint64_t assigned = std::accumulate(cluster_sizes_host.begin(),
                                        cluster_sizes_host.end(), uint64_t{0});
    if (assigned != dataset.rows) {
      throw std::runtime_error(
          "balanced k-means cluster sizes do not sum to n");
    }
    auto [min_cluster, max_cluster] = std::minmax_element(
        cluster_sizes_host.begin(), cluster_sizes_host.end());

    output << opts.label << ",partition-kmeans-flat,0,"
           << "kmeans-balanced-flat-iter" << opts.kmeans_iterations << ','
           << dtype << ',' << dataset.rows << ',' << dataset.dim << ','
           << scaffold::k_cluster_size << ",1,0," << samples << ',' << k
           << ",0," << std::fixed << std::setprecision(6) << build_ms << ','
           << build_ms << ',' << n_clusters << ',' << *min_cluster << ','
           << static_cast<double>(dataset.rows) / n_clusters << ','
           << *max_cluster << ','
           << static_cast<double>(matches) / (samples * k) << ",nan,nan,nan\n";
    output.flush();
    return 0;
  }
  for (int pivot_arity : {2, 3}) {
    auto method = pivot_arity == 2 ? "pivot-binary" : "pivot-ternary";
    std::vector<uint8_t> retained(samples * k, 0);
    double cumulative_partition_ms = 0.0;
    std::size_t checkpoint = 0;
    for (int repeat = 0; repeat < opts.repeats.back(); ++repeat) {
      scaffold::build_params params;
      params.repeats = 1;
      params.pivot_arity = pivot_arity;
      params.seed = repeat == 0
                        ? opts.seed
                        : scaffold::splitmix64(opts.seed ^
                                               (static_cast<uint64_t>(repeat) *
                                                0x9e3779b97f4a7c15ull));
      auto started = clock_type::now();
      auto [leaf_ids, leaves] = partition_once<T>(
          res, raft::make_const_mdspan(device_dataset.view()), params);
      double incremental_ms = elapsed_ms(started, clock_type::now());
      cumulative_partition_ms += incremental_ms;
      rmm::device_uvector<uint32_t> gathered(samples * (k + 1), stream);
      int blocks = static_cast<int>((samples + 255) / 256);
      gather_leaf_labels_kernel<<<blocks, 256, 0, stream>>>(
          device_sample_ids.data(), device_truth.data(), samples, k,
          leaf_ids.data(), gathered.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      std::vector<uint32_t> labels(gathered.size());
      raft::copy(labels.data(), gathered.data(), gathered.size(), stream);
      raft::resource::sync_stream(res);
      for (int64_t sample = 0; sample < samples; ++sample) {
        for (int64_t j = 0; j < k; ++j) {
          retained[sample * k + j] |=
              labels[sample * (k + 1)] == labels[sample * (k + 1) + j + 1];
        }
      }
      if (repeat + 1 != opts.repeats[checkpoint]) {
        continue;
      }
      auto matches =
          std::accumulate(retained.begin(), retained.end(), uint64_t{0});
      int64_t min_leaf = dataset.rows, max_leaf = 0;
      for (auto const &leaf : leaves) {
        auto size = leaf.end - leaf.start;
        min_leaf = std::min(min_leaf, size);
        max_leaf = std::max(max_leaf, size);
      }
      output << opts.label << ",partition,0," << method << ',' << dtype << ','
             << dataset.rows << ',' << dataset.dim << ','
             << scaffold::k_cluster_size << ',' << repeat + 1 << ",0,"
             << samples << ',' << k << ',' << opts.seed << ',' << std::fixed
             << std::setprecision(6) << incremental_ms << ','
             << cumulative_partition_ms << ',' << leaves.size() << ','
             << min_leaf << ','
             << static_cast<double>(dataset.rows) / leaves.size() << ','
             << max_leaf << ','
             << static_cast<double>(matches) / retained.size()
             << ",nan,nan,nan\n";
      output.flush();
      if (++checkpoint == opts.repeats.size()) {
        break;
      }
    }
  }
  int max_degree = opts.repeats.back() * scaffold::k_degree;
  for (int part_count : opts.parts) {
    for (int pivot_arity : {2, 3}) {
      auto method = pivot_arity == 2 ? "pivot-binary" : "pivot-ternary";
      auto offsets = make_offsets(dataset.rows, part_count);
      auto union_graph = raft::make_device_matrix<uint32_t, int64_t>(
          res, dataset.rows, max_degree);
      rmm::device_uvector<uint8_t> union_degrees(dataset.rows, stream);
      RAFT_CUDA_TRY(cudaMemsetAsync(union_degrees.data(), 0,
                                    union_degrees.size(), stream));
      double cumulative_ms = 0.0;
      std::size_t checkpoint = 0;
      for (int repeat = 0; repeat < opts.repeats.back(); ++repeat) {
        scaffold::build_params params;
        params.repeats = 1;
        params.neighbors_per_leaf = scaffold::k_degree;
        params.pivot_arity = pivot_arity;
        params.seed = repeat == 0
                          ? opts.seed
                          : scaffold::splitmix64(
                                opts.seed ^ (static_cast<uint64_t>(repeat) *
                                             0x9e3779b97f4a7c15ull));
        auto started = clock_type::now();
        auto repeat_graph = scaffold::build_once<T>(
            res, raft::make_const_mdspan(device_dataset.view()), offsets,
            params);
        int blocks = static_cast<int>((dataset.rows + 255) / 256);
        scaffold::union_repeat_neighbors_kernel<<<blocks, 256, 0, stream>>>(
            repeat_graph.data_handle(), dataset.rows, union_graph.data_handle(),
            union_degrees.data(), scaffold::k_degree, max_degree);
        RAFT_CUDA_TRY(cudaGetLastError());
        raft::resource::sync_stream(res);
        double incremental_ms = elapsed_ms(started, clock_type::now());
        cumulative_ms += incremental_ms;
        if (repeat + 1 != opts.repeats[checkpoint]) {
          continue;
        }

        auto sample_graph = raft::make_device_matrix<uint32_t, int64_t>(
            res, samples, max_degree);
        rmm::device_uvector<uint8_t> sample_degrees(samples, stream);
        int sample_blocks = static_cast<int>((samples + 255) / 256);
        gather_sample_graph_kernel<<<sample_blocks, 256, 0, stream>>>(
            device_sample_ids.data(), samples, union_graph.data_handle(),
            max_degree, union_degrees.data(), sample_graph.data_handle(),
            sample_degrees.data());
        RAFT_CUDA_TRY(cudaGetLastError());
        std::vector<uint32_t> graph_host(sample_graph.size());
        std::vector<uint8_t> degrees_host(samples);
        raft::copy(graph_host.data(), sample_graph.data_handle(),
                   graph_host.size(), stream);
        raft::copy(degrees_host.data(), sample_degrees.data(), samples, stream);
        raft::resource::sync_stream(res);
        uint64_t matches = 0, cross_matches = 0, cross_total = 0,
                 degree_sum = 0;
        for (int64_t sample = 0; sample < samples; ++sample) {
          degree_sum += degrees_host[sample];
          int query_origin = origin_of(sample_ids[sample], offsets);
          for (int64_t j = 0; j < k; ++j) {
            uint32_t truth_id =
                static_cast<uint32_t>(truth.data[sample * k + j]);
            bool found = false;
            for (int d = 0; d < degrees_host[sample]; ++d) {
              found |= graph_host[sample * max_degree + d] == truth_id;
            }
            matches += found;
            bool cross = origin_of(truth_id, offsets) != query_origin;
            cross_total += cross;
            cross_matches += cross && found;
          }
        }
        output << opts.label << ",implicit-scaffold," << part_count << ','
               << method << ',' << dtype << ',' << dataset.rows << ','
               << dataset.dim << ',' << scaffold::k_cluster_size << ','
               << repeat + 1 << ',' << scaffold::k_degree << ',' << samples
               << ',' << k << ',' << opts.seed << ',' << std::fixed
               << std::setprecision(6) << incremental_ms << ',' << cumulative_ms
               << ",0,0,0,0,nan,"
               << static_cast<double>(matches) / (samples * k) << ','
               << (cross_total == 0
                       ? 0.0
                       : static_cast<double>(cross_matches) / cross_total)
               << ',' << static_cast<double>(degree_sum) / samples << '\n';
        output.flush();
        if (++checkpoint == opts.repeats.size()) {
          break;
        }
      }
    }
  }
  for (int degree : opts.native_degrees) {
    cuvs::neighbors::nn_descent::index_params native_params(
        degree, cuvs::distance::DistanceType::L2Expanded);
    native_params.intermediate_graph_degree = std::max(2 * degree, 64);
    native_params.return_distances = false;
    auto started = clock_type::now();
    auto native = cuvs::neighbors::nn_descent::build(
        res, native_params, raft::make_const_mdspan(device_dataset.view()));
    raft::resource::sync_stream(res);
    double build_ms = elapsed_ms(started, clock_type::now());
    auto graph = native.graph();
    auto contains_truth = [&](int64_t sample, uint32_t truth_id) {
      uint32_t row = sample_ids[sample];
      for (int d = 0; d < degree; ++d) {
        if (graph(row, d) == truth_id) {
          return true;
        }
      }
      return false;
    };

    uint64_t matches = 0;
    for (int64_t sample = 0; sample < samples; ++sample) {
      for (int64_t j = 0; j < k; ++j) {
        matches += contains_truth(
            sample, static_cast<uint32_t>(truth.data[sample * k + j]));
      }
    }
    double recall = static_cast<double>(matches) / (samples * k);
    output << opts.label << ",implicit-native,0,native-nn-descent," << dtype
           << ',' << dataset.rows << ',' << dataset.dim << ",0,1," << degree
           << ',' << samples << ',' << k << ',' << opts.seed << ','
           << std::fixed << std::setprecision(6) << build_ms << ',' << build_ms
           << ",0,0,0,0,nan," << recall << ",nan," << degree << '\n';

    for (int part_count : opts.parts) {
      auto offsets = make_offsets(dataset.rows, part_count);
      uint64_t cross_matches = 0;
      uint64_t cross_total = 0;
      for (int64_t sample = 0; sample < samples; ++sample) {
        int query_origin = origin_of(sample_ids[sample], offsets);
        for (int64_t j = 0; j < k; ++j) {
          uint32_t truth_id = static_cast<uint32_t>(truth.data[sample * k + j]);
          if (origin_of(truth_id, offsets) == query_origin) {
            continue;
          }
          ++cross_total;
          cross_matches += contains_truth(sample, truth_id);
        }
      }
      double cross_recall =
          cross_total == 0 ? 0.0
                           : static_cast<double>(cross_matches) / cross_total;
      output << opts.label << ",implicit-native," << part_count
             << ",native-nn-descent," << dtype << ',' << dataset.rows << ','
             << dataset.dim << ",0,1," << degree << ',' << samples << ',' << k
             << ',' << opts.seed << ',' << std::fixed << std::setprecision(6)
             << build_ms << ',' << build_ms << ",0,0,0,0,nan," << recall << ','
             << cross_recall << ',' << degree << '\n';
    }
    output.flush();
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    auto opts = parse_args(argc, argv);
    return is_uint8_matrix(opts.dataset) ? run<uint8_t>(opts)
                                         : run<float>(opts);
  } catch (std::exception const &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}

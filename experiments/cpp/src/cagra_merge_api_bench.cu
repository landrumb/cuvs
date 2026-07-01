/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_profiler_api.h>

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
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
constexpr std::size_t io_chunk_bytes = 1ull << 30;

struct options {
  std::string dataset;
  std::string queries;
  std::string groundtruth;
  std::string output_csv;
  std::string label;
  std::string implementation;
  uint32_t parts = 2;
  int64_t topk = 12;
  std::size_t graph_degree = 64;
  std::size_t intermediate_graph_degree = 128;
  std::size_t itopk_size = 160;
  bool profile_merge = false;
};

template <typename T> struct bin_matrix {
  uint32_t rows = 0;
  uint32_t dim = 0;
  std::vector<T> data;
};

struct ibin_matrix {
  uint32_t rows = 0;
  uint32_t dim = 0;
  std::vector<int32_t> data;
};

std::string usage() {
  return R"(Usage: CAGRA_MERGE_API_BENCH
  --dataset <base.fbin|base.u8bin>
  --queries <queries.fbin|queries.u8bin>
  --groundtruth <neighbors.ibin>
  --output-csv <results.csv>
  --label <dataset-label>
  --implementation <rebuild|k4-scaffold>
  [--parts <2|4|8>]
  [--graph-degree <int>]
  [--intermediate-graph-degree <int>]
  [--itopk-size <int>]
  [--profile-merge]
)";
}

uint64_t parse_u64(std::string const &value, std::string const &flag) {
  char *end = nullptr;
  auto parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    throw std::runtime_error("Invalid " + flag);
  }
  return parsed;
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
    if (arg == "--dataset")
      opts.dataset = value();
    else if (arg == "--queries")
      opts.queries = value();
    else if (arg == "--groundtruth")
      opts.groundtruth = value();
    else if (arg == "--output-csv")
      opts.output_csv = value();
    else if (arg == "--label")
      opts.label = value();
    else if (arg == "--implementation")
      opts.implementation = value();
    else if (arg == "--parts")
      opts.parts = static_cast<uint32_t>(parse_u64(value(), arg));
    else if (arg == "--graph-degree")
      opts.graph_degree = parse_u64(value(), arg);
    else if (arg == "--intermediate-graph-degree") {
      opts.intermediate_graph_degree = parse_u64(value(), arg);
    } else if (arg == "--itopk-size") {
      opts.itopk_size = parse_u64(value(), arg);
    } else if (arg == "--profile-merge") {
      opts.profile_merge = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << usage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (opts.dataset.empty() || opts.queries.empty() ||
      opts.groundtruth.empty() || opts.output_csv.empty() ||
      opts.label.empty() || opts.implementation.empty()) {
    throw std::runtime_error("All required arguments must be provided\n" +
                             usage());
  }
  if (opts.parts < 2) {
    throw std::runtime_error("--parts must be >= 2");
  }
  if (opts.intermediate_graph_degree < opts.graph_degree) {
    throw std::runtime_error(
        "--intermediate-graph-degree must be >= --graph-degree");
  }
  return opts;
}

void read_exact(std::istream &in, char *dst, std::streamsize bytes,
                std::string const &path) {
  in.read(dst, bytes);
  if (in.gcount() != bytes) {
    throw std::runtime_error("Unexpected EOF while reading " + path);
  }
}

void read_large(std::istream &in, char *dst, std::size_t bytes,
                std::string const &path) {
  std::size_t done = 0;
  while (done < bytes) {
    auto chunk =
        static_cast<std::streamsize>(std::min(io_chunk_bytes, bytes - done));
    in.read(dst + done, chunk);
    if (in.gcount() != chunk) {
      throw std::runtime_error("Unexpected EOF while reading " + path);
    }
    done += static_cast<std::size_t>(chunk);
  }
}

template <typename T> bin_matrix<T> read_bin(std::string const &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open " + path);
  }
  bin_matrix<T> matrix;
  read_exact(in, reinterpret_cast<char *>(&matrix.rows), sizeof(matrix.rows),
             path);
  read_exact(in, reinterpret_cast<char *>(&matrix.dim), sizeof(matrix.dim),
             path);
  auto count = static_cast<std::size_t>(matrix.rows) * matrix.dim;
  auto expected = 2 * sizeof(uint32_t) + count * sizeof(T);
  if (std::filesystem::file_size(path) != expected) {
    throw std::runtime_error("Payload size does not match selected datatype: " +
                             path);
  }
  matrix.data.resize(count);
  read_large(in, reinterpret_cast<char *>(matrix.data.data()),
             count * sizeof(T), path);
  return matrix;
}

ibin_matrix read_ibin(std::string const &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open " + path);
  }
  ibin_matrix matrix;
  read_exact(in, reinterpret_cast<char *>(&matrix.rows), sizeof(matrix.rows),
             path);
  read_exact(in, reinterpret_cast<char *>(&matrix.dim), sizeof(matrix.dim),
             path);
  auto count = static_cast<std::size_t>(matrix.rows) * matrix.dim;
  matrix.data.resize(count);
  read_large(in, reinterpret_cast<char *>(matrix.data.data()),
             count * sizeof(int32_t), path);
  return matrix;
}

bool is_uint8_matrix(std::string const &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open " + path);
  }
  uint32_t rows = 0, dim = 0;
  read_exact(in, reinterpret_cast<char *>(&rows), sizeof(rows), path);
  read_exact(in, reinterpret_cast<char *>(&dim), sizeof(dim), path);
  uint64_t count = static_cast<uint64_t>(rows) * dim;
  uint64_t payload = std::filesystem::file_size(path) - 2 * sizeof(uint32_t);
  if (payload == count)
    return true;
  if (payload == count * sizeof(float))
    return false;
  throw std::runtime_error("Payload is neither uint8 nor float32: " + path);
}

double elapsed_ms(clock_type::time_point start, clock_type::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double recall_at_k(ibin_matrix const &expected,
                   std::vector<uint32_t> const &actual, int64_t topk) {
  if (expected.dim < topk) {
    throw std::runtime_error("Ground truth has fewer than topk columns");
  }
  uint64_t matches = 0;
  for (uint32_t row = 0; row < expected.rows; ++row) {
    for (int64_t col = 0; col < topk; ++col) {
      uint32_t got = actual[static_cast<std::size_t>(row) * topk + col];
      for (int64_t truth_col = 0; truth_col < topk; ++truth_col) {
        int32_t truth =
            expected
                .data[static_cast<std::size_t>(row) * expected.dim + truth_col];
        if (truth >= 0 && got == static_cast<uint32_t>(truth)) {
          ++matches;
          break;
        }
      }
    }
  }
  return static_cast<double>(matches) /
         static_cast<double>(expected.rows * topk);
}

void append_result(options const &opts, char const *dtype, uint32_t rows,
                   uint32_t query_count, double oracle_build_ms,
                   double merge_ms, double search_ms, double recall) {
  bool exists = std::filesystem::exists(opts.output_csv);
  std::ofstream out(opts.output_csv, std::ios::app);
  if (!out) {
    throw std::runtime_error("Could not write " + opts.output_csv);
  }
  if (!exists) {
    out << "dataset,parts,implementation,dtype,rows,queries,graph_degree,"
           "intermediate_graph_degree,itopk,oracle_partition_build_ms_excluded,"
           "merge_api_e2e_ms,search_ms,recall,qps\n";
  }
  double qps = static_cast<double>(query_count) / (search_ms / 1000.0);
  out << opts.label << ',' << opts.parts << ',' << opts.implementation << ','
      << dtype << ',' << rows << ',' << query_count << ',' << opts.graph_degree
      << ',' << opts.intermediate_graph_degree << ',' << opts.itopk_size << ','
      << std::fixed << std::setprecision(6) << oracle_build_ms << ','
      << merge_ms << ',' << search_ms << ',' << recall << ',' << qps << '\n';
}

template <typename T> int run(options const &opts) {
  auto dataset = read_bin<T>(opts.dataset);
  auto queries = read_bin<T>(opts.queries);
  auto groundtruth = read_ibin(opts.groundtruth);
  if (dataset.dim != queries.dim) {
    throw std::runtime_error("Dataset/query dim mismatch");
  }
  if (queries.rows != groundtruth.rows) {
    throw std::runtime_error("Query/ground-truth row mismatch");
  }
  if (opts.parts > dataset.rows) {
    throw std::runtime_error("Too many partitions");
  }

  raft::resources res;
  std::vector<cuvs::neighbors::cagra::index<T, uint32_t>> owned_indices;
  std::vector<cuvs::neighbors::cagra::index<T, uint32_t> *> indices;
  owned_indices.reserve(opts.parts);
  indices.reserve(opts.parts);

  uint32_t base_rows = dataset.rows / opts.parts;
  uint32_t remainder = dataset.rows % opts.parts;
  uint32_t offset = 0;
  auto oracle_start = clock_type::now();
  for (uint32_t part = 0; part < opts.parts; ++part) {
    uint32_t part_rows = base_rows + (part < remainder ? 1 : 0);
    auto view = raft::make_host_matrix_view<const T, int64_t>(
        dataset.data.data() + static_cast<std::size_t>(offset) * dataset.dim,
        part_rows, dataset.dim);
    cuvs::neighbors::cagra::index_params build_params;
    build_params.metric = cuvs::distance::DistanceType::L2Expanded;
    build_params.graph_degree = opts.graph_degree;
    build_params.intermediate_graph_degree = opts.intermediate_graph_degree;
    build_params.attach_dataset_on_build = true;
    build_params.guarantee_connectivity = false;
    build_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
            raft::matrix_extent<int64_t>(part_rows, dataset.dim),
            build_params.metric);
    owned_indices.push_back(
        cuvs::neighbors::cagra::build(res, build_params, view));
    indices.push_back(&owned_indices.back());
    offset += part_rows;
    std::cout << "oracle part " << part << " rows=" << part_rows << " built\n";
  }
  raft::resource::sync_stream(res);
  double oracle_build_ms = elapsed_ms(oracle_start, clock_type::now());

  cuvs::neighbors::cagra::index_params merge_params;
  merge_params.metric = cuvs::distance::DistanceType::L2Expanded;
  merge_params.graph_degree = opts.graph_degree;
  merge_params.intermediate_graph_degree = opts.intermediate_graph_degree;
  merge_params.attach_dataset_on_build = true;
  merge_params.guarantee_connectivity = false;
  merge_params.graph_build_params =
      cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
          raft::matrix_extent<int64_t>(dataset.rows, dataset.dim),
          merge_params.metric);

  raft::resource::sync_stream(res);
  if (opts.profile_merge) {
    RAFT_CUDA_TRY(cudaProfilerStart());
  }
  auto merge_start = clock_type::now();
  auto merged = cuvs::neighbors::cagra::merge(res, merge_params, indices);
  raft::resource::sync_stream(res);
  double merge_ms = elapsed_ms(merge_start, clock_type::now());
  if (opts.profile_merge) {
    RAFT_CUDA_TRY(cudaProfilerStop());
  }

  auto device_queries =
      raft::make_device_matrix<T, int64_t>(res, queries.rows, queries.dim);
  raft::copy(device_queries.data_handle(), queries.data.data(),
             queries.data.size(), raft::resource::get_cuda_stream(res));
  auto neighbors =
      raft::make_device_matrix<uint32_t, int64_t>(res, queries.rows, opts.topk);
  auto distances =
      raft::make_device_matrix<float, int64_t>(res, queries.rows, opts.topk);
  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size = opts.itopk_size;

  auto search_once = [&] {
    cuvs::neighbors::cagra::search(
        res, search_params, merged,
        raft::make_const_mdspan(device_queries.view()), neighbors.view(),
        distances.view());
    raft::resource::sync_stream(res);
  };
  search_once();
  std::vector<double> search_samples;
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto start = clock_type::now();
    search_once();
    search_samples.push_back(elapsed_ms(start, clock_type::now()));
  }
  std::sort(search_samples.begin(), search_samples.end());
  double search_ms = search_samples[search_samples.size() / 2];

  std::vector<uint32_t> actual(neighbors.size());
  raft::copy(actual.data(), neighbors.data_handle(), actual.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  double recall = recall_at_k(groundtruth, actual, opts.topk);

  append_result(opts, std::is_same_v<T, uint8_t> ? "uint8" : "float32",
                dataset.rows, queries.rows, oracle_build_ms, merge_ms,
                search_ms, recall);
  std::cout << "RESULT dataset=" << opts.label << " parts=" << opts.parts
            << " implementation=" << opts.implementation
            << " merge_ms=" << merge_ms << " search_ms=" << search_ms
            << " recall=" << recall
            << " qps=" << (1000.0 * queries.rows / search_ms) << '\n';
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

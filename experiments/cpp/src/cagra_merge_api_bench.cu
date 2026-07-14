/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <neighbors/detail/cagra/cagra_merge.cuh>

#include "cagra_binary_cross_query_baseline.cuh"
#include "kmeans_merge_scaffold.cuh"

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
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
constexpr std::size_t io_chunk_bytes = 1ull << 30;
constexpr int default_scaffold_repeats =
    cuvs::neighbors::cagra::detail::merge_scaffold::k_default_repeats;
constexpr int64_t default_scaffold_candidate_cap = cuvs::neighbors::cagra::
    detail::merge_scaffold::k_cap_to_output_graph_degree;

struct options {
  std::string dataset;
  std::string queries;
  std::string groundtruth;
  std::string output_csv;
  std::string label;
  std::string implementation;
  std::string serialized_index_dir;
  uint32_t parts = 2;
  uint32_t rows = 0;
  int64_t topk = 12;
  std::size_t graph_degree = 64;
  std::size_t intermediate_graph_degree = 128;
  std::size_t itopk_size = 160;
  int scaffold_repeats = default_scaffold_repeats;
  int scaffold_neighbors = 4;
  int scaffold_first_repeat_neighbors = 0;
  uint64_t scaffold_seed = 1234;
  int64_t scaffold_candidate_cap = default_scaffold_candidate_cap;
  int native_knn_degree = 32;
  int kmeans_target_cluster_size = 256;
  int kmeans_iterations = 20;
  int kmeans_tree_branching = 2;
  std::vector<int> scaffold_candidate_cap_list;
  std::vector<int> scaffold_repeat_list;
  std::vector<int> scaffold_neighbor_list;
  std::vector<int> scaffold_first_repeat_neighbor_list;
  std::vector<uint64_t> scaffold_seed_list;
  bool scaffold_repeat_sweep = false;
  bool scaffold_quality = false;
  int64_t quality_sample_rows = 65536;
  int timing_runs = 2;
  bool deserialize_comparison = false;
  bool owning_inputs = false;
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
  --implementation <rebuild|k4-scaffold|ternary-scaffold|binary-cross-query|native-knn|flat-kmeans|kmeans-tree>
  [--parts <2|4|8|16|32|64|128>]
  [--rows <dataset-prefix-rows; default: all>]
  [--graph-degree <int>]
  [--intermediate-graph-degree <int>]
  [--itopk-size <int>]
  [--scaffold-repeats <1-32>]
  [--scaffold-repeat-list <comma-separated counts>]
  [--scaffold-neighbors <1|2|4|8|16|32>]
  [--scaffold-neighbor-list <comma-separated 1|2|4|8|16|32 values>]
  [--scaffold-first-repeat-neighbors <0|1|2|4|8|16|32>]
  [--scaffold-first-repeat-neighbor-list <comma-separated 1|2|4|8|16|32 values>]
  [--scaffold-seed <uint64>]
  [--scaffold-seed-list <comma-separated uint64 values>]
  [--scaffold-candidate-cap <0|at-least-output-degree> (default: output graph degree)]
  [--scaffold-candidate-cap-list <comma-separated ints>]
  [--kmeans-target-cluster-size <positive int>]
  [--kmeans-iterations <1-100>]
  [--kmeans-tree-branching <2|5>]
  [--native-knn-degree <12-128>]
  [--scaffold-repeat-sweep]
  [--scaffold-quality]
  [--quality-sample-rows <int>]
  [--deserialize-comparison]
  [--serialized-index-dir <empty-dir>]
  [--timing-runs <int>]
  [--owning-inputs]
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

std::vector<int> parse_repeat_list(std::string const &value,
                                   std::string const &flag) {
  if (value.empty() || value.front() == ',' || value.back() == ',' ||
      value.find(",,") != std::string::npos) {
    throw std::runtime_error("Invalid " + flag);
  }
  std::vector<int> counts;
  std::size_t start = 0;
  while (start < value.size()) {
    auto end = value.find(',', start);
    auto token = value.substr(start, end - start);
    auto count = parse_u64(token, flag);
    if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Invalid " + flag);
    }
    counts.push_back(static_cast<int>(count));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return counts;
}
std::vector<uint64_t> parse_seed_list(std::string const &value,
                                      std::string const &flag) {
  if (value.empty() || value.front() == ',' || value.back() == ',' ||
      value.find(",,") != std::string::npos) {
    throw std::runtime_error("Invalid " + flag);
  }
  std::vector<uint64_t> seeds;
  std::size_t start = 0;
  while (start < value.size()) {
    auto end = value.find(',', start);
    auto token = value.substr(start, end - start);
    seeds.push_back(parse_u64(token, flag));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return seeds;
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
    else if (arg == "--rows")
      opts.rows = static_cast<uint32_t>(parse_u64(value(), arg));
    else if (arg == "--graph-degree")
      opts.graph_degree = parse_u64(value(), arg);
    else if (arg == "--intermediate-graph-degree") {
      opts.intermediate_graph_degree = parse_u64(value(), arg);
    } else if (arg == "--itopk-size") {
      opts.itopk_size = parse_u64(value(), arg);
    } else if (arg == "--scaffold-repeats") {
      opts.scaffold_repeats = static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--scaffold-repeat-list") {
      opts.scaffold_repeat_list = parse_repeat_list(value(), arg);
    } else if (arg == "--scaffold-neighbors") {
      opts.scaffold_neighbors = static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--scaffold-neighbor-list") {
      opts.scaffold_neighbor_list = parse_repeat_list(value(), arg);
    } else if (arg == "--scaffold-first-repeat-neighbors") {
      opts.scaffold_first_repeat_neighbors =
          static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--scaffold-first-repeat-neighbor-list") {
      opts.scaffold_first_repeat_neighbor_list =
          parse_repeat_list(value(), arg);
    } else if (arg == "--scaffold-seed") {
      opts.scaffold_seed = parse_u64(value(), arg);
    } else if (arg == "--scaffold-seed-list") {
      opts.scaffold_seed_list = parse_seed_list(value(), arg);
    } else if (arg == "--scaffold-candidate-cap") {
      opts.scaffold_candidate_cap =
          static_cast<int64_t>(parse_u64(value(), arg));
    } else if (arg == "--scaffold-candidate-cap-list") {
      opts.scaffold_candidate_cap_list = parse_repeat_list(value(), arg);
    } else if (arg == "--native-knn-degree") {
      opts.native_knn_degree = static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--kmeans-target-cluster-size") {
      opts.kmeans_target_cluster_size =
          static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--kmeans-iterations") {
      opts.kmeans_iterations = static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--kmeans-tree-branching") {
      opts.kmeans_tree_branching = static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--scaffold-repeat-sweep") {
      opts.scaffold_repeat_sweep = true;
    } else if (arg == "--scaffold-quality") {
      opts.scaffold_quality = true;
    } else if (arg == "--quality-sample-rows") {
      opts.quality_sample_rows = static_cast<int64_t>(parse_u64(value(), arg));
    } else if (arg == "--deserialize-comparison") {
      opts.deserialize_comparison = true;
    } else if (arg == "--serialized-index-dir") {
      opts.serialized_index_dir = value();
    } else if (arg == "--timing-runs") {
      opts.timing_runs = static_cast<int>(parse_u64(value(), arg));
    } else if (arg == "--owning-inputs") {
      opts.owning_inputs = true;
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
  if (opts.implementation != "rebuild" &&
      opts.implementation != "k4-scaffold" &&
      opts.implementation != "ternary-scaffold" &&
      opts.implementation != "binary-cross-query" &&
      opts.implementation != "native-knn" &&
      opts.implementation != "flat-kmeans" &&
      opts.implementation != "kmeans-tree") {
    throw std::runtime_error("Unknown --implementation: " +
                             opts.implementation);
  }
  if (opts.parts < 2) {
    throw std::runtime_error("--parts must be >= 2");
  }
  if (opts.native_knn_degree < 12 || opts.native_knn_degree > 128) {
    throw std::runtime_error("--native-knn-degree must be in [12, 128]");
  }
  if (opts.implementation == "native-knn" && opts.owning_inputs) {
    if ((opts.implementation == "flat-kmeans" ||
         opts.implementation == "kmeans-tree") &&
        opts.owning_inputs) {
      throw std::runtime_error(
          "K-means scaffold variants require reusable non-owning inputs");
    }
    if (opts.kmeans_target_cluster_size < 2) {
      throw std::runtime_error(
          "--kmeans-target-cluster-size must be at least 2");
    }
    if (opts.kmeans_iterations < 1 || opts.kmeans_iterations > 100) {
      throw std::runtime_error("--kmeans-iterations must be in [1, 100]");
    }
    if (opts.kmeans_tree_branching != 2 && opts.kmeans_tree_branching != 5) {
      throw std::runtime_error("--kmeans-tree-branching must be 2 or 5");
    }
    throw std::runtime_error(
        "--native-knn requires reusable non-owning inputs");
  }
  if (opts.intermediate_graph_degree < opts.graph_degree) {
    throw std::runtime_error(
        "--intermediate-graph-degree must be >= --graph-degree");
  }
  if (opts.scaffold_repeats < 1 || opts.scaffold_repeats > 32) {
    throw std::runtime_error("--scaffold-repeats must be between 1 and 32");
  }
  for (int repeats : opts.scaffold_repeat_list) {
    if (repeats < 1 || repeats > 32) {
      throw std::runtime_error(
          "--scaffold-repeat-list values must be between 1 and 32");
    }
  }
  auto supported_scaffold_degree = [](int degree) {
    return degree == 1 || degree == 2 || degree == 4 || degree == 8 ||
           degree == 16 || degree == 32;
  };
  if (!supported_scaffold_degree(opts.scaffold_neighbors)) {
    throw std::runtime_error(
        "--scaffold-neighbors must be one of 1, 2, 4, 8, 16, or 32");
  }
  for (int neighbors : opts.scaffold_neighbor_list) {
    if (!supported_scaffold_degree(neighbors)) {
      throw std::runtime_error(
          "--scaffold-neighbor-list values must be 1, 2, 4, 8, 16, or 32");
    }
  }
  if (opts.scaffold_first_repeat_neighbors != 0 &&
      !supported_scaffold_degree(opts.scaffold_first_repeat_neighbors)) {
    throw std::runtime_error(
        "--scaffold-first-repeat-neighbors must be 0, 1, 2, 4, 8, 16, or 32");
  }
  for (int neighbors : opts.scaffold_first_repeat_neighbor_list) {
    if (!supported_scaffold_degree(neighbors)) {
      throw std::runtime_error("--scaffold-first-repeat-neighbor-list values "
                               "must be 1, 2, 4, 8, 16, or 32");
    }
  }

  std::vector<int> validation_repeats;
  if (!opts.scaffold_repeat_list.empty()) {
    validation_repeats = opts.scaffold_repeat_list;
  } else if (opts.scaffold_repeat_sweep) {
    for (int repeats = 1; repeats <= 8; ++repeats) {
      validation_repeats.push_back(repeats);
    }
  } else {
    validation_repeats.push_back(opts.scaffold_repeats);
  }
  std::vector<int> validation_neighbors = opts.scaffold_neighbor_list;
  if (validation_neighbors.empty()) {
    validation_neighbors.push_back(opts.scaffold_neighbors);
  }
  std::vector<int> validation_first_neighbors =
      opts.scaffold_first_repeat_neighbor_list;
  if (validation_first_neighbors.empty()) {
    validation_first_neighbors.push_back(opts.scaffold_first_repeat_neighbors);
  }
  for (int neighbors : validation_neighbors) {
    for (int configured_first_neighbors : validation_first_neighbors) {
      int first_neighbors = configured_first_neighbors == 0
                                ? neighbors
                                : configured_first_neighbors;
      for (int repeats : validation_repeats) {
        int union_degree = first_neighbors + (repeats - 1) * neighbors;
        if (union_degree > 255) {
          throw std::runtime_error(
              "Every scaffold repeat schedule must have union degree <= 255");
        }
      }
    }
  }

  if (opts.scaffold_repeat_sweep && !opts.scaffold_repeat_list.empty()) {
    throw std::runtime_error("--scaffold-repeat-sweep and "
                             "--scaffold-repeat-list are mutually exclusive");
  }
  if (opts.scaffold_repeat_sweep && opts.profile_merge) {
    throw std::runtime_error(
        "--scaffold-repeat-sweep cannot be combined with --profile-merge");
  }
  if (opts.scaffold_repeat_sweep && opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "--scaffold-repeat-sweep requires --implementation k4-scaffold");
  }
  if (!opts.scaffold_repeat_list.empty() &&
      opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "--scaffold-repeat-list requires --implementation k4-scaffold");
  }
  if (!opts.scaffold_neighbor_list.empty() &&
      opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "--scaffold-neighbor-list requires --implementation k4-scaffold");
  }
  if (!opts.scaffold_neighbor_list.empty() && opts.profile_merge) {
    throw std::runtime_error(
        "--scaffold-neighbor-list cannot use --profile-merge");
  }
  if ((opts.scaffold_first_repeat_neighbors != 0 ||
       !opts.scaffold_first_repeat_neighbor_list.empty()) &&
      opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "First-repeat scaffold width requires --implementation k4-scaffold");
  }
  if (!opts.scaffold_first_repeat_neighbor_list.empty() && opts.profile_merge) {
    throw std::runtime_error(
        "--scaffold-first-repeat-neighbor-list cannot use --profile-merge");
  }
  if (!opts.scaffold_seed_list.empty() &&
      opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "--scaffold-seed-list requires --implementation k4-scaffold");
  }
  if (!opts.scaffold_seed_list.empty() && opts.profile_merge) {
    throw std::runtime_error("--scaffold-seed-list cannot use --profile-merge");
  }
  if (opts.scaffold_candidate_cap > 0 &&
      opts.scaffold_candidate_cap < static_cast<int64_t>(opts.graph_degree)) {
    throw std::runtime_error(
        "--scaffold-candidate-cap must be zero or at least --graph-degree");
  }
  for (int cap : opts.scaffold_candidate_cap_list) {
    if (cap > 0 && cap < static_cast<int>(opts.graph_degree)) {
      throw std::runtime_error(
          "--scaffold-candidate-cap-list values must be zero or at least "
          "--graph-degree");
    }
  }
  if ((opts.scaffold_candidate_cap > 0 ||
       !opts.scaffold_candidate_cap_list.empty()) &&
      opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "Scaffold candidate caps require --implementation k4-scaffold");
  }
  if (!opts.scaffold_candidate_cap_list.empty() && opts.profile_merge) {
    throw std::runtime_error(
        "--scaffold-candidate-cap-list cannot use --profile-merge");
  }
  if (opts.scaffold_quality && opts.implementation != "k4-scaffold") {
    throw std::runtime_error(
        "--scaffold-quality requires --implementation k4-scaffold");
  }
  if (opts.quality_sample_rows < 1 || opts.quality_sample_rows > 1000000) {
    throw std::runtime_error(
        "--quality-sample-rows must be between 1 and 1000000");
  }
  if (opts.timing_runs < 1 || opts.timing_runs > 10) {
    throw std::runtime_error("--timing-runs must be between 1 and 10");
  }
  if (opts.deserialize_comparison && opts.serialized_index_dir.empty()) {
    throw std::runtime_error(
        "--serialized-index-dir is required with --deserialize-comparison");
  }
  if (opts.deserialize_comparison &&
      (opts.scaffold_repeat_sweep || !opts.scaffold_repeat_list.empty() ||
       !opts.scaffold_neighbor_list.empty() ||
       opts.scaffold_first_repeat_neighbors != 0 ||
       !opts.scaffold_first_repeat_neighbor_list.empty() ||
       !opts.scaffold_candidate_cap_list.empty() ||
       !opts.scaffold_seed_list.empty() || opts.scaffold_quality ||
       opts.profile_merge)) {
    throw std::runtime_error("--deserialize-comparison cannot be combined with "
                             "sweep, quality, or profiling modes");
  }
  if (opts.owning_inputs &&
      (opts.scaffold_repeat_sweep || !opts.scaffold_repeat_list.empty() ||
       !opts.scaffold_neighbor_list.empty() ||
       !opts.scaffold_first_repeat_neighbor_list.empty() ||
       !opts.scaffold_candidate_cap_list.empty() ||
       !opts.scaffold_seed_list.empty())) {
    throw std::runtime_error(
        "--owning-inputs cannot be reused by a parameter sweep");
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

template <typename T>
bin_matrix<T> read_bin(std::string const &path, uint32_t max_rows = 0) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open " + path);
  }
  bin_matrix<T> matrix;
  uint32_t file_rows = 0;
  read_exact(in, reinterpret_cast<char *>(&file_rows), sizeof(file_rows), path);
  read_exact(in, reinterpret_cast<char *>(&matrix.dim), sizeof(matrix.dim),
             path);
  auto file_count = static_cast<std::size_t>(file_rows) * matrix.dim;
  auto expected = 2 * sizeof(uint32_t) + file_count * sizeof(T);
  if (std::filesystem::file_size(path) != expected) {
    throw std::runtime_error("Payload size does not match selected datatype: " +
                             path);
  }
  if (max_rows > file_rows) {
    throw std::runtime_error("Requested --rows exceeds dataset rows");
  }
  matrix.rows = max_rows == 0 ? file_rows : max_rows;
  auto count = static_cast<std::size_t>(matrix.rows) * matrix.dim;
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

void append_quality_result(
    options const &opts, char const *dtype, uint32_t rows, uint32_t query_count,
    int scaffold_repeats, int scaffold_neighbors,
    int scaffold_first_repeat_neighbors, uint64_t scaffold_seed,
    double oracle_build_ms, double merge_ms, double search_ms, double recall,
    cuvs::neighbors::cagra::detail::merge_scaffold::quality_stats const
        &quality) {
  bool exists = std::filesystem::exists(opts.output_csv);
  std::ofstream out(opts.output_csv, std::ios::app);
  if (!out) {
    throw std::runtime_error("Could not write " + opts.output_csv);
  }
  if (!exists) {
    out << "dataset,parts,implementation,dtype,rows,queries,graph_degree,"
           "intermediate_graph_degree,itopk,scaffold_repeats,"
           "scaffold_neighbors_per_leaf,"
           "scaffold_first_repeat_neighbors_per_leaf,scaffold_seed,"
           "preopt_graph_degree_cap,"
           "quality_sample_rows,"
           "oracle_partition_build_ms_excluded,"
           "merge_api_e2e_ms_instrumented,quality_measurement_ms,"
           "merge_api_ms_excluding_quality_measurement,search_ms,recall,qps,"
           "unique_scaffold_degree_mean,preopt_candidate_rank_mean,"
           "preopt_best_candidate_rank_mean,preopt_top4_candidate_rank_mean,"
           "preopt_fraction_rank_le_16,preopt_fraction_rank_le_32,"
           "preopt_fraction_rank_le_64,"
           "preopt_fraction_rank_le_output_degree,missing_candidates\n";
  }
  double qps = static_cast<double>(query_count) / (search_ms / 1000.0);
  out << opts.label << ',' << opts.parts << ',' << opts.implementation << ','
      << dtype << ',' << rows << ',' << query_count << ',' << opts.graph_degree
      << ',' << opts.intermediate_graph_degree << ',' << opts.itopk_size << ','
      << scaffold_repeats << ',' << scaffold_neighbors << ','
      << scaffold_first_repeat_neighbors << ',' << scaffold_seed << ','
      << opts.scaffold_candidate_cap << ',' << quality.sampled_rows << ','
      << std::fixed << std::setprecision(6) << oracle_build_ms << ','
      << merge_ms << ',' << quality.measurement_ms << ','
      << (merge_ms - quality.measurement_ms) << ',' << search_ms << ','
      << recall << ',' << qps << ',' << quality.unique_degree_mean << ','
      << quality.candidate_rank_mean << ',' << quality.best_candidate_rank_mean
      << ',' << quality.top4_candidate_rank_mean << ','
      << quality.fraction_rank_le_16 << ',' << quality.fraction_rank_le_32
      << ',' << quality.fraction_rank_le_64 << ','
      << quality.fraction_rank_le_output_graph_degree << ','
      << quality.missing_candidates << '\n';
}

void append_deserialize_result(options const &opts, char const *dtype,
                               uint32_t rows, int run,
                               std::string const &implementation,
                               uint64_t serialized_bytes,
                               double oracle_build_ms, double serialize_ms,
                               double deserialize_ms, double merge_ms,
                               double other_overhead_ms, double end_to_end_ms) {
  bool exists = std::filesystem::exists(opts.output_csv);
  std::ofstream out(opts.output_csv, std::ios::app);
  if (!out) {
    throw std::runtime_error("Could not write " + opts.output_csv);
  }
  if (!exists) {
    out << "dataset,parts,implementation,dtype,rows,run,graph_degree,"
           "intermediate_graph_degree,scaffold_repeats,serialized_bytes,"
           "oracle_partition_build_ms_excluded,partition_serialize_ms_excluded,"
           "deserialize_ms,merge_api_ms,other_overhead_ms,total_overhead_ms,"
           "end_to_end_ms,deserialize_pct,merge_api_pct\n";
  }
  double total_overhead_ms = deserialize_ms + other_overhead_ms;
  double deserialize_pct = 100.0 * deserialize_ms / end_to_end_ms;
  double merge_pct = 100.0 * merge_ms / end_to_end_ms;
  out << opts.label << ',' << opts.parts << ',' << implementation << ','
      << dtype << ',' << rows << ',' << run << ',' << opts.graph_degree << ','
      << opts.intermediate_graph_degree << ',' << opts.scaffold_repeats << ','
      << serialized_bytes << ',' << std::fixed << std::setprecision(6)
      << oracle_build_ms << ',' << serialize_ms << ',' << deserialize_ms << ','
      << merge_ms << ',' << other_overhead_ms << ',' << total_overhead_ms << ','
      << end_to_end_ms << ',' << deserialize_pct << ',' << merge_pct << '\n';
}

__device__ int native_partition_of(uint32_t row, int64_t rows, uint32_t parts) {
  int64_t base = rows / parts;
  int64_t remainder = rows % parts;
  int64_t wide_rows = remainder * (base + 1);
  if (static_cast<int64_t>(row) < wide_rows) {
    return static_cast<int>(row / (base + 1));
  }
  return static_cast<int>(remainder + (row - wide_rows) / base);
}

__device__ uint32_t native_partition_start(int part, int64_t rows,
                                           uint32_t parts) {
  int64_t base = rows / parts;
  int64_t remainder = rows % parts;
  if (part < remainder) {
    return static_cast<uint32_t>(part * (base + 1));
  }
  return static_cast<uint32_t>(remainder * (base + 1) +
                               (part - remainder) * base);
}

static __global__ void
filter_native_cross_partition_kernel(uint32_t const *input, int64_t rows,
                                     int degree, uint32_t parts,
                                     uint32_t *output) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  int origin = native_partition_of(static_cast<uint32_t>(row), rows, parts);
  int selected = 0;
  for (int j = 0; j < degree; ++j) {
    uint32_t candidate = input[row * degree + j];
    if (candidate < rows && candidate != row &&
        native_partition_of(candidate, rows, parts) != origin) {
      output[row * degree + selected++] = candidate;
    }
  }

  int fallback_part = (origin + 1) % parts;
  uint32_t fallback = native_partition_start(fallback_part, rows, parts);
  uint32_t fallback_end =
      native_partition_start(fallback_part + 1, rows, parts);
  if (fallback_part + 1 == parts) {
    fallback_end = static_cast<uint32_t>(rows);
  }
  uint32_t fallback_size = fallback_end - fallback;
  for (int j = selected; j < degree; ++j) {
    output[row * degree + j] = fallback + ((j - selected) % fallback_size);
  }
}

template <typename T>
auto merge_with_native_knn(
    raft::resources const &res,
    cuvs::neighbors::cagra::index_params const &params,
    std::vector<cuvs::neighbors::cagra::index<T, uint32_t> *> const &indices,
    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
    uint32_t parts, int native_degree)
    -> cuvs::neighbors::cagra::index<T, uint32_t> {
  auto stream = raft::resource::get_cuda_stream(res);
  cuvs::neighbors::nn_descent::index_params nn_params(native_degree,
                                                      params.metric);
  nn_params.intermediate_graph_degree =
      std::max<std::size_t>(2 * native_degree, 64);
  nn_params.return_distances = false;
  auto native_index =
      cuvs::neighbors::nn_descent::build(res, nn_params, dataset);

  auto native_graph = raft::make_device_matrix<uint32_t, int64_t>(
      res, dataset.extent(0), native_degree);
  raft::copy(native_graph.data_handle(), native_index.graph().data_handle(),
             native_graph.size(), stream);
  auto cross_graph = raft::make_device_matrix<uint32_t, int64_t>(
      res, dataset.extent(0), native_degree);
  int blocks = static_cast<int>((dataset.extent(0) + 255) / 256);
  filter_native_cross_partition_kernel<<<blocks, 256, 0, stream>>>(
      native_graph.data_handle(), dataset.extent(0), native_degree, parts,
      cross_graph.data_handle());
  RAFT_CUDA_TRY(cudaGetLastError());

  std::vector<int64_t> offsets{0};
  int64_t base = dataset.extent(0) / parts;
  int64_t remainder = dataset.extent(0) % parts;
  for (uint32_t part = 0; part < parts; ++part) {
    offsets.push_back(offsets.back() + base + (part < remainder ? 1 : 0));
  }
  auto merged_graph =
      cuvs::neighbors::cagra::detail::merge_scaffold::append_to_input_graphs<
          T, uint32_t>(res, indices, offsets,
                       raft::make_const_mdspan(cross_graph.view()));
  cuvs::neighbors::cagra::detail::graph::sort_knn_graph_device_inplace(
      res, params.metric, dataset, merged_graph.view());
  if (merged_graph.extent(1) > static_cast<int64_t>(params.graph_degree)) {
    merged_graph =
        cuvs::neighbors::cagra::detail::merge_scaffold::cap_sorted_graph(
            res, raft::make_const_mdspan(merged_graph.view()),
            params.graph_degree);
  }

  auto optimized_graph = raft::make_device_matrix<uint32_t, int64_t>(
      res, dataset.extent(0), static_cast<int64_t>(params.graph_degree));
  cuvs::neighbors::cagra::detail::graph::optimize(
      res, merged_graph.view(), optimized_graph.view(),
      params.guarantee_connectivity);
  cuvs::neighbors::cagra::index<T, uint32_t> merged(res, params.metric);
  merged.update_graph(res, std::move(optimized_graph));
  merged.update_dataset(res, dataset);
  raft::resource::sync_stream(res);
  return merged;
}

template <typename T>
auto merge_with_kmeans_scaffold(
    raft::resources const &res,
    cuvs::neighbors::cagra::index_params const &params,
    std::vector<cuvs::neighbors::cagra::index<T, uint32_t> *> const &indices,
    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
    options const &opts) -> cuvs::neighbors::cagra::index<T, uint32_t> {
  std::vector<int64_t> offsets{0};
  for (auto const *index : indices) {
    offsets.push_back(offsets.back() + static_cast<int64_t>(index->size()));
  }
  RAFT_EXPECTS(offsets.back() == dataset.extent(0),
               "K-means scaffold offsets do not cover the dataset");

  auto layout =
      opts.implementation == "flat-kmeans"
          ? fastener_experiment::kmeans_scaffold::flat_balanced(
                res, dataset, opts.kmeans_target_cluster_size,
                opts.kmeans_iterations)
          : fastener_experiment::kmeans_scaffold::lloyd_tree(
                res, dataset, opts.kmeans_tree_branching,
                cuvs::neighbors::cagra::detail::merge_scaffold::k_cluster_size,
                opts.kmeans_iterations);
  std::cout << "kmeans scaffold clusters=" << layout.clusters.size()
            << " mode=" << opts.implementation << '\n';
  auto scaffold = fastener_experiment::kmeans_scaffold::build_graph(
      res, dataset, offsets, layout, 4);
  auto merged_graph =
      cuvs::neighbors::cagra::detail::merge_scaffold::append_to_input_graphs<
          T, uint32_t>(res, indices, offsets,
                       raft::make_const_mdspan(scaffold.view()));
  cuvs::neighbors::cagra::detail::graph::sort_knn_graph_device_inplace(
      res, params.metric, dataset, merged_graph.view());
  if (merged_graph.extent(1) > static_cast<int64_t>(params.graph_degree)) {
    merged_graph =
        cuvs::neighbors::cagra::detail::merge_scaffold::cap_sorted_graph(
            res, raft::make_const_mdspan(merged_graph.view()),
            params.graph_degree);
  }

  auto optimized_graph = raft::make_device_matrix<uint32_t, int64_t>(
      res, dataset.extent(0), static_cast<int64_t>(params.graph_degree));
  cuvs::neighbors::cagra::detail::graph::optimize(
      res, merged_graph.view(), optimized_graph.view(),
      params.guarantee_connectivity);
  cuvs::neighbors::cagra::index<T, uint32_t> merged(res, params.metric);
  merged.update_graph(res, std::move(optimized_graph));
  merged.update_dataset(res, dataset);
  raft::resource::sync_stream(res);
  return merged;
}
template <typename T> int run(options const &opts) {
  auto dataset = read_bin<T>(opts.dataset, opts.rows);
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
  // Keep one device-resident dataset and attach non-owning slices to every
  // partition index. Fastener consumes owning input datasets after a
  // successful merge, while parameter sweeps intentionally reuse the same
  // partition graphs for several independent merge configurations.
  auto device_dataset = raft::make_device_matrix<T, int64_t>(
      res, opts.owning_inputs ? 0 : dataset.rows, dataset.dim);
  if (!opts.owning_inputs) {
    raft::copy(device_dataset.data_handle(), dataset.data.data(),
               dataset.data.size(), raft::resource::get_cuda_stream(res));
  }
  raft::resource::sync_stream(res);

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
    cuvs::neighbors::cagra::index_params build_params;
    build_params.metric = cuvs::distance::DistanceType::L2Expanded;
    build_params.graph_degree = opts.graph_degree;
    build_params.intermediate_graph_degree = opts.intermediate_graph_degree;
    build_params.attach_dataset_on_build = opts.owning_inputs;
    build_params.guarantee_connectivity = false;
    build_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
            raft::matrix_extent<int64_t>(part_rows, dataset.dim),
            build_params.metric);
    if (opts.owning_inputs) {
      auto view = raft::make_host_matrix_view<const T, int64_t>(
          dataset.data.data() + static_cast<std::size_t>(offset) * dataset.dim,
          part_rows, dataset.dim);
      owned_indices.push_back(
          cuvs::neighbors::cagra::build(res, build_params, view));
    } else {
      auto view = raft::make_device_matrix_view<const T, int64_t>(
          device_dataset.data_handle() +
              static_cast<std::size_t>(offset) * dataset.dim,
          part_rows, dataset.dim);
      owned_indices.push_back(
          cuvs::neighbors::cagra::build(res, build_params, view));
      owned_indices.back().update_dataset(res, view);
    }
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

  if (opts.deserialize_comparison) {
    using index_t = cuvs::neighbors::cagra::index<T, uint32_t>;
    std::filesystem::path serialized_dir(opts.serialized_index_dir);
    if (std::filesystem::exists(serialized_dir)) {
      throw std::runtime_error("--serialized-index-dir must not already exist");
    }
    std::filesystem::create_directories(serialized_dir);

    std::vector<std::string> serialized_paths;
    serialized_paths.reserve(opts.parts);
    auto serialize_start = clock_type::now();
    for (uint32_t part = 0; part < opts.parts; ++part) {
      auto path = serialized_dir / ("part_" + std::to_string(part) + ".cag");
      cuvs::neighbors::cagra::serialize(res, path.string(), *indices[part],
                                        true);
      serialized_paths.push_back(path.string());
    }
    raft::resource::sync_stream(res);
    double serialize_ms = elapsed_ms(serialize_start, clock_type::now());
    uint64_t serialized_bytes = 0;
    for (auto const &path : serialized_paths) {
      serialized_bytes += std::filesystem::file_size(path);
    }

    indices.clear();
    owned_indices.clear();
    raft::resource::sync_stream(res);

    auto run_one = [&](std::string const &implementation, int run,
                       bool record) {
      auto total_start = clock_type::now();
      std::vector<index_t> loaded_indices;
      std::vector<index_t *> loaded_ptrs;
      loaded_indices.reserve(opts.parts);
      loaded_ptrs.reserve(opts.parts);
      for (uint32_t part = 0; part < opts.parts; ++part) {
        loaded_indices.emplace_back(res);
      }

      auto deserialize_start = clock_type::now();
      for (uint32_t part = 0; part < opts.parts; ++part) {
        cuvs::neighbors::cagra::deserialize(res, serialized_paths[part],
                                            &loaded_indices[part]);
      }
      raft::resource::sync_stream(res);
      double deserialize_ms = elapsed_ms(deserialize_start, clock_type::now());

      for (auto &index : loaded_indices) {
        loaded_ptrs.push_back(&index);
      }
      auto merge_start = clock_type::now();
      auto merged = [&] {
        if (implementation == "rebuild") {
          return cuvs::neighbors::cagra::detail::merge_rebuild<false>(
              res, merge_params, loaded_ptrs,
              cuvs::neighbors::filtering::none_sample_filter{});
        }
        cuvs::neighbors::cagra::detail::merge_scaffold::build_params
            scaffold_params;
        scaffold_params.repeats = opts.scaffold_repeats;
        scaffold_params.neighbors_per_leaf = opts.scaffold_neighbors;
        scaffold_params.first_repeat_neighbors_per_leaf =
            opts.scaffold_first_repeat_neighbors;
        scaffold_params.seed = opts.scaffold_seed;
        scaffold_params.preopt_graph_degree_cap = opts.scaffold_candidate_cap;
        return cuvs::neighbors::cagra::detail::merge_with_k4_scaffold(
            res, merge_params, loaded_ptrs, scaffold_params);
      }();
      raft::resource::sync_stream(res);
      double merge_ms = elapsed_ms(merge_start, clock_type::now());
      double end_to_end_ms = elapsed_ms(total_start, clock_type::now());
      double other_overhead_ms = end_to_end_ms - deserialize_ms - merge_ms;

      if (record) {
        append_deserialize_result(
            opts, std::is_same_v<T, uint8_t> ? "uint8" : "float32",
            dataset.rows, run, implementation, serialized_bytes,
            oracle_build_ms, serialize_ms, deserialize_ms, merge_ms,
            other_overhead_ms, end_to_end_ms);
      }
      std::cout << (record ? "RESULT" : "WARMUP") << " dataset=" << opts.label
                << " parts=" << opts.parts
                << " implementation=" << implementation << " run=" << run
                << " deserialize_ms=" << deserialize_ms
                << " merge_ms=" << merge_ms
                << " other_overhead_ms=" << other_overhead_ms
                << " end_to_end_ms=" << end_to_end_ms << '\n';
      (void)merged;
    };

    std::string fastener =
        "fastener-repeat" + std::to_string(opts.scaffold_repeats);
    run_one(fastener, 0, false);
    run_one("rebuild", 0, false);
    for (int run = 1; run <= opts.timing_runs; ++run) {
      if (run % 2 == 1) {
        run_one(fastener, run, true);
        run_one("rebuild", run, true);
      } else {
        run_one("rebuild", run, true);
        run_one(fastener, run, true);
      }
    }
    return 0;
  }

  raft::resource::sync_stream(res);
  if (opts.scaffold_repeat_sweep || !opts.scaffold_repeat_list.empty() ||
      !opts.scaffold_neighbor_list.empty() ||
      !opts.scaffold_first_repeat_neighbor_list.empty() ||
      !opts.scaffold_candidate_cap_list.empty() ||
      !opts.scaffold_seed_list.empty()) {
    cuvs::neighbors::cagra::detail::merge_scaffold::build_params warmup_params;
    warmup_params.repeats = opts.scaffold_repeats;
    warmup_params.neighbors_per_leaf =
        opts.scaffold_neighbor_list.empty()
            ? opts.scaffold_neighbors
            : opts.scaffold_neighbor_list.front();
    warmup_params.first_repeat_neighbors_per_leaf =
        opts.scaffold_first_repeat_neighbor_list.empty()
            ? opts.scaffold_first_repeat_neighbors
            : opts.scaffold_first_repeat_neighbor_list.front();
    warmup_params.seed = opts.scaffold_seed;
    warmup_params.preopt_graph_degree_cap =
        opts.scaffold_candidate_cap_list.empty()
            ? opts.scaffold_candidate_cap
            : opts.scaffold_candidate_cap_list.front();
    auto warmup = cuvs::neighbors::cagra::detail::merge_with_k4_scaffold(
        res, merge_params, indices, warmup_params);
    raft::resource::sync_stream(res);
    (void)warmup;
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

  std::vector<int> repeat_counts;
  if (!opts.scaffold_repeat_list.empty()) {
    repeat_counts = opts.scaffold_repeat_list;
  } else if (opts.scaffold_repeat_sweep) {
    for (int repeats = 1; repeats <= 8; ++repeats) {
      repeat_counts.push_back(repeats);
    }
  } else {
    repeat_counts.push_back(opts.scaffold_repeats);
  }

  std::vector<int> neighbor_counts = opts.scaffold_neighbor_list;
  if (neighbor_counts.empty()) {
    neighbor_counts.push_back(opts.scaffold_neighbors);
  }
  std::vector<int> first_neighbor_counts =
      opts.scaffold_first_repeat_neighbor_list;
  if (first_neighbor_counts.empty()) {
    first_neighbor_counts.push_back(opts.scaffold_first_repeat_neighbors);
  }
  std::vector<int64_t> candidate_caps(opts.scaffold_candidate_cap_list.begin(),
                                      opts.scaffold_candidate_cap_list.end());
  if (candidate_caps.empty()) {
    candidate_caps.push_back(opts.scaffold_candidate_cap);
  }
  std::vector<uint64_t> scaffold_seeds = opts.scaffold_seed_list;
  if (scaffold_seeds.empty()) {
    scaffold_seeds.push_back(opts.scaffold_seed);
  }
  std::vector<std::tuple<int, int, int, uint64_t, int64_t>> scaffold_configs;
  for (uint64_t seed : scaffold_seeds) {
    for (int neighbors_per_leaf : neighbor_counts) {
      for (int configured_first_neighbors : first_neighbor_counts) {
        int first_neighbors = configured_first_neighbors == 0
                                  ? neighbors_per_leaf
                                  : configured_first_neighbors;
        for (int repeats : repeat_counts) {
          int64_t candidate_degree = static_cast<int64_t>(opts.graph_degree) +
                                     first_neighbors +
                                     (repeats - 1) * neighbors_per_leaf;
          for (int64_t candidate_cap : candidate_caps) {
            int64_t resolved_candidate_cap =
                candidate_cap == default_scaffold_candidate_cap
                    ? static_cast<int64_t>(opts.graph_degree)
                    : candidate_cap;
            if (resolved_candidate_cap > candidate_degree) {
              std::cout << "SKIP scaffold_neighbors=" << neighbors_per_leaf
                        << " scaffold_first_neighbors=" << first_neighbors
                        << " scaffold_repeats=" << repeats
                        << " candidate_cap=" << resolved_candidate_cap
                        << " candidate_degree=" << candidate_degree << '\n';
              continue;
            }
            scaffold_configs.emplace_back(repeats, neighbors_per_leaf,
                                          first_neighbors, seed,
                                          resolved_candidate_cap);
          }
        }
      }
    }
  }

  for (auto const &[scaffold_repeats, scaffold_neighbors,
                    scaffold_first_neighbors, scaffold_seed, scaffold_cap] :
       scaffold_configs) {
    cuvs::neighbors::cagra::detail::merge_scaffold::quality_stats quality;
    if (opts.profile_merge) {
      RAFT_CUDA_TRY(cudaProfilerStart());
    }
    auto merge_start = clock_type::now();
    auto merged = [&] {
      if (opts.implementation == "rebuild") {
        return cuvs::neighbors::cagra::detail::merge_rebuild<false>(
            res, merge_params, indices,
            cuvs::neighbors::filtering::none_sample_filter{});
      }
      if (opts.implementation == "binary-cross-query") {
        return cuvs::neighbors::cagra::detail::binary_cross_query::merge_tree(
            res, merge_params, std::move(owned_indices));
      }
      if (opts.implementation == "native-knn") {
        return merge_with_native_knn<T>(
            res, merge_params, indices,
            raft::make_const_mdspan(device_dataset.view()), opts.parts,
            opts.native_knn_degree);
      }
      if (opts.implementation == "flat-kmeans" ||
          opts.implementation == "kmeans-tree") {
        return merge_with_kmeans_scaffold<T>(
            res, merge_params, indices,
            raft::make_const_mdspan(device_dataset.view()), opts);
      }

      cuvs::neighbors::cagra::detail::merge_scaffold::build_params
          scaffold_params;
      scaffold_params.repeats = scaffold_repeats;
      scaffold_params.pivot_arity =
          opts.implementation == "ternary-scaffold" ? 3 : 2;
      scaffold_params.neighbors_per_leaf = scaffold_neighbors;
      scaffold_params.first_repeat_neighbors_per_leaf =
          scaffold_first_neighbors;
      scaffold_params.seed = scaffold_seed;
      scaffold_params.preopt_graph_degree_cap = scaffold_cap;
      scaffold_params.quality_sample_rows = opts.quality_sample_rows;
      scaffold_params.quality_stats_output =
          opts.scaffold_quality ? &quality : nullptr;
      return cuvs::neighbors::cagra::detail::merge_with_k4_scaffold(
          res, merge_params, indices, scaffold_params);
    }();
    raft::resource::sync_stream(res);
    double merge_ms = elapsed_ms(merge_start, clock_type::now());
    if (opts.profile_merge) {
      RAFT_CUDA_TRY(cudaProfilerStop());
    }

    auto search_once = [&] {
      cuvs::neighbors::cagra::search(
          res, search_params, merged,
          raft::make_const_mdspan(device_queries.view()), neighbors.view(),
          distances.view());
      raft::resource::sync_stream(res);
    };
    search_once();
    std::vector<double> search_samples;
    for (int sample = 0; sample < 3; ++sample) {
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

    auto result_opts = opts;
    result_opts.scaffold_candidate_cap = scaffold_cap;
    if (opts.scaffold_repeat_sweep || !opts.scaffold_repeat_list.empty()) {
      result_opts.implementation +=
          "-repeat" + std::to_string(scaffold_repeats);
    } else if (opts.implementation == "k4-scaffold" &&
               scaffold_repeats != default_scaffold_repeats) {
      result_opts.implementation +=
          "-repeat" + std::to_string(scaffold_repeats);
    }
    if (opts.implementation == "k4-scaffold" && scaffold_neighbors != 4) {
      result_opts.implementation += "-k" + std::to_string(scaffold_neighbors);
    }
    if (opts.implementation == "k4-scaffold" &&
        scaffold_first_neighbors != scaffold_neighbors) {
      result_opts.implementation +=
          "-first-k" + std::to_string(scaffold_first_neighbors);
    }
    if (opts.implementation == "k4-scaffold" &&
        (!opts.scaffold_seed_list.empty() || scaffold_seed != 1234)) {
      result_opts.implementation += "-seed" + std::to_string(scaffold_seed);
    }
    if (opts.implementation == "k4-scaffold" && scaffold_cap > 0) {
      result_opts.implementation += "-cap" + std::to_string(scaffold_cap);
    }
    if (opts.implementation == "native-knn") {
      result_opts.implementation +=
          "-k" + std::to_string(opts.native_knn_degree);
    }
    if (opts.implementation == "flat-kmeans") {
      result_opts.implementation +=
          "-target" + std::to_string(opts.kmeans_target_cluster_size) +
          "-iter" + std::to_string(opts.kmeans_iterations) + "-k4-cap64";
    }
    if (opts.implementation == "kmeans-tree") {
      result_opts.implementation +=
          "-b" + std::to_string(opts.kmeans_tree_branching) + "-leaf" +
          std::to_string(
              cuvs::neighbors::cagra::detail::merge_scaffold::k_cluster_size) +
          "-iter" + std::to_string(opts.kmeans_iterations) + "-k4-cap64";
    }
    if (opts.scaffold_quality) {
      append_quality_result(
          result_opts, std::is_same_v<T, uint8_t> ? "uint8" : "float32",
          dataset.rows, queries.rows, scaffold_repeats, scaffold_neighbors,
          scaffold_first_neighbors, scaffold_seed, oracle_build_ms, merge_ms,
          search_ms, recall, quality);
    } else {
      append_result(result_opts,
                    std::is_same_v<T, uint8_t> ? "uint8" : "float32",
                    dataset.rows, queries.rows, oracle_build_ms, merge_ms,
                    search_ms, recall);
    }
    std::cout << "RESULT dataset=" << result_opts.label
              << " parts=" << result_opts.parts
              << " implementation=" << result_opts.implementation
              << " merge_ms=" << merge_ms << " search_ms=" << search_ms
              << " recall=" << recall
              << " qps=" << (1000.0 * queries.rows / search_ms)
              << " scaffold_neighbors=" << scaffold_neighbors
              << " scaffold_first_neighbors=" << scaffold_first_neighbors
              << " scaffold_seed=" << scaffold_seed
              << " candidate_cap=" << scaffold_cap;
    if (opts.scaffold_quality) {
      std::cout << " quality_ms=" << quality.measurement_ms
                << " unique_degree_mean=" << quality.unique_degree_mean
                << " candidate_rank_mean=" << quality.candidate_rank_mean
                << " best_rank_mean=" << quality.best_candidate_rank_mean
                << " top4_rank_mean=" << quality.top4_candidate_rank_mean
                << " rank_le_64=" << quality.fraction_rank_le_64
                << " missing=" << quality.missing_candidates;
    }
    std::cout << '\n';
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

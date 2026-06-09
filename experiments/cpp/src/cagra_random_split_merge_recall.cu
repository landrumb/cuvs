/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

enum class graph_build_algo {
  auto_select,
  ivf_pq,
  nn_descent,
};

struct options {
  std::string dataset_path;
  std::string queries_path;
  std::string output_dir;
  int64_t topk = 10;
  uint64_t seed = 1234;
  graph_build_algo build_algo = graph_build_algo::auto_select;
  std::optional<size_t> graph_degree;
  std::optional<size_t> intermediate_graph_degree;
  std::optional<size_t> itopk_size;
};

struct fbin_matrix {
  uint32_t rows = 0;
  uint32_t dim  = 0;
  std::vector<float> data;
};

struct timings {
  double bruteforce_ms        = 0.0;
  double part0_search_ms      = 0.0;
  double part1_search_ms      = 0.0;
  double merge_ms             = 0.0;
  double total_split_search_ms = 0.0;
};

std::string usage()
{
  return R"(Usage:
  CAGRA_RANDOM_SPLIT_MERGE_RECALL \
    --dataset <base.fbin> \
    --queries <query.fbin> \
    --output <dir> \
    [--topk <int>] \
    [--seed <uint64>] \
    [--metric l2] \
    [--graph-build-algo auto|ivf-pq|nn-descent] \
    [--graph-degree <int>] \
    [--intermediate-graph-degree <int>] \
    [--itopk-size <int>]
)";
}

std::string graph_build_algo_to_string(graph_build_algo algo)
{
  switch (algo) {
    case graph_build_algo::auto_select: return "auto";
    case graph_build_algo::ivf_pq: return "ivf-pq";
    case graph_build_algo::nn_descent: return "nn-descent";
  }
  return "unknown";
}

graph_build_algo parse_graph_build_algo(const std::string& value)
{
  if (value == "auto") { return graph_build_algo::auto_select; }
  if (value == "ivf-pq" || value == "ivf_pq" || value == "IVF_PQ") {
    return graph_build_algo::ivf_pq;
  }
  if (value == "nn-descent" || value == "nn_descent" || value == "NN_DESCENT") {
    return graph_build_algo::nn_descent;
  }
  throw std::runtime_error("Invalid --graph-build-algo");
}

uint64_t parse_u64(const std::string& value, const std::string& flag)
{
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') { throw std::runtime_error("Invalid " + flag); }
  return static_cast<uint64_t>(parsed);
}

int64_t parse_i64(const std::string& value, const std::string& flag)
{
  char* end          = nullptr;
  long long parsed   = std::strtoll(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') { throw std::runtime_error("Invalid " + flag); }
  return static_cast<int64_t>(parsed);
}

options parse_args(int argc, char** argv)
{
  options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) { throw std::runtime_error("Missing value for " + flag); }
      return argv[++i];
    };

    if (arg == "--dataset") {
      opts.dataset_path = need_value(arg);
    } else if (arg == "--queries") {
      opts.queries_path = need_value(arg);
    } else if (arg == "--output") {
      opts.output_dir = need_value(arg);
    } else if (arg == "--topk") {
      opts.topk = parse_i64(need_value(arg), arg);
    } else if (arg == "--seed") {
      opts.seed = parse_u64(need_value(arg), arg);
    } else if (arg == "--graph-degree") {
      opts.graph_degree = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--intermediate-graph-degree") {
      opts.intermediate_graph_degree = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--itopk-size") {
      opts.itopk_size = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--metric") {
      auto metric = need_value(arg);
      if (metric != "l2") { throw std::runtime_error("Only --metric l2 is supported"); }
    } else if (arg == "--graph-build-algo") {
      opts.build_algo = parse_graph_build_algo(need_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << usage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.dataset_path.empty()) { throw std::runtime_error("--dataset is required"); }
  if (opts.queries_path.empty()) { throw std::runtime_error("--queries is required"); }
  if (opts.output_dir.empty()) { throw std::runtime_error("--output is required"); }
  if (opts.topk <= 0) { throw std::runtime_error("--topk must be positive"); }
  return opts;
}

cuvs::neighbors::cagra::index_params make_index_params(const options& opts,
                                                       uint32_t rows,
                                                       uint32_t dim)
{
  cuvs::neighbors::cagra::index_params index_params;
  index_params.metric                  = cuvs::distance::DistanceType::L2Expanded;
  index_params.attach_dataset_on_build = true;
  if (opts.graph_degree) { index_params.graph_degree = *opts.graph_degree; }
  if (opts.intermediate_graph_degree) {
    index_params.intermediate_graph_degree = *opts.intermediate_graph_degree;
  }

  switch (opts.build_algo) {
    case graph_build_algo::auto_select: break;
    case graph_build_algo::ivf_pq:
      index_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
          raft::matrix_extent<int64_t>(rows, dim), index_params.metric);
      break;
    case graph_build_algo::nn_descent:
      index_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
          index_params.intermediate_graph_degree, index_params.metric);
      break;
  }

  return index_params;
}

std::string json_escape(const std::string& input)
{
  std::ostringstream out;
  for (char c : input) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

std::string command_line(int argc, char** argv)
{
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) out << ' ';
    out << argv[i];
  }
  return out.str();
}

void read_exact(std::istream& in, char* dst, std::streamsize bytes, const std::string& path)
{
  in.read(dst, bytes);
  if (in.gcount() != bytes) { throw std::runtime_error("Unexpected EOF while reading " + path); }
}

fbin_matrix read_fbin(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("Could not open " + path); }

  fbin_matrix matrix;
  read_exact(in, reinterpret_cast<char*>(&matrix.rows), sizeof(uint32_t), path);
  read_exact(in, reinterpret_cast<char*>(&matrix.dim), sizeof(uint32_t), path);

  const size_t count = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.dim);
  matrix.data.resize(count);
  read_exact(in, reinterpret_cast<char*>(matrix.data.data()), count * sizeof(float), path);
  return matrix;
}

void write_fbin(const std::filesystem::path& path,
                uint32_t rows,
                uint32_t dim,
                const std::vector<float>& data)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

void write_ibin(const std::filesystem::path& path, const std::vector<uint32_t>& ids)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  uint32_t rows = static_cast<uint32_t>(ids.size());
  uint32_t dim  = 1;
  out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
  for (uint32_t id : ids) {
    if (id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("original_ids.ibin requires int32-compatible row IDs");
    }
    int32_t signed_id = static_cast<int32_t>(id);
    out.write(reinterpret_cast<const char*>(&signed_id), sizeof(int32_t));
  }
}

raft::device_matrix<float, int64_t> copy_to_device(raft::device_resources const& res,
                                                   const fbin_matrix& matrix)
{
  auto out = raft::make_device_matrix<float, int64_t>(res, matrix.rows, matrix.dim);
  raft::copy(out.data_handle(),
             matrix.data.data(),
             matrix.data.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return out;
}

raft::device_matrix<float, int64_t> copy_to_device(raft::device_resources const& res,
                                                   const std::vector<float>& data,
                                                   uint32_t rows,
                                                   uint32_t dim)
{
  auto out = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(out.data_handle(), data.data(), data.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return out;
}

std::vector<float> gather_rows(const fbin_matrix& matrix, const std::vector<uint32_t>& row_ids)
{
  std::vector<float> out(static_cast<size_t>(row_ids.size()) * matrix.dim);
  for (size_t local_row = 0; local_row < row_ids.size(); ++local_row) {
    const auto src = static_cast<size_t>(row_ids[local_row]) * matrix.dim;
    const auto dst = local_row * matrix.dim;
    std::copy_n(matrix.data.data() + src, matrix.dim, out.data() + dst);
  }
  return out;
}

std::vector<float> concatenate(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
  std::vector<float> out;
  out.reserve(lhs.size() + rhs.size());
  out.insert(out.end(), lhs.begin(), lhs.end());
  out.insert(out.end(), rhs.begin(), rhs.end());
  return out;
}

template <typename T>
std::vector<T> copy_device_matrix_to_host(raft::device_resources const& res,
                                          raft::device_matrix_view<T, int64_t> matrix)
{
  std::vector<T> host(matrix.size());
  raft::copy(host.data(), matrix.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return host;
}

double elapsed_ms(clock_type::time_point start, clock_type::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename Fn>
double time_cuda(raft::device_resources const& res, Fn&& fn)
{
  auto start = clock_type::now();
  fn();
  raft::resource::sync_stream(res);
  return elapsed_ms(start, clock_type::now());
}

double recall_at_k(const std::vector<int64_t>& expected,
                   const std::vector<uint32_t>& actual,
                   int64_t rows,
                   int64_t k)
{
  uint64_t matches = 0;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t actual_col = 0; actual_col < k; ++actual_col) {
      const auto actual_id = static_cast<int64_t>(actual[static_cast<size_t>(row * k + actual_col)]);
      for (int64_t expected_col = 0; expected_col < k; ++expected_col) {
        if (actual_id == expected[static_cast<size_t>(row * k + expected_col)]) {
          ++matches;
          break;
        }
      }
    }
  }
  return static_cast<double>(matches) / static_cast<double>(rows * k);
}

void write_partition_metadata(const std::filesystem::path& path,
                              const options& opts,
                              uint32_t rows,
                              uint32_t dim,
                              uint32_t local_start,
                              int part_id,
                              const cuvs::neighbors::cagra::index_params& index_params,
                              const cuvs::neighbors::cagra::search_params& search_params)
{
  std::ofstream out(path);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"partition_id\": " << part_id << ",\n"
      << "  \"rows\": " << rows << ",\n"
      << "  \"dim\": " << dim << ",\n"
      << "  \"local_id_start\": " << local_start << ",\n"
      << "  \"local_id_end_exclusive\": " << (local_start + rows) << ",\n"
      << "  \"source_dataset\": \"" << json_escape(opts.dataset_path) << "\",\n"
      << "  \"seed\": " << opts.seed << ",\n"
      << "  \"metric\": \"l2_expanded\",\n"
      << "  \"graph_build_algo\": \"" << graph_build_algo_to_string(opts.build_algo) << "\",\n"
      << "  \"index_params\": {\n"
      << "    \"graph_degree\": " << index_params.graph_degree << ",\n"
      << "    \"intermediate_graph_degree\": " << index_params.intermediate_graph_degree << ",\n"
      << "    \"attach_dataset_on_build\": true\n"
      << "  },\n"
      << "  \"search_params\": {\n"
      << "    \"itopk_size\": " << search_params.itopk_size << "\n"
      << "  }\n"
      << "}\n";
}

void write_manifest(const std::filesystem::path& path,
                    const options& opts,
                    const std::string& cmdline,
                    const fbin_matrix& dataset,
                    const fbin_matrix& queries,
                    uint32_t part0_rows,
                    uint32_t part1_rows,
                    const cuvs::neighbors::cagra::index_params& index_params,
                    const cuvs::neighbors::cagra::search_params& search_params,
                    double recall,
                    const timings& t)
{
  std::ofstream out(path);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"command_line\": \"" << json_escape(cmdline) << "\",\n"
      << "  \"dataset_path\": \"" << json_escape(opts.dataset_path) << "\",\n"
      << "  \"queries_path\": \"" << json_escape(opts.queries_path) << "\",\n"
      << "  \"rows\": " << dataset.rows << ",\n"
      << "  \"dim\": " << dataset.dim << ",\n"
      << "  \"query_rows\": " << queries.rows << ",\n"
      << "  \"topk\": " << opts.topk << ",\n"
      << "  \"seed\": " << opts.seed << ",\n"
      << "  \"metric\": \"l2_expanded\",\n"
      << "  \"split_policy\": \"seeded_random_halves\",\n"
      << "  \"graph_build_algo\": \"" << graph_build_algo_to_string(opts.build_algo) << "\",\n"
      << "  \"partition_order\": [\"part_000\", \"part_001\"],\n"
      << "  \"partitions\": [\n"
      << "    {\"name\": \"part_000\", \"rows\": " << part0_rows << ", \"local_id_start\": 0},\n"
      << "    {\"name\": \"part_001\", \"rows\": " << part1_rows
      << ", \"local_id_start\": " << part0_rows << "}\n"
      << "  ],\n"
      << "  \"index_params\": {\n"
      << "    \"graph_degree\": " << index_params.graph_degree << ",\n"
      << "    \"intermediate_graph_degree\": " << index_params.intermediate_graph_degree << ",\n"
      << "    \"attach_dataset_on_build\": true\n"
      << "  },\n"
      << "  \"search_params\": {\n"
      << "    \"itopk_size\": " << search_params.itopk_size << "\n"
      << "  },\n"
      << "  \"recall\": " << recall << ",\n"
      << "  \"timings_ms\": {\n"
      << "    \"bruteforce\": " << t.bruteforce_ms << ",\n"
      << "    \"part0_search\": " << t.part0_search_ms << ",\n"
      << "    \"part1_search\": " << t.part1_search_ms << ",\n"
      << "    \"merge\": " << t.merge_ms << ",\n"
      << "    \"total_split_search\": " << t.total_split_search_ms << "\n"
      << "  }\n"
      << "}\n";
}

void write_metrics(const std::filesystem::path& path,
                   int64_t topk,
                   double recall,
                   const timings& t)
{
  std::ofstream out(path);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  out << "topk,recall,bruteforce_ms,part0_search_ms,part1_search_ms,merge_ms,"
         "total_split_search_ms\n";
  out << std::fixed << std::setprecision(6) << topk << ',' << recall << ','
      << t.bruteforce_ms << ',' << t.part0_search_ms << ',' << t.part1_search_ms << ','
      << t.merge_ms << ',' << t.total_split_search_ms << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto opts    = parse_args(argc, argv);
    auto cmdline = command_line(argc, argv);

    raft::device_resources res;
    rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                          1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(pool_mr);

    auto dataset = read_fbin(opts.dataset_path);
    auto queries = read_fbin(opts.queries_path);

    if (dataset.dim != queries.dim) {
      throw std::runtime_error("Dataset and query dimensions do not match");
    }
    if (dataset.rows < 2) { throw std::runtime_error("Dataset must contain at least two rows"); }
    if (opts.topk > static_cast<int64_t>(dataset.rows / 2)) {
      throw std::runtime_error("--topk must be <= the smaller partition size");
    }
    if (dataset.rows > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("CAGRA uint32 IDs require <= uint32 max rows");
    }

    std::vector<uint32_t> permutation(dataset.rows);
    std::iota(permutation.begin(), permutation.end(), uint32_t{0});
    std::mt19937_64 rng(opts.seed);
    std::shuffle(permutation.begin(), permutation.end(), rng);

    const uint32_t part0_rows = dataset.rows / 2;
    const uint32_t part1_rows = dataset.rows - part0_rows;
    std::vector<uint32_t> part0_ids(permutation.begin(), permutation.begin() + part0_rows);
    std::vector<uint32_t> part1_ids(permutation.begin() + part0_rows, permutation.end());

    auto part0_host = gather_rows(dataset, part0_ids);
    auto part1_host = gather_rows(dataset, part1_ids);
    auto concat_host = concatenate(part0_host, part1_host);

    auto queries_dev = copy_to_device(res, queries);
    auto part0_dev   = copy_to_device(res, part0_host, part0_rows, dataset.dim);
    auto part1_dev   = copy_to_device(res, part1_host, part1_rows, dataset.dim);
    auto concat_dev  = copy_to_device(res, concat_host, dataset.rows, dataset.dim);

    auto part0_index_params = make_index_params(opts, part0_rows, dataset.dim);
    auto part1_index_params = make_index_params(opts, part1_rows, dataset.dim);

    cuvs::neighbors::cagra::search_params search_params;
    if (opts.itopk_size) { search_params.itopk_size = *opts.itopk_size; }

    std::cout << "Building CAGRA index for part_000 (" << part0_rows
              << " rows, graph_build_algo=" << graph_build_algo_to_string(opts.build_algo)
              << ")\n";
    auto part0_index = cuvs::neighbors::cagra::build(
      res, part0_index_params, raft::make_const_mdspan(part0_dev.view()));

    std::cout << "Building CAGRA index for part_001 (" << part1_rows
              << " rows, graph_build_algo=" << graph_build_algo_to_string(opts.build_algo)
              << ")\n";
    auto part1_index = cuvs::neighbors::cagra::build(
      res, part1_index_params, raft::make_const_mdspan(part1_dev.view()));

    auto bf_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, queries.rows, opts.topk);
    auto bf_distances = raft::make_device_matrix<float, int64_t>(res, queries.rows, opts.topk);

    cuvs::neighbors::brute_force::index_params bf_params;
    bf_params.metric = cuvs::distance::DistanceType::L2Expanded;
    auto bf_index =
      cuvs::neighbors::brute_force::build(res, bf_params, raft::make_const_mdspan(concat_dev.view()));

    timings t;
    std::cout << "Searching brute-force full partition-concatenated dataset\n";
    t.bruteforce_ms = time_cuda(res, [&] {
      cuvs::neighbors::brute_force::search(res,
                                           cuvs::neighbors::brute_force::search_params{},
                                           bf_index,
                                           raft::make_const_mdspan(queries_dev.view()),
                                           bf_neighbors.view(),
                                           bf_distances.view());
    });

    auto part0_neighbors =
      raft::make_device_matrix<uint32_t, int64_t>(res, queries.rows, opts.topk);
    auto part1_neighbors =
      raft::make_device_matrix<uint32_t, int64_t>(res, queries.rows, opts.topk);
    auto part0_distances =
      raft::make_device_matrix<float, int64_t>(res, queries.rows, opts.topk);
    auto part1_distances =
      raft::make_device_matrix<float, int64_t>(res, queries.rows, opts.topk);

    auto split_start = clock_type::now();
    std::cout << "Searching part_000\n";
    t.part0_search_ms = time_cuda(res, [&] {
      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     part0_index,
                                     raft::make_const_mdspan(queries_dev.view()),
                                     part0_neighbors.view(),
                                     part0_distances.view());
    });

    std::cout << "Searching part_001\n";
    t.part1_search_ms = time_cuda(res, [&] {
      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     part1_index,
                                     raft::make_const_mdspan(queries_dev.view()),
                                     part1_neighbors.view(),
                                     part1_distances.view());
    });

    auto in_neighbors =
      raft::make_device_matrix<uint32_t, int64_t>(res, int64_t(queries.rows) * 2, opts.topk);
    auto in_distances =
      raft::make_device_matrix<float, int64_t>(res, int64_t(queries.rows) * 2, opts.topk);
    auto merged_neighbors =
      raft::make_device_matrix<uint32_t, int64_t>(res, queries.rows, opts.topk);
    auto merged_distances =
      raft::make_device_matrix<float, int64_t>(res, queries.rows, opts.topk);
    auto translations = raft::make_device_vector<uint32_t, int64_t>(res, 2);

    std::vector<uint32_t> translations_host{0, part0_rows};
    raft::copy(translations.data_handle(),
               translations_host.data(),
               translations_host.size(),
               raft::resource::get_cuda_stream(res));
    raft::copy(in_neighbors.data_handle(),
               part0_neighbors.data_handle(),
               part0_neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::copy(in_neighbors.data_handle() + part0_neighbors.size(),
               part1_neighbors.data_handle(),
               part1_neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::copy(in_distances.data_handle(),
               part0_distances.data_handle(),
               part0_distances.size(),
               raft::resource::get_cuda_stream(res));
    raft::copy(in_distances.data_handle() + part0_distances.size(),
               part1_distances.data_handle(),
               part1_distances.size(),
               raft::resource::get_cuda_stream(res));

    std::cout << "Merging partition result sets\n";
    t.merge_ms = time_cuda(res, [&] {
      cuvs::neighbors::knn_merge_parts(res,
                                       raft::make_const_mdspan(in_distances.view()),
                                       raft::make_const_mdspan(in_neighbors.view()),
                                       merged_distances.view(),
                                       merged_neighbors.view(),
                                       translations.view());
    });
    t.total_split_search_ms = elapsed_ms(split_start, clock_type::now());

    auto bf_neighbors_host = copy_device_matrix_to_host<int64_t>(res, bf_neighbors.view());
    auto merged_neighbors_host =
      copy_device_matrix_to_host<uint32_t>(res, merged_neighbors.view());
    double recall = recall_at_k(bf_neighbors_host, merged_neighbors_host, queries.rows, opts.topk);

    std::filesystem::path output_dir(opts.output_dir);
    std::filesystem::create_directories(output_dir / "part_000");
    std::filesystem::create_directories(output_dir / "part_001");

    write_fbin(output_dir / "queries.fbin", queries.rows, queries.dim, queries.data);
    write_fbin(output_dir / "part_000" / "dataset.fbin", part0_rows, dataset.dim, part0_host);
    write_fbin(output_dir / "part_001" / "dataset.fbin", part1_rows, dataset.dim, part1_host);
    write_ibin(output_dir / "part_000" / "original_ids.ibin", part0_ids);
    write_ibin(output_dir / "part_001" / "original_ids.ibin", part1_ids);

    cuvs::neighbors::cagra::serialize(
      res, (output_dir / "part_000" / "index.cag").string(), part0_index, true);
    cuvs::neighbors::cagra::serialize(
      res, (output_dir / "part_001" / "index.cag").string(), part1_index, true);

    write_partition_metadata(output_dir / "part_000" / "metadata.json",
                             opts,
                             part0_rows,
                             dataset.dim,
                             0,
                             0,
                             part0_index_params,
                             search_params);
    write_partition_metadata(output_dir / "part_001" / "metadata.json",
                             opts,
                             part1_rows,
                             dataset.dim,
                             part0_rows,
                             1,
                             part1_index_params,
                             search_params);
    write_manifest(output_dir / "manifest.json",
                   opts,
                   cmdline,
                   dataset,
                   queries,
                   part0_rows,
                   part1_rows,
                   part0_index_params,
                   search_params,
                   recall,
                   t);
    write_metrics(output_dir / "metrics.csv", opts.topk, recall, t);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "recall@" << opts.topk << " = " << recall << "\n";
    std::cout << "wrote artifacts to " << output_dir << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n\n" << usage();
    return 1;
  }

  return 0;
}

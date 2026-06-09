/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <neighbors/detail/cagra/graph_core.cuh>
#include <neighbors/detail/nn_descent.cuh>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();

enum class candidate_strategy { none, random_global, query_one, query_all };
enum class insert_mode { replace, append };
enum class replace_policy { random, farthest, nearest };
enum class scratch_algo { nn_descent, ivf_pq, auto_select };

struct options {
  std::string split_dir = "/raid/blandrum/split-wiki";
  std::string groundtruth_path;
  std::string output_csv;
  std::string label = "run";
  int64_t topk = 12;
  size_t graph_degree = 64;
  size_t intermediate_graph_degree = 128;
  size_t candidate_count = 32;
  double sample_rate = 1.0;
  double replace_fraction = 0.25;
  uint64_t seed = 1234;
  size_t nnd_iterations = 20;
  size_t itopk_size = 64;
  size_t candidate_itopk_size = 0;
  size_t query_batch_size = 10000;
  size_t rows_per_part = 0;
  size_t query_count = 0;
  size_t part_start = 0;
  size_t part_count_limit = 0;
  bool skip_scratch = false;
  bool skip_variant = false;
  bool skip_nnd = false;
  bool skip_optimize = false;
  candidate_strategy candidates = candidate_strategy::random_global;
  insert_mode insertion = insert_mode::append;
  replace_policy replacement = replace_policy::random;
  scratch_algo scratch = scratch_algo::nn_descent;
};

struct fbin_matrix {
  uint32_t rows = 0;
  uint32_t dim  = 0;
  std::vector<float> data;
};

struct ibin_matrix {
  uint32_t rows = 0;
  uint32_t dim  = 0;
  std::vector<int32_t> data;
};

struct split_part {
  std::string name;
  std::filesystem::path dir;
  fbin_matrix data;
  std::vector<uint32_t> original_ids;
  size_t global_offset = 0;
};

struct loaded_split {
  std::vector<split_part> parts;
  fbin_matrix queries;
  uint32_t dim = 0;
};

size_t part_count(const loaded_split& split)
{
  return split.parts.size();
}

size_t total_rows(const loaded_split& split)
{
  size_t total = 0;
  for (const auto& part : split.parts) {
    total += part.data.rows;
  }
  return total;
}

struct graph_copy {
  std::vector<uint32_t> graph;
  size_t rows = 0;
  size_t degree = 0;
};

struct variant_timings {
  double load_graph_ms = 0.0;
  double candidate_ms  = 0.0;
  double seed_ms       = 0.0;
  double nnd_ms        = 0.0;
  double sort_ms       = 0.0;
  double optimize_ms   = 0.0;
  double index_ms      = 0.0;
  double search_ms     = 0.0;
};

struct run_metrics {
  double bf_ms             = 0.0;
  double scratch_build_ms  = std::numeric_limits<double>::quiet_NaN();
  double scratch_search_ms = std::numeric_limits<double>::quiet_NaN();
  double scratch_recall    = std::numeric_limits<double>::quiet_NaN();
  double variant_recall    = std::numeric_limits<double>::quiet_NaN();
  variant_timings variant;
};

std::string usage()
{
  return R"(Usage:
  CAGRA_SPLIT_MERGE_CONSTRUCTION [options]

Options:
  --split-dir <dir>              Split artifacts dir (default /raid/blandrum/split-wiki)
  --groundtruth <neighbors.ibin> Use precomputed source-ID ground truth instead of brute force
  --output-csv <file>            Append one CSV row with metrics
  --label <name>                 Run label
  --topk <int>                   Recall/search k (default 12)
  --graph-degree <int>           CAGRA optimized graph degree (default 64)
  --intermediate-graph-degree <int>
                                 NND/CAGRA input graph degree (default 128)
  --candidate-count <int>        Candidates generated per modified row (default 32)
  --candidate-strategy <none|random-global|query-one|query-all>
  --insert-mode <replace|append>
  --replace-policy <random|farthest|nearest>
  --replace-fraction <float>     Fraction of base neighbors to replace (default 0.25)
  --sample-rate <float>          Fraction of rows receiving generated candidates (default 1.0)
  --seed <uint64>                Random seed (default 1234)
  --nnd-iterations <int>         NN-Descent iterations (default 20)
  --itopk-size <int>             Final CAGRA search itopk_size (default 64)
  --candidate-itopk-size <int>   Candidate-generation CAGRA search itopk_size
  --query-batch-size <int>       Candidate-generation query batch size (default 10000)
  --rows-per-part <int>          Use first N rows from each part for quick sanity runs
  --query-count <int>            Use first N queries for quick sanity runs
  --part-start <int>             First sorted part_* directory to load (default 0)
  --part-count <int>             Number of sorted part_* directories to load (default all)
  --scratch-algo <nn-descent|ivf-pq|auto>
  --skip-scratch                 Do not build/search from-scratch CAGRA baseline
  --skip-variant                 Only run brute force and optional scratch baseline
  --skip-nnd                     Sort/optimize the mixed seed graph directly
  --skip-optimize                Search the mixed graph without CAGRA optimize
)";
}

uint64_t parse_u64(const std::string& value, const std::string& flag)
{
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') { throw std::runtime_error("Invalid " + flag); }
  return static_cast<uint64_t>(parsed);
}

double parse_f64(const std::string& value, const std::string& flag)
{
  char* end    = nullptr;
  double value_ = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') { throw std::runtime_error("Invalid " + flag); }
  return value_;
}

candidate_strategy parse_candidate_strategy(const std::string& value)
{
  if (value == "none") return candidate_strategy::none;
  if (value == "random-global" || value == "random") return candidate_strategy::random_global;
  if (value == "query-one" || value == "one-index") return candidate_strategy::query_one;
  if (value == "query-all" || value == "all-indices") return candidate_strategy::query_all;
  throw std::runtime_error("Invalid --candidate-strategy");
}

insert_mode parse_insert_mode(const std::string& value)
{
  if (value == "replace") return insert_mode::replace;
  if (value == "append") return insert_mode::append;
  throw std::runtime_error("Invalid --insert-mode");
}

replace_policy parse_replace_policy(const std::string& value)
{
  if (value == "random") return replace_policy::random;
  if (value == "farthest") return replace_policy::farthest;
  if (value == "nearest") return replace_policy::nearest;
  throw std::runtime_error("Invalid --replace-policy");
}

scratch_algo parse_scratch_algo(const std::string& value)
{
  if (value == "nn-descent" || value == "nn_descent") return scratch_algo::nn_descent;
  if (value == "ivf-pq" || value == "ivf_pq") return scratch_algo::ivf_pq;
  if (value == "auto") return scratch_algo::auto_select;
  throw std::runtime_error("Invalid --scratch-algo");
}

std::string to_string(candidate_strategy value)
{
  switch (value) {
    case candidate_strategy::none: return "none";
    case candidate_strategy::random_global: return "random-global";
    case candidate_strategy::query_one: return "query-one";
    case candidate_strategy::query_all: return "query-all";
  }
  return "unknown";
}

std::string to_string(insert_mode value)
{
  return value == insert_mode::replace ? "replace" : "append";
}

std::string to_string(replace_policy value)
{
  switch (value) {
    case replace_policy::random: return "random";
    case replace_policy::farthest: return "farthest";
    case replace_policy::nearest: return "nearest";
  }
  return "unknown";
}

std::string to_string(scratch_algo value)
{
  switch (value) {
    case scratch_algo::nn_descent: return "nn-descent";
    case scratch_algo::ivf_pq: return "ivf-pq";
    case scratch_algo::auto_select: return "auto";
  }
  return "unknown";
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

    if (arg == "--split-dir") {
      opts.split_dir = need_value(arg);
    } else if (arg == "--groundtruth") {
      opts.groundtruth_path = need_value(arg);
    } else if (arg == "--output-csv") {
      opts.output_csv = need_value(arg);
    } else if (arg == "--label") {
      opts.label = need_value(arg);
    } else if (arg == "--topk") {
      opts.topk = static_cast<int64_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--graph-degree") {
      opts.graph_degree = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--intermediate-graph-degree") {
      opts.intermediate_graph_degree = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--candidate-count") {
      opts.candidate_count = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--candidate-strategy") {
      opts.candidates = parse_candidate_strategy(need_value(arg));
    } else if (arg == "--insert-mode") {
      opts.insertion = parse_insert_mode(need_value(arg));
    } else if (arg == "--replace-policy") {
      opts.replacement = parse_replace_policy(need_value(arg));
    } else if (arg == "--replace-fraction") {
      opts.replace_fraction = parse_f64(need_value(arg), arg);
    } else if (arg == "--sample-rate") {
      opts.sample_rate = parse_f64(need_value(arg), arg);
    } else if (arg == "--seed") {
      opts.seed = parse_u64(need_value(arg), arg);
    } else if (arg == "--nnd-iterations") {
      opts.nnd_iterations = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--itopk-size") {
      opts.itopk_size = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--candidate-itopk-size") {
      opts.candidate_itopk_size = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--query-batch-size") {
      opts.query_batch_size = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--rows-per-part") {
      opts.rows_per_part = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--query-count") {
      opts.query_count = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--part-start") {
      opts.part_start = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--part-count") {
      opts.part_count_limit = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--scratch-algo") {
      opts.scratch = parse_scratch_algo(need_value(arg));
    } else if (arg == "--skip-scratch") {
      opts.skip_scratch = true;
    } else if (arg == "--skip-variant") {
      opts.skip_variant = true;
    } else if (arg == "--skip-nnd") {
      opts.skip_nnd = true;
    } else if (arg == "--skip-optimize") {
      opts.skip_optimize = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << usage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.topk <= 0) { throw std::runtime_error("--topk must be positive"); }
  if (opts.graph_degree == 0 || opts.intermediate_graph_degree == 0) {
    throw std::runtime_error("Graph degrees must be positive");
  }
  if (opts.intermediate_graph_degree < opts.graph_degree) {
    throw std::runtime_error("--intermediate-graph-degree must be >= --graph-degree");
  }
  if (opts.sample_rate < 0.0 || opts.sample_rate > 1.0) {
    throw std::runtime_error("--sample-rate must be in [0, 1]");
  }
  if (opts.replace_fraction < 0.0 || opts.replace_fraction > 1.0) {
    throw std::runtime_error("--replace-fraction must be in [0, 1]");
  }
  if (opts.query_batch_size == 0) { throw std::runtime_error("--query-batch-size must be > 0"); }
  if (opts.output_csv.empty()) {
    opts.output_csv = (std::filesystem::path(opts.split_dir) / "merge_construction_results.csv").string();
  }
  return opts;
}

void read_exact(std::istream& in, char* dst, std::streamsize bytes, const std::string& path)
{
  in.read(dst, bytes);
  if (in.gcount() != bytes) { throw std::runtime_error("Unexpected EOF while reading " + path); }
}

void read_large(std::istream& in, char* dst, size_t bytes, const std::string& path)
{
  constexpr size_t chunk_bytes = 1ull << 30;
  size_t done = 0;
  while (done < bytes) {
    auto chunk = static_cast<std::streamsize>(std::min(chunk_bytes, bytes - done));
    in.read(dst + done, chunk);
    if (in.gcount() != chunk) { throw std::runtime_error("Unexpected EOF while reading " + path); }
    done += static_cast<size_t>(chunk);
  }
}

fbin_matrix read_fbin(const std::filesystem::path& path,
                      std::optional<size_t> row_limit = std::nullopt)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("Could not open " + path.string()); }

  fbin_matrix matrix;
  read_exact(in, reinterpret_cast<char*>(&matrix.rows), sizeof(uint32_t), path.string());
  read_exact(in, reinterpret_cast<char*>(&matrix.dim), sizeof(uint32_t), path.string());

  uint32_t rows_to_read = matrix.rows;
  if (row_limit.has_value()) {
    rows_to_read = std::min<uint32_t>(rows_to_read, static_cast<uint32_t>(*row_limit));
  }
  const size_t count = static_cast<size_t>(rows_to_read) * matrix.dim;
  matrix.data.resize(count);
  read_large(in, reinterpret_cast<char*>(matrix.data.data()), count * sizeof(float), path.string());
  matrix.rows = rows_to_read;
  return matrix;
}

ibin_matrix read_ibin(const std::filesystem::path& path,
                      std::optional<size_t> row_limit = std::nullopt)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("Could not open " + path.string()); }

  ibin_matrix matrix;
  read_exact(in, reinterpret_cast<char*>(&matrix.rows), sizeof(uint32_t), path.string());
  read_exact(in, reinterpret_cast<char*>(&matrix.dim), sizeof(uint32_t), path.string());

  uint32_t rows_to_read = matrix.rows;
  if (row_limit.has_value()) {
    rows_to_read = std::min<uint32_t>(rows_to_read, static_cast<uint32_t>(*row_limit));
  }
  const size_t count = static_cast<size_t>(rows_to_read) * matrix.dim;
  matrix.data.resize(count);
  read_large(in, reinterpret_cast<char*>(matrix.data.data()), count * sizeof(int32_t), path.string());
  matrix.rows = rows_to_read;
  return matrix;
}

std::vector<uint32_t> read_original_ids(const std::filesystem::path& path,
                                        std::optional<size_t> row_limit = std::nullopt)
{
  auto matrix = read_ibin(path, row_limit);
  if (matrix.dim != 1) { throw std::runtime_error("original_ids.ibin must have dim=1"); }
  std::vector<uint32_t> ids(matrix.rows);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (matrix.data[i] < 0) { throw std::runtime_error("original_ids.ibin contains negative IDs"); }
    ids[i] = static_cast<uint32_t>(matrix.data[i]);
  }
  return ids;
}

bool is_partition_dir(const std::filesystem::directory_entry& entry)
{
  if (!entry.is_directory()) return false;
  auto name = entry.path().filename().string();
  if (name.rfind("part_", 0) != 0 || name.size() <= 5) return false;
  return std::all_of(name.begin() + 5, name.end(), [](char c) { return c >= '0' && c <= '9'; });
}

std::vector<std::filesystem::path> partition_dirs(const std::filesystem::path& root)
{
  std::vector<std::filesystem::path> dirs;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (is_partition_dir(entry)) { dirs.push_back(entry.path()); }
  }
  std::sort(dirs.begin(), dirs.end());
  if (dirs.size() < 2) { throw std::runtime_error("Need at least two part_* directories"); }
  return dirs;
}

loaded_split load_split(const options& opts)
{
  std::filesystem::path root(opts.split_dir);
  std::optional<size_t> rows_limit;
  if (opts.rows_per_part > 0) { rows_limit = opts.rows_per_part; }
  std::optional<size_t> query_limit;
  if (opts.query_count > 0) { query_limit = opts.query_count; }

  loaded_split split;
  split.queries = read_fbin(root / "queries.fbin", query_limit);
  split.dim     = split.queries.dim;

  auto dirs = partition_dirs(root);
  if (opts.part_start >= dirs.size()) { throw std::runtime_error("--part-start is out of range"); }
  size_t end_part = dirs.size();
  if (opts.part_count_limit > 0) {
    end_part = std::min(end_part, opts.part_start + opts.part_count_limit);
  }
  if (end_part - opts.part_start < 2) { throw std::runtime_error("Need at least two selected parts"); }

  size_t offset = 0;
  for (size_t part_id = opts.part_start; part_id < end_part; ++part_id) {
    const auto& dir = dirs[part_id];
    split_part part;
    part.name          = dir.filename().string();
    part.dir           = dir;
    part.global_offset = offset;
    part.data          = read_fbin(dir / "dataset.fbin", rows_limit);
    part.original_ids  = read_original_ids(dir / "original_ids.ibin", rows_limit);
    if (part.data.dim != split.dim) { throw std::runtime_error("Dataset/query dimensions do not match"); }
    if (part.original_ids.size() != part.data.rows) {
      throw std::runtime_error("original_ids.ibin row count does not match dataset.fbin");
    }
    offset += part.data.rows;
    split.parts.push_back(std::move(part));
  }
  return split;
}

std::vector<float> concatenate(const loaded_split& split)
{
  std::vector<float> out;
  size_t values = 0;
  for (const auto& part : split.parts) {
    values += part.data.data.size();
  }
  out.reserve(values);
  for (const auto& part : split.parts) {
    out.insert(out.end(), part.data.data.begin(), part.data.data.end());
  }
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

raft::device_matrix<float, int64_t> copy_to_device(raft::device_resources const& res,
                                                   const fbin_matrix& matrix)
{
  return copy_to_device(res, matrix.data, matrix.rows, matrix.dim);
}

template <typename T, typename ViewT>
std::vector<T> copy_device_view_to_host(raft::device_resources const& res, ViewT view)
{
  std::vector<T> host(view.size());
  raft::copy(host.data(), view.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
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

uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

bool selected_row(uint64_t seed, size_t row, double sample_rate)
{
  if (sample_rate >= 1.0) return true;
  if (sample_rate <= 0.0) return false;
  constexpr double denom = static_cast<double>(std::numeric_limits<uint64_t>::max());
  double u = static_cast<double>(splitmix64(seed ^ (row * 0x9e3779b97f4a7c15ull))) / denom;
  return u < sample_rate;
}

std::vector<uint8_t> make_selection(const options& opts, size_t rows)
{
  std::vector<uint8_t> selected(rows, 0);
  for (size_t row = 0; row < rows; ++row) {
    selected[row] = selected_row(opts.seed, row, opts.sample_rate) ? 1 : 0;
  }
  return selected;
}

size_t count_selected(const std::vector<uint8_t>& selected)
{
  return static_cast<size_t>(std::accumulate(selected.begin(), selected.end(), uint64_t{0}));
}

std::vector<size_t> selected_local_rows(const std::vector<uint8_t>& selected,
                                        size_t global_offset,
                                        size_t rows)
{
  std::vector<size_t> out;
  for (size_t local = 0; local < rows; ++local) {
    if (selected[global_offset + local]) { out.push_back(local); }
  }
  return out;
}

graph_copy load_partition_graph(raft::device_resources const& res,
                                const std::filesystem::path& index_path,
                                size_t rows_limit)
{
  cuvs::neighbors::cagra::index<float, uint32_t> index(res);
  cuvs::neighbors::cagra::deserialize(res, index_path.string(), &index);
  auto graph_view = index.graph();
  size_t rows     = std::min<size_t>(rows_limit, graph_view.extent(0));
  size_t degree   = static_cast<size_t>(graph_view.extent(1));
  auto full_graph = copy_device_view_to_host<uint32_t>(res, graph_view);

  graph_copy out;
  out.rows   = rows;
  out.degree = degree;
  out.graph.resize(rows * degree);
  for (size_t row = 0; row < rows; ++row) {
    std::copy_n(full_graph.data() + row * degree, degree, out.graph.data() + row * degree);
  }
  return out;
}

std::vector<uint32_t> make_empty_candidates(size_t rows, size_t candidate_count)
{
  return std::vector<uint32_t>(rows * candidate_count, invalid_id);
}

void generate_random_candidates(const options& opts,
                                size_t rows,
                                std::vector<uint32_t>& candidates,
                                const std::vector<uint8_t>& selected)
{
  if (opts.candidate_count == 0) return;
#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    if (!selected[row]) continue;
    for (size_t c = 0; c < opts.candidate_count; ++c) {
      uint64_t value = splitmix64(opts.seed ^ (row * 0xd6e8feb86659fd93ull) ^
                                  (c * 0xa0761d6478bd642full));
      uint32_t candidate = static_cast<uint32_t>(value % rows);
      if (candidate == row) { candidate = (candidate + 1) % rows; }
      candidates[row * opts.candidate_count + c] = candidate;
    }
  }
}

void fill_query_batch(const fbin_matrix& source,
                      const std::vector<size_t>& selected_rows,
                      size_t begin,
                      size_t rows,
                      std::vector<float>& batch)
{
  batch.resize(rows * source.dim);
  for (size_t out_row = 0; out_row < rows; ++out_row) {
    size_t src_row = selected_rows[begin + out_row];
    std::copy_n(source.data.data() + src_row * source.dim,
                source.dim,
                batch.data() + out_row * source.dim);
  }
}

void insert_candidate(std::vector<uint32_t>& candidates,
                      std::vector<float>& candidate_dists,
                      size_t candidate_count,
                      size_t row,
                      uint32_t candidate,
                      float dist)
{
  if (candidate_count == 0 || candidate == invalid_id) return;
  auto base = row * candidate_count;
  for (size_t c = 0; c < candidate_count; ++c) {
    if (candidates[base + c] == candidate) {
      if (dist < candidate_dists[base + c]) { candidate_dists[base + c] = dist; }
      return;
    }
  }

  size_t slot = 0;
  bool found_empty = false;
  float worst_dist = -std::numeric_limits<float>::infinity();
  for (size_t c = 0; c < candidate_count; ++c) {
    if (candidates[base + c] == invalid_id) {
      slot = c;
      found_empty = true;
      break;
    }
    if (candidate_dists[base + c] > worst_dist) {
      worst_dist = candidate_dists[base + c];
      slot = c;
    }
  }
  if (found_empty || dist < worst_dist) {
    candidates[base + slot] = candidate;
    candidate_dists[base + slot] = dist;
  }
}

void query_partition_with_index(
  raft::device_resources const& res,
  const options& opts,
  const cuvs::neighbors::cagra::index<float, uint32_t>& target_index,
  const fbin_matrix& query_part,
  size_t query_global_offset,
  size_t target_global_offset,
  size_t target_rows_limit,
  const std::vector<size_t>& selected_rows,
  std::vector<uint32_t>& candidates,
  std::vector<float>& candidate_dists)
{
  if (selected_rows.empty() || opts.candidate_count == 0) return;

  cuvs::neighbors::cagra::search_params search_params;
  size_t candidate_itopk = opts.candidate_itopk_size == 0 ? opts.itopk_size : opts.candidate_itopk_size;
  search_params.itopk_size = std::max(candidate_itopk, opts.candidate_count);

  std::vector<float> batch_host;
  for (size_t begin = 0; begin < selected_rows.size(); begin += opts.query_batch_size) {
    size_t rows = std::min(opts.query_batch_size, selected_rows.size() - begin);
    fill_query_batch(query_part, selected_rows, begin, rows, batch_host);
    auto batch_dev = copy_to_device(res, batch_host, static_cast<uint32_t>(rows), query_part.dim);
    auto neigh     = raft::make_device_matrix<uint32_t, int64_t>(res, rows, opts.candidate_count);
    auto dist      = raft::make_device_matrix<float, int64_t>(res, rows, opts.candidate_count);

    cuvs::neighbors::cagra::search(res,
                                   search_params,
                                   target_index,
                                   raft::make_const_mdspan(batch_dev.view()),
                                   neigh.view(),
                                   dist.view());
    auto neigh_host = copy_device_view_to_host<uint32_t>(res, neigh.view());
    auto dist_host  = copy_device_view_to_host<float>(res, dist.view());
    for (size_t out_row = 0; out_row < rows; ++out_row) {
      size_t local_row  = selected_rows[begin + out_row];
      size_t global_row = query_global_offset + local_row;
      for (size_t c = 0; c < opts.candidate_count; ++c) {
        uint32_t local_candidate = neigh_host[out_row * opts.candidate_count + c];
        if (local_candidate < target_rows_limit) {
          auto global_candidate = static_cast<uint32_t>(target_global_offset + local_candidate);
          insert_candidate(candidates,
                           candidate_dists,
                           opts.candidate_count,
                           global_row,
                           global_candidate,
                           dist_host[out_row * opts.candidate_count + c]);
        }
      }
    }
  }
}

void generate_query_candidates(raft::device_resources const& res,
                               const options& opts,
                               const loaded_split& split,
                               const std::vector<uint8_t>& selected,
                               std::vector<uint32_t>& candidates)
{
  if (opts.candidate_count == 0) return;

  std::vector<std::vector<size_t>> selected_by_part;
  selected_by_part.reserve(split.parts.size());
  for (const auto& part : split.parts) {
    selected_by_part.push_back(selected_local_rows(selected, part.global_offset, part.data.rows));
  }

  std::vector<float> candidate_dists(candidates.size(), std::numeric_limits<float>::infinity());
  if (opts.candidates == candidate_strategy::query_one) {
    for (size_t source_id = 0; source_id < split.parts.size(); ++source_id) {
      size_t target_id = (source_id + 1) % split.parts.size();
      const auto& source = split.parts[source_id];
      const auto& target = split.parts[target_id];
      cuvs::neighbors::cagra::index<float, uint32_t> target_index(res);
      cuvs::neighbors::cagra::deserialize(res, (target.dir / "index.cag").string(), &target_index);
      query_partition_with_index(res,
                                 opts,
                                 target_index,
                                 source.data,
                                 source.global_offset,
                                 target.global_offset,
                                 target.data.rows,
                                 selected_by_part[source_id],
                                 candidates,
                                 candidate_dists);
    }
    return;
  }

  for (size_t target_id = 0; target_id < split.parts.size(); ++target_id) {
    const auto& target = split.parts[target_id];
    cuvs::neighbors::cagra::index<float, uint32_t> target_index(res);
    cuvs::neighbors::cagra::deserialize(res, (target.dir / "index.cag").string(), &target_index);
    for (size_t source_id = 0; source_id < split.parts.size(); ++source_id) {
      if (source_id == target_id) continue;
      const auto& source = split.parts[source_id];
      query_partition_with_index(res,
                                 opts,
                                 target_index,
                                 source.data,
                                 source.global_offset,
                                 target.global_offset,
                                 target.data.rows,
                                 selected_by_part[source_id],
                                 candidates,
                                 candidate_dists);
    }
  }
}

float l2_distance(const std::vector<float>& dataset, size_t dim, size_t lhs, size_t rhs)
{
  const float* a = dataset.data() + lhs * dim;
  const float* b = dataset.data() + rhs * dim;
  float acc = 0.0f;
  for (size_t d = 0; d < dim; ++d) {
    float diff = a[d] - b[d];
    acc += diff * diff;
  }
  return acc;
}

bool contains_id(const uint32_t* row, size_t degree, uint32_t id)
{
  for (size_t i = 0; i < degree; ++i) {
    if (row[i] == id) return true;
  }
  return false;
}

std::vector<uint32_t> unique_valid_candidates(const std::vector<uint32_t>& candidates,
                                              size_t row,
                                              size_t candidate_count,
                                              size_t total_rows,
                                              const uint32_t* existing,
                                              size_t existing_degree)
{
  std::vector<uint32_t> out;
  out.reserve(candidate_count);
  for (size_t c = 0; c < candidate_count; ++c) {
    uint32_t id = candidates[row * candidate_count + c];
    if (id == invalid_id || id >= total_rows || id == row) continue;
    if (contains_id(existing, existing_degree, id)) continue;
    if (std::find(out.begin(), out.end(), id) != out.end()) continue;
    out.push_back(id);
  }
  return out;
}

void fill_base_seed_rows(const graph_copy& graph,
                         size_t global_offset,
                         size_t local_rows_limit,
                         size_t seed_degree,
                         std::vector<uint32_t>& seed_graph)
{
  for (size_t row = 0; row < local_rows_limit; ++row) {
    uint32_t* dst = seed_graph.data() + (global_offset + row) * seed_degree;
    for (size_t j = 0; j < graph.degree; ++j) {
      uint32_t local = graph.graph[row * graph.degree + j];
      if (local < local_rows_limit) { dst[j] = static_cast<uint32_t>(global_offset + local); }
    }
  }
}

std::vector<size_t> replacement_positions(const options& opts,
                                          const std::vector<float>& dataset,
                                          size_t dim,
                                          size_t row,
                                          const uint32_t* existing,
                                          size_t base_degree,
                                          size_t replace_count)
{
  std::vector<size_t> positions(base_degree);
  std::iota(positions.begin(), positions.end(), size_t{0});
  if (opts.replacement == replace_policy::random) {
    std::sort(positions.begin(), positions.end(), [&](size_t lhs, size_t rhs) {
      auto l = splitmix64(opts.seed ^ (row * 0x9e3779b97f4a7c15ull) ^ (lhs * 0xbf58476d1ce4e5b9ull));
      auto r = splitmix64(opts.seed ^ (row * 0x9e3779b97f4a7c15ull) ^ (rhs * 0xbf58476d1ce4e5b9ull));
      return l < r;
    });
  } else {
    std::sort(positions.begin(), positions.end(), [&](size_t lhs, size_t rhs) {
      float lhs_dist = existing[lhs] == invalid_id ? std::numeric_limits<float>::infinity()
                                                   : l2_distance(dataset, dim, row, existing[lhs]);
      float rhs_dist = existing[rhs] == invalid_id ? std::numeric_limits<float>::infinity()
                                                   : l2_distance(dataset, dim, row, existing[rhs]);
      if (opts.replacement == replace_policy::nearest) { return lhs_dist < rhs_dist; }
      return lhs_dist > rhs_dist;
    });
  }
  positions.resize(replace_count);
  return positions;
}

std::vector<uint32_t> build_seed_graph(const options& opts,
                                       const loaded_split& split,
                                       const std::vector<graph_copy>& part_graphs,
                                       const std::vector<uint32_t>& candidates,
                                       const std::vector<uint8_t>& selected,
                                       const std::vector<float>& combined_host,
                                       size_t& seed_degree)
{
  if (part_graphs.size() != split.parts.size()) {
    throw std::runtime_error("Partition graph count does not match split parts");
  }
  if (part_graphs.empty()) { throw std::runtime_error("No partition graphs loaded"); }
  size_t base_degree = part_graphs.front().degree;
  for (const auto& graph : part_graphs) {
    if (graph.degree != base_degree) { throw std::runtime_error("Partition graph degrees differ"); }
  }
  size_t rows_total = total_rows(split);
  seed_degree       = base_degree + (opts.insertion == insert_mode::append ? opts.candidate_count : 0);
  seed_degree       = std::max<size_t>(seed_degree, 1);

  std::vector<uint32_t> seed_graph(rows_total * seed_degree, invalid_id);
  for (size_t part_id = 0; part_id < split.parts.size(); ++part_id) {
    fill_base_seed_rows(part_graphs[part_id],
                        split.parts[part_id].global_offset,
                        split.parts[part_id].data.rows,
                        seed_degree,
                        seed_graph);
  }

  if (opts.candidates == candidate_strategy::none || opts.candidate_count == 0) { return seed_graph; }

  size_t replace_count = static_cast<size_t>(std::ceil(base_degree * opts.replace_fraction));
  replace_count        = std::min(replace_count, base_degree);

#pragma omp parallel for
  for (size_t row = 0; row < rows_total; ++row) {
    if (!selected[row]) continue;
    uint32_t* existing = seed_graph.data() + row * seed_degree;
    auto unique = unique_valid_candidates(
      candidates, row, opts.candidate_count, rows_total, existing, base_degree);
    if (unique.empty()) continue;

    if (opts.insertion == insert_mode::append) {
      size_t n_append = std::min(unique.size(), opts.candidate_count);
      for (size_t c = 0; c < n_append; ++c) {
        existing[base_degree + c] = unique[c];
      }
    } else {
      size_t n_replace = std::min(unique.size(), replace_count);
      if (n_replace == 0) continue;
      auto positions = replacement_positions(
        opts, combined_host, split.dim, row, existing, base_degree, n_replace);
      for (size_t i = 0; i < n_replace; ++i) {
        existing[positions[i]] = unique[i];
      }
    }
  }

  return seed_graph;
}

raft::host_matrix<uint32_t, int64_t> seed_to_host_graph(const std::vector<uint32_t>& seed_graph,
                                                        size_t rows,
                                                        size_t seed_degree,
                                                        uint64_t seed)
{
  auto graph = raft::make_host_matrix<uint32_t, int64_t, raft::row_major>(
    static_cast<int64_t>(rows), static_cast<int64_t>(seed_degree));
#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < seed_degree; ++col) {
      uint32_t id = seed_graph[row * seed_degree + col];
      if (id == invalid_id || id >= rows || id == row) {
        uint32_t fallback = static_cast<uint32_t>(
          splitmix64(seed ^ (row * 0xd6e8feb86659fd93ull) ^ (col * 0xa0761d6478bd642full)) % rows);
        if (fallback == row) { fallback = (fallback + 1) % rows; }
        id = fallback;
      }
      graph(row, col) = id;
    }
  }
  return graph;
}

template <typename DatasetView>
raft::host_matrix<uint32_t, int64_t> build_seeded_nnd_graph(
  raft::device_resources const& res,
  const options& opts,
  DatasetView dataset,
  const std::vector<uint32_t>& seed_graph,
  size_t seed_degree,
  double& nnd_ms,
  double& sort_ms)
{
  cuvs::neighbors::nn_descent::index_params nnd_params(opts.intermediate_graph_degree,
                                                       cuvs::distance::DistanceType::L2Expanded);
  nnd_params.graph_degree              = opts.intermediate_graph_degree;
  nnd_params.intermediate_graph_degree = opts.intermediate_graph_degree;
  nnd_params.max_iterations            = opts.nnd_iterations;
  nnd_params.return_distances          = false;

  size_t extended_graph_degree = 0;
  size_t graph_degree          = 0;
  auto build_config = cuvs::neighbors::nn_descent::detail::get_build_config(
    res,
    nnd_params,
    static_cast<size_t>(dataset.extent(0)),
    static_cast<size_t>(dataset.extent(1)),
    nnd_params.metric,
    extended_graph_degree,
    graph_degree);

  std::vector<int> seed_int(seed_graph.size());
#pragma omp parallel for
  for (size_t i = 0; i < seed_graph.size(); ++i) {
    uint32_t id = seed_graph[i];
    seed_int[i] = (id == invalid_id || id > static_cast<uint32_t>(std::numeric_limits<int>::max()))
                    ? std::numeric_limits<int>::max()
                    : static_cast<int>(id);
  }

  auto int_graph = raft::make_host_matrix<int, int64_t, raft::row_major>(
    dataset.extent(0), static_cast<int64_t>(extended_graph_degree));
  cuvs::neighbors::nn_descent::detail::GNND<const float, int> nnd(res, build_config);
  auto empty_distances = raft::make_device_matrix<float, int64_t>(res, 0, 0);

  nnd_ms = time_cuda(res, [&] {
    nnd.build(dataset.data_handle(),
              static_cast<int>(dataset.extent(0)),
              int_graph.data_handle(),
              false,
              empty_distances.data_handle(),
              seed_int.data(),
              seed_degree);
  });

  auto knn_graph = raft::make_host_matrix<uint32_t, int64_t, raft::row_major>(
    dataset.extent(0), static_cast<int64_t>(opts.intermediate_graph_degree));
#pragma omp parallel for
  for (size_t row = 0; row < static_cast<size_t>(dataset.extent(0)); ++row) {
    for (size_t col = 0; col < opts.intermediate_graph_degree; ++col) {
      int id = int_graph(row, col);
      knn_graph(row, col) = id < 0 ? 0u : static_cast<uint32_t>(id);
    }
  }

  sort_ms = time_cuda(res, [&] {
    cuvs::neighbors::cagra::detail::graph::sort_knn_graph(
      res, cuvs::distance::DistanceType::L2Expanded, dataset, knn_graph.view());
  });

  return knn_graph;
}

double recall_at_k(const std::vector<int64_t>& expected,
                   const std::vector<uint32_t>& actual,
                   int64_t rows,
                   int64_t k)
{
  uint64_t matches = 0;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t actual_col = 0; actual_col < k; ++actual_col) {
      auto actual_id = static_cast<int64_t>(actual[static_cast<size_t>(row * k + actual_col)]);
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

std::vector<int64_t> groundtruth_neighbors(const options& opts,
                                           const loaded_split& split,
                                           double& load_map_ms,
                                           size_t& missing_expected)
{
  auto start = clock_type::now();
  std::optional<size_t> query_limit;
  if (opts.query_count > 0) { query_limit = opts.query_count; }
  auto gt = read_ibin(opts.groundtruth_path, query_limit);
  if (gt.rows != split.queries.rows) {
    throw std::runtime_error("Ground truth row count does not match query count");
  }
  if (gt.dim < static_cast<uint32_t>(opts.topk)) {
    throw std::runtime_error("Ground truth dim must be >= topk");
  }

  uint32_t max_original_id = 0;
  for (const auto& part : split.parts) {
    for (uint32_t id : part.original_ids) {
      max_original_id = std::max(max_original_id, id);
    }
  }
  std::vector<int64_t> original_to_global(static_cast<size_t>(max_original_id) + 1, -1);
  for (const auto& part : split.parts) {
    for (size_t local = 0; local < part.original_ids.size(); ++local) {
      uint32_t original = part.original_ids[local];
      original_to_global[original] = static_cast<int64_t>(part.global_offset + local);
    }
  }

  std::vector<int64_t> expected(static_cast<size_t>(gt.rows) * static_cast<size_t>(opts.topk), -1);
  missing_expected = 0;
  for (size_t row = 0; row < gt.rows; ++row) {
    for (int64_t col = 0; col < opts.topk; ++col) {
      int32_t original = gt.data[row * gt.dim + static_cast<size_t>(col)];
      int64_t mapped = -1;
      if (original >= 0 && static_cast<size_t>(original) < original_to_global.size()) {
        mapped = original_to_global[static_cast<size_t>(original)];
      }
      if (mapped < 0) { ++missing_expected; }
      expected[row * static_cast<size_t>(opts.topk) + static_cast<size_t>(col)] = mapped;
    }
  }
  load_map_ms = elapsed_ms(start, clock_type::now());
  return expected;
}

std::vector<int64_t> brute_force_neighbors(raft::device_resources const& res,
                                           raft::device_matrix<float, int64_t>& dataset,
                                           raft::device_matrix<float, int64_t>& queries,
                                           int64_t topk,
                                           double& bf_ms)
{
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, queries.extent(0), topk);
  auto distances = raft::make_device_matrix<float, int64_t>(res, queries.extent(0), topk);
  cuvs::neighbors::brute_force::index_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;
  auto index = cuvs::neighbors::brute_force::build(
    res, params, raft::make_const_mdspan(dataset.view()));
  bf_ms = time_cuda(res, [&] {
    cuvs::neighbors::brute_force::search(res,
                                         cuvs::neighbors::brute_force::search_params{},
                                         index,
                                         raft::make_const_mdspan(queries.view()),
                                         neighbors.view(),
                                         distances.view());
  });
  return copy_device_view_to_host<int64_t>(res, neighbors.view());
}

std::vector<uint32_t> search_index(raft::device_resources const& res,
                                   const options& opts,
                                   const cuvs::neighbors::cagra::index<float, uint32_t>& index,
                                   raft::device_matrix<float, int64_t>& queries,
                                   double& search_ms)
{
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, queries.extent(0), opts.topk);
  auto distances = raft::make_device_matrix<float, int64_t>(res, queries.extent(0), opts.topk);
  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size = opts.itopk_size;
  search_ms = time_cuda(res, [&] {
    cuvs::neighbors::cagra::search(res,
                                   search_params,
                                   index,
                                   raft::make_const_mdspan(queries.view()),
                                   neighbors.view(),
                                   distances.view());
  });
  return copy_device_view_to_host<uint32_t>(res, neighbors.view());
}

cuvs::neighbors::cagra::index_params make_cagra_params(const options& opts)
{
  cuvs::neighbors::cagra::index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = opts.graph_degree;
  params.intermediate_graph_degree = opts.intermediate_graph_degree;
  params.attach_dataset_on_build   = true;
  switch (opts.scratch) {
    case scratch_algo::auto_select: break;
    case scratch_algo::ivf_pq:
      params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(0, 0), params.metric);
      break;
    case scratch_algo::nn_descent: {
      auto nnd = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
        opts.intermediate_graph_degree, params.metric);
      nnd.max_iterations   = opts.nnd_iterations;
      nnd.return_distances = false;
      params.graph_build_params = nnd;
      break;
    }
  }
  return params;
}

void configure_ivf_pq_shape(cuvs::neighbors::cagra::index_params& params,
                            int64_t rows,
                            int64_t dim)
{
  if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
        params.graph_build_params)) {
    params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
      raft::matrix_extent<int64_t>(rows, dim), params.metric);
  }
}

void append_csv(const options& opts,
                const loaded_split& split,
                size_t selected_rows,
                size_t seed_degree,
                const run_metrics& metrics)
{
  bool exists = std::filesystem::exists(opts.output_csv);
  std::ofstream out(opts.output_csv, std::ios::app);
  if (!out) { throw std::runtime_error("Could not write " + opts.output_csv); }
  if (!exists) {
    out << "label,rows,queries,topk,graph_degree,intermediate_graph_degree,seed_degree,"
           "candidate_strategy,insert_mode,replace_policy,sample_rate,selected_rows,"
           "candidate_count,replace_fraction,nnd_iterations,itopk_size,scratch_algo,"
           "bf_ms,scratch_build_ms,scratch_search_ms,scratch_recall,"
           "load_graph_ms,candidate_ms,seed_ms,nnd_ms,sort_ms,optimize_ms,index_ms,"
           "variant_build_ms,variant_search_ms,variant_recall\n";
  }

  double variant_build_ms = metrics.variant.load_graph_ms + metrics.variant.candidate_ms +
                            metrics.variant.seed_ms + metrics.variant.nnd_ms +
                            metrics.variant.sort_ms + metrics.variant.optimize_ms +
                            metrics.variant.index_ms;
  out << std::fixed << std::setprecision(6)
      << opts.label << ',' << total_rows(split) << ','
      << split.queries.rows << ',' << opts.topk << ',' << opts.graph_degree << ','
      << opts.intermediate_graph_degree << ',' << seed_degree << ',' << to_string(opts.candidates)
      << ',' << to_string(opts.insertion) << ',' << to_string(opts.replacement) << ','
      << opts.sample_rate << ',' << selected_rows << ',' << opts.candidate_count << ','
      << opts.replace_fraction << ',' << opts.nnd_iterations << ',' << opts.itopk_size << ','
      << to_string(opts.scratch) << ',' << metrics.bf_ms << ',' << metrics.scratch_build_ms << ','
      << metrics.scratch_search_ms << ',' << metrics.scratch_recall << ','
      << metrics.variant.load_graph_ms << ',' << metrics.variant.candidate_ms << ','
      << metrics.variant.seed_ms << ',' << metrics.variant.nnd_ms << ',' << metrics.variant.sort_ms
      << ',' << metrics.variant.optimize_ms << ',' << metrics.variant.index_ms << ','
      << variant_build_ms << ',' << metrics.variant.search_ms << ',' << metrics.variant_recall
      << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto opts = parse_args(argc, argv);

    raft::device_resources res;
    rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                          1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(pool_mr);

    std::cout << "Loading split data from " << opts.split_dir << "\n";
    auto split = load_split(opts);
    size_t rows_total = total_rows(split);
    if (opts.topk > static_cast<int64_t>(rows_total)) {
      throw std::runtime_error("--topk must be <= total rows");
    }
    auto combined_host = concatenate(split);
    auto selected      = make_selection(opts, rows_total);
    size_t n_selected  = count_selected(selected);

    std::cout << "Rows=" << rows_total << " parts=" << part_count(split)
              << " queries=" << split.queries.rows << " selected=" << n_selected << "\n";

    auto combined_dev = copy_to_device(
      res, combined_host, static_cast<uint32_t>(rows_total), split.dim);
    auto queries_dev = copy_to_device(res, split.queries);

    run_metrics metrics;
    std::vector<int64_t> expected_neighbors;
    if (!opts.groundtruth_path.empty()) {
      size_t missing_expected = 0;
      std::cout << "Loading precomputed ground truth from " << opts.groundtruth_path << "\n";
      expected_neighbors = groundtruth_neighbors(opts, split, metrics.bf_ms, missing_expected);
      if (missing_expected > 0) {
        std::cout << "groundtruth_missing_expected=" << missing_expected << "\n";
      }
    } else {
      std::cout << "Building brute-force ground truth\n";
      expected_neighbors = brute_force_neighbors(res, combined_dev, queries_dev, opts.topk, metrics.bf_ms);
    }

    if (!opts.skip_scratch) {
      std::cout << "Building from-scratch CAGRA baseline (" << to_string(opts.scratch) << ")\n";
      auto params = make_cagra_params(opts);
      configure_ivf_pq_shape(params, combined_dev.extent(0), combined_dev.extent(1));
      cuvs::neighbors::cagra::index<float, uint32_t> scratch_index(res);
      metrics.scratch_build_ms = time_cuda(res, [&] {
        scratch_index = cuvs::neighbors::cagra::build(
          res, params, raft::make_const_mdspan(combined_dev.view()));
      });
      auto scratch_neighbors = search_index(res, opts, scratch_index, queries_dev, metrics.scratch_search_ms);
      metrics.scratch_recall = recall_at_k(expected_neighbors, scratch_neighbors, split.queries.rows, opts.topk);
    }

    size_t seed_degree = 0;
    if (!opts.skip_variant) {
      std::vector<graph_copy> part_graphs;
      part_graphs.reserve(split.parts.size());
      metrics.variant.load_graph_ms = elapsed_ms(clock_type::now(), clock_type::now());
      auto load_start = clock_type::now();
      std::cout << "Loading " << split.parts.size() << " partition CAGRA graphs\n";
      for (const auto& part : split.parts) {
        part_graphs.push_back(load_partition_graph(res, part.dir / "index.cag", part.data.rows));
      }
      metrics.variant.load_graph_ms = elapsed_ms(load_start, clock_type::now());

      auto candidates = make_empty_candidates(rows_total, opts.candidate_count);
      auto candidate_start = clock_type::now();
      if (opts.candidates == candidate_strategy::random_global) {
        std::cout << "Generating random global candidates\n";
        generate_random_candidates(opts, rows_total, candidates, selected);
      } else if (opts.candidates == candidate_strategy::query_one ||
                 opts.candidates == candidate_strategy::query_all) {
        std::cout << "Generating query candidates across partition indexes\n";
        generate_query_candidates(res, opts, split, selected, candidates);
      }
      raft::resource::sync_stream(res);
      metrics.variant.candidate_ms = elapsed_ms(candidate_start, clock_type::now());

      std::cout << "Building seeded initial graph\n";
      auto seed_start = clock_type::now();
      auto seed_graph =
        build_seed_graph(opts, split, part_graphs, candidates, selected, combined_host, seed_degree);
      metrics.variant.seed_ms = elapsed_ms(seed_start, clock_type::now());

      auto knn_graph = raft::make_host_matrix<uint32_t, int64_t, raft::row_major>(0, 0);
      if (opts.skip_nnd) {
        knn_graph = seed_to_host_graph(seed_graph, rows_total, seed_degree, opts.seed);
        if (opts.skip_optimize) {
          std::cout << "Skipping NN-Descent and CAGRA optimize; searching mixed seed graph directly\n";
        } else {
          std::cout << "Skipping NN-Descent; sorting mixed seed graph directly\n";
          metrics.variant.sort_ms = time_cuda(res, [&] {
            cuvs::neighbors::cagra::detail::graph::sort_knn_graph(
              res,
              cuvs::distance::DistanceType::L2Expanded,
              raft::make_const_mdspan(combined_dev.view()),
              knn_graph.view());
          });
        }
      } else {
        std::cout << "Running seeded NN-Descent\n";
        knn_graph = build_seeded_nnd_graph(res,
                                           opts,
                                           raft::make_const_mdspan(combined_dev.view()),
                                           seed_graph,
                                           seed_degree,
                                           metrics.variant.nnd_ms,
                                           metrics.variant.sort_ms);
      }

      std::cout << "Constructing searchable merged index\n";
      std::optional<cuvs::neighbors::cagra::index<float, uint32_t>> merged_index;
      if (opts.skip_optimize) {
        metrics.variant.index_ms = time_cuda(res, [&] {
          merged_index.emplace(res,
                               cuvs::distance::DistanceType::L2Expanded,
                               raft::make_const_mdspan(combined_dev.view()),
                               raft::make_const_mdspan(knn_graph.view()));
        });
      } else {
        std::cout << "Optimizing CAGRA graph\n";
        auto optimized_graph = raft::make_host_matrix<uint32_t, int64_t, raft::row_major>(
          static_cast<int64_t>(rows_total), static_cast<int64_t>(opts.graph_degree));
        metrics.variant.optimize_ms = time_cuda(res, [&] {
          cuvs::neighbors::cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());
        });
        metrics.variant.index_ms = time_cuda(res, [&] {
          merged_index.emplace(res,
                               cuvs::distance::DistanceType::L2Expanded,
                               raft::make_const_mdspan(combined_dev.view()),
                               raft::make_const_mdspan(optimized_graph.view()));
        });
      }
      auto merged_neighbors = search_index(res, opts, *merged_index, queries_dev, metrics.variant.search_ms);
      metrics.variant_recall = recall_at_k(expected_neighbors, merged_neighbors, split.queries.rows, opts.topk);
    }

    append_csv(opts, split, n_selected, seed_degree, metrics);

    double variant_build_ms = metrics.variant.load_graph_ms + metrics.variant.candidate_ms +
                              metrics.variant.seed_ms + metrics.variant.nnd_ms +
                              metrics.variant.sort_ms + metrics.variant.optimize_ms +
                              metrics.variant.index_ms;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "bf_ms=" << metrics.bf_ms << "\n";
    if (!opts.skip_scratch) {
      std::cout << "scratch_build_ms=" << metrics.scratch_build_ms
                << " scratch_search_ms=" << metrics.scratch_search_ms
                << " scratch_recall@" << opts.topk << "=" << metrics.scratch_recall << "\n";
    }
    if (!opts.skip_variant) {
      std::cout << "variant_build_ms=" << variant_build_ms
                << " variant_search_ms=" << metrics.variant.search_ms
                << " variant_recall@" << opts.topk << "=" << metrics.variant_recall << "\n";
      std::cout << "variant components: load_graph=" << metrics.variant.load_graph_ms
                << " candidate=" << metrics.variant.candidate_ms
                << " seed=" << metrics.variant.seed_ms << " nnd=" << metrics.variant.nnd_ms
                << " sort=" << metrics.variant.sort_ms
                << " optimize=" << metrics.variant.optimize_ms
                << " index=" << metrics.variant.index_ms << "\n";
    }
    std::cout << "wrote " << opts.output_csv << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n\n" << usage();
    return 1;
  }
  return 0;
}

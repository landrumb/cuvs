/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

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

constexpr std::size_t io_chunk_bytes = 1ull << 30;

enum class graph_build_algo { auto_select, ivf_pq, nn_descent };

struct options {
  std::string dataset_path;
  std::string queries_path;
  std::string output_dir;
  uint32_t parts = 10;
  uint64_t seed  = 1234;
  graph_build_algo build_algo = graph_build_algo::ivf_pq;
  std::optional<size_t> graph_degree;
  std::optional<size_t> intermediate_graph_degree;
  std::optional<size_t> itopk_size;
  bool force = false;
};

struct fbin_header {
  uint32_t rows = 0;
  uint32_t dim  = 0;
};

struct fbin_matrix {
  uint32_t rows = 0;
  uint32_t dim  = 0;
  std::vector<float> data;
};

struct partition_info {
  std::string name;
  uint32_t rows = 0;
  uint32_t local_id_start = 0;
  double gather_ms = 0.0;
  double write_dataset_ms = 0.0;
  double build_index_ms = 0.0;
  double serialize_index_ms = 0.0;
};

std::string graph_build_algo_to_string(graph_build_algo algo)
{
  switch (algo) {
    case graph_build_algo::auto_select: return "auto";
    case graph_build_algo::ivf_pq: return "ivf-pq";
    case graph_build_algo::nn_descent: return "nn-descent";
  }
  return "unknown";
}

std::string usage()
{
  return R"(Usage:
  CAGRA_RANDOM_KWAY_SPLIT_BUILD \
    --dataset <base.fbin> \
    --queries <queries.fbin> \
    --output <dir> \
    [--parts <int>] \
    [--seed <uint64>] \
    [--graph-build-algo auto|ivf-pq|nn-descent] \
    [--graph-degree <int>] \
    [--intermediate-graph-degree <int>] \
    [--itopk-size <int>] \
    [--force]
)";
}

uint64_t parse_u64(const std::string& value, const std::string& flag)
{
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') { throw std::runtime_error("Invalid " + flag); }
  return static_cast<uint64_t>(parsed);
}

graph_build_algo parse_graph_build_algo(const std::string& value)
{
  if (value == "auto") return graph_build_algo::auto_select;
  if (value == "ivf-pq" || value == "ivf_pq" || value == "IVF_PQ") return graph_build_algo::ivf_pq;
  if (value == "nn-descent" || value == "nn_descent" || value == "NN_DESCENT") {
    return graph_build_algo::nn_descent;
  }
  throw std::runtime_error("Invalid --graph-build-algo");
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
    } else if (arg == "--parts") {
      opts.parts = static_cast<uint32_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--seed") {
      opts.seed = parse_u64(need_value(arg), arg);
    } else if (arg == "--graph-degree") {
      opts.graph_degree = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--intermediate-graph-degree") {
      opts.intermediate_graph_degree = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--itopk-size") {
      opts.itopk_size = static_cast<size_t>(parse_u64(need_value(arg), arg));
    } else if (arg == "--graph-build-algo") {
      opts.build_algo = parse_graph_build_algo(need_value(arg));
    } else if (arg == "--force") {
      opts.force = true;
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
  if (opts.parts < 2) { throw std::runtime_error("--parts must be >= 2"); }
  return opts;
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

template <typename Fn>
double time_host(Fn&& fn)
{
  auto start = clock_type::now();
  fn();
  return elapsed_ms(start, clock_type::now());
}

void read_exact(std::istream& in, char* dst, std::streamsize bytes, const std::string& path)
{
  in.read(dst, bytes);
  if (in.gcount() != bytes) { throw std::runtime_error("Unexpected EOF while reading " + path); }
}

void read_large(std::istream& in, char* dst, std::size_t bytes, const std::string& path)
{
  std::size_t done = 0;
  while (done < bytes) {
    auto chunk = static_cast<std::streamsize>(std::min(io_chunk_bytes, bytes - done));
    in.read(dst + done, chunk);
    if (in.gcount() != chunk) { throw std::runtime_error("Unexpected EOF while reading " + path); }
    done += static_cast<std::size_t>(chunk);
  }
}

void write_large(std::ostream& out, const char* src, std::size_t bytes, const std::string& path)
{
  std::size_t done = 0;
  while (done < bytes) {
    auto chunk = static_cast<std::streamsize>(std::min(io_chunk_bytes, bytes - done));
    out.write(src + done, chunk);
    if (!out) { throw std::runtime_error("Failed while writing " + path); }
    done += static_cast<std::size_t>(chunk);
  }
}

fbin_header read_fbin_header(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("Could not open " + path); }
  fbin_header header;
  read_exact(in, reinterpret_cast<char*>(&header.rows), sizeof(uint32_t), path);
  read_exact(in, reinterpret_cast<char*>(&header.dim), sizeof(uint32_t), path);
  return header;
}

fbin_matrix read_fbin(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("Could not open " + path); }

  fbin_matrix matrix;
  read_exact(in, reinterpret_cast<char*>(&matrix.rows), sizeof(uint32_t), path);
  read_exact(in, reinterpret_cast<char*>(&matrix.dim), sizeof(uint32_t), path);

  const std::size_t count = static_cast<std::size_t>(matrix.rows) * matrix.dim;
  matrix.data.resize(count);
  read_large(in, reinterpret_cast<char*>(matrix.data.data()), count * sizeof(float), path);
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
  write_large(out,
              reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(float),
              path.string());
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

std::vector<float> gather_rows(const fbin_matrix& matrix, const std::vector<uint32_t>& row_ids)
{
  std::vector<float> out(static_cast<std::size_t>(row_ids.size()) * matrix.dim);
#pragma omp parallel for
  for (std::size_t local_row = 0; local_row < row_ids.size(); ++local_row) {
    const auto src = static_cast<std::size_t>(row_ids[local_row]) * matrix.dim;
    const auto dst = local_row * matrix.dim;
    std::copy_n(matrix.data.data() + src, matrix.dim, out.data() + dst);
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

std::string part_name(uint32_t part)
{
  std::ostringstream out;
  out << "part_" << std::setw(3) << std::setfill('0') << part;
  return out.str();
}

void write_partition_metadata(const std::filesystem::path& path,
                              const options& opts,
                              uint32_t rows,
                              uint32_t dim,
                              uint32_t local_start,
                              uint32_t part_id,
                              const cuvs::neighbors::cagra::index_params& index_params,
                              const cuvs::neighbors::cagra::search_params& search_params,
                              const partition_info& info)
{
  std::ofstream out(path);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  out << std::fixed << std::setprecision(6);
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
      << "  \"split_policy\": \"seeded_random_kway\",\n"
      << "  \"graph_build_algo\": \"" << graph_build_algo_to_string(opts.build_algo) << "\",\n"
      << "  \"index_params\": {\n"
      << "    \"graph_degree\": " << index_params.graph_degree << ",\n"
      << "    \"intermediate_graph_degree\": " << index_params.intermediate_graph_degree << ",\n"
      << "    \"attach_dataset_on_build\": true\n"
      << "  },\n"
      << "  \"search_params\": {\n"
      << "    \"itopk_size\": " << search_params.itopk_size << "\n"
      << "  },\n"
      << "  \"timings_ms\": {\n"
      << "    \"gather_rows\": " << info.gather_ms << ",\n"
      << "    \"write_dataset\": " << info.write_dataset_ms << ",\n"
      << "    \"build_index\": " << info.build_index_ms << ",\n"
      << "    \"serialize_index\": " << info.serialize_index_ms << "\n"
      << "  }\n"
      << "}\n";
}

void write_manifest(const std::filesystem::path& path,
                    const options& opts,
                    const std::string& cmdline,
                    const fbin_matrix& dataset,
                    const fbin_header& queries,
                    const cuvs::neighbors::cagra::index_params& index_params,
                    const cuvs::neighbors::cagra::search_params& search_params,
                    const std::vector<partition_info>& partitions,
                    double read_dataset_ms,
                    double shuffle_ms,
                    double copy_queries_ms,
                    double total_ms)
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
      << "  \"partition_count\": " << opts.parts << ",\n"
      << "  \"seed\": " << opts.seed << ",\n"
      << "  \"metric\": \"l2_expanded\",\n"
      << "  \"split_policy\": \"seeded_random_kway\",\n"
      << "  \"graph_build_algo\": \"" << graph_build_algo_to_string(opts.build_algo) << "\",\n"
      << "  \"partition_order\": [";
  for (std::size_t i = 0; i < partitions.size(); ++i) {
    if (i > 0) out << ", ";
    out << "\"" << partitions[i].name << "\"";
  }
  out << "],\n"
      << "  \"partitions\": [\n";
  for (std::size_t i = 0; i < partitions.size(); ++i) {
    const auto& p = partitions[i];
    out << "    {\"name\": \"" << p.name << "\", \"rows\": " << p.rows
        << ", \"local_id_start\": " << p.local_id_start << "}";
    out << (i + 1 == partitions.size() ? "\n" : ",\n");
  }
  out << "  ],\n"
      << "  \"index_params\": {\n"
      << "    \"graph_degree\": " << index_params.graph_degree << ",\n"
      << "    \"intermediate_graph_degree\": " << index_params.intermediate_graph_degree << ",\n"
      << "    \"attach_dataset_on_build\": true\n"
      << "  },\n"
      << "  \"search_params\": {\n"
      << "    \"itopk_size\": " << search_params.itopk_size << "\n"
      << "  },\n"
      << "  \"timings_ms\": {\n"
      << "    \"read_dataset\": " << read_dataset_ms << ",\n"
      << "    \"shuffle\": " << shuffle_ms << ",\n"
      << "    \"copy_queries\": " << copy_queries_ms << ",\n"
      << "    \"total\": " << total_ms << "\n"
      << "  }\n"
      << "}\n";
}

void write_metrics(const std::filesystem::path& path,
                   const std::vector<partition_info>& partitions,
                   double read_dataset_ms,
                   double shuffle_ms,
                   double copy_queries_ms,
                   double total_ms)
{
  std::ofstream out(path);
  if (!out) { throw std::runtime_error("Could not write " + path.string()); }
  out << "scope,part,rows,gather_ms,write_dataset_ms,build_index_ms,serialize_index_ms,"
         "read_dataset_ms,shuffle_ms,copy_queries_ms,total_ms\n";
  out << std::fixed << std::setprecision(6)
      << "all,,0,0,0,0,0," << read_dataset_ms << ',' << shuffle_ms << ',' << copy_queries_ms
      << ',' << total_ms << "\n";
  for (const auto& p : partitions) {
    out << "partition," << p.name << ',' << p.rows << ',' << p.gather_ms << ','
        << p.write_dataset_ms << ',' << p.build_index_ms << ',' << p.serialize_index_ms
        << ",0,0,0,0\n";
  }
}

}  // namespace

int main(int argc, char** argv)
{
  auto total_start = clock_type::now();
  try {
    auto opts    = parse_args(argc, argv);
    auto cmdline = command_line(argc, argv);

    std::filesystem::path output_dir(opts.output_dir);
    if (std::filesystem::exists(output_dir)) {
      if (!opts.force) {
        throw std::runtime_error(
          "Output directory already exists; pass --force to add/overwrite artifacts in it");
      }
    }
    std::filesystem::create_directories(output_dir);

    raft::device_resources res;
    rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                          1024 * 1024 * 1024ull);
    rmm::mr::set_current_device_resource(pool_mr);

    std::cout << "Reading dataset " << opts.dataset_path << "\n";
    fbin_matrix dataset;
    double read_dataset_ms = time_host([&] { dataset = read_fbin(opts.dataset_path); });
    auto queries = read_fbin_header(opts.queries_path);
    if (dataset.dim != queries.dim) {
      throw std::runtime_error("Dataset and query dimensions do not match");
    }
    if (opts.parts > dataset.rows) { throw std::runtime_error("--parts must be <= dataset rows"); }
    if (dataset.rows > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("CAGRA uint32 IDs require <= uint32 max rows");
    }

    std::cout << "Rows=" << dataset.rows << " dim=" << dataset.dim
              << " queries=" << queries.rows << " parts=" << opts.parts << "\n";

    std::vector<uint32_t> permutation(dataset.rows);
    double shuffle_ms = time_host([&] {
      std::iota(permutation.begin(), permutation.end(), uint32_t{0});
      std::mt19937_64 rng(opts.seed);
      std::shuffle(permutation.begin(), permutation.end(), rng);
    });

    double copy_queries_ms = time_host([&] {
      std::filesystem::copy_file(opts.queries_path,
                                 output_dir / "queries.fbin",
                                 std::filesystem::copy_options::overwrite_existing);
    });

    cuvs::neighbors::cagra::search_params search_params;
    if (opts.itopk_size) { search_params.itopk_size = *opts.itopk_size; }

    uint32_t base_rows = dataset.rows / opts.parts;
    uint32_t remainder = dataset.rows % opts.parts;
    uint32_t offset    = 0;
    std::vector<partition_info> partitions;
    partitions.reserve(opts.parts);

    for (uint32_t part = 0; part < opts.parts; ++part) {
      partition_info info;
      info.name           = part_name(part);
      info.rows           = base_rows + (part < remainder ? 1 : 0);
      info.local_id_start = offset;

      auto part_dir = output_dir / info.name;
      std::filesystem::create_directories(part_dir);

      std::vector<uint32_t> ids(permutation.begin() + offset,
                                permutation.begin() + offset + info.rows);
      std::vector<float> part_host;
      std::cout << "Preparing " << info.name << " rows=" << info.rows
                << " local_start=" << info.local_id_start << "\n";
      info.gather_ms = time_host([&] { part_host = gather_rows(dataset, ids); });
      info.write_dataset_ms = time_host([&] {
        write_fbin(part_dir / "dataset.fbin", info.rows, dataset.dim, part_host);
        write_ibin(part_dir / "original_ids.ibin", ids);
      });

      auto index_params = make_index_params(opts, info.rows, dataset.dim);
      {
        auto part_dev = copy_to_device(res, part_host, info.rows, dataset.dim);
        std::optional<cuvs::neighbors::cagra::index<float, uint32_t>> index;
        std::cout << "Building CAGRA index for " << info.name << " ("
                  << graph_build_algo_to_string(opts.build_algo) << ")\n";
        info.build_index_ms = time_cuda(res, [&] {
          index.emplace(cuvs::neighbors::cagra::build(
            res, index_params, raft::make_const_mdspan(part_dev.view())));
        });
        std::cout << "Serializing " << info.name << "\n";
        info.serialize_index_ms = time_cuda(res, [&] {
          cuvs::neighbors::cagra::serialize(
            res, (part_dir / "index.cag").string(), *index, true);
        });
      }

      write_partition_metadata(part_dir / "metadata.json",
                               opts,
                               info.rows,
                               dataset.dim,
                               info.local_id_start,
                               part,
                               index_params,
                               search_params,
                               info);
      partitions.push_back(info);
      offset += info.rows;

      std::cout << std::fixed << std::setprecision(3)
                << info.name << " done: gather=" << info.gather_ms / 1000.0
                << "s write=" << info.write_dataset_ms / 1000.0
                << "s build=" << info.build_index_ms / 1000.0
                << "s serialize=" << info.serialize_index_ms / 1000.0 << "s\n";
    }

    auto manifest_index_params = make_index_params(opts, partitions.front().rows, dataset.dim);
    double total_ms = elapsed_ms(total_start, clock_type::now());
    write_manifest(output_dir / "manifest.json",
                   opts,
                   cmdline,
                   dataset,
                   queries,
                   manifest_index_params,
                   search_params,
                   partitions,
                   read_dataset_ms,
                   shuffle_ms,
                   copy_queries_ms,
                   total_ms);
    write_metrics(output_dir / "metrics.csv",
                  partitions,
                  read_dataset_ms,
                  shuffle_ms,
                  copy_queries_ms,
                  total_ms);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "wrote " << opts.parts << "-way split artifacts to " << output_dir << "\n";
    std::cout << "total setup time " << total_ms / 1000.0 << "s\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n\n" << usage();
    return 1;
  }
  return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

namespace {

using clock_type = std::chrono::high_resolution_clock;

constexpr int64_t kSyntheticRows    = 100000;
constexpr int64_t kSyntheticDim     = 128;
constexpr int64_t kSyntheticQueries = 100;
constexpr int kDefaultMaxRows       = 100000;
constexpr int kDefaultMaxQueries    = 1000;
constexpr int64_t kDefaultTopk       = 12;
constexpr size_t kDefaultGraphDegree = 16;
constexpr size_t kDefaultIntermediateGraphDegree = 32;
constexpr char const* kDefaultBuildAlgo = "ivf_pq";

struct dataset_paths {
  std::filesystem::path base;
  std::filesystem::path query;
};

struct split_options {
  bool show_help{};
  int64_t topk{kDefaultTopk};
  size_t graph_degree{kDefaultGraphDegree};
  size_t intermediate_graph_degree{kDefaultIntermediateGraphDegree};
  std::string build_algo{kDefaultBuildAlgo};
  std::filesystem::path output_dir{"cagra_split_search_output"};
  std::vector<std::string> positional;
};

struct search_recall_result {
  double recall{};
  size_t match_count{};
  size_t total_count{};
  double mean_query_recall{};
  double min_query_recall{1.0};
};

struct run_metadata {
  std::string dataset_source;
  std::string base_path;
  std::string query_path;
  int64_t dataset_rows{};
  int64_t query_rows{};
  int64_t dim{};
  int64_t left_rows{};
  int64_t right_rows{};
  int64_t topk{};
  double left_build_seconds{};
  double right_build_seconds{};
  double left_search_seconds{};
  double right_search_seconds{};
  double combine_seconds{};
  double ground_truth_seconds{};
  search_recall_result recall;
};

double elapsed_seconds(clock_type::time_point start)
{
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

std::optional<dataset_paths> find_default_sift_dataset()
{
  constexpr char const* base_rel  = "sift-128-euclidean/base.fbin";
  constexpr char const* query_rel = "sift-128-euclidean/query.fbin";

  std::vector<std::filesystem::path> roots;
  if (const char* dataset_root = std::getenv("RAPIDS_DATASET_ROOT_DIR")) {
    roots.emplace_back(dataset_root);
  }

  roots.emplace_back("datasets");
  roots.emplace_back("../datasets");
  roots.emplace_back("../../datasets");
  roots.emplace_back("../../../datasets");

  for (auto const& root : roots) {
    auto base  = root / base_rel;
    auto query = root / query_rel;
    if (std::filesystem::exists(base) && std::filesystem::exists(query)) {
      return dataset_paths{base, query};
    }
  }

  return std::nullopt;
}

std::string json_escape(std::string const& text)
{
  std::ostringstream out;
  for (char c : text) {
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

std::string command_line(int argc, char* argv[])
{
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) { out << ' '; }
    out << argv[i];
  }
  return out.str();
}

void usage(char const* program)
{
  std::cout << "Usage: " << program
            << " [--output-dir DIR] [--topk K] [--graph-degree K]\n"
            << "       [--intermediate-graph-degree K] [--build-algo ALGO]\n"
            << "       [base.fbin query.fbin [max_rows] [max_queries]]\n\n"
            << "Builds two CAGRA indexes on contiguous halves of the input dataset. Searches both "
               "indexes independently, offsets the right-half neighbor ids, combines both result "
               "sets by distance, and reports recall@K against exact GPU flat search over the "
               "full dataset.\n\n"
            << "The output directory contains left/ and right/ subdirectories. Each subdirectory "
               "contains a serialized CAGRA index with its attached dataset included, so the two "
               "indexes can be deserialized and passed to cuvs::neighbors::cagra::merge(). A "
               "manifest.json records paths, split offsets, parameters, timings, and recall.\n\n"
            << "Options:\n"
            << "  --build-algo ALGO                 auto | ivf_pq | nn_descent (default "
            << kDefaultBuildAlgo << ")\n"
            << "  --graph-degree K                  output graph degree (default "
            << kDefaultGraphDegree << ")\n"
            << "  --intermediate-graph-degree K     pruning graph degree (default "
            << kDefaultIntermediateGraphDegree << ")\n";
}

bool parse_options(int argc, char* argv[], split_options& options)
{
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    }
    if (arg.rfind("--output-dir=", 0) == 0) {
      options.output_dir = arg.substr(13);
      continue;
    }
    if (arg == "--output-dir") {
      if (i + 1 >= argc) { return false; }
      options.output_dir = argv[++i];
      continue;
    }
    if (arg.rfind("--topk=", 0) == 0) {
      options.topk = std::stoll(arg.substr(7));
      continue;
    }
    if (arg == "--topk") {
      if (i + 1 >= argc) { return false; }
      options.topk = std::stoll(argv[++i]);
      continue;
    }
    if (arg.rfind("--graph-degree=", 0) == 0) {
      options.graph_degree = std::stoul(arg.substr(15));
      continue;
    }
    if (arg == "--graph-degree") {
      if (i + 1 >= argc) { return false; }
      options.graph_degree = std::stoul(argv[++i]);
      continue;
    }
    if (arg.rfind("--intermediate-graph-degree=", 0) == 0) {
      options.intermediate_graph_degree = std::stoul(arg.substr(28));
      continue;
    }
    if (arg == "--intermediate-graph-degree") {
      if (i + 1 >= argc) { return false; }
      options.intermediate_graph_degree = std::stoul(argv[++i]);
      continue;
    }
    if (arg.rfind("--build-algo=", 0) == 0) {
      options.build_algo = arg.substr(13);
      continue;
    }
    if (arg == "--build-algo") {
      if (i + 1 >= argc) { return false; }
      options.build_algo = argv[++i];
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      std::cerr << "Unknown option: " << arg << std::endl;
      return false;
    }
    options.positional.push_back(arg);
  }

  auto const n_positional = options.positional.size();
  if (n_positional == 1 || n_positional > 4) { return false; }
  if (options.topk <= 0) { return false; }
  if (options.graph_degree == 0 || options.intermediate_graph_degree == 0) { return false; }
  if (options.intermediate_graph_degree < options.graph_degree) { return false; }
  if (options.build_algo != "auto" && options.build_algo != "ivf_pq" &&
      options.build_algo != "nn_descent") {
    return false;
  }
  return true;
}

cuvs::neighbors::cagra::index_params make_index_params(split_options const& options)
{
  using namespace cuvs::neighbors;

  cagra::index_params index_params;
  index_params.graph_degree              = options.graph_degree;
  index_params.intermediate_graph_degree = options.intermediate_graph_degree;
  index_params.attach_dataset_on_build   = true;
  index_params.compression               = std::nullopt;

  if (options.build_algo == "ivf_pq") {
    index_params.graph_build_params = cagra::graph_build_params::ivf_pq_params();
  } else if (options.build_algo == "nn_descent") {
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(options.intermediate_graph_degree);
  }

  return index_params;
}

raft::device_matrix<float, int64_t> copy_dataset_rows(
  raft::device_resources const& dev_resources,
  raft::device_matrix_view<const float, int64_t> dataset,
  int64_t row_offset,
  int64_t n_rows)
{
  auto out = raft::make_device_matrix<float, int64_t>(dev_resources, n_rows, dataset.extent(1));
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(out.data_handle(),
             dataset.data_handle() + row_offset * dataset.extent(1),
             out.size(),
             stream);
  return out;
}

struct ground_truth_neighbors {
  std::vector<int64_t> host_neighbors;
  double seconds{};
};

ground_truth_neighbors compute_ground_truth(
  raft::device_resources const& dev_resources,
  raft::device_matrix_view<const float, int64_t> dataset,
  raft::device_matrix_view<const float, int64_t> queries,
  int64_t topk)
{
  using namespace cuvs::neighbors;

  auto start = clock_type::now();
  std::cout << "Computing exact GPU flat search ground truth (" << queries.extent(0)
            << " queries, topk=" << topk << ")" << std::endl;

  brute_force::index_params bf_index_params;
  auto bf_index = brute_force::build(dev_resources, bf_index_params, dataset);

  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources, queries.extent(0), topk);
  auto distances = raft::make_device_matrix<float, int64_t>(dev_resources, queries.extent(0), topk);
  brute_force::search_params bf_search_params;
  brute_force::search(dev_resources,
                      bf_search_params,
                      bf_index,
                      queries,
                      neighbors.view(),
                      distances.view());
  raft::resource::sync_stream(dev_resources);

  auto host_neighbors = raft::make_host_matrix<int64_t, int64_t>(queries.extent(0), topk);
  raft::copy(host_neighbors.data_handle(),
             neighbors.data_handle(),
             neighbors.size(),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  return {{host_neighbors.data_handle(), host_neighbors.data_handle() + host_neighbors.size()},
          elapsed_seconds(start)};
}

search_recall_result compute_recall(std::vector<uint32_t> const& actual_neighbors,
                                    std::vector<int64_t> const& expected_neighbors,
                                    int64_t n_queries,
                                    int64_t topk)
{
  auto stats = calc_recall(actual_neighbors, expected_neighbors, n_queries, topk);
  return {stats.recall,
          stats.match_count,
          stats.total_count,
          stats.mean_node_recall,
          stats.min_node_recall};
}

std::vector<uint32_t> combine_results(raft::device_resources const& dev_resources,
                                      raft::device_matrix_view<uint32_t, int64_t> left_neighbors,
                                      raft::device_matrix_view<float, int64_t> left_distances,
                                      raft::device_matrix_view<uint32_t, int64_t> right_neighbors,
                                      raft::device_matrix_view<float, int64_t> right_distances,
                                      uint32_t right_id_offset)
{
  int64_t const n_queries = left_neighbors.extent(0);
  int64_t const topk      = left_neighbors.extent(1);

  auto left_neighbors_host  = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  auto right_neighbors_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  auto left_distances_host  = raft::make_host_matrix<float, int64_t>(n_queries, topk);
  auto right_distances_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

  auto stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(left_neighbors_host.data_handle(),
             left_neighbors.data_handle(),
             left_neighbors.size(),
             stream);
  raft::copy(right_neighbors_host.data_handle(),
             right_neighbors.data_handle(),
             right_neighbors.size(),
             stream);
  raft::copy(left_distances_host.data_handle(),
             left_distances.data_handle(),
             left_distances.size(),
             stream);
  raft::copy(right_distances_host.data_handle(),
             right_distances.data_handle(),
             right_distances.size(),
             stream);
  raft::resource::sync_stream(dev_resources);

  std::vector<uint32_t> combined(static_cast<size_t>(n_queries * topk));
  std::vector<std::pair<float, uint32_t>> candidates(static_cast<size_t>(2 * topk));

  for (int64_t row = 0; row < n_queries; ++row) {
    for (int64_t col = 0; col < topk; ++col) {
      candidates[static_cast<size_t>(col)] = {left_distances_host(row, col),
                                              left_neighbors_host(row, col)};
      candidates[static_cast<size_t>(topk + col)] = {
        right_distances_host(row, col), right_neighbors_host(row, col) + right_id_offset};
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + topk,
                      candidates.end(),
                      [](auto const& a, auto const& b) { return a.first < b.first; });
    for (int64_t col = 0; col < topk; ++col) {
      combined[static_cast<size_t>(row * topk + col)] = candidates[static_cast<size_t>(col)].second;
    }
  }

  return combined;
}

void write_manifest(std::filesystem::path const& output_dir,
                    std::string const& command,
                    run_metadata const& metadata,
                    cuvs::neighbors::cagra::index_params const& index_params,
                    cuvs::neighbors::cagra::search_params const& search_params,
                    split_options const& options)
{
  auto manifest_path = output_dir / "manifest.json";
  std::ofstream out(manifest_path);
  if (!out) { throw std::runtime_error("Could not open manifest for writing: " + manifest_path.string()); }

  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"command\": \"" << json_escape(command) << "\",\n";
  out << "  \"dataset_source\": \"" << json_escape(metadata.dataset_source) << "\",\n";
  out << "  \"base_path\": \"" << json_escape(metadata.base_path) << "\",\n";
  out << "  \"query_path\": \"" << json_escape(metadata.query_path) << "\",\n";
  out << "  \"dataset\": {\n";
  out << "    \"rows\": " << metadata.dataset_rows << ",\n";
  out << "    \"queries\": " << metadata.query_rows << ",\n";
  out << "    \"dim\": " << metadata.dim << ",\n";
  out << "    \"split\": {\n";
  out << "      \"left\": {\"row_offset\": 0, \"rows\": " << metadata.left_rows
      << ", \"index_path\": \"left/cagra.index\"},\n";
  out << "      \"right\": {\"row_offset\": " << metadata.left_rows << ", \"rows\": "
      << metadata.right_rows << ", \"index_path\": \"right/cagra.index\"}\n";
  out << "    }\n";
  out << "  },\n";
  out << "  \"cagra_index_params\": {\n";
  out << "    \"metric\": " << static_cast<int>(index_params.metric) << ",\n";
  out << "    \"graph_degree\": " << index_params.graph_degree << ",\n";
  out << "    \"intermediate_graph_degree\": " << index_params.intermediate_graph_degree << ",\n";
  out << "    \"attach_dataset_on_build\": "
      << (index_params.attach_dataset_on_build ? "true" : "false") << ",\n";
  out << "    \"guarantee_connectivity\": "
      << (index_params.guarantee_connectivity ? "true" : "false") << ",\n";
  out << "    \"compression\": null,\n";
  out << "    \"graph_build_params\": \"" << json_escape(options.build_algo) << "\"\n";
  out << "  },\n";
  out << "  \"cagra_search_params\": {\n";
  out << "    \"topk\": " << metadata.topk << ",\n";
  out << "    \"itopk_size\": " << search_params.itopk_size << ",\n";
  out << "    \"max_queries\": " << search_params.max_queries << ",\n";
  out << "    \"max_iterations\": " << search_params.max_iterations << ",\n";
  out << "    \"search_width\": " << search_params.search_width << ",\n";
  out << "    \"num_random_samplings\": " << search_params.num_random_samplings << ",\n";
  out << "    \"rand_xor_mask\": " << search_params.rand_xor_mask << "\n";
  out << "  },\n";
  out << "  \"timings_seconds\": {\n";
  out << "    \"left_build\": " << metadata.left_build_seconds << ",\n";
  out << "    \"right_build\": " << metadata.right_build_seconds << ",\n";
  out << "    \"left_search\": " << metadata.left_search_seconds << ",\n";
  out << "    \"right_search\": " << metadata.right_search_seconds << ",\n";
  out << "    \"combine\": " << metadata.combine_seconds << ",\n";
  out << "    \"ground_truth\": " << metadata.ground_truth_seconds << "\n";
  out << "  },\n";
  out << "  \"combined_search_recall\": {\n";
  out << "    \"recall\": " << metadata.recall.recall << ",\n";
  out << "    \"matched\": " << metadata.recall.match_count << ",\n";
  out << "    \"total\": " << metadata.recall.total_count << ",\n";
  out << "    \"mean_query_recall\": " << metadata.recall.mean_query_recall << ",\n";
  out << "    \"min_query_recall\": " << metadata.recall.min_query_recall << "\n";
  out << "  },\n";
  out << "  \"merge_api_compatibility\": {\n";
  out << "    \"compatible\": true,\n";
  out << "    \"reason\": \"Each cagra.index file includes an attached uncompressed dataset with the "
         "same dimension. Deserialize both indexes and pass pointers to cagra::merge.\"\n";
  out << "  }\n";
  out << "}\n";
}

void run_split_search(raft::device_resources const& dev_resources,
                      raft::device_matrix_view<const float, int64_t> dataset,
                      raft::device_matrix_view<const float, int64_t> queries,
                      split_options const& options,
                      run_metadata& metadata,
                      std::string const& command)
{
  using namespace cuvs::neighbors;

  if (dataset.extent(0) < 2) { throw std::runtime_error("Dataset must contain at least two rows"); }
  int64_t const left_rows  = dataset.extent(0) / 2;
  int64_t const right_rows = dataset.extent(0) - left_rows;
  if (options.topk > std::min(left_rows, right_rows)) {
    throw std::runtime_error("topk cannot exceed the smaller split size");
  }

  metadata.dataset_rows = dataset.extent(0);
  metadata.query_rows   = queries.extent(0);
  metadata.dim          = dataset.extent(1);
  metadata.left_rows    = left_rows;
  metadata.right_rows   = right_rows;
  metadata.topk         = options.topk;

  std::filesystem::create_directories(options.output_dir / "left");
  std::filesystem::create_directories(options.output_dir / "right");

  std::cout << "Splitting dataset into left=" << left_rows << " rows and right=" << right_rows
            << " rows" << std::endl;
  auto left_dataset  = copy_dataset_rows(dev_resources, dataset, 0, left_rows);
  auto right_dataset = copy_dataset_rows(dev_resources, dataset, left_rows, right_rows);
  raft::resource::sync_stream(dev_resources);

  auto index_params = make_index_params(options);
  cagra::search_params search_params;

  std::cout << "Building left CAGRA index" << std::endl;
  auto start = clock_type::now();
  auto left_index =
    cagra::build(dev_resources, index_params, raft::make_const_mdspan(left_dataset.view()));
  raft::resource::sync_stream(dev_resources);
  metadata.left_build_seconds = elapsed_seconds(start);

  std::cout << "Building right CAGRA index" << std::endl;
  start = clock_type::now();
  auto right_index =
    cagra::build(dev_resources, index_params, raft::make_const_mdspan(right_dataset.view()));
  raft::resource::sync_stream(dev_resources);
  metadata.right_build_seconds = elapsed_seconds(start);

  auto const left_index_path  = options.output_dir / "left" / "cagra.index";
  auto const right_index_path = options.output_dir / "right" / "cagra.index";
  std::cout << "Saving left index to " << left_index_path << std::endl;
  cagra::serialize(dev_resources, left_index_path.string(), left_index, true);
  std::cout << "Saving right index to " << right_index_path << std::endl;
  cagra::serialize(dev_resources, right_index_path.string(), right_index, true);

  auto left_neighbors  = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, queries.extent(0), options.topk);
  auto right_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, queries.extent(0), options.topk);
  auto left_distances  = raft::make_device_matrix<float, int64_t>(dev_resources, queries.extent(0), options.topk);
  auto right_distances = raft::make_device_matrix<float, int64_t>(dev_resources, queries.extent(0), options.topk);

  std::cout << "Searching left index" << std::endl;
  start = clock_type::now();
  cagra::search(dev_resources,
                search_params,
                left_index,
                queries,
                left_neighbors.view(),
                left_distances.view());
  raft::resource::sync_stream(dev_resources);
  metadata.left_search_seconds = elapsed_seconds(start);

  std::cout << "Searching right index" << std::endl;
  start = clock_type::now();
  cagra::search(dev_resources,
                search_params,
                right_index,
                queries,
                right_neighbors.view(),
                right_distances.view());
  raft::resource::sync_stream(dev_resources);
  metadata.right_search_seconds = elapsed_seconds(start);

  std::cout << "Combining independent search results" << std::endl;
  start = clock_type::now();
  auto combined_neighbors = combine_results(dev_resources,
                                            left_neighbors.view(),
                                            left_distances.view(),
                                            right_neighbors.view(),
                                            right_distances.view(),
                                            static_cast<uint32_t>(left_rows));
  metadata.combine_seconds = elapsed_seconds(start);

  auto ground_truth = compute_ground_truth(dev_resources, dataset, queries, options.topk);
  metadata.ground_truth_seconds = ground_truth.seconds;
  metadata.recall =
    compute_recall(combined_neighbors, ground_truth.host_neighbors, queries.extent(0), options.topk);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Combined split-search recall@" << options.topk << ": " << metadata.recall.recall
            << " (" << metadata.recall.match_count << "/" << metadata.recall.total_count
            << " neighbor slots matched)" << std::endl;
  std::cout << "Per-query recall: mean=" << metadata.recall.mean_query_recall
            << ", min=" << metadata.recall.min_query_recall << std::endl;

  write_manifest(options.output_dir, command, metadata, index_params, search_params, options);
  std::cout << "Wrote manifest to " << (options.output_dir / "manifest.json") << std::endl;
}

}  // namespace

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  split_options options;
  if (!parse_options(argc, argv, options)) {
    usage(argv[0]);
    return 1;
  }
  if (options.show_help) {
    usage(argv[0]);
    return 0;
  }

  auto const& args = options.positional;
  int max_rows    = (args.size() > 2) ? std::atoi(args[2].c_str()) : kDefaultMaxRows;
  int max_queries = (args.size() > 3) ? std::atoi(args[3].c_str()) : kDefaultMaxQueries;
  max_rows        = std::max(max_rows, 1);
  max_queries     = std::max(max_queries, 1);

  run_metadata metadata;
  auto const command = command_line(argc, argv);

  if (args.size() >= 2) {
    std::filesystem::path base_path{args[0]};
    std::filesystem::path query_path{args[1]};
    metadata.dataset_source = "explicit";
    metadata.base_path      = base_path.string();
    metadata.query_path     = query_path.string();

    std::cout << "Loading dataset from " << base_path << std::endl;
    std::cout << "Loading queries from " << query_path << std::endl;
    auto dataset = read_bin_dataset<float, int64_t>(dev_resources, base_path.string(), max_rows);
    auto queries = read_bin_dataset<float, int64_t>(dev_resources, query_path.string(), max_queries);
    raft::resource::sync_stream(dev_resources);

    run_split_search(dev_resources,
                     raft::make_const_mdspan(dataset.view()),
                     raft::make_const_mdspan(queries.view()),
                     options,
                     metadata,
                     command);
    return 0;
  }

  if (auto paths = find_default_sift_dataset()) {
    metadata.dataset_source = "sift-128-euclidean";
    metadata.base_path      = paths->base.string();
    metadata.query_path     = paths->query.string();

    std::cout << "Loading cuVS Bench dataset: sift-128-euclidean" << std::endl;
    std::cout << "Base file: " << paths->base << std::endl;
    std::cout << "Query file: " << paths->query << std::endl;
    auto dataset = read_bin_dataset<float, int64_t>(dev_resources, paths->base.string(), max_rows);
    auto queries = read_bin_dataset<float, int64_t>(dev_resources, paths->query.string(), max_queries);
    raft::resource::sync_stream(dev_resources);

    run_split_search(dev_resources,
                     raft::make_const_mdspan(dataset.view()),
                     raft::make_const_mdspan(queries.view()),
                     options,
                     metadata,
                     command);
    return 0;
  }

  metadata.dataset_source = "synthetic";
  metadata.base_path      = "";
  metadata.query_path     = "";

  std::cout << "sift-128-euclidean files were not found; using synthetic data instead."
            << std::endl;
  auto dataset =
    raft::make_device_matrix<float, int64_t>(dev_resources, kSyntheticRows, kSyntheticDim);
  auto queries =
    raft::make_device_matrix<float, int64_t>(dev_resources, kSyntheticQueries, kSyntheticDim);
  generate_dataset(dev_resources, dataset.view(), queries.view());
  raft::resource::sync_stream(dev_resources);

  run_split_search(dev_resources,
                   raft::make_const_mdspan(dataset.view()),
                   raft::make_const_mdspan(queries.view()),
                   options,
                   metadata,
                   command);
}

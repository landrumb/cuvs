/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/matrix/copy.cuh>
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
constexpr int64_t kMaxGraphRecallNodes = 4096;

struct dataset_paths {
  std::filesystem::path base;
  std::filesystem::path query;
};

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

struct example_options {
  bool search_recall{};
  bool show_help{};
  std::vector<std::string> positional;
};

void usage(char const* program)
{
  std::cout << "Usage: " << program
            << " [--search-recall] [base.fbin query.fbin [max_rows] [max_queries]]\n\n"
            << "With no arguments, this example looks for the built-in cuVS Bench descriptor "
               "layout for sift-128-euclidean under RAPIDS_DATASET_ROOT_DIR or datasets/.\n"
            << "If those files are not present, it falls back to a larger synthetic dataset.\n"
            << "Pass --search-recall to compare CAGRA search results with exact GPU flat "
               "search. This can be slow.\n";
}

bool parse_options(int argc, char* argv[], example_options& options)
{
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--search-recall") {
      options.search_recall = true;
    } else if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    } else if (arg.rfind("--", 0) == 0) {
      std::cerr << "Unknown option: " << arg << std::endl;
      return false;
    } else {
      options.positional.push_back(arg);
    }
  }

  auto const n_positional = options.positional.size();
  if (n_positional == 1 || n_positional > 4) { return false; }
  return true;
}

double elapsed_seconds(clock_type::time_point start)
{
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

std::vector<int64_t> select_graph_recall_nodes(int64_t n_rows)
{
  int64_t n_eval = std::min(n_rows, kMaxGraphRecallNodes);
  std::vector<int64_t> eval_indices(static_cast<size_t>(n_eval));
  if (n_eval == n_rows) {
    std::iota(eval_indices.begin(), eval_indices.end(), int64_t{0});
    return eval_indices;
  }

  std::vector<int64_t> all_indices(static_cast<size_t>(n_rows));
  std::iota(all_indices.begin(), all_indices.end(), int64_t{0});
  std::mt19937 rng(12345);
  std::shuffle(all_indices.begin(), all_indices.end(), rng);
  std::copy(all_indices.begin(), all_indices.begin() + n_eval, eval_indices.begin());
  return eval_indices;
}

template <typename CagraIndex>
void evaluate_graph_recall(raft::device_resources const& dev_resources,
                           CagraIndex const& index,
                           raft::device_matrix_view<const float, int64_t> dataset)
{
  using namespace cuvs::neighbors;

  int64_t const n_rows = dataset.extent(0);
  int64_t const k      = index.graph_degree();
  auto const eval_indices = select_graph_recall_nodes(n_rows);
  int64_t const n_eval = static_cast<int64_t>(eval_indices.size());

  auto eval_indices_dev = raft::make_device_vector<int64_t, int64_t>(dev_resources, n_eval);
  auto eval_queries     = raft::make_device_matrix<float, int64_t>(dev_resources, n_eval, dataset.extent(1));
  auto eval_graph =
    raft::make_device_matrix<typename CagraIndex::graph_index_type, int64_t>(dev_resources, n_eval, k);

  raft::copy(eval_indices_dev.data_handle(),
             eval_indices.data(),
             eval_indices.size(),
             raft::resource::get_cuda_stream(dev_resources));
  raft::matrix::copy_rows(dev_resources,
                          dataset,
                          eval_queries.view(),
                          raft::make_const_mdspan(eval_indices_dev.view()));
  raft::matrix::copy_rows(dev_resources,
                          index.graph(),
                          eval_graph.view(),
                          raft::make_const_mdspan(eval_indices_dev.view()));
  raft::resource::sync_stream(dev_resources);

  std::cout << "Computing ground-truth kNN graph for recall check (" << n_eval
            << " nodes, k=" << k << ")" << std::endl;
  auto gt_start = clock_type::now();

  brute_force::index_params bf_index_params;
  auto bf_index = brute_force::build(dev_resources, bf_index_params, dataset);

  // Request one extra neighbor because brute-force includes the query vector itself.
  auto gt_neighbors =
    raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_eval, k + 1);
  auto gt_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_eval, k + 1);
  brute_force::search_params bf_search_params;
  brute_force::search(dev_resources,
                      bf_search_params,
                      bf_index,
                      raft::make_const_mdspan(eval_queries.view()),
                      gt_neighbors.view(),
                      gt_distances.view());
  raft::resource::sync_stream(dev_resources);

  auto actual_graph_host =
    raft::make_host_matrix<typename CagraIndex::graph_index_type, int64_t>(n_eval, k);
  auto gt_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(n_eval, k + 1);
  raft::copy(actual_graph_host.data_handle(),
             eval_graph.data_handle(),
             eval_graph.size(),
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(gt_neighbors_host.data_handle(),
             gt_neighbors.data_handle(),
             gt_neighbors.size(),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  std::vector<typename CagraIndex::graph_index_type> actual_graph(
    actual_graph_host.data_handle(), actual_graph_host.data_handle() + actual_graph_host.size());
  std::vector<typename CagraIndex::graph_index_type> expected_graph(static_cast<size_t>(n_eval * k));

  for (int64_t row = 0; row < n_eval; ++row) {
    int64_t const query_idx = eval_indices[static_cast<size_t>(row)];
    size_t num_added        = 0;
    for (int64_t j = 0; j < k + 1 && num_added < static_cast<size_t>(k); ++j) {
      auto const neighbor = gt_neighbors_host(row, j);
      if (neighbor == query_idx) { continue; }
      expected_graph[static_cast<size_t>(row * k + num_added)] =
        static_cast<typename CagraIndex::graph_index_type>(neighbor);
      ++num_added;
    }
    while (num_added < static_cast<size_t>(k)) {
      expected_graph[static_cast<size_t>(row * k + num_added)] =
        std::numeric_limits<typename CagraIndex::graph_index_type>::max();
      ++num_added;
    }
  }

  auto stats = calc_graph_recall(actual_graph, expected_graph, n_eval, k);
  print_graph_recall_stats(stats, n_eval, k, n_rows, elapsed_seconds(gt_start));
}

void evaluate_search_recall(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const float, int64_t> dataset,
                            raft::device_matrix_view<const float, int64_t> queries,
                            raft::device_matrix_view<uint32_t, int64_t> actual_neighbors)
{
  using namespace cuvs::neighbors;

  int64_t const n_queries = queries.extent(0);
  int64_t const topk      = actual_neighbors.extent(1);

  std::cout << "Computing exact GPU flat search for recall check (" << n_queries
            << " queries, topk=" << topk << ")" << std::endl;
  auto exact_start = clock_type::now();

  brute_force::index_params bf_index_params;
  auto bf_index = brute_force::build(dev_resources, bf_index_params, dataset);

  auto expected_neighbors =
    raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_queries, topk);
  auto expected_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);
  brute_force::search_params bf_search_params;
  brute_force::search(dev_resources,
                      bf_search_params,
                      bf_index,
                      queries,
                      expected_neighbors.view(),
                      expected_distances.view());
  raft::resource::sync_stream(dev_resources);

  auto actual_neighbors_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  auto expected_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(n_queries, topk);
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(
    actual_neighbors_host.data_handle(), actual_neighbors.data_handle(), actual_neighbors.size(), stream);
  raft::copy(expected_neighbors_host.data_handle(),
             expected_neighbors.data_handle(),
             expected_neighbors.size(),
             stream);
  raft::resource::sync_stream(dev_resources);

  std::vector<uint32_t> actual(actual_neighbors_host.data_handle(),
                               actual_neighbors_host.data_handle() + actual_neighbors_host.size());
  std::vector<int64_t> expected(
    expected_neighbors_host.data_handle(),
    expected_neighbors_host.data_handle() + expected_neighbors_host.size());

  auto stats = calc_recall(actual, expected, n_queries, topk);
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Search recall@" << topk << ": " << stats.recall << " (" << stats.match_count
            << "/" << stats.total_count << " neighbor slots matched)" << std::endl;
  std::cout << "Per-query recall: mean=" << stats.mean_node_recall
            << ", min=" << stats.min_node_recall << std::endl;
  std::cout << "Exact GPU flat search computed in " << elapsed_seconds(exact_start) << " s"
            << std::endl;
}

}  // namespace

void cagra_build_search_simple(raft::device_resources const& dev_resources,
                               raft::device_matrix_view<const float, int64_t> dataset,
                               raft::device_matrix_view<const float, int64_t> queries,
                               bool report_search_recall)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 12;
  int64_t n_queries = queries.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // use default index parameters
  cagra::index_params index_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto build_start = clock_type::now();
  auto index       = cagra::build(dev_resources, index_params, dataset);
  raft::resource::sync_stream(dev_resources);

  std::cout << "Built CAGRA index in " << elapsed_seconds(build_start) << " s" << std::endl;
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  evaluate_graph_recall(dev_resources, index, dataset);

  // use default search parameters
  cagra::search_params search_params;
  // search K nearest neighbors
  std::cout << "Searching CAGRA index (" << n_queries << " queries, topk=" << topk << ")"
            << std::endl;
  auto search_start = clock_type::now();
  cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  raft::resource::sync_stream(dev_resources);

  std::cout << "Searched CAGRA index in " << elapsed_seconds(search_start) << " s" << std::endl;

  if (report_search_recall) {
    evaluate_search_recall(dev_resources, dataset, queries, neighbors.view());
  }

  print_results(dev_resources, neighbors.view(), distances.view());
}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  example_options options;
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

  if (args.size() >= 2) {
    std::filesystem::path base_path{args[0]};
    std::filesystem::path query_path{args[1]};
    std::cout << "Loading dataset from " << base_path << std::endl;
    std::cout << "Loading queries from " << query_path << std::endl;
    auto load_start = clock_type::now();
    auto dataset = read_bin_dataset<float, int64_t>(dev_resources, base_path.string(), max_rows);
    auto queries =
      read_bin_dataset<float, int64_t>(dev_resources, query_path.string(), max_queries);
    raft::resource::sync_stream(dev_resources);
    std::cout << "Loaded dataset and queries in " << elapsed_seconds(load_start) << " s"
              << std::endl;

    cagra_build_search_simple(dev_resources,
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              options.search_recall);
    return 0;
  }

  if (auto paths = find_default_sift_dataset()) {
    std::cout << "Loading cuVS Bench dataset: sift-128-euclidean" << std::endl;
    std::cout << "Base file: " << paths->base << std::endl;
    std::cout << "Query file: " << paths->query << std::endl;
    auto load_start = clock_type::now();
    auto dataset = read_bin_dataset<float, int64_t>(dev_resources, paths->base.string(), max_rows);
    auto queries =
      read_bin_dataset<float, int64_t>(dev_resources, paths->query.string(), max_queries);
    raft::resource::sync_stream(dev_resources);
    std::cout << "Loaded dataset and queries in " << elapsed_seconds(load_start) << " s"
              << std::endl;

    cagra_build_search_simple(dev_resources,
                              raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()),
                              options.search_recall);
    return 0;
  }

  std::cout << "sift-128-euclidean files were not found; using larger synthetic data instead."
            << std::endl;
  auto generate_start = clock_type::now();
  auto dataset =
    raft::make_device_matrix<float, int64_t>(dev_resources, kSyntheticRows, kSyntheticDim);
  auto queries =
    raft::make_device_matrix<float, int64_t>(dev_resources, kSyntheticQueries, kSyntheticDim);
  generate_dataset(dev_resources, dataset.view(), queries.view());
  raft::resource::sync_stream(dev_resources);
  std::cout << "Generated synthetic dataset and queries in " << elapsed_seconds(generate_start)
            << " s" << std::endl;

  cagra_build_search_simple(dev_resources,
                            raft::make_const_mdspan(dataset.view()),
                            raft::make_const_mdspan(queries.view()),
                            options.search_recall);
}

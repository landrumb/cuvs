/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/matrix/copy.cuh>

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

namespace {

using clock_type = std::chrono::high_resolution_clock;

constexpr int64_t kSyntheticRows           = 100000;
constexpr int64_t kSyntheticDim            = 128;
constexpr int64_t kSyntheticQueries        = 100;
constexpr int kDefaultMaxRows              = 100000;
constexpr int kDefaultMaxQueries           = 1000;
constexpr int64_t kSearchTopk              = 12;
constexpr char const* kDefaultMaxIters     = "5,10,20";
constexpr char const* kDefaultTermThresh   = "0.0001,0.001";
constexpr char const* kDefaultIvfFractions = "0,0.25,0.5,0.75,1";
constexpr size_t kNnDescentSegmentSize       = 32;

struct dataset_paths {
  std::filesystem::path base;
  std::filesystem::path query;
};

enum class ivf_pq_selection_mode { closest, random };

struct compare_options {
  bool show_help{};
  std::vector<std::string> positional;
  std::vector<size_t> max_iterations;
  std::vector<float> termination_thresholds;
  std::vector<float> ivf_pq_fractions;
  ivf_pq_selection_mode ivf_pq_selection{ivf_pq_selection_mode::closest};
  std::optional<std::filesystem::path> output_csv;
};

struct build_stage_timings {
  double assembly_seconds{};
  double nn_descent_seconds{};
  double optimize_seconds{};
  size_t nn_descent_iterations{};
};

struct run_result {
  ivf_pq_selection_mode ivf_pq_selection{ivf_pq_selection_mode::closest};
  float ivf_pq_fraction{};
  size_t ivf_seed_cols{};
  size_t random_seed_cols{};
  size_t max_iterations{};
  float termination_threshold{};
  bool succeeded{true};
  std::string error_message;
  double assembly_seconds{};
  double nn_descent_seconds{};
  double optimize_seconds{};
  size_t nn_descent_iterations{};
  double build_seconds{};
  double search_seconds{};
  double search_recall{};
  double mean_query_recall{};
  double min_query_recall{};
};

constexpr double kFailedMetric = std::numeric_limits<double>::quiet_NaN();

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

template <typename T>
std::vector<T> split_csv(std::string const& text, T (*parse)(std::string const&))
{
  std::vector<T> values;
  std::stringstream ss{text};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) { continue; }
    values.push_back(parse(item));
  }
  return values;
}

size_t parse_size(std::string const& text)
{
  return static_cast<size_t>(std::stoul(text));
}

float parse_float(std::string const& text)
{
  return std::stof(text);
}

ivf_pq_selection_mode parse_ivf_pq_selection(std::string const& text)
{
  if (text == "closest") { return ivf_pq_selection_mode::closest; }
  if (text == "random") { return ivf_pq_selection_mode::random; }
  throw std::invalid_argument("expected closest or random");
}

char const* ivf_pq_selection_name(ivf_pq_selection_mode mode)
{
  switch (mode) {
    case ivf_pq_selection_mode::closest: return "closest";
    case ivf_pq_selection_mode::random: return "random";
  }
  return "unknown";
}

void usage(char const* program)
{
  std::cout << "Usage: " << program
            << " [base.fbin query.fbin [max_rows] [max_queries]]\n"
            << "       [--max-iters LIST] [--termination-threshold LIST]\n"
            << "       [--ivf-pq-fractions LIST] [--ivf-pq-selection MODE] [-o CSV]\n\n"
            << "Precomputes two NN-descent seed graphs: IVF-PQ candidates and the default "
               "segmented-random initialization. For each IVF-PQ fraction f in [0, 1], builds a "
               "mixed seed with round(f * intermediate_graph_degree) IVF-PQ neighbors followed by "
               "the leading random-default neighbors, then runs NN-descent refinement, CAGRA "
               "optimize, and search.\n"
            << "When f is strictly between 0 and 1, --ivf-pq-selection controls which IVF-PQ "
               "neighbors are kept: closest (prefix by refined distance, default) or random.\n"
            << "f=0 is pure random-default init; f=1 is pure IVF-PQ init.\n"
            << "Always reports search recall@12 (vs exact GPU flat search) and build/search "
               "runtimes.\n\n"
            << "Dataset arguments match the other experiments examples. With no positional "
               "arguments, looks for sift-128-euclidean under RAPIDS_DATASET_ROOT_DIR or "
               "datasets/, then falls back to synthetic data.\n\n"
            << "Options:\n"
            << "  --max-iters LIST               comma-separated max_iterations values "
               "(default "
            << kDefaultMaxIters << ")\n"
            << "  --termination-threshold LIST   comma-separated termination_threshold values "
               "(default "
            << kDefaultTermThresh << ")\n"
            << "  --ivf-pq-fractions LIST        comma-separated IVF-PQ seed fractions in [0, 1] "
               "(default "
            << kDefaultIvfFractions << ")\n"
            << "  --ivf-pq-selection MODE        closest | random (default closest)\n"
            << "  -o, --output CSV               write summary results to CSV\n";
}

bool parse_options(int argc, char* argv[], compare_options& options)
{
  options.max_iterations         = split_csv<size_t>(kDefaultMaxIters, parse_size);
  options.termination_thresholds = split_csv<float>(kDefaultTermThresh, parse_float);
  options.ivf_pq_fractions       = split_csv<float>(kDefaultIvfFractions, parse_float);

  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    }
    if (arg.rfind("--max-iters=", 0) == 0) {
      options.max_iterations = split_csv<size_t>(arg.substr(12), parse_size);
      continue;
    }
    if (arg == "--max-iters") {
      if (i + 1 >= argc) { return false; }
      options.max_iterations = split_csv<size_t>(argv[++i], parse_size);
      continue;
    }
    if (arg.rfind("--termination-threshold=", 0) == 0) {
      options.termination_thresholds = split_csv<float>(arg.substr(24), parse_float);
      continue;
    }
    if (arg == "--termination-threshold") {
      if (i + 1 >= argc) { return false; }
      options.termination_thresholds = split_csv<float>(argv[++i], parse_float);
      continue;
    }
    if (arg.rfind("--ivf-pq-fractions=", 0) == 0) {
      options.ivf_pq_fractions = split_csv<float>(arg.substr(19), parse_float);
      continue;
    }
    if (arg == "--ivf-pq-fractions") {
      if (i + 1 >= argc) { return false; }
      options.ivf_pq_fractions = split_csv<float>(argv[++i], parse_float);
      continue;
    }
    if (arg.rfind("--ivf-pq-selection=", 0) == 0) {
      try {
        options.ivf_pq_selection = parse_ivf_pq_selection(arg.substr(21));
      } catch (std::exception const&) {
        return false;
      }
      continue;
    }
    if (arg == "--ivf-pq-selection") {
      if (i + 1 >= argc) { return false; }
      try {
        options.ivf_pq_selection = parse_ivf_pq_selection(argv[++i]);
      } catch (std::exception const&) {
        return false;
      }
      continue;
    }
    if (arg == "-o" || arg == "--output") {
      if (i + 1 >= argc) { return false; }
      options.output_csv = argv[++i];
      continue;
    }
    if (arg.rfind("-o=", 0) == 0) {
      options.output_csv = arg.substr(3);
      continue;
    }
    if (arg.rfind("--output=", 0) == 0) {
      options.output_csv = arg.substr(9);
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      std::cerr << "Unknown option: " << arg << std::endl;
      return false;
    }
    options.positional.push_back(arg);
  }

  if (options.max_iterations.empty() || options.termination_thresholds.empty() ||
      options.ivf_pq_fractions.empty()) {
    return false;
  }

  for (float fraction : options.ivf_pq_fractions) {
    if (fraction < 0.0f || fraction > 1.0f) { return false; }
  }

  auto const n_positional = options.positional.size();
  if (n_positional == 1 || n_positional > 4) { return false; }
  return true;
}

double elapsed_seconds(clock_type::time_point start)
{
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

void fill_nn_descent_style_random_neighbors(raft::host_matrix_view<uint32_t, int64_t> graph,
                                          size_t col_offset,
                                          size_t degree)
{
  int64_t const nrow        = graph.extent(0);
  size_t const num_segments = degree / kNnDescentSegmentSize;
  int64_t const row_degree  = graph.extent(1);

  RAFT_EXPECTS(degree % kNnDescentSegmentSize == 0,
                "Random NN-descent initialization requires degree divisible by %zu",
                kNnDescentSegmentSize);

  for (size_t seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
    std::vector<uint32_t> rand_seq((nrow + static_cast<int64_t>(num_segments) - 1) /
                                   static_cast<int64_t>(num_segments));
    std::iota(rand_seq.begin(), rand_seq.end(), uint32_t{0});
    auto gen = std::default_random_engine{seg_idx};
    std::shuffle(rand_seq.begin(), rand_seq.end(), gen);

    for (int64_t i = 0; i < nrow; ++i) {
      size_t idx = static_cast<size_t>(i * row_degree + col_offset + seg_idx * kNnDescentSegmentSize);
      size_t self_in_this_seg = 0;
      for (size_t j = 0; j < kNnDescentSegmentSize; ++j) {
        uint32_t id = rand_seq[idx % rand_seq.size()] * static_cast<uint32_t>(num_segments) +
                      static_cast<uint32_t>(seg_idx);
        if (static_cast<int64_t>(id) == i) {
          ++idx;
          id = rand_seq[idx % rand_seq.size()] * static_cast<uint32_t>(num_segments) +
               static_cast<uint32_t>(seg_idx);
          self_in_this_seg = 1;
        }

        int64_t const out_col =
          static_cast<int64_t>(col_offset + seg_idx * kNnDescentSegmentSize + j);
        graph(i, out_col) =
          j < (rand_seq.size() - self_in_this_seg) && static_cast<int64_t>(id) < nrow
            ? id
            : std::numeric_limits<uint32_t>::max();
        ++idx;
      }
    }
  }
}

size_t ivf_cols_for_fraction(float fraction, size_t intermediate_degree)
{
  fraction = std::clamp(fraction, 0.0f, 1.0f);
  return static_cast<size_t>(std::lround(fraction * static_cast<double>(intermediate_degree)));
}

void assemble_mixed_seed(raft::host_matrix_view<uint32_t, int64_t> knn_graph,
                         raft::host_matrix_view<const uint32_t, int64_t> ivf_pq_init,
                         raft::host_matrix_view<const uint32_t, int64_t> random_init,
                         size_t ivf_cols,
                         ivf_pq_selection_mode ivf_selection)
{
  auto const intermediate_degree = static_cast<size_t>(knn_graph.extent(1));
  auto const random_cols           = intermediate_degree - ivf_cols;

  RAFT_EXPECTS(ivf_pq_init.extent(1) >= static_cast<int64_t>(intermediate_degree),
               "IVF-PQ init graph degree is too small");
  RAFT_EXPECTS(random_init.extent(1) >= static_cast<int64_t>(random_cols),
               "Random init graph does not have enough columns");

  std::vector<size_t> ivf_source_cols(intermediate_degree);

  for (int64_t row = 0; row < knn_graph.extent(0); ++row) {
    if (ivf_cols > 0) {
      if (ivf_selection == ivf_pq_selection_mode::closest) {
        for (size_t col = 0; col < ivf_cols; ++col) {
          knn_graph(row, static_cast<int64_t>(col)) = ivf_pq_init(row, static_cast<int64_t>(col));
        }
      } else {
        std::iota(ivf_source_cols.begin(), ivf_source_cols.end(), size_t{0});
        auto gen = std::default_random_engine{static_cast<uint32_t>(row)};
        std::shuffle(ivf_source_cols.begin(), ivf_source_cols.end(), gen);
        for (size_t col = 0; col < ivf_cols; ++col) {
          knn_graph(row, static_cast<int64_t>(col)) =
            ivf_pq_init(row, static_cast<int64_t>(ivf_source_cols[col]));
        }
      }
    }
    for (size_t col = 0; col < random_cols; ++col) {
      knn_graph(row, static_cast<int64_t>(ivf_cols + col)) =
        random_init(row, static_cast<int64_t>(col));
    }
  }
}

cuvs::neighbors::nn_descent::index_params make_nn_descent_params(
  size_t intermediate_degree,
  cuvs::distance::DistanceType metric,
  size_t max_iterations,
  float termination_threshold)
{
  auto params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(intermediate_degree, metric);
  params.max_iterations        = max_iterations;
  params.termination_threshold = termination_threshold;
  params.return_distances      = false;
  return params;
}

struct ground_truth_neighbors {
  raft::device_matrix<int64_t, int64_t> neighbors;
  std::vector<int64_t> host_neighbors;
};

ground_truth_neighbors compute_ground_truth_neighbors(
  raft::device_resources const& dev_resources,
  raft::device_matrix_view<const float, int64_t> dataset,
  raft::device_matrix_view<const float, int64_t> queries,
  int64_t topk)
{
  using namespace cuvs::neighbors;

  auto gt_start = clock_type::now();
  std::cout << "Computing exact GPU flat search ground truth (" << queries.extent(0)
            << " queries, topk=" << topk << ")" << std::endl;

  brute_force::index_params bf_index_params;
  auto bf_index = brute_force::build(dev_resources, bf_index_params, dataset);

  ground_truth_neighbors gt{
    raft::make_device_matrix<int64_t, int64_t>(dev_resources, queries.extent(0), topk), {}};
  auto gt_distances = raft::make_device_matrix<float, int64_t>(dev_resources, queries.extent(0), topk);
  brute_force::search_params bf_search_params;
  brute_force::search(dev_resources,
                      bf_search_params,
                      bf_index,
                      queries,
                      gt.neighbors.view(),
                      gt_distances.view());
  raft::resource::sync_stream(dev_resources);

  auto gt_host = raft::make_host_matrix<int64_t, int64_t>(queries.extent(0), topk);
  auto stream  = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(gt_host.data_handle(), gt.neighbors.data_handle(), gt.neighbors.size(), stream);
  raft::resource::sync_stream(dev_resources);

  gt.host_neighbors.assign(gt_host.data_handle(), gt_host.data_handle() + gt_host.size());
  std::cout << "Ground truth computed in " << elapsed_seconds(gt_start) << " s" << std::endl;
  return gt;
}

graph_recall_stats compute_search_recall(
  raft::device_resources const& dev_resources,
  raft::device_matrix_view<uint32_t, int64_t> actual_neighbors,
  ground_truth_neighbors const& ground_truth,
  int64_t n_queries,
  int64_t topk)
{
  auto actual_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  auto stream      = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(actual_host.data_handle(),
             actual_neighbors.data_handle(),
             actual_neighbors.size(),
             stream);
  raft::resource::sync_stream(dev_resources);

  std::vector<uint32_t> actual(actual_host.data_handle(),
                               actual_host.data_handle() + actual_host.size());
  return calc_recall(actual, ground_truth.host_neighbors, n_queries, topk);
}

std::pair<cuvs::neighbors::cagra::index<float, uint32_t>, build_stage_timings>
build_mixed_seed_cagra(raft::device_resources const& dev_resources,
                       raft::device_matrix_view<const float, int64_t> dataset,
                       raft::host_matrix_view<const uint32_t, int64_t> ivf_pq_init,
                       raft::host_matrix_view<const uint32_t, int64_t> random_init,
                       float ivf_pq_fraction,
                       ivf_pq_selection_mode ivf_selection,
                       cuvs::neighbors::cagra::index_params const& base_index_params,
                       cuvs::neighbors::nn_descent::index_params const& nn_descent_params)
{
  using namespace cuvs::neighbors;

  build_stage_timings timings{};

  auto const intermediate_degree = base_index_params.intermediate_graph_degree;
  auto const ivf_cols           = ivf_cols_for_fraction(ivf_pq_fraction, intermediate_degree);

  auto knn_graph =
    raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), intermediate_degree);

  auto assembly_start = clock_type::now();
  assemble_mixed_seed(knn_graph.view(), ivf_pq_init, random_init, ivf_cols, ivf_selection);
  timings.assembly_seconds = elapsed_seconds(assembly_start);

  auto nn_descent_start = clock_type::now();
  auto nn_descent_index = nn_descent::build(
    dev_resources, nn_descent_params, dataset, std::make_optional(knn_graph.view()));
  raft::resource::sync_stream(dev_resources);
  timings.nn_descent_seconds    = elapsed_seconds(nn_descent_start);
  timings.nn_descent_iterations = nn_descent_index.num_iterations_executed();

  auto optimize_start = clock_type::now();
  auto cagra_graph =
    raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), base_index_params.graph_degree);
  cagra::helpers::optimize(dev_resources, knn_graph.view(), cagra_graph.view());
  raft::resource::sync_stream(dev_resources);
  auto index = cagra::index<float, uint32_t>(
    dev_resources, base_index_params.metric, dataset, raft::make_const_mdspan(cagra_graph.view()));
  timings.optimize_seconds = elapsed_seconds(optimize_start);

  return {std::move(index), timings};
}

run_result evaluate_configuration(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries,
                                  raft::host_matrix_view<const uint32_t, int64_t> ivf_pq_init,
                                  raft::host_matrix_view<const uint32_t, int64_t> random_init,
                                  cuvs::neighbors::cagra::index_params const& base_index_params,
                                  ground_truth_neighbors const& ground_truth,
                                  float ivf_pq_fraction,
                                  ivf_pq_selection_mode ivf_selection,
                                  size_t max_iterations,
                                  float termination_threshold)
{
  using namespace cuvs::neighbors;

  auto const n_queries           = queries.extent(0);
  auto const topk                = kSearchTopk;
  auto const intermediate_degree = base_index_params.intermediate_graph_degree;

  auto nn_descent_params = make_nn_descent_params(intermediate_degree,
                                                  base_index_params.metric,
                                                  max_iterations,
                                                  termination_threshold);

  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  run_result result{};
  result.ivf_pq_selection      = ivf_selection;
  result.ivf_pq_fraction       = ivf_pq_fraction;
  result.ivf_seed_cols         = ivf_cols_for_fraction(ivf_pq_fraction, intermediate_degree);
  result.random_seed_cols        = intermediate_degree - result.ivf_seed_cols;
  result.max_iterations          = max_iterations;
  result.termination_threshold   = termination_threshold;

  try {
    auto [index, build_timings] = build_mixed_seed_cagra(dev_resources,
                                                         dataset,
                                                         ivf_pq_init,
                                                         random_init,
                                                         ivf_pq_fraction,
                                                         ivf_selection,
                                                         base_index_params,
                                                         nn_descent_params);
    result.assembly_seconds      = build_timings.assembly_seconds;
    result.nn_descent_seconds    = build_timings.nn_descent_seconds;
    result.optimize_seconds      = build_timings.optimize_seconds;
    result.nn_descent_iterations = build_timings.nn_descent_iterations;
    result.build_seconds         = build_timings.nn_descent_seconds + build_timings.optimize_seconds;

    cagra::search_params search_params;
    auto search_start = clock_type::now();
    cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
    raft::resource::sync_stream(dev_resources);
    result.search_seconds = elapsed_seconds(search_start);

    auto recall_stats        = compute_search_recall(dev_resources, neighbors.view(), ground_truth, n_queries, topk);
    result.search_recall     = recall_stats.recall;
    result.mean_query_recall = recall_stats.mean_node_recall;
    result.min_query_recall  = recall_stats.min_node_recall;
  } catch (std::exception const& ex) {
    result.succeeded         = false;
    result.error_message     = ex.what();
    result.assembly_seconds  = kFailedMetric;
    result.nn_descent_seconds = kFailedMetric;
    result.optimize_seconds  = kFailedMetric;
    result.build_seconds     = kFailedMetric;
    result.search_seconds    = kFailedMetric;
    result.search_recall     = kFailedMetric;
    result.mean_query_recall = kFailedMetric;
    result.min_query_recall  = kFailedMetric;
  }
  return result;
}

void print_run_result(run_result const& result)
{
  std::cout << std::fixed << std::setprecision(4);
  if (!result.succeeded) {
    std::cout << "[ivf_sel=" << ivf_pq_selection_name(result.ivf_pq_selection)
              << " ivf_frac=" << result.ivf_pq_fraction << " ivf_cols=" << result.ivf_seed_cols
              << " random_cols=" << result.random_seed_cols << "] FAILED: " << result.error_message
              << std::endl;
    return;
  }
  std::cout << "[ivf_sel=" << ivf_pq_selection_name(result.ivf_pq_selection)
            << " ivf_frac=" << result.ivf_pq_fraction << " ivf_cols=" << result.ivf_seed_cols
            << " random_cols=" << result.random_seed_cols << "] nn_iters=" << result.nn_descent_iterations
            << "/" << result.max_iterations << ", assembly=" << result.assembly_seconds
            << " s, nn_descent=" << result.nn_descent_seconds
            << " s, optimize=" << result.optimize_seconds << " s, build=" << result.build_seconds
            << " s, search=" << result.search_seconds << " s, recall@" << kSearchTopk << "="
            << result.search_recall << ", mean_query_recall=" << result.mean_query_recall
            << ", min_query_recall=" << result.min_query_recall << std::endl;
}

std::string csv_escape(std::string const& text)
{
  if (text.find_first_of(",\"\n\r") == std::string::npos) { return text; }
  std::string escaped;
  escaped.reserve(text.size() + 2);
  escaped.push_back('"');
  for (char ch : text) {
    if (ch == '"') {
      escaped.append("\"\"");
    } else {
      escaped.push_back(ch);
    }
  }
  escaped.push_back('"');
  return escaped;
}

void write_csv_metric(std::ostream& out, double value)
{
  if (std::isnan(value)) {
    return;
  }
  out << std::fixed << std::setprecision(6) << value;
}

void write_results_csv(std::filesystem::path const& path, std::vector<run_result> const& results)
{
  std::ofstream out{path};
  if (!out) {
    throw std::runtime_error("failed to open CSV output file: " + path.string());
  }

  out << "max_iters,term_thresh,ivf_sel,ivf_frac,ivf_cols,rand_cols,nn_iters,assembly_s,"
         "nndescent_s,optimize_s,build_s,search_s,recall,mean_q_recall,min_q_recall,succeeded,"
         "error_message\n";

  for (auto const& result : results) {
    out << result.max_iterations << ',' << result.termination_threshold << ','
        << ivf_pq_selection_name(result.ivf_pq_selection) << ',' << result.ivf_pq_fraction << ','
        << result.ivf_seed_cols << ',' << result.random_seed_cols << ',';
    if (result.succeeded) {
      out << result.nn_descent_iterations << ',';
      write_csv_metric(out, result.assembly_seconds);
      out << ',';
      write_csv_metric(out, result.nn_descent_seconds);
      out << ',';
      write_csv_metric(out, result.optimize_seconds);
      out << ',';
      write_csv_metric(out, result.build_seconds);
      out << ',';
      write_csv_metric(out, result.search_seconds);
      out << ',';
      write_csv_metric(out, result.search_recall);
      out << ',';
      write_csv_metric(out, result.mean_query_recall);
      out << ',';
      write_csv_metric(out, result.min_query_recall);
      out << ",1,\n";
    } else {
      out << ",,,,,,,,,0," << csv_escape(result.error_message) << "\n";
    }
  }
}

void print_summary_table(std::vector<run_result> const& results)
{
  std::cout << "\nSummary (search recall@" << kSearchTopk << ")\n";
  std::cout << std::left << std::setw(8) << "max_it" << std::setw(12) << "term_thresh"
            << std::setw(10) << "ivf_sel" << std::setw(10) << "ivf_frac" << std::setw(10)
            << "ivf_cols" << std::setw(12) << "rand_cols" << std::setw(10) << "nn_iters"
            << std::setw(12) << "assembly_s" << std::setw(12) << "nndescent_s"
            << std::setw(12) << "optimize_s" << std::setw(12) << "build_s" << std::setw(12)
            << "search_s" << std::setw(10) << "recall" << std::setw(14) << "mean_q_recall"
            << "min_q_recall\n";

  std::cout << std::fixed << std::setprecision(4);
  for (auto const& result : results) {
    std::cout << std::left << std::setw(8) << result.max_iterations << std::setw(12)
              << result.termination_threshold << std::setw(10)
              << ivf_pq_selection_name(result.ivf_pq_selection) << std::setw(10)
              << result.ivf_pq_fraction
              << std::setw(10) << result.ivf_seed_cols << std::setw(12) << result.random_seed_cols;
    if (!result.succeeded) {
      std::cout << std::setw(10) << "-" << std::setw(12) << "-" << std::setw(12) << "-"
                << std::setw(12) << "-" << std::setw(12) << "-" << std::setw(12) << "-";
    } else {
      std::cout << std::setw(10) << result.nn_descent_iterations << std::setw(12)
                << result.assembly_seconds << std::setw(12) << result.nn_descent_seconds
                << std::setw(12) << result.optimize_seconds << std::setw(12) << result.build_seconds
                << std::setw(12) << result.search_seconds;
    }
    if (!result.succeeded) {
      std::cout << std::setw(10) << "FAILED" << std::setw(14) << "-" << "-\n";
    } else {
      std::cout << std::setw(10) << result.search_recall << std::setw(14)
                << result.mean_query_recall << result.min_query_recall << "\n";
    }
  }
}

void run_comparison(raft::device_resources const& dev_resources,
                    raft::device_matrix_view<const float, int64_t> dataset,
                    raft::device_matrix_view<const float, int64_t> queries,
                    compare_options const& options)
{
  using namespace cuvs::neighbors;

  cagra::index_params base_index_params;
  auto const intermediate_degree = base_index_params.intermediate_graph_degree;
  RAFT_EXPECTS(intermediate_degree % kNnDescentSegmentSize == 0,
                "intermediate_graph_degree must be divisible by %zu for random initialization",
                kNnDescentSegmentSize);

  auto ground_truth =
    compute_ground_truth_neighbors(dev_resources, dataset, queries, kSearchTopk);

  auto dataset_host =
    raft::make_host_matrix<float, int64_t>(dataset.extent(0), dataset.extent(1));
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(dataset_host.data_handle(), dataset.data_handle(), dataset.size(), stream);
  raft::resource::sync_stream(dev_resources);

  std::cout << "Building IVF-PQ initialization graph" << std::endl;
  auto ivf_pq_start = clock_type::now();
  auto ivf_pq_params =
    cagra::graph_build_params::ivf_pq_params(dataset.extents(), base_index_params.metric);
  auto ivf_pq_init =
    raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), intermediate_degree);
  cagra::build_knn_graph(dev_resources, dataset_host.view(), ivf_pq_init.view(), ivf_pq_params);
  raft::resource::sync_stream(dev_resources);
  std::cout << "IVF-PQ initialization built in " << elapsed_seconds(ivf_pq_start) << " s"
            << std::endl;

  std::cout << "Building default NN-descent random initialization graph" << std::endl;
  auto random_start = clock_type::now();
  auto random_init =
    raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), intermediate_degree);
  fill_nn_descent_style_random_neighbors(random_init.view(), 0, intermediate_degree);
  std::cout << "Default random initialization built in " << elapsed_seconds(random_start) << " s\n"
            << std::endl;
  std::cout << "IVF-PQ neighbor selection for mixed seeds: "
            << ivf_pq_selection_name(options.ivf_pq_selection) << std::endl
            << std::endl;

  std::vector<run_result> results;
  for (auto const max_iterations : options.max_iterations) {
    for (auto const termination_threshold : options.termination_thresholds) {
      std::cout << "=== max_iterations=" << max_iterations
                << " termination_threshold=" << std::fixed << std::setprecision(6)
                << termination_threshold << " ===" << std::endl;

      for (auto const ivf_pq_fraction : options.ivf_pq_fractions) {
        auto result = evaluate_configuration(dev_resources,
                                             dataset,
                                             queries,
                                             ivf_pq_init.view(),
                                             random_init.view(),
                                             base_index_params,
                                             ground_truth,
                                             ivf_pq_fraction,
                                             options.ivf_pq_selection,
                                             max_iterations,
                                             termination_threshold);
        print_run_result(result);
        results.push_back(result);
      }
      std::cout << std::endl;
    }
  }

  print_summary_table(results);

  if (options.output_csv.has_value()) {
    write_results_csv(options.output_csv.value(), results);
    std::cout << "\nWrote results to " << options.output_csv.value() << std::endl;
  }
}

}  // namespace

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  compare_options options;
  if (!parse_options(argc, argv, options)) {
    usage(argv[0]);
    return 1;
  }
  if (options.show_help) {
    usage(argv[0]);
    return 0;
  }

  auto const& args = options.positional;
  int max_rows     = (args.size() > 2) ? std::atoi(args[2].c_str()) : kDefaultMaxRows;
  int max_queries  = (args.size() > 3) ? std::atoi(args[3].c_str()) : kDefaultMaxQueries;
  max_rows         = std::max(max_rows, 1);
  max_queries      = std::max(max_queries, 1);

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
    std::cout << "Loaded dataset and queries in "
              << elapsed_seconds(load_start) << " s\n" << std::endl;

    run_comparison(dev_resources,
                   raft::make_const_mdspan(dataset.view()),
                   raft::make_const_mdspan(queries.view()),
                   options);
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
    std::cout << "Loaded dataset and queries in "
              << elapsed_seconds(load_start) << " s\n" << std::endl;

    run_comparison(dev_resources,
                   raft::make_const_mdspan(dataset.view()),
                   raft::make_const_mdspan(queries.view()),
                   options);
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
            << " s\n" << std::endl;

  run_comparison(dev_resources,
                 raft::make_const_mdspan(dataset.view()),
                 raft::make_const_mdspan(queries.view()),
                 options);
}

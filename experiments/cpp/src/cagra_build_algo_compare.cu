/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Sweep CAGRA graph-build time for NN-Descent vs IVF-PQ across dataset sizes,
 * to locate the size regime (if any) where the NN-Descent path is faster.
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
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
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double elapsed_ms(clock_type::time_point start, clock_type::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename Fn>
double time_cuda(raft::device_resources const& res, Fn&& fn)
{
  raft::resource::sync_stream(res);
  auto start = clock_type::now();
  fn();
  raft::resource::sync_stream(res);
  return elapsed_ms(start, clock_type::now());
}

struct options {
  std::string dataset =
    "/raid/blandrum/local_datasets/wiki_all_10M/base.10M.fbin";
  std::string output_csv = "/raid/blandrum/local_datasets/wiki_all_10M/build_algo_compare.csv";
  std::vector<size_t> sizes = {
    100000, 250000, 500000, 1000000, 2000000, 4000000, 8000000, 10000000};
  size_t graph_degree              = 64;
  size_t intermediate_graph_degree = 128;
  size_t nnd_iterations            = 20;  // CAGRA default for nn-descent
  int repeats                      = 2;
};

void read_exact(std::ifstream& in, char* dst, size_t bytes, const std::string& path)
{
  in.read(dst, static_cast<std::streamsize>(bytes));
  if (!in) { throw std::runtime_error("Short read on " + path); }
}

// Load the leading `max_rows` rows of an fbin (uint32 rows, uint32 dim, float data).
struct host_matrix {
  uint32_t rows = 0;
  uint32_t dim  = 0;
  std::vector<float> data;
};

host_matrix load_prefix(const std::string& path, size_t max_rows)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("Could not open " + path); }
  host_matrix m;
  read_exact(in, reinterpret_cast<char*>(&m.rows), sizeof(uint32_t), path);
  read_exact(in, reinterpret_cast<char*>(&m.dim), sizeof(uint32_t), path);
  uint32_t rows_to_read = std::min<uint32_t>(m.rows, static_cast<uint32_t>(max_rows));
  const size_t count    = static_cast<size_t>(rows_to_read) * m.dim;
  m.data.resize(count);
  read_exact(in, reinterpret_cast<char*>(m.data.data()), count * sizeof(float), path);
  m.rows = rows_to_read;
  return m;
}

enum class algo { nn_descent, ivf_pq };

const char* algo_name(algo a) { return a == algo::nn_descent ? "nn_descent" : "ivf_pq"; }

cuvs::neighbors::cagra::index_params make_params(const options& opts, algo a, int64_t rows, int64_t dim)
{
  cuvs::neighbors::cagra::index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = opts.graph_degree;
  params.intermediate_graph_degree = opts.intermediate_graph_degree;
  params.attach_dataset_on_build   = true;
  if (a == algo::ivf_pq) {
    params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
      raft::matrix_extent<int64_t>(rows, dim), params.metric);
  } else {
    auto nnd = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      opts.intermediate_graph_degree, params.metric);
    nnd.max_iterations        = opts.nnd_iterations;
    nnd.return_distances      = false;
    params.graph_build_params = nnd;
  }
  return params;
}

// Returns build time in ms, or NaN if the build threw (e.g. degenerate graph).
double build_once(raft::device_resources const& res,
                  const options& opts,
                  algo a,
                  raft::device_matrix_view<const float, int64_t> dataset)
{
  auto params = make_params(opts, a, dataset.extent(0), dataset.extent(1));
  cuvs::neighbors::cagra::index<float, uint32_t> index(res);
  try {
    return time_cuda(res, [&] {
      index = cuvs::neighbors::cagra::build(res, params, dataset);
    });
  } catch (const std::exception& e) {
    raft::resource::sync_stream(res);
    std::cerr << "  [" << algo_name(a) << " build failed at rows=" << dataset.extent(0)
              << "]: " << e.what() << "\n";
    return std::numeric_limits<double>::quiet_NaN();
  }
}

options parse_args(int argc, char** argv)
{
  options o;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value");
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dataset") {
      o.dataset = need(i);
    } else if (arg == "--output-csv") {
      o.output_csv = need(i);
    } else if (arg == "--sizes") {
      o.sizes.clear();
      std::stringstream ss(need(i));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) o.sizes.push_back(std::stoull(tok));
      }
    } else if (arg == "--graph-degree") {
      o.graph_degree = std::stoull(need(i));
    } else if (arg == "--intermediate-degree") {
      o.intermediate_graph_degree = std::stoull(need(i));
    } else if (arg == "--nnd-iterations") {
      o.nnd_iterations = std::stoull(need(i));
    } else if (arg == "--repeats") {
      o.repeats = std::stoi(need(i));
    } else {
      throw std::runtime_error("unknown arg: " + arg);
    }
  }
  return o;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto opts = parse_args(argc, argv);

    raft::device_resources res;
    // Managed upstream so the largest (10M x 768 ~ 30.7 GB) datasets plus build
    // working sets cannot hard-OOM on an 80 GB card.
    rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                          size_t{8} * 1024 * 1024 * 1024);
    rmm::mr::set_current_device_resource(pool_mr);

    size_t max_rows = *std::max_element(opts.sizes.begin(), opts.sizes.end());
    std::cout << "Loading up to " << max_rows << " rows from " << opts.dataset << " ...\n";
    auto host = load_prefix(opts.dataset, max_rows);
    std::cout << "Loaded rows=" << host.rows << " dim=" << host.dim << "\n";

    // Upload the full prefix once; sub-sizes are a leading device sub-view.
    auto dev = raft::make_device_matrix<float, int64_t>(res, host.rows, host.dim);
    raft::copy(dev.data_handle(), host.data.data(), host.data.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    std::ofstream csv(opts.output_csv);
    csv << "rows,algo,run,build_ms\n";

    std::cout << "\nWarmup build (init CUDA/cuBLAS/kmeans paths)...\n";
    {
      int64_t wr = std::min<int64_t>(50000, host.rows);
      auto sub = raft::make_device_matrix_view<const float, int64_t>(dev.data_handle(), wr, host.dim);
      build_once(res, opts, algo::ivf_pq, sub);
      build_once(res, opts, algo::nn_descent, sub);
    }

    std::cout << "\n"
              << std::setw(12) << "rows" << std::setw(16) << "ivf_pq_ms"
              << std::setw(16) << "nndescent_ms" << std::setw(12) << "nnd/ivf"
              << std::setw(14) << "winner" << "\n";

    for (size_t n : opts.sizes) {
      int64_t rows = static_cast<int64_t>(std::min<size_t>(n, host.rows));
      auto sub = raft::make_device_matrix_view<const float, int64_t>(dev.data_handle(), rows, host.dim);

      double best_ivf = std::numeric_limits<double>::infinity();
      double best_nnd = std::numeric_limits<double>::infinity();
      for (int r = 0; r < opts.repeats; ++r) {
        double t_ivf = build_once(res, opts, algo::ivf_pq, sub);
        csv << rows << ",ivf_pq," << r << "," << std::fixed << std::setprecision(3) << t_ivf << "\n";
        if (!std::isnan(t_ivf)) best_ivf = std::min(best_ivf, t_ivf);

        double t_nnd = build_once(res, opts, algo::nn_descent, sub);
        csv << rows << ",nn_descent," << r << "," << std::fixed << std::setprecision(3) << t_nnd << "\n";
        if (!std::isnan(t_nnd)) best_nnd = std::min(best_nnd, t_nnd);
      }
      csv.flush();
      const bool ok = std::isfinite(best_ivf) && std::isfinite(best_nnd);
      const char* winner = !ok ? "FAIL" : (best_nnd < best_ivf ? "nn_descent" : "ivf_pq");
      std::cout << std::setw(12) << rows << std::setw(16) << std::fixed << std::setprecision(1)
                << best_ivf << std::setw(16) << best_nnd << std::setw(12) << std::setprecision(3)
                << (best_nnd / best_ivf) << std::setw(14) << winner << "\n";
    }

    std::cout << "\nWrote " << opts.output_csv << "\n";
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

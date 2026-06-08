/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal CAGRA construction driver intended for profiling (e.g. with Nsight
// Systems / nvprof / ncu). It loads a dataset from a binary file using the
// integrated dataset loader (`read_bin_dataset` from common.cuh) and times the
// CAGRA graph build, with an optional warmup pass so the profiler can isolate
// steady-state behavior.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

// Configure CAGRA build parameters from the command line and select the graph
// build algorithm.
cuvs::neighbors::cagra::index_params make_index_params(size_t intermediate_graph_degree,
                                                       size_t graph_degree,
                                                       const std::string& build_algo)
{
  using namespace cuvs::neighbors;

  cagra::index_params index_params;
  index_params.intermediate_graph_degree = intermediate_graph_degree;
  index_params.graph_degree              = graph_degree;

  if (build_algo == "ivf_pq") {
    index_params.graph_build_params = cagra::graph_build_params::ivf_pq_params();
  } else if (build_algo == "nn_descent") {
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(intermediate_graph_degree);
  } else if (build_algo == "auto") {
    // Leave graph_build_params as std::monostate -> CAGRA picks a heuristic.
  } else {
    std::cerr << "Unknown build algo '" << build_algo << "', using 'auto'." << std::endl;
  }

  return index_params;
}

template <typename T>
void profile_cagra_build(raft::device_resources const& dev_resources,
                         raft::device_matrix_view<const T, int64_t> dataset,
                         const cuvs::neighbors::cagra::index_params& index_params,
                         int warmup_iters,
                         int timed_iters)
{
  using namespace cuvs::neighbors;

  std::cout << "Dataset: " << dataset.extent(0) << " x " << dataset.extent(1) << std::endl;
  std::cout << "intermediate_graph_degree=" << index_params.intermediate_graph_degree
            << " graph_degree=" << index_params.graph_degree << std::endl;

  // Warmup passes: amortize one-time costs (context, allocator pool growth,
  // kernel JIT) so the profiled region reflects steady-state construction.
  for (int i = 0; i < warmup_iters; ++i) {
    std::cout << "Warmup build " << (i + 1) << "/" << warmup_iters << std::endl;
    auto index = cagra::build(dev_resources, index_params, dataset);
    raft::resource::sync_stream(dev_resources);
  }

  // Timed passes: this is the region you want to capture in the profiler.
  for (int i = 0; i < timed_iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    auto index = cagra::build(dev_resources, index_params, dataset);
    raft::resource::sync_stream(dev_resources);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Build " << (i + 1) << "/" << timed_iters << ": " << elapsed.count() << " s, "
              << index.size() << " vectors, graph degree " << index.graph_degree() << std::endl;
  }
}

void usage()
{
  std::cout
    << "Usage: ./CAGRA_BUILD_PROFILE <data filename> <datatype> "
       "[graph_degree] [intermediate_graph_degree] [build_algo] [warmup_iters] [timed_iters] "
       "[max_N]\n\n"
       "  data filename               binary file: [uint32 N][uint32 dim][N*dim values]\n"
       "  datatype                    one of: float, int8, uint8\n"
       "  graph_degree                output graph degree (default 64)\n"
       "  intermediate_graph_degree   pruning graph degree (default 128)\n"
       "  build_algo                  auto | ivf_pq | nn_descent (default auto)\n"
       "  warmup_iters                untimed warmup builds (default 1)\n"
       "  timed_iters                 timed builds (default 3)\n"
       "  max_N                       cap number of loaded rows (default all)\n";
  std::exit(1);
}

int main(int argc, char* argv[])
{
  if (argc < 3) usage();

  std::string data_fname              = argv[1];
  std::string dtype                   = argv[2];
  size_t graph_degree                 = (argc > 3) ? std::stoul(argv[3]) : 64;
  size_t intermediate_graph_degree    = (argc > 4) ? std::stoul(argv[4]) : 128;
  std::string build_algo              = (argc > 5) ? argv[5] : "auto";
  int warmup_iters                    = (argc > 6) ? std::atoi(argv[6]) : 1;
  int timed_iters                     = (argc > 7) ? std::atoi(argv[7]) : 3;
  int max_N                           = (argc > 8) ? std::atoi(argv[8]) : INT_MAX;

  raft::device_resources dev_resources;

  // Pool allocator avoids repeated cudaMalloc/cudaFree churn from polluting the
  // profile across iterations.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  auto index_params = make_index_params(intermediate_graph_degree, graph_degree, build_algo);

  if (dtype == "float") {
    auto dataset = read_bin_dataset<float, int64_t>(dev_resources, data_fname, max_N);
    profile_cagra_build<float>(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               index_params,
                               warmup_iters,
                               timed_iters);
  } else if (dtype == "int8") {
    auto dataset = read_bin_dataset<int8_t, int64_t>(dev_resources, data_fname, max_N);
    profile_cagra_build<int8_t>(dev_resources,
                                raft::make_const_mdspan(dataset.view()),
                                index_params,
                                warmup_iters,
                                timed_iters);
  } else if (dtype == "uint8") {
    auto dataset = read_bin_dataset<uint8_t, int64_t>(dev_resources, data_fname, max_N);
    profile_cagra_build<uint8_t>(dev_resources,
                                 raft::make_const_mdspan(dataset.view()),
                                 index_params,
                                 warmup_iters,
                                 timed_iters);
  } else {
    std::cerr << "Unsupported datatype: " << dtype << std::endl;
    usage();
  }

  return 0;
}

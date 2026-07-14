# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(cuvs_experiments)

project(cuvs_experiments LANGUAGES CXX CUDA)

rapids_cpm_init()
include(../../cpp/cmake/thirdparty/get_cutlass.cmake)
set(BUILD_CUVS_C_LIBRARY OFF)
include(../cmake/thirdparty/get_cuvs.cmake)

add_executable(CAGRA_RANDOM_SPLIT_MERGE_RECALL src/cagra_random_split_merge_recall.cu)
target_link_libraries(
  CAGRA_RANDOM_SPLIT_MERGE_RECALL PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(CAGRA_RANDOM_KWAY_SPLIT_BUILD src/cagra_random_kway_split_build.cu)
set_target_properties(
  CAGRA_RANDOM_KWAY_SPLIT_BUILD PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_RANDOM_KWAY_SPLIT_BUILD PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(CAGRA_SPLIT_MERGE_CONSTRUCTION src/cagra_split_merge_construction.cu)
target_include_directories(
  CAGRA_SPLIT_MERGE_CONSTRUCTION PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src
)
set_target_properties(
  CAGRA_SPLIT_MERGE_CONSTRUCTION PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_SPLIT_MERGE_CONSTRUCTION PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(
  CAGRA_MERGE_API_BENCH src/cagra_merge_api_bench.cu src/kmeans_min_cluster_fused.cu)
target_include_directories(
  CAGRA_MERGE_API_BENCH
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src
)
set_target_properties(
  CAGRA_MERGE_API_BENCH PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_MERGE_API_BENCH
  PRIVATE
    cuvs::cuvs
    CUDA::cuda_driver
    nvidia::cutlass::cutlass
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

foreach(leaf_size IN ITEMS 64 128 256 512)
  set(target CAGRA_MERGE_API_BENCH_LEAF_${leaf_size})
  add_executable(
    ${target} src/cagra_merge_api_bench.cu src/kmeans_min_cluster_fused.cu)
  target_include_directories(
    ${target}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src
  )
  target_compile_definitions(
    ${target} PRIVATE CUVS_MERGE_SCAFFOLD_CLUSTER_SIZE=${leaf_size}
  )
  set_target_properties(
    ${target}
    PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20 OUTPUT_NAME cagra_merge_api_bench_leaf_${leaf_size}
  )
  target_link_libraries(
    ${target}
    PRIVATE
      cuvs::cuvs
      CUDA::cuda_driver
      nvidia::cutlass::cutlass
      $<TARGET_NAME_IF_EXISTS:conda_env>
  )
endforeach()

add_executable(
  CAGRA_PARTITION_QUALITY_BENCH
  src/cagra_partition_quality_bench.cu src/kmeans_min_cluster_fused.cu)
target_include_directories(
  CAGRA_PARTITION_QUALITY_BENCH
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src
)
set_target_properties(
  CAGRA_PARTITION_QUALITY_BENCH PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_PARTITION_QUALITY_BENCH
  PRIVATE
    cuvs::cuvs
    CUDA::cuda_driver
    nvidia::cutlass::cutlass
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(CAGRA_STREAMING_BUILD_BENCH src/cagra_streaming_build_bench.cu)
target_include_directories(
  CAGRA_STREAMING_BUILD_BENCH
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src
)
set_target_properties(
  CAGRA_STREAMING_BUILD_BENCH PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_STREAMING_BUILD_BENCH
  PRIVATE cuvs::cuvs CUDA::cuda_driver $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(CAGRA_BUILD_ALGO_COMPARE src/cagra_build_algo_compare.cu)
set_target_properties(
  CAGRA_BUILD_ALGO_COMPARE PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_BUILD_ALGO_COMPARE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)

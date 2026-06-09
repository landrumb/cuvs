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

add_executable(CAGRA_BUILD_ALGO_COMPARE src/cagra_build_algo_compare.cu)
set_target_properties(
  CAGRA_BUILD_ALGO_COMPARE PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20
)
target_link_libraries(
  CAGRA_BUILD_ALGO_COMPARE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)

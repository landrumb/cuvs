# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(cuvs_experiments)

project(cuvs_experiments LANGUAGES CXX CUDA)
find_package(Threads)

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../cmake/thirdparty/get_cuvs.cmake)

# -------------- compile tasks ----------------- #
add_executable(CAGRA_BUILD_PROFILE src/cagra_build_profile.cu)
add_executable(CAGRA_EXAMPLE src/cagra_example.cu)
add_executable(CAGRA_IVF_PQ_NNDESCENT_EXAMPLE src/cagra_ivf_pq_nndescent_example.cu)
add_executable(CAGRA_BUILD_COMPARE src/cagra_build_compare.cu)
add_executable(CAGRA_IVF_PQ_RANDOM_HYBRID_EXAMPLE src/cagra_ivf_pq_random_hybrid_example.cu)
add_executable(CAGRA_SPLIT_SEARCH src/cagra_split_search.cu)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(CAGRA_BUILD_PROFILE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_IVF_PQ_NNDESCENT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_BUILD_COMPARE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_IVF_PQ_RANDOM_HYBRID_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_SPLIT_SEARCH PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)

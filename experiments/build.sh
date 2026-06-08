#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cuvs experiments build script

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg clean; then
  rm -rf cpp/build
  exit 0
fi

function gpuArch {
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-gpu\-arch" || true; } ) ]]; then
        GPU_ARCH_ARG=$(echo "$ARGS" | { grep -Eo "\-\-gpu\-arch=.+( |$)" || true; })
        if [[ -n ${GPU_ARCH_ARG} ]]; then
            echo "${GPU_ARCH_ARG}" | sed -e 's/--gpu-arch=//' -e 's/ .*//'
            return
        fi
    fi
    if hasArg --allgpuarch; then
        echo "RAPIDS"
        return
    fi
    # Default to the architecture of the GPU in the system
    echo "NATIVE"
}

# Set up build configuration
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
BUILD_TYPE=Release
CUVS_REPO_REL=""
EXTRA_CMAKE_ARGS=()

CUVS_CMAKE_CUDA_ARCHITECTURES=$(gpuArch)

# Root of experiments
EXPERIMENTS_DIR=$(dirname "$(realpath "$0")")

# Reuse the already-built libcuvs from ../cpp/build when present; otherwise CPM
# fetches and builds cuvs from source (slow). Override with CPM_cuvs_SOURCE or
# LIB_BUILD_DIR.
if [[ ${CUVS_REPO_REL} != "" ]]; then
  CUVS_REPO_PATH=$(readlink -f "${CUVS_REPO_REL}")
  EXTRA_CMAKE_ARGS+=("-DCPM_cuvs_SOURCE=${CUVS_REPO_PATH}")
else
  LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXPERIMENTS_DIR}/../cpp/build")}
  EXTRA_CMAKE_ARGS+=("-Dcuvs_ROOT=${LIB_BUILD_DIR}")
fi

build_experiment() {
  experiment_dir=${1}
  experiment_dir="${EXPERIMENTS_DIR}/${experiment_dir}"
  build_dir="${experiment_dir}/build"

  cmake -S "${experiment_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCUVS_NVTX=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUVS_CMAKE_CUDA_ARCHITECTURES}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "${EXTRA_CMAKE_ARGS[@]}"

  cmake --build "${build_dir}" -j"${PARALLEL_LEVEL}"
}

build_experiment cpp

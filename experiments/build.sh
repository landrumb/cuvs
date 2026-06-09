#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cuVS experiments build script

set -e -u -o pipefail

NUMARGS=$#
ARGS=$*

EXPERIMENTS_DIR=$(cd "$(dirname "$0")"; pwd)
CPP_SOURCE_DIR="${EXPERIMENTS_DIR}/cpp"
CPP_BUILD_DIR="${CPP_BUILD_DIR:-${CPP_SOURCE_DIR}/build}"
LIB_BUILD_DIR="${LIB_BUILD_DIR:-$(readlink -f "${EXPERIMENTS_DIR}/../cpp/build")}"

BUILD_TYPE="${BUILD_TYPE:-Release}"
PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)}"
EXTRA_CMAKE_ARGS=()
CACHE_ARGS=()

HELP="$0 [<target>] [<flag> ...] [--cmake-args=\"<args>\"] [--cache-tool=<tool>]
 where <target> is:
   clean            - remove experiments build artifacts

 and <flag> is:
   -g                          - build for debug
   --allgpuarch                - build for all supported GPU architectures
   --gpu-arch=\"<arch>\"        - build for specific GPU architectures (e.g. \"80-real;90-real\")
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options
   --cache-tool=<tool>         - pass a build cache tool (e.g. ccache, sccache, distcc)
   -h                          - print this text

 default action (no args) is to build all experiments
"

function hasArg {
  (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
  if [[ $(echo "$ARGS" | { grep -Eo "\-\-cmake\-args" || true; } | wc -l) -gt 1 ]]; then
    echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
    exit 1
  fi

  if [[ -n $(echo "$ARGS" | { grep -E "\-\-cmake\-args" || true; }) ]]; then
    local extra_cmake_args
    extra_cmake_args=$(echo "$ARGS" | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
    if [[ -n ${extra_cmake_args} ]]; then
      ARGS=${ARGS//$extra_cmake_args/}
      extra_cmake_args=$(echo "$extra_cmake_args" | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
      read -ra EXTRA_CMAKE_ARGS <<< "$extra_cmake_args"
    fi
  fi
}

function cacheTool {
  if [[ $(echo "$ARGS" | { grep -Eo "\-\-cache\-tool" || true; } | wc -l) -gt 1 ]]; then
    echo "Multiple --cache-tool options were provided, please provide only one: ${ARGS}"
    exit 1
  fi

  if [[ -n $(echo "$ARGS" | { grep -E "\-\-cache\-tool" || true; }) ]]; then
    local cache_tool
    cache_tool=$(echo "$ARGS" | sed -e 's/.*--cache-tool=//' -e 's/ .*//')
    if [[ -n ${cache_tool} ]]; then
      ARGS=${ARGS//--cache-tool=$cache_tool/}
      CACHE_ARGS=("-DCMAKE_CUDA_COMPILER_LAUNCHER=${cache_tool}"
                  "-DCMAKE_C_COMPILER_LAUNCHER=${cache_tool}"
                  "-DCMAKE_CXX_COMPILER_LAUNCHER=${cache_tool}")
    fi
  fi
}

function gpuArch {
  if hasArg --allgpuarch && [[ -n $(echo "$ARGS" | { grep -E "\-\-gpu\-arch" || true; }) ]]; then
    echo "Error: Cannot specify both --gpu-arch and --allgpuarch"
    echo "Use either:"
    echo "  --gpu-arch=\"80-real;90-real\"    (for specific architectures)"
    echo "  --allgpuarch        (for all supported architectures)"
    exit 1
  fi

  if [[ $(echo "$ARGS" | { grep -Eo "\-\-gpu\-arch" || true; } | wc -l) -gt 1 ]]; then
    echo "Error: Multiple --gpu-arch options were provided. Please combine architectures into a single option."
    echo "Instead of: --gpu-arch=80-real --gpu-arch=90-real"
    echo "Use:       --gpu-arch=\"80-real;90-real\""
    exit 1
  fi

  if [[ -n $(echo "$ARGS" | { grep -E "\-\-gpu\-arch" || true; }) ]]; then
    local gpu_arch_arg
    gpu_arch_arg=$(echo "$ARGS" | { grep -Eo "\-\-gpu\-arch=.+( |$)" || true; })
    if [[ -n ${gpu_arch_arg} ]]; then
      echo "${gpu_arch_arg}" | sed -e 's/--gpu-arch=//' -e 's/ .*//'
      return
    fi
  fi

  if hasArg --allgpuarch; then
    echo "RAPIDS"
    return
  fi

  echo "NATIVE"
}

function validArgs {
  if [[ -n ${ARGS// /} ]]; then
    for arg in ${ARGS}; do
      case ${arg} in
        clean|-g|-h|--allgpuarch|--gpu-arch=*) ;;
        *)
          echo "Invalid option: ${arg}"
          echo "${HELP}"
          exit 1
          ;;
      esac
    done
  fi
}

if hasArg -h; then
  echo "${HELP}"
  exit 0
fi

if hasArg -g; then
  BUILD_TYPE=Debug
fi

if hasArg clean; then
  rm -rf "${CPP_BUILD_DIR}"
  exit 0
fi

cmakeArgs
cacheTool
CUVS_CMAKE_CUDA_ARCHITECTURES=$(gpuArch)
validArgs

case ${CUVS_CMAKE_CUDA_ARCHITECTURES} in
  "RAPIDS") echo "Building for *ALL* supported GPU architectures..." ;;
  "NATIVE") echo "Building for the architecture of the GPU in the system..." ;;
  *) echo "Building for specified GPU architectures: ${CUVS_CMAKE_CUDA_ARCHITECTURES}" ;;
esac

cmake -S "${CPP_SOURCE_DIR}" -B "${CPP_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -Dcuvs_ROOT="${LIB_BUILD_DIR}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUVS_CMAKE_CUDA_ARCHITECTURES}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "${CACHE_ARGS[@]}" \
  "${EXTRA_CMAKE_ARGS[@]}"

cmake --build "${CPP_BUILD_DIR}" -j"${PARALLEL_LEVEL}"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="${script_dir}/run_scaffold_degree_quality.sh"
binary="${BINARY:-${script_dir}/build-local/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/merge_api_results/scaffold_cap64_repeat_quality.csv}"
parts="${PARTS:-128}"

run_sweep() {
  local neighbors="$1"
  local repeats="$2"
  env \
    BINARY="${binary}" \
    OUTPUT_CSV="${output_csv}" \
    PARTS="${parts}" \
    NEIGHBORS="${neighbors}" \
    REPEATS="${repeats}" \
    CANDIDATE_CAPS="0 64" \
    bash "${runner}"
}

# k8-r32 would create a 256-edge union, one past the current uint8 degree limit.
# Run every valid k4/k8 repeat in one shared-input pass, then add k4-r32.
run_sweep "4 8" "1 2 4 8 16"
run_sweep "4" "32"

echo "COMPLETE output_csv=${output_csv}"

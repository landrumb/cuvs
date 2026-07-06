#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary="${BINARY:-${script_dir}/build-local/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/merge_api_results/fanin_2_128_raw.csv}"

parts=(2 4 8 16 32 64 128)
implementations=(
  k4-scaffold
  k4-scaffold-repeat8
  k4-scaffold-repeat16
  k4-scaffold-repeat32
  binary-cross-query
)

has_result() {
  local dataset="$1"
  local part_count="$2"
  local implementation="$3"
  [[ -f "${output_csv}" ]] || return 1
  awk -F, -v dataset="${dataset}" -v parts="${part_count}" -v implementation="${implementation}" '
    NR > 1 && $1 == dataset && $2 == parts && $3 == implementation { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${output_csv}"
}

run_case() {
  local label="$1"
  local dataset="$2"
  local queries="$3"
  local groundtruth="$4"
  local part_count="$5"
  local implementation="$6"
  local cli_implementation="${implementation}"

  if [[ "${implementation}" == k4-scaffold-repeat* ]]; then
    cli_implementation=k4-scaffold
  fi

  if has_result "${label}" "${part_count}" "${implementation}"; then
    echo "SKIP dataset=${label} parts=${part_count} implementation=${implementation}"
    return
  fi

  echo "START dataset=${label} parts=${part_count} implementation=${implementation}"
  args=(
    "${binary}"
    --dataset "${dataset}"
    --queries "${queries}"
    --groundtruth "${groundtruth}"
    --output-csv "${output_csv}"
    --label "${label}"
    --implementation "${cli_implementation}"
    --parts "${part_count}"
    --graph-degree 64
    --intermediate-graph-degree 128
    --itopk-size 160
  )
  case "${implementation}" in
    k4-scaffold) args+=(--scaffold-repeats 2 --scaffold-candidate-cap 0) ;;
    k4-scaffold-repeat8) args+=(--scaffold-repeats 8 --scaffold-candidate-cap 0) ;;
    k4-scaffold-repeat16) args+=(--scaffold-repeats 16 --scaffold-candidate-cap 0) ;;
    k4-scaffold-repeat32) args+=(--scaffold-repeats 32 --scaffold-candidate-cap 0) ;;
  esac
  "${args[@]}"
}

run_dataset() {
  local label="$1"
  local dataset="$2"
  local queries="$3"
  local groundtruth="$4"
  for part_count in "${parts[@]}"; do
    for implementation in "${implementations[@]}"; do
      run_case "${label}" "${dataset}" "${queries}" "${groundtruth}" \
        "${part_count}" "${implementation}"
    done
  done
}

mkdir -p "$(dirname -- "${output_csv}")"

run_dataset \
  Wiki-1M \
  /home/coder/cuvs/datasets/wiki_all_1M/base.1M.fbin \
  /home/coder/cuvs/datasets/wiki_all_1M/queries.fbin \
  /home/coder/cuvs/datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin

run_dataset \
  OpenAI-2M \
  /raid/blandrum/openai/openai_base.bin \
  /raid/blandrum/openai/openai_query.bin \
  /raid/blandrum/openai/openai-2M.GT

run_dataset \
  YFCC-10M \
  /raid/blandrum/yfcc/base.10M.u8bin \
  /raid/blandrum/yfcc/query.public.100K.u8bin \
  /raid/blandrum/yfcc/unfiltered.GT.public.ibin

echo "COMPLETE output_csv=${output_csv}"

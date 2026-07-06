#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary="${BINARY:-${script_dir}/build-local/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/merge_api_results/scaffold_seed_quality.csv}"
part_count="${PARTS:-128}"
repeat_count="${REPEATS:-2}"
neighbors_per_leaf="${NEIGHBORS:-4}"
quality_sample_rows="${QUALITY_SAMPLE_ROWS:-65536}"
read -r -a seeds <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 1234}"

has_result() {
  local dataset="$1"
  local seed="$2"
  [[ -f "${output_csv}" ]] || return 1
  awk -F, -v dataset="${dataset}" -v parts="${part_count}" \
    -v repeats="${repeat_count}" -v neighbors="${neighbors_per_leaf}" -v seed="${seed}" '
    NR > 1 && $1 == dataset && $2 == parts && $10 == repeats &&
      $11 == neighbors && $12 == neighbors && $13 == seed { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${output_csv}"
}

run_case() {
  local label="$1"
  local dataset="$2"
  local queries="$3"
  local groundtruth="$4"
  local missing=()
  local seed
  for seed in "${seeds[@]}"; do
    if ! has_result "${label}" "${seed}"; then
      missing+=("${seed}")
    fi
  done
  if ((${#missing[@]} == 0)); then
    echo "SKIP dataset=${label} parts=${part_count} seeds=complete"
    return
  fi

  local seed_list
  seed_list="$(IFS=,; echo "${missing[*]}")"
  echo "START dataset=${label} parts=${part_count} repeats=${repeat_count} neighbors=${neighbors_per_leaf} seeds=${seed_list}"
  "${binary}" \
    --dataset "${dataset}" \
    --queries "${queries}" \
    --groundtruth "${groundtruth}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --implementation k4-scaffold \
    --parts "${part_count}" \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --itopk-size 160 \
    --scaffold-repeats "${repeat_count}" \
    --scaffold-neighbors "${neighbors_per_leaf}" \
    --scaffold-seed-list "${seed_list}" \
    --scaffold-quality \
    --quality-sample-rows "${quality_sample_rows}" \
    --scaffold-candidate-cap 0
}

mkdir -p "$(dirname -- "${output_csv}")"

run_case \
  Wiki-1M \
  /home/coder/cuvs/datasets/wiki_all_1M/base.1M.fbin \
  /home/coder/cuvs/datasets/wiki_all_1M/queries.fbin \
  /home/coder/cuvs/datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin

run_case \
  OpenAI-2M \
  /raid/blandrum/openai/openai_base.bin \
  /raid/blandrum/openai/openai_query.bin \
  /raid/blandrum/openai/openai-2M.GT

run_case \
  YFCC-10M \
  /raid/blandrum/yfcc/base.10M.u8bin \
  /raid/blandrum/yfcc/query.public.100K.u8bin \
  /raid/blandrum/yfcc/unfiltered.GT.public.ibin

echo "COMPLETE output_csv=${output_csv}"

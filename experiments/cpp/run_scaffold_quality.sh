#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary="${BINARY:-${script_dir}/build-local/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/merge_api_results/scaffold_rank_quality.csv}"
quality_sample_rows="${QUALITY_SAMPLE_ROWS:-65536}"
read -r -a parts <<< "${PARTS:-128 8 2 32}"
read -r -a repeats <<< "${REPEATS:-1 2 4 8 16 32}"

has_result() {
  local dataset="$1"
  local part_count="$2"
  local repeat_count="$3"
  [[ -f "${output_csv}" ]] || return 1
  awk -F, -v dataset="${dataset}" -v parts="${part_count}" -v repeats="${repeat_count}" '
    NR > 1 && $1 == dataset && $2 == parts && $10 == repeats { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${output_csv}"
}

run_case() {
  local label="$1"
  local dataset="$2"
  local queries="$3"
  local groundtruth="$4"
  local part_count="$5"
  local missing=()
  local repeat_count
  for repeat_count in "${repeats[@]}"; do
    if ! has_result "${label}" "${part_count}" "${repeat_count}"; then
      missing+=("${repeat_count}")
    fi
  done
  if ((${#missing[@]} == 0)); then
    echo "SKIP dataset=${label} parts=${part_count} repeats=complete"
    return
  fi

  local repeat_list
  repeat_list="$(IFS=,; echo "${missing[*]}")"
  echo "START dataset=${label} parts=${part_count} repeats=${repeat_list}"
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
    --scaffold-repeat-list "${repeat_list}" \
    --scaffold-quality \
    --quality-sample-rows "${quality_sample_rows}" \
    --scaffold-candidate-cap 0
}

mkdir -p "$(dirname -- "${output_csv}")"

for part_count in "${parts[@]}"; do
  run_case \
    Wiki-1M \
    /home/coder/cuvs/datasets/wiki_all_1M/base.1M.fbin \
    /home/coder/cuvs/datasets/wiki_all_1M/queries.fbin \
    /home/coder/cuvs/datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin \
    "${part_count}"

  run_case \
    OpenAI-2M \
    /raid/blandrum/openai/openai_base.bin \
    /raid/blandrum/openai/openai_query.bin \
    /raid/blandrum/openai/openai-2M.GT \
    "${part_count}"

  run_case \
    YFCC-10M \
    /raid/blandrum/yfcc/base.10M.u8bin \
    /raid/blandrum/yfcc/query.public.100K.u8bin \
    /raid/blandrum/yfcc/unfiltered.GT.public.ibin \
    "${part_count}"
done

echo "COMPLETE output_csv=${output_csv}"

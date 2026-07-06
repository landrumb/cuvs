#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary="${BINARY:-${script_dir}/build-local/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/merge_api_results/scaffold_degree_quality.csv}"
quality_sample_rows="${QUALITY_SAMPLE_ROWS:-65536}"
seed="${SEED:-1234}"
candidate_cap="${CANDIDATE_CAP:-0}"
read -r -a candidate_caps <<< "${CANDIDATE_CAPS:-}"
read -r -a parts <<< "${PARTS:-128}"
read -r -a neighbors <<< "${NEIGHBORS:-1 2 8 16 32}"
read -r -a repeats <<< "${REPEATS:-1 2 4}"
read -r -a first_neighbors <<< "${FIRST_NEIGHBORS:-}"

has_complete_result() {
  local dataset="$1"
  local part_count="$2"
  [[ -f "${output_csv}" ]] || return 1

  local -a configured_first_neighbors=("${first_neighbors[@]}")
  if ((${#configured_first_neighbors[@]} == 0)); then
    configured_first_neighbors=(0)
  fi
  local -a configured_caps=("${candidate_caps[@]}")
  if ((${#configured_caps[@]} == 0)); then
    configured_caps=("${candidate_cap}")
  fi

  local expected=0 desired="|"
  local neighbors_per_leaf first_config first_degree repeat_count cap candidate_degree
  for neighbors_per_leaf in "${neighbors[@]}"; do
    for first_config in "${configured_first_neighbors[@]}"; do
      first_degree="${first_config}"
      if ((first_degree == 0)); then
        first_degree="${neighbors_per_leaf}"
      fi
      for repeat_count in "${repeats[@]}"; do
        candidate_degree=$((64 + first_degree + (repeat_count - 1) * neighbors_per_leaf))
        for cap in "${configured_caps[@]}"; do
          if ((cap > candidate_degree)); then
            continue
          fi
          desired+="${repeat_count},${neighbors_per_leaf},${first_degree},${cap}|"
          expected=$((expected + 1))
        done
      done
    done
  done

  local observed
  observed="$(awk -F, -v dataset="${dataset}" -v parts="${part_count}" \
    -v seed="${seed}" -v desired="${desired}" '
    NR > 1 && $1 == dataset && $2 == parts && $13 == seed {
      key = "|" $10 "," $11 "," $12 "," $14 "|"
      if (index(desired, key) > 0) { seen[key] = 1 }
    }
    END { print length(seen) }
  ' "${output_csv}")"
  [[ "${observed}" -eq "${expected}" ]]
}

run_case() {
  local label="$1"
  local dataset="$2"
  local queries="$3"
  local groundtruth="$4"
  local part_count="$5"
  if has_complete_result "${label}" "${part_count}"; then
    echo "SKIP dataset=${label} parts=${part_count} configurations=complete"
    return
  fi

  local neighbor_list repeat_list first_neighbor_list candidate_cap_list
  neighbor_list="$(IFS=,; echo "${neighbors[*]}")"
  repeat_list="$(IFS=,; echo "${repeats[*]}")"
  first_neighbor_list="$(IFS=,; echo "${first_neighbors[*]}")"
  candidate_cap_list="$(IFS=,; echo "${candidate_caps[*]}")"
  echo "START dataset=${label} parts=${part_count} neighbors=${neighbor_list} first_neighbors=${first_neighbor_list:-default} repeats=${repeat_list} caps=${candidate_cap_list:-${candidate_cap}}"
  local args=(
    --dataset "${dataset}"
    --queries "${queries}"
    --groundtruth "${groundtruth}"
    --output-csv "${output_csv}"
    --label "${label}"
    --implementation k4-scaffold
    --parts "${part_count}"
    --graph-degree 64
    --intermediate-graph-degree 128
    --itopk-size 160
    --scaffold-repeat-list "${repeat_list}"
    --scaffold-neighbor-list "${neighbor_list}"
    --scaffold-seed "${seed}"
    --scaffold-quality
    --quality-sample-rows "${quality_sample_rows}"
  )
  if [[ -n "${first_neighbor_list}" ]]; then
    args+=(--scaffold-first-repeat-neighbor-list "${first_neighbor_list}")
  fi
  if [[ -n "${candidate_cap_list}" ]]; then
    args+=(--scaffold-candidate-cap-list "${candidate_cap_list}")
  else
    args+=(--scaffold-candidate-cap "${candidate_cap}")
  fi
  "${binary}" "${args[@]}"
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

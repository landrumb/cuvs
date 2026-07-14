#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
binary="${BINARY:-${experiment_root}/build-presentation/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/data/partition_variants.csv}"
read -r -a datasets <<< "${DATASETS:-Wiki-1M OpenAI-2M YFCC-10M}"
read -r -a parts <<< "${PARTS:-2 8 128}"
mkdir -p "$(dirname -- "${output_csv}")" "${script_dir}/logs"

dataset_paths() {
  case "$1" in
    Wiki-1M)
      printf '%s\n' \
        /raid/blandrum/local_datasets/wiki_all_1M/base.1M.fbin \
        /raid/blandrum/local_datasets/wiki_all_1M/queries.fbin \
        /raid/blandrum/local_datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin
      ;;
    OpenAI-2M)
      printf '%s\n' \
        /raid/blandrum/openai/openai_base.bin \
        /raid/blandrum/openai/openai_query.bin \
        /raid/blandrum/openai/openai-2M.GT
      ;;
    YFCC-10M)
      printf '%s\n' \
        /raid/blandrum/yfcc/base.10M.u8bin \
        /raid/blandrum/yfcc/query.public.100K.u8bin \
        /raid/blandrum/yfcc/unfiltered.GT.public.ibin
      ;;
    *)
      echo "Unknown dataset: $1" >&2
      return 1
      ;;
  esac
}

has_result() {
  local label="$1"
  local part_count="$2"
  local implementation="$3"
  [[ -f "${output_csv}" ]] || return 1
  awk -F, -v d="${label}" -v p="${part_count}" -v i="${implementation}" '
    NR > 1 && $1 == d && $2 == p && $3 == i { found = 1 }
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
  local expected="$7"
  shift 7
  if has_result "${label}" "${part_count}" "${expected}"; then
    echo "SKIP dataset=${label} parts=${part_count} implementation=${expected}"
    return
  fi
  local log="${script_dir}/logs/partition_variant_${label}_${part_count}_${implementation}.log"
  echo "START dataset=${label} parts=${part_count} implementation=${implementation}"
  "${binary}" \
    --dataset "${dataset}" \
    --queries "${queries}" \
    --groundtruth "${groundtruth}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --parts "${part_count}" \
    --implementation "${implementation}" \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --itopk-size 160 \
    "$@" 2>&1 | tee -a "${log}"
}

for label in "${datasets[@]}"; do
  mapfile -t paths < <(dataset_paths "${label}")
  for part_count in "${parts[@]}"; do
    run_case "${label}" "${paths[0]}" "${paths[1]}" "${paths[2]}" \
      "${part_count}" ternary-scaffold ternary-scaffold
    run_case "${label}" "${paths[0]}" "${paths[1]}" "${paths[2]}" \
      "${part_count}" native-knn native-knn-k32 --native-knn-degree 32
  done
done

echo "COMPLETE output_csv=${output_csv}"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
binary="${BINARY:-${experiment_root}/build-presentation/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/data/kmeans_merge_pareto.csv}"
read -r -a datasets <<< "${DATASETS:-Wiki-1M OpenAI-2M YFCC-10M}"
read -r -a flat_iterations <<< "${FLAT_ITERATIONS:-1 2 5 10 20}"
target_cluster_size="${TARGET_CLUSTER_SIZE:-256}"
tree_iterations="${TREE_ITERATIONS:-5}"
mkdir -p "$(dirname -- "${output_csv}")" "${script_dir}/logs"

dataset_paths() {
  case "$1" in
    Wiki-1M)
      printf '%s\n' \
        /raid/blandrum/local_datasets/wiki_all_1M/base.1M.fbin \
        /raid/blandrum/local_datasets/wiki_all_1M/queries.fbin \
        /raid/blandrum/local_datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin \
        wiki1m
      ;;
    OpenAI-2M)
      printf '%s\n' \
        /raid/blandrum/openai/openai_base.bin \
        /raid/blandrum/openai/openai_query.bin \
        /raid/blandrum/openai/openai-2M.GT \
        openai2m
      ;;
    YFCC-10M)
      printf '%s\n' \
        /raid/blandrum/yfcc/base.10M.u8bin \
        /raid/blandrum/yfcc/query.public.100K.u8bin \
        /raid/blandrum/yfcc/unfiltered.GT.public.ibin \
        yfcc10m
      ;;
    *)
      echo "Unknown dataset: $1" >&2
      return 1
      ;;
  esac
}

has_result() {
  local label="$1"
  local implementation="$2"
  [[ -f "${output_csv}" ]] || return 1
  awk -F, -v d="${label}" -v i="${implementation}" '
    NR > 1 && $1 == d && $2 == 8 && $3 == i { count++ }
    END { exit(count == 1 ? 0 : 1) }
  ' "${output_csv}"
}

run_case() {
  local label="$1"
  local dataset="$2"
  local queries="$3"
  local groundtruth="$4"
  local key="$5"
  local implementation="$6"
  local expected="$7"
  shift 7
  if has_result "${label}" "${expected}"; then
    echo "SKIP dataset=${label} implementation=${expected}"
    return
  fi
  if [[ -f "${output_csv}" ]] &&
     awk -F, -v d="${label}" -v i="${expected}" '
       NR > 1 && $1 == d && $2 == 8 && $3 == i { found = 1 }
       END { exit(found ? 0 : 1) }
     ' "${output_csv}"; then
    echo "Ambiguous duplicate/partial result: ${label} ${expected}" >&2
    exit 2
  fi
  local log="${script_dir}/logs/kmeans_merge_pareto_${key}_${expected}.log"
  echo "START dataset=${label} implementation=${expected}"
  "${binary}" \
    --dataset "${dataset}" \
    --queries "${queries}" \
    --groundtruth "${groundtruth}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --parts 8 \
    --implementation "${implementation}" \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --itopk-size 160 \
    "$@" 2>&1 | tee -a "${log}"
}

for label in "${datasets[@]}"; do
  mapfile -t paths < <(dataset_paths "${label}")
  for iterations in "${flat_iterations[@]}"; do
    expected="flat-kmeans-target${target_cluster_size}-iter${iterations}-k4-cap64"
    run_case "${label}" "${paths[0]}" "${paths[1]}" "${paths[2]}" "${paths[3]}" \
      flat-kmeans "${expected}" \
      --kmeans-target-cluster-size "${target_cluster_size}" \
      --kmeans-iterations "${iterations}"
  done
  for branching in 2 5; do
    expected="kmeans-tree-b${branching}-leaf256-iter${tree_iterations}-k4-cap64"
    run_case "${label}" "${paths[0]}" "${paths[1]}" "${paths[2]}" "${paths[3]}" \
      kmeans-tree "${expected}" \
      --kmeans-tree-branching "${branching}" \
      --kmeans-iterations "${tree_iterations}"
  done
done

echo "COMPLETE output_csv=${output_csv}"

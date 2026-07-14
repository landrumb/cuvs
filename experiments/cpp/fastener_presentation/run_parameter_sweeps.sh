#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${experiment_root}/build-presentation}"
default_binary="${BINARY:-${build_dir}/CAGRA_MERGE_API_BENCH}"

read -r -a datasets <<< "${DATASETS:-Wiki-1M OpenAI-2M YFCC-10M}"
read -r -a parts <<< "${PARTS:-2 8 128}"
read -r -a repeat_values <<< "${REPEAT_VALUES:-1 2 4 8 16 32}"
read -r -a neighbor_values <<< "${NEIGHBOR_VALUES:-1 2 4 8 16}"
read -r -a leaf_values <<< "${LEAF_VALUES:-64 128 256 512}"

data_dir="${script_dir}/data"
log_dir="${script_dir}/logs"
mkdir -p "${data_dir}" "${log_dir}"

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
      echo "Unknown dataset label: $1" >&2
      return 1
      ;;
  esac
}

has_result() {
  local csv="$1"
  local dataset="$2"
  local part_count="$3"
  local implementation="$4"
  [[ -f "${csv}" ]] || return 1
  awk -F, \
    -v dataset="${dataset}" \
    -v parts="${part_count}" \
    -v implementation="${implementation}" '
      NR > 1 && $1 == dataset && $2 == parts && $3 == implementation {
        found = 1
      }
      END { exit(found ? 0 : 1) }
    ' "${csv}"
}

check_batch() {
  local csv="$1"
  local dataset="$2"
  local part_count="$3"
  shift 3
  local expected=("$@")
  local present=0
  local implementation
  for implementation in "${expected[@]}"; do
    if has_result "${csv}" "${dataset}" "${part_count}" "${implementation}"; then
      present=$((present + 1))
    fi
  done
  if (( present == ${#expected[@]} )); then
    return 0
  fi
  if (( present != 0 )); then
    echo "Partial batch in ${csv}: dataset=${dataset} parts=${part_count} present=${present}/${#expected[@]}" >&2
    echo "Move the partial CSV aside or complete it explicitly before resuming." >&2
    return 2
  fi
  return 1
}

run_benchmark() {
  local family="$1"
  local binary="$2"
  local output_csv="$3"
  local label="$4"
  local dataset="$5"
  local queries="$6"
  local groundtruth="$7"
  local part_count="$8"
  shift 8

  local log_file="${log_dir}/${family}_${label}_${part_count}.log"
  echo "START family=${family} dataset=${label} parts=${part_count} binary=${binary}"
  "${binary}" \
    --dataset "${dataset}" \
    --queries "${queries}" \
    --groundtruth "${groundtruth}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --parts "${part_count}" \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --itopk-size 160 \
    "$@" 2>&1 | tee -a "${log_file}"
}

for label in "${datasets[@]}"; do
  mapfile -t paths < <(dataset_paths "${label}")
  dataset="${paths[0]}"
  queries="${paths[1]}"
  groundtruth="${paths[2]}"

  for part_count in "${parts[@]}"; do
    baseline_csv="${data_dir}/rebuild_baseline.csv"
    if has_result "${baseline_csv}" "${label}" "${part_count}" rebuild; then
      echo "SKIP family=rebuild dataset=${label} parts=${part_count}"
    else
      run_benchmark rebuild "${default_binary}" "${baseline_csv}" \
        "${label}" "${dataset}" "${queries}" "${groundtruth}" "${part_count}" \
        --implementation rebuild
    fi

    repeat_csv="${data_dir}/repeats.csv"
    repeat_implementations=()
    repeat_csv_values=()
    for value in "${repeat_values[@]}"; do
      repeat_implementations+=("k4-scaffold-repeat${value}-cap64")
      repeat_csv_values+=("${value}")
    done
    if check_batch "${repeat_csv}" "${label}" "${part_count}" \
        "${repeat_implementations[@]}"; then
      echo "SKIP family=repeats dataset=${label} parts=${part_count}"
    else
      batch_status=$?
      if (( batch_status == 2 )); then
        exit 2
      fi
      repeat_list="$(IFS=,; echo "${repeat_csv_values[*]}")"
      run_benchmark repeats "${default_binary}" "${repeat_csv}" \
        "${label}" "${dataset}" "${queries}" "${groundtruth}" "${part_count}" \
        --implementation k4-scaffold --scaffold-repeat-list "${repeat_list}"
    fi

    neighbor_csv="${data_dir}/neighbors_per_leaf.csv"
    neighbor_implementations=()
    neighbor_csv_values=()
    for value in "${neighbor_values[@]}"; do
      if [[ "${value}" == 4 ]]; then
        neighbor_implementations+=("k4-scaffold-cap64")
      else
        neighbor_implementations+=("k4-scaffold-k${value}-cap64")
      fi
      neighbor_csv_values+=("${value}")
    done
    if check_batch "${neighbor_csv}" "${label}" "${part_count}" \
        "${neighbor_implementations[@]}"; then
      echo "SKIP family=neighbors dataset=${label} parts=${part_count}"
    else
      batch_status=$?
      if (( batch_status == 2 )); then
        exit 2
      fi
      neighbor_list="$(IFS=,; echo "${neighbor_csv_values[*]}")"
      run_benchmark neighbors "${default_binary}" "${neighbor_csv}" \
        "${label}" "${dataset}" "${queries}" "${groundtruth}" "${part_count}" \
        --implementation k4-scaffold --scaffold-neighbor-list "${neighbor_list}"
    fi

    for leaf_size in "${leaf_values[@]}"; do
      leaf_binary="${build_dir}/cagra_merge_api_bench_leaf_${leaf_size}"
      leaf_csv="${data_dir}/leaf_size_${leaf_size}.csv"
      if has_result "${leaf_csv}" "${label}" "${part_count}" k4-scaffold-cap64; then
        echo "SKIP family=leaf${leaf_size} dataset=${label} parts=${part_count}"
      else
        run_benchmark "leaf${leaf_size}" "${leaf_binary}" "${leaf_csv}" \
          "${label}" "${dataset}" "${queries}" "${groundtruth}" "${part_count}" \
          --implementation k4-scaffold
      fi
    done
  done
done

echo "COMPLETE data_dir=${data_dir}"

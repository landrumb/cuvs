#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
binary="${BINARY:-${experiment_root}/build-presentation/CAGRA_MERGE_API_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/data/wiki_scaling.csv}"
read -r -a sizes <<< "${SIZES:-1000000 2000000 4000000 6000000 8000000 10000000}"
read -r -a parts <<< "${PARTS:-2 8 128}"

dataset=/raid/blandrum/local_datasets/wiki_all_10M/base.10M.fbin
queries=/raid/blandrum/local_datasets/wiki_all_10M/queries.fbin
label=Wiki-10M-prefix
mkdir -p "$(dirname -- "${output_csv}")" "${script_dir}/logs"

groundtruth_for_rows() {
  case "$1" in
    1000000)
      echo /raid/blandrum/local_datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin
      ;;
    2000000)
      echo "${script_dir}/groundtruth/wiki_prefix_2M_neighbors.ibin"
      ;;
    4000000)
      echo "${script_dir}/groundtruth/wiki_prefix_4M_neighbors.ibin"
      ;;
    6000000)
      echo "${script_dir}/groundtruth/wiki_prefix_6M_neighbors.ibin"
      ;;
    8000000)
      echo /raid/blandrum/local_datasets/wiki_all_8M/groundtruth.8M.neighbors.ibin
      ;;
    10000000)
      echo /raid/blandrum/local_datasets/wiki_all_10M/groundtruth.10M.neighbors.ibin
      ;;
    *)
      echo "No exact query ground truth registered for rows=$1" >&2
      return 1
      ;;
  esac
}

has_result() {
  local rows="$1"
  local part_count="$2"
  local implementation="$3"
  [[ -f "${output_csv}" ]] || return 1
  awk -F, \
    -v dataset="${label}" \
    -v parts="${part_count}" \
    -v implementation="${implementation}" \
    -v rows="${rows}" '
      NR > 1 && $1 == dataset && $2 == parts && $3 == implementation && $5 == rows {
        found = 1
      }
      END { exit(found ? 0 : 1) }
    ' "${output_csv}"
}

run_case() {
  local rows="$1"
  local part_count="$2"
  local implementation="$3"
  local expected_implementation="$4"
  local groundtruth="$5"
  if has_result "${rows}" "${part_count}" "${expected_implementation}"; then
    echo "SKIP rows=${rows} parts=${part_count} implementation=${expected_implementation}"
    return
  fi

  log_file="${script_dir}/logs/wiki_scaling_${rows}_${part_count}_${implementation}.log"
  echo "START rows=${rows} parts=${part_count} implementation=${implementation}"
  "${binary}" \
    --dataset "${dataset}" \
    --rows "${rows}" \
    --queries "${queries}" \
    --groundtruth "${groundtruth}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --parts "${part_count}" \
    --implementation "${implementation}" \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --itopk-size 160 2>&1 | tee -a "${log_file}"
}

for rows in "${sizes[@]}"; do
  groundtruth="$(groundtruth_for_rows "${rows}")"
  if [[ ! -f "${groundtruth}" ]]; then
    echo "Missing exact ground truth for rows=${rows}: ${groundtruth}" >&2
    exit 2
  fi
  for part_count in "${parts[@]}"; do
    run_case "${rows}" "${part_count}" rebuild rebuild "${groundtruth}"
    run_case "${rows}" "${part_count}" k4-scaffold k4-scaffold-cap64 "${groundtruth}"
  done
done

echo "COMPLETE output_csv=${output_csv}"

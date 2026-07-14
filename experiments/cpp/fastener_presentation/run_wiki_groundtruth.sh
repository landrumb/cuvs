#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset=/raid/blandrum/local_datasets/wiki_all_10M/base.10M.fbin
queries=/raid/blandrum/local_datasets/wiki_all_10M/queries.fbin
output_dir="${script_dir}/groundtruth"
read -r -a sizes <<< "${SIZES:-2000000 4000000 6000000}"
mkdir -p "${output_dir}" "${script_dir}/logs"

validation_neighbors="${output_dir}/wiki_prefix_1M_validation_neighbors.ibin"
validation_report="${output_dir}/wiki_prefix_1M_validation.json"
if [[ ! -f "${validation_report}" ]]; then
  echo "START exact ground-truth validation rows=1000000"
  python3 "${script_dir}/generate_query_groundtruth.py" \
    --dataset "${dataset}" \
    --queries "${queries}" \
    --rows 1000000 \
    --output "${validation_neighbors}" \
    --k 12 \
    --batch-size "${BATCH_SIZE:-32}" \
    2>&1 | tee -a "${script_dir}/logs/wiki_groundtruth_1M_validation.log"
  python3 "${script_dir}/validate_query_groundtruth.py" \
    --generated "${validation_neighbors}" \
    --reference /raid/blandrum/local_datasets/wiki_all_1M/groundtruth.1M.neighbors.ibin \
    --output "${validation_report}" \
    --k 12 \
    2>&1 | tee -a "${script_dir}/logs/wiki_groundtruth_1M_validation.log"
else
  echo "SKIP exact ground-truth validation report=${validation_report}"

fi
for rows in "${sizes[@]}"; do
  millions=$((rows / 1000000))
  output="${output_dir}/wiki_prefix_${millions}M_neighbors.ibin"
  metadata="${output%.ibin}.json"
  if [[ -f "${output}" && -f "${metadata}" ]]; then
    echo "SKIP exact ground truth rows=${rows} output=${output}"
    continue
  fi
  echo "START exact ground truth rows=${rows} output=${output}"
  python3 "${script_dir}/generate_query_groundtruth.py" \
    --dataset "${dataset}" \
    --queries "${queries}" \
    --rows "${rows}" \
    --output "${output}" \
    --k 12 \
    --batch-size "${BATCH_SIZE:-32}" \
    2>&1 | tee -a "${script_dir}/logs/wiki_groundtruth_${millions}M.log"
done

echo "COMPLETE output_dir=${output_dir}"

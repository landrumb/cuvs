#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
binary="${BINARY:-${experiment_root}/build-presentation/CAGRA_PARTITION_QUALITY_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/data/kmeans_flat_clustering.csv}"
groundtruth_dir="${script_dir}/groundtruth"
kmeans_iterations="${KMEANS_ITERATIONS:-20}"
read -r -a datasets <<< "${DATASETS:-Wiki-1M OpenAI-2M YFCC-10M}"
mkdir -p "$(dirname -- "${output_csv}")" "${script_dir}/logs"

dataset_info() {
  case "$1" in
    Wiki-1M)
      printf '%s\n' /raid/blandrum/local_datasets/wiki_all_1M/base.1M.fbin wiki1m
      ;;
    OpenAI-2M)
      printf '%s\n' /raid/blandrum/openai/openai_base.bin openai2m
      ;;
    YFCC-10M)
      printf '%s\n' /raid/blandrum/yfcc/base.10M.u8bin yfcc10m
      ;;
    *)
      echo "Unknown dataset: $1" >&2
      return 1
      ;;
  esac
}

record_count() {
  local label="$1"
  if [[ ! -f "${output_csv}" ]]; then
    echo 0
    return
  fi
  awk -F, -v d="${label}" 'NR > 1 && $1 == d { count++ } END { print count + 0 }' "${output_csv}"
}

for label in "${datasets[@]}"; do
  mapfile -t info < <(dataset_info "${label}")
  dataset="${info[0]}"
  key="${info[1]}"
  sample_ids="${groundtruth_dir}/${key}_sample_ids.ibin"
  self_gt="${groundtruth_dir}/${key}_self_gt_k12.ibin"
  if [[ ! -f "${sample_ids}" || ! -f "${self_gt}" ]]; then
    echo "Missing self ground truth for ${label}; run run_partition_quality.sh first." >&2
    exit 2
  fi

  count="$(record_count "${label}")"
  if (( count >= 1 )); then
    echo "SKIP flat k-means dataset=${label} records=${count}"
    continue
  fi

  echo "START flat k-means dataset=${label} k=ceil(n/256) iterations=${kmeans_iterations}"
  "${binary}" \
    --dataset "${dataset}" \
    --sample-ids "${sample_ids}" \
    --self-groundtruth "${self_gt}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --repeats 1 \
    --parts 8 \
    --native-degrees 12 \
    --kmeans-flat-only \
    --kmeans-iterations "${kmeans_iterations}" \
    --seed 1234 \
    2>&1 | tee -a "${script_dir}/logs/kmeans_flat_${key}.log"
done

echo "COMPLETE output_csv=${output_csv}"

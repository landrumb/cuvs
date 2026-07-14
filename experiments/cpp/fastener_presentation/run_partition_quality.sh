#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
binary="${BINARY:-${experiment_root}/build-presentation/CAGRA_PARTITION_QUALITY_BENCH}"
output_csv="${OUTPUT_CSV:-${script_dir}/data/partition_quality.csv}"
groundtruth_dir="${script_dir}/groundtruth"
read -r -a datasets <<< "${DATASETS:-Wiki-1M OpenAI-2M YFCC-10M}"
mkdir -p "$(dirname -- "${output_csv}")" "${groundtruth_dir}" "${script_dir}/logs"

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
  metadata="${groundtruth_dir}/${key}_self_gt_k12.json"

  if [[ -f "${sample_ids}" && -f "${self_gt}" && -f "${metadata}" ]]; then
    echo "SKIP self ground truth dataset=${label}"
  else
    echo "START self ground truth dataset=${label}"
    python3 "${script_dir}/generate_self_groundtruth.py" \
      --dataset "${dataset}" \
      --label "${key}" \
      --output-dir "${groundtruth_dir}" \
      --sample-size "${SAMPLE_SIZE:-4096}" \
      --k 12 \
      --seed "${SAMPLE_SEED:-20260709}" \
      --batch-size "${BATCH_SIZE:-32}" \
      2>&1 | tee -a "${script_dir}/logs/self_groundtruth_${key}.log"
  fi

  count="$(record_count "${label}")"
  if (( count >= 60 )); then
    echo "SKIP partition quality dataset=${label} records=${count}"
    continue
  fi
  if (( count != 0 )); then
    echo "Partial partition-quality batch for ${label}: ${count}/60 records" >&2
    echo "Move the partial CSV aside or finish the dataset explicitly before resuming." >&2
    exit 2
  fi

  echo "START partition quality dataset=${label}"
  "${binary}" \
    --dataset "${dataset}" \
    --sample-ids "${sample_ids}" \
    --self-groundtruth "${self_gt}" \
    --output-csv "${output_csv}" \
    --label "${label}" \
    --repeats 1,2,4,8,16,32 \
    --parts 2,8,128 \
    --native-degrees 12,32,64 \
    --seed 1234 \
    2>&1 | tee -a "${script_dir}/logs/partition_quality_${key}.log"
done

echo "COMPLETE output_csv=${output_csv}"

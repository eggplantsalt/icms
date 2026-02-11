#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_offline_icms.sh <NPROC>
NPROC=${1:-1}
PROJ_ROOT=/opt/data/private/openvla_icms

export HF_HOME=${PROJ_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJ_ROOT}/hf_cache
export HF_DATASETS_CACHE=${PROJ_ROOT}/hf_cache
export TORCH_HOME=${PROJ_ROOT}/torch_cache

python -m research.icms.offline_icms \
  --vla_path openvla/openvla-7b \
  --cache_dir ${PROJ_ROOT}/hf_cache \
  --data_root_dir ${PROJ_ROOT}/datasets \
  --probe_root_dir ${PROJ_ROOT}/probe \
  --artifact_dir ${PROJ_ROOT}/artifacts/icms_openvla-7b \
  --batch_size 8 \
  --max_samples 500 \
  --sensitivity_samples 128 \
  --r 128 \
  --epsilon 1e-3 \
  --probe_dataset_name bridge_orig

#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_method_train.sh <NPROC>
NPROC=${1:-1}
PROJ_ROOT=/opt/data/private/openvla_icms

export HF_HOME=${PROJ_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJ_ROOT}/hf_cache
export HF_DATASETS_CACHE=${PROJ_ROOT}/hf_cache
export TORCH_HOME=${PROJ_ROOT}/torch_cache
export WANDB_DIR=${PROJ_ROOT}/wandb

torchrun --standalone --nnodes 1 --nproc-per-node ${NPROC} vla-scripts/finetune.py \
  --config configs/method_icsm_hsw_thermostat.yaml

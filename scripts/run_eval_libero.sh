#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_eval_libero.sh <CHECKPOINT_PATH>
CHECKPOINT_PATH=${1:?"Please provide checkpoint path"}
PROJ_ROOT=/opt/data/private/openvla_icms

export HF_HOME=${PROJ_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJ_ROOT}/hf_cache
export HF_DATASETS_CACHE=${PROJ_ROOT}/hf_cache
export TORCH_HOME=${PROJ_ROOT}/torch_cache

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ${CHECKPOINT_PATH} \
  --task_suite_name libero_spatial \
  --center_crop True \
  --run_id_note icms_hsw \
  --use_wandb False

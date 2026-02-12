#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_baseline_lora.sh <NPROC>
NPROC=${1:-1}
PROJ_ROOT=/opt/data/private/openvla_icms

export HF_HOME=${PROJ_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJ_ROOT}/hf_cache
export HF_DATASETS_CACHE=${PROJ_ROOT}/hf_cache
export TORCH_HOME=${PROJ_ROOT}/torch_cache
export WANDB_DIR=${PROJ_ROOT}/wandb
export PYTHONPATH=/workspace/openvla:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nnodes 1 --nproc-per-node ${NPROC} vla-scripts/finetune.py \
  --config configs/baseline_lora.yaml

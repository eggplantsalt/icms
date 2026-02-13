#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_eval_libero.sh <CHECKPOINT_PATH> [NUM_TRIALS]
CHECKPOINT_PATH=${1:?"Please provide checkpoint path"}
NUM_TRIALS=${2:-3}
PROJ_ROOT=/opt/data/private/openvla_icms
EXP_ID=$(basename "${CHECKPOINT_PATH}")
STATS_DIR=${PROJ_ROOT}/runs/${EXP_ID}
if [[ -f "${CHECKPOINT_PATH}/dataset_statistics.json" ]]; then
  STATS_DIR=${CHECKPOINT_PATH}
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "openvla" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate openvla
fi

export HF_HOME=${PROJ_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJ_ROOT}/hf_cache
export HF_DATASETS_CACHE=${PROJ_ROOT}/hf_cache
export TORCH_HOME=${PROJ_ROOT}/torch_cache
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/openvla:/workspace/openvla/LIBERO:${PYTHONPATH:-}

if [[ -z "${DISPLAY:-}" ]]; then
  if ! command -v Xvfb >/dev/null 2>&1; then
    apt-get update && apt-get install -y libegl-dev xvfb libgl1-mesa-dri libgl1-mesa-dev libgl1-mesa-glx libstdc++6
  fi
  if command -v Xvfb >/dev/null 2>&1; then
    if [[ -f /tmp/.X9-lock ]]; then
      echo "[info] Reusing existing Xvfb :9"
    else
      Xvfb :9 &
      sleep 1
    fi
    export DISPLAY=:9
  else
    echo "[warn] Xvfb not found; if eval fails, install it and set DISPLAY."
  fi
fi
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6.0.34" ]]; then
    ln -sf "${CONDA_PREFIX}/lib/libstdc++.so.6.0.34" "${CONDA_PREFIX}/lib/libstdc++.so.6"
  fi
fi
export MUJOCO_GL=glx

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ${CHECKPOINT_PATH} \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task ${NUM_TRIALS} \
  --dataset_stats_dir ${STATS_DIR} \
  --run_id_note icms_hsw \
  --use_wandb False

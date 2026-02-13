#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_method_train.sh [NPROC] [TARGET_STEPS] [SAVE_STEPS] [EVAL_EVERY] [PATIENCE]
# 推荐：把 `server_profile` 写到 configs/method_icsm_hsw_thermostat.yaml 中，脚本自动切换 4090 / v100x8 参数。
NPROC=${1:-auto}
TARGET_STEPS=${2:-10000}
SAVE_STEPS=${3:-1000}
EVAL_EVERY=${4:-1000}
PATIENCE=${5:-5}
PROJ_ROOT=/opt/data/private/openvla_icms
CFG_PATH=configs/method_icsm_hsw_thermostat.yaml
OPENVLA_ENV=${OPENVLA_ENV:-/root/miniconda3/envs/openvla}
OPENVLA_PY=${OPENVLA_PY:-${OPENVLA_ENV}/bin/python}
OPENVLA_TORCHRUN=${OPENVLA_TORCHRUN:-${OPENVLA_ENV}/bin/torchrun}

if [[ ! -x "${OPENVLA_PY}" ]] || [[ ! -x "${OPENVLA_TORCHRUN}" ]]; then
  echo "[error] openvla env binaries not found under ${OPENVLA_ENV}" >&2
  exit 1
fi

export PATH=${OPENVLA_ENV}/bin:${PATH}

export HF_HOME=${PROJ_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${PROJ_ROOT}/hf_cache
export HF_DATASETS_CACHE=${PROJ_ROOT}/hf_cache
export TORCH_HOME=${PROJ_ROOT}/torch_cache
export WANDB_DIR=${PROJ_ROOT}/wandb
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/openvla:/workspace/openvla/LIBERO:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-8}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-8}

if [[ "${CLEAN_STALE_GPU_PROCS:-0}" == "1" ]]; then
  pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F, '/python/{gsub(/ /, "", $1); print $1}')
  if [[ -n "${pids}" ]]; then
    echo "[warn] Killing stale GPU python processes: ${pids}"
    for p in ${pids}; do
      kill -9 "$p" 2>/dev/null || true
    done
    sleep 1
  fi
fi

SERVER_PROFILE=${SERVER_PROFILE:-$(awk -F': *' '$1=="server_profile" {print $2}' ${CFG_PATH} | sed 's/#.*$//' | tr -d '"' | xargs | tail -n 1)}
SERVER_PROFILE=${SERVER_PROFILE:-4090_1gpu}

case "${SERVER_PROFILE}" in
  4090_1gpu)
    DEFAULT_NPROC=1
    DEFAULT_BATCH=2
    DEFAULT_ACCUM=4
    DEFAULT_PROBE_BATCH=4
    DEFAULT_THERMO_UPDATE=100
    DEFAULT_THERMO_WARMUP=100
    DEFAULT_RLDS_FRAME_CALLS=24
    DEFAULT_RLDS_TRAJ_THREADS=8
    DEFAULT_RLDS_READ_THREADS=8
    ;;
  v100_8gpu)
    DEFAULT_NPROC=8
    DEFAULT_BATCH=1
    DEFAULT_ACCUM=8
    DEFAULT_PROBE_BATCH=4
    DEFAULT_THERMO_UPDATE=200
    DEFAULT_THERMO_WARMUP=100
    DEFAULT_RLDS_FRAME_CALLS=16
    DEFAULT_RLDS_TRAJ_THREADS=2
    DEFAULT_RLDS_READ_THREADS=2
    ;;
  *)
    echo "[error] unknown server_profile='${SERVER_PROFILE}'. use: 4090_1gpu | v100_8gpu" >&2
    exit 1
    ;;
esac

if [[ "${NPROC}" == "auto" ]]; then
  NPROC=${DEFAULT_NPROC}
fi

BATCH_SIZE=${BATCH_SIZE:-${DEFAULT_BATCH}}
GRAD_ACCUM=${GRAD_ACCUM:-${DEFAULT_ACCUM}}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-${DEFAULT_PROBE_BATCH}}
THERMO_UPDATE=${THERMO_UPDATE:-${DEFAULT_THERMO_UPDATE}}
THERMO_WARMUP=${THERMO_WARMUP:-${DEFAULT_THERMO_WARMUP}}
RLDS_FRAME_CALLS=${RLDS_FRAME_CALLS:-${DEFAULT_RLDS_FRAME_CALLS}}
RLDS_TRAJ_THREADS=${RLDS_TRAJ_THREADS:-${DEFAULT_RLDS_TRAJ_THREADS}}
RLDS_READ_THREADS=${RLDS_READ_THREADS:-${DEFAULT_RLDS_READ_THREADS}}

if [[ -z "${RESUME_ADAPTER:-}" ]]; then
  RESUME_ADAPTER=$(ls -1dt ${PROJ_ROOT}/tmp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0002+lora-r16+dropout-0.0--method--image_aug \
    ${PROJ_ROOT}/tmp/openvla-7b+libero_spatial_no_noops+b32+lr-0.0002+lora-r16+dropout-0.0--method--image_aug 2>/dev/null | head -n 1)
fi

if [[ -z "${RESUME_ADAPTER}" ]] || [[ ! -d "${RESUME_ADAPTER}" ]]; then
  echo "[error] RESUME_ADAPTER not found. Set env RESUME_ADAPTER=/path/to/adapter" >&2
  exit 1
fi

if [[ -z "${RESUME_STEP:-}" ]]; then
  RESUME_STEP=$("${OPENVLA_PY}" - <<'PY'
import os, torch
p = os.environ['RESUME_ADAPTER']
s = os.path.join(p, 'trainer_state.pt')
if not os.path.exists(s):
    print(0)
else:
    d = torch.load(s, map_location='cpu')
    print(int(d.get('global_step', 0)))
PY
)
fi

echo "[info] resume_adapter=${RESUME_ADAPTER}"
echo "[info] resume_step=${RESUME_STEP}"
echo "[info] server_profile=${SERVER_PROFILE}"
echo "[info] target_steps=${TARGET_STEPS}, save_steps=${SAVE_STEPS}, eval_every=${EVAL_EVERY}, patience=${PATIENCE}"
echo "[info] nproc=${NPROC}, batch_size=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}, probe_batch=${PROBE_BATCH_SIZE}, thermostat_update=${THERMO_UPDATE}"
echo "[info] rlds_frame_calls=${RLDS_FRAME_CALLS}, rlds_traj_threads=${RLDS_TRAJ_THREADS}, rlds_read_threads=${RLDS_READ_THREADS}"

"${OPENVLA_TORCHRUN}" --standalone --nnodes 1 --nproc_per_node ${NPROC} vla-scripts/finetune.py \
  --config ${CFG_PATH} \
  --learning_rate 2e-4 \
  --batch_size ${BATCH_SIZE} \
  --grad_accumulation_steps ${GRAD_ACCUM} \
  --probe_batch_size ${PROBE_BATCH_SIZE} \
  --thermostat_update_interval ${THERMO_UPDATE} \
  --thermostat_warmup_steps ${THERMO_WARMUP} \
  --rlds_frame_parallel_calls ${RLDS_FRAME_CALLS} \
  --rlds_traj_transform_threads ${RLDS_TRAJ_THREADS} \
  --rlds_traj_read_threads ${RLDS_READ_THREADS} \
  --max_steps ${TARGET_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --resume_adapter_dir ${RESUME_ADAPTER} \
  --resume_global_step ${RESUME_STEP} \
  --enable_periodic_eval True \
  --eval_every_steps ${EVAL_EVERY} \
  --eval_num_trials_per_task 1 \
  --early_stopping_enabled True \
  --early_stopping_patience ${PATIENCE} \
  --early_stopping_min_delta 0.1

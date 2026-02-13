#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash scripts/run_long_train_with_periodic_eval.sh [NPROC] [TARGET_STEPS] [EVAL_EVERY] [NUM_TRIALS]
# 默认: 1卡, 10000步, 每1000步评测一次, 每task 1个trial

NPROC=${1:-1}
TARGET_STEPS=${2:-10000}
EVAL_EVERY=${3:-1000}
NUM_TRIALS=${4:-1}

PROJ_ROOT=/opt/data/private/openvla_icms
CFG_PATH=configs/method_icsm_hsw_thermostat.yaml
RESUME_ADAPTER=${RESUME_ADAPTER:-${PROJ_ROOT}/tmp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0002+lora-r16+dropout-0.0--method--image_aug}
RUN_ID_NOTE=${RUN_ID_NOTE:-method}
OPENVLA_ENV=${OPENVLA_ENV:-/root/miniconda3/envs/openvla}
OPENVLA_PY=${OPENVLA_PY:-${OPENVLA_ENV}/bin/python}
OPENVLA_TORCHRUN=${OPENVLA_TORCHRUN:-${OPENVLA_ENV}/bin/torchrun}
export RESUME_ADAPTER

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

EXP_ID=$(basename "${RESUME_ADAPTER}")
TRAIN_LOG_DIR=${PROJ_ROOT}/runs/train_logs
EVAL_PROGRESS_DIR=${PROJ_ROOT}/runs/eval_progress
mkdir -p "${TRAIN_LOG_DIR}" "${EVAL_PROGRESS_DIR}"

TIMESTAMP=$(date +%Y_%m_%d-%H_%M_%S)
TRAIN_LOG=${TRAIN_LOG_DIR}/TRAIN-${EXP_ID}-${TIMESTAMP}.log
PROGRESS_CSV=${EVAL_PROGRESS_DIR}/PROGRESS-${EXP_ID}.csv

if [[ ! -f "${PROGRESS_CSV}" ]]; then
  echo "timestamp,step,success_rate_percent,eval_log" > "${PROGRESS_CSV}"
fi

read_step() {
  "${OPENVLA_PY}" - <<'PY'
import os, torch
p = os.environ['RESUME_ADAPTER']
s = os.path.join(p, 'trainer_state.pt')
if not os.path.exists(s):
    print(0)
else:
    d = torch.load(s, map_location='cpu')
    print(int(d.get('global_step', 0)))
PY
}

START_STEP=$(read_step)
echo "[info] resume_adapter=${RESUME_ADAPTER}"
echo "[info] start_step=${START_STEP}, target_steps=${TARGET_STEPS}, eval_every=${EVAL_EVERY}"
echo "[info] train_log=${TRAIN_LOG}"
echo "[info] progress_csv=${PROGRESS_CSV}"

if (( START_STEP >= TARGET_STEPS )); then
  echo "[warn] START_STEP (${START_STEP}) >= TARGET_STEPS (${TARGET_STEPS}), skip training."
  exit 0
fi

"${OPENVLA_TORCHRUN}" --standalone --nnodes 1 --nproc_per_node "${NPROC}" vla-scripts/finetune.py \
  --config "${CFG_PATH}" \
  --learning_rate 2e-4 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --max_steps "${TARGET_STEPS}" \
  --save_steps "${EVAL_EVERY}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --resume_adapter_dir "${RESUME_ADAPTER}" \
  --resume_global_step "${START_STEP}" \
  > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "[info] train_pid=${TRAIN_PID}"

NEXT_EVAL=$(( ((START_STEP / EVAL_EVERY) + 1) * EVAL_EVERY ))
LAST_EVAL_STEP=0

while kill -0 "${TRAIN_PID}" >/dev/null 2>&1; do
  sleep 45
  CUR_STEP=$(read_step)
  echo "[monitor] step=${CUR_STEP}, next_eval=${NEXT_EVAL}"

  if (( CUR_STEP >= NEXT_EVAL )) && (( CUR_STEP > LAST_EVAL_STEP )); then
    echo "[eval] trigger at step=${CUR_STEP}"
    bash scripts/run_eval_libero.sh "${RESUME_ADAPTER}" "${NUM_TRIALS}" || true

    LATEST_EVAL_LOG=$(ls -1t ${PROJ_ROOT}/runs/eval_logs/EVAL-libero_spatial-openvla-*.txt | head -n 1)
    RATE=$(grep -Eo '# successes: [0-9]+ \([0-9.]+%\)' "${LATEST_EVAL_LOG}" | tail -n 1 | sed -E 's/.*\(([0-9.]+)%\)/\1/' || true)
    RATE=${RATE:-NA}
    NOW=$(date +%Y-%m-%dT%H:%M:%S)
    echo "${NOW},${CUR_STEP},${RATE},${LATEST_EVAL_LOG}" >> "${PROGRESS_CSV}"

    LAST_EVAL_STEP=${CUR_STEP}
    NEXT_EVAL=$((NEXT_EVAL + EVAL_EVERY))
  fi
done

wait "${TRAIN_PID}" || true
FINAL_STEP=$(read_step)
echo "[done] training exited, final_step=${FINAL_STEP}" 

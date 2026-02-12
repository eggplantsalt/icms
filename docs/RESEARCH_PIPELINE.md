# 研究型微调管线（ICSM + HSW + Thermostat）

本文件提供从 0 到跑通 baseline / method / eval 的完整流程与产物说明，默认不写入 /workspace（全部落盘到 /opt/data/private）。

## 0) 运行环境与镜像注意事项

- 激活环境：`conda activate openvla`
- 若无法连接 HuggingFace：`export HF_ENDPOINT=https://hf-mirror.com`
- 大文件禁止写入 /workspace；数据集保留在 `/opt/data/private/modified_libero_rlds`，缓存与训练产物统一放在 `/opt/data/private/openvla_icms/`。

默认目录（可通过参数覆盖，但必须位于 /opt/data/private）：

```
HF_CACHE=/opt/data/private/openvla_icms/hf_cache
DATA_ROOT=/opt/data/private/modified_libero_rlds
PROBE_ROOT=/opt/data/private/openvla_icms/probe
ARTIFACT_ROOT=/opt/data/private/openvla_icms/artifacts
RUN_ROOT=/opt/data/private/openvla_icms/runs
TMP_ROOT=/opt/data/private/openvla_icms/tmp
```

推荐环境变量：

```bash
export HF_HOME=/opt/data/private/openvla_icms/hf_cache
export TRANSFORMERS_CACHE=/opt/data/private/openvla_icms/hf_cache
export HF_DATASETS_CACHE=/opt/data/private/openvla_icms/hf_cache
export TORCH_HOME=/opt/data/private/openvla_icms/torch_cache
export WANDB_DIR=/opt/data/private/openvla_icms/wandb
```

## 1) 目录结构与产物

离线 ICSM 产物（每层）：

```
/opt/data/private/openvla_icms/artifacts/icms_openvla-7b/
  U<layer>_f.pt
  U<layer>_p.pt
  mu<layer>.pt
  C_T<layer>.pt
  meta.json
```

训练产物：

```
/opt/data/private/openvla_icms/runs/<exp_id>/
  config.yaml
  dataset_statistics.json
  (adapter or merged checkpoint)
```

## 2) 运行命令（1 卡 / 8 卡）

### 2.1 离线 ICSM（Teacher-only）

```bash
bash scripts/run_offline_icms.sh 1
```

可通过直接调用脚本自定义参数：

```bash
python -m research.icms.offline_icms \
  --vla_path openvla/openvla-7b \
  --cache_dir /opt/data/private/openvla_icms/hf_cache \
  --probe_root_dir /opt/data/private/openvla_icms/probe \
  --artifact_dir /opt/data/private/openvla_icms/artifacts/icms_openvla-7b \
  --probe_dataset_name libero_spatial_no_noops \
  --max_samples 500 \
  --sensitivity_samples 128
```

### 2.2 Baseline LoRA 微调

```bash
bash scripts/run_baseline_lora.sh 1
# 或 8 卡
bash scripts/run_baseline_lora.sh 8
```

### 2.3 Method 训练（LoRA + HSW + Thermostat）

```bash
bash scripts/run_method_train.sh 1
# 或 8 卡
bash scripts/run_method_train.sh 8
```

### 2.4 Method 训练（FSDP / train.py 入口，可选）

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --method_enabled True \
  --probe_root_dir /opt/data/private/openvla_icms/probe \
  --artifact_dir /opt/data/private/openvla_icms/artifacts \
  --icms_artifact_dir /opt/data/private/openvla_icms/artifacts/icms_openvla-7b \
  --probe_dataset_name libero_spatial_no_noops \
  --probe_batch_size 16 \
  --hsw_beta 1.0 \
  --hsw_gamma 1.0
```

### 2.5 评测（LIBERO）

```bash
bash scripts/run_eval_libero.sh /opt/data/private/openvla_icms/runs/<exp_id>
```

## 3) 日志字段（method 模式）

在 W&B 或日志输出中包含：

- `drift_d`：漂移 $d = ||C_S - C_T||_F$
- `hsw_beta` / `hsw_gamma`
- `hsw_g_norm` / `hsw_gprime_norm`
- 训练指标：`train_loss`、`action_accuracy`、`l1_loss`

## 4) Sanity Checks

- Hidden states / instruction mask：
  - `python research/probe/smoke_test_hidden_and_mask.py`
  - `python research/probe/smoke_test_alignment.py`
- 离线 ICSM 最小测试：
  - `python research/icms/smoke_test_offline_icms.py`
- HSW 逻辑测试：
  - `python research/hooks/smoke_test_hsw.py`
- Thermostat 更新逻辑测试：
  - `python research/thermostat/smoke_test_thermostat.py`

## 5) 磁盘安全自检

```bash
python scripts/check_disk_safety.py \
  --workspace_root /workspace \
  --allowed_root /opt/data/private \
  --project_root /opt/data/private/openvla_icms \
  --max_workspace_file_gb 1
```

## 6) 关键实现位置

- 离线 ICSM：`research/icms/offline_icms.py`
- HSW Hook：`research/hooks/hsw_hook.py`
- Thermostat：`research/thermostat/thermostat.py`
- LoRA 训练入口（baseline/method 共享）：`vla-scripts/finetune.py`
- Probe 数据构建：`research/probe/probe_dataset.py`

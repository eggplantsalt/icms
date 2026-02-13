# OpenVLA 训练与断点续训教程

本文是可直接执行的训练手册，覆盖：
- 如何找到“正在训练”的终端
- 如何从 `run_method_train.sh` 启动/续训
- 训练命令需要哪些参数、这些参数在代码哪里定义
- 如何开启“每 N 步自动评测 + 早停”

---

## 1) 我看不到训练终端了，怎么找？

### 1.1 先查训练进程在哪个终端（TTY）
```bash
ps -ef | grep -E 'run_method_train.sh|vla-scripts/finetune.py|torchrun --standalone' | grep -v grep
```

你会看到类似：
- `... pts/5 ... bash scripts/run_method_train.sh ...`

这表示训练是从 `pts/5` 这个终端启动的。

### 1.2 如果原终端窗口已经丢失/关闭
VS Code 无法“重新附着”到已丢失的历史终端会话。建议直接新开终端看日志：

```bash
tail -f /opt/data/private/openvla_icms/runs/train_logs/RUN_METHOD_RESUME_*.log
```

更稳妥（只看最新一个文件）：
```bash
LATEST=$(ls -1t /opt/data/private/openvla_icms/runs/train_logs/RUN_METHOD_RESUME_*.log | head -n 1)
echo "$LATEST"
tail -f "$LATEST"
```

> 说明：训练日志与终端输出是同一份信息。即使终端丢了，日志仍可实时看。

---

## 2) 推荐启动方式（run_method_train）

### 2.0 先选服务器档位（新增开关）
在 `configs/method_icsm_hsw_thermostat.yaml` 设置：

```yaml
server_profile: 4090_1gpu   # 或 v100_8gpu
```

说明：
- `4090_1gpu`：默认 `nproc=1`，更激进的单卡吞吐参数。
- `v100_8gpu`：默认 `nproc=8`，更保守的每卡 batch + 更稀疏 thermostat 更新。
- 如果命令第1个参数写 `auto` 或不传，会按 `server_profile` 自动选择卡数。

### 2.1 标准续训命令（推荐）
```bash
cd /workspace/openvla
export RESUME_ADAPTER=/opt/data/private/openvla_icms/tmp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0002+lora-r16+dropout-0.0--method--image_aug
bash scripts/run_method_train.sh auto 10000 1000 1000 5
```

参数含义：
- `1`：GPU 进程数（`NPROC`）
- `10000`：目标总步数（`TARGET_STEPS`）
- `1000`：保存间隔（`SAVE_STEPS`）
- `1000`：每 N 步评测一次（`EVAL_EVERY`）
- `5`：早停 patience（`PATIENCE`）

### 2.2 不传 `RESUME_ADAPTER` 时
脚本会自动在以下目录里选最新 adapter：
- `/opt/data/private/openvla_icms/tmp/openvla-7b+libero_spatial_no_noops+b16+...`
- `/opt/data/private/openvla_icms/tmp/openvla-7b+libero_spatial_no_noops+b32+...`

### 2.3 强制指定续训步数（可选）
```bash
export RESUME_STEP=500
bash scripts/run_method_train.sh 1 10000 1000 1000 5
```

不指定时会自动从 `RESUME_ADAPTER/trainer_state.pt` 读取 `global_step`。

---

## 3) 训练参数在哪定义？

### 3.1 启动脚本参数入口
- 文件：`scripts/run_method_train.sh`
- 位置：脚本开头的参数解析
  - `NPROC=${1:-1}`
  - `TARGET_STEPS=${2:-10000}`
  - `SAVE_STEPS=${3:-1000}`
  - `EVAL_EVERY=${4:-1000}`
  - `PATIENCE=${5:-5}`

### 3.2 训练主配置（YAML）
- 文件：`configs/method_icsm_hsw_thermostat.yaml`
- 关键字段：
  - 训练：`batch_size` `learning_rate` `max_steps` `save_steps`
  - 续训：`resume_adapter_dir` `resume_global_step`
  - 评测：`enable_periodic_eval` `eval_every_steps` `eval_num_trials_per_task`
  - 早停：`early_stopping_enabled` `early_stopping_patience` `early_stopping_min_delta`

### 3.3 代码默认值定义（最终来源）
- 文件：`vla-scripts/finetune.py`
- 位置：`FinetuneConfig` dataclass

> 优先级：命令行参数 > YAML 配置 > `FinetuneConfig` 默认值。

---

## 4) 每 N 步自动评测 + 早停（已集成）

现在 `finetune.py` 已内置：
- 每到 `eval_every_steps` 且完成 checkpoint 保存后，自动调用 `experiments/robot/libero/run_libero_eval.py`
- 自动解析成功率
- 若启用早停：连续 `early_stopping_patience` 次评测无提升（阈值 `early_stopping_min_delta`）则停止训练

推荐组合：
```yaml
enable_periodic_eval: true
eval_every_steps: 1000
eval_num_trials_per_task: 1
early_stopping_enabled: true
early_stopping_patience: 5
early_stopping_min_delta: 0.1
```

---

## 5) 常用排查命令

### 5.1 查看是否在训练
```bash
ps -ef | grep -E 'run_method_train.sh|vla-scripts/finetune.py' | grep -v grep
```

### 5.2 看最新训练日志
```bash
LATEST=$(ls -1t /opt/data/private/openvla_icms/runs/train_logs/RUN_METHOD_RESUME_*.log | head -n 1)
tail -n 200 "$LATEST"
```

### 5.3 看评测日志
```bash
ls -1t /opt/data/private/openvla_icms/runs/eval_logs/EVAL-libero_spatial-openvla-*.txt | head
```

### 5.4 看 OOM 线索
```bash
dmesg | tail -n 80
```

---

## 6) 一键模板（直接复制）

### 6.1 续训（推荐）
```bash
cd /workspace/openvla
export HF_ENDPOINT=https://hf-mirror.com
export RESUME_ADAPTER=/opt/data/private/openvla_icms/tmp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0002+lora-r16+dropout-0.0--method--image_aug
bash scripts/run_method_train.sh 1 10000 1000 1000 5
```

### 6.2 从头训练（不续训）
```bash
cd /workspace/openvla
export HF_ENDPOINT=https://hf-mirror.com
unset RESUME_ADAPTER
export RESUME_STEP=0
bash scripts/run_method_train.sh 1 10000 1000 1000 5
```

如果你希望，我可以再给你补一版“多卡训练参数模板（2卡/4卡）”直接贴在这个文档末尾。

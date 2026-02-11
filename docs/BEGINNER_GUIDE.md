# 零基础上手指南（不要求现在运行）

这份指南面向完全没有代码基础的同学，目的是让你**知道要看哪些文件、改哪些配置、以及各模块在做什么**。即使暂时不运行，也能读懂项目结构与操作流程。

> 重要约束：**任何大文件都不能写入 /workspace**。所有模型、数据、缓存、产物必须放在 `/opt/data/private/openvla_icms/` 下。

---

## 1) 你要先理解的三个“概念”

1. **模型（OpenVLA）**：已经预训练好的视觉-语言-动作模型。
2. **下游微调（LoRA）**：只训练少量 LoRA 参数来适配新任务。
3. **研究方法（ICSM + HSW + Thermostat）**：
   - ICSM：离线统计“语义子空间”。
   - HSW：训练时做梯度手术，保护语义能力。
   - Thermostat：动态调节 beta/gamma，控制漂移。

---

## 2) 目录结构和你需要关注的文件

### 2.1 最重要的目录

- `vla-scripts/`：训练入口
  - [vla-scripts/finetune.py](vla-scripts/finetune.py)：LoRA 微调入口（baseline/method 共用）
  - [vla-scripts/train.py](vla-scripts/train.py)：FSDP 全量训练入口（可选，method 也已支持）

- `research/`：研究方法实现
  - [research/icms/offline_icms.py](research/icms/offline_icms.py)：离线 ICSM（Teacher-only）
  - [research/hooks/hsw_hook.py](research/hooks/hsw_hook.py)：HSW 梯度手术
  - [research/thermostat/thermostat.py](research/thermostat/thermostat.py)：Thermostat 控制器
  - [research/probe/hidden_and_mask.py](research/probe/hidden_and_mask.py)：指令 mask + hidden 对齐
  - [research/probe/probe_dataset.py](research/probe/probe_dataset.py)：探针集加载

- `configs/`：可直接使用的配置
  - [configs/baseline_lora.yaml](configs/baseline_lora.yaml)
  - [configs/method_icsm_hsw_thermostat.yaml](configs/method_icsm_hsw_thermostat.yaml)

- `scripts/`：一键命令模板
  - [scripts/run_offline_icms.sh](scripts/run_offline_icms.sh)
  - [scripts/run_baseline_lora.sh](scripts/run_baseline_lora.sh)
  - [scripts/run_method_train.sh](scripts/run_method_train.sh)
  - [scripts/run_eval_libero.sh](scripts/run_eval_libero.sh)
  - [scripts/check_disk_safety.py](scripts/check_disk_safety.py)

- `docs/`：文档
  - [docs/RESEARCH_PIPELINE.md](docs/RESEARCH_PIPELINE.md)：完整跑法与产物说明
  - [docs/RESEARCH_EVIDENCE.md](docs/RESEARCH_EVIDENCE.md)：源码确认与证据
  - [docs/OVERVIEW.md](docs/OVERVIEW.md)：项目全局介绍

---

## 3) 必须牢记的磁盘规则

- **任何大文件不能写入 /workspace**。
- 必须使用 `/opt/data/private/openvla_icms/` 作为总根目录。
- 统一目录结构：

```
/opt/data/private/openvla_icms/
  hf_cache/
  datasets/
  probe/
  artifacts/
  runs/
  tmp/
```

如果你不确定是否安全，先运行磁盘自检：

```bash
python scripts/check_disk_safety.py \
  --workspace_root /workspace \
  --allowed_root /opt/data/private \
  --project_root /opt/data/private/openvla_icms \
  --max_workspace_file_gb 1
```

---

## 4) 不运行也能看懂的“流程图”

1. **离线阶段（ICSM）**
   - 使用探针集（probe）统计子空间。
   - 产物在 `/opt/data/private/openvla_icms/artifacts/icms_openvla-7b/`。

2. **训练阶段（LoRA + HSW + Thermostat）**
   - baseline：只用 LoRA。
   - method：加载 ICSM 产物，启用 HSW + Thermostat。
   - 产物在 `/opt/data/private/openvla_icms/runs/`。

3. **评测阶段（LIBERO）**
   - 使用现有 eval 脚本，读取训练产物。

---

## 5) 零基础“该看哪里、该改哪里”

### 5.1 你只需要改的三个地方

1. **配置文件**（最重要）：
   - baseline：修改 [configs/baseline_lora.yaml](configs/baseline_lora.yaml)
   - method：修改 [configs/method_icsm_hsw_thermostat.yaml](configs/method_icsm_hsw_thermostat.yaml)
  - 注意：`dataset_name` 必须与本地实际数据集名称一致（LIBERO 不是 bridge_orig）。

2. **数据路径**：
   - `data_root_dir`：训练数据（RLDS）
   - `probe_root_dir`：探针数据（probe）

3. **训练产物目录**：
   - `run_root_dir`：训练日志与 checkpoint

### 5.2 你不需要改的文件

- `prismatic/` 目录：核心模型与训练逻辑，不建议新手改。
- `research/` 目录：研究方法实现，除非要做算法改动。

---

## 6) “一键命令”对应什么（即使不运行也要懂）

- `run_offline_icms.sh`
  - 做离线统计，生成子空间。

- `run_baseline_lora.sh`
  - 只训练 LoRA，作为对照组。

- `run_method_train.sh`
  - 训练 LoRA + HSW + Thermostat。

- `run_eval_libero.sh`
  - 用 LIBERO 评测训练结果。

---

## 7) 你应该关心的日志字段（method 模式）

- `drift_d`：漂移大小 $d = ||C_S - C_T||_F$
- `hsw_beta`, `hsw_gamma`：梯度手术系数
- `hsw_g_norm`, `hsw_gprime_norm`：梯度范数对比
- `train_loss`, `action_accuracy`, `l1_loss`：训练基本指标

---

## 8) 常见问题（先看这里）

1. **不能写 /workspace**：所有数据/模型/产物必须放 /opt/data/private。
2. **连不上 HuggingFace**：
   - `export HF_ENDPOINT=https://hf-mirror.com`
3. **环境问题**：
   - `conda activate openvla`

---

## 9) 如果你想“深入读代码”

推荐阅读顺序：

1. [vla-scripts/finetune.py](vla-scripts/finetune.py)
2. [research/icms/offline_icms.py](research/icms/offline_icms.py)
3. [research/hooks/hsw_hook.py](research/hooks/hsw_hook.py)
4. [research/thermostat/thermostat.py](research/thermostat/thermostat.py)
5. [prismatic/training/strategies/base_strategy.py](prismatic/training/strategies/base_strategy.py)

---

## 10) 你需要记住的一句话

**所有大文件只能在 `/opt/data/private/openvla_icms/`，不要写入 /workspace。**

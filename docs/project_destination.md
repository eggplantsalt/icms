

## Prompt #1（总需求版｜一口气交付完整论文代码库）

你是一个顶级科研工程实现者（research engineer）。你正在一个本地 git 仓库中工作：**openvla 官方开源代码库（PyTorch）**。你的任务是：**在不破坏原有训练/评测能力的前提下**，为一篇研究论文实现一个完整可复现的训练框架，研究“VLA 下游微调的灾难性遗忘”，并落地我们的方法：**ICSM + HSW + Thermostat**（Teacher 仅离线一次，训练仅 Student）。

> 重要：你必须通过“实际读取代码与运行命令”完成工作，不能凭空猜接口。
> 你必须提供足够证据（文件路径、关键函数签名、真实命令、产物目录结构、日志字段），保证新成员可以按文档一键复现。

我们的硬件资源是一张4090，或者八张V100，所以运行文件要提供参数接口，可以供用户选择是1张卡还是8张卡
我们使用的是conda虚拟环境，openvla的虚拟环境我已经给你配置好了。你用的时候可能需要先conda activate openvla,期间遇到环境问题可以尝试在虚拟环境里安装。如果出现连不上huggingface或者github的情况，可以先配置huggingface国内镜像export HF_ENDPOINT=https://hf-mirror.com
，github连不上就多试几次


---
我们运行在镜像环境中：/workspace 位于容器文件系统下，容量有限。
任何大文件（模型/数据集/缓存/ckpt）写入 /workspace 都可能导致镜像崩溃。

因此你必须满足：

## -1.1 唯一允许的大文件落盘位置

所有大模型、数据集、缓存、训练产物必须落在：

/opt/data/private/<your_project_dir>/...（你需要创建目录）

建议统一路径（可调整但必须都在 /opt/data/private）：

HF_CACHE=/opt/data/private/<proj>/hf_cache

DATA_ROOT=/opt/data/private/<proj>/datasets

PROBE_ROOT=/opt/data/private/<proj>/probe

ARTIFACT_ROOT=/opt/data/private/<proj>/artifacts

RUN_ROOT=/opt/data/private/<proj>/runs

TMP_ROOT=/opt/data/private/<proj>/tmp

## -1.2 代码层必须“强制”不写 /workspace

你实现的所有脚本/入口必须支持并默认使用这些参数（默认值必须在 /opt/data/private/...）：

--cache_dir

--data_root_dir

--probe_root_dir

--artifact_dir

--run_root_dir

--tmp_dir

并且所有 HuggingFace 加载必须支持：

cache_dir=...（显式传入）

支持通过 env 覆盖缓存位置

必须在文档与脚本里推荐/默认设置：

HF_HOME=/opt/data/private/<proj>/hf_cache

TRANSFORMERS_CACHE=/opt/data/private/<proj>/hf_cache

HF_DATASETS_CACHE=/opt/data/private/<proj>/hf_cache

TORCH_HOME=/opt/data/private/<proj>/torch_cache（如用到）

（如用 wandb）WANDB_DIR=/opt/data/private/<proj>/wandb 或直接支持 --use_wandb False

## -1.3 /workspace 只允许放代码与小文件

/workspace/openvla 只能包含：

源码

小配置（yaml/json）

文档

极小日志（最好也重定向到 /opt）

禁止写入：

模型权重

数据集

checkpoints / adapters

大型 cache

-1.4 必须提供“磁盘安全自检”

你需要新增一个脚本（或在 docs 中给出可运行命令），用于：

检查本次实验相关的目录是否都在 /opt/data/private

扫描 /workspace 下是否出现异常大文件（例如 >1GB），一旦出现给出明确报警与清理建议
# 0) 全局目标（论文级 + 工程交付）

## 0.1 论文级目标

我们从一个预训练 VLA（pi0-FAST base，或 repo 当前选用的预训练基座）出发，在下游动作任务（如 LIBERO）微调时研究灾难性遗忘。我们提出的方法不使用 replay 数据避免遗忘，而是通过训练反向传播阶段的“梯度手术”抑制对语义能力有害的梯度分量，同时保留动作学习的可塑性。

最终要证明：

1. 下游动作成功率不降（最好提升）
2. 通用能力遗忘显著减轻（相对 baseline：普通 LoRA 微调）
3. 子空间手术 + Thermostat 闭环控制是关键贡献（有消融）

## 0.2 工程交付目标（必须能跑通）

你需要在仓库内新增一个“研究型微调管线”，支持：

* **Baseline**：普通 LoRA 微调（作为对照）
* **Method**：LoRA + ICSM（离线） + HSW（训练时梯度手术） + Thermostat（闭环调参）
* 训练可跑通（至少 1 GPU debug；多 GPU/FSDP 保持兼容或最小改动）
* 评测可跑通（复用现有 eval 脚本入口，或提供等价接口）
* 可复现：配置、命令行、日志、产物目录清晰，一键复现关键实验
* 文档：新增 docs 说明如何跑 baseline / method / 产物是什么 / 复现实验目录结构

---

# 1) 冻结技术路线（不要偏离）

* Codebase：必须基于 openvla 官方 repo（PyTorch）
* Backbone：预训练 VLA 权重（以你在 repo 里确认的实际基座为准；如果已有 pi0-FAST 路径则使用 pi0-FAST）
* 微调方式：LoRA（PEFT），只对后部若干层挂 LoRA（层范围可配置）
* 方法类型：训练时梯度改造（不是 replay / 不是新 loss / 不是新 action head 结构）
* Teacher：只离线一次，不在训练阶段常驻
* HSW：默认只在最后 4 层做（算力可控）
* 双组保护：最后 4 层分 A/B 两组，各自共享一套子空间字典

---

# 2) 方法 pipeline（必须严格实现）

## 2.1 数据：两套数据

(1) Downstream train set（下游动作微调数据）

* 输入：images + instruction text (+ state 可选) + action labels/tokens
* 输出：训练 batch（由 repo 的 dataset/processor/collator 产生）
* loss：**使用 repo 默认 action prediction loss**，不要发明新 loss

(2) Probe set（探针集，只用于几何统计与漂移监控）

* N\_probe 默认 500（可配置）
* 每条包含 image + instruction/state 文本（动作 token 可无）
* probe 不用于梯度更新（除了离线 Teacher 统计）
* 目标：构造语义子空间 + 监控漂移

## 2.2 离线阶段（Teacher-only，一次性）

Teacher = 预训练模型（冻结）

离线要做：

1. instruction focusing：只对 instruction tokens 的 hidden 做统计

   * **如果 tokenizer 提供 loss\_mask（动作区）**：instr\_mask = (\~loss\_mask) & attention\_mask
   * 否则必须从 repo 实际 prompt/模板中可靠定位 instruction token span（必须可解释、可审计）
2. mean pooling：对 instr\_mask 的 hidden mean pooling 得到每条样本一个向量 s\_i ∈ R^D

   * 必须处理分母为 0 的情况（mask 全 False 必须安全处理并计数）
3. PCA/SVD：对中心化矩阵 X \[N,D] 做 SVD，取 **Vh.T\[:, \:r]** 得候选方向（r 默认 128）
4. 敏感度矫正：对每个候选方向 u\_k 做 broadcast 注入到 instruction hidden，测输出分布变化（KL 或等价指标），按脆弱度重排方向

   * 产出 Uf（Fragile）/ Up（Plastic）/ Unull（剩余）
5. Teacher baseline 统计：mu 与 C\_T（协方差矩阵）用于 Thermostat

离线产物落盘（必须有）：

* `artifacts/icms_<backbone_name>/`

  * `U15_f.pt, U15_p.pt, mu15.pt, C_T15.pt`
  * `U17_f.pt, U17_p.pt, mu17.pt, C_T17.pt`
  * `meta.json`（层号、r、epsilon、probe hash、版本信息、commit）

代表层与双组策略（若 backbone 为 18 层 0..17）：

* Group A：层 14,15 共享 U(15)
* Group B：层 16,17 共享 U(17)

> 若实际层数不同，你必须从源码确认层结构，并给出等价的“最后四层 + 两代表层”映射，并写入 meta.json 和文档。

## 2.3 在线训练（Student-only）

Student 从同一预训练权重加载，进行 LoRA 微调。loss 不变，但 backward 做 HSW，并用 Thermostat 动态调 β/γ。

(1) LoRA

* 仅 LoRA 参数可训练，其余冻结（可配置）
* 层范围与模块类型（attn/mlp）可配置

(2) HSW 梯度手术（训练时）
在高风险层注册 backward hook（默认最后 4 层）：

* 对梯度 g 分解：gf = proj(g, Uf), gp = proj(g, Up), gn = g - gf - gp
* 重组：g' = gn + beta \* gp + gamma \* gf
* 范数只缩不放：g' = g' \* min(1, ||g||/(||g'||+eps))

(3) Thermostat（闭环控制）
每隔 K steps：

* 从 probe 随机抽小 batch（例如 64）
* Student forward 抓代表层 hidden，做相同 pooling 得 C\_S
* 漂移度量：d = ||C\_S - C\_T||\_F（两个代表层可平均）
* 控制律：d 越大 => gamma 越小（更强抑制 fragile），beta 随 d 下降但不归零
* 必须有 calibration/warmup：训练开始先只测不更，估计噪声基线并设阈值

---

# 3) 你必须先做的“源码确认”（必须真实读代码）

在开始实现前，必须运行并记录以下命令输出（写入 docs/ 或日志中）：

* `pwd`
* `ls`
* `tree -L 3`（或 find 替代）
* `rg` 搜索入口/关键字：`finetune.py`, `train.py`, `run_.*eval`, `loss_mask`, `output_hidden_states`, `hidden_states`, `predict_action`, `processor`, `PrismaticProcessor`, `datasets/rlds`, `draccus` 等
* 打开并阅读关键文件（若存在）：

  * README.md
  * 训练/微调脚本（train/finetune）
  * eval 脚本（LIBERO/bridge）
  * processor/modeling 文件（确认 hidden\_states）
  * dataset/collator（确认 batch 字段、loss\_mask/labels）

你必须明确写出：

* 默认 prompt/输入格式来自哪里（文件路径）
* hidden\_states 能否通过 output\_hidden\_states 获取、返回结构是什么（类/函数名）
* batch 中 action 区如何定义（loss\_mask 或 labels）

---

# 4) 代码结构要求（你可以自行决定，但必须最小侵入式）

建议（可调整）：

* `research/`

  * `probe/`（mask + hidden + pooling + covariance）
  * `icms/`（离线构建子空间 + 敏感度矫正）
  * `hooks/`（HSW 投影/梯度手术）
  * `thermostat/`（闭环控制）
  * `runner/`（统一训练入口：baseline/method）
* `configs/`

  * baseline\_lora.yaml
  * method\_icsm\_hsw\_thermostat.yaml
* `scripts/`

  * run\_offline\_icms.sh
  * run\_baseline\_lora.sh
  * run\_method\_train.sh
  * run\_eval\_libero.sh
* `docs/`

  * RESEARCH\_PIPELINE.md（跑法 + 产物 + 日志字段）
  * OVERVIEW\.md（更新：把研究 pipeline 写进全局上下文）
* `artifacts/`（离线产物与实验输出）

硬性要求：

* baseline 与 method 必须共享尽可能多的训练代码路径（避免不可控差异）
* 所有新增功能必须可以通过 config 开关关闭（method\_off => baseline）
* 日志必须包含：step、loss、d、beta、gamma、||g||/||g'||（至少在 debug 模式）

---

# 5) 运行命令与可复现性（必须提供）

你必须提供至少以下命令模板（结构必须可运行，路径可以用占位符）：

1. 离线 ICSM（Teacher-only，一次）

* 输入：probe 数据路径、代表层、r、epsilon、输出目录
* 输出：artifacts/icms\_\*/ 下的 pt 文件与 meta.json

2. Baseline LoRA 微调

* 输入：下游数据路径、输出目录、LoRA 配置

3. Method 训练（LoRA + HSW + Thermostat）

* 输入：下游数据路径、probe 数据路径、离线产物目录、hook 层映射、K、控制律参数

4. Eval（LIBERO）

* 复用现有 eval 脚本或提供 wrapper
* 输入：checkpoint/adapters、task suite、其它 eval 参数

---

# 6) 验收标准（你必须自己给出自检与证据）

离线阶段验收：

* 产物落盘齐全、shape 合法（U: \[D,r]、C: \[D,D]）
* meta.json 信息完整
* 对同一 probe 固定 seed 可复现（hash/输出一致）

训练阶段验收：

* baseline 能跑（loss 下降）
* method 能跑（loss 下降）
* hook 可开关对比（关闭 hook 时等价 baseline）
* 日志里能看到 d/beta/gamma 在变化（Thermostat 在跑）
* 梯度范数守恒项生效（||g'|| ≤ ||g||）

评测验收：

* 至少 LIBERO eval 能跑通一条命令
* 输出 success rate 或 repo 原生指标

---

# 7) 最终交付（你必须产出）

* 新增的全部代码文件（按你设计的结构）
* 配置文件（baseline/method）
* 运行脚本（offline/baseline/method/eval）
* 文档（RESEARCH\_PIPELINE.md + 更新 OVERVIEW\.md）
* 一份“证据报告”：列出你运行过的命令、读过的关键文件、关键接口签名、以及产物目录树

---

## 输出要求（非常重要）

你必须：

* 先完成源码确认（命令与读文件）再写代码
* 每实现一个关键模块，附带最小可运行的 sanity check（不要求 pytest，但要有可执行脚本）
* 最终给出“从 0 到跑通 baseline + method + eval”的完整文档与命令

开始执行。

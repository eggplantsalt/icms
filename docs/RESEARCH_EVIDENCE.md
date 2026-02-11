# 源码确认与证据记录

本文件记录执行的命令与输出、关键文件的读取证据，以及接口确认结论。

## 1) 命令执行记录

### 1.1 `pwd` / `ls` / `find -maxdepth 3`

命令：

```bash
cd /workspace/openvla && pwd && ls && find . -maxdepth 3 -print
```

输出：

```text
/workspace/openvla
LIBERO   Makefile   dlimp_openvla  experiments       prismatic       requirements-min.txt  scripts
LICENSE  README.md  docs           openvla.egg-info  pyproject.toml  research              vla-scripts
.
./vla-scripts
./vla-scripts/finetune.py
./vla-scripts/train.py
./vla-scripts/extern
./vla-scripts/extern/verify_openvla.py
./vla-scripts/extern/convert_openvla_weights_to_hf.py
./vla-scripts/deploy.py
./scripts
./scripts/additional-datasets
./scripts/additional-datasets/lrv_instruct.py
./scripts/additional-datasets/lvis_instruct_4v.py
./scripts/pretrain.py
./scripts/preprocess.py
./scripts/extern
./scripts/extern/convert_prismatic_weights_to_hf.py
./scripts/extern/verify_prismatic.py
./scripts/generate.py
./.git
./.git/info
./.git/info/exclude
./.git/hooks
./.git/hooks/prepare-commit-msg.sample
./.git/hooks/pre-applypatch.sample
./.git/hooks/update.sample
./.git/hooks/post-update.sample
./.git/hooks/pre-receive.sample
./.git/hooks/pre-push.sample
./.git/hooks/fsmonitor-watchman.sample
./.git/hooks/pre-merge-commit.sample
./.git/hooks/applypatch-msg.sample
./.git/hooks/pre-rebase.sample
./.git/hooks/pre-commit.sample
./.git/hooks/commit-msg.sample
./.git/description
./.git/logs
./.git/logs/HEAD
./.git/logs/refs
./.git/index
./.git/branches
./.git/HEAD
./.git/config
./.git/objects
./.git/objects/info
./.git/objects/pack
./.git/refs
./.git/refs/tags
./.git/refs/remotes
./.git/refs/heads
./.git/packed-refs
./.gitignore
./.pre-commit-config.yaml
./pyproject.toml
./README.md
./prismatic
./prismatic/preprocessing
./prismatic/preprocessing/materialize.py
./prismatic/preprocessing/download.py
./prismatic/preprocessing/__init__.py
./prismatic/preprocessing/datasets
./prismatic/util
./prismatic/util/batching_utils.py
./prismatic/util/data_utils.py
./prismatic/util/nn_utils.py
./prismatic/util/__init__.py
./prismatic/util/torch_utils.py
./prismatic/vla
./prismatic/vla/materialize.py
./prismatic/vla/action_tokenizer.py
./prismatic/vla/__init__.py
./prismatic/vla/datasets
./prismatic/py.typed
./prismatic/models
./prismatic/models/materialize.py
./prismatic/models/vlms
./prismatic/models/load.py
./prismatic/models/backbones
./prismatic/models/__init__.py
./prismatic/models/registry.py
./prismatic/models/vlas
./prismatic/extern
./prismatic/extern/hf
./prismatic/extern/__init__.py
./prismatic/__init__.py
./prismatic/conf
./prismatic/conf/models.py
./prismatic/conf/__init__.py
./prismatic/conf/datasets.py
./prismatic/conf/vla.py
./prismatic/training
./prismatic/training/strategies
./prismatic/training/materialize.py
./prismatic/training/metrics.py
./prismatic/training/__init__.py
./prismatic/overwatch
./prismatic/overwatch/overwatch.py
./prismatic/overwatch/__init__.py
./experiments
./experiments/robot
./experiments/robot/libero
./experiments/robot/robot_utils.py
./experiments/robot/openvla_utils.py
./experiments/robot/bridge
./LIBERO
./LIBERO/scripts
./LIBERO/scripts/check_dataset_integrity.py
./LIBERO/scripts/config_copy.py
./LIBERO/scripts/collect_demonstration.py
./LIBERO/scripts/create_dataset.py
./LIBERO/scripts/get_affordance_info.py
./LIBERO/scripts/create_template.py
./LIBERO/scripts/libero_100_collect_demonstrations.py
./LIBERO/scripts/get_dataset_info.py
./LIBERO/scripts/init_path.py
./LIBERO/scripts/create_libero_task_example.py
./LIBERO/libero
./LIBERO/libero/lifelong
./LIBERO/libero/libero
./LIBERO/libero/configs
./LIBERO/.git
./LIBERO/.git/info
./LIBERO/.git/hooks
./LIBERO/.git/description
./LIBERO/.git/logs
./LIBERO/.git/index
./LIBERO/.git/branches
./LIBERO/.git/HEAD
./LIBERO/.git/config
./LIBERO/.git/objects
./LIBERO/.git/refs
./LIBERO/.git/packed-refs
./LIBERO/notebooks
./LIBERO/notebooks/quick_guide_algo.ipynb
./LIBERO/notebooks/quick_walkthrough.ipynb
./LIBERO/notebooks/custom_object_example.ipynb
./LIBERO/notebooks/procedural_creation_walkthrough.ipynb
./LIBERO/notebooks/custom_assets
./LIBERO/templates
./LIBERO/templates/problem_class_template.py
./LIBERO/templates/scene_template.xml
./LIBERO/.gitignore
./LIBERO/requirements.txt
./LIBERO/images
./LIBERO/images/fig1.png
./LIBERO/images/libero_logo.png
./LIBERO/README.md
./LIBERO/setup.py
./LIBERO/benchmark_scripts
./LIBERO/benchmark_scripts/check_task_suites.py
./LIBERO/benchmark_scripts/shasum_files.py
./LIBERO/benchmark_scripts/download_libero_datasets.py
./LIBERO/benchmark_scripts/render_single_task.py
./LIBERO/benchmark_scripts/init_path.py
./LIBERO/LICENSE
./research
./research/probe
./research/probe/smoke_test_alignment.py
./research/probe/smoke_test_hidden_and_mask.py
./research/probe/__pycache__
./research/probe/hidden_and_mask.py
./requirements-min.txt
./Makefile
./openvla.egg-info
./openvla.egg-info/requires.txt
./openvla.egg-info/PKG-INFO
./openvla.egg-info/top_level.txt
./openvla.egg-info/dependency_links.txt
./openvla.egg-info/SOURCES.txt
./LICENSE
./dlimp_openvla
./dlimp_openvla/.gitignore
./dlimp_openvla/.pre-commit-config.yaml
./dlimp_openvla/rlds_converters
./dlimp_openvla/rlds_converters/README.md
./dlimp_openvla/rlds_converters/setup.py
./dlimp_openvla/rlds_converters/bridge_dataset
./dlimp_openvla/rlds_converters/dataset_builder.py
./dlimp_openvla/README.md
./dlimp_openvla/dlimp
./dlimp_openvla/dlimp/dataset.py
./dlimp_openvla/dlimp/utils.py
./dlimp_openvla/dlimp/augmentations.py
./dlimp_openvla/dlimp/transforms
./dlimp_openvla/dlimp/__init__.py
./dlimp_openvla/setup.py
./dlimp_openvla/legacy_converters
./dlimp_openvla/legacy_converters/bridgedata
./dlimp_openvla/legacy_converters/ego4d
./dlimp_openvla/legacy_converters/somethingsomething
./dlimp_openvla/legacy_converters/kinetics
./docs
./docs/PROBE_HIDDEN_AND_MASK.md
./docs/project_destination.md
./docs/OVERVIEW.md
```

### 1.2 `rg` 关键字搜索

命令：

```bash
rg "finetune.py|train.py|run_.*eval|loss_mask|output_hidden_states|hidden_states|predict_action|processor|PrismaticProcessor|datasets/rlds|draccus"
```

输出（节选）：

```text
README.md
56:processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
70:inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
71:action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
78:embodiments](./vla-scripts/finetune.py); this script supports different fine-tuning modes -- including (quantized) 
157:fine-tuning is `vla-scripts/finetune.py`. (If you instead wish to do full fine-tuning, please see the
184:torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
278:torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
364:from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
403:The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We use 
404:[`draccus`](https://pypi.org/project/draccus) to provide a modular, dataclass-based interface for specifying VLA 
484:python experiments/robot/bridge/run_bridgev2_eval.py \
555:python experiments/robot/libero/run_libero_eval.py \

vla-scripts/finetune.py
113:@draccus.wrap()
157:    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

vla-scripts/train.py
106:@draccus.wrap()

prismatic/extern/hf/modeling_prismatic.py
301:        output_hidden_states: Optional[bool] = None,
307:        output_hidden_states = (
413:                output_hidden_states=output_hidden_states,
444:            hidden_states=language_model_output.hidden_states,

prismatic/vla/datasets/datasets.py
23:from prismatic.models.backbones.llm.prompting import PromptBuilder
35:    prompt_builder_fn: Type[PromptBuilder]

experiments/robot/openvla_utils.py
64:    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
```

### 1.3 `rg loss_mask`

命令：

```bash
rg "loss_mask"
```

输出：

```text
docs/project_destination.md
223:* `rg` 搜索入口/关键字：`finetune.py`, `train.py`, `run_.*eval`, `loss_mask`, `output_hidden_states`, `hidden_states`
```

### 1.4 `rg output_hidden_states|hidden_states`

命令：

```bash
rg -n "output_hidden_states|hidden_states" prismatic/extern/hf/modeling_prismatic.py
```

输出：

```text
169:    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
301:        output_hidden_states: Optional[bool] = None,
307:        output_hidden_states = (
339:                output_hidden_states=output_hidden_states,
357:                output_hidden_states=output_hidden_states,
413:                output_hidden_states=output_hidden_states,
444:            hidden_states=language_model_output.hidden_states,
```

## 2) 关键文件阅读记录（接口证据）

### 2.1 默认 prompt / 输入格式来源

- 推理 prompt：`experiments/robot/openvla_utils.py` 中 `OpenVLA` 分支拼接模板：`In: What action should the robot take to {task_label.lower()}?\nOut:`
- 训练 prompt：`prismatic/vla/datasets/datasets.py` 中 `RLDSBatchTransform.__call__` 使用 PromptBuilder 构造 `What action should the robot take to {lang}?`

### 2.2 hidden_states 获取方式

- `prismatic/extern/hf/modeling_prismatic.py` 中 `PrismaticForConditionalGeneration.forward` 接收 `output_hidden_states` 并在返回的 `PrismaticCausalLMOutputWithPast` 中暴露 `hidden_states`

### 2.3 action 区域定义方式

- `prismatic/vla/datasets/datasets.py` 中 `RLDSBatchTransform.__call__` 将 labels 的非 action 部分设为 `IGNORE_INDEX`，训练 loss 由 labels 掩码定义，而非 `loss_mask`

"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from itertools import cycle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from research.hooks.hsw_hook import HSWState, HSWHookManager, load_subspaces
from research.hooks.layer_utils import get_llm_layers
from research.probe.probe_dataset import build_probe_dataloader, build_probe_jsonl_dataloader
from research.thermostat.thermostat import Thermostat, ThermostatConfig, ThermostatState

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)
    server_profile: Optional[str] = None                             # Optional launcher profile flag (e.g., 4090_1gpu / v100_8gpu)

    # Directory Paths
    cache_dir: Path = Path("/opt/data/private/openvla_icms/hf_cache")
    data_root_dir: Path = Path("/opt/data/private/openvla_icms/datasets")        # Path to Open-X dataset directory
    probe_root_dir: Path = Path("/opt/data/private/openvla_icms/probe")
    artifact_dir: Path = Path("/opt/data/private/openvla_icms/artifacts")
    run_root_dir: Path = Path("/opt/data/private/openvla_icms/runs")             # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("/opt/data/private/openvla_icms/tmp")           # Temporary directory for LoRA weights before fusing
    tmp_dir: Path = Path("/opt/data/private/openvla_icms/tmp")
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    min_learning_rate: float = 1e-5                                 # Minimum LR for cosine schedule
    lr_scheduler: str = "none"                                       # none | cosine
    lr_warmup_steps: int = 0                                         # Linear warmup steps before schedule
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 5000                             # Dataloader shuffle buffer size (can reduce if OOM)
    rlds_frame_parallel_calls: int = 32                             # RLDS frame transform parallel calls
    rlds_traj_transform_threads: int = 8                            # RLDS trajectory transform worker threads
    rlds_traj_read_threads: int = 8                                 # RLDS trajectory read worker threads
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    merge_lora_during_training: bool = False                        # Merge LoRA into full model at save time (high memory)

    # Periodic Evaluation & Early Stopping
    enable_periodic_eval: bool = False                               # Run LIBERO eval every N training steps
    eval_every_steps: int = 1000                                     # Eval interval in steps (must be multiple of save_steps)
    eval_task_suite_name: str = "libero_spatial"                    # LIBERO task suite for periodic eval
    eval_num_trials_per_task: int = 1                                # Number of trials per task during periodic eval
    eval_center_crop: bool = True                                    # Whether to use center crop during eval
    eval_run_id_note: Optional[str] = None                           # Optional extra run_id note for eval logs

    early_stopping_enabled: bool = False                             # Stop training when eval metric stops improving
    early_stopping_patience: int = 5                                 # Number of eval rounds without improvement before stop
    early_stopping_min_delta: float = 0.1                            # Minimum improvement in success rate (percentage points)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    use_wandb: bool = True                                          # Whether to log to W&B
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # Research Method Parameters
    method_enabled: bool = False                                    # Enable ICSM + HSW + Thermostat
    icms_artifact_dir: Optional[Path] = None                         # Override artifact dir for ICSM outputs
    hsw_layer_ids: Optional[List[int]] = None                        # Layer IDs to hook; default: last 4 layers
    hsw_beta: float = 1.0                                            # Initial beta
    hsw_gamma: float = 1.0                                           # Initial gamma
    hsw_eps: float = 1e-8                                            # Stability epsilon for norm scaling
    prompt_template: str = "In: {instruction}\nOut:"                 # Probe prompt template

    # Resume Parameters
    resume_adapter_dir: Optional[Path] = None                        # LoRA adapter dir to resume from
    resume_full_model_dir: Optional[Path] = None                     # Full model dir to resume from (non-LoRA)
    resume_global_step: int = 0                                      # Global step offset for logging/schedules

    # Probe Parameters (Thermostat)
    probe_dataset_name: str = "bridge_orig"
    probe_jsonl: Optional[Path] = None
    probe_image_root: Optional[Path] = None
    probe_batch_size: int = 16

    thermostat_update_interval: int = 100
    thermostat_warmup_steps: int = 200
    thermostat_min_beta: float = 0.2
    thermostat_max_beta: float = 1.0
    thermostat_min_gamma: float = 0.1
    thermostat_max_gamma: float = 1.0
    thermostat_k_beta: float = 1.0
    thermostat_k_gamma: float = 1.0

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    os.environ.setdefault("HF_HOME", str(cfg.cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cfg.cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cfg.cache_dir))
    os.environ.setdefault("TORCH_HOME", str(cfg.cache_dir.parent / "torch_cache"))
    os.environ.setdefault("WANDB_DIR", str(cfg.cache_dir.parent / "wandb"))

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True, cache_dir=cfg.cache_dir)

    if cfg.use_lora and cfg.resume_full_model_dir is not None:
        raise ValueError("resume_full_model_dir is only valid when use_lora is False")

    model_load_path = cfg.resume_full_model_dir or cfg.vla_path
    vla = AutoModelForVision2Seq.from_pretrained(
        model_load_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        if cfg.resume_adapter_dir is not None:
            if not cfg.resume_adapter_dir.exists():
                raise FileNotFoundError(f"LoRA adapter dir not found: {cfg.resume_adapter_dir}")
            vla = PeftModel.from_pretrained(vla, cfg.resume_adapter_dir, is_trainable=True)
            print(f"Resumed LoRA adapter from: {cfg.resume_adapter_dir}")
        else:
            vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Optional LR scheduler (warmup + cosine)
    scheduler = None
    if cfg.lr_scheduler == "cosine":
        total_steps = max(1, cfg.max_steps)
        warmup_steps = max(0, cfg.lr_warmup_steps)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            min_ratio = cfg.min_learning_rate / cfg.learning_rate
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    resume_state_step = None
    if cfg.resume_adapter_dir is not None:
        trainer_state_path = cfg.resume_adapter_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            trainer_state = torch.load(trainer_state_path, map_location="cpu")
            optimizer.load_state_dict(trainer_state.get("optimizer", {}))
            if scheduler is not None and trainer_state.get("scheduler") is not None:
                scheduler.load_state_dict(trainer_state["scheduler"])
            resume_state_step = int(trainer_state.get("global_step", 0))
            if distributed_state.is_main_process:
                print(f"Resumed optimizer/scheduler from: {trainer_state_path}")

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # === 研究方法：ICSM + HSW + Thermostat（可开关） ===
    hsw_manager = None
    hsw_state = None
    thermostat = None
    thermostat_state = None
    probe_iter = None
    if cfg.method_enabled:
        icms_dir = cfg.icms_artifact_dir or (cfg.artifact_dir / f"icms_{cfg.vla_path.split('/')[-1]}")
        meta_path = icms_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"ICSM meta.json not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        llm_layers = get_llm_layers(vla)
        num_layers = len(llm_layers)
        hsw_layers = cfg.hsw_layer_ids or list(range(num_layers - 4, num_layers))

        group_mapping = meta.get("group_mapping", {})
        group_a = group_mapping.get("group_a", [num_layers - 4, num_layers - 3])
        group_b = group_mapping.get("group_b", [num_layers - 2, num_layers - 1])
        rep_layers = group_mapping.get("rep_layers", [num_layers - 3, num_layers - 1])

        rep_to_subspace = load_subspaces(str(icms_dir), rep_layers, device=torch.device(device_id))
        layer_to_subspace = {}
        for layer_id in hsw_layers:
            if layer_id in group_a:
                rep = rep_layers[0]
            else:
                rep = rep_layers[-1]
            layer_to_subspace[layer_id] = rep_to_subspace[rep]

        hsw_state = HSWState(beta=cfg.hsw_beta, gamma=cfg.hsw_gamma, eps=cfg.hsw_eps)
        hsw_manager = HSWHookManager(vla, layer_to_subspace, hsw_state)
        hsw_manager.register()

        teacher_stats = {}
        for rep in rep_layers:
            mu = torch.load(icms_dir / f"mu{rep}.pt", map_location="cpu")
            c_t = torch.load(icms_dir / f"C_T{rep}.pt", map_location="cpu")
            teacher_stats[rep] = {"mu": mu, "C_T": c_t}

        thermostat_cfg = ThermostatConfig(
            update_interval=cfg.thermostat_update_interval,
            warmup_steps=cfg.thermostat_warmup_steps,
            min_beta=cfg.thermostat_min_beta,
            max_beta=cfg.thermostat_max_beta,
            min_gamma=cfg.thermostat_min_gamma,
            max_gamma=cfg.thermostat_max_gamma,
            k_beta=cfg.thermostat_k_beta,
            k_gamma=cfg.thermostat_k_gamma,
        )
        thermostat = Thermostat(
            teacher_stats=teacher_stats,
            rep_layer_ids=rep_layers,
            config=thermostat_cfg,
            prompt_template=cfg.prompt_template,
            processor_or_tokenizer=processor,
        )
        thermostat_state = ThermostatState(beta=cfg.hsw_beta, gamma=cfg.hsw_gamma)

        if cfg.probe_jsonl is not None:
            probe_loader = build_probe_jsonl_dataloader(
                processor,
                jsonl_path=cfg.probe_jsonl,
                image_root=cfg.probe_image_root,
                prompt_template=cfg.prompt_template,
                batch_size=cfg.probe_batch_size,
            )
        else:
            probe_loader = build_probe_dataloader(
                processor,
                data_root_dir=cfg.probe_root_dir,
                dataset_name=cfg.probe_dataset_name,
                prompt_template=cfg.prompt_template,
                batch_size=cfg.probe_batch_size,
            )
        probe_iter = cycle(probe_loader)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        frame_transform_parallel_calls=cfg.rlds_frame_parallel_calls,
        traj_transform_threads=cfg.rlds_traj_transform_threads,
        traj_read_threads=cfg.rlds_traj_read_threads,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_step_times = deque(maxlen=20)
    optimizer_step_t0 = time.time()

    # Train!
    if cfg.resume_global_step > 0:
        start_step = int(cfg.resume_global_step)
        if resume_state_step is not None and resume_state_step > 0 and resume_state_step != start_step:
            if distributed_state.is_main_process:
                print(
                    "[warn] resume_global_step does not match trainer_state global_step: "
                    f"cfg={start_step} state={resume_state_step}"
                )
    else:
        start_step = int(resume_state_step) if resume_state_step is not None else 0
    if start_step >= cfg.max_steps:
        raise ValueError("resume_global_step must be < max_steps")

    if cfg.enable_periodic_eval:
        if cfg.eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be > 0 when enable_periodic_eval=True")
        if cfg.eval_every_steps % cfg.save_steps != 0:
            raise ValueError("eval_every_steps must be a multiple of save_steps so eval always uses latest checkpoint")

    best_eval_success = -float("inf")
    no_improve_evals = 0

    def _run_periodic_eval(step: int) -> Optional[float]:
        checkpoint_path = adapter_dir if cfg.use_lora else run_dir
        run_note = cfg.eval_run_id_note or (cfg.run_id_note or "train")
        run_note = f"{run_note}-step{step}"

        eval_cmd = [
            sys.executable,
            "experiments/robot/libero/run_libero_eval.py",
            "--model_family",
            "openvla",
            "--pretrained_checkpoint",
            str(checkpoint_path),
            "--task_suite_name",
            cfg.eval_task_suite_name,
            "--center_crop",
            str(cfg.eval_center_crop),
            "--num_trials_per_task",
            str(cfg.eval_num_trials_per_task),
            "--dataset_stats_dir",
            str(run_dir),
            "--run_id_note",
            run_note,
            "--use_wandb",
            "False",
        ]

        if distributed_state.is_main_process:
            print(f"[eval] Running periodic eval at step {step}: {' '.join(eval_cmd)}")
        eval_env = os.environ.copy()
        for key in [
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_ERROR_FILE",
            "ACCELERATE_USE_DISTRIBUTED",
        ]:
            eval_env.pop(key, None)
        result = subprocess.run(eval_cmd, capture_output=True, text=True, env=eval_env)
        if distributed_state.is_main_process:
            if result.returncode != 0:
                print(f"[eval] failed at step {step} with code {result.returncode}")
                if result.stderr:
                    print(result.stderr[-2000:])
                return None

            eval_log_dir = cfg.run_root_dir / "eval_logs"
            pattern = f"EVAL-{cfg.eval_task_suite_name}-openvla-*--{run_note}.txt"
            candidates = sorted(eval_log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                print(f"[eval] no eval log found for step {step} (pattern={pattern})")
                return None

            log_path = candidates[0]
            success_rate = None
            regex = re.compile(r"# successes:\s+\d+\s+\(([0-9.]+)%\)")
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = regex.search(line)
                    if m:
                        success_rate = float(m.group(1))

            if success_rate is None:
                print(f"[eval] unable to parse success rate from log: {log_path}")
                return None

            print(f"[eval] step {step} success_rate={success_rate:.2f}% ({log_path})")
            return success_rate

        return None

    with tqdm.tqdm(total=cfg.max_steps - start_step, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            should_stop = False
            is_optimizer_step = (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute gradient step index
            gradient_step_idx = start_step + (batch_idx // cfg.grad_accumulation_steps)

            # Thermostat 更新（各 rank 都执行，保证广播一致）
            thermo_metrics = None
            if cfg.method_enabled and is_optimizer_step:
                should_update_thermostat = (
                    gradient_step_idx < cfg.thermostat_warmup_steps
                    or gradient_step_idx % cfg.thermostat_update_interval == 0
                )
                if should_update_thermostat:
                    probe_batch = next(probe_iter)
                    probe_batch = {
                        "input_ids": probe_batch["input_ids"].to(device_id),
                        "attention_mask": probe_batch["attention_mask"].to(device_id),
                        "pixel_values": probe_batch["pixel_values"].to(device_id, dtype=torch.bfloat16),
                        "instructions": probe_batch["instructions"],
                    }
                    thermo_metrics = thermostat.maybe_update(gradient_step_idx, vla, probe_batch, thermostat_state)
                hsw_state.beta = thermostat_state.beta
                hsw_state.gamma = thermostat_state.gamma

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0 and cfg.use_wandb:
                payload = {
                    "train_loss": smoothened_loss,
                    "action_accuracy": smoothened_action_accuracy,
                    "l1_loss": smoothened_l1_loss,
                }
                if cfg.method_enabled and thermo_metrics is not None:
                    payload.update(
                        {
                            "drift_d": thermo_metrics["d"],
                            "hsw_beta": thermo_metrics["beta"],
                            "hsw_gamma": thermo_metrics["gamma"],
                            "hsw_g_norm": hsw_state.last_g_norm,
                            "hsw_gprime_norm": hsw_state.last_gprime_norm,
                        }
                    )
                wandb.log(payload, step=gradient_step_idx)

            # Optimizer Step
            if is_optimizer_step:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                progress.update()
                step_time = time.time() - optimizer_step_t0
                optimizer_step_t0 = time.time()
                recent_step_times.append(step_time)

                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    avg_step_time = sum(recent_step_times) / len(recent_step_times) if recent_step_times else step_time
                    if cfg.method_enabled and thermo_metrics is not None:
                        print(
                            "[method] step", gradient_step_idx,
                            "lr", round(current_lr, 8),
                            "loss", round(smoothened_loss, 6),
                            "sec/step", round(avg_step_time, 3),
                            "d", round(thermo_metrics["d"], 6),
                            "base", round(thermo_metrics.get("baseline", 0.0), 6),
                            "beta", round(thermo_metrics["beta"], 6),
                            "gamma", round(thermo_metrics["gamma"], 6),
                            "g", round(hsw_state.last_g_norm, 6),
                            "gpre", round(hsw_state.last_gprime_norm_pre, 6),
                            "gpost", round(hsw_state.last_gprime_norm_post, 6),
                            "scale", round(hsw_state.last_scale, 6),
                            "gf", round(hsw_state.last_gf_norm, 6),
                            "gp", round(hsw_state.last_gp_norm, 6),
                            "gn", round(hsw_state.last_gn_norm, 6),
                        )
                    else:
                        print(
                            "[train] step", gradient_step_idx,
                            "lr", round(current_lr, 8),
                            "loss", round(smoothened_loss, 6),
                            "acc", round(smoothened_action_accuracy, 6),
                            "l1", round(smoothened_l1_loss, 6),
                            "sec/step", round(avg_step_time, 3),
                        )

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if is_optimizer_step and gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If we keep all checkpoints, save into a step-specific directory
                    if cfg.save_latest_checkpoint_only:
                        checkpoint_dir = run_dir
                    else:
                        checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                    def _save_trainer_state(save_dir: Path) -> None:
                        state = {"global_step": gradient_step_idx, "optimizer": optimizer.state_dict()}
                        if scheduler is not None:
                            state["scheduler"] = scheduler.state_dict()
                        torch.save(state, save_dir / "trainer_state.pt")

                    # Save Processor & Weights
                    processor.save_pretrained(checkpoint_dir)
                    if cfg.use_lora:
                        # Always keep latest adapter in adapter_dir; optionally also keep per-step history.
                        vla.module.save_pretrained(adapter_dir)
                        _save_trainer_state(adapter_dir)
                        if not cfg.save_latest_checkpoint_only:
                            vla.module.save_pretrained(checkpoint_dir)
                            _save_trainer_state(checkpoint_dir)
                    else:
                        vla.module.save_pretrained(checkpoint_dir)
                        _save_trainer_state(checkpoint_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and memory-heavy; keep disabled during training to avoid OOM.
                if cfg.use_lora and cfg.merge_lora_during_training:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        cache_dir=cfg.cache_dir,
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

                if cfg.enable_periodic_eval and gradient_step_idx % cfg.eval_every_steps == 0:
                    eval_success = None
                    if distributed_state.is_main_process:
                        eval_success = _run_periodic_eval(gradient_step_idx)

                        if (
                            cfg.early_stopping_enabled
                            and eval_success is not None
                        ):
                            if eval_success > (best_eval_success + cfg.early_stopping_min_delta):
                                best_eval_success = eval_success
                                no_improve_evals = 0
                                print(
                                    f"[early-stop] improvement at step {gradient_step_idx}: "
                                    f"best={best_eval_success:.2f}%"
                                )
                            else:
                                no_improve_evals += 1
                                print(
                                    f"[early-stop] no improvement ({no_improve_evals}/{cfg.early_stopping_patience}) "
                                    f"at step {gradient_step_idx}; best={best_eval_success:.2f}%"
                                )
                                if no_improve_evals >= cfg.early_stopping_patience:
                                    should_stop = True
                                    print("[early-stop] patience reached, requesting stop.")

                    stop_signal = torch.tensor([1 if should_stop else 0], device=device_id, dtype=torch.int32)
                    dist.broadcast(stop_signal, src=0)
                    if int(stop_signal.item()) == 1:
                        print(f"Early stopping at step {gradient_step_idx}.")
                        break

            # Stop training when max_steps is reached
            if gradient_step_idx >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()

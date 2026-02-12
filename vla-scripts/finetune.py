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
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

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
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
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

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
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
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Thermostat 更新（各 rank 都执行，保证广播一致）
            thermo_metrics = None
            if cfg.method_enabled:
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

            if (
                cfg.method_enabled
                and distributed_state.is_main_process
                and gradient_step_idx % 10 == 0
                and (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            ):
                if thermo_metrics is not None:
                    print(
                        "[method] step", gradient_step_idx,
                        "loss", round(smoothened_loss, 6),
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

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
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

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()

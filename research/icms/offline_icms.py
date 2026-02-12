"""离线 ICSM 构建：探针统计 + 子空间 + 脆弱度排序。

只运行一次 Teacher，产出 Uf/Up/mu/C_T 与 meta.json。
"""

import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import hashlib

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

import draccus

from research.probe.hidden_and_mask import (
    build_instruction_mask_for_hidden,
    extract_hidden_states,
    _infer_patch_len,
)
from research.probe.probe_dataset import build_probe_dataloader, build_probe_jsonl_dataloader
from research.hooks.layer_utils import get_llm_layers


DEFAULT_PROMPT_TEMPLATE = "In: {instruction}\nOut:"


@dataclass
class OfflineICMSConfig:
    vla_path: str = "openvla/openvla-7b"
    cache_dir: Path = Path("/opt/data/private/openvla_icms/hf_cache")
    data_root_dir: Path = Path("/opt/data/private/modified_libero_rlds")
    probe_root_dir: Path = Path("/opt/data/private/modified_libero_rlds")
    artifact_dir: Path = Path("/opt/data/private/openvla_icms/artifacts")
    tmp_dir: Path = Path("/opt/data/private/openvla_icms/tmp")

    probe_dataset_name: str = "bridge_orig"
    probe_jsonl: Optional[Path] = None
    probe_image_root: Optional[Path] = None

    batch_size: int = 8
    max_samples: int = 500
    sensitivity_samples: int = 128
    r: int = 128
    epsilon: float = 1e-3

    fragile_dim: Optional[int] = None
    plastic_dim: Optional[int] = None

    layer_ids: Optional[List[int]] = None
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    seed: int = 7
    use_bf16: bool = True

    probe_shuffle_buffer_size: int = 1_000
    probe_num_parallel_calls: int = 1


def _set_seed(seed: int) -> None:
    # 固定随机种子，保证可复现。
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _hash_instructions(instructions: List[str]) -> str:
    # 对指令列表做哈希，写入 meta.json 方便复现核对。
    h = hashlib.sha256()
    for instr in instructions:
        h.update(instr.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parents[2])).decode().strip()
    except Exception:
        return "unknown"


def _resolve_layer_ids(model, layer_ids: Optional[List[int]]) -> List[int]:
    # 允许负索引（从尾部开始）。
    layers = get_llm_layers(model)
    num_layers = len(layers)
    if layer_ids is not None:
        resolved = []
        for layer_id in layer_ids:
            idx = layer_id if layer_id >= 0 else num_layers + layer_id
            if idx < 0 or idx >= num_layers:
                raise IndexError(f"layer_id {layer_id} out of range for {num_layers} layers")
            resolved.append(idx)
        return resolved
    return [num_layers - 3, num_layers - 1]


def _default_group_mapping(layer_ids: List[int], num_layers: int) -> Dict[str, List[int]]:
    last_four = [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1]
    rep_a = last_four[1]
    rep_b = last_four[3]
    return {
        "group_a": [last_four[0], last_four[1]],
        "group_b": [last_four[2], last_four[3]],
        "rep_layers": [rep_a, rep_b],
        "requested_layers": layer_ids,
    }


def _build_probe_loader(cfg: OfflineICMSConfig, processor) -> DataLoader:
    # probe 支持 RLDS 或 JSONL 两种来源。
    if cfg.probe_jsonl is not None:
        return build_probe_jsonl_dataloader(
            processor,
            jsonl_path=cfg.probe_jsonl,
            image_root=cfg.probe_image_root,
            prompt_template=cfg.prompt_template,
            batch_size=cfg.batch_size,
        )
    return build_probe_dataloader(
        processor,
        data_root_dir=cfg.probe_root_dir,
        dataset_name=cfg.probe_dataset_name,
        prompt_template=cfg.prompt_template,
        batch_size=cfg.batch_size,
        shuffle_buffer_size=cfg.probe_shuffle_buffer_size,
        frame_num_parallel_calls=cfg.probe_num_parallel_calls,
    )


def _pool_instruction_hidden(hidden: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    # 对指令 token 做 mean pooling（避免空 mask 除零）。
    mask = mask.to(hidden.device)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def _compute_covariance(x: torch.Tensor) -> torch.Tensor:
    # 计算协方差矩阵，用于 Thermostat。
    x_centered = x - x.mean(dim=0, keepdim=True)
    return (x_centered.T @ x_centered) / max(1, x_centered.shape[0])


def _extract_hidden_by_layers(model, inputs: Dict[str, torch.Tensor], layer_ids: List[int]) -> Dict[int, torch.Tensor]:
    return extract_hidden_states(model, inputs, layer_ids=layer_ids)


def _collect_probe_batches(
    cfg: OfflineICMSConfig,
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[int, List[torch.Tensor]], List[str]]:
    # 采样 probe batch，并对指定层做 pooling。
    layer_ids = _resolve_layer_ids(model, cfg.layer_ids)
    probe_loader = _build_probe_loader(cfg, processor)

    pooled_per_layer: Dict[int, List[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}
    instruction_log: List[str] = []

    seen = 0
    batch_idx = 0
    for batch in probe_loader:
        if seen >= cfg.max_samples:
            break

        inputs = {
            "input_ids": batch["input_ids"].to(device=device),
            "attention_mask": batch["attention_mask"].to(device=device),
            "pixel_values": batch["pixel_values"].to(device=device, dtype=dtype),
        }
        instructions = batch["instructions"]

        with torch.no_grad():
            hidden_by_layer = _extract_hidden_by_layers(model, inputs, layer_ids)
        for layer_id, hidden in hidden_by_layer.items():
            mask = build_instruction_mask_for_hidden(
                processor,
                model,
                inputs,
                instructions[0] if len(instructions) == 1 else instructions,
                prompt_template=cfg.prompt_template,
            )
            pooled = _pool_instruction_hidden(hidden, mask)
            pooled_per_layer[layer_id].append(pooled.cpu())

        instruction_log.extend(instructions)
        seen += inputs["input_ids"].shape[0]
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"[icms] pooled {seen}/{cfg.max_samples} probe samples", flush=True)

    return pooled_per_layer, instruction_log


def _compute_svd_subspace(x: torch.Tensor, r: int) -> torch.Tensor:
    # 对中心化矩阵做 SVD，取前 r 个方向。
    x_centered = x - x.mean(dim=0, keepdim=True)
    if x_centered.dtype in (torch.bfloat16, torch.float16):
        x_centered = x_centered.float()
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    max_r = min(vh.shape[0], vh.shape[1])
    r = min(r, max_r)
    return vh.T[:, :r]


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(dim=-1)


def _compute_sensitivities(
    model,
    processor,
    direction_matrix: torch.Tensor,
    inputs_list: List[Dict[str, torch.Tensor]],
    masks_list: List[torch.BoolTensor],
    layer_id: int,
    epsilon: float,
) -> torch.Tensor:
    # 对每个方向做小扰动，计算输出分布变化（KL）。
    device = direction_matrix.device
    model_dtype = next(model.parameters()).dtype
    if direction_matrix.dtype != model_dtype:
        direction_matrix = direction_matrix.to(dtype=model_dtype)
    layers = get_llm_layers(model)
    layer = layers[layer_id]
    r = direction_matrix.shape[1]

    baseline_logits: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs in inputs_list:
            out = model(**inputs, return_dict=True)
            logits = out.logits
            patch_len = int(_infer_patch_len(model, inputs))
            pos = inputs["attention_mask"].sum(dim=1) - 1 + patch_len
            baseline_logits.append(logits[torch.arange(logits.shape[0]), pos])

    print(f"[icms] sensitivity: layer={layer_id} r={r} batches={len(inputs_list)}", flush=True)

    sens = torch.zeros(r, device=device)
    for k in range(r):
        direction = direction_matrix[:, k].view(1, 1, -1)

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                mask = masks_list[_hook.step].to(hidden.device)
                new_hidden = hidden + epsilon * direction * mask.unsqueeze(-1)
                return (new_hidden,) + output[1:]
            mask = masks_list[_hook.step].to(output.device)
            return output + epsilon * direction * mask.unsqueeze(-1)

        handle = layer.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                kl_vals = []
                for idx, inputs in enumerate(inputs_list):
                    _hook.step = idx
                    out = model(**inputs, return_dict=True)
                    logits = out.logits
                    patch_len = int(_infer_patch_len(model, inputs))
                    pos = inputs["attention_mask"].sum(dim=1) - 1 + patch_len
                    pert_logits = logits[torch.arange(logits.shape[0]), pos]

                    p = torch.softmax(baseline_logits[idx], dim=-1)
                    q = torch.softmax(pert_logits, dim=-1)
                    kl_vals.append(_kl_divergence(p, q).mean())
                sens[k] = torch.stack(kl_vals).mean()
        finally:
            handle.remove()

        if (k + 1) % 8 == 0 or (k + 1) == r:
            print(f"[icms] sensitivity progress: {k + 1}/{r}", flush=True)

    return sens


def _prepare_sensitivity_batches(
    cfg: OfflineICMSConfig,
    processor,
    model,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.BoolTensor]]:
    loader = _build_probe_loader(cfg, processor)
    inputs_list: List[Dict[str, torch.Tensor]] = []
    masks_list: List[torch.BoolTensor] = []

    seen = 0
    batch_idx = 0
    for batch in loader:
        if seen >= cfg.sensitivity_samples:
            break
        inputs = {
            "input_ids": batch["input_ids"].to(device=device),
            "attention_mask": batch["attention_mask"].to(device=device),
            "pixel_values": batch["pixel_values"].to(device=device, dtype=dtype),
        }
        instructions = batch["instructions"]
        mask = build_instruction_mask_for_hidden(
            processor,
            model,
            inputs,
            instructions[0] if len(instructions) == 1 else instructions,
            prompt_template=cfg.prompt_template,
        )
        inputs_list.append(inputs)
        masks_list.append(mask)
        seen += inputs["input_ids"].shape[0]
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"[icms] prepared {seen}/{cfg.sensitivity_samples} sensitivity samples", flush=True)

    return inputs_list, masks_list


def run_offline_icms_core(cfg: OfflineICMSConfig) -> None:
    # 离线 ICSM 主流程。
    _set_seed(cfg.seed)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cfg.cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cfg.cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cfg.cache_dir))
    os.environ.setdefault("TORCH_HOME", str(cfg.cache_dir.parent / "torch_cache"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (cfg.use_bf16 and device.type == "cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True, cache_dir=cfg.cache_dir)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=cfg.cache_dir,
    ).to(device)
    model.eval()

    layer_ids = _resolve_layer_ids(model, cfg.layer_ids)
    pooled_per_layer, instruction_log = _collect_probe_batches(cfg, model, processor, device, dtype)

    probe_hash = _hash_instructions(instruction_log)

    inputs_list, masks_list = _prepare_sensitivity_batches(cfg, processor, model, device, dtype)

    num_layers = len(get_llm_layers(model))
    group_meta = _default_group_mapping(layer_ids, num_layers)

    meta = {
        "vla_path": cfg.vla_path,
        "layer_ids": layer_ids,
        "r": cfg.r,
        "epsilon": cfg.epsilon,
        "probe_hash": probe_hash,
        "fragile_dim": cfg.fragile_dim,
        "plastic_dim": cfg.plastic_dim,
        "prompt_template": cfg.prompt_template,
        "group_mapping": group_meta,
        "commit": _get_git_commit(),
    }

    for layer_id, pooled_list in pooled_per_layer.items():
        x = torch.cat(pooled_list, dim=0)
        mu = x.mean(dim=0)
        cov = _compute_covariance(x)

        subspace = _compute_svd_subspace(x, cfg.r).to(device)
        sens = _compute_sensitivities(
            model,
            processor,
            subspace,
            inputs_list,
            masks_list,
            layer_id=layer_id,
            epsilon=cfg.epsilon,
        )

        order = torch.argsort(sens, descending=True)
        fragile_dim = cfg.fragile_dim or max(1, cfg.r // 2)
        plastic_dim = cfg.plastic_dim or max(1, cfg.r // 4)
        fragile_idx = order[:fragile_dim]
        plastic_idx = order[-plastic_dim:]

        uf = subspace[:, fragile_idx].cpu()
        up = subspace[:, plastic_idx].cpu()

        torch.save(uf, cfg.artifact_dir / f"U{layer_id}_f.pt")
        torch.save(up, cfg.artifact_dir / f"U{layer_id}_p.pt")
        torch.save(mu.cpu(), cfg.artifact_dir / f"mu{layer_id}.pt")
        torch.save(cov.cpu(), cfg.artifact_dir / f"C_T{layer_id}.pt")

    with open(cfg.artifact_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@draccus.wrap()
def run_offline_icms(cfg: OfflineICMSConfig) -> None:
    run_offline_icms_core(cfg)


if __name__ == "__main__":
    run_offline_icms()

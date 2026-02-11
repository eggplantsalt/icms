from __future__ import annotations

"""探针输入、hidden states 与指令掩码工具。

用途：
1) 构建 OpenVLA prompt 输入。
2) 获取指定层 hidden states。
3) 生成指令区间 mask（支持多模态 hidden 对齐）。
"""

from typing import Dict, List

import logging

import torch
from transformers import PreTrainedTokenizerBase

from research.hooks.layer_utils import get_llm_layers
logger = logging.getLogger(__name__)


_DEFAULT_PROMPT_TEMPLATE = "In: {instruction}\nOut:"


def build_inputs(processor, image, instruction: str, device, dtype) -> Dict[str, torch.Tensor]:
    # 使用与 OpenVLA 推理一致的默认 prompt 形状。
    prompt = _DEFAULT_PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt")

    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if key == "pixel_values" and value.is_floating_point():
            inputs[key] = value.to(device=device, dtype=dtype)
        else:
            inputs[key] = value.to(device=device)

    return inputs


def extract_hidden_states(model, inputs: Dict[str, torch.Tensor], layer_ids: List[int]) -> Dict[int, torch.Tensor]:
    # 前向时开启 hidden_states 输出。
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True.")

    # LLM hidden_states 通常包含 embedding 输出，需要按 LLM 层数做偏移对齐。
    llm_layers = get_llm_layers(model)
    offset = 1 if len(hidden_states) > len(llm_layers) else 0
    num_layers = len(llm_layers)
    selected: Dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        idx = layer_id if layer_id >= 0 else num_layers + layer_id
        if idx < 0 or idx >= num_layers:
            raise IndexError(f"layer_id {layer_id} is out of range for {num_layers} layers")
        actual_idx = idx + offset
        if actual_idx < 0 or actual_idx >= len(hidden_states):
            raise IndexError(f"layer_id {layer_id} maps to hidden_states[{actual_idx}] out of range")
        selected[layer_id] = hidden_states[actual_idx]

    return selected


def build_instruction_mask(
    processor_or_tokenizer,
    inputs: Dict[str, torch.Tensor],
    instruction: str | List[str],
    prompt_template: str | None = None,
) -> torch.BoolTensor:
    # 构建只覆盖指令 token 的布尔 mask。
    tokenizer = _get_tokenizer(processor_or_tokenizer)
    prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE

    if "{instruction}" not in prompt_template:
        raise ValueError("prompt_template must include '{instruction}' placeholder")

    prefix, suffix = prompt_template.split("{instruction}", 1)
    if "Out:" not in suffix:
        raise ValueError("prompt_template must include 'Out:' after instruction")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids to be 2D [B,T], got shape {tuple(input_ids.shape)}")

    batch_size, seq_len = input_ids.shape
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)

    instructions = instruction if isinstance(instruction, list) else [instruction] * batch_size
    if len(instructions) != batch_size:
        raise ValueError("Number of instructions must match batch size")

    for batch_idx in range(batch_size):
        instr = instructions[batch_idx]
        prompt_text = f"{prefix}{instr}{suffix}"

        prompt_ids = _encode(tokenizer, prompt_text, add_special_tokens=False)
        prefix_ids = _encode(tokenizer, prefix, add_special_tokens=False)
        prefix_instruction_ids = _encode(tokenizer, f"{prefix}{instr}", add_special_tokens=False)

        instruction_start = len(prefix_ids)
        instruction_end = len(prefix_instruction_ids)

        instruction_start, instruction_end = _refine_instruction_span_with_offsets(
            tokenizer, prompt_text, len(prefix), instruction_start, instruction_end
        )

        valid_len = int(attention_mask[batch_idx].sum().item())
        seq = input_ids[batch_idx, :valid_len].tolist()
        prompt_start = _find_subsequence(seq, prompt_ids)
        if prompt_start == -1:
            raise ValueError(
                "Prompt tokens were not found in input_ids; cannot align instruction mask. "
                "Check that the same prompt template was used to build inputs."
            )

        start = prompt_start + instruction_start
        end = prompt_start + instruction_end
        if end > valid_len:
            raise ValueError("Instruction span exceeds the non-padding token length.")

        mask[batch_idx, start:end] = True

    mask = mask & attention_mask.bool()
    return mask


def build_instruction_mask_for_hidden(
    processor_or_tokenizer,
    model,
    inputs: Dict[str, torch.Tensor],
    instruction: str,
    prompt_template: str | None = None,
) -> torch.BoolTensor:
    # 对齐多模态 hidden：在 BOS 后插入 patch span。
    base_mask = build_instruction_mask(
        processor_or_tokenizer=processor_or_tokenizer,
        inputs=inputs,
        instruction=instruction,
        prompt_template=prompt_template,
    )

    patch_len = _infer_patch_len(model, inputs)
    if patch_len == 0:
        return base_mask

    batch_size = base_mask.shape[0]
    patch_mask = torch.zeros((batch_size, patch_len), dtype=torch.bool, device=base_mask.device)

    # Align with PrismaticForConditionalGeneration.forward: insert patches after BOS token (index 1).
    hidden_mask = torch.cat([base_mask[:, :1], patch_mask, base_mask[:, 1:]], dim=1)
    return hidden_mask


def _get_tokenizer(processor_or_tokenizer) -> PreTrainedTokenizerBase:
    if hasattr(processor_or_tokenizer, "tokenizer"):
        return processor_or_tokenizer.tokenizer
    return processor_or_tokenizer


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, add_special_tokens: bool) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_attention_mask=False)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded["input_ids"])


def _find_out_marker_start(
    tokenizer: PreTrainedTokenizerBase, prompt_ids: List[int], min_start: int
) -> int | None:
    candidates = ["\nOut:", " Out:", "Out:"]
    starts: List[int] = []
    for marker in candidates:
        marker_ids = _encode(tokenizer, marker, add_special_tokens=False)
        if not marker_ids:
            continue
        start = _find_subsequence(prompt_ids, marker_ids, start_at=min_start)
        if start != -1:
            starts.append(start)

    return min(starts) if starts else None


def _refine_instruction_span_with_offsets(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    prefix_char_len: int,
    fallback_start: int,
    fallback_end: int,
) -> tuple[int, int]:
    if not getattr(tokenizer, "is_fast", False):
        return fallback_start, fallback_end

    encoded = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return fallback_start, fallback_end

    out_char_start = prompt_text.find("Out:", prefix_char_len)
    if out_char_start == -1:
        raise ValueError("Prompt text does not contain 'Out:'; cannot build instruction mask.")

    instr_char_start = prefix_char_len
    instr_char_end = out_char_start

    instr_token_start = None
    instr_token_end = None
    for idx, (start, end) in enumerate(offsets):
        if end <= instr_char_start:
            continue
        if start >= instr_char_end:
            if instr_token_end is None:
                instr_token_end = idx
            break
        if instr_token_start is None:
            instr_token_start = idx

    if instr_token_start is None or instr_token_end is None:
        return fallback_start, fallback_end

    return instr_token_start, instr_token_end


def _find_subsequence(sequence: List[int], subseq: List[int], start_at: int = 0) -> int:
    if not subseq:
        return -1
    max_start = len(sequence) - len(subseq)
    for idx in range(start_at, max_start + 1):
        if sequence[idx : idx + len(subseq)] == subseq:
            return idx
    return -1


def _infer_patch_len(model, inputs: Dict[str, torch.Tensor]) -> int:
    # 从 vision_backbone 输出推断 patch token 的数量。
    if hasattr(model, "module"):
        model = model.module
    if "pixel_values" not in inputs:
        raise ValueError("inputs must include 'pixel_values' to infer patch token count")
    if not hasattr(model, "vision_backbone"):
        raise ValueError("model must expose 'vision_backbone' to infer patch token count")

    pixel_values = inputs["pixel_values"]
    if pixel_values.dim() != 4:
        raise ValueError(f"Expected pixel_values to be 4D [B,C,H,W], got shape {tuple(pixel_values.shape)}")

    with torch.no_grad():
        patch_features = model.vision_backbone(pixel_values)

    if patch_features.dim() != 3:
        raise ValueError(
            "Expected vision_backbone output to be 3D [B,P,D], got shape "
            f"{tuple(patch_features.shape)}"
        )

    return int(patch_features.shape[1])

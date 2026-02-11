from __future__ import annotations

import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research.probe.hidden_and_mask import (  # noqa: E402
    build_inputs,
    build_instruction_mask_for_hidden,
    extract_hidden_states,
)


def _find_subsequence(sequence: list[int], subseq: list[int]) -> int:
    if not subseq:
        return -1
    max_start = len(sequence) - len(subseq)
    for idx in range(max_start + 1):
        if sequence[idx : idx + len(subseq)] == subseq:
            return idx
    return -1


def _hidden_window_tokens(tokenizer, input_ids: list[int], patch_len: int, start: int, end: int) -> list[str]:
    tokens: list[str] = []
    for idx in range(start, end):
        if idx == 0:
            tokens.append(tokenizer.convert_ids_to_tokens(input_ids[0]))
        elif 1 <= idx <= patch_len:
            tokens.append("<PATCH>")
        else:
            token_idx = idx - patch_len
            if token_idx < len(input_ids):
                tokens.append(tokenizer.convert_ids_to_tokens(input_ids[token_idx]))
            else:
                tokens.append("<PAD>")
    return tokens


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    image = Image.new("RGB", (224, 224), color=(0, 0, 0))
    instruction = "pick up the black cup"

    inputs = build_inputs(processor, image, instruction, device=device, dtype=dtype)
    input_ids = inputs["input_ids"][0].tolist()
    input_len = len(input_ids)

    with torch.no_grad():
        hidden = extract_hidden_states(model, inputs, layer_ids=[-1])
    hidden_len = int(hidden[-1].shape[1])

    hidden_mask = build_instruction_mask_for_hidden(processor, model, inputs, instruction)
    mask_len = int(hidden_mask.shape[1])
    patch_len = hidden_len - input_len
    mask_patch_len = mask_len - input_len

    print("input_ids_len:", input_len)
    print("hidden_len:", hidden_len)
    print("mask_len:", mask_len)
    print("patch_len:", patch_len)
    print("mask_patch_len:", mask_patch_len)
    if mask_len != hidden_len:
        print("WARNING: mask_len does not match hidden_len; alignment may be incorrect.")

    bos_window_end = min(mask_len, max(8, 2 + patch_len))
    bos_tokens = _hidden_window_tokens(processor.tokenizer, input_ids, patch_len, 0, bos_window_end)
    bos_mask = hidden_mask[0, 0:bos_window_end].tolist()
    print("BOS window tokens:", bos_tokens)
    print("BOS window mask:", bos_mask)
    if patch_len > 0:
        patch_span = hidden_mask[0, 1 : 1 + patch_len]
        print("patch_span_all_false:", bool((patch_span == 0).all().item()))

    prompt = f"In: {instruction}\nOut:"
    prompt_ids = processor.tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_start = _find_subsequence(input_ids, prompt_ids)
    out_ids = processor.tokenizer("\nOut:", add_special_tokens=False).input_ids
    out_start = _find_subsequence(prompt_ids, out_ids)
    if out_start == -1:
        out_ids = processor.tokenizer("Out:", add_special_tokens=False).input_ids
        out_start = _find_subsequence(prompt_ids, out_ids)

    if prompt_start != -1 and out_start != -1:
        out_input_idx = prompt_start + out_start
        out_hidden_idx = out_input_idx + (patch_len if out_input_idx >= 1 else 0)
        window_start = max(0, out_hidden_idx - 3)
        window_end = min(mask_len, out_hidden_idx + len(out_ids) + 3)
        out_tokens = _hidden_window_tokens(processor.tokenizer, input_ids, patch_len, window_start, window_end)
        out_mask = hidden_mask[0, window_start:window_end].tolist()
        print("Out window tokens:", out_tokens)
        print("Out window mask:", out_mask)
    else:
        print("WARNING: Unable to locate Out: token span for hidden-aligned window display.")


if __name__ == "__main__":
    main()

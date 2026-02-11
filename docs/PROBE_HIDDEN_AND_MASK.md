# Hidden States + Instruction Mask

This module provides minimal utilities for (A) extracting hidden states and (B) building an instruction-only token mask, based on the actual OpenVLA prompt in the repo.

## Prompt Source (OpenVLA default)

The default inference prompt is defined in [experiments/robot/openvla_utils.py](experiments/robot/openvla_utils.py) under the `OpenVLA` branch:

- `prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"`

The utilities here use the same template shape (`In: {instruction}\nOut:`) unless a `prompt_template` override is provided.

## Hidden States Retrieval

Hidden states are exposed by the HF model in [prismatic/extern/hf/modeling_prismatic.py](prismatic/extern/hf/modeling_prismatic.py). In `PrismaticForConditionalGeneration.forward`, `output_hidden_states=True` is forwarded to the language model, and `hidden_states` are returned in `PrismaticCausalLMOutputWithPast`:

- `outputs = model(**inputs, output_hidden_states=True, return_dict=True)`
- `outputs.hidden_states` is a tuple of `[B, T, D]` tensors.

## Instruction Mask Rules

- Mask covers the instruction content between `In:` and `Out:` tokens.
- By default, the `In:` prefix and the `Out:` marker are **excluded** from the mask.
- Tokens at and after `Out:` are **always excluded**.

### Out: Marker Failure Handling

If the tokenized `Out:` marker cannot be located, `build_instruction_mask` **raises a ValueError** with a clear message. This avoids silently producing a wrong mask.

### Notes on Alignment

The mask length matches `input_ids` length. When the model runs with images, hidden states are computed over a multimodal sequence (vision patch tokens are inserted after the BOS token). If you need a mask aligned to those hidden states, you must insert a block of `False` tokens for the patch span after BOS.

## Multimodal Alignment

In [prismatic/extern/hf/modeling_prismatic.py](prismatic/extern/hf/modeling_prismatic.py), `PrismaticForConditionalGeneration.forward` builds multimodal embeddings by inserting projected vision patches **after the BOS token**:

- `multimodal_embeddings = torch.cat([input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1)`

This means `hidden_states` length is `input_ids_len + patch_len`, where `patch_len` is the number of vision patches produced by `model.vision_backbone(pixel_values)`.

The helper `build_instruction_mask_for_hidden(...)` computes `patch_len` from `model.vision_backbone` and inserts a `False` span of that length right after BOS so the mask aligns to hidden states. If `pixel_values` or `vision_backbone` is unavailable, it raises a `ValueError` rather than guessing.

To verify lengths in your environment, run `research/probe/smoke_test_hidden_and_mask.py`, which prints `input_ids_len` and `hidden_states[-1]_len`. For multimodal inputs, the two lengths differ as described above, matching the insertion rule in `PrismaticForConditionalGeneration.forward`.

## Files Added

- `research/probe/hidden_and_mask.py`: build inputs, extract hidden states, build instruction mask.
- `research/probe/smoke_test_hidden_and_mask.py`: runnable sanity check.
- `research/probe/smoke_test_alignment.py`: alignment check for hidden-length mask and patch span.

from __future__ import annotations  # Enable postponed evaluation of annotations.
#
import os  # Path utilities for repo root discovery.
import sys  # sys.path injection for local imports.
#
import torch  # Tensor operations and device selection.
from PIL import Image  # Create a dummy image for the smoke test.
from transformers import AutoModelForVision2Seq, AutoProcessor  # HF auto classes for OpenVLA.
#
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # Resolve repo root.
if REPO_ROOT not in sys.path:  # Ensure repo root is importable.
    sys.path.insert(0, REPO_ROOT)  # Prepend repo root to sys.path.
#
from research.probe.hidden_and_mask import (  # noqa: E402
    build_inputs,  # Build processor inputs.
    build_instruction_mask,  # Build instruction-only mask on text sequence.
    extract_hidden_states,  # Extract hidden states from model outputs.
)
#

def _find_subsequence(sequence: list[int], subseq: list[int]) -> int:  # Find subseq start index.
    if not subseq:  # Empty subsequence is invalid here.
        return -1  # Signal not found.
    max_start = len(sequence) - len(subseq)  # Last possible start index.
    for idx in range(max_start + 1):  # Scan through all candidate starts.
        if sequence[idx : idx + len(subseq)] == subseq:  # Exact match check.
            return idx  # Return first match.
    return -1  # No match found.
#

def main() -> None:  # Entry point for the smoke test.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32  # Select compute dtype.
#
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)  # Load processor.
    model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", trust_remote_code=True)  # Load model.
    model = model.to(device=device, dtype=dtype)  # Move model to device.
    model.eval()  # Set eval mode.
#
    image = Image.new("RGB", (224, 224), color=(0, 0, 0))  # Dummy black image.
    instruction = "pick up the black cup"  # Example instruction text.
#
    inputs = build_inputs(processor, image, instruction, device=device, dtype=dtype)  # Build model inputs.
    input_ids = inputs["input_ids"]  # Token IDs for the prompt.
    input_len = int(input_ids.shape[1])  # Text sequence length.
#
    print("input_ids.shape:", tuple(input_ids.shape))  # Print token shape.
#
    with torch.no_grad():  # Disable gradients for inference.
        hidden = extract_hidden_states(model, inputs, layer_ids=[0, -1])  # Fetch hidden states.
    for layer_id, tensor in hidden.items():  # Iterate over selected layers.
        print(f"hidden[{layer_id}].shape:", tuple(tensor.shape))  # Print hidden shape per layer.
#
    hidden_len = int(hidden[-1].shape[1])  # Hidden sequence length.
    print("input_ids_len:", input_len)  # Print input length.
    print("hidden_states[-1]_len:", hidden_len)  # Print hidden length.
    print("len_equal:", input_len == hidden_len)  # Report equality.
#
    instr_mask = build_instruction_mask(processor, inputs, instruction)  # Build instruction mask on text tokens.
    print("instr_mask.sum():", int(instr_mask.sum().item()))  # Print number of masked tokens.
#
    tokenizer = processor.tokenizer  # Convenience handle for tokenizer.
    prompt = f"In: {instruction}\nOut:"  # Construct prompt string.
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids  # Tokenize prompt.
    seq = input_ids[0].tolist()  # Convert batch-0 input_ids to list.
    prompt_start = _find_subsequence(seq, prompt_ids)  # Locate prompt span.
    out_ids = tokenizer("\nOut:", add_special_tokens=False).input_ids  # Tokenize Out: with newline.
    out_start = _find_subsequence(prompt_ids, out_ids)  # Find Out: in prompt tokens.
    if out_start == -1:  # If newline form not found.
        out_ids = tokenizer("Out:", add_special_tokens=False).input_ids  # Tokenize Out: without newline.
        out_start = _find_subsequence(prompt_ids, out_ids)  # Find Out: again.
#
    if prompt_start != -1 and out_start != -1:  # Ensure both prompt and Out: are found.
        out_global = prompt_start + out_start  # Convert to global index in input_ids.
        window_start = max(0, out_global - 3)  # Window start around Out:.
        window_end = min(len(seq), out_global + len(out_ids) + 3)  # Window end around Out:.
        tokens = tokenizer.convert_ids_to_tokens(seq[window_start:window_end])  # Tokens around Out:.
        mask_vals = instr_mask[0, window_start:window_end].tolist()  # Mask values around Out:.
        print("Out: window tokens:", tokens)  # Print tokens for inspection.
        print("Out: window mask:", mask_vals)  # Print mask flags for inspection.
    else:  # Out: span missing.
        print("WARNING: Unable to locate Out: token span in input_ids for display.")  # Warn.
#

if __name__ == "__main__":  # Script execution guard.
    main()  # Run main.

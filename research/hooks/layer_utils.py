from __future__ import annotations

"""模型解包与 LLM 层访问工具。"""

from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import PeftModel
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - optional import
    FSDP = None


def unwrap_model(model):
    # 逐层剥离 DDP/FSDP/PEFT 包装。
    if isinstance(model, DDP):
        model = model.module
    if FSDP is not None and isinstance(model, FSDP):
        model = model.module
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    return model


def get_llm_layers(model) -> List[torch.nn.Module]:
    # 获取底层 LLM 的 transformer layers 列表。
    model = unwrap_model(model)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return list(model.language_model.model.layers)
    if hasattr(model, "llm_backbone") and hasattr(model.llm_backbone, "llm"):
        if hasattr(model.llm_backbone.llm, "model"):
            return list(model.llm_backbone.llm.model.layers)
    raise ValueError("Unable to resolve LLM layers from model")

from __future__ import annotations

"""Thermostat 闭环控制：根据漂移调节 beta/gamma。"""

from dataclasses import dataclass, field
import contextlib
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist

from research.probe.hidden_and_mask import build_instruction_mask_for_hidden, extract_hidden_states


@dataclass
class ThermostatConfig:
    update_interval: int = 100
    warmup_steps: int = 200
    min_beta: float = 0.2
    max_beta: float = 1.0
    min_gamma: float = 0.1
    max_gamma: float = 1.0
    k_beta: float = 1.0
    k_gamma: float = 1.0


@dataclass
class ThermostatState:
    beta: float
    gamma: float
    drift_baseline: Optional[float] = None
    drift_history: List[float] = field(default_factory=list)

    def update_baseline(self, d: float) -> None:
        self.drift_history.append(d)
        self.drift_baseline = sum(self.drift_history) / len(self.drift_history)


class Thermostat:
    def __init__(
        self,
        teacher_stats: Dict[int, Dict[str, torch.Tensor]],
        rep_layer_ids: List[int],
        config: ThermostatConfig,
        prompt_template: str,
        processor_or_tokenizer,
    ) -> None:
        self.teacher_stats = teacher_stats
        self.rep_layer_ids = rep_layer_ids
        self.config = config
        self.prompt_template = prompt_template
        self.processor_or_tokenizer = processor_or_tokenizer

    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=0, keepdim=True)
        return (x_centered.T @ x_centered) / max(1, x_centered.shape[0])

    def _pool_instruction_hidden(self, hidden: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _compute_drift(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        # 计算 Student 协方差与 Teacher 协方差的距离。
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
        }
        instructions = batch["instructions"]

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            hidden_by_layer = extract_hidden_states(model, inputs, self.rep_layer_ids)

        drifts = []
        for layer_id in self.rep_layer_ids:
            hidden = hidden_by_layer[layer_id]
            mask = build_instruction_mask_for_hidden(
                self.processor_or_tokenizer,
                model,
                inputs,
                instructions,
                prompt_template=self.prompt_template,
            )
            pooled = self._pool_instruction_hidden(hidden, mask)
            c_s = self._compute_covariance(pooled)
            c_t = self.teacher_stats[layer_id]["C_T"].to(c_s.device)
            drifts.append(torch.norm(c_s - c_t, p="fro"))

        return float(torch.stack(drifts).mean().item())

    def maybe_update(
        self,
        step: int,
        model,
        batch: Dict[str, torch.Tensor],
        state: ThermostatState,
    ) -> Dict[str, float]:
        # 预热阶段只估计漂移基线；之后按间隔更新 beta/gamma。
        if step < self.config.warmup_steps:
            # warmup 阶段每步都更新基线，避免 update_interval 太大导致基线样本过少
            d = self._compute_drift(model, batch)
            state.update_baseline(d)
            return {"d": d, "beta": state.beta, "gamma": state.gamma, "baseline": state.drift_baseline or 0.0}

        if step % self.config.update_interval != 0:
            return {"d": state.drift_baseline or 0.0, "beta": state.beta, "gamma": state.gamma, "baseline": state.drift_baseline or 0.0}

        d = self._compute_drift(model, batch)
        baseline = state.drift_baseline or d
        if state.drift_baseline is None:
            state.drift_baseline = baseline
        delta = max(0.0, d - baseline)

        beta = self.config.max_beta / (1.0 + self.config.k_beta * delta)
        gamma = self.config.max_gamma / (1.0 + self.config.k_gamma * delta)

        beta = max(self.config.min_beta, min(self.config.max_beta, beta))
        gamma = max(self.config.min_gamma, min(self.config.max_gamma, gamma))

        state.beta = beta
        state.gamma = gamma

        if dist.is_initialized():
            tensor = torch.tensor([beta, gamma], device=batch["input_ids"].device)
            dist.broadcast(tensor, src=0)
            state.beta = float(tensor[0].item())
            state.gamma = float(tensor[1].item())

        return {"d": d, "beta": state.beta, "gamma": state.gamma, "baseline": state.drift_baseline or baseline}

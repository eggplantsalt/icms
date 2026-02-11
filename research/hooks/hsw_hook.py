from __future__ import annotations

"""HSW 梯度手术：在指定层对梯度做子空间投影重组。"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from research.hooks.layer_utils import get_llm_layers, unwrap_model


@dataclass
class HSWState:
    beta: float
    gamma: float
    eps: float = 1e-8

    last_g_norm: float = 0.0
    last_gprime_norm: float = 0.0

    def update_norms(self, g_norm: torch.Tensor, gprime_norm: torch.Tensor) -> None:
        # 记录最近一次梯度范数，便于日志输出。
        self.last_g_norm = float(g_norm.detach().cpu().item())
        self.last_gprime_norm = float(gprime_norm.detach().cpu().item())


class HSWProjector:
    def __init__(self, uf: torch.Tensor, up: torch.Tensor) -> None:
        self.uf = uf
        self.up = up
        self.uf_t = uf.t()
        self.up_t = up.t()

    def project(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将梯度分解为 fragile / plastic / null 三部分。
        orig_shape = g.shape
        g_flat = g.reshape(-1, orig_shape[-1])

        gf = (g_flat @ self.uf) @ self.uf_t
        gp = (g_flat @ self.up) @ self.up_t
        gn = g_flat - gf - gp

        return gf.reshape(orig_shape), gp.reshape(orig_shape), gn.reshape(orig_shape)


class HSWHookManager:
    def __init__(
        self,
        model,
        layer_to_subspace: Dict[int, tuple[torch.Tensor, torch.Tensor]],
        state: HSWState,
    ) -> None:
        self.model = model
        self.layer_to_subspace = layer_to_subspace
        self.state = state
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.projectors: Dict[int, HSWProjector] = {}

    def register(self) -> None:
        # 在目标层注册 backward hook。
        layers = get_llm_layers(self.model)
        for layer_id, (uf, up) in self.layer_to_subspace.items():
            layer = layers[layer_id]
            projector = HSWProjector(uf=uf, up=up)
            self.projectors[layer_id] = projector

            def _hook(_module, grad_inputs, grad_outputs, _layer_id=layer_id):
                # 对输出梯度做子空间重组并范数约束。
                if not grad_outputs:
                    return grad_outputs
                grad = grad_outputs[0]
                if grad is None:
                    return grad_outputs
                projector = self.projectors[_layer_id]
                gf, gp, gn = projector.project(grad)
                beta = self.state.beta
                gamma = self.state.gamma
                gprime = gn + beta * gp + gamma * gf

                g_norm = torch.norm(grad)
                gprime_norm = torch.norm(gprime)
                scale = torch.minimum(torch.tensor(1.0, device=gprime.device), g_norm / (gprime_norm + self.state.eps))
                gprime = gprime * scale

                self.state.update_norms(g_norm, gprime_norm)
                return (gprime,) + grad_outputs[1:]

            handle = layer.register_full_backward_hook(_hook)
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_subspaces(artifact_dir: str, layer_ids: List[int], device: torch.device) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    # 读取离线 ICSM 产物（Uf/Up）。
    layer_to_subspace: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_id in layer_ids:
        uf = torch.load(f"{artifact_dir}/U{layer_id}_f.pt", map_location=device)
        up = torch.load(f"{artifact_dir}/U{layer_id}_p.pt", map_location=device)
        layer_to_subspace[layer_id] = (uf.to(device), up.to(device))
    return layer_to_subspace

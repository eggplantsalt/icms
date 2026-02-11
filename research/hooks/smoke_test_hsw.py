from __future__ import annotations

import torch

from research.hooks.hsw_hook import HSWProjector, HSWState


def main() -> None:
    torch.manual_seed(7)
    d = 16
    r = 4

    uf = torch.randn(d, r)
    up = torch.randn(d, r)
    uf, _ = torch.linalg.qr(uf)
    up, _ = torch.linalg.qr(up)

    projector = HSWProjector(uf=uf, up=up)
    state = HSWState(beta=0.5, gamma=0.2, eps=1e-8)

    g = torch.randn(2, 3, d)
    gf, gp, gn = projector.project(g)
    gprime = gn + state.beta * gp + state.gamma * gf

    g_norm = torch.norm(g)
    gprime_norm = torch.norm(gprime)
    scale = torch.minimum(torch.tensor(1.0), g_norm / (gprime_norm + state.eps))
    gprime = gprime * scale

    print("g_norm:", float(g_norm))
    print("gprime_norm:", float(torch.norm(gprime)))
    print("norm_preserved:", float(torch.norm(gprime)) <= float(g_norm))


if __name__ == "__main__":
    main()

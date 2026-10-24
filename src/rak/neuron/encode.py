"""Poisson spike encoder (Phase 1.5)."""

from __future__ import annotations

import torch
from torch import Tensor


def poisson_encode(images: Tensor, intensity: float, n_steps: int,
                   max_rate_hz: float = 63.75, seed: int = 0) -> Tensor:
    """Encode images into Poisson spike trains.

    images: (B, n_in) in [0, 1]. Return (n_steps, B, n_in) float32.
    """
    g = torch.Generator().manual_seed(seed)
    rates = images * max_rate_hz * intensity  # Hz
    prob = rates * (1.0 / 1000.0)  # per ms (dt=1ms)
    prob = prob.clamp(0, 1)
    u = torch.rand(n_steps, images.shape[0], images.shape[1], generator=g)
    return (u < prob.unsqueeze(0)).to(torch.float32)


def _demo():
    images = torch.rand(4, 16)
    sp = poisson_encode(images, intensity=2.0, n_steps=100)
    assert sp.shape == (100, 4, 16)
    assert set(sp.unique().tolist()) <= {0.0, 1.0}
    print(f"ok: shape={tuple(sp.shape)}, mean_rate={sp.mean():.4f}")


if __name__ == "__main__":
    _demo()
"""Intrinsic plasticity: adaptive threshold (Phase 2 / Fitur 1B.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def adaptive_threshold_step(
    theta: Tensor, spikes: Tensor, beta: float, alpha: float, theta_0: float = 0.0
) -> Tensor:
    """θ += β·z; θ -= (θ - θ_0)·α. Return new theta."""
    theta = theta + beta * spikes
    theta = theta - (theta - theta_0) * alpha
    return theta


def _demo():
    theta = torch.zeros(4)
    alpha = 0.01  # slow
    beta = 0.5
    spikes = torch.ones(4)
    for _ in range(100):
        theta = adaptive_threshold_step(theta, spikes, beta, alpha)
    assert float(theta.mean()) > 0
    print(f"ok: theta={float(theta.mean()):.3f}")


if __name__ == "__main__":
    _demo()
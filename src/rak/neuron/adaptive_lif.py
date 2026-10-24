"""Adaptive LIF neuron (Phase 1.2).

Homeostasis: threshold naik setiap spike, decay per timestep.
Formula:
    θ += β_θ · z
    θ -= (θ - θ_0) · α_θ
    v_th_eff = v_th + θ
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rak.neuron.lif import LIF, LIFConfig


@dataclass
class AdaptiveLIFConfig(LIFConfig):
    beta_theta: float = 0.004
    theta_0: float = 0.0
    tau_theta_ms: float = 10000.0

    def __post_init__(self):
        super().__post_init__()
        if self.beta_theta < 0:
            raise ValueError("beta_theta harus >= 0")
        if self.tau_theta_ms <= 0:
            raise ValueError("tau_theta_ms harus > 0")


class AdaptiveLIF(LIF):
    """LIF dengan adaptive threshold (homeostasis)."""

    def __init__(self, n: int, config: AdaptiveLIFConfig | None = None, seed: int = 0):
        super().__init__(n, config or AdaptiveLIFConfig(), seed)
        self.theta = torch.zeros(n)

    def reset(self):
        super().reset()
        self.theta = torch.zeros(self.n)

    def step(self, I: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        alpha = cfg.dt_ms / cfg.tau_mem_ms
        alpha_theta = cfg.dt_ms / cfg.tau_theta_ms
        v_new = self.v + alpha * (cfg.v_rest - self.v + I)
        v_th_eff = cfg.v_th + self.theta
        spikes = (v_new >= v_th_eff).to(torch.float32)
        v_new = torch.where(spikes > 0, torch.full_like(v_new, cfg.v_reset), v_new)
        # Homeostasis
        self.theta = self.theta + cfg.beta_theta * spikes
        self.theta = self.theta - (self.theta - cfg.theta_0) * alpha_theta
        self.v = v_new
        return spikes, v_new

    def firing_rate(self, current: float, n_steps: int = 500) -> float:
        I = torch.full((n_steps, self.n), current)
        sp = self.simulate(I)
        return float(sp.sum() / (n_steps * self.n) * (1000.0 / self.config.dt_ms))


def _demo():
    lif = AdaptiveLIF(4)
    I = torch.full((500,), 15.0)
    sp, v = lif.simulate(I, record=True)
    # Firing rate harus menurun (adaptasi)
    r1 = float(sp[:100].sum()) / 100
    r2 = float(sp[-100:].sum()) / 100
    print(f"ok: early_rate={r1:.2f}, late_rate={r2:.2f}, adaptasi={r1 > r2}")


if __name__ == "__main__":
    _demo()
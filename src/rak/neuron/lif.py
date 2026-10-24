"""Leaky Integrate-and-Fire neuron (Phase 1).

Bentuk implisit timestep — stabil, tanpa ODE solver.
Formula: V[t] = V[t-1] + (dt/tau_m) * (v_rest - V[t-1] + I[t])
         z[t] = 1 jika V[t] >= v_th, else 0
         V[t] = v_reset jika z[t]=1
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LIFConfig:
    tau_mem_ms: float = 20.0
    v_rest: float = -65.0
    v_th: float = -52.0
    v_reset: float = -65.0
    dt_ms: float = 1.0

    def __post_init__(self):
        if self.tau_mem_ms <= 0:
            raise ValueError("tau_mem_ms harus > 0")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms harus > 0")
        if self.v_reset >= self.v_th:
            raise ValueError("v_reset harus < v_th")


class LIF:
    """LIF neuron: tensor-based, batch + time dims.

    State: V (membran potential). Reset: set ke v_reset saat spike.
    """

    def __init__(self, n: int, config: LIFConfig | None = None, seed: int = 0):
        self.n = n
        self.config = config or LIFConfig()
        self.v = torch.full((n,), self.config.v_rest)

    def reset(self):
        self.v = torch.full((self.n,), self.config.v_rest)

    def step(self, I: Tensor) -> tuple[Tensor, Tensor]:
        """Satu timestep. I: (batch, n) atau (n,). Return (spikes, V)."""
        cfg = self.config
        alpha = cfg.dt_ms / cfg.tau_mem_ms
        v_new = self.v + alpha * (cfg.v_rest - self.v + I)
        spikes = (v_new >= cfg.v_th).to(torch.float32)
        v_new = torch.where(spikes > 0, torch.full_like(v_new, cfg.v_reset), v_new)
        self.v = v_new
        return spikes, v_new

    def simulate(self, I: Tensor, record: bool = False):
        """I: (T,) atau (T, n) atau (T, batch, n). Return spikes [T, ...]."""
        if I.dim() == 1:
            I = I.unsqueeze(-1)
        T = I.shape[0]
        spikes_out = []
        v_out = [] if record else None
        self.reset()
        for t in range(T):
            sp, v = self.step(I[t])
            spikes_out.append(sp)
            if record:
                v_out.append(v.clone())
        sp_tensor = torch.stack(spikes_out)
        if record:
            return sp_tensor, torch.stack(v_out)
        return sp_tensor

    def firing_rate(self, current: float, n_steps: int = 500) -> float:
        I = torch.full((n_steps, self.n), current)
        sp = self.simulate(I)
        return float(sp.sum() / (n_steps * self.n) * (1000.0 / self.config.dt_ms))


def _demo():
    """Self-check: LIF harus fire pada input di atas threshold."""
    lif = LIF(4)
    I = torch.full((200,), 15.0)
    sp, v = lif.simulate(I, record=True)
    assert sp.sum() > 0, "LIF harus fire"
    print(f"ok: spikes={int(sp.sum())}, v_range=[{v.min():.1f}, {v.max():.1f}]")


if __name__ == "__main__":
    _demo()
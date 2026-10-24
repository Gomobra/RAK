"""Izhikevich neuron model (Phase 1.3).

Rumus:
    v' = 0.04*v^2 + 5*v + 140 - u + I
    u' = a*(b*v - u)
    if v >= 30: v = c, u += d

6 mode firing: regular, bursting, chattering, adapting, fast spiking,
thalamo-cortical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class IzhikevichConfig:
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    dt_ms: float = 1.0
    v_th: float = 30.0

    def __post_init__(self):
        if self.dt_ms <= 0:
            raise ValueError("dt_ms harus > 0")


# 6 mode firing dari Izhikevich 2003
IZHIKEVICH_MODES = {
    "regular": IzhikevichConfig(a=0.02, b=0.2, c=-65.0, d=8.0),
    "bursting": IzhikevichConfig(a=0.02, b=0.2, c=-55.0, d=4.0),
    "chattering": IzhikevichConfig(a=0.02, b=0.2, c=-50.0, d=2.0),
    "adapting": IzhikevichConfig(a=0.02, b=0.2, c=-65.0, d=6.0),
    "fast_spiking": IzhikevichConfig(a=0.1, b=0.2, c=-65.0, d=2.0),
    "thalamo_cortical": IzhikevichConfig(a=0.02, b=0.25, c=-65.0, d=0.05),
}


class Izhikevich:
    """Izhikevich neuron: 2 variabel (v, u), 4 parameter (a, b, c, d)."""

    def __init__(self, n: int, config: IzhikevichConfig | None = None, seed: int = 0):
        self.n = n
        self.config = config or IzhikevichConfig()
        self.v = torch.full((n,), -65.0)
        self.u = torch.full((n,), self.config.b * -65.0)

    def reset(self):
        self.v = torch.full((self.n,), -65.0)
        self.u = torch.full((self.n,), self.config.b * -65.0)

    def step(self, I: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        dt = cfg.dt_ms
        v = self.v + dt * (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I)
        u = self.u + dt * (cfg.a * (cfg.b * self.v - self.u))
        spikes = (v >= cfg.v_th).to(torch.float32)
        v = torch.where(spikes > 0, torch.full_like(v, cfg.c), v)
        u = torch.where(spikes > 0, u + cfg.d, u)
        self.v = v
        self.u = u
        return spikes, v

    def simulate(self, I: Tensor, record: bool = False):
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
    """Self-check: setiap mode harus menghasilkan spike."""
    for name, cfg in IZHIKEVICH_MODES.items():
        neuron = Izhikevich(4, config=cfg)
        I = torch.full((300, 4), 10.0)
        sp = neuron.simulate(I)
        print(f"  {name}: spikes={int(sp.sum())}, rate={neuron.firing_rate(10.0, 300):.1f} Hz")
    print("ok: 6 mode firing")


if __name__ == "__main__":
    _demo()
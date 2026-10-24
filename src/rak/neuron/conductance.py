"""Conductance-based LIF neuron (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor


@dataclass
class ConductanceConfig:
    tau_mem_ms: float = 100.0
    tau_ge_ms: float = 1.0
    tau_gi_ms: float = 2.0
    v_rest: float = -65.0
    e_syn_e: float = 0.0
    e_syn_i: float = -100.0
    v_th: float = -52.0
    v_reset: float = -65.0
    refrac_ms: float = 5.0
    spike_refrac_ms: float | None = None
    dt_ms: float = 0.5
    ge_jump: float = 1.0
    gi_jump: float = 1.0

    gamma_ge: float = field(init=False)
    gamma_gi: float = field(init=False)

    def __post_init__(self):
        for name in ("tau_mem_ms", "tau_ge_ms", "tau_gi_ms", "dt_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} harus > 0")
        if self.v_reset >= self.v_th:
            raise ValueError("v_reset harus < v_th")
        if self.refrac_ms < 0:
            raise ValueError("refrac_ms harus >= 0")
        if self.spike_refrac_ms is None:
            object.__setattr__(self, "spike_refrac_ms", self.refrac_ms)
        object.__setattr__(self, "gamma_ge", math.exp(-self.dt_ms / self.tau_ge_ms))
        object.__setattr__(self, "gamma_gi", math.exp(-self.dt_ms / self.tau_gi_ms))


@dataclass
class ConductanceState:
    v: Tensor
    ge: Tensor
    gi: Tensor
    timer: Tensor | None


class ConductanceLIF(torch.nn.Module):
    def __init__(self, n: int, strict_stability: bool = False,
                 config: ConductanceConfig | None = None):
        super().__init__()
        self.n = n
        self.config = config or ConductanceConfig()
        self.strict_stability = strict_stability
        self.v_th = self.config.v_th
        self.v_reset = self.config.v_reset

    def init_state(self, shape=None):
        if shape is None:
            shape = (self.n,)
        v = torch.full(shape, self.config.v_rest)
        ge = torch.zeros(shape)
        gi = torch.zeros(shape)
        timer = torch.zeros(shape)
        return ConductanceState(v=v, ge=ge, gi=gi, timer=timer)

    def step(self, ge_drive, gi_drive, state, v_th=None):
        cfg = self.config
        ge = cfg.gamma_ge * state.ge + cfg.ge_jump * ge_drive
        gi = cfg.gamma_gi * state.gi + cfg.gi_jump * gi_drive
        # implicit Euler
        g_total = 1.0 + ge + gi
        v_ss = (cfg.v_rest + cfg.e_syn_e * ge + cfg.e_syn_i * gi) / g_total
        beta = cfg.dt_ms * g_total / cfg.tau_mem_ms
        alpha = 1.0 - beta
        # clamp alpha for stability
        alpha = alpha.clamp(-1.0, 1.0)
        v_new = alpha * state.v + (1 - alpha) * v_ss

        # refractory
        timer = state.timer + cfg.dt_ms if state.timer is not None else None
        if cfg.refrac_ms > 0 and timer is not None:
            blocked = (timer < cfg.refrac_ms).to(torch.float32)
            v_new = torch.where(blocked > 0, torch.full_like(v_new, cfg.v_reset), v_new)

        threshold = self.v_th if v_th is None else v_th
        spikes = (v_new >= threshold).to(torch.float32)
        if timer is not None:
            spikes = torch.where(timer >= cfg.spike_refrac_ms, spikes, torch.zeros_like(spikes))

        v_new = torch.where(spikes > 0, torch.full_like(v_new, cfg.v_reset), v_new)
        new_timer = None
        if timer is not None:
            new_timer = torch.where(spikes > 0, torch.zeros_like(timer), timer)

        return spikes, ConductanceState(v=v_new, ge=ge, gi=gi, timer=new_timer)

    def to(self, device):
        return self
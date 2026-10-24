"""Spike trace & Poisson encoder (Phase 1.5)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def poisson_spike_train(rate_hz: float, duration_ms: float, dt_ms: float = 1.0,
                        generator: torch.Generator | None = None) -> Tensor:
    """Generate Poisson spike train. rate_hz: laju spike (Hz). Return (T,) float."""
    T = int(duration_ms / dt_ms)
    p = rate_hz * dt_ms / 1000.0
    if generator is None:
        generator = torch.Generator()
    u = torch.rand(T, generator=generator)
    return (u < p).to(torch.float32)


def make_trace(n: int, value: float = 0.0) -> Tensor:
    """Spike trace: x[t] = gamma * x[t-1] + z[t]."""
    return torch.full((n,), value)


def trace_step(trace: Tensor, spike: Tensor, gamma: float) -> Tensor:
    """Satu step trace decay + increment."""
    return gamma * trace + spike


def _demo():
    # Trace monoton turun setelah spike berhenti
    trace = make_trace(4, value=1.0)
    gamma = math.exp(-1.0 / 20.0)
    vals = [float(trace.mean())]
    for _ in range(20):
        trace = trace_step(trace, torch.zeros(4), gamma)
        vals.append(float(trace.mean()))
    assert vals[-1] < vals[0], "trace harus turun"
    print(f"ok: trace {vals[0]:.3f} -> {vals[-1]:.3f}")


if __name__ == "__main__":
    _demo()
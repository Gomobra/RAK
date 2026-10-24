"""STDP — Spike-Timing-Dependent Plasticity (Phase 2 / Fitur 2.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def make_trace(n: int, value: float = 0.0) -> Tensor:
    return torch.full((n,), value)


def trace_step(trace: Tensor, spike: Tensor, gamma: float) -> Tensor:
    return gamma * trace + spike


def stdp_step(
    w: Tensor,
    z_pre: Tensor,
    z_post: Tensor,
    trace_pre: Tensor,
    trace_post: Tensor,
    eta_plus: float = 1e-3,
    eta_minus: float = 1e-3,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
    w_min: float = 0.0,
    w_max: float = 1.0,
    dt_ms: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Satu timestep STDP (trace-based, additive).

    Urutan: decay trace → update W → bump trace.
    pre sebelum post → naik (LTP), post sebelum pre → turun (LTD).
    """
    gamma_pre = math.exp(-dt_ms / tau_pre)
    gamma_post = math.exp(-dt_ms / tau_post)

    trace_pre_d = gamma_pre * trace_pre
    trace_post_d = gamma_post * trace_post

    # LTP: pre trace * post spike
    dw = eta_plus * torch.outer(trace_pre_d, z_post)
    # LTD: post trace * pre spike
    dw = dw - eta_minus * torch.outer(z_pre, trace_post_d)

    w_new = (w + dw).clamp(w_min, w_max)

    trace_pre_new = trace_pre_d + z_pre
    trace_post_new = trace_post_d + z_post

    return w_new, trace_pre_new, trace_post_new


def stdp_window(dt_ms: float, tau_pre: float = 20.0, tau_post: float = 20.0,
                a_plus: float = 1.0, a_minus: float = 1.0,
                max_ms: float = 100.0) -> tuple[Tensor, Tensor]:
    """STDP window analitik untuk plot dan verifikasi."""
    half = max(1, int(round(max_ms / dt_ms)))
    dts = torch.linspace(-half * dt_ms, half * dt_ms, 2 * half + 1)
    dw = torch.where(
        dts > 0,
        a_plus * torch.exp(-dts / tau_pre),
        torch.where(dts < 0, -a_minus * torch.exp(dts / tau_post), torch.zeros_like(dts)),
    )
    return dts, dw


def _demo():
    w = torch.zeros(4, 3)
    tp = torch.zeros(4)
    tq = torch.zeros(3)
    # pre spike di t=0, post spike di t=1 → LTP
    zp = torch.tensor([1.0, 0, 0, 0])
    zq = torch.zeros(3)
    w, tp, tq = stdp_step(w, zp, zq, tp, tq, eta_plus=0.1, eta_minus=0.1)
    # pre trace naik
    assert float(tp[0]) > 0
    # post spike di t=1
    zq = torch.tensor([1.0, 0, 0])
    w2, tp2, tq2 = stdp_step(w, torch.zeros(4), zq, tp, tq, eta_plus=0.1, eta_minus=0.1)
    assert (w2[0, 0] > 0).item(), "LTP harus naik"
    print(f"ok: w[0,0]={w2[0,0]:.3f}")


if __name__ == "__main__":
    _demo()
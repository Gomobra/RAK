"""Synaptic scaling: hard column normalization + soft Oja (Phase 2 / Fitur 1B.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def hard_normalize(w: Tensor, target_sum: float = 78.0) -> Tensor:
    """Column normalization: sum over rows = target_sum. W: (n_in, n_out)."""
    col_sums = w.sum(dim=0).clamp_min(1e-9)
    return w * (target_sum / col_sums).unsqueeze(0)


def soft_scale(w: Tensor, power: float = 0.5, eps: float = 1e-9) -> Tensor:
    """Oja-style soft scaling per neuron (column)."""
    mean_abs = w.abs().mean(dim=0).clamp_min(eps)
    return w / (mean_abs ** power)


def _demo():
    w = torch.rand(10, 5) * 2
    w2 = hard_normalize(w, target_sum=78.0)
    assert torch.allclose(w2.sum(dim=0), torch.full((5,), 78.0), atol=1e-4)
    print(f"ok: col sums = {w2.sum(dim=0).tolist()}")


if __name__ == "__main__":
    _demo()
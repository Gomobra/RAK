"""Lateral inhibition (Phase 2 / Fitur 1B.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def build_inhibition(n: int, strength: float = 17.0, seed: int = 0) -> Tensor:
    """I->E all-to-all (i != j) with given strength per synapse."""
    g = torch.Generator().manual_seed(seed)
    w = torch.full((n, n), strength)
    w.fill_diagonal_(0.0)
    return w


def apply_inhibition(z_exc: Tensor, w_ie: Tensor) -> Tensor:
    """I_inh = W_inh @ z_exc. z_exc: (B, n_e) → (B, n_e)."""
    return z_exc @ w_ie


def _demo():
    w = build_inhibition(4, strength=17.0)
    assert (torch.diag(w) == 0).all()
    assert (w[~torch.eye(4, dtype=torch.bool)] == 17.0).all()
    print(f"ok: shape={tuple(w.shape)}, diag=0, offdiag=17.0")


if __name__ == "__main__":
    _demo()
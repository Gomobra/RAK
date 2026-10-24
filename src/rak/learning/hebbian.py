"""Hebbian learning rules (Phase 2 / Fitur 2.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def hebbian_step(w: Tensor, x_pre: Tensor, x_post: Tensor, eta: float) -> Tensor:
    """Δw = η · x_pre · x_post."""
    return w + eta * torch.outer(x_pre, x_post)


def bcm_step(w: Tensor, x_pre: Tensor, x_post: Tensor, eta: float, theta: Tensor) -> Tensor:
    """BCM: Δw = η · x_post · (x_post - θ) · x_pre."""
    return w + eta * torch.outer(x_pre, x_post * (x_post - theta))


def oja_step(w: Tensor, x_pre: Tensor, x_post: Tensor, eta: float, beta: float = 1.0) -> Tensor:
    """Oja: Δw = η · (x_pre·x_post - β·x_post²·w)."""
    return w + eta * (torch.outer(x_pre, x_post) - beta * x_post.pow(2) * w)


def _demo():
    w = torch.zeros(4, 3)
    x_pre = torch.ones(4)
    x_post = torch.ones(3)
    w2 = hebbian_step(w, x_pre, x_post, eta=0.1)
    assert (w2 > 0).all()
    print(f"ok: hebbian {w2.mean():.3f}")


if __name__ == "__main__":
    _demo()
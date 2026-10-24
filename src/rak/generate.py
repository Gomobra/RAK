"""Generative readout — generate & reconstruct images from SNN weights.

Tanpa backprop, tanpa decoder terpisah. Bobot X->E sudah berisi template
digit; rekonstruksi adalah kombinasi linear template dengan spike count.
"""

from __future__ import annotations

import torch
from torch import Tensor


def generate_template(w: Tensor, neuron_idx: int) -> Tensor:
    """Template digit dari neuron j: reshape bobot (784,) -> (28, 28)."""
    return w[:, neuron_idx].reshape(28, 28)


def generate_class(w: Tensor, assignments: Tensor, class_idx: int,
                   seed: int = 0) -> Tensor:
    """Pilih neuron yang assigned ke class_idx, return template-nya."""
    g = torch.Generator().manual_seed(seed)
    candidates = (assignments == class_idx).nonzero(as_tuple=True)[0]
    if candidates.numel() == 0:
        raise ValueError(f"tidak ada neuron untuk kelas {class_idx}")
    j = candidates[torch.randint(0, candidates.numel(), (1,), generator=g).item()]
    return generate_template(w, int(j))


def reconstruct(w: Tensor, counts: Tensor) -> Tensor:
    """Rekonstruksi linear: counts (B, n_e) @ w.T (n_e, 784) -> (B, 784)."""
    return (counts.to(torch.float64) @ w.T.to(torch.float64)).reshape(-1, 28, 28).to(torch.float32)


def generate_samples(w: Tensor, n: int = 10, seed: int = 0) -> Tensor:
    """Generate n gambar dari n neuron acak."""
    g = torch.Generator().manual_seed(seed)
    n_e = w.shape[1]
    idx = torch.randperm(n_e, generator=g)[:n]
    return torch.stack([generate_template(w, int(i)) for i in idx])  # (n, 28, 28)
"""SpikeLayer — banyak neuron (Phase 1.4).

Simulasi T timestep, return spike tensor [T, batch, N].
"""

from __future__ import annotations

import torch
from torch import Tensor

from rak.neuron.lif import LIF, LIFConfig


class SpikeLayer:
    """Layer neuron: simulasi T timestep.

    Args:
        n_in: jumlah input neuron
        n_out: jumlah output neuron
        seed: seed generator
    """

    def __init__(self, n_in: int, n_out: int, seed: int = 0, config: LIFConfig | None = None):
        self.n_in = n_in
        self.n_out = n_out
        self.neuron = LIF(n_out, config=config, seed=seed)
        # Bobot input→output: uniform [0, 1] scaled
        g = torch.Generator().manual_seed(seed)
        self.w = torch.rand(n_in, n_out, generator=g) * 0.1

    def reset(self):
        self.neuron.reset()

    def forward(self, x: Tensor, state=None):
        """x: (T, batch, n_in) → (spikes: (T, batch, n_out), V: (T, batch, n_out))."""
        T, batch, n_in = x.shape
        assert n_in == self.n_in
        spikes_out = []
        v_out = []
        self.reset()
        for t in range(T):
            I = x[t] @ self.w  # (batch, n_out)
            sp, v = self.neuron.step(I)
            spikes_out.append(sp)
            v_out.append(v.clone())
        return torch.stack(spikes_out), torch.stack(v_out)

    def encode(self, images: Tensor, n_steps: int = 100, max_rate_hz: float = 100.0, seed: int = 0) -> Tensor:
        """Encode gambar jadi spike train Poisson. images: (batch, n_in) → (n_steps, batch, n_in)."""
        g = torch.Generator().manual_seed(seed)
        probs = images * (max_rate_hz / 1000.0)
        u = torch.rand(n_steps, images.shape[0], images.shape[1], generator=g)
        return (u < probs.unsqueeze(0)).to(torch.float32)
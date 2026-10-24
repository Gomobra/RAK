"""Test SpikeLayer — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.neuron.layer import SpikeLayer


def test_spike_layer_shape():
    layer = SpikeLayer(n_in=16, n_out=8, seed=1)
    x = torch.rand(20, 4, 16).lt(0.3).float()
    y, v = layer.forward(x, None)
    assert y.shape == (20, 4, 8), f"shape salah: {tuple(y.shape)}"
    assert v.shape == (20, 4, 8)


def test_spike_layer_binary():
    layer = SpikeLayer(n_in=16, n_out=8, seed=1)
    x = torch.rand(20, 4, 16).lt(0.3).float()
    y, _ = layer.forward(x, None)
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_spike_layer_encode():
    layer = SpikeLayer(n_in=16, n_out=8, seed=1)
    images = torch.rand(4, 16)
    sp = layer.encode(images, n_steps=50)
    assert sp.shape == (50, 4, 16)
    assert set(sp.unique().tolist()) <= {0.0, 1.0}
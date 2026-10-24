"""Test Poisson encoder — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.neuron.encode import poisson_encode


def test_poisson_encode_shape():
    images = torch.rand(4, 16)
    sp = poisson_encode(images, intensity=2.0, n_steps=100)
    assert sp.shape == (100, 4, 16)


def test_poisson_encode_binary():
    images = torch.rand(4, 16)
    sp = poisson_encode(images, intensity=2.0, n_steps=100)
    assert set(sp.unique().tolist()) <= {0.0, 1.0}


def test_poisson_encode_rate_scales_with_intensity():
    images = torch.full((2, 16), 0.5)
    sp1 = poisson_encode(images, intensity=1.0, n_steps=1000, seed=1)
    sp2 = poisson_encode(images, intensity=2.0, n_steps=1000, seed=1)
    assert sp2.mean() > sp1.mean(), "rate harus naik dengan intensitas"


def test_poisson_encode_deterministic():
    images = torch.rand(4, 16)
    a = poisson_encode(images, 2.0, 100, seed=42)
    b = poisson_encode(images, 2.0, 100, seed=42)
    assert torch.equal(a, b)
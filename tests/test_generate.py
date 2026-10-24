"""Test generative readout — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.generate import generate_class, generate_samples, generate_template, reconstruct


def _weights(n_e=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(784, n_e, generator=g)


def test_generate_template_shape():
    w = _weights()
    img = generate_template(w, 0)
    assert img.shape == (28, 28)


def test_generate_samples_shape():
    w = _weights()
    imgs = generate_samples(w, n=5, seed=1)
    assert imgs.shape == (5, 28, 28)


def test_generate_class_returns_template():
    w = _weights()
    assignments = torch.tensor([0, 0, 1, 1, 2] + [0] * 15)
    img = generate_class(w, assignments, class_idx=1, seed=0)
    assert img.shape == (28, 28)
    # Should be one of the assigned neurons' templates
    candidates = (assignments == 1).nonzero(as_tuple=True)[0]
    assert any(torch.allclose(img, generate_template(w, int(c))) for c in candidates)


def test_reconstruct_shape():
    w = _weights(n_e=10)
    counts = torch.rand(4, 10)
    rec = reconstruct(w, counts)
    assert rec.shape == (4, 28, 28)


def test_reconstruct_zero_counts_gives_zero():
    w = _weights(n_e=10)
    counts = torch.zeros(2, 10)
    rec = reconstruct(w, counts)
    assert torch.allclose(rec, torch.zeros(2, 28, 28))


def test_reconstruct_linear():
    w = _weights(n_e=5)
    # Single neuron activation should reconstruct that neuron's template
    counts = torch.zeros(1, 5)
    counts[0, 2] = 1.0
    rec = reconstruct(w, counts)
    assert torch.allclose(rec[0], generate_template(w, 2))
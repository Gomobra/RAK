"""Test homeostasis — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.homeostasis.inhibition import apply_inhibition, build_inhibition
from rak.homeostasis.intrinsic import adaptive_threshold_step
from rak.homeostasis.scaling import hard_normalize, soft_scale


def test_hard_normalize_column_sum():
    w = torch.rand(10, 5) * 2
    w2 = hard_normalize(w, target_sum=78.0)
    assert torch.allclose(w2.sum(dim=0), torch.full((5,), 78.0), atol=1e-4)


def test_hard_normalize_shape():
    w = torch.rand(784, 400)
    w2 = hard_normalize(w, target_sum=78.0)
    assert w2.shape == w.shape


def test_soft_scale_reduces_large():
    w = torch.tensor([[10.0, 1.0], [10.0, 1.0]])
    w2 = soft_scale(w, power=1.0)
    assert w2.abs().mean(dim=0)[0] < w.abs().mean(dim=0)[0]


def test_adaptive_threshold_increases():
    theta = torch.zeros(4)
    spikes = torch.ones(4)
    new = adaptive_threshold_step(theta, spikes, beta=0.5, alpha=0.01)
    assert (new > theta).all()


def test_adaptive_threshold_decays():
    theta = torch.full((4,), 5.0)
    spikes = torch.zeros(4)
    new = adaptive_threshold_step(theta, spikes, beta=0.5, alpha=0.1)
    assert (new < theta).all()


def test_build_inhibition_diagonal_zero():
    w = build_inhibition(8, strength=17.0)
    assert (torch.diag(w) == 0).all()


def test_build_inhibition_offdiag_value():
    w = build_inhibition(8, strength=17.0)
    off = w[~torch.eye(8, dtype=torch.bool)]
    assert (off == 17.0).all()


def test_apply_inhibition_shape():
    z = torch.rand(4, 8)
    w = build_inhibition(8, strength=17.0)
    out = apply_inhibition(z, w)
    assert out.shape == (4, 8)
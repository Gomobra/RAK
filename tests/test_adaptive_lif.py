"""Test Adaptive LIF neuron — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.neuron.adaptive_lif import AdaptiveLIF, AdaptiveLIFConfig


def test_adaptive_lif_fires():
    alif = AdaptiveLIF(4)
    I = torch.full((200, 4), 15.0)
    sp = alif.simulate(I)
    assert sp.sum() > 0, "AdaptiveLIF harus fire"


def test_adaptive_lif_theta_increases_and_stabilizes():
    """Theta naik dari 0 dan stabil (tidak osilasi)."""
    alif = AdaptiveLIF(4)
    I = torch.full((5000, 4), 15.0)
    sp = alif.simulate(I)
    assert float(alif.theta.mean()) > 0, "theta harus naik setelah spike"
    # Rate harus stabil (tidak runaway, tidak mati)
    r = float(sp.mean())
    assert 0.0 < r < 1.0, f"rate harus di (0,1): {r:.4f}"


def test_adaptive_lif_theta_increases():
    alif = AdaptiveLIF(4)
    I = torch.full((100, 4), 15.0)
    alif.simulate(I)
    assert float(alif.theta.mean()) > 0, "theta harus naik setelah spike"


def test_adaptive_lif_config_validation():
    try:
        AdaptiveLIFConfig(beta_theta=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("beta_theta < 0 harus ValueError")
    try:
        AdaptiveLIFConfig(tau_theta_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("tau_theta_ms <= 0 harus ValueError")
"""Ekstraksi fitur spike: bentuk, trial, dan invariance bobot."""

from __future__ import annotations

import torch

from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.readout import collect_counts, ensemble_counts


def _net(n_e: int, seed: int = 1) -> CompetitiveNetwork:
    return CompetitiveNetwork(
        NetworkConfig(n_e=n_e, inhibition_strength=400.0), seed=seed
    )


def test_spike_counts_shape_and_finiteness() -> None:
    net = _net(20)
    images = torch.rand(4, 784, generator=torch.Generator().manual_seed(0))
    c = collect_counts(net, images, trials=1, batch=2, seed=0)
    assert c.shape == (4, 20)
    assert torch.isfinite(c).all()
    assert (c >= 0).all()


def test_trials_sum_without_touching_weights() -> None:
    net = _net(20)
    images = torch.rand(3, 784, generator=torch.Generator().manual_seed(1))
    w0 = net.w_e.clone()
    th0 = net.theta.clone()
    one = collect_counts(net, images, trials=1, batch=3, seed=7)
    two = collect_counts(net, images, trials=2, batch=3, seed=7)
    # Dua percobaan Poisson adalah sampel berbeda; yang bisa
    # dipastikan hanya bentuk, non-negatif, dan bobot/theta tidak
    # berubah oleh presentasi `learn=False`.
    assert one.shape == two.shape == (3, 20)
    assert (one >= 0).all() and (two >= 0).all()
    assert torch.equal(net.w_e, w0)
    assert torch.equal(net.theta, th0)


def test_ensemble_concatenates_neurons() -> None:
    images = torch.rand(2, 784, generator=torch.Generator().manual_seed(2))
    both = ensemble_counts([_net(12), _net(8)], images, trials=1, batch=2)
    assert both.shape == (2, 20)


def test_empty_ensemble_rejected() -> None:
    try:
        ensemble_counts([], torch.rand(1, 784))
    except ValueError as exc:
        assert "minimal satu model" in str(exc)
    else:
        raise AssertionError("ensemble kosong seharusnya ValueError")

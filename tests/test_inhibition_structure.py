"""Struktur inhibisi I->E dan batas theta.

Dua rezim yang didukung (lihat `use_reference_params`):

- reference (default): persis acuan Diehl & Cook 2015 — E->I one-to-one
  10.4, I->E all-to-all 17.0 (i != j). Simetri dipecah oleh noise
  weight init + STDP + delay, bukan oleh konektivitas inhibisi.
- legacy: inhibisi sparse ternormalisasi (rak lama).

Test di bawah memeriksa invarian yang berlaku untuk rezim yang aktif.
"""

from __future__ import annotations

import torch

from rak.network.competitive import CompetitiveNetwork, NetworkConfig


def _net(n_e: int = 40, strength: float = 800.0, **kw) -> CompetitiveNetwork:
    return CompetitiveNetwork(
        NetworkConfig(
            n_e=n_e, inhibition_strength=strength, rule="variant3", **kw
        ),
        seed=3,
    )


def test_e_to_i_is_one_to_one() -> None:
    """E -> I satu-satu (diagonal) di kedua rezim."""
    net = _net(n_e=16)
    diag = torch.diag(net.w_ei)
    off = net.w_ei - torch.diag(diag)
    assert torch.count_nonzero(off) == 0
    if net.config.use_reference_params:
        assert torch.allclose(diag, torch.full((16,), 10.4))
    else:
        assert torch.allclose(
            diag, torch.full((16,), float(net.config.e_to_i_strength))
        )


def test_i_to_e_reference_is_all_to_all_without_self() -> None:
    """Rezim reference: I->E all-to-all, i != j, bobot 17.0."""
    net = _net(n_e=16)  # default reference; strength diabaikan
    assert net.config.use_reference_params
    assert torch.count_nonzero(torch.diag(net.w_ie)) == 0
    off_diag = net.w_ie[~torch.eye(16, dtype=torch.bool)]
    assert torch.allclose(
        off_diag, torch.full_like(off_diag, float(net.config.inhibition_strength))
    )


def test_i_to_e_legacy_columns_sum_to_strength() -> None:
    """Rezim legacy: total inhibisi per neuron = strength."""
    net = _net(n_e=40, strength=800.0, use_reference_params=False)
    col = net.w_ie.sum(dim=0)
    assert torch.allclose(col, torch.full_like(col, 800.0), atol=1e-4)


def test_i_to_e_legacy_is_sparse() -> None:
    """Rezim legacy: konektivitas acak, ada nol, tidak ada kolom kosong."""
    net = _net(
        n_e=60, strength=800.0, use_reference_params=False,
        inhib_connectivity=0.5,
    )
    frac_zero = float((net.w_ie == 0).double().mean())
    assert 0.2 < frac_zero < 0.8
    assert bool((net.w_ie.sum(dim=0) > 0).all())


def test_i_to_e_legacy_differs_between_neurons() -> None:
    """Rezim legacy: pola inhibisi berbeda antar neuron."""
    net = _net(n_e=60, strength=800.0, use_reference_params=False)
    cols = net.w_ie.T
    norm = cols / cols.norm(dim=1, keepdim=True).clamp_min(1e-9)
    gram = norm @ norm.T
    off = gram[~torch.eye(60, dtype=torch.bool)]
    assert float(off.max()) < 0.99


def test_theta_never_exceeds_theta_max() -> None:
    """Theta dibatasi, apa pun laju fire-nya."""
    cfg = NetworkConfig(n_e=20, rule="variant3")
    net = CompetitiveNetwork(cfg, seed=1)
    enc_spikes = torch.rand(
        700, 4, 784, generator=torch.Generator().manual_seed(0)
    )
    enc_spikes = (enc_spikes < 0.2).to(torch.float32)

    for _ in range(3):
        net.present_batch(enc_spikes)

    assert float(net.theta.max()) <= cfg.theta_max + 1e-4
    assert bool(torch.isfinite(net.theta).all())

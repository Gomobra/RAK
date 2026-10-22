"""Struktur inhibisi I->E dan batas theta.

Dua properti yang harus dijamin kode, bukan diasumsikan:

1. Setiap neuron eksitasi menerima TOTAL inhibisi yang sama
   (`inhibition_strength`), terlepas dari ukuran jaringan dan terlepas
   dari berapa neuron inhibitor yang menyala. Kalau properti ini
   rusak, laju fire per neuron ikut berubah saat `n_e` diubah -- dan
   itu yang membuat penskalaan 100 -> 400 neuron gagal (0.103 ->
   0.22 spike/neuron).

2. Konektivitas I->E harus ACAK, bukan seragam. Inhibisi seragam
   membuat kompetisi simetris sempurna: semua neuron eksitasi
   menerima sinyal inhibisi yang identik, sehingga konvergen ke
   receptive field yang sama (cosine 0.637 terukur).

3. `theta` tidak boleh melewati `theta_max`.
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


def test_i_to_e_columns_sum_to_inhibition_strength() -> None:
    """Total inhibisi per neuron eksitasi = `inhibition_strength`."""
    for n_e, strength in ((20, 400.0), (40, 800.0), (80, 1600.0)):
        net = _net(n_e=n_e, strength=strength)
        col = net.w_ie.sum(dim=0)
        assert torch.allclose(col, torch.full_like(col, strength), atol=1e-4), (
            f"n_e={n_e} kolom I->E tidak berjumlah {strength}: "
            f"min {float(col.min()):.3f} max {float(col.max()):.3f}"
        )


def test_i_to_e_total_is_constant_across_network_size() -> None:
    """Properti penskalaan: total inhibisi tidak bergantung `n_e`.

    Inilah yang gagal di run 400 neuron: laju fire per neuron naik
    dua kali lipat saat n_e naik 4x.
    """
    a = _net(n_e=20, strength=800.0).w_ie.sum(dim=0).mean()
    b = _net(n_e=80, strength=800.0).w_ie.sum(dim=0).mean()
    assert torch.allclose(a, b, atol=1e-4)


def test_i_to_e_is_sparse_and_has_no_empty_columns() -> None:
    """Konektivitas acak: ada nol, tapi tidak ada kolom kosong."""
    net = _net(n_e=60, strength=800.0, inhib_connectivity=0.5)
    frac_zero = float((net.w_ie == 0).double().mean())
    assert 0.2 < frac_zero < 0.8, f"sparsitas tidak wajar: {frac_zero:.3f}"
    assert bool((net.w_ie.sum(dim=0) > 0).all())


def test_i_to_e_differs_between_excitatory_neurons() -> None:
    """Setiap neuron eksitasi punya pola inhibisi yang BERBEDA.

    Inhibisi seragam = simetri sempurna = semua neuron belajar hal
    yang sama.
    """
    net = _net(n_e=60, strength=800.0)
    # Bandingkan kolom: dua kolom identik berarti neuron itu menerima
    # inhibisi dari sumber yang sama persis.
    cols = net.w_ie.T
    norm = cols / cols.norm(dim=1, keepdim=True).clamp_min(1e-9)
    gram = norm @ norm.T
    off = gram[~torch.eye(60, dtype=torch.bool)]
    assert float(off.max()) < 0.99, (
        f"ada dua neuron dengan pola inhibisi identik "
        f"(cosine {float(off.max()):.3f})"
    )


def test_theta_never_exceeds_theta_max() -> None:
    """Theta dibatasi, apa pun laju fire-nya."""
    cfg = NetworkConfig(n_e=20, inhibition_strength=400.0, rule="variant3")
    net = CompetitiveNetwork(cfg, seed=1)
    enc_spikes = torch.rand(700, 4, 784, generator=torch.Generator().manual_seed(0))
    enc_spikes = (enc_spikes < 0.2).to(torch.float32)

    for _ in range(3):
        net.present_batch(enc_spikes)

    assert float(net.theta.max()) <= cfg.theta_max + 1e-4
    assert bool(torch.isfinite(net.theta).all())

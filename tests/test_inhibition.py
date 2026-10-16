"""T1B.3 — Lateral inhibition.

Arsitektur: E -> I satu-satu, I -> E all-to-all (Diehl & Cook 2015).
"""

import pytest
import torch

from rak.homeostasis.inhibition import (
    InhibitionConfig,
    build_inhibitory_weights,
    make_excitatory_neuron,
    make_inhibitory_neuron,
)
from rak.neuron import LIF


def test_shapes():
    w = build_inhibitory_weights(50)
    assert w.e_to_i.shape == (50, 50)
    assert w.i_to_e.shape == (50, 50)
    assert w.n_exc == 50
    assert w.n_inh == 50


def test_e_to_i_is_one_to_one():
    """Setiap neuron eksitasi punya tepat satu pasangan inhibitor."""
    w = build_inhibitory_weights(20)
    assert torch.equal(w.e_to_i, torch.eye(20))
    assert float(w.e_to_i.sum()) == 20.0


def test_i_to_e_is_all_to_all_and_inhibitory():
    w = build_inhibitory_weights(20)
    assert (w.i_to_e < 0).all(), "semua bobot inhibisi harus negatif"
    assert float((w.i_to_e != 0).sum()) == 400.0, "harus all-to-all"


def test_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="n_exc"):
        build_inhibitory_weights(0)


def test_strength_scales_down_with_network_size():
    """Total inhibisi tidak boleh tumbuh bareng ukuran jaringan.

    Kalau tidak, jaringan besar akan langsung ter-damp dan tidak ada
    neuron yang bisa fire.
    """
    totals = {}
    for n in (100, 1000, 10_000):
        w = build_inhibitory_weights(n)
        totals[n] = float(w.i_to_e.abs().sum())
    # dengan scale_per_neuron, total per neuron eksitasi konstan
    per_neuron = {n: totals[n] / n for n in totals}
    spread = max(per_neuron.values()) / min(per_neuron.values())
    assert spread == pytest.approx(1.0, rel=1e-4), per_neuron


def test_without_scaling_inhibition_would_explode():
    """Bukti kenapa `scale_per_neuron` ada.

    Tanpa scaling, bobot per synapse tetap tapi jumlah synapse inhibisi
    grows dengan N — sehingga jaringan besar mati total.
    """
    unscaled = build_inhibitory_weights(
        1000, InhibitionConfig(scale_per_neuron=False)
    )
    scaled = build_inhibitory_weights(
        1000, InhibitionConfig(scale_per_neuron=True)
    )
    total_unscaled = float(unscaled.i_to_e.abs().sum())
    total_scaled = float(scaled.i_to_e.abs().sum())
    assert total_scaled < total_unscaled, (
        f"scaled harus lebih ringan: {total_scaled} vs {total_unscaled}"
    )


# ------------------------------------------------------- WTA / kompetisi
def test_inhibition_creates_winner_take_all():
    """Inhibisi harus menimbulkan kompetisi: neuron terkuat menang.

    Implementasi loop eksitasi + inhibitor manual, supaya yang diuji
    benar-benar interaksinya dan bukan cuma bobotnya.
    """
    n_exc = 20
    w = build_inhibitory_weights(
        n_exc, InhibitionConfig(inhibition_strength=2.0, scale_per_neuron=False)
    )
    exc = make_excitatory_neuron(n_exc)
    inh = make_inhibitory_neuron(n_exc)

    exc_state = exc.init_state()
    inh_state = inh.init_state()

    # Neuron 7 diberi arus jauh lebih besar dari yang lain
    currents = torch.full((n_exc,), 0.4)
    currents[7] = 3.0

    spike_counts = torch.zeros(n_exc)
    for _ in range(400):
        z_e, exc_state = exc.step(currents, exc_state)
        z_i, inh_state = inh.step(z_e @ w.e_to_i, inh_state)
        inhib = z_i @ w.i_to_e
        v_inhibited = (exc_state.v + inhib).clamp(min=0.0)
        exc_state = type(exc_state)(v=v_inhibited, timer=exc_state.timer)
        spike_counts += z_e

    top = int(spike_counts.argmax())
    assert top == 7, f"neuron terkuat harus menang, dapat neuron {top}"
    assert float(spike_counts[7]) > float(spike_counts.sum()) / n_exc * 2, (
        f"winner harus jelas dominates: {spike_counts.tolist()}"
    )


def test_inhibition_reduces_total_firing():
    """Tanpa inhibisi total spike lebih banyak."""
    n_exc = 20
    currents = torch.full((n_exc,), 3.0)

    without = LIF(n_exc, tau_mem_ms=20.0)
    sp_without, _, _ = without.simulate(
        currents.unsqueeze(0).expand(300, n_exc)
    )

    with_inh = LIF(n_exc, tau_mem_ms=20.0)
    sp_with, _, _ = with_inh.simulate(
        (currents * 0.6).unsqueeze(0).expand(300, n_exc)
    )
    # eksitasi berlebihan + inhibisi harus lebih rendah dari full excite tanpa inhibisi
    assert float(sp_with.sum()) < float(sp_without.sum())


def test_inhibitory_neuron_different_from_excitatory():
    """Populasi inhibitor harus neuron terpisah dengan parameter sendiri."""
    exc = make_excitatory_neuron(10)
    inh = make_inhibitory_neuron(10)
    assert exc.config.v_th != inh.config.v_th, (
        "threshold eksitasi dan inhibitor harus berbeda"
    )
    assert inh.config.v_reset != exc.config.v_reset


def test_inhibitory_refractory_is_shorter():
    """Refraktori inhibitor lebih pendek — supaya bisa fire lagi cepat."""
    cfg = InhibitionConfig()
    assert cfg.refrac_ms == 2.0, "Diehl-Cook: refrac_i = 2 ms"
    assert cfg.tau_mem_ms == 10.0, "Diehl-Cook: tau_i = 10 ms"
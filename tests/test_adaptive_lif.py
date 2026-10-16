"""T1.2 — Adaptive LIF tests."""

import pytest
import torch

from rak.neuron.adaptive_lif import AdaptiveLIF, AdaptiveLIFConfig
from rak.neuron.lif import LIF


def test_adaptive_state_has_theta():
    lif = AdaptiveLIF(8)
    state = lif.init_state()
    assert state.theta.shape == (8,)
    assert float(state.theta.abs().sum()) == 0.0


def test_adaptive_config_inherits_lif_validation():
    with pytest.raises(ValueError, match="v_reset"):
        AdaptiveLIFConfig(v_reset=2.0, v_th=1.0)


# ---------------------------------------------------------------- T1.2.2
def test_theta_increases_on_spike():
    lif = AdaptiveLIF(1, beta_theta=0.1)
    state = lif.init_state()
    state.v.fill_(1.5)  # pastikan fire
    spikes, state = lif.step(torch.tensor([10.0]), state)
    assert float(spikes[0]) == 1.0
    assert float(state.theta[0]) > 0.0


def test_theta_decays_without_spike():
    lif = AdaptiveLIF(1, tau_adapt_ms=100.0, beta_theta=0.1)
    state = lif.init_state()
    state.theta.fill_(1.0)
    _, state = lif.step(torch.tensor([0.0]), state)
    assert float(state.theta[0]) < 1.0


# ---------------------------------------------------------------- T1.2.3
def test_rate_declines_under_sustained_input():
    """T1.2.3 — firing rate menurun saat input konstan.

    Efeknya: theta naik tiap spike, jadi v_eff naik, jadi neuron butuh
    lebih banyak input untuk fire lagi.
    """
    alif = AdaptiveLIF(
        1, tau_adapt_ms=2_000.0, beta_theta=0.05, tau_mem_ms=20.0
    )
    n = 4_000
    spikes, _, _ = alif.simulate(torch.full((n,), 2.0))

    early = float(spikes[:1000].sum())
    late = float(spikes[-1000:].sum())
    assert late < early, f"rate tidak menurun: awal={early} akhir={late}"


def test_longer_detector_gives_stronger_suppression():
    """tau_adapt panjang = homeostasis lebih persisten = rate lebih rendah.

    Ini inti aturan Zenke 2013: tau_adapt adalah "seberapa lama
    neuron ingat bahwa dia terlalu aktif".
    """
    rates = {}
    for tau_adapt in (500.0, 5_000.0):
        alif = AdaptiveLIF(
            1,
            tau_adapt_ms=tau_adapt,
            beta_theta=0.05,
            tau_mem_ms=20.0,
        )
        spikes, _, _ = alif.simulate(torch.full((4_000,), 2.0))
        rates[tau_adapt] = float(spikes[-1_000:].sum())
    assert rates[5_000.0] < rates[500.0], rates


def test_adaptive_rate_lower_than_plain_lif():
    """T1.2.4 — bandingkan dengan LIF biasa."""
    alif = AdaptiveLIF(
        1, tau_adapt_ms=2_000.0, beta_theta=0.05, tau_mem_ms=20.0
    )
    plain = LIF(1, tau_mem_ms=20.0)
    cur = 2.0
    assert alif.firing_rate(cur, n_steps=4_000) < plain.firing_rate(
        cur, n_steps=4_000
    )


def test_adaptive_recovers_after_silence():
    """Setelah input berhenti, neuron harus pulih dan bisa fire lagi."""
    alif = AdaptiveLIF(
        1, tau_adapt_ms=200.0, beta_theta=0.1, tau_mem_ms=20.0
    )
    cur = torch.cat([torch.full((1_000,), 2.0), torch.full((2_000,), 0.0)])
    spikes, _, _ = alif.simulate(cur)
    assert float(spikes[-1_000:].sum()) == 0.0
    assert float(spikes[:1_000].sum()) > 0.0


def test_theta_decay_matches_analytic():
    """Decay harus mempertahankan fraksi alpha_theta per timestep."""
    cfg_tau = 500.0
    alif = AdaptiveLIF(1, tau_adapt_ms=cfg_tau)
    state = alif.init_state()
    state.theta.fill_(1.0)
    _, state = alif.step(torch.tensor([0.0]), state)
    expected = alif.config.alpha_theta
    assert float(state.theta[0]) == pytest.approx(expected, abs=1e-6)


def test_simulate_records_theta():
    alif = AdaptiveLIF(4)
    cur = torch.full((100,), 2.0)
    cur = cur.unsqueeze(-1).expand(100, 4)
    spikes, _, (v_rec, theta_rec) = alif.simulate(cur, record=True)
    assert v_rec.shape == (100, 4)
    assert theta_rec.shape == (100, 4)
    assert float(theta_rec[-1].sum()) > 0.0
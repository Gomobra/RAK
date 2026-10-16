"""T1.1 — LIF neuron tests."""

import pytest
import torch

from rak.neuron.lif import LIF, LIFConfig


# ---------------------------------------------------------------- T1.1.1
def test_lif_runs_and_shapes():
    lif = LIF(4)
    state = lif.init_state()
    spikes, state = lif.step(torch.tensor([1.0, 0.5, 2.0, 0.0]), state)
    assert spikes.shape == (4,)
    assert state.v.shape == (4,)
    assert spikes.dtype == torch.float32


def test_lif_config_rejects_bad_reset():
    with pytest.raises(ValueError, match="v_reset"):
        LIFConfig(v_reset=1.0, v_th=1.0)


def test_lif_config_rejects_bad_tau():
    with pytest.raises(ValueError, match="tau_mem_ms"):
        LIFConfig(tau_mem_ms=0.0)


# ---------------------------------------------------------------- T1.1.2
def test_constant_input_gives_stable_rate():
    """Input konstan -> firing rate stabil (t-invariant)."""
    lif = LIF(1)
    n = 2000
    cur = torch.full((n,), 1.5)
    spikes, _, _ = lif.simulate(cur)

    # fire rate harus stabil setelah fase transien
    rates = spikes[200:].reshape(-1, 200).sum(dim=1)
    spread = float(rates.max() - rates.min())
    assert spread <= 1, f"rate tidak stabil, spread={spread}"


def test_subthreshold_input_never_spikes():
    lif = LIF(1, v_th=1.0, tau_mem_ms=10.0)
    spikes, _, _ = lif.simulate(torch.full((500,), 0.05))
    assert float(spikes.sum()) == 0.0


def test_suprathreshold_input_keeps_firing():
    lif = LIF(1, tau_mem_ms=10.0)
    spikes, _, _ = lif.simulate(torch.full((500,), 5.0))
    assert float(spikes.sum()) > 50.0


# ---------------------------------------------------------------- T1.1.3
def test_varying_input_changes_spike_pattern():
    lif = LIF(1)
    n = 1000
    cur = torch.cat([torch.zeros(n // 2), torch.full((n // 2,), 3.0)])
    spikes, _, _ = lif.simulate(cur)
    first_half = float(spikes[: n // 2].sum())
    second_half = float(spikes[n // 2 :].sum())
    assert first_half == 0.0
    # tau=20ms, beta=0.05: dari v=0 ke v_th=1 butuh ~14 timestep,
    # jadi ~35 spike di 500 timestep. Ambil margin longgar.
    assert second_half > 20.0


def test_higher_input_gives_higher_rate():
    lif = LIF(1)
    low = lif.firing_rate(0.8, n_steps=1000)
    high = lif.firing_rate(3.0, n_steps=1000)
    assert high > low


def test_spikes_are_binary():
    lif = LIF(8)
    cur = torch.randn(100, 8) * 2
    spikes, _, _ = lif.simulate(cur)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------- T1.1.4
def test_recorded_potential_tracks_input():
    lif = LIF(1, v_leak=0.0, tau_mem_ms=100.0)
    cur = torch.full((200,), 0.5)
    spikes, _, v_rec = lif.simulate(cur, record=True)
    assert v_rec.shape == (200, 1)
    # v harus naik monoton menuju steady state
    v = v_rec[:, 0]
    assert torch.all(v[1:] >= v[:-1] - 1e-6)


# ---------------------------------------------------------------- T1.1.5
def test_smaller_tau_gives_higher_rate():
    """tau lebih kecil = neuron lebih responsif = rate lebih tinggi."""
    slow = LIF(1, tau_mem_ms=100.0)
    fast = LIF(1, tau_mem_ms=10.0)
    cur = 2.0
    assert fast.firing_rate(cur) > slow.firing_rate(cur)


def test_higher_threshold_gives_lower_rate():
    low_th = LIF(1, v_th=0.5)
    high_th = LIF(1, v_th=2.0)
    assert high_th.firing_rate(2.0) < low_th.firing_rate(2.0)


def test_reset_value_affects_rate():
    """v_reset tinggi -> neuron lebih mudah fire lagi (ISI lebih pendek).

    Ini counterintuitive tapi benar: kalau v_reset = 0.9 dan
    v_th = 1.0, neuron hanya butuh sedikit input untuk mencapai
    threshold lagi. v_reset = 0 (reset penuh) justru memperlambat.
    """
    deep_reset = LIF(1, v_reset=0.0)
    shallow_reset = LIF(1, v_reset=0.9)
    assert shallow_reset.firing_rate(3.0) > deep_reset.firing_rate(3.0)


def test_numerical_stability_large_tau_small_dt():
    """tau >> dt harus tetap stabil (alasan bentuk implisit)."""
    lif = LIF(1, tau_mem_ms=10_000.0, dt_ms=1.0)
    spikes, _, v_rec = lif.simulate(
        torch.full((500,), 100.0), record=True
    )
    v = v_rec[:, 0]
    assert torch.isfinite(v).all()
    assert float(v.max()) < 1e6, f"membran meledak: {float(v.max())}"
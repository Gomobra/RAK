"""Refractory period tests — item 0 Phase 2.

Refractory period (Diehl & Cook 2015 pakai 5 ms untuk neuron eksitasi)
tidak ada di Phase 1 dan lupa diimplementasikan. Tanpa itu, neuron bisa
fire beruntun di beberapa timestep berturut-turut dan STDP kehilangan
makna: "kapan" spike terjadi jadi tidak terdefinisi.

Bug yang ketahuan saat menulis test di sini:
masking aritmetika `a*x + b*y` dengan `y = inf` menghasilkan `0*inf = nan`.
Akibatnya neuron diam SELAMANYA setelah spike pertama. Semua masking
harus pakai `torch.where`.
"""

import pytest
import torch

from rak.neuron import AdaptiveLIF, LIF, SpikeLayer
from rak.neuron.lif import LIFConfig


def _isi(spikes: torch.Tensor) -> list[int]:
    idx = spikes.nonzero().flatten().tolist()
    return [b - a for a, b in zip(idx, idx[1:])]


# ------------------------------------------------------------- config
def test_refrac_rejects_negative():
    with pytest.raises(ValueError, match="refrac_ms"):
        LIFConfig(refrac_ms=-1.0)


def test_refrac_defaults_to_off():
    """Default 0 = refractory mati, supaya test Phase 1 tidak berubah."""
    assert LIFConfig().refrac_ms == 0.0


# -------------------------------------------------------------- neuron
@pytest.mark.parametrize("cls", [LIF, AdaptiveLIF])
def test_timer_starts_infinite(cls):
    """Timer awal harus `inf`: neuron belum pernah spike, jadi boleh fire."""
    n = cls(1, refrac_ms=5.0)
    state = n.init_state()
    assert state.timer is not None
    assert torch.isinf(state.timer).all()


@pytest.mark.parametrize("cls", [LIF, AdaptiveLIF])
def test_no_nan_after_first_spike(cls):
    """Regression: 0 * inf = nan bikin neuron diam selamanya."""
    n = cls(1, refrac_ms=5.0, tau_mem_ms=20.0)
    spikes, state, _ = n.simulate(torch.full((300,), 1.5))
    assert not torch.isnan(state.v).any()
    assert not torch.isnan(state.timer).any()
    assert float(spikes.sum()) > 1.0, "neuron harus spike lebih dari sekali"


@pytest.mark.parametrize("cls", [LIF, AdaptiveLIF])
def test_refractory_enforces_minimum_isi(cls):
    """ISI tidak boleh lebih kecil dari refrac_ms."""
    n = cls(1, refrac_ms=8.0, tau_mem_ms=20.0, dt_ms=1.0)
    spikes, _, _ = n.simulate(torch.full((500,), 1.5))
    isi = _isi(spikes)
    assert len(isi) >= 3, "butuh beberapa spike untuk mengecek ISI"
    assert min(isi) >= 8, f"ISI {min(isi)} < refrac 8 ms"


@pytest.mark.parametrize("cls", [LIF, AdaptiveLIF])
def test_refractory_reduces_firing_rate(cls):
    """Refractory harus menurunkan rate, bukan menaikkan."""
    rates = {}
    for r in (0.0, 5.0):
        n = cls(1, refrac_ms=r, tau_mem_ms=20.0)
        sp, _, _ = n.simulate(torch.full((1000,), 10.0))
        rates[r] = float(sp.sum())
    assert rates[5.0] < rates[0.0], rates


@pytest.mark.parametrize("cls", [LIF, AdaptiveLIF])
def test_refractory_blocks_immediate_respike(cls):
    """Tepat setelah spike, neuron tidak boleh fire lagi di timestep berikutnya."""
    n = cls(1, refrac_ms=10.0, tau_mem_ms=20.0)
    state = n.init_state()
    cur = torch.tensor([100.0])  # sangat kuat, pasti mau fire tiap timestep
    spikes = []
    for _ in range(6):
        spk, state = n.step(cur, state)
        spikes.append(float(spk[0]))
    assert spikes[0] == 1.0
    assert spikes[1] == 0.0, "tidak boleh fire di timestep setelah spike"
    assert spikes[2] == 0.0
    assert sum(spikes) == 1.0, "selama refrac aktif, hanya boleh 1 spike"


@pytest.mark.parametrize("cls", [LIF, AdaptiveLIF])
def test_refractory_recovers_after_period(cls):
    """Setelah periode refrac habis, neuron harus bisa fire lagi."""
    n = cls(1, refrac_ms=5.0, tau_mem_ms=20.0)
    state = n.init_state()
    cur = torch.tensor([100.0])
    fired_again = False
    for t in range(60):
        spk, state = n.step(cur, state)
        if t > 6 and float(spk[0]) == 1.0:
            fired_again = True
            break
    assert fired_again, "neuron tidak pernah fire lagi setelah refrac"


def test_timer_is_per_neuron():
    """Timer harus independen per neuron, bukan scalar bersama."""
    n = LIF(4, refrac_ms=5.0)
    state = n.init_state()
    # refrac = 5 ms, jadi timer 2.0 masih DI DALAM periode refrac
    # (setelah +1 jadi 3.0 masih < 5), sedangkan 100.0 sudah lewat.
    state.timer = torch.tensor([0.0, float("inf"), 2.0, float("inf")])
    state.v.fill_(99.0)
    spikes, state = n.step(torch.full((4,), 100.0), state)
    # neuron 1 dan 3 belum pernah spike -> boleh fire
    assert float(spikes[1]) == 1.0 and float(spikes[3]) == 1.0
    # neuron 0 baru spike (timer 0 -> 1) dan neuron 2 masih dalam refrac
    assert float(spikes[0]) == 0.0 and float(spikes[2]) == 0.0


def test_timer_past_refrac_is_allowed():
    """Timer jauh melewati refrac harusnya neuron BOLEH fire."""
    n = LIF(1, refrac_ms=5.0)
    state = n.init_state()
    state.timer = torch.tensor([100.0])
    state.v.fill_(99.0)
    spikes, state = n.step(torch.tensor([100.0]), state)
    assert float(spikes[0]) == 1.0


# --------------------------------------------------------------- layer
def test_layer_refractory_state_present():
    layer = SpikeLayer(n_in=10, n_out=5, refrac_ms=5.0, seed=1)
    state = layer.init_state(3)
    assert "timer" in state
    assert torch.isinf(state["timer"]).all()


def test_layer_without_refractory_has_no_timer():
    layer = SpikeLayer(n_in=10, n_out=5, refrac_ms=0.0, seed=1)
    assert "timer" not in layer.init_state(3)


def test_layer_refractory_lowers_density():
    densities = {}
    for r in (0.0, 5.0):
        layer = SpikeLayer(n_in=784, n_out=100, seed=1, refrac_ms=r)
        x = layer.encode(torch.full((1, 784), 0.3), n_steps=350, seed=1)
        with torch.no_grad():
            y, _, _ = layer.forward(x, None)
        densities[r] = float(y.mean())
    assert densities[5.0] < densities[0.0], densities


def test_layer_refractory_no_nan():
    layer = SpikeLayer(n_in=64, n_out=20, seed=1, refrac_ms=5.0)
    x = layer.encode(torch.full((1, 64), 0.5), n_steps=200, seed=1)
    with torch.no_grad():
        _, state, _ = layer.forward(x, None)
    assert torch.isfinite(state["timer"]).all()
    assert torch.isfinite(state["v"]).all()


def test_layer_reset_restores_infinite_timer():
    layer = SpikeLayer(n_in=10, n_out=5, refrac_ms=5.0, seed=1)
    state = layer.init_state(2)
    x = torch.rand(50, 2, 10).lt(0.5).float()
    with torch.no_grad():
        _, state, _ = layer.forward(x, state)
    layer.reset_state_(state)
    assert torch.isinf(state["timer"]).all()
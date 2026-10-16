"""T1.5 — spike trace & Poisson encoder tests."""

import pytest
import torch

from rak.neuron.trace import (
    PoissonEncoder,
    decay_to_target,
    poisson_spike_train,
    spike_trace,
)


def test_decay_to_target_math():
    # exp(-1/50) untuk tau=50, dt=1
    g = decay_to_target(50.0, 1.0)
    assert g == pytest.approx(0.9802, abs=1e-4)


def test_decay_rejects_zero_tau():
    with pytest.raises(ValueError, match="tau_ms"):
        decay_to_target(0.0)


def test_tau_larger_means_slower_decay():
    """tau lebih besar = decay lebih pelan = gamma lebih dekat ke 1."""
    assert decay_to_target(100.0) > decay_to_target(10.0)
    assert decay_to_target(10.0) > 0.9


# ---------------------------------------------------------------- T1.5.1
def test_trace_matches_recurrence():
    spikes = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]])
    tau, dt = 50.0, 1.0
    g = decay_to_target(tau, dt)
    x = spike_trace(spikes, tau, dt)

    manual = []
    state = 0.0
    for z in spikes[0]:
        state = g * state + float(z)
        manual.append(state)
    assert torch.allclose(x[0], torch.tensor(manual), atol=1e-6)


def test_trace_spike_sets_to_one():
    spikes = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    x = spike_trace(spikes, 50.0)
    assert float(x[0, 2]) == pytest.approx(1.0)


# ---------------------------------------------------------------- T1.5.2
def test_trace_accepts_initial_value():
    """`initial` = nilai trace SEBELUM timestep pertama (setara x[-1]).

    Jadi x[0] = gamma * initial + z[0], sesuai rekurensi.
    """
    spikes = torch.zeros(1, 3)
    x0 = spike_trace(spikes, 50.0, initial=1.0)
    assert float(x0[0, 0]) == pytest.approx(0.9802, abs=1e-4)
    assert float(x0[0, 1]) < float(x0[0, 0])


def test_trace_accepts_tensor_initial():
    spikes = torch.zeros(4, 3)
    x0 = spike_trace(spikes, 50.0, initial=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert x0.shape == (4, 3)
    assert float(x0[2, 0]) == pytest.approx(3.0 * 0.9802, abs=1e-4)


# ---------------------------------------------------------------- T1.5.3
def test_trace_monotonic_decay_after_spike_stops():
    spikes = torch.tensor([[1.0] + [0.0] * 20])
    x = spike_trace(spikes, 50.0)
    tail = x[0, 1:]
    assert torch.all(tail[1:] < tail[:-1]), "trace harus monoton turun"
    assert float(tail[-1]) > 0.0


def test_trace_is_causal():
    """Trace di t tidak boleh bergantung spike di t+1."""
    a = torch.tensor([[1.0, 1.0, 1.0]])
    b = torch.tensor([[1.0, 1.0, 0.0]])
    xa = spike_trace(a, 50.0)
    xb = spike_trace(b, 50.0)
    assert float(xa[0, 1]) == pytest.approx(float(xb[0, 1]))


# ---------------------------------------------------------------- T1.5.4
def test_poisson_shape_and_binary():
    rates = torch.full((10,), 100.0)
    s = poisson_spike_train(rates, duration_ms=100.0, dt_ms=1.0)
    assert s.shape == (10, 100)
    assert set(s.unique().tolist()) <= {0.0, 1.0}


def test_poisson_rate_approximates_target():
    """Spike per neuron harus mendekati rate * durasi.

    `s.mean()` adalah PROBABILITAS per timestep, bukan jumlah spike per
    neuron. Yang benar dijumlahkan lewat sumbu waktu dulu.
    """
    target_hz = 500.0
    rates = torch.full((2000,), target_hz)
    s = poisson_spike_train(rates, duration_ms=100.0, dt_ms=1.0)
    expected_spikes = target_hz * 0.1  # 500 Hz * 0.1 s = 50 spike/neuron
    actual_spikes = float(s.sum(dim=-1).mean().item())
    assert actual_spikes == pytest.approx(expected_spikes, rel=0.10)


def test_poisson_zero_rate_gives_no_spikes():
    s = poisson_spike_train(torch.zeros(10), duration_ms=100.0)
    assert float(s.sum()) == 0.0


def test_poisson_full_rate_gives_all_spikes():
    s = poisson_spike_train(
        torch.full((10,), 1000.0), duration_ms=100.0, dt_ms=1.0
    )
    assert float(s.mean().item()) == 1.0


def test_poisson_is_reproducible_with_generator():
    rates = torch.full((50,), 200.0)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = poisson_spike_train(rates, 100.0, generator=g1)
    b = poisson_spike_train(rates, 100.0, generator=g2)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    rates = torch.full((50,), 200.0)
    a = poisson_spike_train(rates, 100.0, generator=torch.Generator().manual_seed(1))
    b = poisson_spike_train(rates, 100.0, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


# ---------------------------------------------------------------- Encoder
def test_encoder_scales_with_value():
    enc = PoissonEncoder(seed=7)
    low = enc.encode(torch.full((1, 100), 0.2), max_rate_hz=1000.0)
    high = enc.encode(torch.full((1, 100), 0.9), max_rate_hz=1000.0)
    assert float(high.sum()) > float(low.sum())


def test_encoder_reset_makes_deterministic():
    enc = PoissonEncoder()
    x = torch.full((1, 20), 0.5)
    enc.reset(99)
    a = enc.encode(x)
    enc.reset(99)
    b = enc.encode(x)
    assert torch.equal(a, b)
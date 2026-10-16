"""T1.4 — SpikeLayer tests."""

import time

import pytest
import torch

from rak.neuron.layer import SpikeLayer


def test_forward_shapes():
    layer = SpikeLayer(n_in=16, n_out=8, seed=1)
    x = torch.rand(20, 4, 16).lt(0.3).float()
    y, state, aux = layer.forward(x, None)
    assert y.shape == (20, 4, 8)
    assert state["v"].shape == (4, 8)
    assert state["trace_pre"].shape == (4, 16)
    assert state["trace_post"].shape == (4, 8)


def test_rejects_wrong_ndim():
    layer = SpikeLayer(n_in=16, n_out=8)
    with pytest.raises(ValueError, match="3D"):
        layer.forward(torch.rand(20, 16), None)


def test_rejects_wrong_n_in():
    layer = SpikeLayer(n_in=16, n_out=8)
    with pytest.raises(ValueError, match="n_in"):
        layer.forward(torch.rand(20, 4, 7), None)


def test_rejects_mismatched_state():
    layer = SpikeLayer(n_in=16, n_out=8)
    x = torch.rand(20, 4, 16).lt(0.3).float()
    bad_state = layer.init_state(batch_size=9)
    with pytest.raises(ValueError, match="state v"):
        layer.forward(x, bad_state)


def test_rejects_unknown_neuron_kind():
    with pytest.raises(ValueError, match="neuron"):
        SpikeLayer(n_in=4, n_out=4, neuron="izhikevich")


# ---------------------------------------------------------------- T1.4.2
def test_state_contains_traces():
    layer = SpikeLayer(n_in=10, n_out=5)
    st = layer.init_state(batch_size=2)
    for key in ("v", "trace_pre", "trace_post"):
        assert key in st


def test_adaptive_layer_has_theta_in_state():
    layer = SpikeLayer(n_in=10, n_out=5, neuron="adaptive")
    st = layer.init_state(batch_size=2)
    assert "theta" in st


def test_reset_state_zeroes():
    layer = SpikeLayer(n_in=10, n_out=5)
    st = layer.init_state(3)
    st["v"].fill_(9.0)
    st["trace_pre"].fill_(9.0)
    layer.reset_state_(st)
    assert float(st["v"].abs().sum()) == 0.0
    assert float(st["trace_pre"].abs().sum()) == 0.0


# ---------------------------------------------------------------- T1.4.4
def test_state_carries_across_batches_when_reused():
    """State harus berevolusi kalau dipakai ulang untuk batch berikutnya.

    Yang diperiksa adalah `state["v"]`, bukan spike output — dengan
    input kuat semua neuron saturasi, jadi spike output bisa identik
    meski state jelas sudah berubah.
    """
    layer = SpikeLayer(n_in=200, n_out=20, seed=3)
    x = layer.encode(torch.full((1, 200), 0.3), n_steps=30, seed=3)
    _, state1, _ = layer.forward(x, None)
    _, state2, _ = layer.forward(x, state1)
    assert not torch.equal(state1["v"], state2["v"]), (
        "state v harus berubah setelah diproses lagi"
    )


def test_fresh_state_gives_identical_output():
    layer = SpikeLayer(n_in=10, n_out=5, seed=3)
    x = torch.rand(30, 2, 10).lt(0.5).float()
    y1, _, _ = layer.forward(x, None)
    y2, _, _ = layer.forward(x, None)
    assert torch.equal(y1, y2)


# ---------------------------------------------------------------- T1.4.5
@pytest.mark.slow
def test_speed_benchmark_100x100():
    """T1.4.5 — 100 neuron, 100 timestep, ukur throughput."""
    n, T = 100, 100
    layer = SpikeLayer(n_in=n, n_out=n, seed=1)
    x = torch.rand(T, 1, n).lt(0.05).float()

    layer.forward(x, None)  # warmup
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        y, _, _ = layer.forward(x, None)
    elapsed = time.perf_counter() - t0

    neuron_updates = T * 1 * n * reps
    print(
        f"\n[bench] {n}x{n}, T={T}: "
        f"{neuron_updates / elapsed:,.0f} neuron-update/s, "
        f"{float(y.sum()) * reps / elapsed:,.0f} spike/s"
    )
    assert neuron_updates / elapsed > 10_000, "terlalu lambat untuk dipakai"


def test_weight_calibrated_to_fire():
    """Bobot harus dikalibrasi supaya neuron bisa mencapai threshold.

    Steady-state v LIF implisit = input current, jadi kalau bobot
    tidak dikalibrasi, layer bisa diam total (density 0).
    """
    layer = SpikeLayer(n_in=784, n_out=100, seed=1)
    x = layer.encode(torch.full((1, 784), 1.0), n_steps=350, seed=1)
    with torch.no_grad():
        y, _, _ = layer.forward(x, None)
    assert float(y.sum()) > 0, "layer tidak fire sama sekali"
    assert float(y.mean()) > 0.01, f"density terlalu rendah: {float(y.mean()):.4f}"


def test_response_is_graded_not_binary():
    """Peningkatan intensitas input harus menaikkan spike density.

    Ini yang bikin LIF bisa menyimpan informasi intensitas, bukan cuma
    "ada spike / tidak ada spike". Kalau responsnya binary, MNIST akan
    kehilangan semua informasi grayscale.
    """
    layer = SpikeLayer(n_in=784, n_out=100, seed=1)
    densities = []
    for val in (0.13, 0.25, 0.50, 1.00):
        x = layer.encode(torch.full((1, 784), val), n_steps=350, seed=1)
        with torch.no_grad():
            y, _, _ = layer.forward(x, None)
        densities.append(float(y.mean()))
    for lo, hi in zip(densities, densities[1:]):
        assert hi > lo, f"densitas tidak gradasi: {densities}"
    # Rasio max/min minimal 5x supaya gradasi-nya berarti, bukan noise
    assert densities[-1] / max(densities[0], 1e-6) > 5.0, densities


def test_zero_input_produces_no_spike():
    layer = SpikeLayer(n_in=64, n_out=20, seed=1)
    x = torch.zeros(200, 1, 64)
    with torch.no_grad():
        y, _, _ = layer.forward(x, None)
    assert float(y.sum()) == 0.0


def test_encode_shapes():
    layer = SpikeLayer(n_in=28, n_out=10, seed=1)
    values = torch.rand(3, 28)
    s = layer.encode(values, n_steps=50, seed=1)
    assert s.shape == (50, 3, 28)
    assert set(s.unique().tolist()) <= {0.0, 1.0}


def test_encode_n_steps_controls_duration():
    layer = SpikeLayer(n_in=10, n_out=4, dt_ms=1.0)
    values = torch.rand(2, 10)
    for n in (10, 100, 350):
        assert layer.encode(values, n_steps=n).shape[0] == n


def test_layer_end_to_end_encode_forward():
    layer = SpikeLayer(n_in=28, n_out=10, seed=1)
    values = torch.rand(4, 28)
    x = layer.encode(values, n_steps=100, seed=1)
    y, state, aux = layer.forward(x, None)
    assert y.shape == (100, 4, 10)
    assert aux["syn_current"].shape == (100, 4, 10)


def test_parameters_counted():
    layer = SpikeLayer(n_in=100, n_out=50)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params >= 100 * 50  # minimal bobot


def test_trace_pre_matches_input_trace():
    """trace_pre di akhir harus cocok dengan spike_trace() atas input.

    SpikeLayer pakai sumbu waktu PERTAMA (T, B, N), sedangkan
    `spike_trace` memakai sumbu waktu TERAKHIR (..., T). Jadi input
    perlu di-transpose dulu.
    """
    from rak.neuron.trace import spike_trace

    layer = SpikeLayer(n_in=6, n_out=4, tau_trace_ms=50.0, seed=1)
    x = torch.rand(25, 1, 6).lt(0.4).float()
    _, state, _ = layer.forward(x, None)
    # (T, B, N) -> (B, N, T)
    expected = spike_trace(x[:, 0].T, tau_ms=50.0)[..., -1]
    assert torch.allclose(state["trace_pre"], expected, atol=1e-5)
"""T1.3 — Izhikevich tests.

Bentuk ODE yang dipakai: `dv/dt = 0.04v^2 + 5v + 140 - w + I`,
`dw/dt = a(bv - w)` — mengikuti Brian2 / Izhikevich (2003).
"""

import pytest
import torch

from rak.neuron.izhikevich import Izhikevich, IzhikevichConfig


def test_izhikevich_rejects_zero_dt():
    with pytest.raises(ValueError, match="dt_ms"):
        IzhikevichConfig(dt_ms=0.0)


def test_rejects_unsafe_dt():
    """dt di luar rentang tervalidasi harus ditolak, bukan diam-diam dipakai.

    Batas 1.0 ms berasal dari uji konvergensi
    (`test_isi_converges_across_dt`): di atas itu error ISI > 3.4%.
    """
    with pytest.raises(ValueError, match="tervalidasi"):
        IzhikevichConfig(dt_ms=5.0)
    assert IzhikevichConfig(dt_ms=0.5).dt_ms == 0.5
    assert IzhikevichConfig(dt_ms=1.0).dt_ms == 1.0


@pytest.mark.slow
def test_isi_converges_across_dt():
    """Dokumentasi konvergensi numerik — dasar dari batas dt = 1.0 ms.

    Bandingkan ISI FISIK (bukan jumlah timestep) pada window waktu yang
    sama untuk beberapa dt. Kalau integrasi benar, ISI harus konverge.
    """
    window_ms = 2400.0
    isi_ms = {}
    for dt in (0.05, 0.2, 0.5, 1.0):
        z = Izhikevich(1, dt_ms=dt)
        steps = int(round(window_ms / dt))
        sp, _, _ = z.simulate(torch.full((steps,), 10.0))
        idx = sp.nonzero().flatten()
        assert len(idx) > 5, f"dt={dt} menghasilkan terlalu sedikit spike"
        isi_ms[dt] = float((idx[1:] - idx[:-1]).float().mean()) * dt

    reference = isi_ms[0.05]
    for dt, v in isi_ms.items():
        dev = abs(v - reference) / reference
        assert dev < 0.05, f"dt={dt}: ISI devi {dev:.1%} dari referensi"
    # dt besar harus tetap monoton mendekati referensi
    assert isi_ms[1.0] > reference, "ISI harus naik (kurang akurat) dengan dt besar"


@pytest.mark.slow
def test_no_overflow_with_huge_input():
    """Input raksasa tidak boleh menghasilkan NaN/Inf.

    Karena reset langsung memaksa v ke c = -65, suku 0.04*v^2 tidak
    pernah sempat meledak: setiap timestep yang mendorong v jauh
    langsung di-cancel oleh reset.
    """
    for i_val in (1e4, 1e6, 1e12):
        z = Izhikevich(1)
        sp, _, (v, w) = z.simulate(
            torch.full((500,), i_val), record=True
        )
        assert torch.isfinite(v).all(), f"v tidak finite untuk I={i_val}"
        assert torch.isfinite(w).all(), f"w tidak finite untuk I={i_val}"
        # v harus selalu <= threshold (reset bekerja)
        assert float(v.max()) <= z.config.v_threshold + 1e-3


def test_state_shapes():
    izh = Izhikevich(16)
    state = izh.init_state()
    assert state.v.shape == (16,)
    assert state.w.shape == (16,)
    assert float(state.v[0]) == pytest.approx(-65.0)  # v = c (reset value)
    assert float(state.w[0]) == 0.0


def test_resting_no_spike():
    """Tanpa input, neuron harus diam — bukan fire setiap timestep."""
    izh = Izhikevich(1)
    spikes, _, _ = izh.simulate(torch.zeros(3000))
    assert float(spikes.sum()) == 0.0


def test_drifts_to_analytic_rest_point():
    """v harus konverge ke titik istirahat COUPLED dari dua persamaan.

    Bukan akar dari `0.04v^2 + 5v + 140 = 0` saja — karena `-w` di
    dv/dt berarti w juga bergerak. Di titik diam:
        dw/dt = 0  ->  w = b*v
        dv/dt = 0  ->  0.04v^2 + 5v + 140 - 0.2v = 0
                   ->  0.04v^2 + 4.8v + 140 = 0
                   ->  v = (-4.8 ± sqrt(23.04 - 22.4)) / 0.08
    Akar yang stabil (f' < 0) adalah -70, dengan w = 0.2 * -70 = -14.
    """
    izh = Izhikevich(1)
    _, state, (v_rec, w_rec) = izh.simulate(torch.zeros(4000), record=True)
    assert float(v_rec[-1, 0]) == pytest.approx(-70.0, abs=0.2)
    assert float(w_rec[-1, 0]) == pytest.approx(-14.0, abs=0.2)


def test_strong_input_spikes_regularly():
    """I = 10 (RS mode) harus menghasilkan spike berulang."""
    izh = Izhikevich(1)
    spikes, _, _ = izh.simulate(torch.full((2000,), 10.0))
    n = float(spikes.sum())
    assert n > 10.0, f"harus fire untuk I=10, dapat {n} spike"

    idx = spikes.nonzero().flatten()
    isi = (idx[1:] - idx[:-1]).float()
    assert float(isi.std()) < 0.15 * float(isi.mean()), (
        f"ISI tidak konsisten (std={float(isi.std()):.2f}, "
        f"mean={float(isi.mean()):.2f}) — bukan regular spiking"
    )


# ---------------------------------------------------------------- T1.3.2
# Parameter set dari Izhikevich (2003) tabel mode firing
MODES = {
    "regular_spiking": dict(a=0.02, b=0.20, c=-65.0, d=8.0, i=10.0),
    "intrinsically_bursting": dict(a=-0.02, b=0.20, c=-55.0, d=4.0, i=15.0),
    "chattering": dict(a=-0.02, b=0.20, c=-50.0, d=2.0, i=15.0),
    "spike_freq_adaptation": dict(a=0.001, b=0.20, c=-65.0, d=8.0, i=30.0),
    "low_threshold": dict(a=0.02, b=0.20, c=-60.0, d=4.0, i=4.0),
}


def _run_mode(p: dict) -> torch.Tensor:
    izh = Izhikevich(1, a=p["a"], b=p["b"], c=p["c"], d=p["d"])
    spikes, _, _ = izh.simulate(torch.full((3000,), p["i"]))
    return spikes


@pytest.mark.parametrize("mode", sorted(MODES))
def test_each_firing_mode_produces_spikes(mode):
    p = MODES[mode]
    assert float(_run_mode(p).sum()) > 0.0, f"mode {mode} tidak spike"


def test_firing_modes_differ():
    """T1.3.2 — mode berbeda harus punya signature spike berbeda."""
    sigs = {m: _run_mode(p) for m, p in MODES.items()}
    unique = {tuple(s.int().tolist()) for s in sigs.values()}
    assert len(unique) == len(MODES), (
        "beberapa mode menghasilkan spike train identik"
    )


def test_chattering_mode_bursts():
    """Chattering mode harus menghasilkan spike beruntun rapat.

    Di mode ini isi ~3.5 ms — jauh lebih pendek dari refractory period
    mode lain. Itu signature "chattering".
    """
    p = MODES["chattering"]
    spikes = _run_mode(p)
    idx = spikes.nonzero().flatten()
    isi = (idx[1:] - idx[:-1]).float()
    assert float(isi.mean()) < 0.5 * float(
        _isi_mean(MODES["regular_spiking"])
    ), (
        f"chattering harus jauh lebih cepat dari RS: "
        f"{float(isi.mean()):.1f} vs {_isi_mean(MODES['regular_spiking']):.1f}"
    )


def _isi_mean(p: dict) -> float:
    idx = _run_mode(p).nonzero().flatten()
    return float((idx[1:] - idx[:-1]).float().mean())


def test_regular_spiking_isi_is_consistent():
    p = MODES["regular_spiking"]
    idx = _run_mode(p).nonzero().flatten()
    isi = (idx[1:] - idx[:-1]).float()
    assert float(isi.std()) < 0.10 * float(isi.mean())


def test_threshold_reset_applied():
    izh = Izhikevich(1, c=-60.0, d=5.0)
    state = izh.init_state()
    state.v.fill_(100.0)  # pastikan langsung fire
    spikes, state = izh.step(torch.zeros(1), state)
    assert float(spikes[0]) == 1.0
    assert float(state.v[0]) == pytest.approx(-60.0)
    assert float(state.w[0]) == pytest.approx(5.0)


def test_no_reset_without_spike():
    izh = Izhikevich(1)
    state = izh.init_state()
    v_before = float(state.v[0])
    _, state = izh.step(torch.zeros(1), state)
    assert float(state.v[0]) != pytest.approx(v_before), (
        "neuron tanpa spike harus berevolusi (bukan di-reset)"
    )


def test_no_nan_on_long_run():
    izh = Izhikevich(1)
    _, _, (v_rec, w_rec) = izh.simulate(
        torch.full((5000,), 25.0), record=True
    )
    assert torch.isfinite(v_rec).all()
    assert torch.isfinite(w_rec).all()
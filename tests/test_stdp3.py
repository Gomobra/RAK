r"""Test STDP Eq. 3 Diehl & Cook — rule utama paper.

Kesalahan yang sudah dikoreksi ada di sini, jadi ada testnya:

  Versi pertama modul ini mengimplementasikan bentuk 3-trace dari
  kode Brian2. Itu varian KETIGA di paper, bukan rule utama. Rule
  utama (Eq. 3) cuma punya satu trace presinaptik plus target
  `x_tar`. Test di file ini mengunci bentuk yang benar, dan
  `test_no_pre_triggered_ltd` mengunci sifat yang paling sering
  disalahpahami: rule ini TIDAK punya suku LTD dari spike pre.

Yang paling bernilai: `test_window_matches_empirical`, `test_x_tar_is_self_calibrating`,
dan `test_depression_only_at_active_neuron`.
"""

import math

import pytest
import torch

from rak.learning.stdp3 import (
    Eq3Config,
    Eq3State,
    eq3_step,
    eq3_window,
    make_eq3_state,
)


def _cfg(**kw) -> Eq3Config:
    base = dict(eta=1e-3, mu=0.2, tau_pre_ms=20.0, dt_ms=0.5,
                w_min=0.0, w_max=1.0)
    base.update(kw)
    return Eq3Config(**base)


def _run(z_pre_seq, z_post_seq, w0, cfg, n_in, n_out):
    state = make_eq3_state(n_in, n_out, cfg)
    w = w0.clone()
    for z_pre, z_post in zip(z_pre_seq, z_post_seq):
        w, state = eq3_step(z_pre, z_post, w, state, cfg)
    return w, state


# ================================================= bentuk dasar
def test_potentiation_when_pre_trace_exceeds_target():
    """`x_pre > x_tar` harus menaikkan bobot."""
    cfg = _cfg(x_tar=0.1)
    w = torch.full((2, 2), 0.5)
    state = Eq3State(pre=torch.ones(2), x_tar=torch.full((2,), 0.1))
    _, state = eq3_step(torch.zeros(2), torch.zeros(2), w, state, cfg)
    state.pre.fill_(1.0)
    w_new, _ = eq3_step(torch.zeros(2), torch.ones(2), w, state, cfg)
    assert float(w_new.mean()) > float(w.mean())


def test_depression_when_pre_trace_below_target():
    """`x_pre < x_tar` harus menurunkan bobot."""
    cfg = _cfg(x_tar=2.0)
    w = torch.full((2, 2), 0.5)
    state = Eq3State(pre=torch.zeros(2), x_tar=torch.full((2,), 2.0))
    w_new, _ = eq3_step(torch.zeros(2), torch.ones(2), w, state, cfg)
    assert float(w_new.mean()) < float(w.mean())


def test_no_pre_triggered_ltd():
    """TIDAK ADA LTD dari spike presinaptik.

    Ini perbedaan paling mendasar dari STDP klasik dan sumber
    kebingungan paling sering. Kalau test ini gagal, implementasi
    sudah kembali ke bentuk Bi-Poo dan ADR-0003 jadi tidak terpenuhi.
    """
    cfg = _cfg(x_tar=0.0)
    w = torch.full((4, 4), 0.5)
    state = Eq3State(pre=torch.zeros(4), x_tar=torch.zeros(4))

    # 200 timestep spike pre tanpa spike post sama sekali
    w_after = w.clone()
    st = state
    for _ in range(200):
        w_after, st = eq3_step(
            torch.ones(4), torch.zeros(4), w_after, st, cfg
        )
    assert torch.equal(w_after, w), (
        "spike presinaptik tidak boleh mengubah bobot tanpa "
        "spike postsinaptik"
    )


def test_weight_unchanged_without_post_spikes():
    """Konsekuensi lain: neuron yang tidak pernah menyala tidak belajar."""
    cfg = _cfg()
    w = torch.full((8, 8), 0.4)
    state = make_eq3_state(8, 8, cfg)
    for _ in range(500):
        w, state = eq3_step(torch.ones(8), torch.zeros(8), w, state, cfg)
    assert torch.equal(w, torch.full((8, 8), 0.4))


def test_simultaneous_spike_uses_pre_decay_only():
    """Spike bareng: `x_pre` yang dipakai adalah yang SEBELUM bump."""
    cfg = _cfg(x_tar=0.0)
    w = torch.full((1, 1), 0.5)
    # x_pre = 0 sebelum timestep, jadi diff = 0 - 0 = 0, tanpa efek
    state = Eq3State(pre=torch.zeros(1), x_tar=torch.zeros(1))
    w_new, _ = eq3_step(torch.ones(1), torch.ones(1), w, state, cfg)
    assert float(w_new[0, 0]) == pytest.approx(0.5, abs=1e-12)

    # Dengan x_pre yang sudah ada, bareng spike tetap memakainya
    state = Eq3State(pre=torch.ones(1), x_tar=torch.zeros(1))
    w_new, _ = eq3_step(torch.ones(1), torch.ones(1), w, state, cfg)
    assert float(w_new[0, 0]) > 0.5


# ================================================= x_tar
def test_x_tar_tracks_observed_trace():
    """EMA harus mendekati rata-rata jejak pre berbobot bobot.

    `observed[j] = Σ_i w[i,j]·x_pre[i] / Σ_i w[i,j]`

    Bobot dibuat TIDAK seragam supaya rata-rata berbobot benar-benar
    beda dari rata-rata biasa — kalau test ini pakai w seragam, dia
    hanya menguji rata-rata sederhana yang kebetulan cocok.
    """
    cfg = _cfg(x_tar_tau_ms=50.0, x_tar=None)
    # Kolom 0 berbobot di presinaptik yang ADA jejaknya; kolom 1
    # berbobot di presinaptik yang TIDAK. Kalau keduanya dibobot di
    # baris yang sama, rata-rata berbobot keduanya identik dan test
    # ini tidak menguji apa pun.
    w = torch.tensor([[3.0, 0.0],
                      [0.0, 1.0]])
    pre = torch.tensor([4.0, 0.0])              # jejak hanya di presinaptik 0

    state = Eq3State(pre=pre, x_tar=torch.zeros(2))
    # Hanya presinaptik 0 yang menyala, supaya presinaptik 1 jejaknya
    # benar-benar nol. Kalau keduanya dinyalakan, `pre[1]` ikut naik
    # ke steady state dan kolom 1 ikut dapat target yang sama — test
    # jadi tidak membedakan apa pun.
    for _ in range(2000):
        _, state = eq3_step(
            torch.tensor([1.0, 0.0]), torch.ones(2), w, state, cfg
        )

    # kolom 0: semua bobot di presinaptik yang berjejak ->observed -> pre_ss
    # kolom 1: bobotnya di presinaptik tanpa jejak ->observed -> 0
    pre_ss = 1.0 / (1.0 - cfg.gamma_pre)
    assert float(state.x_tar[0]) == pytest.approx(pre_ss, rel=0.05), (
        f"kolom berat harus dekati jejak steady {pre_ss:.2f}, "
        f"dapat {float(state.x_tar[0]):.3f}"
    )
    assert float(state.x_tar[1]) == pytest.approx(0.0, abs=0.05), (
        f"kolom ringan harus dekati 0, dapat {float(state.x_tar[1]):.3f}"
    )
    assert float(state.x_tar[0]) > float(state.x_tar[1])


def test_x_tar_is_self_calibrating():
    """`x_tar` harus bergerak ke arah rata-rata, bukan diam.

    Kalau `x_tar` tidak pernah bergerak, seluruh bobot akan monoton
    naik sampai `w_max` — itu bukan STDP, itu unbounded potentiation.
    """
    cfg = _cfg(x_tar_tau_ms=100.0)
    g = torch.Generator().manual_seed(1)
    n_in, n_out = 32, 8
    w = torch.rand(n_in, n_out, generator=g) * 0.3 + 0.01
    state = make_eq3_state(n_in, n_out, cfg)

    initial = state.x_tar.clone()
    for _ in range(100):
        zp = (torch.rand(n_in, generator=g) < 0.1).float()
        zq = (torch.rand(n_out, generator=g) < 0.1).float()
        w, state = eq3_step(zp, zq, w, state, cfg)

    assert not torch.equal(initial, state.x_tar), "x_tar tidak bergerak"
    assert float(state.x_tar.max()) > 0.01


def test_fixed_x_tar_option_bypasses_ema():
    """Kalau `x_tar` dikunci ke konstanta, EMA tidak boleh jalan."""
    cfg = _cfg(x_tar=0.42)
    g = torch.Generator().manual_seed(2)
    n_in, n_out = 16, 4
    w = torch.rand(n_in, n_out, generator=g) * 0.3
    state = make_eq3_state(n_in, n_out, cfg)

    for _ in range(200):
        zp = (torch.rand(n_in, generator=g) < 0.2).float()
        zq = (torch.rand(n_out, generator=g) < 0.2).float()
        w, state = eq3_step(zp, zq, w, state, cfg)

    assert torch.allclose(state.x_tar, torch.full((n_out,), 0.42))


def test_x_tar_saturates_at_mean_trace():
    """Dalam jangka panjang `x_tar` menutup ke jejak steady-state.

    Yang diuji: spike pre TERUS-MENERUS membuat `x_pre` konverge ke
    `1/(1-gamma)`, dan `x_tar` harus mengejar nilai itu — bukan ke
    nilai spike pertamanya.

    Versi pertama test ini mengisi `state.pre` sekali lalu tidak
    pernah di-bump, jadi `x_pre` meluruh ke nol dan `x_tar` ikut
    meluruh. Test itu salah, bukan kodenya.
    """
    cfg = _cfg(x_tar_tau_ms=200.0, x_tar=None)
    w = torch.full((4, 4), 0.5)
    state = Eq3State(pre=torch.zeros(4), x_tar=torch.zeros(4))
    for _ in range(5000):
        _, state = eq3_step(torch.ones(4), torch.ones(4), w, state, cfg)

    pre_ss = 1.0 / (1.0 - cfg.gamma_pre)
    assert float(state.pre[0]) == pytest.approx(pre_ss, rel=1e-3)
    assert float(state.x_tar[0]) == pytest.approx(pre_ss, rel=0.05), (
        f"x_tar harus dekati {pre_ss:.2f}, dapat {float(state.x_tar[0]):.3f}"
    )


# ================================================= window analitik
def test_window_matches_empirical():
    """Loop diskret harus cocok dengan `eta·x_pre·(w_max−w)^μ`."""
    cfg = _cfg(x_tar=0.0)
    dts, analytic = eq3_window(cfg.dt_ms, cfg, max_ms=50.0)

    worst, worst_dt = 0.0, 0.0
    for dt, want in zip(dts.tolist(), analytic.tolist()):
        if dt <= 0:
            continue
        k = int(round(dt / cfg.dt_ms))
        w = torch.full((1, 1), 0.5)
        state = Eq3State(pre=torch.zeros(1), x_tar=torch.zeros(1))
        total = 0.0
        for t in range(k + 1):
            z_pre = torch.tensor([1.0]) if t == 0 else torch.zeros(1)
            z_post = torch.tensor([1.0]) if t == k else torch.zeros(1)
            w_new, state = eq3_step(z_pre, z_post, w, state, cfg)
            total += float(w_new[0, 0] - w[0, 0])
            w = w_new
        err = abs(total - want) / max(abs(want), 1e-12)
        if err > worst:
            worst, worst_dt = err, dt

    assert worst < 1e-3, f"galat {worst:.2e} di dt={worst_dt}ms"


def test_window_has_no_negative_branch():
    """Tidak ada update untuk `dt < 0` — konsekuensi desain."""
    cfg = _cfg()
    dts, win = eq3_window(cfg.dt_ms, cfg, max_ms=50.0)
    assert float(win[dts < 0].abs().max()) == 0.0


def test_window_decay_matches_closed_form():
    cfg = _cfg()
    dts, win = eq3_window(1.0, cfg, max_ms=100.0)
    for dt, got in zip(dts.tolist(), win.tolist()):
        if dt <= 0:
            continue
        want = cfg.eta * math.exp(-dt / cfg.tau_pre_ms) * (cfg.w_max / 2) ** cfg.mu
        assert got == pytest.approx(want, rel=1e-6)


# ================================================= mu
def test_mu_zero_is_linear():
    """mu=0 → (w_max−w)^0 = 1 -> linear."""
    cfg = _cfg(mu=0.0, x_tar=0.0)
    w = torch.tensor([[0.2], [0.8]])
    state = Eq3State(pre=torch.ones(2), x_tar=torch.zeros(2))
    gains = []
    for row in range(2):
        state_r = Eq3State(pre=torch.ones(1), x_tar=torch.zeros(1))
        _, state_r = eq3_step(torch.zeros(1), torch.zeros(1),
                              torch.full((1, 1), float(w[row, 0])), state_r, cfg)
        after, _ = eq3_step(torch.zeros(1), torch.ones(1),
                            torch.full((1, 1), float(w[row, 0])), state_r, cfg)
        gains.append(float(after[0, 0]) - float(w[row, 0]))

    assert gains[0] == pytest.approx(gains[1], rel=1e-5), (
        f"mu=0 harus linear, dapat {gains}"
    )


def test_higher_mu_stronger_dependence_on_weight():
    """mu besar → synapse yang sudah besar tumbuh lebih lambat."""
    w_lo = torch.tensor([[0.2]])
    w_hi = torch.tensor([[0.8]])

    def gain(cfg, w):
        state = Eq3State(pre=torch.ones(1), x_tar=torch.zeros(1))
        after, _ = eq3_step(torch.zeros(1), torch.ones(1), w, state, cfg)
        return float(after[0, 0]) - float(w[0, 0])

    for mu in (0.5, 2.0):
        cfg = _cfg(mu=mu, x_tar=0.0)
        g_lo, g_hi = gain(cfg, w_lo), gain(cfg, w_hi)
        assert g_lo > g_hi, f"mu={mu}: bobot kecil harus tumbuh lebih cepat"


def test_weight_dependence_weakens_growth_near_max():
    """Tujuan `(w_max − w)^μ`: perlambatan dekat batas."""
    cfg = _cfg(mu=1.0, x_tar=0.0)
    gains = []
    for start in (0.1, 0.5, 0.9):
        w = torch.full((1, 1), start)
        state = Eq3State(pre=torch.ones(1), x_tar=torch.zeros(1))
        after, _ = eq3_step(torch.zeros(1), torch.ones(1), w, state, cfg)
        gains.append(float(after[0, 0]) - start)
    assert gains[0] > gains[1] > gains[2], gains


# ================================================= batas
def test_weights_stay_in_bounds():
    cfg = _cfg(eta=0.1)
    g = torch.Generator().manual_seed(3)
    n_in, n_out = 16, 8
    w = torch.rand(n_in, n_out, generator=g)
    state = make_eq3_state(n_in, n_out, cfg)
    for _ in range(500):
        zp = (torch.rand(n_in, generator=g) < 0.5).float()
        zq = (torch.rand(n_out, generator=g) < 0.5).float()
        w, state = eq3_step(zp, zq, w, state, cfg)
    assert float(w.min()) >= cfg.w_min - 1e-7
    assert float(w.max()) <= cfg.w_max + 1e-7


def test_input_not_mutated():
    cfg = _cfg()
    w = torch.full((2, 2), 0.5)
    w0 = w.clone()
    state = make_eq3_state(2, 2, cfg)
    eq3_step(torch.ones(2), torch.ones(2), w, state, cfg)
    assert torch.equal(w, w0)


def test_trace_shape_validated():
    cfg = _cfg()
    w = torch.full((4, 4), 0.5)
    bad = Eq3State(pre=torch.zeros(16), x_tar=torch.zeros(4))
    with pytest.raises(ValueError, match="per-neuron"):
        eq3_step(torch.zeros(4), torch.zeros(4), w, bad, cfg)


def test_no_nan_with_degenerate_weights():
    """Kolom nol tidak boleh menghasilkan NaN."""
    cfg = _cfg()
    w = torch.zeros(4, 4)
    state = Eq3State(pre=torch.ones(4), x_tar=torch.zeros(4))
    w_new, _ = eq3_step(torch.zeros(4), torch.ones(4), w, state, cfg)
    assert torch.isfinite(w_new).all()


def test_eta_scales_growth_linearly():
    """Kenaikan bobot harus linear terhadap eta."""
    gains = []
    for eta in (1e-3, 2e-3, 4e-3):
        cfg = _cfg(eta=eta, x_tar=0.0)
        w = torch.full((1, 1), 0.5)
        state = Eq3State(pre=torch.ones(1), x_tar=torch.zeros(1))
        after, _ = eq3_step(torch.zeros(1), torch.ones(1), w, state, cfg)
        gains.append(float(after[0, 0]) - 0.5)
    # toleransi longkar: arithmetic dilakukan dalam float32
    assert gains[0] * 2 == pytest.approx(gains[1], rel=1e-4)
    assert gains[1] * 2 == pytest.approx(gains[2], rel=1e-4)


# ================================================= batch
def test_batched_matches_looped():
    cfg = _cfg(x_tar=0.0)
    gen = torch.Generator().manual_seed(4)
    zp_seq = (torch.rand(5, 3, 4, generator=gen) < 0.4).float()
    zq_seq = (torch.rand(5, 3, 4, generator=gen) < 0.4).float()
    w0 = torch.rand(4, 4, generator=gen)

    st = make_eq3_state(4, 4, cfg)
    w = w0.clone()
    out = []
    for t in range(5):
        w, st = eq3_step(zp_seq[t], zq_seq[t], w, st, cfg)
        out.append(w.clone())

    # Batched memakai SATU matriks bobot bersama dan update-nya
    # dirata-ratakan, jadi hasilnya dekat dengan rata-rata jalur
    # individual — bukan identik dengan salah satunya. Bedanya O(dw^2)
    # karena `(w_max - w)^mu` tidak linear.
    for row in range(3):
        ws = w0.clone()
        ss = make_eq3_state(4, 4, cfg)
        for t in range(5):
            ws, ss = eq3_step(zp_seq[t, row], zq_seq[t, row], ws, ss, cfg)
        assert (float(out[0].max()) < float(ws.max()) + 1e-6), (
            "batched tidak boleh melebihi jalur individual"
        )

    # Yang WAJIB identik: batch BERUKURAN 1 harus sama persis dengan
    # jalur non-batched. (Batch 3 di atas tidak akan sama dengan baris
    # pertamanya, karena `w` dipakai bersama dan update-nya
    # dirata-ratakan.)
    one_pre = zp_seq[:, :1]        # (5, 1, n_in) — tetap 3D
    one_post = zq_seq[:, :1]
    w1 = w0.clone()
    s1 = make_eq3_state(4, 4, cfg)
    out1 = []
    for t in range(5):
        w1, s1 = eq3_step(one_pre[t], one_post[t], w1, s1, cfg)
        out1.append(w1.clone())

    ws = w0.clone()
    ss = make_eq3_state(4, 4, cfg)
    out_single = []
    for t in range(5):
        ws, ss = eq3_step(zp_seq[t, 0], zq_seq[t, 0], ws, ss, cfg)
        out_single.append(ws.clone())

    for t in range(5):
        assert torch.allclose(out1[t], out_single[t], atol=1e-6), (
            f"batch-1 harus identik dengan non-batched, beda di t={t}: "
            f"maks {float((out1[t] - out_single[t]).abs().max()):.2e}"
        )


# ================================================= konfigurasi
def test_config_validation():
    with pytest.raises(ValueError, match="eta"):
        _cfg(eta=-1.0)
    with pytest.raises(ValueError, match="tau_pre_ms"):
        _cfg(tau_pre_ms=0.0)
    with pytest.raises(ValueError, match="w_min"):
        _cfg(w_min=2.0, w_max=1.0)
    with pytest.raises(ValueError, match="x_tar"):
        _cfg(x_tar=-1.0)


def test_gamma_matches_closed_form():
    cfg = _cfg()
    assert cfg.gamma_pre == pytest.approx(
        math.exp(-cfg.dt_ms / cfg.tau_pre_ms)
    )


def test_paper_reference_values():
    """Nilai yang disebut paper: mu = 0.2, tau_pre = 20 ms, w_max = 1."""
    cfg = Eq3Config()
    assert cfg.mu == 0.2
    assert cfg.tau_pre_ms == 20.0
    assert cfg.w_max == 1.0
    assert cfg.dt_ms == 0.5
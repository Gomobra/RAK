"""T2.1 — Hebbian / BCM / Oja tests.

Yang diuji di sini BUKAN "update menghasilkan tensorBx" — itu asumsi
penulis. Yang diuji adalah PROPERTI MATEMATIS yang harus benar
apa pun nilai parameternya:

    - Hebbian tanpa normalisasi = divergen (membuktikan BCM/Oja perlu)
    - Oja punya titik seimbang
    - BCM dua arah (LTP di atas ambang, LTD di bawah)
    - BCM theta bergerak lambat, tidak meniru aktivitas instan
"""

import math

import pytest
import torch

from rak.learning.hebbian import (
    BCMConfig,
    HebbianConfig,
    OjaConfig,
    bcm_theta,
    bcm_update,
    hebbian_update,
    oja_update,
)


def _rand(n_in, n_out, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_in, n_out, generator=g) * scale


# ------------------------------------------------------------- Hebbian
def test_hebbian_update_shape_and_outer_product():
    w = torch.zeros(4, 3)
    x_pre = torch.tensor([1.0, 2.0, 0.0, 0.0])
    x_post = torch.tensor([0.5, 0.0, 1.0])
    dw = hebbian_update(w, x_pre, x_post)
    # kolom 0: eta * x_pre * 0.5 ; kolom 2: eta * x_pre * 1.0
    assert dw.shape == (4, 3)
    assert float(dw[1, 0]) == pytest.approx(2.0 * 0.5 * 1e-4, rel=1e-5)
    assert float(dw[1, 2]) == pytest.approx(2.0 * 1.0 * 1e-4, rel=1e-5)
    assert float(dw[0, 1]) == pytest.approx(0.0, abs=1e-12)


def test_hebbian_does_not_mutate_weights():
    w = _rand(5, 5, seed=1)
    w0 = w.clone()
    hebbian_update(w, torch.ones(5), torch.ones(5))
    assert torch.equal(w, w0), "update harus mengembalikan dW, bukan mutasi w"


def test_hebbian_diverges_without_normalization():
    """Bukti bahwa plain Hebbian TIDAK PERNAH BERHENTI.

    Pertumbuhannya LINEAR (w += eta * x_pre * x_post tiap langkah), jadi
    tidak ada titik seimbang. Kita buktikan dengan menunjukkan bahwa
   pertumbuhan pada putaran kedua sama besarnya dengan putaran pertama —
    kalau ada normalisasi, putaran kedua akan melambat.
    """
    w = torch.full((50, 50), 0.01)
    x_pre = torch.ones(50)
    x_post = torch.ones(50)

    for _ in range(10_000):
        w = w + hebbian_update(w, x_pre, x_post)
    after_first = float(w.mean())

    for _ in range(10_000):
        w = w + hebbian_update(w, x_pre, x_post)
    after_second = float(w.mean())

    delta_first = after_first - 0.01
    delta_second = float(w.mean()) - after_first
    assert delta_second == pytest.approx(delta_first, rel=1e-4), (
        f"pertumbuhan harus konstan (divergen), "
        f"{delta_first:.5f} lalu {delta_second:.5f}"
    )


def test_hebbian_all_neurons_become_identical():
    """Divergensi Hebbian = amnesia: semua neuron jadi sama."""
    w = torch.full((30, 30), 0.01)
    x_pre = torch.ones(30)
    x_post = torch.ones(30)
    for _ in range(1500):
        w = w + hebbian_update(w, x_pre, x_post)
    rows = w[0]
    spread = float((w - rows.unsqueeze(0)).abs().max())
    assert spread == pytest.approx(0.0, abs=1e-6), (
        f"semua baris harus identik, selisih maks {spread:.2e}"
    )


# ------------------------------------------------------------------- Oja
def test_oja_decays_when_pre_already_large():
    """Suku gesek harus mengalahkan Hebbian kalau W sudah besar."""
    w = torch.full((4, 4), 10.0)
    dw = oja_update(w, torch.ones(4), torch.ones(4))
    assert (dw < 0).all(), f"semua dW harus < 0, dapat max={float(dw.max()):.3f}"


def test_oja_potentiates_when_pre_is_small():
    w = torch.full((4, 4), 1e-4)
    dw = oja_update(w, torch.ones(4), torch.ones(4))
    assert (dw > 0).all()


def test_oja_has_equilibrium():
    """Titik seimbang: di w* = a_pre / (beta * a_post), dW = 0."""
    a_pre = torch.tensor([2.0])
    a_post = torch.tensor([0.5])
    beta = 1.0
    w_star = a_pre / (beta * a_post)
    dw = oja_update(w_star.reshape(1, 1), a_pre, a_post)
    assert float(dw.abs().max()) < 1e-6, f"dW harus ~0 di titik seimbang"


def test_oja_keeps_norms_bounded():
    """Oja harus mencegah blowup pada input berulang."""
    n = 40
    w = torch.full((n, n), 0.01)
    x = torch.ones(n)
    for _ in range(5000):
        w = w + oja_update(w, x, x)
    assert torch.isfinite(w).all()
    assert float(w.abs().max()) < 1e3, f"Oja tidak membatasi: {float(w.abs().max()):.1f}"


def test_oja_norm_growth_is_sublinear():
    """Pertumbuhan norma harus melambat (tidak linier seperti Hebbian)."""
    n = 20
    w = torch.full((n, n), 0.01)
    x = torch.ones(n)
    norms = []
    for _ in range(3000):
        w = w + oja_update(w, x, x)
        if len(norms) < 3:
            norms.append(float(w.norm()))
        norms.append(float(w.norm()))
    d1 = norms[1000] - norms[0]
    d3 = norms[3000] - norms[2000]
    assert d3 < d1, f"pertumbuhan harus melambat: awal {d1:.3f} akhir {d3:.3f}"


def test_oja_slower_than_hebbian_at_matched_rate():
    """Dengan laju belajar SAMA, Oja harus lebih kecil dari Hebbian.

    Membandingkan dua aturan dengan learning rate berbeda tidak
    membuktikan apa pun — jadi lajunya harus disamakan dulu supaya
    satu-satunya perbedaan adalah suku gesek.
    """
    w = torch.full((4, 4), 0.5)
    xp = torch.ones(4)
    xq = torch.ones(4)
    rate = 1e-3
    d_heb = hebbian_update(w, xp, xq, HebbianConfig(eta=rate))
    d_oja = oja_update(w, xp, xq, OjaConfig(kappa=rate, beta=1.0))
    assert float(d_oja.abs().sum()) < float(d_heb.abs().sum())


def test_oja_equals_hebbian_when_weight_is_zero():
    """Di w = 0, suku gesek Oja ikut nol -> identik dengan Hebbian."""
    w = torch.zeros(4, 4)
    xp = torch.rand(4)
    xq = torch.rand(4)
    rate = 1e-3
    d_heb = hebbian_update(w, xp, xq, HebbianConfig(eta=rate))
    d_oja = oja_update(w, xp, xq, OjaConfig(kappa=rate, beta=1.0))
    assert torch.allclose(d_heb, d_oja, atol=1e-12)


# ------------------------------------------------------------------- BCM
def test_bcm_potentiates_above_threshold():
    """a_j di atas theta -> LTP."""
    theta = torch.tensor([1.0])
    a_post = torch.tensor([3.0])
    a_pre = torch.ones(2)
    dw = bcm_update(torch.zeros(2, 1), a_pre, a_post, theta)
    assert (dw > 0).all(), "harus LTP"


def test_bcm_depresses_below_threshold():
    """a_j di bawah theta -> LTD."""
    theta = torch.tensor([5.0])
    a_post = torch.tensor([1.0])
    a_pre = torch.ones(2)
    dw = bcm_update(torch.zeros(2, 1), a_pre, a_post, theta)
    assert (dw < 0).all(), "harus LTD"


def test_bcm_is_zero_at_threshold():
    theta = torch.tensor([2.0])
    a_post = torch.tensor([2.0])
    dw = bcm_update(torch.zeros(2, 1), torch.ones(2), a_post, theta)
    assert float(dw.abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_bcm_theta_tracks_mean_square_slowly():
    """theta harus mendekati rata-rata kuadrat, tapi BUKAN serta-merta.

    Filter order-1 dengan time constant tau: setelah 1*tau baru 63%
    konvergen. Jadipenuh butuh ~10*tau.
    """
    tau = 100.0
    cfg = BCMConfig(tau_theta_ms=tau)
    theta = torch.zeros(3)
    a = torch.full((3,), 2.0)  # a^2 = 4
    after_one = bcm_theta(theta, a, cfg)
    assert 0.0 < float(after_one[0]) < 4.0, (
        f"satu langkah harus kecil, bukan langsung 4, dapat {float(after_one[0])}"
    )

    # Setelah 1 * tau: harus ~63% dari 4
    for _ in range(int(tau)):
        theta = bcm_theta(theta, a, cfg)
    assert float(theta[0]) == pytest.approx(4.0 * (1 - math.exp(-1)), rel=0.02)

    # Setelah 10 * tau: konvergen
    for _ in range(9 * int(tau)):
        theta = bcm_theta(theta, a, cfg)
    assert float(theta[0]) == pytest.approx(4.0, rel=0.01)


def test_bcm_theta_never_negative():
    """theta tidak boleh minus tak hingga saat neuron diam."""
    theta = torch.zeros(3)
    cfg = BCMConfig(tau_theta_ms=10.0)
    for _ in range(200):
        theta = bcm_theta(theta, torch.zeros(3), cfg)
    assert float(theta.min()) >= cfg.theta_min


def test_bcm_theta_respects_min():
    cfg = BCMConfig(tau_theta_ms=1.0, theta_min=0.5)
    theta = torch.zeros(3)
    for _ in range(50):
        theta = bcm_theta(theta, torch.zeros(3), cfg)
    assert float(theta.min()) >= 0.5


def test_bcm_faster_detector_tracks_more_closely():
    """tau_detector kecil -> theta lebih dekat ke a^2 pada langkah sama."""
    a = torch.full((3,), 2.0)
    fast = bcm_theta(torch.zeros(3), a, BCMConfig(tau_theta_ms=10.0))
    slow = bcm_theta(torch.zeros(3), a, BCMConfig(tau_theta_ms=10_000.0))
    assert float(fast[0]) > float(slow[0])


def test_bcm_equilibrium_above_sustained_activity():
    """Dengan aktivitas konstan, neuron harus ter-depress (theta naik)."""
    cfg = BCMConfig(kappa=1e-2, tau_theta_ms=100.0)
    theta = torch.zeros(1)
    w = torch.full((1, 1), 0.1)
    a_post = torch.full((1,), 3.0)
    a_pre = torch.ones(1)
    for _ in range(3000):
        dw = bcm_update(w, a_pre, a_post, theta, cfg)
        w = (w + dw).clamp(0.0, 1.0)
        theta = bcm_theta(theta, a_post, cfg)
    # theta_ha = a^2 = 9 > a = 3, jadi (a - theta) < 0 -> LTD terus
    assert float(w[0, 0]) < 0.1, (
        f"neuron over-active harus ter-depress, dapat w={float(w[0, 0]):.4f}"
    )


def test_bcm_selective_strengthening():
    """Dua neuron post: yang over-active di-depress, yang lain menguat."""
    cfg = BCMConfig(kappa=1e-2, tau_theta_ms=200.0)
    theta = torch.zeros(2)
    w = torch.full((2, 2), 0.2)
    a_pre = torch.ones(2)
    a_post = torch.tensor([4.0, 0.2])  # neuron 0 over-active, neuron 1 diam
    for _ in range(2000):
        dw = bcm_update(w, a_pre, a_post, theta, cfg)
        w = (w + dw).clamp(0.0, 1.0)
        theta = bcm_theta(theta, a_post, cfg)
    assert float(w[0, 0]) < 0.2, "neuron over-active harus turun"
    assert float(w[0, 1]) > 0.2, "neuron yang boleh bereaksi harus menguat"


# ------------------------------------------------------------- kontra
def test_hebbian_never_shrinks_weights():
    """Hebbian monoton: tidak pernah mengurangi bobot."""
    w = _rand(3, 3, seed=2, scale=5.0)
    dw = hebbian_update(w, torch.rand(3), torch.rand(3))
    assert (dw >= 0).all()


def test_oja_can_be_negative_whereas_hebbian_cannot():
    w = torch.full((3, 3), 2.0)
    assert (hebbian_update(w, torch.ones(3), torch.ones(3)) >= 0).all()
    assert (oja_update(w, torch.ones(3), torch.ones(3)) < 0).all()


def test_updates_are_finite_for_reasonable_inputs():
    for fn in (hebbian_update, bcm_update, oja_update):
        w = _rand(8, 8, seed=3, scale=0.5)
        args = (w, torch.rand(8), torch.rand(8))
        if fn is bcm_update:
            dw = fn(*args, torch.rand(8) + 0.1)
        else:
            dw = fn(*args)
        assert torch.isfinite(dw).all(), f"{fn.__name__} menghasilkan non-finite"
"""T1B.1 — Intrinsic plasticity (adaptive threshold) sebagai homeostasis."""

import warnings

import pytest
import torch

from rak.homeostasis.intrinsic import (
    TIME_SCALES,
    HomeostaticConfig,
    make_homeostatic_neuron,
    validate_timescales,
)
from rak.homeostasis.scaling import rate_uniformity


# ------------------------------------------------------------ konfigurasi
def test_default_is_realistic_hour():
    """Default harus realistic, bukan mode demo."""
    cfg = HomeostaticConfig()
    assert cfg.tau_theta_ms == TIME_SCALES["realistic_hour"]


def test_diehl_cook_scale_is_available():
    """Nilai yang dipakai paper (1e7 ms) harus bisa dipilih."""
    cfg = HomeostaticConfig(tau_theta_ms=TIME_SCALES["diehl_cook_2.8h"])
    assert cfg.tau_theta_ms == 10_000_000.0


def test_rejects_nonpositive_tau():
    with pytest.raises(ValueError, match="tau_theta_ms"):
        HomeostaticConfig(tau_theta_ms=0.0)


def test_alpha_theta_is_valid_decay():
    cfg = HomeostaticConfig(tau_theta_ms=1000.0, dt_ms=1.0)
    assert 0.0 < cfg.alpha_theta < 1.0
    assert cfg.alpha_theta == pytest.approx(0.999, abs=1e-6)


# ------------------------------------------------------------ validasi
def test_zenke_scale_has_no_warnings():
    """Skala yang benar-Jenke harus lolos bersih."""
    cfg = HomeostaticConfig(tau_theta_ms=TIME_SCALES["realistic_hour"])
    assert validate_timescales(cfg) == []


def test_subsecond_tau_is_flagged():
    """tau_theta < 1 detik melanggar Zenke 2013 -> harus diperingatkan."""
    cfg = HomeostaticConfig(tau_theta_ms=100.0)
    warns = validate_timescales(cfg)
    assert warns, "tau_theta sub-detik harus diberi peringatan"
    assert any("Zenke" in w for w in warns)


def test_theta_faster_than_membrane_is_flagged():
    cfg = HomeostaticConfig(tau_theta_ms=5.0, tau_mem_ms=20.0)
    warns = validate_timescales(cfg)
    assert any("tau_mem" in w for w in warns)


def test_tight_ratio_is_flagged():
    """Homeostasis harus JAUH lebih lambat dari membran (>100x)."""
    cfg = HomeostaticConfig(tau_theta_ms=500.0, tau_mem_ms=20.0)
    warns = validate_timescales(cfg)
    assert any("100x" in w for w in warns)


# ------------------------------------------------------ builder
def test_make_neuron_returns_adaptive_lif():
    from rak.neuron import AdaptiveLIF

    n = make_homeostatic_neuron(10)
    assert isinstance(n, AdaptiveLIF)
    assert n.n_neurons == 10
    assert n.config.tau_adapt_ms == TIME_SCALES["realistic_hour"]


def test_make_neuron_with_scale_name():
    n = make_homeostatic_neuron(4, scale="demo_seconds")
    assert n.config.tau_adapt_ms == TIME_SCALES["demo_seconds"]


def test_make_neuron_rejects_unknown_scale():
    with pytest.raises(ValueError, match="scale"):
        make_homeostatic_neuron(4, scale="ngawur")


def test_strict_mode_raises_on_bad_scale():
    """`strict=True` mengubah peringatan jadi exception."""
    with pytest.raises(ValueError):
        make_homeostatic_neuron(
            4, config=HomeostaticConfig(tau_theta_ms=100.0), strict=True
        )


def test_lenient_mode_warns_on_bad_scale():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_homeostatic_neuron(4, config=HomeostaticConfig(tau_theta_ms=100.0))
    assert any(issubclass(c.category, RuntimeWarning) for c in caught)


# ------------------------------------------------------------ perilaku
def test_theta_grows_and_decays():
    # tau_theta=100ms sengaja melanggar Zenke — test ini cuma mau lihat
    # mekanisme decay-nya cepat, bukan mengklaim konfigurasi ini valid.
    with pytest.warns(RuntimeWarning, match="Zenke"):
        n = make_homeostatic_neuron(1, config=HomeostaticConfig(
            tau_theta_ms=100.0, beta_theta=0.1, tau_mem_ms=20.0))
    state = n.init_state()
    state.v.fill_(10.0)
    _, state = n.step(torch.tensor([10.0]), state)
    assert float(state.theta[0]) > 0.0
    theta_after_spike = float(state.theta[0])

    for _ in range(20):
        _, state = n.step(torch.tensor([0.0]), state)
    assert float(state.theta[0]) < theta_after_spike, (
        "theta harus meluruh saat tidak ada spike"
    )


def test_longer_tau_theta_means_longer_memory():
    """tau_theta = "seberapa lama neuron ingat bahwa dia terlalu aktif".

    Diukur langsung: berapa timestep sampai theta meluruh jadi setengah.
    Harus ~tau_theta * ln(2). Menguji mechanismenya tanpa harus menunggu
    ribuan spike supaya ambang efeknya kelihatan.
    """
    halvings = {}
    for scale in ("demo_seconds", "zenke_minimum"):
        tau = TIME_SCALES[scale]
        n = make_homeostatic_neuron(
            1,
            config=HomeostaticConfig(
                tau_theta_ms=tau, beta_theta=1e-3, tau_mem_ms=20.0
            ),
        )
        state = n.init_state()
        state.theta.fill_(1.0)
        steps = 0
        # cukup 1.5 * ln(2) ~ 1.04 * tau untuk sampai separuh
        for steps in range(1, int(tau * 1.2)):
            _, state = n.step(torch.tensor([0.0]), state)
            if float(state.theta[0]) <= 0.5:
                break
        halvings[scale] = steps

    ratio = halvings["zenke_minimum"] / halvings["demo_seconds"]
    expected = TIME_SCALES["zenke_minimum"] / TIME_SCALES["demo_seconds"]
    assert ratio == pytest.approx(expected, rel=0.1), (
        f"rasio half-life {ratio:.2f} harus ~{expected:.2f} (= rasio tau)"
    )


def test_steady_state_theta_depends_on_beta_times_tau():
    """Insight untuk tuning Phase 3: theta_ss = beta_theta * rate * tau_theta.

    Di steady state, teka-tekan equilibrium:

        theta_ss * (1 - gamma) = beta_theta * rate * tau_theta * (1 - gamma)
        theta_ss = beta_theta * rate * tau_theta

    Kalau produk `beta_theta * tau_theta` dijaga konstan, laju firing
    steady-state-nya juga konstan — SEBUk apa pun tau-nya. Itu yang
    membuat pengujian "tau panjang = penJARANGAN lebih kuat" menyesatkan:
    efeknya hilang kalau beta ikut diskalakan.

    Konsekuensi praktis: untuk menghindari neuron mati, yang penting
    menjaga `beta_theta * tau_theta` kecil — bukan salah satu saja.

    Asumsinya: simulasi harus jauh lebih panjang dari tau_theta supaya
    steady state tercapai.
    """
    tau = 300.0
    rate_input = 1.2
    rates = {}
    for beta in (1e-3, 1e-2, 1e-1):
        n = make_homeostatic_neuron(
            1,
            config=HomeostaticConfig(
                tau_theta_ms=tau, beta_theta=beta, tau_mem_ms=20.0
            ),
        )
        T = int(25 * tau)
        sp, _, _ = n.simulate(torch.full((T,), rate_input))
        rates[beta] = float(sp[-T // 10:].sum()) / (T / 10)

    # beta besar -> theta_ss besar -> neuron throttled -> rate turun
    assert rates[1e-3] > rates[1e-2] > rates[1e-1], rates


def test_matched_beta_times_tau_gives_matched_rate():
    """Konfirmasi langsung: produk sama -> laju sama, apa pun tau-nya."""
    tau_a, tau_b = 200.0, 1600.0
    rate_input = 1.2
    rates = []
    for tau in (tau_a, tau_b):
        beta = 0.02 / tau  # produk beta*tau dijaga = 0.02
        n = make_homeostatic_neuron(
            1,
            config=HomeostaticConfig(
                tau_theta_ms=tau, beta_theta=beta, tau_mem_ms=20.0
            ),
        )
        T = int(25 * tau)
        sp, _, _ = n.simulate(torch.full((T,), rate_input))
        rates.append(float(sp[-T // 10:].sum()) / (T / 10))

    assert rates[0] == pytest.approx(rates[1], rel=0.15), (
        f"produk beta*tau sama harus memberi laju sama: {rates}"
    )



def test_intrinsic_plasticity_caps_population_rate():
    """T1B.1.3 — total laju populasi dibatasi, meski antar neuron beda.

    Ini yang sebenarnya dilakukan theta: membatasi DAYA, bukan
    menyamakan rate. Inputnya sengaja tidak seragam supaya kita bisa
    melihat theta bekerja di atas perbedaan itu.

    PENTING: adaptive threshold BUKAN equalizer. Diuji terpisah di
    `test_theta_caps_firing_rate_not_equalize_it`. Yang menyamakan
    rate antar neuron adalah synaptic scaling (kolom) dan lateral
    inhibition.
    """
    n_exc = 60
    g = torch.Generator().manual_seed(0)
    currents = torch.rand(4000, n_exc, generator=g) * 3.0

    from rak.neuron import LIF

    plain = LIF(n_exc, tau_mem_ms=20.0)
    sp_plain, _, _ = plain.simulate(currents)
    rate_plain = float(sp_plain[-1000:].mean().item())

    adaptive = make_homeostatic_neuron(
        n_exc,
        config=HomeostaticConfig(
            tau_theta_ms=TIME_SCALES["realistic_hour"],
            beta_theta=1e-3,
            tau_mem_ms=20.0,
        ),
    )
    sp_ad, _, _ = adaptive.simulate(currents)
    rate_ad = float(sp_ad[-1000:].mean().item())

    assert rate_ad < rate_plain, (
        f"theta harus membatasi laju populasi: "
        f"dengan={rate_ad:.4f} tanpa={rate_plain:.4f}"
    )


def test_adaptive_never_dies_completely():
    """Homeostasis tidak boleh mematikan neuron sepenuhnya.

    Kalau theta melebihi jangkauan v, neuron jadi "dead" dan tidak bisa
    dipulihkan — itu yang harus dicegah.
    """
    n = make_homeostatic_neuron(
        1,
        config=HomeostaticConfig(
            tau_theta_ms=TIME_SCALES["realistic_hour"],
            beta_theta=0.05,
            tau_mem_ms=20.0,
        ),
    )
    # Input sangat besar selama lama -> neuron akan throttle
    sp, state, _ = n.simulate(torch.full((5000,), 100.0))
    assert torch.isfinite(state.theta).all()
    assert float(state.theta[0]) < 100.0, "theta tidak boleh meledak"

    # Setelah input berhenti, harus bisa fire lagi
    sp2, _, _ = n.simulate(torch.full((3000,), 0.0))
    assert float(sp2.sum()) == 0.0, "tanpa input tidak boleh fire"
    sp3, _, _ = n.simulate(torch.full((2000,), 5.0))
    assert float(sp3.sum()) > 0.0, "harus pulih setelah input kembali"
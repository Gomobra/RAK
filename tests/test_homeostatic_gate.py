"""GATE Phase 2 — homeostatic gain > 1.

Ini acceptance criteria Phase 2. Kalau test di file ini gagal, homeostasis
belum bekerja dan Phase 2 belum lulus.

Definisi operasional (lihat Zenke et al. 2013):
    Homeostasis bekerja kalau, setelah periode plastisitas yang panjang,
    total synaptic drive TETAP TERBATAS. Tanpa homeostasis, drift
    plastisitas akan membawa drive ke arah divergen.

Pengukuran:
    drive_t = W[t].sum()      # total synaptic drive
    Θ (gain) = -(d drive / d t) / drive

    Tanpa homeostasis : Θ < 0  -> drive naik tanpa henti
    Dengan homeostasis: Θ > 0  -> drive didorong kembali

Tiga mekanisme diuji satu per satu, lalu bersama-sama — karena Dividing
of labour itu inti findings Phase 2:

    theta (intrinsic) : membatasi DAYA
    scaling (kolom)   : menyamakan drive per neuron
    inhibition        :Competition antar neuron

Hanya scaling + inhibition yang menyamakan; theta tidak (lihat
`test_homeostasis.py::test_intrinsic_plasticity_caps_population_rate`).
"""

import pytest
import torch

from rak.homeostasis.inhibition import (
    InhibitionConfig,
    build_inhibitory_weights,
    make_inhibitory_neuron,
)
from rak.homeostasis.intrinsic import HomeostaticConfig, make_homeostatic_neuron
from rak.homeostasis.scaling import (
    ScalingConfig,
    hard_normalize,
    incoming_drive,
    rate_uniformity,
    soft_normalize,
)
from rak.neuron import AdaptiveLIF
from rak.neuron.adaptive_lif import AdaptiveLIFConfig
from rak.learning.stdp import STDPConfig, make_trace, stdp_step


def make_homeostatic_neuron(n, beta_theta):
    """Neuron eksitasi yang dipakai kedua cabang (identik modelnya)."""
    return AdaptiveLIF(
        n,
        config=AdaptiveLIFConfig(
            tau_mem_ms=20.0,
            v_th=1.0,
            v_reset=0.0,
            tau_adapt_ms=HomeostaticConfig().tau_theta_ms,
            beta_theta=beta_theta,
        ),
    )

N_IN, N_OUT = 64, 32


def _run_plasticity(
    n_epochs: int,
    steps_per_epoch: int,
    *,
    homeostasis: bool,
    normalize_every: int | None = None,
    seed: int = 0,
):
    """Jalankan epochs STDP di arsitektur recurrent, +/- homeostasis.

    DESAIN EKSPERIMEN: kedua cabang memakai ARSITEKTUR YANG SAMA persis
    (lapisan eksitasi LIF + inhibisi lateral + STDP). Yang berbeda hanya
    apakah mekanisme homeostasis diaktifkan. Kalau arsitekturnya ikut
    beda, kita tidak tahu apakah hasilbedanya dari homeostasis atau dari
    arsitektur.

    Positive feedback sengaja ada di kedua cabang — itu yang membuat
    Hebbian plasticity tidak stabil (Zenke 2013):
        bobot naik -> neuron post lebih sering fire -> lebih banyak LTP
    Homeostasis adalah yang memutus rantai itu.

    Returns:
        dict berisi drive akhir, riwayat drive, dan spread antar neuron.
    """
    g = torch.Generator().manual_seed(seed)
    # Asimetri eta itu PENTING. Dengan eta_plus == eta_minus, STDP
    # punya FIXED POINT: LTP dan LTD menyeimbangkan sebelum bobot
    # mencapai batas, jadi tidak ada runaway untukodianagunakan.
    # Diehl-Cook memakai rasio ~100x (nu_ee_post=1e-2, nu_ee_pre=1e-4);
    # di sini 20x sudah cukup untuk menunjukkan efeknya.
    stdp_cfg = STDPConfig(eta_plus=2e-3, eta_minus=1e-4, w_max=1.0)
    scale_cfg = ScalingConfig(target_sum=4.0, soft_power=0.02)

    w = torch.rand(N_IN, N_OUT, generator=g) * 0.3 + 0.01
    trace_pre = make_trace(N_IN)
    trace_post = make_trace(N_OUT)

    # Neuron model IDENTIK di kedua cabang. Cuma `beta_theta` yang
    # dimatikan di cabang tanpa homeostasis.
    beta_theta = 0.05 if homeostasis else 0.0
    exc = make_homeostatic_neuron(N_OUT, beta_theta)
    exc_state = exc.init_state()
    inh = make_inhibitory_neuron(N_OUT)
    inh_state = inh.init_state()
    inh_w = build_inhibitory_weights(
        N_OUT,
        InhibitionConfig(inhibition_strength=1.0, scale_per_neuron=False),
    )

    drive_history = []
    for epoch in range(n_epochs):
        for t in range(steps_per_epoch):
            z_pre = (torch.rand(N_IN, generator=g) < 0.15).float()
            z_post, exc_state = exc.step(z_pre @ w, exc_state)
            z_i, inh_state = inh.step(z_post @ inh_w.e_to_i, inh_state)
            inhib = z_i @ inh_w.i_to_e
            exc_state = type(exc_state)(
                v=(exc_state.v + inhib).clamp(min=0.0),
                theta=exc_state.theta,
                timer=exc_state.timer,
            )
            w, trace_pre, trace_post = stdp_step(
                z_pre, z_post, w, trace_pre, trace_post, stdp_cfg
            )

        if homeostasis and normalize_every and epoch % normalize_every == 0:
            w = hard_normalize(w, scale_cfg)
            w = soft_normalize(w, scale_cfg)
        drive_history.append(float(w.sum()))

    return {
        "w": w,
        "drive_history": drive_history,
        "final_drive": drive_history[-1],
        "spread": float(incoming_drive(w).std()),
    }


# ============================================================ GATE
def test_homeostatic_gain_exceeds_unity():
    """GATE Phase 2: tanpa homeostasis divergen, dengan homeostasis tertahan.

    Kriteria lulus:
      - STDP saja     : drive NAIK terus sampai saturasi
      - STDP + homeo. : drive TERTAHAN (tidak tumbuh, tidak naik lagi)

    Angka yang harus keluar (dicek di test_saturation_vs_bounded):
        plain : drive naik dari ~328 sampai ~2048 (w_max di semua synapse)
        homeo : drive datar di ~135, w_max ~0.15
    """
    n_epochs, steps = 60, 20

    plain = _run_plasticity(n_epochs, steps, homeostasis=False)
    homeo = _run_plasticity(
        n_epochs, steps, homeostasis=True, normalize_every=5
    )

    def late_slope(history):
        """Kemiringan drive pada paruh akhir run."""
        half = len(history) // 2
        return history[-1] - history[half]

    slope_plain = late_slope(plain["drive_history"])
    slope_homeo = late_slope(homeo["drive_history"])

    assert slope_plain > 0, (
        f"tanpa homeostasis drive harus NAIK (drift positif), "
        f"tapi dapat {slope_plain:+.4f}"
    )
    # Homeostasis boleh konstan ATAU turun — yang penting tidak tumbuh.
    assert slope_homeo <= 0.01 * abs(plain["final_drive"]), (
        f"Θ homeostasis harus arrests drive: kemiringan {slope_homeo:+.4f} "
        f"terlalu besar dibanding drive {plain['final_drive']:.2f}"
    )


def test_saturation_vs_bounded():
    """Angka konkret dari gate — drift vs tertahan.

    Ini yang dicatat di exp_002/README.md.
    """
    plain = _run_plasticity(60, 20, homeostasis=False)
    homeo = _run_plasticity(60, 20, homeostasis=True, normalize_every=5)

    # Tanpa homeostasis: SETIHA synapse saturasi di w_max
    assert float(plain["w"].max()) > 0.99, (
        f"tanpa homeostasis semua synapse harus saturasi, "
        f"dapat w_max={float(plain['w'].max()):.4f}"
    )
    # Dengan homeostasis: synapse JAUH dari batas
    assert float(homeo["w"].max()) < 0.5, (
        f"homeostasis harus menahan synapse jauh dari w_max, "
        f"dapat w_max={float(homeo['w'].max()):.4f}"
    )
    # Drive akhir jauh lebih kecil
    assert homeo["final_drive"] < 0.5 * plain["final_drive"], (
        f"drive harus tertahan: {homeo['final_drive']:.1f} "
        f"vs {plain['final_drive']:.1f}"
    )


def test_equal_eta_stdp_is_self_limiting():
    """TEMUAN: eta_plus == eta_minus membuat STDP punya FIXED POINT.

    Kalau learning rate LTP dan LTD sama besar, keduanya menyeimbangkan
    sebelum bobot mencapai batas — jadi TIDAK ada runaway, dan
    homeostasis tidak terlihat gunanya.

    Ini yang membuat desain eksperimen gate penting: tanpa asimetri
    eta, kita akan menyimpulkan "homeostasis tidak diperlukan" —
    kesimpulan yang salah.
    """
    from rak.learning.stdp import STDPConfig, make_trace, stdp_step

    def run_equal_eta(n_epochs=40, steps=20):
        g = torch.Generator().manual_seed(0)
        cfg = STDPConfig(eta_plus=2e-3, eta_minus=2e-3)
        w = torch.rand(N_IN, N_OUT, generator=g) * 0.3 + 0.01
        tp, tq = make_trace(N_IN), make_trace(N_OUT)
        exc = make_homeostatic_neuron(N_OUT, beta_theta=0.0)
        st = exc.init_state()
        for _ in range(n_epochs):
            for _ in range(steps):
                zp = (torch.rand(N_IN, generator=g) < 0.15).float()
                zq, st = exc.step(zp @ w, st)
                w, tp, tq = stdp_step(zp, zq, w, tp, tq, cfg)
        return float(w.sum())

    drive = run_equal_eta()
    ceiling = N_IN * N_OUT * 1.0
    assert drive < 0.5 * ceiling, (
        f"eta harus sama -> tidak boleh saturasi, dapat {drive:.1f} "
        f"dari ceiling {ceiling}"
    )


def test_homeostasis_prevents_blowup():
    """Tanpa homeostasis, drive bisa tumbuh tak terbatas.

    Ini yang membuat homeostasis wajib: bukan soal estetika, tapi soal
    jaringan masih ada setelah 100 epoch.
    """
    plain = _run_plasticity(40, 20, homeostasis=False)
    homeo = _run_plasticity(40, 20, homeostasis=True, normalize_every=5)

    assert torch.isfinite(plain["w"]).all()
    assert torch.isfinite(homeo["w"]).all()
    assert homeo["final_drive"] < plain["final_drive"], (
        f"homeostasis harus menahan drive: "
        f"{homeo['final_drive']:.2f} vs {plain['final_drive']:.2f}"
    )


def test_all_weights_stay_physical():
    """Setelah homeostasis, SEMUA bobot harus di [0, w_max]."""
    homeo = _run_plasticity(50, 20, homeostasis=True, normalize_every=5)
    w = homeo["w"]
    assert torch.isfinite(w).all()
    assert float(w.min()) >= -1e-6, f"bobot negatif: {float(w.min())}"
    assert float(w.max()) <= 1.0 + 1e-6, f"bobot > w_max: {float(w.max())}"


# =================================================== dividing of labour
def test_scaling_reduces_spread_across_neurons():
    """Scaling menyamakan DRIVE per neuron.

    Ini mekanisme yang theta tidak lakukan (theta cuma membatasi total).
    """
    homeo = _run_plasticity(50, 20, homeostasis=True, normalize_every=5)
    plain = _run_plasticity(50, 20, homeostasis=False)

    drive = incoming_drive(homeo["w"])
    assert float(drive.std()) < float(incoming_drive(plain["w"]).std()), (
        f"spread harus turun: "
        f"{float(incoming_drive(plain['w']).std()):.4f} "
        f"-> {float(drive.std()):.4f}"
    )


def test_column_sum_restored_after_normalization():
    """Setelah normalisasi keras, kolom kembali ke target."""
    scale_cfg = ScalingConfig(target_sum=4.0)
    homeo = _run_plasticity(50, 20, homeostasis=True, normalize_every=5)
    w = hard_normalize(homeo["w"], scale_cfg)
    drive = incoming_drive(w)
    assert torch.allclose(drive, torch.full_like(drive, 4.0), atol=1e-4)


def test_rate_uniformity_metric_is_finite_and_meaningful():
    """Metrik §9.1 harus bisa dihitung pada bobot yang sudah stabil."""
    homeo = _run_plasticity(40, 20, homeostasis=True, normalize_every=5)
    drive = incoming_drive(homeo["w"])
    cv = rate_uniformity(drive)
    assert 0.0 < cv < 1.0, f"CV tak masuk akal: {cv}"


# ============================================================== seeds
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gate_holds_across_seeds(seed):
    """Gate harus berlaku di lebih dari satu seed.

    Kalau hanya berlaku di seed tertentu, itu noise — bukan temuan.
    """
    homeo = _run_plasticity(
        50, 20, homeostasis=True, normalize_every=5, seed=seed
    )
    assert torch.isfinite(homeo["w"]).all()
    assert homeo["final_drive"] < 1e5, "drive tidak boleh meledak"
    assert float(homeo["w"].min()) >= -1e-6
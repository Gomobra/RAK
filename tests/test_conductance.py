r"""Test LIF conductance-based.

T1B.4 / Phase 3 — fondasi synapse conductance Diehl & Cook 2015.

Fokus test di sini SIFAT MATEMATIS, bukan angka keluaran implementasi.
Alasannya sudah terbukti berkali-kali di Phase 1 dan 2: test yang
memeriksa nilai balik dari rumus yang sama dengan yang ditulis
pengnya tidak akan menangkap apa pun. Yang menangkap bug adalah
properti yang harus benar untuk alasan yang BERBEDA dari cara
kodenya ditulis:

    v_ss adalah kombinasi konveks dari {v_rest, E_e, E_i}
        -> tidak bisa keluar dari rentang itu, untuk SEMUA ge, gi >= 0

Properti ini tidak evidente dari kode (`effective` cuma membagi dua
nilai), jadi testnya benar-benar menguji.

Sedikit lebih jarang: limit `ge -> besar` harus mendarat di `E_e`,
dan limit `gi -> ∞` di `E_i`. Itu fisika synapse, bukan aritmetika.
"""

import pytest
import torch

from rak.neuron.conductance import (
    ConductanceConfig,
    ConductanceLIF,
    ConductanceState,
)


def _cfg(**kw) -> ConductanceConfig:
    """Config Diehl & Cook default untuk neuron eksitasi."""
    base = dict(
        tau_mem_ms=100.0,
        tau_ge_ms=1.0,
        tau_gi_ms=2.0,
        v_rest=-65.0,
        e_syn_e=0.0,
        e_syn_i=-100.0,
        v_th=-52.0,
        v_reset=-65.0,
        refrac_ms=5.0,
        dt_ms=0.5,
    )
    base.update(kw)
    return ConductanceConfig(**base)


def _neuron(*, strict_stability: bool = True, **kw) -> ConductanceLIF:
    """Neuron dengan config Diehl-Cook.

    `strict_stability` dipisah dari parameter config karena itu flag
    modul, bukan parameter neuron.
    """
    return ConductanceLIF(4, config=_cfg(**kw), strict_stability=strict_stability)


def _drive(shape, value):
    return torch.full(shape, float(value))


# ================================================== PROPERTI UTAMA
def test_v_ss_stays_within_reversal_bounds():
    """`v_ss` tidak pernah keluar dari [min, max] potensial.

    Ini heart test file ini. Kalau `effective` salah, neuron akan punya
    steady state di luar jangkauan potensial reversal — dan itu
    akan muncul sebagai "akurasi MNIST aneh" yang sangat mahal dicari.
    """
    neuron = _neuron()
    lo, hi = _cfg().reversal_bounds

    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        ge = torch.rand(4, generator=gen) * 500.0
        gi = torch.rand(4, generator=gen) * 500.0
        _, v_ss = neuron.effective(ge, gi)
        assert float(v_ss.min()) >= lo - 1e-4, f"v_ss {float(v_ss.min())} < {lo}"
        assert float(v_ss.max()) <= hi + 1e-4, f"v_ss {float(v_ss.max())} > {hi}"


def test_v_ss_is_a_convex_combination():
    """Nilai `v_ss` harus sama dengan rata-rata berbobot.

    Membuktikan formula bukan sekadar "di dalam rentang" — kalau
    pembobotnya salah, nilainya masih di dalam rentang tapi salah
    secara kuantitatif.
    """
    neuron = _neuron()
    cfg = _cfg()

    ge = torch.tensor([0.0, 1.0, 3.0, 17.0])
    gi = torch.tensor([0.0, 0.0, 2.0, 5.0])
    _, v_ss = neuron.effective(ge, gi)

    g_total = 1.0 + ge + gi
    expected = (
        cfg.v_rest + ge * cfg.e_syn_e + gi * cfg.e_syn_i
    ) / g_total
    assert torch.allclose(v_ss, expected, atol=1e-5)


def test_no_input_means_rest_potential():
    """Tanpa conductance, v_ss harus persis v_rest."""
    neuron = _neuron()
    _, v_ss = neuron.effective(torch.zeros(4), torch.zeros(4))
    assert torch.allclose(
        v_ss, torch.full((4,), _cfg().v_rest), atol=1e-6
    )


# ================================================== limit / fisika
def test_excitatory_limit_is_e_syn_e():
    """`ge -> besar` harus mendarat di `E_e`, bukan di tak hingga.

    Kesalahan klasik: lupa membagi dengan G, jadi v_ss ikut naik tanpa
    batas. Itu kelihatan jelas di sini dan tidak terlihat di tes lain.
    """
    neuron = _neuron()
    cfg = _cfg()

    for ge in (1e2, 1e4, 1e6):
        _, v_ss = neuron.effective(
            torch.full((1,), ge), torch.zeros(1)
        )
        assert abs(float(v_ss[0]) - cfg.e_syn_e) < 1.0, (
            f"ge={ge}: v_ss={float(v_ss[0]):.4f}, "
            f"harus mendekati {cfg.e_syn_e}"
        )


def test_inhibitory_limit_is_e_syn_i():
    """`gi -> besar` harus mendarat di `E_i`."""
    neuron = _neuron()
    cfg = _cfg()

    for gi in (1e2, 1e4, 1e6):
        _, v_ss = neuron.effective(
            torch.zeros(1), torch.full((1,), gi)
        )
        assert abs(float(v_ss[0]) - cfg.e_syn_i) < 1.0, (
            f"gi={gi}: v_ss={float(v_ss[0]):.4f}, "
            f"harus mendekati {cfg.e_syn_i}"
        )


def test_excitatory_is_depolarizing():
    """Dari v_rest, masuknya spike eksitasi harus menaikkan v.

    E_e = 0 mV dan v_rest = -65 mV, jadi spike harus MENARIK v ke arah
    0 — kalau tidak,-ge tidak ada artinya.
    """
    neuron = _neuron()
    state = neuron.init_state()
    v_start = float(state.v[0])

    state = state.__class__(
        v=state.v,
        ge=state.ge,
        gi=state.gi,
        timer=state.timer,
    )
    _, after = neuron.step(_drive((4,), 1.0), torch.zeros(4), state)
    assert float(after.v[0]) > v_start, (
        f"v turun dari {v_start} ke {float(after.v[0])} — "
        "eksitasi jadi menghiperpolar, salah"
    )


def test_inhibitory_is_hyperpolarizing():
    """Dari v_rest, spike inhibitor harus menurunkan v."""
    neuron = _neuron()
    state = neuron.init_state()
    v_start = float(state.v[0])

    _, after = neuron.step(torch.zeros(4), _drive((4,), 1.0), state)
    assert float(after.v[0]) < v_start, (
        f"v naik dari {v_start} ke {float(after.v[0])} — "
        "inhibisi jadi mendepolar, salah"
    )


def test_more_excitatory_gives_more_depolarization():
    """Monoton: lebih banyak ge -> v_ss lebih tinggi."""
    neuron = _neuron()
    _, v_ss = neuron.effective(torch.tensor([0.0, 1.0, 5.0, 50.0]), torch.zeros(4))
    assert torch.all(v_ss[1:] > v_ss[:-1]), v_ss.tolist()


# ================================================== decay
def test_conductance_decay_is_exact_geometric():
    """Tanpa input baru, conductance harus `g * gamma^T` persis.

    Ini yang membedakan peluruhan eksponensial yang benar dari sekadar
    "nilai turun kira-kira".
    """
    cfg = _cfg()
    neuron = _neuron()
    state = neuron.init_state()
    state.ge.fill_(1.0)

    for step in range(1, 12):
        _, state = neuron.step(torch.zeros(4), torch.zeros(4), state)
        expected = cfg.gamma_ge**step
        assert float(state.ge[0]) == pytest.approx(expected, rel=1e-5), (
            f"t={step}: dapat {float(state.ge[0]):.8f}, "
            f"harus {expected:.8f}"
        )


def test_inhibitory_decays_slower_than_excitatory():
    """tau_gi (2 ms) > tau_ge (1 ms) -> inhibitory Conductance bertahan."""
    cfg = _cfg()
    assert cfg.gamma_gi > cfg.gamma_ge, (
        f"gamma_gi={cfg.gamma_gi} harus > gamma_ge={cfg.gamma_ge}"
    )


def test_decay_halves_at_expected_time():
    """Waktu paruh conductance harus ln(2) * tau, bukan tebakan."""
    import math

    cfg = _cfg()
    neuron = _neuron()
    state = neuron.init_state()
    state.ge.fill_(1.0)

    # Cari timestep saat ge <= 0.5
    half_at = None
    for t in range(1, 200):
        _, state = neuron.step(torch.zeros(4), torch.zeros(4), state)
        if float(state.ge[0]) <= 0.5:
            half_at = t
            break

    expected_ms = math.log(2) * cfg.tau_ge_ms
    assert half_at is not None
    # diskret, jadi toleransi satu timestep
    assert abs(half_at * cfg.dt_ms - expected_ms) <= cfg.dt_ms + 1e-6, (
        f"paruh pada {half_at * cfg.dt_ms}ms, harus ~{expected_ms:.3f}ms"
    )


def test_conductance_never_negative():
    """Peluruhan tidak boleh memunculkan conductance negatif.

    Ge negatif akan menarik v ke ARAH YANG SALAH lewat `numerator`,
    dan tidak akan ketahuan kalau Ge kecil.
    """
    neuron = _neuron()
    state = neuron.init_state()
    state.ge.fill_(1.0)
    state.gi.fill_(2.0)

    for _ in range(500):
        _, state = neuron.step(torch.zeros(4), torch.zeros(4), state)
    assert float(state.ge.min()) >= 0.0
    assert float(state.gi.min()) >= 0.0


# ================================================== stabilitas
def test_stability_limit_matches_formula():
    """`dt*G = tau` harus jadi batasnya, persis."""
    cfg = _cfg()
    bound = cfg.stable_ge_bound()
    g_at_bound = 1.0 + bound
    assert cfg.dt_ms * g_at_bound / cfg.tau_mem_ms == pytest.approx(1.0)


def test_unstable_conductance_raises():
    """Melewati batas harus DITOLAK, bukan diam-diam berosilasi."""
    neuron = _neuron(strict_stability=True)
    state = neuron.init_state()
    # PENTING: guard jalan SESUDAH decay. Dengan gamma_ge = 0.607,
    # mengisi ge = 300 menghasilkan 182 setelah meluruh — di bawah
    # batas 199, jadi tidak naik error. Yang perlu mengisi nilai
    # yang SETELAH decay masih di atas batas: 300/0.607 = 495.
    limit = _cfg().stable_ge_bound()
    state.ge.fill_(limit * 2.0)

    with pytest.raises(ValueError, match="tidak stabil"):
        neuron.step(torch.zeros(4), torch.zeros(4), state)


def test_error_message_explains_the_cause():
    """Pesan error harus menyebut(sum_k W = 78) — penyebab sebenarnya."""
    neuron = _neuron(strict_stability=True)
    state = neuron.init_state()
    state.ge.fill_(_cfg().stable_ge_bound() * 3.0)

    with pytest.raises(ValueError) as exc:
        neuron.step(torch.zeros(4), torch.zeros(4), state)
    msg = str(exc.value)
    assert "78" in msg, "sebut nilai kolom yang wajar"
    assert "tau" in msg, "sebut cara menaikkan stabilitas"


def test_just_below_limit_does_not_raise():
    neuron = _neuron(strict_stability=True)
    state = neuron.init_state()
    # Setelah decay harus tetap di bawah batas, tapi sudah jelas di atas
    # `tau/dt - 1`.
    limit = _cfg().stable_ge_bound()
    state.ge.fill_(limit * 1.5)
    _, after = neuron.step(torch.zeros(4), torch.zeros(4), state)
    assert float(after.ge.max()) < limit + 1.0, (
        f"harus di bawah batas {limit:.1f}, dapat {float(after.ge.max()):.1f}"
    )
    assert torch.isfinite(after.v).all()


def test_strict_disabled_skips_check():
    """Matikan guard harus benar-benar melewati — bukan Half jalan."""
    neuron = _neuron(strict_stability=False)
    state = neuron.init_state()
    state.ge.fill_(5000.0)
    _, after = neuron.step(torch.zeros(4), torch.zeros(4), state)
    assert after.v is not None  # tidak exception


# ================================================== numerik
def test_no_nan_with_extreme_but_valid_input():
    """Input besar tapi sah tidak boleh menghasilkan NaN/inf.

    `0 * inf = nan` sudah pernah membunuh neuron diam-diam di Phase 1.
    """
    neuron = _neuron()
    state = neuron.init_state()
    limit = _cfg().stable_ge_bound()
    for ge, gi in ((0.0, 0.0), (limit * 0.9, 0.0), (0.0, limit * 0.9),
                   (limit * 0.4, limit * 0.4)):
        state.ge.fill_(ge)
        state.gi.fill_(gi)
        _, out = neuron.step(_drive((4,), 10.0), _drive((4,), 10.0), state)
        assert torch.isfinite(out.v).all(), (ge, gi)
        assert torch.isfinite(out.ge).all()
        assert torch.isfinite(out.gi).all()


def test_state_fields_are_finite_at_init():
    state = _neuron().init_state()
    for name in ("v", "ge", "gi"):
        assert torch.isfinite(getattr(state, name)).all(), name


# ================================================== refraktori
def test_refractory_blocks_spikes():
    neuron = _neuron(refrac_ms=5.0)
    state = neuron.init_state()
    state.v.fill_(10.0)  # jauh di atas threshold
    spikes, state = neuron.step(_drive((4,), 5.0), torch.zeros(4), state)
    assert float(spikes.sum()) > 0, "harus bisa fire"

    # timestep berikutnya harus diblokir walau v tinggi
    state.v.fill_(10.0)
    spikes2, _ = neuron.step(_drive((4,), 5.0), torch.zeros(4), state)
    assert float(spikes2.sum()) == 0.0, "refraktori tidak bekerja"


def test_no_nan_after_first_spike():
    """Regresi: `0 * inf = nan` di masking timer."""
    neuron = _neuron(refrac_ms=5.0)
    state = neuron.init_state()
    state.v.fill_(10.0)
    _, state = neuron.step(_drive((4,), 5.0), torch.zeros(4), state)
    for _ in range(30):
        _, state = neuron.step(torch.zeros(4), torch.zeros(4), state)
    assert torch.isfinite(state.v).all(), "v jadi NaN setelah spike"


# ================================================== batch
def test_batched_matches_looped():
    """Batched (B, n) harus identik dengan loop per baris."""
    neuron = _neuron()
    gen = torch.Generator().manual_seed(11)
    drive = (torch.rand(6, 3, 4, generator=gen) * 2).clamp(max=1.0)

    batched = neuron.init_state((3, 4))
    batched_out = []
    for t in range(6):
        z, batched = neuron.step(drive[t], torch.zeros(3, 4), batched)
        batched_out.append(z)

    for row in range(3):
        single = neuron.init_state((4,))
        single_out = []
        for t in range(6):
            z, single = neuron.step(drive[t, row], torch.zeros(4), single)
            single_out.append(z)
        for t in range(6):
            assert torch.allclose(batched_out[t][row], single_out[t]), (
                f"baris {row} timestep {t} beda"
            )


def test_spikes_are_binary():
    neuron = _neuron()
    state = neuron.init_state()
    state.v.fill_(999.0)
    spikes, _ = neuron.step(_drive((4,), 50.0), torch.zeros(4), state)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}, spikes.unique()


# ================================================== konfigurasi
def test_config_rejects_bad_parameters():
    with pytest.raises(ValueError, match="tau_mem_ms"):
        _cfg(tau_mem_ms=0.0)
    with pytest.raises(ValueError, match="dt_ms"):
        _cfg(dt_ms=0.0)
    with pytest.raises(ValueError, match="v_reset"):
        _cfg(v_reset=-10.0, v_th=-52.0)
    with pytest.raises(ValueError, match="refrac_ms"):
        _cfg(refrac_ms=-1.0)


def test_diehl_cook_defaults_are_preserved():
    """Angka acuan harus persis seperti di kode aslinya."""
    cfg = _cfg()
    assert cfg.tau_mem_ms == 100.0
    assert cfg.tau_ge_ms == 1.0
    assert cfg.tau_gi_ms == 2.0
    assert cfg.v_rest == -65.0
    assert cfg.e_syn_e == 0.0
    assert cfg.e_syn_i == -100.0
    assert cfg.refrac_ms == 5.0
    assert cfg.dt_ms == 0.5


def test_inhibitory_population_uses_own_reversal():
    """Populasi I memakai -85 mV, bukan -100 mV."""
    cfg = _cfg(e_syn_i=-85.0)
    assert cfg.reversal_bounds == (-85.0, 0.0)
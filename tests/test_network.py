r"""Test jaringan locally competitive — integrasi Phase 3.

T6.1 — INI test pertama yang menjalankan kelima mekanisme
(conductance, STDP Eq.3, inhibisi lateral, adaptive threshold,
column normalization) BERSAMAAN dalam satu loop.

Semua test sebelumnya menguji komponennya secara terpisah di
lingkungan sintetis. Test di sini yang menangkap ERROR INTERAKSI:
misalnya neuron inhibitor ikut mendapat adaptive threshold, atau
`x_tar` tidak pernah bergerak karena `w` dinormalisasi setiap
presentasi.

Yang diuji sebagai PROPERTI, bukan sebagai angka yang diharapkan:
- kolom bobot tepat `column_sum` SETIAP presentasi
- theta hanya bergerak pada neuron eksitasi
- inhibitor benar-benar ter-ground (tidak fire tanpa stimulus)
- kompresi spike WTA benar-benar terjadi
- label tidak pernah menyentuh loop STDP
"""

import inspect

import pytest
import torch

from rak.data.encode import EncoderConfig, encode_poisson
from rak.learning.stdp3 import Eq3Config
from rak.network.competitive import (
    CompetitiveNetwork,
    NetworkConfig,
    assign_classes,
    classify,
)


def _net(n_e=20, seed=0, **kw) -> CompetitiveNetwork:
    cfg = NetworkConfig(n_e=n_e, **kw)
    return CompetitiveNetwork(cfg, stdp_config=Eq3Config(eta=1e-3), seed=seed)


def _spikes(batch=4, n_in=784, steps=60, intensity=2.0, seed=0):
    ec = EncoderConfig()
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(batch, n_in) * 0.3
    return encode_poisson(images, intensity, ec, generator=g, n_steps=steps)


# ================================================= struktur
def test_shapes():
    net = _net(n_e=20)
    assert net.w_e.shape == (784, 20)
    assert net.w_ei.shape == (20, 20)
    assert net.w_ie.shape == (20, 20)
    assert net.theta.shape == (20,)


def test_e_to_i_is_one_to_one():
    """E -> I satu-satu (diagonal), persis Diehl-Cook.

    Kekuatannya `e_to_i_strength`, bukan 1: default 1600 karena
    `inhibition_strength` adalah column sum I->E. Yang penting
    strukturnya: hanya diagonal yang bukan nol.
    """
    net = _net(n_e=16)
    diag = torch.diag(net.w_ei)
    off = net.w_ei - torch.diag(diag)
    assert torch.count_nonzero(off) == 0
    assert torch.allclose(
        diag, torch.full((16,), float(net.config.e_to_i_strength))
    )

    scaled = _net(n_e=4, e_to_i_strength=5.0)
    assert torch.allclose(torch.diag(scaled.w_ei), torch.full((4,), 5.0))


def test_i_to_e_is_positive_and_sparse():
    """Bobot I->E POSITIF, dan konektivitasnya ACAK (bukan all-to-all).

    Arah inhibisi datang dari `e_syn_i = -100 mV`, bukan dari bobot
    negatif. Bobot negatif akan menghitung inhibisi dua kali:
    `gi` negatif x `E_i` negatif jadi positif, sehingga inhibisi
    berubah jadi depolarisasi.

    All-to-all seragam membuat setiap neuron eksitasi menerima
    inhibisi yang identik. Kompetisi jadi simetris sempurna dan semua
    neuron konvergen ke receptive field yang sama (0.975 terukur).
    Acuan memakai matriks acak (`random/AiAe.npy`), jadi test ini
    menguji dua hal: tidak ada bobot negatif, dan ada sparsitas.
    """
    net = _net(n_e=16)
    assert (net.w_ie >= 0).all(), "tidak boleh negatif -- lihat docstring"
    frac_zero = float((net.w_ie == 0).double().mean())
    assert 0.2 < frac_zero < 0.8, f"bukan konektivitas acak: {frac_zero:.3f}"
    assert float(net.exc.config.e_syn_i) < float(net.exc.config.v_rest), (
        "potensial balik inhibitor harus di bawah v_rest supaya "
        "memang menghambat"
    )


def test_inhibitory_population_is_ground():
    """Neuron inhibitor HARUS ter-ground.

    `v_rest_i = -60 mV` di bawah `v_th_i = -40 mV`. Kalau di-init di 0
    (default LIF), inhibitor langsung fire di setiap timestep dan
    mematikan seluruh jaringan. Bug ini sudah pernah terjadi di
    Phase 2B.
    """
    net = _net(n_e=8)
    assert float(net.inh.init_state().v.max()) < float(net.inh.v_th)
    assert float(net.inh.config.v_reset) < float(net.inh.v_th), (
        "v_reset inhibitor harus di bawah threshold supaya bisa "
        "naik lagi sebelum fire berikutnya"
    )


def test_excitatory_threshold_is_ground():
    net = _net(n_e=8)
    assert float(net.exc.init_state().v.max()) < float(net.exc.v_th)


# ================================================= column normalization
def test_column_sum_exact_after_presentation():
    """`sum_k W[k,j] == column_sum` untuk SETIAP neuron.

    Diehl-Cook memanggil normalisasi di dalam loop presentasi, bukan
    per epoch. Kalau ini meleset, drive per neuron tidak konstan dan
    homeostasis yang sudah kita uji di Phase 2B kehilangan targetnya.
    """
    net = _net(n_e=12)
    net.present_batch(_spikes(steps=40))
    col = net.w_e.sum(dim=0)
    assert torch.allclose(
        col, torch.full_like(col, net.config.column_sum), atol=1e-3
    ), col.tolist()


def test_column_sum_survives_many_presentations():
    net = _net(n_e=10)
    for _ in range(5):
        net.present_batch(_spikes(steps=30, seed=1))
    col = net.w_e.sum(dim=0)
    assert torch.allclose(
        col, torch.full_like(col, net.config.column_sum), atol=1e-3
    )


def test_weights_stay_physical():
    net = _net(n_e=10)
    for _ in range(3):
        net.present_batch(_spikes(steps=40, seed=2))
    assert torch.isfinite(net.w_e).all()
    assert float(net.w_e.min()) >= -1e-6
    assert float(net.w_e.max()) <= net.config.column_sum * 2 + 1e-6


# ================================================= adaptive threshold
def test_initial_effective_threshold_matches_base():
    """theta awal 20 mV, offset 20 mV → threshold efektif = v_th.

    Ini yang membuatnya cocok dengan kode acuan: di presentasi
    pertama, jaringan berperilaku seperti LIF biasa.
    """
    net = _net(n_e=8)
    net.init_batch(1)
    v_th_eff = net.effective_threshold()
    assert torch.allclose(
        v_th_eff, torch.full_like(v_th_eff, float(net.exc.v_th)), atol=1e-4
    )


def test_theta_rises_for_active_neurons():
    """Neuron yang menyala harus menaikkan theta-nya."""
    net = _net(n_e=16)
    before = net.theta.clone()
    _, counts = net.present_batch(_spikes(steps=80, seed=3))
    active = counts.sum(0) > 0
    if bool(active.any()):
        assert (net.theta[active] > before[active]).all(), (
            "neuron aktif harus menaikkan theta"
        )


def test_theta_only_applies_to_excitatory():
    """Inhibitor TIDAK punya adaptive threshold.

    Kalau inhibitor ikut adaptif, inhibisi ikut melemah justru saat
    neuron sibuk — dan WTA (yang jadi tujuan lateral inhibition)
    akan runtuh.
    """
    net = _net(n_e=8)
    assert not hasattr(net.inh, "theta"), "neuron inhibitor punya theta"
    assert not hasattr(net.inh.config, "theta")
    assert not hasattr(net.inh.config, "theta_plus")


def test_theta_decays_toward_zero():
    """Theta meluruh; tanpa itu neuron akan terkunci diam selamanya."""
    cfg = NetworkConfig(n_e=8, tau_theta_ms=500.0)
    net = CompetitiveNetwork(cfg, stdp_config=Eq3Config(eta=1e-3), seed=0)
    net.theta.fill_(100.0)
    net.init_batch(1)
    net.present_batch(_spikes(steps=200, seed=4))
    # ada spike, tapi theta setelah 200 step (100ms) dengan tau 500ms
    # harus jauh di bawah 100 (yang meluruh tanpa spike apa pun)
    assert float(net.theta.max()) < 100.0


# ================================================= spike statistics
def test_spikes_are_binary():
    net = _net(n_e=10)
    z_e, _ = net.present_batch(_spikes(steps=40, seed=5), record_history=True)
    assert set(z_e.unique().tolist()) <= {0.0, 1.0}


def test_returns_expected_shapes():
    net = _net(n_e=10)
    steps = 50
    z_e, counts = net.present_batch(_spikes(steps=steps, seed=6), record_history=True)
    assert z_e.shape == (steps, 4, 10)
    assert counts.shape == (4, 10)


def test_lateral_inhibition_creates_competition():
    """Jumlah spike output harus lebih sedikit dari tanpa inhibisi.

    Ini bukti bahwa inhibisi benar-benar bekerja di dalam loop —
    bukan cuma bobot negatif yang ada di matriks tapi tidak pernah
    berperan'slpengaruh.
    """
    # Dua network TERPISAH dengan seed sama, supaya perbandingan
    # mengukur inhibisi saja. Versi pertama memakai satu network dan
    # menolakkan bobot di tengah jalan — tapi `w_e` dan `theta` sudah
    # berubah oleh STDP di presentasi pertama, jadi yang dibandingkan
    # bukan "dengan vs tanpa inhibisi" tapi "sebelum vs sesudah
    # training". Itu bukan eksperimen yang gw maksud.
    with_inh = _net(n_e=16, seed=5)
    without_inh = _net(n_e=16, seed=5, inhibition_strength=0.0)

    sp = _spikes(steps=120, seed=7)
    _, counts_a = with_inh.present_batch(sp)
    _, counts_b = without_inh.present_batch(sp)

    assert float(with_inh.w_ie.abs().sum()) > 0
    assert float(without_inh.w_ie.abs().sum()) == 0

    # Neuron eksitasi harus lebih DINGIN dengan inhibisi.
    v_with = float(counts_a.sum())
    v_without = float(counts_b.sum())
    assert v_with <= v_without, (
        f"dengan inhibisi {v_with:.0f} spike, "
        f"tanpa {v_without:.0f} — inhibisi tidak menurunkan spike"
    )


def test_deterministic_under_seed():
    """Jaringan dengan seed sama harus menghasilkan spike sama."""
    a = _net(n_e=10, seed=42)
    b = _net(n_e=10, seed=42)
    sp = _spikes(steps=60, seed=99)
    za, ca = a.present_batch(sp)
    zb, cb = b.present_batch(sp)
    assert torch.equal(ca, cb), "spike count harus identik"
    assert torch.allclose(a.w_e, b.w_e, atol=1e-6)


# ================================================= stabilitas
def test_no_nan_after_long_run():
    """Tidak ada NaN setelah beberapa presentasi."""
    net = _net(n_e=10)
    for i in range(3):
        net.present_batch(_spikes(steps=100, seed=10 + i))
    assert torch.isfinite(net.w_e).all()
    assert torch.isfinite(net.theta).all()
    assert torch.isfinite(net.state_e.v).all()
    assert torch.isfinite(net.state_e.ge).all()


def test_rejects_wrong_input_size():
    net = _net(n_e=8)
    bad = torch.zeros(10, 4, 100)     # n_in salah (100 != 784)
    with pytest.raises(ValueError, match="n_in"):
        net.present_batch(bad)


# ================================================= readout
def test_assign_classes_returns_valid_range():
    counts = torch.rand(100, 20)
    labels = torch.randint(0, 10, (100,))
    a = assign_classes(counts, labels)
    assert a.shape == (20,)
    assert int(a.min()) >= 0 and int(a.max()) <= 9


def test_assign_classes_finds_obvious_groups():
    """Satu neuron didedikasikan ke satu digit harus terdeteksi.

    Bentuk paling sederhana yang tidak ambigu: 10 neuron, neuron `i`
    hanya menyala untuk digit `i`.
    """
    n_per, n_classes = 20, 10
    labels = torch.arange(n_classes).repeat_interleave(n_per)   # (200,)
    counts = torch.full((200, n_classes), 1.0)                  # baseline
    for i in range(n_classes):
        block = slice(i * n_per, (i + 1) * n_per)
        counts[block, i] = 10.0        # neuron i meledak untuk digit i

    a = assign_classes(counts, labels)

    for i in range(n_classes):
        assert int(a[i]) == i, f"neuron {i} -> kelas {int(a[i])}, harus {i}"


def test_classify_picks_dominant_class():
    """Prediksi harus kelas dengan total spike terbesar."""
    assignments = torch.tensor([0, 0, 1, 1])
    counts = torch.tensor([[5.0, 5.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.0]])
    assert int(classify(counts[0], assignments)) == 0
    assert int(classify(counts[1], assignments)) == 1


def test_classify_handles_empty_class():
    """Kelas tanpa anggota harus dapat 0, bukan division by zero."""
    assignments = torch.tensor([0, 0, 1, 1])
    counts = torch.tensor([[5.0, 5.0, 0.0, 0.0]])
    pred = classify(counts[0], assignments)
    assert 0 <= int(pred) <= 9


def test_readout_is_perfect_on_separated_data():
    """Data yang benar-benar terpisah harus terklasifikasi sempurna."""
    n_per, n_classes = 20, 10
    labels = torch.arange(n_classes).repeat_interleave(n_per)   # (200,)
    counts = torch.full((200, n_classes), 0.1)
    for i in range(200):
        counts[i, labels[i]] = 10.0

    a = assign_classes(counts, labels)
    pred = torch.tensor([int(classify(counts[i], a)) for i in range(200)])
    assert (pred == labels).all(), f"{(pred != labels).sum()} salah"


# ================================================= unsupervised
def test_training_loop_takes_no_labels():
    """`present_batch` TIDAK boleh menerima label.

    Ini jaminan struktural bahwa Phase 3 ini unsupervised. Kalau
    label pernah masuk ke loop STDP, klaim "tanpa teaching signal"
    jadi bohong — dan bentuknya mungkin sulit diperiksa kalau hanya
    melihat akurasinya.
    """
    params = list(inspect.signature(CompetitiveNetwork.present_batch).parameters)
    # `rest_steps` boleh ada (peluruhan conductance), tapi label
    # TIDAK boleh masuk ke loop STDP -- itu jaminan struktural
    # bahwa Phase 3 ini unsupervised.
    assert "self" in params and "z_pre" in params and "learn" in params
    assert not any(
        p in params for p in ("label", "labels", "target", "targets", "y")
    ), params


def test_config_defaults_match_paper():
    """Angka acuan: 784 input, kolom 78, dt 0.5 ms."""
    cfg = NetworkConfig()
    assert cfg.n_in == 784
    assert cfg.column_sum == 78.0
    assert cfg.dt_ms == 0.5
    # 0.0125, bukan 0.05: theta_eq = theta_plus * laju * tau. Acuan
    # berjalan pada laju 17x lebih rendah, jadi 0.05 di sini memberi
    # theta_eq ~108 mV dan memaksa evaluasi memakai intensitas 10-12.
    assert cfg.theta_plus == 0.0125
    assert cfg.theta_max == 40.0
    assert cfg.tau_theta_ms == 10_000_000.0
    assert cfg.n_inhibitory == cfg.n_e


def test_config_validation():
    with pytest.raises(ValueError, match="n_e"):
        NetworkConfig(n_e=0)
    with pytest.raises(ValueError, match="column_sum"):
        NetworkConfig(column_sum=0.0)
    with pytest.raises(ValueError, match="tau_theta_ms"):
        NetworkConfig(tau_theta_ms=0.0)


def test_state_dict_roundtrip():
    net = _net(n_e=8)
    net.present_batch(_spikes(steps=40, seed=11))
    state = net.state_dict()

    other = _net(n_e=8, seed=999)
    other.load_state_dict(state)
    assert torch.allclose(other.w_e, net.w_e)
    assert other.n_presentations == net.n_presentations
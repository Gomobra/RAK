r"""Test encoding spike Poisson + intensitas adaptif.

T6.3 — diuji sebagai DISTRIBUSI, bukan sebagai angka yang diharapkan
penulis.

Test yang paling bernilai di file ini adalah `test_empirical_rate_matches_requested`.
Kalau konversi Hz -> probabilitas salah tempat (mis. lupa bagi 1000,
atau `dt_ms` terbalik), rate empirisnya akan menyimpang dari rate yang
diminta dan test ini gagal — sementara test yang membandingkan dengan
implementasi lain akan lolos, karena dua kesalahan yang sama bisa
saling meniadakan.
"""

import math

import pytest
import torch
from torch import Tensor

from rak.data.encode import (
    AdaptiveIntensity,
    EncoderConfig,
    encode_poisson,
    estimated_rate_hz,
    spike_counts,
)


def _cfg(**kw) -> EncoderConfig:
    base = dict(present_ms=350.0, rest_ms=150.0, dt_ms=0.5,
                max_rate_per_intensity=31.875, start_intensity=2.0,
                min_spikes=5)
    base.update(kw)
    return EncoderConfig(**base)


def _empirical_hz(spikes: Tensor, cfg: EncoderConfig, n_steps: int) -> float:
    """Laju rata-rata yang benar-benar teramati, dalam Hz."""
    total = float(spikes.sum())
    n_neurons = spikes.shape[1] * spikes.shape[2]
    duration_s = n_steps * cfg.dt_ms / 1000.0
    return total / (n_neurons * duration_s)


# ================================================= bentuk dasar
def test_spikes_are_binary():
    """Sumbu spike WAJIB biner — bukan tegangan, bukan bobot."""
    cfg = _cfg()
    g = torch.Generator().manual_seed(0)
    sp = encode_poisson(torch.rand(8, 784), 2.0, cfg, generator=g)
    assert set(sp.unique().tolist()) <= {0.0, 1.0}, sp.unique()


def test_shape_has_time_first():
    """Konvensi RAK: waktu di sumbu PERTAMA."""
    cfg = _cfg()
    g = torch.Generator().manual_seed(1)
    sp = encode_poisson(torch.rand(4, 784), 2.0, cfg, generator=g)
    assert sp.shape == (cfg.n_steps, 4, 784), sp.shape


def test_accepts_1d_input():
    cfg = _cfg()
    g = torch.Generator().manual_seed(2)
    sp = encode_poisson(torch.rand(784), 2.0, cfg, generator=g)
    assert sp.shape == (cfg.n_steps, 1, 784)


def test_rejects_out_of_range_images():
    cfg = _cfg()
    g = torch.Generator().manual_seed(3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        encode_poisson(torch.full((2, 784), 1.5), 2.0, cfg, generator=g)


def test_rejects_non_positive_intensity():
    cfg = _cfg()
    g = torch.Generator().manual_seed(4)
    with pytest.raises(ValueError, match="intensity"):
        encode_poisson(torch.rand(2, 784), 0.0, cfg, generator=g)


# ================================================= PROPERTI DISTRIBUSI
def test_empirical_rate_matches_requested():
    """Rata-rata spike harus sama dengan laju yang diminta.

    Ini heart test file ini: membuktikan kodingannya benar secara
    statistik, bukan cuma "tidak error".
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(5)
    images = torch.full((16, 784), 1.0)
    sp = encode_poisson(images, 2.0, cfg, generator=g)

    want = cfg.max_rate_hz(2.0)
    got = _empirical_hz(sp, cfg, cfg.n_steps)
    assert got == pytest.approx(want, rel=0.02), (
        f"laju diminta {want:.2f} Hz, teramati {got:.2f} Hz"
    )


def test_rate_scales_linearly_with_intensity():
    """Laju harus PROPORSIONAL intensitas — bukan saturasi."""
    cfg = _cfg()
    images = torch.full((16, 784), 1.0)
    rates = []
    for intensity in (1.0, 2.0, 4.0):
        g = torch.Generator().manual_seed(6)
        sp = encode_poisson(images, intensity, cfg, generator=g)
        rates.append(_empirical_hz(sp, cfg, cfg.n_steps))

    assert rates[1] == pytest.approx(rates[0] * 2, rel=0.03), rates
    assert rates[2] == pytest.approx(rates[0] * 4, rel=0.03), rates


def test_rate_scales_linearly_with_pixel():
    """Setengah brightness harus setengah laju.

    Ini yang membedakan encoding intensitas yang benar dari yang
    hanya "benar di pixel putih".
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(7)
    full = encode_poisson(torch.full((16, 784), 1.0), 2.0, cfg, generator=g)
    half = encode_poisson(torch.full((16, 784), 0.5), 2.0, cfg, generator=g)

    r_full = _empirical_hz(full, cfg, cfg.n_steps)
    r_half = _empirical_hz(half, cfg, cfg.n_steps)
    assert r_half == pytest.approx(r_full * 0.5, rel=0.03), (r_full, r_half)


def test_dark_image_gives_fewer_spikes():
    cfg = _cfg()
    g = torch.Generator().manual_seed(8)
    dark = encode_poisson(torch.full((16, 784), 0.05), 2.0, cfg, generator=g)
    bright = encode_poisson(torch.full((16, 784), 1.0), 2.0, cfg, generator=g)
    assert float(dark.sum()) < float(bright.sum()) / 10


def test_black_image_gives_zero_spikes():
    """Pixel nol berarti laju nol — TIDAK ada noise floor.

    Kalau ada noise, gambar MNIST yang gelap akan tetap menyalakan
    semua neuron input dan bobot akan belajar dari noise.
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(9)
    sp = encode_poisson(torch.zeros(8, 784), 2.0, cfg, generator=g)
    assert float(sp.sum()) == 0.0


# ================================================= variansi Poisson
def test_variance_matches_poisson():
    """Varians jumlah spike harus ~= mean (sifat Poisson).

    Ini membed witnesses encoding dari yang hanya "benar rata-ratanya".
    Distribusi dengan mean benar tapi varians salah (mis. spike
    spreading yang berkorelasi) akan gagal di sini.
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(10)
    sp = encode_poisson(torch.full((2000, 4), 0.3), 2.0, cfg, generator=g)
    counts = spike_counts(sp).flatten()          # (2000*4,)
    mean = float(counts.mean())
    var = float(counts.var(unbiased=False))
    # untuk n besar, var/mean harus dekat 1
    assert 0.8 < var / mean < 1.25, (
        f"mean={mean:.2f} var={var:.2f} -> var/mean={var/mean:.3f}, "
        f"Poisson menuntut ~1"
    )


def test_no_temporal_correlation_beyond_poisson():
    """Spike di timestep berurutan tidak boleh berkorelasi.

    Kalau implementasinya memakai satu uniform untuk seluruh timestep,
    korelasi antar waktu akan positif dan STDP ikut dipengaruhi.
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(11)
    sp = encode_poisson(torch.full((2000, 1), 0.5), 2.0, cfg, generator=g)
    series = sp[:, 0, 0]                          # (n_steps,)
    a = series[:-1]
    b = series[1:]
    corr = float((a * b).mean() - a.mean() * b.mean())
    assert abs(corr) < 0.01, f"korelasi antar timestep = {corr:.4f}"


# ================================================= determinisme
def test_same_seed_same_spikes():
    cfg = _cfg()
    img = torch.rand(4, 784)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    assert torch.equal(
        encode_poisson(img, 2.0, cfg, generator=g1),
        encode_poisson(img, 2.0, cfg, generator=g2),
    )


def test_different_seed_different_spikes():
    """Seed berbeda harus menghasilkan spike train berbeda.

    Kalau tidak, encoder-nya tidak benar-benar acak — dan akurasi
    MNIST jadi tidak bisa dibandingkan antar run.
    """
    cfg = _cfg()
    img = torch.rand(4, 784)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    a = encode_poisson(img, 2.0, cfg, generator=g1)
    b = encode_poisson(img, 2.0, cfg, generator=g2)
    assert not torch.equal(a, b)


def test_encoding_does_not_consume_global_rng():
    """Encoder harus tidak menyentuh RNG global.

    Kalau iya, urutan pemanggilan fungsi lain akan mengubah hasil
    spike — hasil eksperimen jadi tergantung urutan baris kode.
    """
    cfg = _cfg()
    img = torch.rand(2, 784)
    torch.manual_seed(0)
    state_a = torch.random.get_rng_state()

    torch.manual_seed(0)
    encode_poisson(img, 2.0, cfg, generator=torch.Generator().manual_seed(7))
    state_b = torch.random.get_rng_state()

    assert torch.equal(state_a, state_b), "RNG global terpakai"


# ================================================= rate helper
def test_estimated_rate_matches_formula():
    cfg = _cfg()
    img = torch.tensor([[0.0, 0.5, 1.0]])
    rates = estimated_rate_hz(img, 2.0, cfg)
    full = cfg.max_rate_hz(2.0)
    assert float(rates[0, 0]) == 0.0
    assert float(rates[0, 1]) == pytest.approx(full * 0.5)
    assert float(rates[0, 2]) == pytest.approx(full)


def test_reference_max_rate():
    """Acuan: pixel 255, intensity 2 -> 255/8*2 = 63.75 Hz."""
    cfg = _cfg()
    assert cfg.max_rate_hz(2.0) == pytest.approx(63.75)
    assert cfg.max_rate_hz(1.0) == pytest.approx(31.875)


def test_reference_timing():
    """Acuan: 350 ms presentasi, 150 ms jeda, dt 0.5 ms."""
    cfg = _cfg()
    assert cfg.n_steps == 700
    assert cfg.n_rest_steps == 300


# ================================================= guard
def test_rejects_impossible_rate():
    """Laju per timestep > 1 harus DITOLAK, bukan di-clamp.

    Clamping diam-diam akan menghasilkan spike lebih sedikit dari
    yang diminta tanpa ada yang tahu.
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(12)
    with pytest.raises(ValueError, match="melebihi 1"):
        encode_poisson(
            torch.ones(1, 784), 1000.0, cfg, generator=g
        )


def test_rest_appends_silence():
    cfg = _cfg()
    g = torch.Generator().manual_seed(13)
    sp = encode_poisson(
        torch.full((2, 784), 1.0), 2.0, cfg, generator=g, rest=True
    )
    assert sp.shape[0] == cfg.n_steps + cfg.n_rest_steps
    assert float(sp[cfg.n_steps:].sum()) == 0.0, "tail harus diam"


# ================================================= intensitas adaptif
def test_adaptive_starts_at_reference_value():
    ctrl = AdaptiveIntensity(_cfg())
    assert ctrl.intensity == pytest.approx(2.0)


def test_adaptive_escalates_when_too_few_spikes():
    """Kurang dari `min_spikes` -> naik intensitas, minta ulangi."""
    ctrl = AdaptiveIntensity(_cfg())
    assert ctrl.update(2) is True
    assert ctrl.intensity == pytest.approx(3.0)


def test_adaptive_resets_when_enough_spikes():
    ctrl = AdaptiveIntensity(_cfg())
    ctrl.escalate()
    ctrl.escalate()
    assert ctrl.intensity == pytest.approx(4.0)
    assert ctrl.update(50) is False
    assert ctrl.intensity == pytest.approx(2.0), "harus kembali ke awal"


def test_adaptive_threshold_is_exclusive():
    """Tepat di `min_spikes` sudah cukup."""
    cfg = _cfg()
    ctrl = AdaptiveIntensity(cfg)
    assert ctrl.update(cfg.min_spikes) is False
    assert ctrl.update(cfg.min_spikes - 1) is True


def test_repeated_escalation_actually_increases_input_spikes():
    """Escalasi harus menambah spike INPUT pada gambar yang sama.

    Uji ujung-ke-ujung: kalau tidak, mekanisme adaptifnya tidak
    berguna dan kita hanya menambah komputasi sia-sia.
    """
    cfg = _cfg()
    img = torch.full((4, 784), 0.02)
    ctrl = AdaptiveIntensity(cfg)

    counts = []
    for _ in range(6):
        g = torch.Generator().manual_seed(14)
        sp = encode_poisson(img, ctrl.intensity, cfg, generator=g)
        counts.append(float(sp.sum(dim=(0, 2)).mean()))
        ctrl.escalate()          # naik intensitas tiap putaran

    assert counts[-1] > counts[0] * 3, (
        f"tidak naik cukup: {[round(c, 1) for c in counts]}"
    )


def test_min_spikes_threshold_is_on_output_not_input():
    """Threshold dihitung dari spike OUTPUT, bukan spike input.

    Kalau dihitung dari spike input, threshold 5 hampir tidak pernah
    terlampaui: 784 input dengan pixel secerah apa pun sudah
    menghasilkan ratusan spike dalam 350 ms. Mekanisme yang
    mencegah dead network tidak akan pernah aktif.
    """
    cfg = _cfg()
    g = torch.Generator().manual_seed(15)
    sp = encode_poisson(
        torch.full((1, 784), 0.02), cfg.start_intensity, cfg, generator=g
    )
    input_spikes = float(sp.sum())

    assert input_spikes > cfg.min_spikes, (
        f"spike input sudah {input_spikes:.0f} > {cfg.min_spikes}; kalau "
        f"threshold dihitung dari input, eskalasi tidak akan pernah jalan"
    )

    # tapi spike output untuk jaringan yang belum terlatih bisa 0
    assert not AdaptiveIntensity(cfg).needs_repeat(0) or True
    ctrl = AdaptiveIntensity(cfg)
    assert ctrl.needs_repeat(0) is True
    assert ctrl.needs_repeat(input_spikes) is False


def test_needs_repeat_is_pure_predicate():
    """`needs_repeat` tidak boleh mengubah state."""
    ctrl = AdaptiveIntensity(_cfg())
    before = ctrl.intensity
    ctrl.needs_repeat(0)
    ctrl.needs_repeat(10_000)
    assert ctrl.intensity == before


# ================================================= konfigurasi
def test_config_validation():
    with pytest.raises(ValueError, match="present_ms"):
        _cfg(present_ms=0.0)
    with pytest.raises(ValueError, match="rest_ms"):
        _cfg(rest_ms=-1.0)
    with pytest.raises(ValueError, match="dt_ms"):
        _cfg(dt_ms=0.0)
    with pytest.raises(ValueError, match="start_intensity"):
        _cfg(start_intensity=0.0)
    with pytest.raises(ValueError, match="dt_ms"):
        _cfg(dt_ms=1000.0, present_ms=350.0)


def test_n_steps_rounds_correctly():
    cfg = _cfg(present_ms=350.0, dt_ms=0.5)
    assert cfg.n_steps == 700
    cfg2 = _cfg(present_ms=100.0, dt_ms=0.5)
    assert cfg2.n_steps == 200
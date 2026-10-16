"""T1B.2 — Synaptic scaling (hard column normalization + soft Oja-style)."""

import pytest
import torch

from rak.homeostasis.scaling import (
    ScalingConfig,
    hard_normalize,
    incoming_drive,
    rate_uniformity,
    soft_normalize,
    target_sum_for,
)


# ------------------------------------------------------- definisi shape
def test_incoming_drive_is_column_sum():
    """Drive masuk per neuron postsynaptic = jumlah kolom."""
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (n_in=2, n_out=2)
    drive = incoming_drive(w)
    assert drive.shape == (2,)
    assert float(drive[0]) == pytest.approx(4.0)  # 1 + 3 (kolom 0)
    assert float(drive[1]) == pytest.approx(6.0)  # 2 + 4 (kolom 1)


def test_incoming_drive_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        incoming_drive(torch.zeros(5))


# ------------------------------------------------------------- hard
def test_hard_normalize_hits_target_exactly():
    cfg = ScalingConfig(target_sum=78.0)
    w = torch.rand(784, 100, generator=torch.Generator().manual_seed(0)) * 0.3
    wn = hard_normalize(w, cfg)
    drive = incoming_drive(wn)
    assert torch.allclose(drive, torch.full_like(drive, 78.0), atol=1e-4)


def test_hard_normalize_preserves_relative_ratios():
    """Normalisasi kolom harus TIDAK mengubah rasio antar synapse.

    Kalau rasio berubah, informasi yang disimpan neuron ikut berubah —
    itu bukan homeostasis, itu penghapusan memori.
    """
    w = torch.tensor([[1.0, 5.0], [3.0, 5.0]])  # kolom 0: rasio 1:3
    wn = hard_normalize(w, ScalingConfig(target_sum=1.0))
    ratio_before = float(w[1, 0] / w[0, 0])
    ratio_after = float(wn[1, 0] / wn[0, 0])
    assert ratio_after == pytest.approx(ratio_before, rel=1e-5)


def test_hard_normalize_is_idempotent():
    cfg = ScalingConfig(target_sum=78.0)
    w = torch.rand(100, 20, generator=torch.Generator().manual_seed(1)) * 0.5
    once = hard_normalize(w, cfg)
    twice = hard_normalize(once, cfg)
    assert torch.allclose(once, twice, atol=1e-6)


def test_hard_normalize_handles_zero_column():
    """Kolom nol harus tidak menghasilkan NaN/inf."""
    w = torch.zeros(10, 4)
    w[:, 0] = 1.0  # hanya kolom 0 nonzero
    wn = hard_normalize(w, ScalingConfig(target_sum=10.0))
    assert torch.isfinite(wn).all()
    assert float(wn[:, 1].abs().sum()) == 0.0


def test_hard_normalize_inplace_option():
    w = torch.rand(20, 5, generator=torch.Generator().manual_seed(2)) * 0.4
    out = hard_normalize(w, ScalingConfig(target_sum=8.0), inplace=True)
    assert out is w, "inplace=True harus mengembalikan objek yang sama"
    assert float(incoming_drive(w)[0]) == pytest.approx(8.0, abs=1e-5)


def test_hard_normalize_does_not_mutate_by_default():
    w = torch.rand(20, 5, generator=torch.Generator().manual_seed(3)) * 0.4
    w0 = w.clone()
    hard_normalize(w, ScalingConfig(target_sum=8.0))
    assert torch.equal(w, w0)


# --------------------------------------------------------- proporsional
def test_target_sum_proportional_scales_with_n_in():
    cfg = ScalingConfig(proportional=True, target_sum_per_input=0.0995)
    assert target_sum_for(784, cfg) == pytest.approx(78.0, abs=0.01)
    assert target_sum_for(100, cfg) == pytest.approx(9.95, abs=0.01)


def test_target_sum_fixed_ignores_n_in():
    cfg = ScalingConfig(proportional=False, target_sum=78.0)
    assert target_sum_for(100, cfg) == 78.0
    assert target_sum_for(10000, cfg) == 78.0


def test_diehl_cook_value_is_default():
    """Nilai default harus persis 78 — angka dari source aslinya."""
    assert ScalingConfig().target_sum == 78.0


# ------------------------------------------------------------- soft
def test_soft_normalize_reduces_magnitude():
    w = torch.rand(50, 10, generator=torch.Generator().manual_seed(4)) * 2.0 + 1.0
    wn = soft_normalize(w, ScalingConfig(soft_power=0.1))
    assert float(wn.mean()) < float(w.mean())


def test_soft_normalize_equalizes_columns():
    """Soft normalization harus mengurangi perbedaan antar neuron."""
    w = torch.rand(60, 10, generator=torch.Generator().manual_seed(5)) * 3.0 + 0.1
    drive_before = incoming_drive(w)
    wn = soft_normalize(w, ScalingConfig(soft_power=0.05))
    drive_after = incoming_drive(wn)

    spread_before = float(drive_before.std() / drive_before.mean())
    spread_after = float(drive_after.std() / drive_after.mean())
    assert spread_after < spread_before, (
        f"CV harus turun: {spread_before:.3f} -> {spread_after:.3f}"
    )


def test_soft_power_controls_strength():
    """p kecil = koreksi kecil dan halus."""
    w = torch.rand(40, 8, generator=torch.Generator().manual_seed(6)) * 2.0 + 0.5
    weak = soft_normalize(w, ScalingConfig(soft_power=0.001))
    strong = soft_normalize(w, ScalingConfig(soft_power=0.5))
    assert float(strong.mean()) < float(weak.mean())


def test_soft_normalize_moves_columns_toward_target():
    """Kolom di atas target menyusut, kolom di bawah target membesar.

    INI yang benar untuk homeostasis. Versi lama membagi dengan mean
    tanpa target, yang justru memperbesar neuron yang kurang aktif.
    """
    w = torch.tensor([[3.0, 0.2], [3.0, 0.2], [3.0, 0.2], [3.0, 0.2]])
    wn = soft_normalize(w, ScalingConfig(soft_power=1.0, soft_target=1.0))
    hi_in = float(w[:, 0].abs().mean())
    lo_in = float(w[:, 1].abs().mean())
    hi_out = float(wn[:, 0].abs().mean())
    lo_out = float(wn[:, 1].abs().mean())

    assert hi_out < hi_in, f"kolom over-target harus menyusut: {hi_in} -> {hi_out}"
    assert lo_out > lo_in, f"kolom under-target harus membesar: {lo_in} -> {lo_out}"


def test_soft_normalize_converges_columns_to_target():
    w = torch.rand(60, 10, generator=torch.Generator().manual_seed(7)) * 2.0 + 0.2
    cfg = ScalingConfig(soft_power=1.0, soft_target=1.0)
    for _ in range(200):
        w = soft_normalize(w, cfg)
    means = w.abs().mean(dim=0)
    assert torch.allclose(means, torch.ones_like(means), atol=0.1), (
        f"semua kolom harus dekati target, dapat {means.min():.3f}..{means.max():.3f}"
    )


def test_soft_normalize_with_zero_power_is_noop():
    """p = 0 tidak melakukan apa-apa — knob yang aman."""
    w = torch.rand(20, 5, generator=torch.Generator().manual_seed(11)) * 3.0
    wn = soft_normalize(w, ScalingConfig(soft_power=0.0))
    assert torch.allclose(wn, w)


def test_soft_normalize_preserves_sign_structure():
    """Bobot negatif harus tetap negatif relatif."""
    w = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    wn = soft_normalize(w, ScalingConfig(soft_power=0.1))
    assert float(wn[0, 0]) > 0 and float(wn[1, 0]) > 0
    assert float(wn[0, 1]) < 0 and float(wn[1, 1]) < 0


def test_combined_hard_then_soft_is_stable():
    """Kombinasi hard + soft (urutan Phase 2B) harus tetap terbatas."""
    w = torch.rand(200, 20, generator=torch.Generator().manual_seed(8)) * 0.5
    cfg_hard = ScalingConfig(target_sum=20.0)
    cfg_soft = ScalingConfig(soft_power=0.01)
    for _ in range(50):
        w = hard_normalize(w, cfg_hard)
        w = soft_normalize(w, cfg_soft)
    assert torch.isfinite(w).all()
    drive = incoming_drive(w)
    assert float(drive.min()) > 0.0, "tidak boleh ada neuron tanpa drive"


# ------------------------------------------------------- rate uniformity
def test_rate_uniformity_zero_for_identical():
    assert rate_uniformity(torch.ones(10)) == pytest.approx(0.0, abs=1e-9)


def test_rate_uniformity_positive_for_varied():
    rates = torch.tensor([1.0, 2.0, 3.0, 10.0])
    assert rate_uniformity(rates) > 0.5


def test_rate_uniformity_infinite_for_all_zero():
    assert rate_uniformity(torch.zeros(5)) == float("inf")


def test_rate_uniformity_handles_empty():
    assert rate_uniformity(torch.zeros(0)) == float("inf")
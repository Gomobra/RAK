"""Guard konfigurasi STDP + integritas figure window.

T2.2 — test yang lebih dulu tidak ada, dan keputusannya ternyata
menyelamatkan figure 03 dari berbohong.

Konteks: figure "STDP window variants" mengklaim menunjukkan
perbedaan varian multiplicative pada tiga nilai bobot. Kenyataannya
ketiga kurvanya identik persis, karena CLI mengirim
`STDPConfig(mu=1.0)` TANPA `algorithm=`, sehingga default `"additive"`
yang dipakai dan `mu` dibuang diam-diam.

Test di sini menutup dua jalan yang membuat bug itu bisa lolos:
  1. guard di `STDPConfig` — menyetel `mu` pada varian yang tidak
     membacanya sekarang error, bukan no-op senyap;
  2. `panel_configs` — figure per-algoritma sekarang benar-benar
     memakai algoritma yang di-label-kan.
"""

import pytest
import torch

from rak.learning.stdp import ALGORITHMS, STDPConfig, A_plus, stdp_window
from rak.neuron.viz import panel_configs


# =============================================================== guard
@pytest.mark.parametrize("algo", ["additive", "additive_step", "multiplicative_relu"])
def test_mu_on_non_pow_variant_raises(algo):
    """Menyetel `mu` di varian yang tidak membacanya harus DITOLAK.

    Kesalahan ini yang membuat figure 03 menampilkan `additive` tiga
    kali sambil mengklaim multiplicative. Kalau ini hanya
    warning, bug yang sama akan bisa terulang.
    """
    with pytest.raises(ValueError, match="multiplicative_pow"):
        STDPConfig(algorithm=algo, mu=1.0)


def test_mu_on_pow_is_allowed():
    cfg = STDPConfig(algorithm="multiplicative_pow", mu=2.5)
    assert cfg.mu == 2.5


def test_mu_zero_allowed_on_every_algorithm():
    """mu=0 berarti "tidak aktif" — sah untuk semua varian."""
    for algo in ALGORITHMS:
        assert STDPConfig(algorithm=algo, mu=0.0).algorithm == algo


def test_guard_error_names_both_options():
    """Pesan error harus menyarankan jalan keluar yang benar."""
    with pytest.raises(ValueError) as exc:
        STDPConfig(algorithm="additive", mu=1.0)
    msg = str(exc.value)
    assert "multiplicative_pow" in msg
    assert "multiplicative_relu" in msg


def test_omitting_algorithm_is_still_allowed():
    """Jangan sampai guard membuat pemakaian umum jadi ribet."""
    assert STDPConfig().algorithm == "additive"


# ========================================================== panel_configs
def test_panel_configs_covers_every_algorithm():
    panels = panel_configs(STDPConfig())
    assert set(panels) == set(ALGORITHMS)


def test_panel_configs_sets_the_right_algorithm():
    """INI test yang menangkap bug label: tiap panel harus punya
    `algorithm` yang sama dengan namanya."""
    for name, cfg in panel_configs(STDPConfig()).items():
        assert cfg.algorithm == name, (
            f"panel {name!r} dikonfigurasi sebagai {cfg.algorithm!r}"
        )


def test_panel_configs_clears_mu_for_non_pow():
    """`mu` hanya boleh hidup di panel multiplicative_pow.

    Kalau diteruskan, `STDPConfig` akan menolak — jadi ini juga
    menjaga `plot_stdp_window` tidak meledak.
    """
    panels = panel_configs(STDPConfig(algorithm="multiplicative_pow", mu=2.0))
    assert panels["multiplicative_pow"].mu == 2.0
    for algo in ("additive", "additive_step", "multiplicative_relu"):
        assert panels[algo].mu == 0.0, algo


def test_panel_configs_actually_changes_the_curves():
    """Panel yang berbeda harus menghasilkan window yang BERBEDA.

    Kalau ini gagal, figure per-algoritma tidak membandingkan apa pun
    — persis kondisi figure 03 sebelum diperbaiki.
    """
    panels = panel_configs(STDPConfig(mu=1.0, algorithm="multiplicative_pow"))
    windows = {}
    for name, cfg in panels.items():
        _, win = stdp_window(1.0, cfg, w_ref=0.5, max_ms=60.0)
        windows[name] = win

    # additive dan additive_step boleh identik (keduanya tidak
    # bergantung w), tapi TIDAK boleh sama dengan multiplicative_pow
    assert not torch.allclose(
        windows["additive"], windows["multiplicative_pow"]
    ), "panel additive dan multiplicative_pow tidak boleh sama"

    # dan multiplicative_pow dengan mu=1 harus sama dengan relu —
    # itu konsekuensi matematis (x^1 == x), bukan kebetulan
    assert torch.allclose(
        windows["multiplicative_pow"], windows["multiplicative_relu"]
    ), "mu=1 harus identik dengan multiplicative_relu"


def test_panels_differ_in_weight_dependence():
    """Yang membedakan multiplicative dari additive: reaksi ke bobot.

    Diuji pada `A_plus` supaya tidak bergantung pada sampling window.
    """
    w = torch.tensor([[0.2], [0.8]])
    panels = panel_configs(STDPConfig(mu=1.0, algorithm="multiplicative_pow"))

    additive_span = (
        float(A_plus(w, panels["additive"]).max())
        - float(A_plus(w, panels["additive"]).min())
    )
    mult_span = (
        float(A_plus(w, panels["multiplicative_pow"]).max())
        - float(A_plus(w, panels["multiplicative_pow"]).min())
    )

    assert additive_span == pytest.approx(0.0, abs=1e-9)
    assert mult_span > 1e-4, f"multiplicative harus bereaksi ke w, dapat {mult_span}"
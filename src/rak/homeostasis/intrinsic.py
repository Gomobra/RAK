"""Intrinsic plasticity (adaptive threshold) sebagai mekanisme homeostasis.

T1B.1 — neuron's yang terlalu aktif menaikkan threshold-nya sendiri,
sehingga butuh lebih banyak input untuk fire lagi.

Mekanisme yang sudah ada di `AdaptiveLIF`:
```
tiap spike     : theta += beta_theta
tiap timestep  : theta = theta_0 + alpha_theta * (theta - theta_0)
                 dengan alpha_theta = 1 - dt / tau_theta
```

Modul ini TIDAK mengimplementasikan theta untuk kedua kalinya. Yang
dilakukan: menentukan konfigurasi yang SAH menurut aturan Zenke 2013,
menyediakan mode cepat/lambat, dan memvalidasi invarian.

Aturan Zenke et al. 2013 (Plos Comput Biol) yang jadi acuan:

    tau_theta harus reacting pada skala detik atau lebih panjang.
    Kalau terlalu lambat, stabilitas background network hilang.
    Sinaptic weight berubah pada detik–menit; homeostasis bereaksi
    pada jam–hari. Kingscales inilah yang bikin homeostasis "inert"
    terhadap fluktuasi cepat — persis yang diinginkan.

Diehl & Cook 2015 memakai `tc_theta = 1e7 ms` ≈ 2.8 jam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rak.neuron.adaptive_lif import AdaptiveLIF, AdaptiveLIFConfig

__all__ = [
    "HomeostaticConfig",
    "TIME_SCALES",
    "make_homeostatic_neuron",
    "validate_timescales",
]


@dataclass(frozen=True)
class HomeostaticConfig:
    """Konfigurasi homeostasis intrinsik.

    `tau_theta_ms` adalah parameter kunci. Band yang benar menurut
    Zenke 2013:

        < 1 detik    : GAGAL — homeostasis bereaksi secepat plasticity,
                       jadi tidak sempat menyeimbangkan drift
        1 s – 10 mnt : batas bawah yang disebut paper (demo)
        ~ jam        : nilai yang dipakai Diehl-Cook untuk MNIST
    """

    tau_mem_ms: float = 20.0
    tau_theta_ms: float = 3_600_000.0  # 1 jam
    beta_theta: float = 0.05
    theta_0: float = 0.0
    v_th: float = 1.0
    v_reset: float = 0.0
    dt_ms: float = 1.0
    refrac_ms: float = 0.0

    alpha_theta: float = field(init=False)

    def __post_init__(self) -> None:
        if self.tau_theta_ms <= 0:
            raise ValueError(
                f"tau_theta_ms harus > 0, dapat {self.tau_theta_ms}"
            )
        object.__setattr__(
            self, "alpha_theta", 1.0 - self.dt_ms / self.tau_theta_ms
        )


#: Mode siap pakai untuk `HomeostaticConfig`.
#:
#: `fast` hanya untuk unit test — dipakai untdemonstrasi mekanisme
#: dalam hitungan detik. BUKAN untuk eksperimen sungguhan.
#:
#: `realistic` mengikuti Diehl-Cook (1 jam) — ini yang dipakai Phase 3.
TIME_SCALES = {
    "demo_seconds": 5_000.0,
    "zenke_minimum": 60_000.0,
    "paper_minute": 600_000.0,
    "realistic_hour": 3_600_000.0,
    "diehl_cook_2.8h": 10_000_000.0,
}


def validate_timescales(cfg: HomeostaticConfig) -> list[str]:
    """Periksa invarian homeostasis terhadap aturan Zenke 2013.

    Returns:
        Daftar peringatan (bukan error — mekanisme tetap jalan, tapi
        condong tidak stabil).
    """
    warnings: list[str] = []

    ratio = cfg.tau_theta_ms / cfg.tau_mem_ms
    if cfg.tau_theta_ms < 1_000.0:
        warnings.append(
            f"tau_theta = {cfg.tau_theta_ms} ms (< 1 detik). Zenke 2013: "
            "homeostasis harus bereaksi pada skala detik atau lebih "
            "panjang, kalau tidak tidak sempat menyeimbangkan drift."
        )
    if ratio < 100.0:
        warnings.append(
            f"tau_theta/tau_mem = {ratio:.0f}x (< 100x). Homeostasis "
            "harus jauh lebih lambat dari dinamika membran, kalau tidak "
            "dia ikut berosilasi bersama plastisitas."
        )
    if cfg.tau_theta_ms < cfg.tau_mem_ms:
        warnings.append(
            "tau_theta lebih kecil dari tau_mem — secara fisika tidak "
            "bisa, dan jadi tanda konfigurasi salah."
        )
    return warnings


def make_homeostatic_neuron(
    n_neurons: int,
    config: HomeostaticConfig | None = None,
    *,
    scale: str | None = None,
    strict: bool = False,
) -> AdaptiveLIF:
    """Buat `AdaptiveLIF` dengan konfigurasi homeostasis yang tervalidasi.

    Args:
        n_neurons: jumlah neuron
        config: konfigurasi lengkap. Kalau None, pakai default (1 jam).
        scale: nama dari `TIME_SCALES` untuk cepat set `tau_theta_ms`.
            Diabaikan kalau `cfg` diberikan.
        strict: kalau True, peringatan Zenke jadi exception.

    Returns:
        `AdaptiveLIF` yang sudah dikonfigurasi.

    Raises:
        ValueError: kalau `strict=True` dan aturan Zenke dilanggar.
    """
    if config is None:
        if scale is None:
            config = HomeostaticConfig()
        else:
            if scale not in TIME_SCALES:
                raise ValueError(
                    f"scale harus salah satu dari {sorted(TIME_SCALES)}, "
                    f"dapat {scale!r}"
                )
            config = HomeostaticConfig(tau_theta_ms=TIME_SCALES[scale])

    issues = validate_timescales(config)
    if issues:
        msg = "Homeostatic config melanggar Zenke 2013:\n  - " + "\n  - ".join(
            issues
        )
        if strict:
            raise ValueError(msg)
        import warnings as _warnings

        _warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return AdaptiveLIF(
        n_neurons,
        config=AdaptiveLIFConfig(
            tau_mem_ms=config.tau_mem_ms,
            v_th=config.v_th,
            v_reset=config.v_reset,
            tau_adapt_ms=config.tau_theta_ms,
            theta_0=config.theta_0,
            beta_theta=config.beta_theta,
            dt_ms=config.dt_ms,
            refrac_ms=config.refrac_ms,
        ),
    )
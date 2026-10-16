"""Synaptic scaling — hard (column normalization) + soft (Oja-style).

T1B.2 — menjaga total synaptic drive per neuron tetap terkendali.

Dua mekanisme, keduanya dipakai bersamaan:

**HARD — column normalization ke konstanta (Diehl & Cook 2015):**
    col_sums = W.sum(dim=0)
    W *= target_sum / (col_sums + eps)

Dipanggil PERIODIK (setiap N presentasi contoh), bukan tiap timestep.
`target_sum = 78.0` untuk 784 input, persis seperti di
`Diehl&Cook_spiking_MNIST.py`.

**SOFT — Oja-style kontinu:**
    W /= (mean_k(|W_kj|) + eps)^p

Lebih halus, per-synapse, tapi butuh tuning `p`.

Definisi shape (dipin di ADR / PRD C7):
    W[k, j]  k = input neuron, j = neuron POSTSYNAPTIC
    Kolom (`sum over k`) = total drive masuk ke neuron postsynaptic j.

Jadi "per-neuron" di sini berarti per-KOLOM. Penulisan "per-row" yang
ada di PRD v1.1 ambigu — sudah dibetulkan supaya tidak ada dua
interpretasi.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "ScalingConfig",
    "target_sum_for",
    "hard_normalize",
    "soft_normalize",
    "incoming_drive",
    "rate_uniformity",
]


@dataclass(frozen=True)
class ScalingConfig:
    """Parameter synaptic scaling."""

    #: Target `sum_k W[k, j]` untuk tiap neuron postsynaptic.
    #: 78.0 = nilai Diehl-Cook untuk 784 input.
    target_sum: float = 78.0

    #: Kalau True, `target_sum` dihitung ulang dari `n_in` memakai
    #: `target_sum_per_input`. Dipakai kalau arsitektur berubah ukuran.
    proportional: bool = False
    target_sum_per_input: float = 0.0995  # 78 / 784

    #: Eksponen untuk soft normalization. p kecil = koreksi halus.
    soft_power: float = 0.01
    #: Mean absolut per kolom yang dikejar `soft_normalize`.
    soft_target: float = 1.0
    eps: float = 1e-8


def target_sum_for(n_in: int, cfg: ScalingConfig | None = None) -> float:
    """Target jumlah bobot untuk `n_in` input.

    Kalau `cfg.proportional`, diskalakan dari `n_in`. Kalau tidak,
    `cfg.target_sum` dipakai apa adanya.
    """
    cfg = cfg or ScalingConfig()
    if cfg.proportional:
        return cfg.target_sum_per_input * n_in
    return cfg.target_sum


def incoming_drive(w: Tensor) -> Tensor:
    """Total drive masuk per neuron postsynaptic, shape (n_out,).

    `W[k, j]` = synapse dari input k ke output j, jadi
    `W.sum(dim=0)` menjumlahkan semua synapse yang masuk ke neuron j.
    """
    if w.ndim != 2:
        raise ValueError(f"W harus 2D (n_in, n_out), dapat {w.ndim}D")
    return w.sum(dim=0)


def hard_normalize(
    w: Tensor, cfg: ScalingConfig | None = None, inplace: bool = False
) -> Tensor:
    """Normalisasi KOLOM ke `target_sum` (Diehl-Cook).

    Setiap synapse dalam satu neuron postsynaptic diskalakan dengan
    faktor yang sama, jadi RASIO antar-synapse di neuron itu tidak
    berubah. Yang dinormalisasi hanya total drive-nya.

    Args:
        w: (n_in, n_out)
        cfg: konfigurasi. Kalau `proportional=True`, target mengikuti n_in.
        inplace: kalau True, `w` dimodifikasi langsung.

    Returns:
        W yang sudah dinormalisasi (tensor baru kalau `inplace=False`).
    """
    cfg = cfg or ScalingConfig()
    target = target_sum_for(w.shape[0], cfg)
    col_sums = w.sum(dim=0)  # (n_out,)
    scale = target / (col_sums.abs() + cfg.eps)

    if inplace:
        w.mul_(scale.unsqueeze(0))
        return w
    return w * scale.unsqueeze(0)


def soft_normalize(
    w: Tensor, cfg: ScalingConfig | None = None, inplace: bool = False
) -> Tensor:
    """Soft normalization per-neuron menuju TARGET, kontinu.

        W[k, j] *= (target / (mean_k |W[k, j]| + eps))^p

    Bentuknya soft karena p kecil: koreksi mendekati linear dan
   gary, bukan hard snapping seperti `hard_normalize`.

    Kenapa ada `target` dan bukan sekadar `W /= mean^p`: membagi dengan
    mean tanpa target itu SALAH untuk homeostasis. Kolom yang
    mean-nya 0.8 akan dengan sendirinya jadi 1/0.8 = 1.25 kali
    lebih besar — justru memperbesar neuron yang memang kurang
    aktif. Homeostasis harus mendekatkan ke target, bukan selalu
    membagi.

    Efek yang diharapkan: kolom dengan mean di atas target menyusut,
    kolom di bawah target membesar, dan spread antar neuron mengecil.
    """
    cfg = cfg or ScalingConfig()
    mean_abs = w.abs().mean(dim=0)  # (n_out,)
    factor = (cfg.soft_target / (mean_abs + cfg.eps)).clamp(
        min=0.0, max=1e6
    ).pow(cfg.soft_power)

    if inplace:
        w.mul_(factor.unsqueeze(0))
        return w
    return w * factor.unsqueeze(0)


def rate_uniformity(
    rates: Tensor, eps: float = 1e-9
) -> float:
    """Ukuran ketidak-merataan firing rate: std / mean.

    Homeostasis yang bekerja harus membuat nilai ini MENURUN seiring
    waktu. Ini metrik turnaround untuk Fase 2B (lihat PRD §9.1).

    Args:
        rates: firing rate per neuron, (n_neurons,)

    Returns:
        Koefisien variasi. 0 = semua neuron sama rate (ideal).
    """
    if rates.numel() == 0:
        return float("inf")
    mean = float(rates.mean().item())
    if abs(mean) < eps:
        return float("inf")
    return float(rates.std().item()) / abs(mean)
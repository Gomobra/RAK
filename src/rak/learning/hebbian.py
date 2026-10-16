"""Aturan belajar Hebbian dan variasinya.

T2.1 — tiga aturan, semuanya berbasis aktivitas lokal saja (pre dan post),
tanpa error signal global.

Rumus diambil persis dari Nengo (`nengo/learning_rules.py`), yang sudah
battle-tested:

    Hebbian : dW_ij = eta * x_i * x_j
    BCM     : dW_ij = kappa * a_j * (a_j - theta_j) * a_i
    Oja     : dW_ij = kappa * (a_i * a_j - beta * a_j^2 * W_ij)

Kenapa BCM dan Oja ada kalau plain Hebbian cukup:

**Hebbian murni meledak.** dW_ij = eta * x_i * x_j tidak punya mekanisme
penahan. Setiap synapse yang sering spike akan tumbuh tanpa batas, semua
neuron menjadi identik, dan jaringan kehilangan kapasitas membedakan input.

**Oja** menambahkan suku penahan `- beta * a_j^2 * W_ij`. Kalau W sudah
besar, suku negatif mulai mendominasi — ada titik seimbang. Secara
matematis Oja adalah normalisasi L2 dari Hebbian.

**BCM** menambahkan ambang `theta_j` yang bergerak: kalau neuron post
selalu di atas ambang, bobotnya justru turun. Ini yang membuat neuron
bisa specialize pada stimulus tertentu.

Konvensi shape: `W[k, j]`, `x[k]`, `a[j]`, `theta[j]` — sama seperti
`SpikeLayer.weight` (n_in, n_out).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "HebbianConfig",
    "BCMConfig",
    "OjaConfig",
    "BCMState",
    "hebbian_update",
    "bcm_theta",
    "bcm_update",
    "oja_update",
]


# ---------------------------------------------------------------- Hebbian
@dataclass(frozen=True)
class HebbianConfig:
    """Plain Hebbian."""

    eta: float = 1e-4


def hebbian_update(
    w: Tensor,
    x_pre: Tensor,
    x_post: Tensor,
    config: HebbianConfig | None = None,
) -> Tensor:
    """dW_ij = eta * x_pre_i * x_post_j.

    Args:
        w: bobot saat ini, (n_in, n_out). Tidak dimodifikasi.
        x_pre: aktivitas pre, (n_in,)
        x_post: aktivitas post, (n_out,)

    Returns:
        Tensor dW dengan shape sama seperti w.

    Catatan: pakai `x` (trace) BUKAN spike biner. Kalau pakai spike biner,
    synapse hanya berubah saat kedua neuron fire bersamaan di timestep yang
    SAMA — di diskrit hampir tidak pernah terjadi, jadi synapse nyaris
    tidak pernah belajar.
    """
    cfg = config or HebbianConfig()
    return cfg.eta * torch.outer(x_pre, x_post)


# -------------------------------------------------------------------- BCM
@dataclass(frozen=True)
class BCMConfig:
    """Bienenstock-Cooper-Munroe (Bienenstock, Cooper & Munro 1982)."""

    kappa: float = 1e-3
    #: Time constant rate detector (ms). Mengatur seberapa cepat theta_j
    #: mengikuti aktivitas rata-rata. Mengikuti aturan homeostasis: harus
    #: jauh lebih lambat dari dinamika neuron.
    tau_theta_ms: float = 1e4
    #: Proteksi theta dari bawah — kalau neuron diam, theta cenderung
    #: menuju nilai ini, bukan minus tak hingga.
    theta_min: float = 0.0


@dataclass
class BCMState:
    """State BCM: ambang bergerak per neuron postsynaptic."""

    theta: Tensor


def bcm_theta(
    theta: Tensor,
    post_activity: Tensor,
    config: BCMConfig | None = None,
) -> Tensor:
    """Ambang bergerak theta_j: rata-rata kuadrat aktivitas post.

    `theta[j] <- theta[j] + (a[j]^2 - theta[j]) / tau_detector`

    Dipakai low-pass, bukan rata-rata berjalan sederhana, karena BCM butuh
    ambang yang bereaksi LAMBAT — kalau cepat, dia cuma meniru aktivitas
    instan dan jadi tidak ada gunanya.
    """
    cfg = config or BCMConfig()
    inv_tau = 1.0 / max(cfg.tau_theta_ms, 1.0)
    updated = theta + inv_tau * (post_activity.pow(2) - theta)
    return updated.clamp(min=cfg.theta_min)


def bcm_update(
    w: Tensor,
    a_pre: Tensor,
    a_post: Tensor,
    theta: Tensor,
    config: BCMConfig | None = None,
) -> Tensor:
    """dW_ij = kappa * a_j * (a_j - theta_j) * a_i.

    Sinyal LTD kalau a_j < theta_j, LTP kalau a_j > theta_j.

    Returns:
        Tensor dW dengan shape sama seperti w.
    """
    cfg = config or BCMConfig()
    sign = a_post - theta  # < 0 LTD, > 0 LTP
    d_post = a_post * sign
    return cfg.kappa * torch.outer(a_pre, d_post)


# -------------------------------------------------------------------- Oja
@dataclass(frozen=True)
class OjaConfig:
    """Oja rule — Hebbian + normalisasi L2."""

    kappa: float = 1e-3
    beta: float = 1.0


def oja_update(
    w: Tensor,
    a_pre: Tensor,
    a_post: Tensor,
    config: OjaConfig | None = None,
) -> Tensor:
    """dW_ij = kappa * (a_i * a_j - beta * a_j^2 * W_ij).

    Suku kedua `- beta * a_j^2 * W_ij` adalah gaya gesek yang oppose
    pertumbuhan synapse yang sudah besar. Kalau W_ij sudah besar dan
    neuron post aktif, gaya gesek melebihi gaya Hebbian sehingga bobot
    turun — ada titik seimbang yang tidak blown up.
    """
    cfg = config or OjaConfig()
    hebb = torch.outer(a_pre, a_post)
    decay = a_post.pow(2).unsqueeze(0) * w  # (1, n_out) * (n_in, n_out)
    return cfg.kappa * (hebb - cfg.beta * decay)
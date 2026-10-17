r"""Variant 3-trace — bentuk yang dipakai kode acuan `peter-u-diehl/stdp-mnist`.

T6.5 — ini bentuk rule yanggw implementasikan pertama di Phase 2,
lalu SAYA SINGKIRKAN dengan alasan "itu cuma varian ketiga, rule
utamanya Eq. 3". Keputusan itu kemungkinan salah, dan run MNIST
pertama membuktikannya:

    korelasi antar kolom bobot : 0.94 (naik monoton ke arah 1)
    neuron per kelas            : [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]

BENTUKNYA
----------
    dpre/dt   = -pre   / tau_pre
    dpost1/dt = -post1 / tau_post1
    dpost2/dt = -post2 / tau_post2

    spike pre  :  w -= nu_pre  * post1
    spike post :  w += nu_post * pre * post2before
                  post1 = 1 ;  post2 = 1

APA YANG BIKIN BERBEDA DARI EQ. 3
---------------------------------
Eq. 3 (yang gw pakai sebelumnya):

    dw = eta · (x_pre − x_tar) · (w_max − w)^mu

TIDAK ADA jejak postsinaptik. Satu-satunya pembeda antar neuron adalah
`x_tar[j]` — dan itu cuma adaptasi level laju, bukan urutan waktu.

Variant ini punya suku `post2before`, dan itu seluruh inti mekanisme
yang gw lewatkan:

    LTP = nu_post · x_pre · post2before

`post2before` adalah jejak aktivitas postsinaptik yang BARU SAJA
terjadi (tau 40 ms). Kalau neuron itu baru fire sekarang saja,
`post2before` ~= 0 dan LTP-nya nol. Kalau neuron itu fire berulang,
`post2before` besar dan LTP-nya kuat.

Artinya: **neuron yang hanya sesekali menyALA tidak bisa
memperkuat synapse-nya.** Bobot hanya tumbuh pada neuron yang
konsisten menang — persis perilaku yang diminta jaringan, dan tepat
yang memecah simetri yang menghancurkan Eq. 3.

Konsekuensi yang harus diterima: **LTP pertama selalu nol**,
karena `post2` mulai dari 0. Neuron perlu fire minimal dua kali
sebelum bisa mulai belajar. Itu bukan bug, itu konsekuensi desain.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["Variant3Config", "Variant3State", "variant3_step", "make_variant3_state"]


@dataclass(frozen=True)
class Variant3Config:
    """Parameter variant 3-trace.

    Default persis `peter-u-diehl/stdp-mnist`.
    """

    tau_pre_ms: float = 20.0
    tau_post1_ms: float = 20.0
    tau_post2_ms: float = 40.0

    #: Learning rate LTD (dipicu spike presinaptik).
    nu_pre: float = 1e-4
    #: Learning rate LTP (dipicu spike postsinaptik).
    nu_post: float = 1e-2

    w_min: float = 0.0
    w_max: float = 1.0
    dt_ms: float = 0.5

    def __post_init__(self) -> None:
        for name in ("tau_pre_ms", "tau_post1_ms", "tau_post2_ms", "dt_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} harus > 0")
        if self.nu_pre < 0 or self.nu_post < 0:
            raise ValueError("nu harus >= 0")
        if self.w_min >= self.w_max:
            raise ValueError("w_min harus < w_max")

    def _gamma(self, tau: float) -> float:
        return float(torch.exp(torch.tensor(-self.dt_ms / tau)).item())

    @property
    def gamma_pre(self) -> float:
        return self._gamma(self.tau_pre_ms)

    @property
    def gamma_post1(self) -> float:
        return self._gamma(self.tau_post1_ms)

    @property
    def gamma_post2(self) -> float:
        return self._gamma(self.tau_post2_ms)

    @property
    def learning_rate_ratio(self) -> float:
        if self.nu_pre == 0:
            return float("inf")
        return self.nu_post / self.nu_pre


@dataclass
class Variant3State:
    """Tiga jejak, per-neuron."""

    pre: Tensor    # (n_in,)
    post1: Tensor  # (n_out,)
    post2: Tensor  # (n_out,)


def make_variant3_state(n_pre: int, n_post: int) -> Variant3State:
    return Variant3State(
        pre=torch.zeros(n_pre, dtype=torch.float32),
        post1=torch.zeros(n_post, dtype=torch.float32),
        post2=torch.zeros(n_post, dtype=torch.float32),
    )


def variant3_step(
    z_pre: Tensor,
    z_post: Tensor,
    w: Tensor,
    state: Variant3State,
    cfg: Variant3Config,
) -> tuple[Tensor, Variant3State]:
    """Satu timestep variant 3-trace.

    Args:
        z_pre: spike presinaptik, (n_in,) atau (B, n_in).
        z_post: spike postsinaptik, (n_out,) atau (B, n_out).
        w: bobot, (n_in, n_out) atau (B, n_in, n_out).
        state: jejak sebelum timestep ini.
        cfg: konfigurasi.

    Returns:
        (w_baru, state_baru)

    URUTAN — decay, update, bump:
    Spike di timestep ini tidak boleh mengerosi kreditnya sendiri
    (aturan ADR-0004, berlaku sama untuk bentuk ini).
    """
    if w.dim() == 3:
        _, n_in, n_out = w.shape
    else:
        n_in, n_out = w.shape

    # --- 1. decay ------------------------------------------------------
    pre_d = cfg.gamma_pre * state.pre.expand(z_pre.shape)
    post1_d = cfg.gamma_post1 * state.post1.expand(z_post.shape)
    post2_d = cfg.gamma_post2 * state.post2.expand(z_post.shape)

    fired = z_post > 0
    if not bool(fired.any()):
        return w, Variant3State(pre=pre_d + z_pre,
                                post1=post1_d + z_post,
                                post2=post2_d + z_post)

    w_ref = w if w.dim() == 2 else w[0]

    # --- 2. LTD, dipicu spike presinaptik -----------------------------
    # Aturan ini memang punya suku LTD berbasis spike pre — itulah
    # bedanya dari Eq. 3. Di sini itu disengaja.
    dw_ltd = torch.einsum("bi,j->bij", z_pre * cfg.nu_pre, post1_d)

    # --- 3. LTP, dipicu spike postsinaptik, DIKUNCI post2before ------
    # `post2_d` sudah bernilai SEBELUM di-bump — itu `post2before`.
    # Kalau nilainya 0 (neuron belum pernah menyala), LTP-nya 0.
    dw_ltp = torch.einsum("bi,bj->bij", pre_d * cfg.nu_post, post2_d) * fired

    dw = dw_ltp - dw_ltd

    # Rata-rata batch: `w` dipakai BERSAMA seluruh batch. Dijumlahkan
    # akan memberi learning rate B kali lebih besar.
    if dw.dim() == 3:
        dw = dw.mean(dim=0)

    w_new = (w + dw).clamp(cfg.w_min, cfg.w_max)

    # --- 4. bump -------------------------------------------------------
    return w_new, Variant3State(
        pre=pre_d + z_pre,
        post1=post1_d + z_post,
        post2=post2_d + z_post,
    )
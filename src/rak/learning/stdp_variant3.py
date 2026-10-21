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
    # JANGAN keluar lebih awal hanya karena tidak ada spike post.
    # Suku LTD dipicu spike PRE, jadi ia harus tetap jalan. Versi
    # pertama keluar di sini dan diam-diam menghapus seluruh LTD
    # pada timestep yang hanya punya spike presinaptik.
    if not bool(fired.any()) and not bool((z_pre > 0).any()):
        return w, Variant3State(pre=pre_d + z_pre,
                                post1=post1_d + z_post,
                                post2=post2_d + z_post)

    w_ref = w if w.dim() == 2 else w[0]

    # --- 2 & 3. LTD (spike pre) dan LTP (spike post) -----------------
    # Bentuk non-batched lebih dulu; jalur batch di bawah.
    if z_pre.dim() == 1:
        dw = (
            cfg.nu_post * torch.outer(pre_d, post2_d) * z_post
            - cfg.nu_pre * torch.outer(z_pre, post1_d)
        )
    else:
        # Rata-rata batch DULU, baru perkalikan. Operasinya linear
        # dalam `pre_d` dan `post2_d` untuk `z_pre`/`z_post` yang
        # tetap, jadi hasilnya sama dengan menjumlahkan lalu membagi —
        # tapi tidak pernah membentuk tensor (B, n_in, n_out).
        #
        # `w` dipakai BERSAMA seluruh batch; dijumlahkan akan memberi
        # learning rate B kali lebih besar.
        # Outer product PER SAMPEL, lalu DIJUMLAHKAN.
        #
        # Dua kesalahan sebelumnya, keduanya mematikan pembelajaran:
        #
        # 1. `torch.outer(mean_pre, mean_post2)` menghitung
        #    E[pre] * E[post2]. Itu BUKAN E[pre * post2]. Produk dari
        #    rata-rata membuang justru sinyal yang dicari STDP:
        #    koinsidensi antara trace pre yang tinggi dan spike post.
        #    Setelah dirata-ratakan lebih dulu, korelasi antar gambar
        #    hilang dan yang tersisa hanya rata-rata populasi.
        #
        # 2. `.mean(dim=0)` membagi update dengan ukuran batch. Satu
        #    epoch 60.000 gambar jadi setara ~15 gambar pembelajaran,
        #    bukan 60.000. Bobot yang keluar dari run sebelumnya
        #    memang nyaris tidak berubah dari inisialisasi seragam.
        #
        # Menjumlahkan per sampel = sama dengan memproses gambar satu
        # per satu (urutan berbeda, total sama).
        dw = (
            cfg.nu_post
            * torch.einsum("bi,bj->bij", pre_d, post2_d)
            * fired.to(pre_d.dtype).unsqueeze(1)
            - cfg.nu_pre * torch.einsum("bi,bj->bij", z_pre, post1_d)
        ).sum(dim=0)

    w_new = (w + dw).clamp(cfg.w_min, cfg.w_max)

    # --- 4. bump -------------------------------------------------------
    return w_new, Variant3State(
        pre=pre_d + z_pre,
        post1=post1_d + z_post,
        post2=post2_d + z_post,
    )
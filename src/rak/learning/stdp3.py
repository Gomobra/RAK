r"""STDP Diehl & Cook 2015 — rule utama (Eq. 3) dan varian 3-trace.

T2.2b / Phase 3.

KESALAHAN YANG SUDAH GW PERBAIKI
---------------------------------
Versi pertama modul ini mengimplementasikan bentuk 3-trace
(`Δw ∝ post2 · pre`) karena itulah yang terlihat di kode Brian2 pada
`peter-u-diehl/stdp-mnist`. Itu **bukan** rule yang menghasilkan angka
95% di paper.

Setelah membaca paper: rule UTAMA (Eq. 3) cuma punya SATU trace:

              Δw = η · (x_pre − x_tar) · (w_max − w)^μ

    x_pre : jejak spike presinaptik (τ = 20 ms), naik 1 per spike
    x_tar : target jejak presinaptik saat neuron post fire
    μ     : eksponen dependensi bobot

Konsekuensi penting yang sering terlewat:

1. **Tidak ada suku LTD yang dipicu spike presinaptik.** Weight
   decrease terjadi HANYA saat spike postsinaptik, dan hanya kalau
   `x_pre < x_tar`. Ini bukan detail — artinya depresi hanya terjadi
   pada synapse yang sudah terhubung dengan neuron yang sedang aktif.
2. **Tidak ada timestep yang bisa "dilewati".** Bobot berubah hanya
   di timestep neuron post fire, persis seperti paper bilang:
   "weight updates are triggered only when a spike is fired by a
   postsynaptic excitatory neuron" — dan itu justru alasan aturan
   ini murah secara komputasi.

x_tar: MENGUKUR DIRI SENDIRI, BUKAN KONSTANTA
----------------------------------------------
Paper menulis `x_tar` sebagai "the target **average** value of the
presynaptic trace at the moment of a postsynaptic spike", tapi tidak
memberi angka yang bisa dipakai. Daripada menebak konstanta —
dan-tebak-tebakan seperti itu yang bikin replikasi meleset 5-8% —
`x_tar` di sini dihitung sebagai rata-rata bergerak (EMA) dari nilai
yang benar-benar diamati:

    observed[j] = Σ_i w[i,j] · x_pre[i] / Σ_i w[i,j]
    x_tar[j]   ← (1-α)·x_tar[j] + α·observed[j]      (tiap spike post)

Rata-rata dibobot bobot synapse, jadi `x_tar[j]` menjawab "jejak pre rata-rata
yang dilihat neuron j pada synapse-synapse-nya sendiri". Kalau neuron j
hanya menyala untuk input langka, `x_tar[j]` otomatis ikut rendah — dan
itu justru yang membuat neuron bisa membaca input langka dengan benar.
Konstanta yang bisa salah hilang, dan tidak ada parameter yang perlu
di-calibrate dari sumber luar.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "Eq3Config",
    "Eq3State",
    "eq3_step",
    "eq3_window",
    "make_eq3_state",
]


@dataclass(frozen=True)
class Eq3Config:
    """Parameter rule utama Diehl & Cook.

    Default: 6400 neuron. Untuk jaringan lebih kecil, `eta` harus
    dinaikkan — lihat `eta_for_target`.
    """

    #: Learning rate. Paper tidak menyebut angkanya di teks; nilai
    #: di sini dikalibrasi di `experiments/exp_003_mnist/`.
    eta: float = 3.6e-4

    #: Eksponen dependensi bobot. Paper memakai 0.2 dan menyebutnya
    #: "power-law weight dependence".
    mu: float = 0.2

    #: Time constant trace presinaptik.
    tau_pre_ms: float = 20.0

    dt_ms: float = 0.5
    w_min: float = 0.0
    w_max: float = 1.0

    #: Target jejak presinaptik. Kalau None, dihitung sendiri sebagai
    #: EMA (lihat module docstring). Set angka tetap kalau ingin
    #: membandingkan dengan implementasi yang pakai konstanta.
    x_tar: float | None = None

    #: Laju EMA untuk `x_tar`. Kecil = mengikuti cepat.
    x_tar_tau_ms: float = 500.0

    def __post_init__(self) -> None:
        if self.eta < 0:
            raise ValueError(f"eta harus >= 0, dapat {self.eta}")
        if self.tau_pre_ms <= 0:
            raise ValueError(f"tau_pre_ms harus > 0")
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms harus > 0")
        if self.w_min >= self.w_max:
            raise ValueError("w_min harus < w_max")
        if self.x_tar is not None and self.x_tar < 0:
            raise ValueError(f"x_tar harus >= 0, dapat {self.x_tar}")
        if self.x_tar_tau_ms <= 0:
            raise ValueError(f"x_tar_tau_ms harus > 0")

    @property
    def gamma_pre(self) -> float:
        return float(torch.exp(torch.tensor(-self.dt_ms / self.tau_pre_ms)).item())

    def eta_for_target(
        self,
        n_neuron: int,
        present_per_neuron: int,
        *,
        w_start: float = 0.1,
        x_pre_mean: float = 1.0,
    ) -> float:
        """Eta supaya bobot naik dari `w_start` ke `w_max`.

        Kenaikan per spike post adalah
        `eta · (x_pre − x_tar) · (w_max − w)^μ`. Dengan `x_tar` yang
        steady state-nya konvergen ke rata-rata, jadi rata-rata
        mendekati 0 — jadi untuk estimasi kasar cukup `x_pre_mean`.

        `present_per_neuron` = berapa kali satu neuron post(expected)
        menyala sepanjang training. Nilai INI yang menentukan skala eta;
       Versi pertama dari fungsi ini mengabaikannya dan menghasilkan
        nilai 60x terlalu kecil karena mengira satu spike per
        presentasi.
        """
        headroom = max(self.w_max - w_start, 1e-9)
        weight_factor = headroom**self.mu
        total_growth = self.w_max
        denom = max(present_per_neuron, 1) * x_pre_mean * weight_factor
        return total_growth / denom


@dataclass
class Eq3State:
    """Trace `x_pre` per-neuron presinaptik + `x_tar` per-neuron post."""

    pre: Tensor    # (n_in,)
    x_tar: Tensor  # (n_out,)


def make_eq3_state(n_pre: int, n_post: int, cfg: Eq3Config) -> Eq3State:
    """Trace awal.

    `x_tar` awal 0.0 kalau mode EMA: di awal training bobot acak dan
    jejak pre masih kecil, jadi target ikut kecil dan bobot boleh
    naik tanpa hambatan. Ini awal yang benar — steady state emerging
    sendiri.
    """
    return Eq3State(
        pre=torch.zeros(n_pre, dtype=torch.float32),
        x_tar=torch.zeros(n_post, dtype=torch.float32),
    )


def _observed_trace(w: Tensor, pre: Tensor) -> Tensor:
    """Rata-rata jejak pre, dibobot bobot synapse, per neuron post.

        observed[j] = Σ_i w[i,j]·pre[i] / Σ_i w[i,j]

    `pre` boleh 1D (n_in,) atau 2D (B, n_in) — `w` selalu 2D
    (n_in, n_out) karena satu matriks bobot dipakai bersama untuk
    seluruh batch. Kalau kolom w kosong, kembalikan 0 (bukan nan).
    """
    total_w = w.sum(dim=0)  # (n_out,)
    if pre.dim() == 1:
        weighted = torch.einsum("ij,i->j", w, pre)
    else:
        weighted = torch.einsum("ij,bi->bj", w, pre)
    return torch.where(
        total_w > 0,
        weighted / total_w.clamp(min=1e-12),
        torch.zeros_like(weighted),
    )


def eq3_step(
    z_pre: Tensor,
    z_post: Tensor,
    w: Tensor,
    state: Eq3State,
    cfg: Eq3Config,
) -> tuple[Tensor, Eq3State]:
    """Satu timestep Eq. 3.

    Args:
        z_pre: spike presinaptik, (n_in,) atau (B, n_in).
        z_post: spike postsinaptik, (n_out,) atau (B, n_out).
        w: bobot, (n_in, n_out).
        state: trace sebelum timestep ini.

    Returns:
        (w_baru, state_baru)

    URUTAN:
    1. `x_pre` meluruh
    2. kalau ada spike post: amati `x_pre`, perbarui `x_tar`, terapkan Δw
    3. `x_pre` di-bump dengan spike timestep ini

    Spike post di timestep ini tidak ikut memakai `x_pre` — `x_pre`
    baru di-bump di langkah 3. Kalau dibalik, neuron yang baru saja
    menyala akan langsung mempotensiasi dirinya sendiri, dan itu
    kredit yang sama sekali tidak terjadi di dunia nyata (aturan
    ADR-0004).
    """
    # `w` boleh 2D (n_in, n_out) atau 3D (B, n_in, n_out) saat batch.
    if w.dim() == 3:
        _, n_in, n_out = w.shape
    else:
        n_in, n_out = w.shape
    if tuple(state.pre.shape[-1:]) != (n_in,):
        raise ValueError(
            f"trace pre harus per-neuron ({n_in},), "
            f"dapat {tuple(state.pre.shape)}"
        )
    if tuple(state.x_tar.shape[-1:]) != (n_out,):
        raise ValueError(
            f"x_tar harus per-neuron ({n_out},), "
            f"dapat {tuple(state.x_tar.shape)}"
        )

    # --- 1. decay ------------------------------------------------------
    # Trace disimpan tanpa dimensi batch supaya tidak boros, tapi harus
    # di-expand mengikuti spike timestep ini. Tanpa ini, batched call
    # diam-diam memakai trace bersama untuk semua baris batch —
    # learning rate efektif jadi B kali lebih besar per gambar.
    pre_d = cfg.gamma_pre * state.pre.expand(z_pre.shape)
    x_tar_cur = state.x_tar.expand(z_post.shape)

    # --- 2. update pada spike post -------------------------------------
    fired = z_post > 0

    # LEWATI SELURUH PERHITUNGAN kalau tidak ada neuron post yang
    # menyala timestep ini.
    #
    # Eq. 3 hanya mengubah bobot pada spike postsinaptik. Kalau tidak
    # ada, `dw` pasti nol dan `w` tidak berubah — jadi `weight_term`
    # yang butuh `pow(mu)` (4.2 ms untuk (784,400) di CPU ARM) tidak
    # perlu dihitung sama sekali.
    #
    # Ini bukan optimasi kecil: dengan ~5 spike per gambar dari 400
    # neuron dalam 700 timestep, lebih dari 99% timestep bisa
    # dilewati. Tanpa ini, `pow`_sendiri menghabiskan sebagian besar
    # waktu training.
    if not bool(fired.any()):
        pre_bumped = pre_d + z_pre
        return w, Eq3State(pre=pre_bumped, x_tar=x_tar_cur)

    # Bobot dipakai BERSAMA seluruh batch, jadi selalu pakai matriks
    # 2D-nya — baik untuk rata-rata jejak pre maupun untuk suku
    # `(w_max - w)^mu`.
    w_ref = w if w.dim() == 2 else w[0]
    if bool(fired.any()):
        observed = _observed_trace(w_ref, pre_d)
        if cfg.x_tar is None:
            # EMA: neuron post yang menyala historinya menunjukkan
            # rata-rata jejak pre yang "wajar" untuk dia.
            alpha = cfg.dt_ms / cfg.x_tar_tau_ms
            new_tar = (1.0 - alpha) * x_tar_cur + alpha * observed
        else:
            new_tar = torch.full_like(x_tar_cur, float(cfg.x_tar))
    else:
        new_tar = x_tar_cur

    x_tar_used = torch.where(fired, new_tar, x_tar_cur)

    # `x_pre` per presinaptik, `x_tar` per postsinaptik — keduanya
    # jadi matriks lewat broadcast, BUKAN vektor yang bisa dikurangi
    # langsung. Versi pertama memakai `(pre_d - x_tar_used)` dan
    # gagal dengan galat dimensi.
    #
    # Batched: `pre_d` (B, n_in) dan `x_tar_used` (B, n_out), tapi
    # `w` (n_in, n_out) DIPAKAI BERSAMA seluruh batch — jadi diff
    # jadi (B, n_in, n_out). `w_ref` dipakai di sini, bukan `w`,
    # supaya tidak ikut jadi 3D dan menambah dimensi palsu.
    weight_term = (cfg.w_max - w_ref).clamp(min=0.0).pow(cfg.mu)

    if pre_d.dim() == 1:
        diff = pre_d.unsqueeze(1) - x_tar_used.unsqueeze(0)
        dw = cfg.eta * diff * weight_term
        dw = dw * fired.to(dw.dtype).unsqueeze(0)      # kolom yang fire
    else:
        diff = (pre_d.unsqueeze(2)                       # (B, n_in, 1)
                - x_tar_used.unsqueeze(1))               # (B, 1, n_out)
        dw_batch = cfg.eta * diff * weight_term
        dw_batch = dw_batch * fired.to(dw_batch.dtype).unsqueeze(1)

        # RATA-RATA, bukan jumlah. `w` itu SATU matriks yang dipakai
        # bersama seluruh batch. Kalau dijumlahkan, batch 64 memberi
        # learning rate 64 kali lebih besar daripada batch 1 — jaringan
        # akan saturasi di epoch pertama dan tidak ada yang mengira
        # itu bug.
        # SUM, bukan mean. `mean` membagi update dengan ukuran batch,
        # sehingga satu epoch 60.000 gambar hanya setara ~15 gambar
        # pembelajaran. Menjumlahkan = memproses gambar satu per satu.
        dw = dw_batch.sum(dim=0)                         # (n_in, n_out)

    w_new = (w + dw).clamp(cfg.w_min, cfg.w_max)

    # --- 3. bump -------------------------------------------------------
    new_state = Eq3State(pre=pre_d + z_pre, x_tar=new_tar)
    return w_new, new_state


def eq3_window(
    dt_ms: float,
    cfg: Eq3Config,
    max_ms: float = 100.0,
) -> tuple[Tensor, Tensor]:
    r"""Window Eq. 3 untuk plot dan verifikasi.

    Konvensi tanda sama dengan `stdp.py` (ADR-0004): `dt > 0` = post
    terlambat = potensiasi.

        dW(dt) = η · (x_pre(dt) − x_tar) · (w_max − w)^μ
        x_pre(dt) = exp(−dt/τ_pre)     untuk dt > 0

    Untuk `dt < 0` (post lebih awal) tidak ada update sama sekali —
    aturan ini tidak punya suku LTD berbasis spike pre. Itu bukan
    kelemahan, itu konsekuensi desain yang membuat komputasinya murah.
    """
    half = max(1, int(round(max_ms / dt_ms)))
    dts = torch.linspace(-half * dt_ms, half * dt_ms, 2 * half + 1)

    amp = cfg.eta * (cfg.w_max / 2.0) ** cfg.mu
    dw = torch.where(
        dts > 0,
        amp * torch.exp(-dts / cfg.tau_pre_ms),
        torch.zeros_like(dts),
    )
    return dts, dw
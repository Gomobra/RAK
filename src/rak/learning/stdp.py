r"""Spike-Timing-Dependent Plasticity, versi trace-based.

T2.2 — STDP tanpa daftar pasangan spike eksplisit.

Rumus:
    dW_ij = A_plus(W_ij)  * x_pre_i * z_post_j
          - A_minus(W_ij) * z_pre_i  * x_post_j

dengan trace eksponensial:
    x_pre[t]  = gamma_pre  * x_pre[t-1]  + z_pre[t]
    x_post[t] = gamma_post * x_post[t-1] + z_post[t]
    gamma = exp(-dt / tau)

Mengapa trace-based, bukan pair-based (rumus klasik Bi & Poo 1998):

Pair-based butuh daftar (t_pre, t_post) untuk setiap pasangan synapse dan
menghitung exp(-|t_pre - t_post| / tau) untuk tiap pasangan. Itu O(K)
per synapse dengan K = jumlah spike. Trace-based cuma O(1) per synapse
per timestep: dua perkalian dan dua penjumlahan.

Kasus simultaneous spike (t_pre == t_post) dapat dihitung dengan trace:
karena update terjadi SEBELUM trace di-bump, ``x_post[t]`` masih 0 kalau
post neuron belum pernah spike. Hasilnya dW = 0 — yang benar, karena
tanpa urutan waktu tidak ada credit assignment yang bisa diberikan.

EMBERAPA urutan ini penting:

    untuk setiap timestep t:
        1. dW <- hitung dari trace SAAT INI (isi spike s.d. t-1)
        2. W  <- clamp(W + dW)
        3. trace <- decay + bump dengan spike timestep t

Kalau trace di-bump lebih dulu, neuron yang pre DAN post fire di timestep
yang sama akan saling "melihat" spike-nya sendiri, dan dW jadi bukan nol.
 Itu kredit yang tidak pernah terjadi di dunia nyata.

Empat varian A_plus/A_minus mengikuti Norse:
    additive             : A = eta
    additive_step        : A_plus = eta_plus * H(w_max - w)
                           A_minus = eta_minus * H(w - w_min)
    multiplicative_pow   : A_plus = eta_plus * (w_max - w)^mu
                           A_minus = eta_minus * (w - w_min)^mu
    multiplicative_relu  : A_plus = eta_plus * relu(w_max - w)
                           A_minus = eta_minus * relu(w - w_min)

Varian multiplicative membuat A_plus menuju 0 saat W mendekati w_max,
sehingga pertumbuhan melambat kuadratik — itu alasan varian ini ada.

Klaim yang tepat: bobot **tetap terbatas**, bukan "tidak pernah melewati
batas". Satu langkah dengan eta besar masih bisa melewati w_max, dan
karena trace yang jenuh membuat langkah update ikut membesar, W cenderung
berosilasi di sekitar w_max, bukan konverge monotonik.

Catatan lain yang sering disalahpahami: kalau pre dan post fire DENGAN
FREKUENSI SAMA, LTP dan LTD saling menyeimbangkan dan W mencapai fixed
point. Itu bukan bug — justru di situlah stabilizing role STDP muncul,
selain dari clamping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "STDPConfig",
    "ALGORITHMS",
    "stdp_step",
    "stdp_window",
    "make_trace",
    "A_plus",
    "A_minus",
]

ALGORITHMS = (
    "additive",
    "additive_step",
    "multiplicative_pow",
    "multiplicative_relu",
)


@dataclass(frozen=True)
class STDPConfig:
    """Parameter STDP. Frozen karena konfigurasi, bukan state."""

    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0
    eta_plus: float = 1e-3
    eta_minus: float = 1e-3
    w_min: float = 0.0
    w_max: float = 1.0
    #: Eksponen untuk multiplicative_pow.
    mu: float = 0.0
    algorithm: str = "additive"
    dt_ms: float = 1.0
    #: Kalau False, W tidak di-clip ke [w_min, w_max].
    hardbound: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm harus salah satu dari {ALGORITHMS}, "
                f"dapat {self.algorithm!r}"
            )
        if self.tau_pre_ms <= 0 or self.tau_post_ms <= 0:
            raise ValueError(
                f"tau harus > 0, dapat tau_pre={self.tau_pre_ms} "
                f"tau_post={self.tau_post_ms}"
            )
        if self.w_min >= self.w_max:
            raise ValueError(
                f"w_min ({self.w_min}) harus < w_max ({self.w_max})"
            )
        if self.eta_plus < 0 or self.eta_minus < 0:
            raise ValueError(
                f"eta harus >= 0, dapat eta_plus={self.eta_plus} "
                f"eta_minus={self.eta_minus}"
            )
        if self.mu != 0.0 and self.algorithm not in (
            "multiplicative_pow",
        ):
            # Tanpa guard ini, `STDPConfig(mu=1.0)` diam-diam memakai
            # default algorithm="additive" dan `mu` dibuang — yang
            # terjadi di figure 03 sebelum diperbaiki: tiga kurva w_ref
            # identik persis, tapi figurnyaDll Claim "memperlihatkan
            # varian multiplicative".
            raise ValueError(
                f"mu={self.mu} hanya dipakai oleh algorithm="
                f"'multiplicative_pow'. Got {self.algorithm!r}, yang "
                f"tidak membaca mu sama sekali. Kalau yang dimaksud "
                f"multiplicative tanpa pangkat, pakai "
                f"algorithm='multiplicative_relu'."
            )

    @property
    def gamma_pre(self) -> float:
        return float(torch.exp(torch.tensor(-self.dt_ms / self.tau_pre_ms)).item())

    @property
    def gamma_post(self) -> float:
        return float(torch.exp(torch.tensor(-self.dt_ms / self.tau_post_ms)).item())


# ----------------------------------------------------------------------
# Bentuk A_plus / A_minus
# ----------------------------------------------------------------------
def A_plus(w: Tensor, cfg: STDPConfig) -> Tensor:
    """Amplitudo LTP sebagai fungsi bobot saat ini."""
    if cfg.algorithm == "additive":
        a = torch.full_like(w, cfg.eta_plus)
    elif cfg.algorithm == "additive_step":
        a = cfg.eta_plus * (w < cfg.w_max).to(w.dtype)
    elif cfg.algorithm == "multiplicative_pow":
        a = cfg.eta_plus * (cfg.w_max - w).clamp(min=0.0).pow(cfg.mu)
    else:  # multiplicative_relu
        a = cfg.eta_plus * (cfg.w_max - w).clamp(min=0.0)
    return a


def A_minus(w: Tensor, cfg: STDPConfig) -> Tensor:
    """Amplitudo LTD sebagai fungsi bobot saat ini."""
    if cfg.algorithm == "additive":
        a = torch.full_like(w, cfg.eta_minus)
    elif cfg.algorithm == "additive_step":
        a = cfg.eta_minus * (w > cfg.w_min).to(w.dtype)
    elif cfg.algorithm == "multiplicative_pow":
        a = cfg.eta_minus * (w - cfg.w_min).clamp(min=0.0).pow(cfg.mu)
    else:  # multiplicative_relu
        a = cfg.eta_minus * (w - cfg.w_min).clamp(min=0.0)
    return a


# ----------------------------------------------------------------------
# Trace
# ----------------------------------------------------------------------
def make_trace(n: int, value: float = 0.0, like: Tensor | None = None) -> Tensor:
    """Trace baru (per-neuron) dengan nilai awal `value`.

    Trace bersifat **per-neuron**, bukan per-synapse. itu karena
    `dW_ij` hanya butuh `x_pre_i` dan `x_post_j` — tidak perlu nilai
    yang berbeda untuk tiap synapse (i, j). Menyimpan trace per-synapse
    akan memakai memori n_in x n_out, bukan n_in + n_out.
    """
    dtype = like.dtype if like is not None else torch.float32
    return torch.full((n,), value, dtype=dtype)


# ----------------------------------------------------------------------
# Inti STDP
# ----------------------------------------------------------------------
def stdp_step(
    z_pre: Tensor,
    z_post: Tensor,
    w: Tensor,
    trace_pre: Tensor,
    trace_post: Tensor,
    cfg: STDPConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Satu timestep STDP.

    Args:
        z_pre: spike pre timestep ini, (n_in,)
        z_post: spike post timestep ini, (n_out,)
        w: bobot sekarang, (n_in, n_out). Tidak dimodifikasi.
        trace_pre: trace pre SEBELUM timestep ini, (n_in,) — per-neuron
        trace_post: trace post SEBELUM timestep ini, (n_out,) — per-neuron
        cfg: konfigurasi

    Returns:
        (w_baru, trace_pre_baru, trace_post_baru)

    Urutan di dalam fungsi ini adalah inti kebenaran STDP: bobot dihitung
    dari trace yang belum memuat spike timestep ini.
    """
    n_in, n_out = w.shape
    if trace_pre.shape != (n_in,):
        raise ValueError(
            f"trace_pre harus per-neuron shape ({n_in},), "
            f"dapat {tuple(trace_pre.shape)}. Trace per-synapse boros dan "
            "tidak dipakai rumus STDP mana pun."
        )
    if trace_post.shape != (n_out,):
        raise ValueError(
            f"trace_post harus per-neuron shape ({n_out},), "
            f"dapat {tuple(trace_post.shape)}"
        )

    # --- 1. Decay trace DULU ------------------------------------------
    # Trace harus meluruh SEBELUM update supaya nilainya mencerminkan
    # "sudah meluruh selama dt timestep". Kalau bump-dulu-update,
    # trace baru meluruh (dt - 1) kali dan STDP window diskretnya
    # menyimpang dari rumus eksponensial sebesar satu faktor gamma.
    tp_dec = cfg.gamma_pre * trace_pre
    tq_dec = cfg.gamma_post * trace_post

    # --- 2. Weight update, pakai trace yang SUDAH meluruh ------------
    ap = A_plus(w, cfg)
    am = A_minus(w, cfg)
    dw = ap * torch.outer(tp_dec, z_post) - am * torch.outer(z_pre, tq_dec)
    w_new = w + dw
    if cfg.hardbound:
        w_new = w_new.clamp(cfg.w_min, cfg.w_max)

    # --- 3. Bump trace dengan spike timestep INI ---------------------
    # Setelah weight update, supaya spike di timestep ini tidak ikut
    # berkontribusi ke update-nya sendiri.
    trace_pre_new = tp_dec + z_pre
    trace_post_new = tq_dec + z_post

    return w_new, trace_pre_new, trace_post_new

# ----------------------------------------------------------------------
# Window analitik (untuk verifikasi + plot)
# ----------------------------------------------------------------------
def stdp_window(
    dt_ms: float,
    cfg: STDPConfig,
    w_ref: float | None = None,
    max_ms: float = 100.0,
) -> tuple[Tensor, Tensor]:
    """STDP window dari rumus eksponensial, untuk plot dan verifikasi.

    Konvensi tanda (konsisten dengan `stdp_step`):

    ==============  =========  ====================================
    `dt`           -meaning    dW
    ==============  =========  ====================================
    `dt > 0`        post       `+A_plus · exp(-dt/tau_pre)`
                    TERLAMBAT  (potensiasi — pre mendahului post)
    `dt < 0`        post       `-A_minus · exp(dt/tau_post)`
                    LEBIH AWAL (depresi — post mendahului pre)
    `dt = 0`        simultan   0
    ==============  =========  ====================================

    Perhatikan arah tanda: spike pre yang mendahului post adalah
    **potensiasi**. Jadi daerah positif ada di `dt > 0`, bukan sebaliknya.
    (Kebanyakan plot STDP di literatur memakai kebalikan konvensi ini —
   attles itu labels sumbu yang berbeda, bukan rumusnya.)

    Args:
        dt_ms: timestep simulasi
        cfg: konfigurasi
        w_ref: bobot referensi untuk varian multiplicative. Default: titik
            tengah rentang [w_min, w_max].
        max_ms: rentang +/- yang diplot

    Returns:
        (dts, dW) dengan dua tensor 1-D sepanjang `2*max_ms/dt_ms + 1`.
    """
    if w_ref is None:
        w_ref = 0.5 * (cfg.w_min + cfg.w_max)

    # Jumlah titik GANJIL supaya dt = 0 pasti ada di sampling — tanpa
    # itu, test "window nol di lag nol" hanya bisa perkiraan.
    half = max(1, int(round(max_ms / dt_ms)))
    dts = torch.linspace(-half * dt_ms, half * dt_ms, 2 * half + 1)
    w = torch.full((1, 1), w_ref)
    a_p = float(A_plus(w, cfg)[0, 0])
    a_m = float(A_minus(w, cfg)[0, 0])

    dw = torch.where(
        dts > 0,
        a_p * torch.exp(-dts / cfg.tau_pre_ms),
        torch.where(
            dts < 0,
            -a_m * torch.exp(dts / cfg.tau_post_ms),
            torch.zeros_like(dts),
        ),
    )
    return dts, dw

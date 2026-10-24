r"""Encoding MNIST jadi spike train Poisson, dengan intensitas adaptif.

T6.3 / Phase 3.

Kode acuan (`peter-u-diehl/stdp-mnist`) melakukan dua hal yang
sering tidak ditulis di paper:

    rates = pixel_uint8 / 8 * input_intensity

dan

    if np.sum(current_spike_count) < 5:
        input_intensity += 1        # ulangi gambar yang SAMA

Jadi laju spike BUKAN konstan — kalau jaringan terlalu pelan, gambar
dipresentasikan ulang dengan intensitas lebih tinggi sampai total spike
cukup. Itu membuat laju efektif beradaptasi ke jaringan yang sedang
dibangun, dan tanpa itu neuron awal yang bobotnya nol tidak pernah
mendapat kesempatan menyala sama sekali (dead network).

APA YANG DIUJI DI SINI — PROPERTI, BUKAN ASUMSI
-------------------------------------------------
Yang penting bukan "kode menghasilkan angka yang saya">, tapi apakah
distribusinya benar-benar Poisson dengan laju yang diminta:

    E[spike per timestep | rate r Hz] = r * dt / 1000

Itu diuji lewat rata-rata empiris pada sample besar. Kalau
implementasi salah memindahkan konversi Hz->probabilitas, test ini
gagal — sementara test yang membandingkan ke implementasi lain
akan lolos diam-diam, karena dua kesalahan yang sama bisa saling
meniadakan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "EncoderConfig",
    "encode_poisson",
    "AdaptiveIntensity",
    "spike_counts",
    "estimated_rate_hz",
]


@dataclass(frozen=True)
class EncoderConfig:
    """Parameter encoding spike.

    Default mengikuti `peter-u-diehl/stdp-mnist`.
    """

    #: Lama satu gambar dipresentasikan.
    present_ms: float = 350.0

    #: Jeda tanpa stimulus antar presentasi. Bukan cuma untuk
    #: "istirahat" — membran dan conductance harus sempat meluruh,
    #: kalau tidak, jejak dari gambar sebelumnya masih hidup saat
    #: gambar berikutnya mulai dan STDP melihat CREDIT yang salah.
    rest_ms: float = 0.0

    dt_ms: float = 0.5

    #: Laju maksimum per satuan intensitas, dalam Hz.
    #:
    #: 255 / 8 = 31.875. Dengan `intensity = 2` (default acuan),
    #: pixel 255 jadi 63.75 Hz. Angka 31.875 inilah yang jadi rumus
    #: `max_rate_hz(pixel, intensity)` — bukan "255/4" seperti
    #: tertulis di ADR-0003, meskipun angkanya kebetulan sama.
    max_rate_per_intensity: float = 31.875

    #: Intensitas awal. Acuan pakai 2.0.
    start_intensity: float = 2.0

    #: Ambang total spike OUTPUT di bawah ini -> presentasi diulang
    #: dengan intensitas lebih tinggi.
    #:
    #: Default 25, bukan 5 dari kode acuan. Kode acuan
    #: presents satu gambar pada satu waktu, jadi "total spike" di sana berarti
    #: Across the WHOLE population. Di sini batch 64 gambar diproses
    #: bersamaan dan `per_image` membandingkan per gambar — angka 5
    #: jadi terlalu kecil untuk menahan gambar yang sunyi, dan
    #: gambar-gambar itu lolos tanpa pernah diulang. Run 100 neuron
    #: (inhibisi 400) berakhir dengan 25 spike per presentasi TOTAL
    #: dengan 0 neuron mati — artinya hampir semua gambar lolos gate
    #: tanpa diulang, dan jaringan tidak pernah mendapat tekanan untuk
    #: menyala lebih.
    #:
    #: PENTING: ini menghitung spike neuron EKSITASI, bukan spike
    #: input. Kode acuan mengukur `spike_counters['Ae']`. Versi
    #: pertama dokumentasi kita menyimpulkannya menghitung spike
    #: input — dan itu hampir tidak pernah terjadi, karena 784 input
    #: dengan pixel secerah apa pun sudah menghasilkan ratusan spike
    #: dalam 350 ms. Threshold-nya jadi tidak akan pernah aktif, dan
    #: mekanisme yang justru mencegah dead network jadi tidak pernah
    #: berjalan.
    min_spikes: int = 5

    def __post_init__(self) -> None:
        if self.present_ms <= 0:
            raise ValueError(f"present_ms harus > 0")
        if self.rest_ms < 0:
            raise ValueError(f"rest_ms harus >= 0")
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms harus > 0")
        if self.dt_ms > self.present_ms:
            raise ValueError(
                f"dt_ms ({self.dt_ms}) > present_ms ({self.present_ms})"
            )
        if self.max_rate_per_intensity <= 0:
            raise ValueError("max_rate_per_intensity harus > 0")
        if self.start_intensity <= 0:
            raise ValueError("start_intensity harus > 0")
        if self.min_spikes < 0:
            raise ValueError("min_spikes harus >= 0")

    @property
    def n_steps(self) -> int:
        """Jumlah timestep untuk satu presentasi."""
        return int(round(self.present_ms / self.dt_ms))

    @property
    def n_rest_steps(self) -> int:
        return int(round(self.rest_ms / self.dt_ms))

    def max_rate_hz(self, intensity: float | None = None) -> float:
        """Laju spike maksimum (Hz) untuk pixel putih penuh.

        `images` sudah dalam [0, 1] (pixel uint8 dibagi 255), jadi
        `255/8 = 31.875` sudah JUMLAH faktor pngannya. Versi pertama
        mengalikan 255 lagi dan menghasilkan 16256 Hz — 255 kali
        terlalu besar. Guard probabilitas > 1 yang menangkapnya.
        """
        i = self.start_intensity if intensity is None else intensity
        return self.max_rate_per_intensity * i


def estimated_rate_hz(
    images: Tensor, intensity: float, cfg: EncoderConfig
) -> Tensor:
    """Laju spike yang diminta per neuron input, dalam Hz.

    Args:
        images: (B, n_in) atau (n_in,), nilai dalam [0, 1].
        intensity: pengali intensitas.
        cfg: konfigurasi.

    Returns:
        (B, n_in) atau (n_in,) laju dalam Hz.
    """
    return images * (cfg.max_rate_per_intensity * intensity)


def encode_poisson(
    images: Tensor,
    intensity: float,
    cfg: EncoderConfig,
    *,
    generator: torch.Generator | None = None,
    n_steps: int | None = None,
    rest: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    """Ubah intensitas gambar jadi spike train biner.

    Args:
        images: (B, n_in) nilai [0, 1], atau (n_in,).
        intensity: pengali laju.
        cfg: konfigurasi.
        generator: sumber randomness. Wajib diberikan supaya hasil
            bisa direproduksi — spike Poisson tanpa seed tidak
            bisa diulang, dan hasil MNIST tidak bisa dibandingkan
            antar run.
        n_steps: override jumlah timestep.
        rest: kalau True, tambahkan tail diam (untuk jeda).

    Returns:
        (n_steps, B, n_in) float32, nilai 0.0 atau 1.0. Kalau `out`
        diberikan, hasil ditulis ke sana dan tensor yang sama
        dikembalikan (tanpa alokasi baru).

    Catatan: spike digenerate per timestep dari distribusi Binomial
    dengan p = rate * dt / 1000. Mengambil `p` per neuron PER
    TIMESTEP (bukan sekali untuk seluruh presentasi) penting: kalau
    tidak, korelasi antar neuron di timestep yang sama salah, dan
    STDP ikut salah.
    """
    if images.dim() == 1:
        images = images.unsqueeze(0)
    if images.dim() != 2:
        raise ValueError(
            f"images harus (n_in,) atau (B, n_in), "
            f"dapat {tuple(images.shape)}"
        )
    if float(images.min()) < -1e-6 or float(images.max()) > 1.0 + 1e-6:
        raise ValueError(
            f"images harus dalam [0, 1], rentang saat ini "
            f"[{float(images.min()):.3f}, {float(images.max()):.3f}]"
        )
    if intensity <= 0:
        raise ValueError(f"intensity harus > 0, dapat {intensity}")

    steps = n_steps if n_steps is not None else cfg.n_steps
    n_batch, n_in = images.shape

    rates_hz = estimated_rate_hz(images, intensity, cfg)
    prob = rates_hz * (cfg.dt_ms / 1000.0)

    if float(prob.max()) > 1.0:
        raise ValueError(
            f"probabilitas spike melebihi 1 ({float(prob.max()):.3f}). "
            f"rate per timestep tidak boleh melebihi 1/dt = "
            f"{1000.0 / cfg.dt_ms:.0f} Hz. Turunkan intensitas atau "
            f"perkecil dt."
        )

    if out is None:
        out = torch.empty((steps, n_batch, n_in), dtype=torch.float32)
    # generate in-place ke buffer yang sudah ada -> tidak ada
    # alokasi 280 MB per presentasi.
    torch.rand((steps, n_batch, n_in), generator=generator, out=out)
    # bandingkan in-place (float32 tensor jadi 0.0/1.0)
    out.lt_(prob.unsqueeze(0).expand_as(out))
    # lt_ pada float32 tensor menghasilkan bool in-place? Tidak --
    # PyTorch lt_ pada float tensor mengubah nilai jadi 0.0/1.0
    # float (bukan bool). Verifikasi aman.
    return out


def spike_counts(spikes: Tensor) -> Tensor:
    """Jumlah spike per elemen, (..., n_in) -> (...)."""
    return spikes.sum(dim=0)


@dataclass
class AdaptiveIntensity:
    """Pengatur intensitas presentasi, ala Diehl & Cook.

    Aturan dari kode acuan: kalau total spike OUTPUT dari satu
    gambar < `min_spikes`, gambar yang SAMA dipresentasikan ulang
    dengan `intensity += 1`.
    Intensitas kembali ke awal setelah gambar mendapat cukup spike.

    Kenapa ini penting: di awal training bobot masih kecil, jadi
    membran tidak pernah melewati threshold. Tanpa eskalasi, jaringan
    tidak pernah mulai belajar — dan gejalanya (akurasi ~10%,
    setara tebakan acak) sangat mudah disalahartikan sebagai "STDP
    tidak bekerja".
    """

    cfg: EncoderConfig
    intensity: float | None = None

    def __post_init__(self) -> None:
        if self.intensity is None:
            self.intensity = self.cfg.start_intensity
        if self.intensity <= 0:
            raise ValueError(f"intensity harus > 0")

    def escalate(self) -> float:
        """Naikkan intensitas satu tingkat. Returns intensitas baru."""
        self.intensity += 1.0
        return self.intensity

    def reset(self) -> float:
        """Kembalikan ke intensitas awal setelah presentasi berhasil."""
        self.intensity = self.cfg.start_intensity
        return self.intensity

    def needs_repeat(self, total_spikes: int | float) -> bool:
        """Apakah gambar ini perlu dipresentasikan ulang?"""
        return float(total_spikes) < self.cfg.min_spikes

    def update(self, total_spikes: int | float) -> bool:
        """Terapkan aturan. Returns True kalau harus diulang."""
        if self.needs_repeat(total_spikes):
            self.escalate()
            return True
        self.reset()
        return False
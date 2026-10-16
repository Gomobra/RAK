"""Visualisasi neuron.

T1.1.4 — plot V(t) dan spike train.
T1.2.4 — bandingkan LIF vs Adaptive LIF (overlay V, θ, dan spike rate).

Semua fungsi memakai matplotlib dengan backend non-interaktif supaya
aman dipanggil dari CLI/CI. Tidak ada plotting di dalam loop simulasi —
plot selalu setelah data terkumpul.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

__all__ = [
    "plot_membrane",
    "plot_comparison",
    "plot_spike_raster",
    "plot_adaptation",
    "plot_stdp_window",
    "plot_theta_adaptation",
    "plot_drive_history",
    "plot_column_normalization",
    "plot_association_learning",
    "panel_configs",
]


def _prep_axes(n_rows: int = 1, figsize: tuple = (10.0, 3.0)):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize, sharex=True, squeeze=False
    )
    return fig, [ax for row in axes for ax in row]


def _prep_grid(
    n_rows: int, n_cols: int, figsize: tuple = (10.0, 8.0)
):
    """Subplot `n_rows` x `n_cols`, dikembalikan sebagai array 2D.

    `_prep_axes` selalu kolom tunggal, jadi tidak bisa dipakai untuk
    figure yang butuh tata letak 2x2. Yang ini dikembalikan apa adanya
    (`axes[r][c]`) supaya posisi panel tidak perlu dihitung manual.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False
    )
    return fig, axes


def plot_membrane(
    v_rec: Tensor,
    spikes: Tensor,
    out_path: str | Path,
    title: str = "LIF neuron",
    v_th: float | None = None,
    threshold_label: str = r"$v_{th}$",
) -> Path:
    """Plot V(t) dan spike train untuk SATU neuron.

    Spike digambar sebagai garis vertikal dari bawah ke atas — bentuk
    standar raster plot di neuroscience.

    Args:
        v_rec: (T,) rekaman potensial membran
        spikes: (T,) spike biner
        out_path: file tujuan (png/svg/pdf)
        title: judul plot
        v_th: kalau diisi, gambar garis threshold

    Returns:
        Path file yang ditulis.
    """
    if v_rec.ndim != 1 or spikes.ndim != 1:
        raise ValueError(
            f"v_rec dan spikes harus 1D, dapat {v_rec.ndim}D / {spikes.ndim}D"
        )
    if v_rec.shape != spikes.shape:
        raise ValueError(
            f"shape tidak cocok: v_rec {tuple(v_rec.shape)} vs "
            f"spikes {tuple(spikes.shape)}"
        )

    v = v_rec.detach().cpu().numpy()
    s = spikes.detach().cpu().numpy()
    t = torch.arange(v.shape[0]).numpy()

    fig, (ax_v, ax_s) = _prep_axes(2, figsize=(10.0, 6.0))

    ax_v.plot(t, v, lw=1.0, color="#2563eb")
    if v_th is not None:
        ax_v.axhline(v_th, ls="--", lw=0.8, color="#dc2626", label=threshold_label)
        ax_v.legend(loc="upper right", fontsize=8)
    ax_v.set_ylabel("V (mV)")
    ax_v.set_title(f"{title} — {len(s.nonzero()[0])} spike")
    ax_v.grid(alpha=0.3)

    idx = s.nonzero()[0]
    # Sumbu spike harus BINER (0/1), bukan mewarisi skala tegangan dari
    # panel di atas. Kalau tidak, panel ini menampilkan angka -60/-40
    # yang tidak ada artinya sebagai "keadaan spike".
    ax_s.vlines(idx, 0.0, 1.0, color="#dc2626", lw=1.0)
    ax_s.set_ylim(-0.15, 1.15)
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(["no spike", "spike"])
    ax_s.set_ylabel("state")
    ax_s.set_xlabel("time (ms)")
    ax_s.grid(alpha=0.3)

    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def plot_comparison(
    results: dict[str, tuple[Tensor, Tensor]],
    out_path: str | Path,
    title: str = "LIF vs Adaptive LIF",
) -> Path:
    """Overlay potensial membran beberapa model.

    T1.2.4 — bandingkan LIF biasa dengan Adaptive LIF pada input sama.

    Args:
        results: {nama: (v_rec, spikes)} — semua shape (T,) sama
        out_path: file tujuan
    """
    if not results:
        raise ValueError("results tidak boleh kosong")

    lengths = {tuple(v.shape) for v, _ in results.values()}
    if len(lengths) != 1:
        raise ValueError(f"semua rekaman harus sama panjang, dapat {lengths}")

    n_models = len(results)
    fig, axes = _prep_axes(n_models, figsize=(10.0, 2.2 * n_models))

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]
    for ax, (name, (v_rec, spikes)) in zip(axes, results.items()):
        color = colors[list(results).index(name) % len(colors)]
        v = v_rec.detach().cpu().numpy()
        s = spikes.detach().cpu().numpy()
        t = torch.arange(v.shape[0]).numpy()

        ax.plot(t, v, lw=1.0, color=color)
        idx = s.nonzero()[0]
        lo, hi = float(v.min()), float(v.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.vlines(idx, lo - pad, hi, color=color, lw=0.8, alpha=0.7)
        ax.set_ylim(lo - pad, hi)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title(f"{title} — total {len(idx)} spike", fontsize=10)

    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def plot_spike_raster(
    spikes: Tensor,
    out_path: str | Path,
    title: str = "Spike raster",
    max_neurons: int = 100,
) -> Path:
    """Raster plot spike untuk banyak neuron (sparse neuron di satu layer).

    Args:
        spikes: (T, batch, n_neurons) — sama dengan output SpikeLayer
        out_path: file tujuan
        max_neurons: batasi tampil supaya file tidak memblender
    """
    if spikes.ndim != 3:
        raise ValueError(
            f"spikes harus 3D (T, batch, n_neurons), dapat {spikes.ndim}D"
        )

    T, B, N = spikes.shape
    n_show = min(N, max_neurons)
    s = spikes[:, 0, :n_show].detach().cpu().numpy()

    fig, axes = _prep_axes(1, figsize=(10.0, max(3.0, n_show * 0.12)))
    ax = axes[0]
    for neuron in range(n_show):
        idx = s[:, neuron].nonzero()[0]
        ax.vlines(idx, neuron, neuron + 1, color="#111827", lw=0.7)

    ax.set_xlim(0, T)
    ax.set_ylim(0, n_show)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron")
    ax.set_title(f"{title} — {T} steps, density={float(s.mean()):.4f}")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def plot_adaptation(
    v_rec: Tensor,
    theta_rec: Tensor,
    spikes: Tensor,
    out_path: str | Path,
    title: str = "Adaptive LIF — mekanisme adaptasi",
    v_th: float = 1.0,
) -> Path:
    """Plot terpisah V, theta, dan spike untuk Adaptive LIF.

    T1.2.4 — memisahkan kurva membuat mekanismenya terlihat:

    Panel 1  V      — naik ke v_eff, lalu reset ke v_reset
    Panel 2  theta  — naik tiap spike, meluruh pelan (tau_adapt)
    Panel 3  v_eff  — = v_th + theta, GARIS BERSINGKUNG dengan puncak V
    Panel 4  spike  — biner, makin jarang seiring v_eff naik

    Panel 3 adalah kunci: apa yang sebenarnya dinaikkan bukan tegangan
    membran, tapi **threshold efektif**-nya. Tanpa panel ini, pengamat
    akan salah mengira adaptasi terjadi karena V naik.

    Args:
        v_rec: (T,) potensial membran
        theta_rec: (T,) threshold adaptasi
        spikes: (T,) spike biner
        out_path: file tujuan
        v_th: threshold dasar
    """
    for name, t in (("v_rec", v_rec), ("theta_rec", theta_rec), ("spikes", spikes)):
        if t.ndim != 1:
            raise ValueError(f"{name} harus 1D, dapat {t.ndim}D")
    if not (v_rec.shape == theta_rec.shape == spikes.shape):
        raise ValueError(
            f"shape harus sama: v={tuple(v_rec.shape)} "
            f"theta={tuple(theta_rec.shape)} spikes={tuple(spikes.shape)}"
        )

    v = v_rec.detach().cpu().numpy()
    th = theta_rec.detach().cpu().numpy()
    s = spikes.detach().cpu().numpy()
    t_ax = torch.arange(v.shape[0]).numpy()
    v_eff = v_th + th

    fig, axes = _prep_axes(4, figsize=(10.0, 9.0))

    axes[0].plot(t_ax, v, lw=1.0, color="#2563eb")
    axes[0].set_ylabel("V (mV)", color="#2563eb")
    axes[0].tick_params(axis="y", labelcolor="#2563eb")
    axes[0].axhline(v_th, ls=":", lw=0.8, color="#94a3b8")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_ax, th, lw=1.2, color="#059669")
    axes[1].set_ylabel(r"$\theta$", color="#059669")
    axes[1].tick_params(axis="y", labelcolor="#059669")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_ax, v_eff, lw=1.2, color="#7c3aed", label=r"$v_{th}+\theta$")
    axes[2].axhline(v_th, ls="--", lw=0.8, color="#dc2626", label=r"$v_{th}$")
    axes[2].set_ylabel("effective threshold", color="#7c3aed")
    axes[2].tick_params(axis="y", labelcolor="#7c3aed")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(alpha=0.3)

    idx = s.nonzero()[0]
    axes[3].vlines(idx, 0.0, 1.0, color="#dc2626", lw=1.0)
    axes[3].set_ylim(-0.15, 1.15)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["no spike", "spike"])
    axes[3].set_ylabel("state")
    axes[3].set_xlabel("time (ms)")
    axes[3].grid(alpha=0.3)

    axes[0].set_title(
        f"{title} — {len(idx)} spike, "
        f"v_eff naik {float(th[0]):.3f} -> {float(th[-1]):.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def panel_configs(cfg: STDPConfig) -> dict[str, STDPConfig]:
    """Satu `STDPConfig` turunan per varian, untuk figure per-algoritma.

    Dipisah dari `plot_stdp_window` supaya bisa dites tanpa merender
    gambar. Ada karena bug nyata: versi lama loop `for ax, algo in
    zip(axes, ALGORITHMS)` memakai `algo` CUMA sebagai label, lalu
    meneruskan `cfg` apa adanya ke `stdp_window` — yang mendispatch ke
    `cfg.algorithm`. Hasilnya keempat panel menampilkan algoritma yang
    sama dengan empat nama berbeda.

    `mu` ikut diturunkan: hanya `multiplicative_pow` yang membacanya,
    dan `STDPConfig` menolak `mu` pada varian lain.
    """
    from rak.learning.stdp import ALGORITHMS

    return {
        algo: replace(
            cfg, algorithm=algo, mu=cfg.mu if algo == "multiplicative_pow" else 0.0
        )
        for algo in ALGORITHMS
    }


def plot_stdp_window(
    cfg,
    out_path: str | Path,
    max_ms: float = 100.0,
    dt_ms: float = 1.0,
    title: str | None = None,
    w_refs: tuple[float, ...] | None = None,
) -> Path:
    """Plot STDP window dari rumus eksponensial.

    T2.2.4 — keempat varian A_plus/A_minus, masing-masing untuk beberapa
    nilai bobot referensi.

    Konvensi tanda (lihat `rak.learning.stdp.stdp_window`):
      - `dt > 0`  post TERLAMBAT -> **potensiasi** (daerah positif)
      - `dt < 0`  post LEBIH AWAL -> **depresi** (daerah negatif)

    Perhatikan: sebagian besar plot STDP di literatur memakai tanda
    kebalikan. Yang membedakan di sini adalah LEBAR tiap daerah — LTP
    ditentukan `tau_pre`, LTD oleh `tau_post`.

    Args:
        cfg: konfigurasi STDP
        out_path: file tujuan
        max_ms: rentang lag yang diplot
        dt_ms: resolusi titik
        title: judul; default dari parameter cfg
        w_refs: bobot referensi untuk varian multiplicative. Kalau
            None, pakai satu nilai di tengah [w_min, w_max] — hasilnya
            keempat varian akan IDENTIK kalau `mu = 0`, karena
            `(w_max - w)^0 = 1`. Untuk melihat perbedaan varian,
            pakai beberapa nilai, mis. `(0.1, 0.5, 0.9)`.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from rak.learning.stdp import ALGORITHMS, stdp_window

    fig, axes = _prep_axes(len(ALGORITHMS), figsize=(10.0, 2.1 * len(ALGORITHMS)))
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    refs = w_refs or (0.5 * (cfg.w_min + cfg.w_max),)
    shades = [1.0, 0.55, 0.3]

    per_algo = panel_configs(cfg)

    for ax, algo, color in zip(axes, ALGORITHMS, colors):
        # `panel_cfg` (bukan `cfg`) yang dipakai — inilah yang mencegah
        # keempat panel menampilkan algoritma yang sama dengan label
        # berbeda. Lihat `panel_configs`.
        panel_cfg = per_algo[algo]
        all_win = []
        for k, wr in enumerate(refs):
            dts, win = stdp_window(dt_ms, panel_cfg, w_ref=wr, max_ms=max_ms)
            ax.plot(dts, win, lw=1.5, color=color, alpha=shades[k % len(shades)],
                    label=f"w={wr}")
            all_win.append(win)

        peak = max(float(w.abs().max()) for w in all_win)
        pos = dts > 0
        neg = dts < 0
        ax.fill_between(dts[pos].numpy(), 0, all_win[0][pos].numpy(),
                        color="#2563eb", alpha=0.10)
        ax.fill_between(dts[neg].numpy(), 0, all_win[0][neg].numpy(),
                        color="#dc2626", alpha=0.10)
        ax.axhline(0.0, lw=0.6, color="#94a3b8")
        ax.axvline(0.0, lw=0.6, color="#94a3b8", ls=":")
        ax.set_ylabel(
            f"{algo}\ndW", fontsize=8, rotation=0, ha="right", va="center"
        )
        ax.grid(alpha=0.25)
        ax.set_ylim(-peak * 1.25, peak * 1.25)
        if len(refs) > 1:
            ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("lag post relatif pre (ms)   "
                        "kiri = post lebih awal (LTD), kanan = post terlambat (LTP)")
    fig.suptitle(
        title or (
            f"STDP window — tau_pre={cfg.tau_pre_ms}ms, "
            f"tau_post={cfg.tau_post_ms}ms, "
            f"eta+/eta-={cfg.eta_plus}/{cfg.eta_minus}"
            + (f", mu={cfg.mu}" if cfg.mu else "")
        ),
        fontsize=11,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def plot_theta_adaptation(
    decay_runs: dict[str, Tensor],
    rate_runs: dict[str, Tensor],
    out_path: str | Path,
    v_th: float = 1.0,
) -> Path:
    """Tampilkan mekanisme tau_theta: seberapa lama neuron "ingat".

    T1B.1 / ADR-0005.

    Args:
        decay_runs: {label: theta (T,)} — theta mulai dari 1.0, TANPA
            spike, jadi kurvanya murni menunjukkan laju peluruhan.
        rate_runs: {label: spike (T,)} — efek ke laju firing pada input
            konstan.
        out_path: file tujuan
        v_th: threshold dasar
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    import numpy as np

    fig, axes = _prep_grid(2, 2, figsize=(11.0, 8.5))
    colors = ["#059669", "#2563eb", "#7c3aed"]

    for k, (label, theta) in enumerate(decay_runs.items()):
        color = colors[k % len(colors)]
        th = theta.detach().cpu().numpy()
        t = torch.arange(len(th)).numpy()
        axes[0][0].plot(t, th, lw=2.0, color=color, label=label)
        axes[0][1].plot(t, v_th + th, lw=2.0, color=color, label=label)

    axes[0][0].axhline(0.0, lw=0.6, color="#94a3b8")
    axes[0][0].set_yscale("log")
    axes[0][0].set_ylabel(r"$\theta$  (log scale)")
    axes[0][0].set_xlabel("time (ms)")
    axes[0][0].set_title(
        "Peluruhan theta tanpa spike\ntau pendek meluruh cepat, "
        "tau jam hampir datar",
        fontsize=10,
    )
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(alpha=0.25, which="both")

    axes[0][1].axhline(v_th, ls="--", lw=0.9, color="#dc2626",
                       label=r"$v_{th}$ (tanpa adaptasi)")
    axes[0][1].set_ylabel(r"effective threshold  $v_{th}+\theta$")
    axes[0][1].set_xlabel("time (ms)")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(alpha=0.25)

    # Panel 3 — spike kumulatif. Raster mentah di 20 detik itu blok
    # solid (beta=0 fire tiap ~35 ms), jadi yang informatif adalah
    # jumlah spike kumulatif: bedanya kelihatan di skala apa pun.
    for k, (label, spikes) in enumerate(rate_runs.items()):
        color = colors[k % len(colors)]
        sp = spikes.detach().cpu().numpy()
        axes[1][0].plot(
            np.cumsum(sp), lw=2.0, color=color,
            label=f"{label} -> {int(sp.sum())} spike",
        )

    axes[1][0].set_xlabel("time (ms)")
    axes[1][0].set_ylabel("spike kumulatif")
    axes[1][0].set_title(
        "beta_theta = rem daya (tau_theta tetap)\nbeta besar -> "
        "jauh lebih sedikit spike",
        fontsize=10,
    )
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(alpha=0.25)

    # Panel 4 — raster biner 0/1, di-zoom ke 1.5 detik supaya ketiga
    # kurva masih bisa dibedakan. Sumbu y dibatasi 0 dan 1 supaya
    # tidak ada interpretasi "tinggi spike = besar".
    zoom = 1500
    for k, (label, spikes) in enumerate(rate_runs.items()):
        color = colors[k % len(colors)]
        sp = spikes.detach().cpu().numpy()[:zoom]
        idx = sp.nonzero()[0]
        axes[1][1].vlines(idx, 0, 1, color=color, lw=0.9, alpha=0.85,
                          label=f"{label} ({len(idx)} spike)")

    axes[1][1].set_ylim(0, 1)
    axes[1][1].set_yticks([0, 1])
    axes[1][1].set_yticklabels(["no spike", "spike"])
    axes[1][1].set_xlim(0, zoom)
    axes[1][1].set_xlabel("time (ms)")
    axes[1][1].set_ylabel("state")
    axes[1][1].set_title(
        f"raster biner, zoom {zoom} ms pertama", fontsize=10
    )
    axes[1][1].legend(fontsize=8, loc="upper right")
    axes[1][1].grid(alpha=0.25)

    fig.suptitle(
        "tau_theta = seberapa lama neuron ingat dia terlalu aktif",
        fontsize=12,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def plot_drive_history(
    plain: list[float],
    homeo: list[float],
    out_path: str | Path,
    ceiling: float | None = None,
) -> Path:
    """Plot drive historis dengan vs tanpa homeostasis.

    **Ini figure utama Phase 2** — visualisasi reproduksi temuan
    Zenke et al. 2013. Tanpa homeostasis, drive total naik monoton
    sampai saturasi; dengan homeostasis, drive tertahan datar.

    Args:
        plain: daftar drive tanpa homeostasis, per epoch
        homeo: daftar drive dengan homeostasis, per epoch
        out_path: file tujuan
        ceiling: plafon keras (`n_in * n_out * w_max`), digambar garis
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = _prep_axes(1, figsize=(10.0, 5.0))
    ax = ax[0]

    ep = torch.arange(1, len(plain) + 1).numpy()
    ax.plot(ep, plain, lw=2.0, color="#dc2626",
            label=f"STDP saja → {plain[-1]:.0f}")
    ax.plot(ep, homeo, lw=2.0, color="#059669",
            label=f"STDP + homeostasis → {homeo[-1]:.0f}")
    ax.fill_between(ep, 0, plain, color="#dc2626", alpha=0.10)
    ax.fill_between(ep, 0, homeo, color="#059669", alpha=0.10)

    if ceiling:
        ax.axhline(ceiling, ls="--", lw=1.0, color="#94a3b8",
                   label=f"plafon keras (n_in·n_out·w_max = {ceiling:.0f})")

    ax.set_xlabel("epoch")
    ax.set_ylabel("total synaptic drive  ΣW")
    ax.set_title(
        "Homeostatic gain > 1 (Zenke 2013) — tanpa rem, drive saturasi penuh",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def plot_column_normalization(
    before: Tensor,
    after: Tensor,
    out_path: str | Path,
    target_sum: float = 78.0,
) -> Path:
    """Visualisasi column normalization (Diehl-Cook).

    Panel kiri: drive per neuron SEBELUM normalisasi (tidak seragam).
    Panel kanan: SESUDAH — semua kolom tepat di `target_sum`.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from rak.homeostasis.scaling import incoming_drive

    d_before = incoming_drive(before).detach().cpu().numpy()
    d_after = incoming_drive(after).detach().cpu().numpy()
    idx = torch.arange(len(d_before)).numpy()

    fig, axes = _prep_grid(2, 2, figsize=(11.0, 8.0))

    axes[0][0].bar(idx, d_before, color="#dc2626", alpha=0.85)
    axes[0][0].axhline(target_sum, ls="--", lw=1.0, color="#64748b",
                       label=f"target = {target_sum:.2f}")
    axes[0][0].set_title(
        f"sebelum — CV={d_before.std() / d_before.mean():.3f}",
        fontsize=10,
    )
    axes[0][0].set_ylabel(r"$\sum_k W[k,j]$")
    axes[0][0].set_xlabel("neuron postsynaptic")
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(alpha=0.25, axis="y")

    axes[1][0].bar(idx, d_after, color="#059669", alpha=0.85)
    axes[1][0].axhline(target_sum, ls="--", lw=1.0, color="#dc2626",
                       label=f"target_sum = {target_sum:.2f}")
    axes[1][0].set_title(
        f"sesudah — CV={d_after.std() / max(d_after.mean(), 1e-9):.3f}",
        fontsize=10,
    )
    axes[1][0].set_xlabel("neuron postsynaptic")
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(alpha=0.25, axis="y")

    # Heatmap: normalisasi kolom mengubah SKALA tiap kolom, bukan
    # STRUKTUR di dalamnya. Kalau baris dalam satu kolom punya pola
    # yang sama sebelum dan sesudah, rasio antar synapse terjaga —
    # itu yang bikin ini homeostasis dan bukan penghapusan memori.
    wb = before.detach().cpu().numpy()
    wa = after.detach().cpu().numpy()
    vmax = float(max(wb.max(), 1e-9))

    im0 = axes[0][1].imshow(wb, aspect="auto", cmap="magma",
                           vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[0][1].set_title(f"sebelum — w_max={vmax:.3f}", fontsize=10)
    axes[0][1].set_xlabel("neuron postsynaptic")
    axes[0][1].set_ylabel("input")
    fig.colorbar(im0, ax=axes[0][1], fraction=0.046)

    im1 = axes[1][1].imshow(wa, aspect="auto", cmap="magma",
                           vmin=0.0, vmax=vmax, interpolation="nearest")
    # PENTING soal judul: kalau target_sum = drive rata-rata, maka w_max
    # hampir TIDAK berubah — yang dikompresi adalah kolom yang di atas
    # rata-rata dan yang dikoreksi yang di bawahnya. Menulis "skala
    # turun" di sini akan salah; yang terjadi adalah "tinggi kolom
    # diratakan, pola di dalam kolom tetap".
    axes[1][1].set_title(
        f"sesudah — kolom diratakan, isi tetap\n"
        f"w_max {vmax:.2f}->{float(wa.max()):.2f}, "
        f"w_min {float(wb.min()):.3f}->{float(wa.min()):.3f}",
        fontsize=10,
    )
    axes[1][1].set_xlabel("neuron postsynaptic")
    axes[1][1].set_ylabel("input")
    fig.colorbar(im1, ax=axes[1][1], fraction=0.046)

    fig.suptitle(
        "Synaptic scaling — kolom dinormalisasi ke konstanta "
        "(Diehl & Cook 2015)", fontsize=11,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path


def _close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)

def plot_association_learning(
    epochs_axis: list[int],
    weight_ab: list[float],
    weight_aa: list[float],
    response_ab: list[float],
    response_bb: list[float],
    v_th: float = 1.0,
    out_path: str | Path = "figures/07_association_learning.png",
) -> Path:
    """Kurva belajar asosiasi A -> B (GATE STDP bisa belajar).

    Panel kiri: bobot synapse yang DIAJARKAN (A->B) naik, sementara
    synapse yang tidak (A->A) tetap datar. Itu bukti credit
    assignment — bukan drift global.

    Panel tengah: bobot melintasi `v_th`. Selama w < v_th, neuron TIDAK
    bisa fire sama sekali: dengan satu kanal input, `v_ss = w`, jadi
    `w = 0.83` tidak pernah menembus `1.0`. Ada titik balik yang jelas
    di sini dan itu yang membuat association "terlihat" di percobaan.

    Panel kanan: respons spike. Naik dari 0 ke 100% saat bobot melewati
    v_th, dan tetap 0% untuk asosiasi yang tidak diajarkan.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = _prep_grid(1, 3, figsize=(13.0, 4.2))

    axes[0][0].plot(epochs_axis, weight_ab, lw=2.2, color="#059669",
                    label="A->B (diajarkan)")
    axes[0][0].plot(epochs_axis, weight_aa, lw=2.2, color="#dc2626",
                    label="A->A (tidak)")
    axes[0][0].axhline(v_th, ls="--", lw=1.0, color="#94a3b8",
                       label=f"v_th = {v_th}")
    axes[0][0].set_xlabel("epoch")
    axes[0][0].set_ylabel("bobot w")
    axes[0][0].set_title("Hanya synapse yang diajarkan yang tumbuh", fontsize=10)
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(alpha=0.25)

    axes[0][1].plot(epochs_axis, weight_ab, lw=2.2, color="#059669")
    axes[0][1].axhline(v_th, ls="--", lw=1.2, color="#dc2626",
                       label=f"v_th = {v_th}")
    axes[0][1].set_yscale("log")
    axes[0][1].set_xlabel("epoch")
    axes[0][1].set_ylabel("bobot w (log)")
    axes[0][1].set_title(
        f"Pelajaran terjadi di w > v_th\n1 kanal input -> v_ss = w",
        fontsize=10,
    )
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(alpha=0.25, which="both")

    axes[0][2].plot(epochs_axis, response_ab, lw=2.2, color="#059669",
                    label="A memicu B")
    axes[0][2].plot(epochs_axis, response_bb, lw=2.2, color="#dc2626",
                    label="B memicu B (tidak diajarkan)")
    axes[0][2].set_ylim(-5, 105)
    axes[0][2].set_xlabel("epoch")
    axes[0][2].set_ylabel("respons spike (%)")
    axes[0][2].set_title("Asosiasi terbentuk, dan selektif", fontsize=10)
    axes[0][2].legend(fontsize=8)
    axes[0][2].grid(alpha=0.25)

    fig.suptitle(
        "GATE: STDP mengubah jaringan karena pengalaman "
        "(asosiasi A->B, 2 neuron x 2 neuron)", fontsize=12,
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _close(fig)
    return path

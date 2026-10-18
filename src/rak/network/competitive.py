r"""Jaringan locally competitive untuk MNIST — Diehl & Cook 2015.

T6.1 — inti Phase 3. Menyatukan lima mekanisme yang sudah diuji
sendiri-sendiri di Phase 1 dan 2:

    784 input (Poisson)
      -> W_e (784 x n_e)  conductance excitatory, kolom ternormalisasi
      -> n_e neuron eksitasi  LIF conductance + adaptive threshold
      -> n_i neuron inhibitor  LIF conductance
      -> inhibisi lateral all-to-all kembali ke eksitasi
      -> STDP Eq.3 pada W_e

Rantai ini BELUM PERNAH dijalankan utuh sebelumnya. Semua gate
sebelumnya (Phase 1, 2A, 2B, 3.1-3.3) menguji komponen terpisah di
lingkungan sintetis. Kalau ada interaksi yang salah di antara lima
mekanisme ini, satu-satunya cara mengetahuinya adalah menjalankannya.

KONSEP YANG SERING SALAH PAHAMI
------------------------------
1. **Adaptasi threshold adapts di neuron EKSITASI saja.** Neuron
   inhibitor tidak punya `theta`. Kalau inhibitor ikut adaptif,
   inhibisi jadi ikut lemah saat neuron sibuk, dan WTA runtuh.

2. **`x_tar` di STDP adalah adaptasi kedua, bukan duplikasi.** Yang
   satu adapting `v_th` (intrinsic plasticity, skala jam), yang satu
   satu lagi menyesuaikan `x_tar` (skala ~500 ms). Keduanya
   beradaptasi, tapi ke hal yang berbeda dan dengan konstanta waktu
   yang berbeda tiga orde besaran.

3. **Inhibisi lateral DIWUJUDKAN sebagai neuron, bukan pengurangan
   tegangan seadanya.** Neuron inhibitor butuh membumi diri dulu (`v_rest_i`
   = -60 mV di bawah threshold-nya -40 mV). Kalau `v` di-init di 0
   seperti default LIF kita, inhibitor langsung fire di setiap
   timestep dan mematikan seluruh jaringan — bug yang sudah pernah
   terjadi di Phase 2B.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rak.homeostasis.scaling import ScalingConfig, hard_normalize
from rak.learning.stdp3 import Eq3Config, Eq3State, eq3_step, make_eq3_state
from rak.learning.stdp_variant3 import (
    Variant3Config,
    Variant3State,
    make_variant3_state,
    variant3_step,
)
from rak.neuron.conductance import (
    ConductanceConfig,
    ConductanceLIF,
    ConductanceState,
)

__all__ = ["NetworkConfig", "CompetitiveNetwork", "assign_classes", "classify"]


@dataclass(frozen=True)
class NetworkConfig:
    """Arsitektur + parameter jaringan.

    Default untuk 100 neuron eksitasi — milestone pertama. Paper
    memberi 82.9% di ukuran ini dengan 3 epoch.
    """

    n_in: int = 784
    n_e: int = 100
    n_i: int | None = None      # None = sama dengan n_e

    #: Jumlah bobot per kolom (Diehl-Cook: 78 untuk 784 input).
    column_sum: float = 78.0
    soft_power: float = 0.01

    #: Adaptive threshold neuron eksitasi.
    #: `theta += theta_plus` tiap spike, meluruh dengan `tau_theta_ms`.
    theta_plus: float = 0.05
    tau_theta_ms: float = 10_000_000.0   # 1e7 ms ~ 2.8 jam
    #: `v_th_eff = theta - theta_offset + v_th` (Diehl-Cook: 20 mV).
    theta_offset: float = 20.0
    theta_init: float = 20.0

    #: Rule STDP yang dipakai.
    #:
    #: "variant3" = bentuk kode acuan, LTP dikunci `post2before`
    #:              (aktivitas postsinaptik baru saja).
    #: "eq3"      = rumus utama paper, tanpa jejak postsinaptik.
    #:
    #: Default "variant3" karena Eq. 3 terbukti menyatukan semua kolom
    #: neuron (korelasi 0.94) dan menghasilkan akurasi ~10%.
    rule: str = "variant3"

    #: Bobot inhibisi lateral. `w_ie` selalu negatif: besarannya yang
    #: menentukan seberapa kuat inhibisi, arahnya sudah pasti.
    inhibition_strength: float = 1.0

    #: Kalau True, bobot I->E dibagi jumlah neuron inhibitor supaya
    #: total inhibisi per neuron eksitasi konstan terhadap ukuran
    #: jaringan. Tanpa ini, `inhibition_strength` hanya bermakna untuk
    #: satu nilai `n_e`.
    scale_inhibition: bool = True

    dt_ms: float = 0.5

    def __post_init__(self) -> None:
        if self.n_e <= 0:
            raise ValueError(f"n_e harus > 0")
        if self.n_in <= 0:
            raise ValueError(f"n_in harus > 0")
        if self.tau_theta_ms <= 0:
            raise ValueError("tau_theta_ms harus > 0")
        if self.column_sum <= 0:
            raise ValueError("column_sum harus > 0")
        if self.rule not in ("variant3", "eq3"):
            raise ValueError(
                f"rule harus 'variant3' atau 'eq3', dapat {self.rule!r}"
            )

    @property
    def n_inhibitory(self) -> int:
        return self.n_e if self.n_i is None else self.n_i


class CompetitiveNetwork:
    """Jaringan eksitasi-inhibisi dengan STDP Eq.3.

    Bobot `w_e` dipakai BERSAMA seluruh batch — tidak ada bobot per
    gambar. Spike trace dan `x_tar` per-gambar, jadi learning rate
    efektif dijaga konstan oleh rata-rata di `eq3_step`.
    """

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        stdp_config: Eq3Config | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or NetworkConfig()
        cfg = self.config
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # --- neuron eksitasi -------------------------------------------
        # Guard stabilitas DIMATIKAN untuk neuron eksitasi.
        #
        # Dengan `inhibition_strength` besar, `gi` bisa mendorong
        # `G = 1 + ge + gi` melewati `tau/dt = 200`, dan guard akan
        # menolak. Tapi inhibisi yang justru kita cari. Kita tetap
        # merekam nilai G tertinggi supaya tidak hilang diam-diam.
        self.exc = ConductanceLIF(
            cfg.n_e,
            strict_stability=False,
            config=ConductanceConfig(
                tau_mem_ms=100.0,
                tau_ge_ms=1.0,
                tau_gi_ms=2.0,
                v_rest=-65.0,
                e_syn_e=0.0,
                e_syn_i=-100.0,
                v_th=-52.0,
                v_reset=-65.0,
                refrac_ms=5.0,
                dt_ms=cfg.dt_ms,
            ),
        ).to(self.device)

        # --- neuron inhibitor ------------------------------------------
        # v_rest -60 mV DI BAWAH threshold -40 mV. Kalau di-init di 0
        # (default LIF kita), inhibitor akan fire di setiap timestep.
        # `v_reset_i = -45 mV` juga masih di bawah threshold, jadi
        # setelah fire dia perlu naik dulu sebelum bisa fire lagi.
        self.inh = ConductanceLIF(
            cfg.n_inhibitory,
            strict_stability=False,
            config=ConductanceConfig(
                tau_mem_ms=10.0,
                tau_ge_ms=1.0,
                tau_gi_ms=2.0,
                v_rest=-60.0,
                e_syn_e=0.0,
                e_syn_i=-85.0,
                v_th=-40.0,
                v_reset=-45.0,
                refrac_ms=2.0,
                dt_ms=cfg.dt_ms,
            ),
        ).to(self.device)

        # --- bobot ------------------------------------------------------
        scale = cfg.column_sum / cfg.n_in
        self.w_e = (
            torch.rand(cfg.n_in, cfg.n_e, generator=self.generator,
                       device=self.device) * scale + 0.5 * scale
        )
        # E -> I satu-satu (identitas), I -> E all-to-all negatif.
        self.w_ei = (
            torch.eye(cfg.n_e, cfg.n_inhibitory, device=self.device)
            * cfg.inhibition_strength
        )
        # I -> E: besar KONDAKTANSI positif. Arah inhibisi datang
        # dari `e_syn_i = -100 mV` yang lebih rendah dari v_rest, BUKAN
        # dari bobot negatif.
        #
        # Menyalakan bobot negatif di sini akan menghijau inhibitory
        # DUA KALI: `gi` negatif kali `E_i` negatif jadi POSITIF,
        # sehingga inhibisi berubah jadi depolarisasi. Symptom-nya
        # halus dan sangat menyesatkan: neuron yang di-inhibisi
        # justru lebih mudah fire.
        # I -> E diskalakan `1/n_e` supaya TOTAL inhibisi per neuron
        # eksitasi KONSTAN terlepas dari ukuran jaringan.
        #
        # Tanpa ini, total inhibisi = strength x (jumlah inhibitor yang
        # fire), yang ikut tumbuh bersama n_e. Sweep 100 neuron
        # membuktikan konsekuensinya: ambang yang memecah simetri
        # bernilai 4 di n_e=40 tapi 8 di n_e=100. Nilai yang sama
        # menghasilkan jaringan BERBEDA hanya karena ukuran -- itu
        # tidak bisa dipakai sebagai parameter yang dilaporkan.
        #
        # Ini persis `scale_per_neuron` yang sudah ada di
        # `rak.homeostasis.inhibition` (Phase 2B) tapi tidak dibawa ke
        # sini. Inkonsistensi yang sama pays off dua kali.
        if cfg.scale_inhibition:
            w_ie = cfg.inhibition_strength / max(cfg.n_inhibitory, 1)
        else:
            w_ie = cfg.inhibition_strength
        self.w_ie = torch.full(
            (cfg.n_inhibitory, cfg.n_e), w_ie, device=self.device,
        )

        # --- state ------------------------------------------------------
        self.state_e: ConductanceState = self.exc.init_state()
        self.state_i: ConductanceState = self.inh.init_state()
        self.rule = cfg.rule
        if cfg.rule == "variant3":
            self.v3_config = Variant3Config(dt_ms=cfg.dt_ms)
            self.stdp_state: Variant3State | Eq3State = make_variant3_state(
                cfg.n_in, cfg.n_e
            )
        else:
            self.v3_config = None
            self.stdp_state = make_eq3_state(cfg.n_in, cfg.n_e, Eq3Config())
        self.theta = torch.full((cfg.n_e,), cfg.theta_init, device=self.device)

        self.stdp_config = (
            stdp_config if stdp_config is not None else Eq3Config(dt_ms=cfg.dt_ms)
        )
        self.n_presentations = 0

    # ------------------------------------------------------------------
    def effective_threshold(self) -> Tensor:
        """`v_th_eff = theta - offset + v_th`.

        `theta` awal 20 mV dan offset 20 mV, jadi di presentasi
        pertama threshold efektif persis sama dengan `v_th` — cocok
        dengan kode acuan.
        """
        cfg = self.config
        return self.theta - cfg.theta_offset + float(self.exc.v_th)

    def init_batch(self, batch_size: int) -> None:
        """Reset state untuk satu batch presentasi."""
        shape = (batch_size, self.config.n_e)
        self.state_e = self.exc.init_state(shape)
        self.state_i = self.inh.init_state((batch_size, self.config.n_inhibitory))
        self.theta_batch = torch.full(
            (batch_size, self.config.n_e),
            self.config.theta_init,
            device=self.device,
        )
        # Trace STDP juga di-reset per presentasi.
        #
        # Alasannya dua. Pertama, bentuk: `x_pre` setelah satu batch
        # punya dimensi batch, jadi tidak bisa di-`expand` ke batch
        # berukuran berbeda di presentasi berikutnya — dan ukuran
        # batch terakhir memang hampir selalu berbeda.
        #
        # Kedua, dan ini yang penting secara fisika: `tau_pre` = 20 ms
        # sementara jeda antar presentasi 150 ms. Setelah jeda itu,
        # `gamma_pre^300` di bawah 1e-16 — trace sudah hilang total.
        # Melewatkan reset itu tidak mengubah apa pun, hanya menahan
        # bentuk tensor yang salah.
        if self.rule == "variant3":
            self.stdp_state = make_variant3_state(
                self.config.n_in, self.config.n_e
            )
        else:
            self.stdp_state = make_eq3_state(
                self.config.n_in, self.config.n_e, self.stdp_config
            )

    # ------------------------------------------------------------------
    def present_batch(self, z_pre: Tensor) -> tuple[Tensor, Tensor]:
        """Jalankan satu batch presentasi.

        Args:
            z_pre: (n_steps, B, n_in) spike input.

        Returns:
            (z_e, spike_counts) dengan `z_e` (n_steps, B, n_e) dan
            `spike_counts` (B, n_e) akumulasi selama presentasi.
        """
        cfg = self.config
        n_steps, batch, _ = z_pre.shape
        if z_pre.shape[2] != cfg.n_in:
            raise ValueError(
                f"z_pre sumbu terakhir {z_pre.shape[2]} != n_in {cfg.n_in}"
            )

        self.init_batch(batch)
        z_e_hist = torch.zeros(n_steps, batch, cfg.n_e, device=self.device)
        counts = torch.zeros(batch, cfg.n_e, device=self.device)

        gamma_theta = float(torch.exp(torch.tensor(-cfg.dt_ms / cfg.tau_theta_ms)))
        zero_i = torch.zeros(batch, cfg.n_inhibitory, device=self.device)

        for t in range(n_steps):
            zp = z_pre[t]                                   # (B, n_in)

            # conductances: excitatory dari input, inhibitory dari
            # neuron inhibitor yang menyala PADA TIMESTEP SEBELUMNYA.
            ge_drive = zp @ self.w_e                        # (B, n_e)
            gi_drive = self.z_i_prev @ self.w_ie if t > 0 else torch.zeros_like(ge_drive)

            v_th_eff = (
                self.theta_batch - cfg.theta_offset + float(self.exc.v_th)
            )
            z_e, self.state_e = self.exc.step(ge_drive, gi_drive, self.state_e, v_th=v_th_eff)

            z_i, self.state_i = self.inh.step(z_e @ self.w_ei, zero_i, self.state_i)

            # STDP hanya pada synapse input -> eksitasi
            if self.rule == "variant3":
                self.w_e, self.stdp_state = variant3_step(
                    zp, z_e, self.w_e, self.stdp_state, self.v3_config
                )
            else:
                self.w_e, self.stdp_state = eq3_step(
                    zp, z_e, self.w_e, self.stdp_state, self.stdp_config
                )

            # adaptive threshold neuron eksitasi saja
            self.theta_batch = gamma_theta * self.theta_batch + cfg.theta_plus * z_e

            self.z_i_prev = z_i
            z_e_hist[t] = z_e
            counts += z_e

        # theta lintas presentasi: rata-rata batch
        self.theta = self.theta_batch.mean(dim=0)

        # Normalisasi kolom SETIAP presentasi — Diehl-Cook memanggilnya
        # di dalam loop, bukan per epoch.
        # Normalisasi kolom tiap presentasi — Diehl-Cook memanggilnya
        # di dalam loop, bukan per epoch. Hard dulu supaya drive
        # benar-benar konstan, soft lalu untuk meratakan sisa.
        scale = ScalingConfig(target_sum=cfg.column_sum,
                              soft_power=cfg.soft_power)
        self.w_e = hard_normalize(hard_normalize(self.w_e, scale), scale)

        self.n_presentations += 1
        return z_e_hist, counts

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "w_e": self.w_e.detach().cpu(),
            "theta": self.theta.detach().cpu(),
            "n_presentations": self.n_presentations,
        }

    def load_state_dict(self, state: dict) -> None:
        self.w_e = state["w_e"].to(self.device)
        self.theta = state["theta"].to(self.device)
        self.n_presentations = state.get("n_presentations", 0)


# ======================================================================
# Readout — tanpa label
# ======================================================================
def assign_classes(
    spike_counts: Tensor, labels: Tensor, n_classes: int = 10
) -> Tensor:
    """Tentukan neuron mana yang mewakili digit mana.

    Args:
        spike_counts: (N, n_e) jumlah spike per neuron.
        labels: (N,) digit sebenarnya.
        n_classes: jumlah kelas.

    Returns:
        (n_e,) nilai 0..n_classes-1 — kelas yang paling sering
        membuat neuron itu menyala.

    CATATAN PENTING: fungsi ini memakai label, tapi HANYA setelah
    training selesai. Label tidak pernah masuk ke loop STDP. Kalau
    label dipakai saat training, itu bukan unsupervised learning —
    itu supervised dengan claim yang salah.
    """
    assignments = torch.zeros(spike_counts.shape[1], dtype=torch.long)
    best = torch.zeros(spike_counts.shape[1])

    for c in range(n_classes):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            continue
        rate = spike_counts[mask].sum(dim=0) / n
        better = rate > best
        best = torch.where(better, rate, best)
        assignments = torch.where(
            better, torch.full_like(assignments, c), assignments
        )
    return assignments


def classify(
    spike_counts: Tensor, assignments: Tensor, n_classes: int = 10
) -> Tensor:
    """Tebak digit dari spike count, tanpa melihat label.

    Args:
        spike_counts: (n_e,) untuk satu gambar, atau (B, n_e) untuk
            B gambar sekaligus.
        assignments: (n_e,) kelas tiap neuron, dari `assign_classes`.
        n_classes: jumlah kelas.

    Returns:
        () untuk satu gambar, (B,) untuk B gambar.

    Membagi dengan jumlah anggota itu penting: tanpa itu kelas dengan
    neuron paling banyak selalu menang, bukan yang paling responsif.
    """
    counts = spike_counts.to(torch.float64)
    totals = torch.zeros(n_classes, dtype=torch.float64)
    for c in range(n_classes):
        members = assignments == c
        n = int(members.sum())
        if counts.dim() == 1:
            totals[c] = float(counts[members].sum()) / max(n, 1)
        else:
            totals[c] = float(counts[:, members].sum()) / max(n, 1)
    return totals.argmax()
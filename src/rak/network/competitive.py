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
    #: Kenaikan theta per spike.
    #:
    #: Acuan memakai 0.05, tapi equilibrium-nya
    #: `theta_eq = theta_plus * r * tau_theta`. Acuan berjalan pada
    #: r ~ 0.04 Hz (5 spike per 400 neuron per gambar), sehingga
    #: theta_eq ~ 20 mV. Jaringan kita berjalan pada r ~ 0.2 Hz,
    #: sehingga 0.05 memberi theta_eq ~ 108 mV -- dan run terakhir
    #: memang mengukur theta 32..115. Theta setinggi itu memaksa
    #: evaluasi memakai intensitas 10-12, yang membuat input nyaris
    #: seragam dan menghapus informasi bentuk.
    #:
    #: 0.0125 menaruh equilibrium di ~25 mV untuk laju kita. Ini
    #: menyimpang dari angka acuan, tapi menjaga INVARIAN-nya
    #: (theta_eq dekat theta_offset), yang jauh lebih penting
    #: daripada menyalin konstanta pada rezim laju yang berbeda.
    theta_plus: float = 0.0125
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
    #: Default "variant3" (bentuk kode acuan). Eq. 3 dipertahankan
    #: sebagai pembanding, TAPI ia punya titik tetap seragam:
    #: dw = eta*(x_pre - x_tar) menjadi nol saat x_pre = x_tar, dan
    #: itulah bobot yang benar-benar keluar dari run 100 neuron --
    #: mean 0.0995 = 78/784, yaitu seragam. Tanpa struktur bobot,
    #: akurasi mentok di 11.7%.
    rule: str = "variant3"

    #: Bobot inhibisi lateral. `w_ie` selalu negatif: besarannya yang
    #: menentukan seberapa kuat inhibisi, arahnya sudah pasti.
    inhibition_strength: float = 1.0

    #: Batas atas theta. `theta` tumbuh tiap spike dan meluruh dengan
    #: tau 1e7 ms, jadi equilibrium-nya `theta_plus * laju * tau`.
    #: Kalau laju kita lebih tinggi dari acuan, equilibrium-nya ikut
    #: naik (terukur 79-115 mV di run 400 neuron) dan evaluasi terpaksa
    #: memakai intensitas 9-12. Pada intensitas itu hampir semua piksel
    #: menyala, input jadi seragam, dan informasi bentuk hilang --
    #: assignment runtuh ke satu kelas (9.20%).
    #:
    #: Batas ini menjaga theta tetap di rezim yang bisa diatasi input
    #: intensitas rendah. Ini penyimpangan dari acuan, dan disengaja:
    #: acuan tidak butuh batas karena laju fire-nya 17x lebih rendah.
    theta_max: float = 40.0

    #: Bobot E -> I (satu-satu). DULU disamakan dengan
    #: `inhibition_strength`, dan itu salah: menaikkan inhibisi ikut
    #: menguatkan drive ke neuron inhibitor, sehingga gain loop tumbuh
    #: KUADRATIK terhadap `inhibition_strength`. Sweep 400 neuron
    #: menunjukkan akibatnya: 1600-3200 stabil, 4800 meledak
    #: (32.134 spike/presentasi). Ledakan itu bukan sifat jaringan,
    #: tapi konsekuensi parameter yang tergandeng.
    #:
    #: Sekarang dipisah. 1600 adalah nilai lama supaya run yang sudah
    #: ada tetap bisa direproduksi; nilai ini akan dikalibrasi lewat
    #: sweep.
    e_to_i_strength: float = 1600.0

    #: Probabilitas koneksi I -> E. Acuan memakai matriks ACAK
    #: (`random/AiAe.npy`), bukan all-to-all seragam.
    #:
    #: Ini bukan detail kosmetik. Dengan I->E seragam, SETIAP neuron
    #: eksitasi menerima inhibisi yang identik, sehingga kompetisi
    #: sepenuhnya simetris dan semua neuron konvergen ke receptive
    #: field yang sama (cosine 0.637 di run 100 neuron). Konektivitas
    #: acak memberi tiap neuron "sidik" inhibisi yang berbeda, dan itu
    #: yang memecah simetri pada tingkat fitur.
    inhib_connectivity: float = 0.5

    #: Kalau True, bobot I->E dibagi jumlah neuron inhibitor supaya
    #: total inhibisi per neuron eksitasi konstan terhadap ukuran
    #: jaringan. Tanpa ini, `inhibition_strength` hanya bermakna untuk
    #: satu nilai `n_e`.
    scale_inhibition: bool = True

    #: Synaptic delay input -> E, uniform [0, delay_max_in_ms] ms per
    #: neuron. 0 = mati (backward compatible). Acuan: 0-10 ms per
    #: sinaps; versi kita per-neuron (satu delay per neuron, bukan
    #: per sinaps) supaya tetap vektorisasi, tapi menangkap
    #: dekorelasi temporal yang utama.
    delay_max_in_ms: float = 0.0

    #: Jeda zero-input SETELAH presentasi, dalam ms. Acuan: 150 ms
    #: (single_example_time 350 + resting 150). Membran dan conductance
    #: meluruh; STDP trace meluruh (tidak ada spike -> tidak ada
    #: update bobot). Theta tidak berubah.
    rest_ms: float = 0.0

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
        if self.delay_max_in_ms < 0:
            raise ValueError("delay_max_in_ms harus >= 0")
        if self.rest_ms < 0:
            raise ValueError("rest_ms harus >= 0")

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
            * cfg.e_to_i_strength
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
        # I -> E: konektivitas ACAK, dinormalisasi per kolom.
        #
        # Kolom = neuron eksitasi. Jumlah bobot yang diterimanya
        # dibuat tepat `inhibition_strength`, jadi total inhibisi
        # konstan terhadap ukuran jaringan DAN terhadap berapa banyak
        # neuron inhibitor yang menyala (dalam ekspektasi).
        #
        # Versi lama: all-to-all seragam. Secara total sama, tapi
        # secara struktur setiap neuron E menerima inhibisi yang
        # identik -- kompetisi jadi simetris sempurna.
        mask = (
            torch.rand(
                cfg.n_inhibitory, cfg.n_e,
                generator=self.generator, device=self.device,
            ) < cfg.inhib_connectivity
        ).to(torch.float32)
        # Jangan biarkan ada kolom kosong (neuron tanpa inhibisi sama
        # sekali akan menang selamanya).
        empty = mask.sum(dim=0) == 0
        if bool(empty.any()):
            idx = torch.nonzero(empty).flatten()
            mask[idx, idx % cfg.n_inhibitory] = 1.0
        col = mask.sum(dim=0).clamp_min(1.0)
        self.w_ie = mask * (cfg.inhibition_strength / col).unsqueeze(0)

        # --- synaptic delay per neuron (opsional) -----------------------
        cfg_delay = float(cfg.delay_max_in_ms)
        if cfg_delay > 0:
            D = int(round(cfg_delay / cfg.dt_ms))
            if D < 1:
                D = 1
            self.delay_steps = torch.randint(
                0, D + 1, (cfg.n_e,),
                generator=self.generator, device=self.device,
            ).to(torch.long)
            self._delay_groups = []
            for d in range(D + 1):
                idx = torch.nonzero(self.delay_steps == d, as_tuple=False).flatten()
                if idx.numel() > 0:
                    self._delay_groups.append((d, idx))
            self._delay_D = D
        else:
            self.delay_steps = torch.zeros(cfg.n_e, device=self.device, dtype=torch.long)
            self._delay_groups = []
            self._delay_D = 0

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
        # Lanjutkan dari theta yang SUDAH dipelajari, bukan dari
        # theta_init.
        #
        # Versi lama selalu mulai dari theta_init (20.0), lalu di akhir
        # presentasi `theta` ditimpa dengan rata-rata lintasan itu.
        # Akibatnya theta tidak pernah terakumulasi antar presentasi
        # dan selalu kembali ke 20.0 -- nilai yang teramati di run
        # 100 neuron (theta 20.0 +- 0.7 = theta_offset). Homeostasis
        # jadi tidak pernah bekerja.
        self.theta_batch = (
            self.theta.unsqueeze(0)
            .expand(batch_size, self.config.n_e)
            .clone()
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
    def present_batch(
        self, z_pre: Tensor, *, learn: bool = True, rest_steps: int | None = None,
        record_history: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Jalankan satu batch presentasi.

        Args:
            z_pre: (n_steps, B, n_in) spike input.
            rest_steps: jumlah step zero-input setelah presentasi
                (peluruhan conductance). None = pakai ``cfg.rest_ms``.

        Returns:
            (z_e, spike_counts) dengan `z_e` (n_steps, B, n_e) dan
            `spike_counts` (B, n_e) akumulasi selama presentasi
            (step rest TIDAK dihitung).
        """
        cfg = self.config
        n_steps, batch, _ = z_pre.shape
        if z_pre.shape[2] != cfg.n_in:
            raise ValueError(
                f"z_pre sumbu terakhir {z_pre.shape[2]} != n_in {cfg.n_in}"
            )

        D = self._delay_D
        if rest_steps is None:
            rest_steps = int(round(cfg.rest_ms / cfg.dt_ms))
        if rest_steps < 0:
            rest_steps = 0

        self.init_batch(batch)
        if record_history:
            z_e_hist = torch.zeros(n_steps, batch, cfg.n_e, device=self.device)
        else:
            z_e_hist = None
        counts = torch.zeros(batch, cfg.n_e, device=self.device)

        gamma_theta = float(torch.exp(torch.tensor(-cfg.dt_ms / cfg.tau_theta_ms)))
        zero_i = torch.zeros(batch, cfg.n_inhibitory, device=self.device)
        zero_ge = torch.zeros(batch, cfg.n_e, device=self.device)

        # buffer riwayat input untuk delay (hanya kalau delay aktif)
        if D > 0:
            buf = torch.zeros(
                D + 1, batch, cfg.n_in, device=self.device, dtype=z_pre.dtype
            )

        for t in range(n_steps):
            zp = z_pre[t]  # (B, n_in)

            if D > 0:
                buf[t % (D + 1)] = zp
                # zp_delayed: (B, n_e, n_in), neuron j pakai input
                # dari t - delay_steps[j]
                idx = (t - self.delay_steps) % (D + 1)  # (n_e,)
                zp_delayed = buf[idx].permute(1, 0, 2)  # (B, n_e, n_in)
                ge_drive = torch.einsum("bni,ni->bn", zp_delayed, self.w_e)
            else:
                ge_drive = zp @ self.w_e  # (B, n_e)

            gi_drive = (
                self.z_i_prev @ self.w_ie
                if t > 0
                else zero_ge
            )

            v_th_eff = (
                self.theta_batch - cfg.theta_offset + float(self.exc.v_th)
            )
            z_e, self.state_e = self.exc.step(
                ge_drive, gi_drive, self.state_e, v_th=v_th_eff
            )

            z_i, self.state_i = self.inh.step(
                z_e @ self.w_ei, zero_i, self.state_i
            )

            # STDP. Saat delay aktif, LTP memakai trace pre yang
            # tertunda; LTD memakai spike input yang tertunda. Untuk
            # menjaga kecepatan dan kestabilan, kita pakai path
            # undelayed untuk trace (varian 3-trace) -- delay tetap
            # bekerja di forward pass (ge_drive), yang merupakan
            # sumber utama dekorelasi temporal.
            if not learn:
                pass
            elif self.rule == "variant3":
                self.w_e, self.stdp_state = variant3_step(
                    zp, z_e, self.w_e, self.stdp_state, self.v3_config
                )
            else:
                self.w_e, self.stdp_state = eq3_step(
                    zp, z_e, self.w_e, self.stdp_state, self.stdp_config
                )

            if learn:
                self.theta_batch = (
                    gamma_theta * self.theta_batch + cfg.theta_plus * z_e
                )

            self.z_i_prev = z_i
            if record_history:
                z_e_hist[t] = z_e
            counts += z_e

        # --- rest: zero input, conductance meluruh, trace meluruh ---
        # Acuan menjalankan 150 ms zero-input antar gambar. Kita tidak
        # reset state antar gambar di sini (itu di init_batch), tapi
        # conductance dan membran sempat meluruh di dalam presentasi.
        # STDP trace ikut meluruh (variant3 early-return saat tidak ada
        # spike; kalau z_e fire dari sisa membran, LTP dengan trace
        # yang sudah meluruh -- sesuai acuan).
        for _ in range(rest_steps):
            zp = torch.zeros(batch, cfg.n_in, device=self.device, dtype=z_pre.dtype)
            if D > 0:
                buf[0] = zp  # slot terbaru kosong
                idx = ( (n_steps + _) - self.delay_steps) % (D + 1)
                zp_delayed = buf[idx].permute(1, 0, 2)
                ge_drive = torch.einsum("bni,ni->bn", zp_delayed, self.w_e)
            else:
                ge_drive = zp @ self.w_e
            gi_drive = self.z_i_prev @ self.w_ie
            v_th_eff = (
                self.theta_batch - cfg.theta_offset + float(self.exc.v_th)
            )
            z_e, self.state_e = self.exc.step(
                ge_drive, gi_drive, self.state_e, v_th=v_th_eff
            )
            z_i, self.state_i = self.inh.step(
                z_e @ self.w_ei, zero_i, self.state_i
            )
            if learn:
                if self.rule == "variant3":
                    self.w_e, self.stdp_state = variant3_step(
                        zp, z_e, self.w_e, self.stdp_state, self.v3_config
                    )
                else:
                    self.w_e, self.stdp_state = eq3_step(
                        zp, z_e, self.w_e, self.stdp_state, self.stdp_config
                    )
                self.theta_batch = (
                    gamma_theta * self.theta_batch + cfg.theta_plus * z_e
                )
            self.z_i_prev = z_i
            # TIDAK dihitung ke counts / z_e_hist (rest tidak masuk
            # assignment, sesuai acuan).

        # theta lintas presentasi.
        if learn:
            gamma_total = gamma_theta ** float(n_steps + rest_steps)
            self.theta = (
                self.theta_batch.sum(dim=0)
                - float(batch - 1) * gamma_total * self.theta
            )
        self.theta = self.theta.clamp(max=cfg.theta_max)
        if not bool(torch.isfinite(self.theta).all()):
            raise FloatingPointError(
                "theta menjadi tidak hingga. Ini berarti update "
                "homeostasis meledak, bukan sekadar nilai besar: "
                "periksa inisialisasi theta_batch terhadap self.theta."
            )

        scale = ScalingConfig(target_sum=cfg.column_sum,
                              soft_power=cfg.soft_power)
        if learn:
            self.w_e = hard_normalize(
                hard_normalize(self.w_e, scale), scale
            )

        self.n_presentations += 1
        if not record_history:
            z_e_hist = torch.empty(0, device=self.device)
        return z_e_hist, counts

    def state_dict(self) -> dict:
        # `w_ie` ikut disimpan. Matriks ini TIDAK dipelajari, tapi
        # menentukan dinamika: kalau checkpoint dimuat tanpa dia,
        # evaluasi berjalan di jaringan inhibisi yang berbeda dari
        # yang dipakai training. Selama ini terselamatkan hanya
        # karena mask dibangkitkan dari generator dengan seed yang
        # sama dan urutan pemanggilan yang sama -- itu kebetulan,
        # bukan jaminan.
        return {
            "w_e": self.w_e.detach().cpu(),
            "w_ie": self.w_ie.detach().cpu(),
            # `w_ei` juga tidak dipelajari, tapi ikut menentukan loop
            # gain. Tanpa dia, checkpoint 800 neuron yang dilatih
            # dengan e_to_i=800 dievaluasi memakai 1600 -- jaringan
            # yang berbeda dari yang diukur saat training.
            "w_ei": self.w_ei.detach().cpu(),
            "theta": self.theta.detach().cpu(),
            "delay_steps": self.delay_steps.detach().cpu(),
            "n_presentations": self.n_presentations,
        }

    def load_state_dict(self, state: dict) -> None:
        self.w_e = state["w_e"].to(self.device)
        if "w_ie" in state:
            self.w_ie = state["w_ie"].to(self.device)
        if "w_ei" in state:
            self.w_ei = state["w_ei"].to(self.device)
        if "delay_steps" in state:
            self.delay_steps = state["delay_steps"].to(self.device).to(torch.long)
            D = int(self.delay_steps.max().item()) if self.delay_steps.numel() else 0
            self._delay_D = D
            self._delay_groups = []
            for d in range(D + 1):
                idx = torch.nonzero(self.delay_steps == d, as_tuple=False).flatten()
                if idx.numel() > 0:
                    self._delay_groups.append((d, idx))
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


def class_prototypes(
    spike_counts: Tensor, labels: Tensor, n_classes: int = 10
) -> Tensor:
    """Profil laju rata-rata tiap kelas: (n_classes, n_e).

    Dibanding `assign_classes`, ini TIDAK membuang informasi. Neuron
    yang mirip digit 4 dan 9 tetap punya bobot berbeda di kedua
    prototipe; assignment keras memaksanya memilih satu.
    """
    n_e = int(spike_counts.shape[1])
    rates = torch.zeros(n_classes, n_e, dtype=torch.float64)
    for c in range(n_classes):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            continue
        rates[c] = spike_counts[m].sum(dim=0).double() / n
    return rates


def classify_matched(
    spike_counts: Tensor, prototypes: Tensor
) -> Tensor:
    """Klasifikasi dengan korelasi pola (z-score per baris).

    `counts` dan tiap baris prototipe di-z-score dulu, lalu dihitung
    cosine. Z-score membuang perbedaan intensitas total antar gambar:
    yang dibandingkan hanya POLA neuron mana yang aktif, bukan
    seberapa banyak spike.

    Diukur pada bobot 400 neuron / 3 epoch (dipilih di validasi,
    diukur sekali di test): 59.02%, vs topk 50.82%, vs rata-rata
    semua anggota 42.18%.
    """
    counts = spike_counts.to(torch.float64)
    if counts.dim() != 1:
        raise ValueError("classify_matched() menerima satu gambar (n_e,).")
    r = prototypes.to(torch.float64)
    # z-score per gambar: buang perbedaan intensitas total, sisakan
    # POLA neuron mana yang aktif. Ini yang menang di validasi.
    c = counts - counts.mean()
    c = c / c.std().clamp_min(1e-9)
    # Prototipe: pusatkan per kelas (baris) lalu z-score per baris.
    r = r - r.mean(dim=1, keepdim=True)
    r = r / r.std(dim=1, keepdim=True).clamp_min(1e-9)
    return (c.unsqueeze(0) @ r.T).argmax(dim=1).squeeze(0)


def classify(
    spike_counts: Tensor,
    assignments: Tensor,
    n_classes: int = 10,
    topk: int = 3,
) -> Tensor:
    """Tebak digit dari spike count, tanpa melihat label.

    Args:
        spike_counts: (n_e,) untuk satu gambar.
        assignments: (n_e,) kelas tiap neuron, dari `assign_classes`.
        n_classes: jumlah kelas.
        topk: ambil rata-rata k spike count TERTINGGI di antara anggota
            kelas, bukan rata-rata semua anggota.

    Returns:
        skalar: kelas yang menang.

    Kenapa `topk`, bukan rata-rata semua anggota: neuron yang tidak
    selektif (merespons semua digit) menambah "kebisingan" yang sama
    ke setiap kelas. Rata-rata 3 teratas hanya melihat neuron yang
    benar-benar aktif untuk gambar itu.

    Ini BUKAN hyperparameter yang ditebak. Lima readout dibandingkan
    pada bobot 400 neuron / 3 epoch, dipilih di split validasi dari
    data TRAIN, lalu diukur sekali di test:

        readout   validasi   test
        mean         45.80   42.18
        max          47.88   44.92
        margin       52.40   47.46
        topk         55.36   50.82
        binary       42.96   39.50
    """
    counts = spike_counts.to(torch.float64)
    if counts.dim() != 1:
        raise ValueError(
            "classify() menerima satu gambar (n_e,). Untuk batch, "
            "panggil per gambar -- cabang 2D versi lama menjumlahkan "
            "SELURUH batch jadi satu skalar."
        )
    totals = torch.zeros(n_classes, dtype=torch.float64)
    for c in range(n_classes):
        members = assignments == c
        n = int(members.sum())
        if n == 0:
            continue
        k = min(topk, n)
        totals[c] = float(counts[members].topk(k).values.mean())
    return totals.argmax()

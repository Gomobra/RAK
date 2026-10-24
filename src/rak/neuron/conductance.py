r"""Neuron LIF dengan synapse conductance-based.

T1B.4 / Phase 3 — Diehl & Cook 2015 memakai synapse conductance-based,
bukan current-based. Ini bukan detail kecil: di current-based synapse,
current yang masuk tidak bergantung pada tegangan neuron, sedangkan di
conductance-based synapse, current = g * (E_syn - v) — jadi makin
hiperpolar, makin besar tarikannya. Effeknya pada fase 3 adalah nyata
dan tidak bisa diabaikan.

ALIABRA YANG MEMBUAT INI MURAH
------------------------------
Persamaan membran dengan conductance:

    tau dv/dt = (v_rest - v) + ge*(E_e - v) + gi*(E_i - v)
              = v_rest + ge*E_e + gi*E_i - v * (1 + ge + gi)
              = G * (v_ss - v)

dengan

    G   = 1 + ge + gi
    v_ss = (v_rest + ge*E_e + gi*E_i) / G

Jadi bentuk implisit yang sudah kita pakai di `LIF` (yang butuh
`alpha` dan `beta`) TETAP PAKAI — yang berubah cuma `alpha` dan `beta`
jadi fungsi dynamis dari conductance:

    beta[t]  = dt * G[t] / tau          # = dt / tau_eff
    alpha[t] = 1 - beta[t]
    v[t+1]   = alpha[t]*v[t] + beta[t]*v_ss[t]

Tidak perlu kelas neuron yang sepenuhnya terpisah, tidak perlu
backward pass, dan tidak ada loop tambahan per timestep selain
penghitungan `G` dan `v_ss`.

SIFAT PENTING YANG BISA DIUJI (bukan asumsi)
--------------------------------------------
Karena `v_ss` adalah **kombinasi konveks** dari `{v_rest, E_e, E_i}`
dengan bobot `{1, ge, gi} / G`, maka:

    min(v_rest, E_e, E_i)  <=  v_ss  <=  max(v_rest, E_e, E_i)

selalu berlaku, untuk semua ge, gi >= 0. Neuron tidak pernah punya
steady-state di luar rentang itu — tidak ada divergensi tersembunyi
yang bisa lolos di phase 3. Ini diuji sebagai properti, bukan
sebagai angka yang kita cek pakai asumsi.

Catatan E neuron vs I neuron: potensial balik sinaps inhibitor
BERBEDA untuk kedua populasi (E: -100 mV, I: -85 mV di Diehl-Cook).
Sifat di atas tetap berlaku untuk masing-masing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

__all__ = ["ConductanceConfig", "ConductanceState", "ConductanceLIF"]


@dataclass(frozen=True)
class ConductanceConfig:
    """Parameter neuron conductance-based.

    Units: potensial dalam mV, waktu dalam ms, conductance dimensionless
    (dalam satuan nS, sudah dinormalisasi).
    """

    #: Time constant membran. Diehl & Cook: 100 ms untuk E, 10 ms untuk I.
    tau_mem_ms: float = 100.0

    #: Time constant conductance excitatory.
    tau_ge_ms: float = 1.0

    #: Time constant conductance inhibitor.
    tau_gi_ms: float = 2.0

    #: Potensial istirahat. Diehl & Cook: -65 mV (E), -60 mV (I).
    v_rest: float = -65.0

    #: Potensial balik synapse eksitasi. Diehl & Cook: 0 mV.
    e_syn_e: float = 0.0

    #: Potensial balik synapse inhibisi. Diehl & Cook: -100 mV (E pop),
    #: -85 mV (I pop).
    e_syn_i: float = -100.0

    v_th: float = -52.0
    v_reset: float = -65.0
    #: Membran di-clamp ke v_reset selama ini (ms).
    refrac_ms: float = 5.0
    #: Spike dicegah selama ini (ms). Acuan Diehl-Cook: clamp 5 ms
    #: tapi spike dicegah 50 ms. Default = refrac_ms (perilaku lama).
    spike_refrac_ms: float | None = None
    dt_ms: float = 0.5

    #: Berapa banyak conductance ditambahkan per spike presinaptik.
    #: Diehl-Cook menambah tepat 1.0 dan bobot W (dalam satuan nS) yang
    #: menentukan skalanya.
    ge_jump: float = 1.0
    gi_jump: float = 1.0

    gamma_ge: float = field(init=False)
    gamma_gi: float = field(init=False)

    def __post_init__(self) -> None:
        for name in ("tau_mem_ms", "tau_ge_ms", "tau_gi_ms", "dt_ms"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} harus > 0, dapat {value}")
        if self.v_reset >= self.v_th:
            raise ValueError(
                f"v_reset ({self.v_reset}) harus < v_th ({self.v_th})"
            )
        if self.refrac_ms < 0:
            raise ValueError(f"refrac_ms harus >= 0, dapat {self.refrac_ms}")
        sr = self.refrac_ms if self.spike_refrac_ms is None else self.spike_refrac_ms
        if sr < 0:
            raise ValueError(f"spike_refrac_ms harus >= 0, dapat {sr}")
        object.__setattr__(self, "spike_refrac_ms", float(sr))

        object.__setattr__(self, "gamma_ge", _exp(-self.dt_ms / self.tau_ge_ms))
        object.__setattr__(self, "gamma_gi", _exp(-self.dt_ms / self.tau_gi_ms))

    # ------------------------------------------------------------------
    @property
    def reversal_bounds(self) -> tuple[float, float]:
        """Rentang steady-state yang tidak bisa dilanggar.

        Lihat module docstring: `v_ss` adalah kombinasi konveks dari
        potensial istirahat dan dua potensial balik, jadi selalu di
        antara nilai-nilai itu.
        """
        candidates = (self.v_rest, self.e_syn_e, self.e_syn_i)
        return min(candidates), max(candidates)

    def stable_ge_bound(self) -> float:
        """Nilai `ge` maksimum sebelum membran tidak stabil.

        Stabilitas bentuk implisit butuh `dt * G <= tau`, dengan
        `G = 1 + ge + gi`. Untuk `gi = 0`:

            ge_max = tau/dt - 1

        Melebihi ini tidak "tidak akuratlagi" — `alpha` berubah tanda,
        membran berosilasi gaining nilai tiap timestep sampai overflow.
        Karena itu di-raise, bukan di-clamp diam-diam.
        """
        return self.tau_mem_ms / self.dt_ms - 1.0


def _exp(x: float) -> float:
    import math

    return math.exp(x)


@dataclass
class ConductanceState:
    """State neuron conductance.

    `ge`/`gi` TIDAK boleh `inf`. Kalau tidak, `0 * inf = nan` dan
    neuron mati diam selamanya — bug yang sudah pernah terjadi sekali
    di Phase 1 pada masking refraktori.
    """

    v: Tensor
    ge: Tensor
    gi: Tensor
    timer: Tensor | None


class ConductanceLIF(nn.Module):
    """LIF dengan synapse conductance-based.

    Batching: `ge_drive` boleh punya dimensi batch di depan, mis.
    (B, n_neurons) untuk B gambar paralel. Broadcasting yang sama
    seperti `LIF`.
    """

    def __init__(
        self,
        n_neurons: int = 1,
        config: ConductanceConfig | None = None,
        *,
        strict_stability: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config or ConductanceConfig(**kwargs)
        self.n_neurons = n_neurons
        self.strict_stability = strict_stability

        self.register_buffer(
            "v_th", torch.as_tensor(self.config.v_th, dtype=torch.float32)
        )
        self.register_buffer(
            "v_reset", torch.as_tensor(self.config.v_reset, dtype=torch.float32)
        )

    def init_state(self, shape: tuple[int, ...] | None = None) -> ConductanceState:
        shape = shape or (self.n_neurons,)
        return ConductanceState(
            v=torch.full(shape, self.config.v_rest, dtype=torch.float32),
            ge=torch.zeros(shape, dtype=torch.float32),
            gi=torch.zeros(shape, dtype=torch.float32),
            timer=torch.full(shape, float("inf"), dtype=torch.float32),
        )

    reset_state = init_state

    # ------------------------------------------------------------------
    def effective(self, ge: Tensor, gi: Tensor) -> tuple[Tensor, Tensor]:
        """`(G, v_ss)` dari conductance saat ini.

        Ini satu-satunya tempat di mana fisika synapseconductance
        tinggal. Semua pembulatan implisit dilakukan di `step`.

        Returns:
            G: konduktansi total dimensionless, selalu >= 1
            v_ss: steady state, selalu di dalam `reversal_bounds`
        """
        cfg = self.config
        g_total = 1.0 + ge + gi
        numerator = cfg.v_rest + ge * cfg.e_syn_e + gi * cfg.e_syn_i
        return g_total, numerator / g_total

    # ------------------------------------------------------------------
    def step(
        self,
        ge_drive: Tensor,
        gi_drive: Tensor,
        state: ConductanceState,
        v_th: Tensor | None = None,
    ) -> tuple[Tensor, ConductanceState]:
        """Satu timestep.

        Args:
            ge_drive: conductance eksitasi yang DITAMBAHKAN timestep ini,
                sudah berbobot W. Shape (n_neurons,) atau (B, n_neurons).
            gi_drive: conductance inhibisi yang ditambahkan, sudah berbobot.
            state: state sebelumnya.
            v_th: threshold EFEKTIF untuk timestep ini. Kalau None,
                pakai `self.v_th` (tetap). Thesis adaptif Diehl-Cook
                melewati lewat parameter ini — bentuknya

                    v_th_eff = theta - offset + v_th

                dengan `theta` yang adaptif dan `offset` konstanta.
                Diehl-Cook memakai offset = 20 mV dengan `theta` awal
                20 mV, sehingga threshold efektif di timestep pertama
                persis sama dengan `v_th`.

        Returns:
            (spikes, new_state)
        """
        cfg = self.config

        # 1. Conductance: decay lalu bump. Decay dulu supaya nilai
        #    mencerminkan "sudah meluruh selama dt".
        ge = cfg.gamma_ge * state.ge + cfg.ge_jump * ge_drive
        gi = cfg.gamma_gi * state.gi + cfg.gi_jump * gi_drive

        # 2. Membran implisit dengan G dan v_ss dinamis
        g_total, v_ss = self.effective(ge, gi)
        beta = cfg.dt_ms * g_total / cfg.tau_mem_ms
        if self.strict_stability:
            self._check_stability(beta, g_total)
        alpha = 1.0 - beta
        v_new = alpha * state.v + beta * v_ss

        # 3. Refraktori
        timer = self._advance_timer(state)
        if cfg.refrac_ms > 0.0 and timer is not None:
            blocked = (timer < cfg.refrac_ms).to(torch.float32)
            v_new = torch.where(
                blocked > 0,
                self.v_reset.expand_as(v_new),
                v_new,
            )

        threshold = self.v_th if v_th is None else v_th
        spikes = (v_new >= threshold).to(torch.float32)
        if timer is not None:
            spikes = torch.where(
                timer >= cfg.spike_refrac_ms, spikes, torch.zeros_like(spikes)
            )

        v_new = torch.where(spikes > 0, self.v_reset.expand_as(v_new), v_new)

        new_timer = None
        if timer is not None:
            new_timer = torch.where(
                spikes > 0,
                torch.zeros_like(timer),
                timer + cfg.dt_ms,
            )

        return spikes, ConductanceState(
            v=v_new, ge=ge, gi=gi, timer=new_timer
        )

    # ------------------------------------------------------------------
    def _check_stability(self, beta: Tensor, g_total: Tensor) -> None:
        """Tolak `dt*G > tau`.

        Bentuk implisit butuh `beta = dt*G/tau <= 1`. Di atas itu `alpha`
        berubah tanda: membran tidak lagi membocor ke arah steady state,
        tapi justru_gain setiap timestep sampai overflow float32.

        Kenapa ini bukan ttl teoritis: dengan `sum_k W[k,j] = 78` dan
        784 input yang semuanya fire bersamaan dalam satu timestep,
        `ge` melonjak 78 per spike dan steady state-nya
        `78/(1-gamma_ge) = 198` — sedangkan batas stabilnya
        `tau/dt - 1 = 199`. Kita mengsitir tepi itu, dan tidak akan
        tahu dari mana kalau tidak dicek.

        Dengan `strict_stability=False`, pengecekan dilewati — hanya
        masuk akal kalau pemanggil sudah tahu conductancenya kecil.
        """
        worst = float(beta.max())
        if worst > 1.0:
            cfg = self.config
            limit = cfg.stable_ge_bound()
            raise ValueError(
                f"Membran tidak stabil: dt*G/tau = {worst:.3f} > 1. "
                f"G tertinggi = {float(g_total.max()):.1f}. Dengan "
                f"dt={cfg.dt_ms}ms, tau={cfg.tau_mem_ms}ms, conductance "
                f"total harus <= {limit:.1f} (tau/dt - 1, dengan gi=0). "
                f"Penyebab biasanya: bobot kolom terlalu besar "
                f"(sum_k W > ~78) atau spike inputarusakan di banyak "
                f"presynaptic neuron sekaligus. Turunkan bobot, "
                f"naikkan tau_mem, atau set strict_stability=False "
                f"kalau memang sudah paham risikonya."
            )

    # ------------------------------------------------------------------
    def _advance_timer(self, state: ConductanceState) -> Tensor | None:
        cfg = self.config
        if cfg.refrac_ms <= 0.0 or state.timer is None:
            return None
        return torch.where(
            state.timer < float("inf"),
            state.timer + cfg.dt_ms,
            torch.full_like(state.timer, float("inf")),
        )

    # ------------------------------------------------------------------
    def simulate(
        self,
        ge_drive: Tensor,
        gi_drive: Tensor,
        *,
        record: bool = False,
    ):
        """Jalankan deret waktu untuk SATU presentasi.

        Args:
            ge_drive: (T, n_neurons)
            gi_drive: (T, n_neurons)
            record: kalau True, kembalikan juga (v, ge, gi).
        """
        cfg = self.config
        ge_drive = ge_drive.reshape(-1, self.n_neurons)
        gi_drive = gi_drive.reshape(-1, self.n_neurons)

        state = self.init_state()
        n_steps = ge_drive.shape[0]
        spikes = torch.zeros(n_steps, self.n_neurons, dtype=torch.float32)
        v_hist = torch.zeros(n_steps, self.n_neurons, dtype=torch.float32)
        ge_hist = torch.zeros(n_steps, self.n_neurons, dtype=torch.float32)
        gi_hist = torch.zeros(n_steps, self.n_neurons, dtype=torch.float32)

        for t in range(n_steps):
            z, state = self.step(ge_drive[t], gi_drive[t], state)
            spikes[t] = z
            if record:
                v_hist[t] = state.v
                ge_hist[t] = state.ge
                gi_hist[t] = state.gi

        if record:
            return spikes, None, (v_hist, ge_hist, gi_hist)
        return spikes, None, None
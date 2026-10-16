r"""Leaky Integrate-and-Fire neuron.

T1.1 — model neuron paling sederhana yang masih realistis.

Catatan desain penting:
Rumus LIF yang biasa ditulis orang di literatur:

    V[t] = V[t-1] + (dt/tau) * (V_rest - V[t-1] + R*I[t])

Itu bentuk EULER EKSPLISIT. Masalahnya: kalau `dt > tau`, coefficient
`(dt/tau) > 1` jadi gain > 1, membran bisa meledak — atau, lebih buruk,
neuron bisa "beku" di threshold lalu gagal menembakkan spike selamanya
karena pembulatan floating point.

Kita pakai bentuk IMPLISIT:

              (1 - dt/tau)         (dt/tau)
    V[t]  =  ----------- * V[t-1] + ------- * (V_rest + R*I[t])
                 alpha                  beta

Dengan `alpha = 1 - dt/tau` dan `beta = dt/tau`, alpha selalu < 1 untuk
tau > 0, jadi stabil untuk semua timestep. Ini bentuk yang dipakai
Norse dan snnTorch. Stability menang atas kesesuaian-notasi-paper.

Units: potensial dalam mV, waktu dalam ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

__all__ = ["LIFConfig", "LIFState", "LIF"]


@dataclass(frozen=True)
class LIFConfig:
    """Parameter LIF. Frozen karena ini konfigurasi, bukan state."""

    tau_mem_ms: float = 20.0
    v_leak: float = 0.0
    v_th: float = 1.0
    v_reset: float = 0.0
    dt_ms: float = 1.0

    #: Periode refraktori dalam ms. 0.0 = tidak ada (default, menjaga
    #: kompatibilitas dengan test Phase 1). Diehl & Cook 2015 pakai 5 ms
    #: untuk neuron eksitasi. Tanpa refractory, neuron bisa fire beruntun
    #: di beberapa timestep berturut-turut dan STDP kehilangan makna:
    #: "kapan" spike terjadi jadi tidak terdefinisi lagi.
    refrac_ms: float = 0.0

    alpha: float = field(init=False)  # 1 - dt/tau, selalu < 1
    beta: float = field(init=False)   # dt/tau

    def __post_init__(self) -> None:
        if self.tau_mem_ms <= 0:
            raise ValueError(f"tau_mem_ms harus > 0, dapat {self.tau_mem_ms}")
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms harus > 0, dapat {self.dt_ms}")
        if self.v_reset >= self.v_th:
            raise ValueError(
                f"v_reset ({self.v_reset}) harus < v_th ({self.v_th}), "
                "kalau tidak neuron akan langsung fire lagi setelah reset"
            )
        if self.dt_ms > self.tau_mem_ms:
            raise ValueError(
                f"dt_ms ({self.dt_ms}) > tau_mem_ms ({self.tau_mem_ms}) "
                "menyebabkan beta > 1 dan alpha < 0: membran akan "
                "berosilasi (berganti tanda) alih-alih membocor, dan "
                "neuron bisa fire di SETIAP timestep tanpa bailout. "
                "Turunkan dt_ms atau naikkan tau_mem_ms."
            )
        if self.refrac_ms < 0:
            raise ValueError(
                f"refrac_ms harus >= 0, dapat {self.refrac_ms}"
            )
        object.__setattr__(self, "beta", self.dt_ms / self.tau_mem_ms)
        object.__setattr__(self, "alpha", 1.0 - self.beta)


@dataclass
class LIFState:
    """State LIF. Berubah tiap timestep, di-reset antar batch."""

    v: Tensor  # membran potential
    #: Waktu (ms) sejak spike terakhir per neuron. `inf` = belum pernah
    #: spike, supaya selalu lolos pemeriksaan refraktori. `None` kalau
    #: refractory period dimatikan (`refrac_ms = 0`).
    timer: Tensor | None


class LIF(nn.Module):
    """Leaky Integrate-and-Fire.

    Non-batched (satu neuron) maupun batched (N neuron paralel) — keduanya
    lewat tensor broadcasting yang sama.

    Konvensi shape:
        `simulate(current)` menerima `current` dengan shape (T, ...) —
        waktu di sumbu PERTAMA. Kalau 1D, diasumsikan (T,) yaitu T timestep
        dari satu fitur.
    """

    def __init__(
        self,
        n_neurons: int = 1,
        config: LIFConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config or LIFConfig(**kwargs)
        self.n_neurons = n_neurons

        # Buffer supaya ikut pindah device / tersimpan di checkpoint,
        # dan bisa di-track kalau nanti dipakai homeostasis.
        self.register_buffer(
            "v_th", torch.as_tensor(self.config.v_th, dtype=torch.float32)
        )
        self.register_buffer(
            "v_reset",
            torch.as_tensor(self.config.v_reset, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def init_state(self, shape: tuple[int, ...] | None = None) -> LIFState:
        """State baru, default v = v_leak, timer = belum pernah spike."""
        shape = shape or (self.n_neurons,)
        v = torch.full(shape, self.config.v_leak, dtype=torch.float32)
        timer = torch.full(shape, float("inf"), dtype=torch.float32)
        return LIFState(v=v, timer=timer)

    reset_state = init_state

    # ------------------------------------------------------------------
    # Dinamika
    # ------------------------------------------------------------------

    def step(self, current: Tensor, state: LIFState) -> tuple[Tensor, LIFState]:
        """Satu timestep.

        Args:
            current: input current, broadcastable ke state.v
            state: LIFState sebelumnya

        Returns:
            (spikes, new_state). spikes = 1.0 kalau neuron fire.
        """
        cfg = self.config
        # `v_leak` adalah potensial istirahat saat input nol. Tanpa
        # suku ini, v_leak cuma jadi offset dan v akan konverge ke 0
        # apa pun nilai v_leak — itu bukan semantik "leak".
        # Steady-state: v_ss = v_leak + I.
        v_new = cfg.alpha * state.v + cfg.beta * (current + cfg.v_leak)

        timer = self._advance_timer(state)

        # Neuron dalam periode refraktori tidak boleh fire, dan v-nya
        # di-clamp ke v_reset (perilaku standar, sama seperti Brian2).
        if cfg.refrac_ms > 0.0 and timer is not None:
            refractory = timer < cfg.refrac_ms
            blocked = refractory.to(torch.float32)
            v_new = blocked * self.v_reset + (1.0 - blocked) * v_new

        spikes = (v_new >= self.v_th).to(torch.float32)
        if timer is not None:
            # `torch.where` BUKAN masking aritmetika: masking `a*x + b*y`
            # dengan y = inf menghasilkan 0*inf = nan dan neuron jadi
            # diam selamanya setelah spike pertama.
            spikes = torch.where(
                timer >= cfg.refrac_ms, spikes, torch.zeros_like(spikes)
            )

        # Reset hanya neuron yang fire — neuron lain tidak boleh di-reset.
        v_reset = torch.where(
            spikes > 0, self.v_reset.expand_as(v_new), v_new
        )

        # Timer di-reset hanya untuk neuron yang baru fire.
        new_timer = None
        if timer is not None:
            new_timer = torch.where(
                spikes > 0,
                torch.zeros_like(timer),
                timer + cfg.dt_ms,
            )

        return spikes, LIFState(v=v_reset, timer=new_timer)

    def _advance_timer(self, state: LIFState) -> Tensor | None:
        """Naikkan timer satu timestep, atau None kalau refractory mati.

        Timer dihitung SEBELUM pemeriksaan spike, jadi kalau neuron baru
        fire di timestep t, di timestep t+1 timer-nya sudah `dt_ms` —
        persis seperti definisi refractory period.
        """
        if self.config.refrac_ms <= 0.0 or state.timer is None:
            return None
        return state.timer + self.config.dt_ms

    def forward(self, current: Tensor, state: LIFState) -> tuple[Tensor, LIFState]:
        return self.step(current, state)

    # ------------------------------------------------------------------
    # Simulasi
    # ------------------------------------------------------------------

    @staticmethod
    def _as_time_major(
        current: Tensor, n_steps: int | None
    ) -> tuple[Tensor, bool]:
        """Normalisasi input jadi (T, ...) dan kasih flag untuk undo."""
        was_1d = current.ndim == 1
        if was_1d:
            # (T,) -> (T, 1): T timestep, satu fitur
            current = current.unsqueeze(-1)
        if n_steps is not None:
            current = current[:n_steps]
        return current, was_1d

    def simulate(
        self,
        current: Tensor,
        state: LIFState | None = None,
        n_steps: int | None = None,
        record: bool = False,
    ) -> tuple[Tensor, LIFState, Tensor | None]:
        """Jalankan simulasi multi timestep.

        Args:
            current: (T, ...) — waktu di sumbu PERTAMA. 1D = (T,).
            state: state awal. Default dari `init_state`.
            n_steps: potong input ke N timestep pertama
            record: kalau True, kembalikan rekaman v (T, ...)

        Returns:
            (spikes, final_state, v_recording)
        """
        current, was_1d = self._as_time_major(current, n_steps)
        T = current.shape[0]
        if state is None:
            state = self.init_state(shape=tuple(current.shape[1:]))

        spikes, v_rec = [], []
        for t in range(T):
            spk, state = self.step(current[t], state)
            if record:
                v_rec.append(state.v)
            spikes.append(spk)

        stacked = torch.stack(spikes, dim=0)
        if was_1d:
            stacked = stacked.squeeze(-1)

        return stacked, state, (torch.stack(v_rec, dim=0) if record else None)

    # ------------------------------------------------------------------
    # Introspeksi
    # ------------------------------------------------------------------

    def firing_rate(
        self, current: float | Tensor, n_steps: int = 1000
    ) -> float:
        """Firing rate stationary (spike/detik) untuk input konstan.

        Dipakai di T1.1.2 — input konstan harus menghasilkan rate yang
        stabil, tidak drift ke 0 atau ke spike setiap timestep.
        """
        cur = torch.full((n_steps,), float(current), dtype=torch.float32)
        spikes, _, _ = self.simulate(cur)
        seconds = n_steps * self.config.dt_ms / 1000.0
        return float(spikes.sum().item()) / seconds

    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"n_neurons={self.n_neurons}, tau_mem={cfg.tau_mem_ms}ms, "
            f"v_th={cfg.v_th}, v_reset={cfg.v_reset}, dt={cfg.dt_ms}ms"
        )
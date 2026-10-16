"""Adaptive LIF — LIF dengan adaptasi threshold.

T1.2 — neuron yang sering fire jadi kurang sensitif.

Mekanisme:
    setiap spike    : theta += beta_theta
    setiap timestep : theta = theta_0 + alpha_theta * (theta - theta_0)

Dimana alpha_theta = 1 - dt/tau_adapt. Effeknya: kalau neuron
konstan fire, theta naik sampai v_eff = v_th + theta naik dan neuron
berhenti fire. Kalau neuron diam, theta decay balik ke theta_0.

Kenapa namanya `tau_adapt` dan bukan `tau_detector`: yang diukur neuron
adalah **seberapa lama ia mengingat bahwa dia terlalu aktif** — itu
adalah adaptasi. "Detector" menyiratkan ada komponen sensor terpisah,
padahal tidak ada; yang ada hanya time constant decay untuk theta.

CATATAN PENTUK untuk Phase 2 (Homeostasis, T1B.1):
Parameter `tau_adapt` di sini default-nya SINGKAT (~5 detik). Itu
benar untuk Adaptive LIF sebagai demo adaptasi.

Untuk homeostasis yang menjaga stabilitas jaringan, tau_adapt harus
JAUH lebih lambat (detik-ke-menit, aturan Zenke et al. 2013). Phase 2
akan mengubah default ini secara eksplisit. Jangan reuse default Phase 1
tanpa menaikkan tau_adapt secara sadar.

Units: potensial mV, waktu ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rak.neuron.lif import LIF, LIFConfig, LIFState

__all__ = ["AdaptiveLIFConfig", "AdaptiveLIFState", "AdaptiveLIF"]


@dataclass(frozen=True)
class AdaptiveLIFConfig(LIFConfig):
    """Config LIF + adaptasi threshold."""

    tau_adapt_ms: float = 5_000.0
    theta_0: float = 0.0
    beta_theta: float = 0.05

    alpha_theta: float = 0.0  # dihitung di __post_init__

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau_adapt_ms <= 0:
            raise ValueError(
                f"tau_adapt_ms harus > 0, dapat {self.tau_adapt_ms}"
            )
        object.__setattr__(
            self, "alpha_theta", 1.0 - self.dt_ms / self.tau_adapt_ms
        )


@dataclass
class AdaptiveLIFState(LIFState):
    """State Adaptive LIF: LIF state + threshold adaptasi."""

    theta: Tensor


class AdaptiveLIF(LIF):
    """LIF dengan adaptive threshold (intrinsic plasticity).

    Subclass LIF, cuma override `step`. Kontrak timestep identik.
    """

    def __init__(
        self,
        n_neurons: int = 1,
        config: AdaptiveLIFConfig | None = None,
        **kwargs,
    ) -> None:
        # Penting: kalau config None, kita yang membangunnya. Kalau
        # diteruskan ke LIF.__init__ apa adanya, dia akan membuat
        # LIFConfig biasa yang tidak punya tau_adapt_ms / theta_0.
        config = config or AdaptiveLIFConfig(**kwargs)
        super().__init__(n_neurons, config=config)

    def init_state(
        self, shape: tuple[int, ...] | None = None
    ) -> AdaptiveLIFState:
        lif_state = super().init_state(shape)
        theta = torch.full(
            lif_state.v.shape, self.config.theta_0, dtype=torch.float32
        )
        return AdaptiveLIFState(
            v=lif_state.v, theta=theta, timer=lif_state.timer
        )

    reset_state = init_state

    def step(
        self, current: Tensor, state: AdaptiveLIFState
    ) -> tuple[Tensor, AdaptiveLIFState]:
        """Satu timestep.

        Urutan itu penting dan deliberate:
          1. decay theta ke arah theta_0
          2. hitung v_eff dari theta SAAT INI, lalu threshold LIF
          3. kalau fire, reset v dan naikkan theta untuk timestep BERIKUTNYA

        Kalau theta dinaikkan sebelum threshold dipakai, neuron yang
        baru fire akan langsung ikut menaikkan threshold-nya sendiri
        di timestep yang sama — itu coupling yang tidak kita mau.

        Refractory period ditangani dengan logika yang sama seperti LIF:
        timer naik satu timestep, neuron dalam periode refraktori
        diblokir spike-nya dan v-nya di-clamp ke v_reset.
        """
        cfg: AdaptiveLIFConfig = self.config  # type: ignore[assignment]

        # 1. decay theta menuju theta_0, mempertahankan fraksi alpha_theta
        theta = cfg.theta_0 + cfg.alpha_theta * (state.theta - cfg.theta_0)

        timer = self._advance_timer(state)

        # 2. LIF dynamics dengan threshold effective
        v_eff = self.v_th + theta
        v_next = cfg.alpha * state.v + cfg.beta * (current + cfg.v_leak)

        if cfg.refrac_ms > 0.0 and timer is not None:
            v_next = torch.where(
                timer < cfg.refrac_ms,
                self.v_reset.expand_as(v_next),
                v_next,
            )

        spikes = (v_next >= v_eff).to(torch.float32)
        if timer is not None:
            # torch.where, bukan masking aritmetika — `a*x + b*y` dengan
            # y = inf menghasilkan 0*inf = nan (lihat LIF.step).
            spikes = torch.where(
                timer >= cfg.refrac_ms, spikes, torch.zeros_like(spikes)
            )
        v_reset = torch.where(
            spikes > 0, self.v_reset.expand_as(v_next), v_next
        )

        # 3. bump theta kalau ada spike
        theta = theta + cfg.beta_theta * spikes

        new_timer = None
        if timer is not None:
            new_timer = torch.where(
                spikes > 0,
                torch.zeros_like(timer),
                timer + cfg.dt_ms,
            )

        return spikes, AdaptiveLIFState(
            v=v_reset, theta=theta, timer=new_timer
        )

    def simulate(
        self,
        current: Tensor,
        state: AdaptiveLIFState | None = None,
        n_steps: int | None = None,
        record: bool = False,
    ):
        """Multi timestep, rekam v DAN theta.

        Return shape sama dengan `LIF.simulate`, tapi `record=True`
        mengembalikan tuple `(v_rec, theta_rec)`.
        """
        current, was_1d = LIF._as_time_major(current, n_steps)
        T = current.shape[0]
        if state is None:
            state = self.init_state(shape=tuple(current.shape[1:]))

        spikes, v_rec, theta_rec = [], [], []
        for t in range(T):
            spk, state = self.step(current[t], state)
            if record:
                v_rec.append(state.v)
                theta_rec.append(state.theta)
            spikes.append(spk)

        stacked = torch.stack(spikes, dim=0)
        if was_1d:
            stacked = stacked.squeeze(-1)

        return (
            stacked,
            state,
            (
                (torch.stack(v_rec, dim=0), torch.stack(theta_rec, dim=0))
                if record
                else None
            ),
        )

    def extra_repr(self) -> str:
        cfg: AdaptiveLIFConfig = self.config  # type: ignore[assignment]
        return (
            f"n_neurons={self.n_neurons}, tau_mem={cfg.tau_mem_ms}ms, "
            f"v_th={cfg.v_th}, tau_adapt={cfg.tau_adapt_ms}ms, "
            f"beta_theta={cfg.beta_theta}, dt={cfg.dt_ms}ms"
        )
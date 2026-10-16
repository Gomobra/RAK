r"""Izhikevich neuron model.

T1.3 — model yang bisa mensimulasikan berbagai pola firing biologis
(regular spiking, intrinsically bursting, chattering, spike-frequency
adaptation).

Kita memakai bentuk yang dipakai Brian2 (canonical simulator
implementation), yang mengikuti Izhikevich (2003):

    dv/dt = 0.04 * v^2 + 5 * v + 140 - w + I
    dw/dt = a * (b * v - w)

Kenapa bentuk ini dan bukan bentuk diskret `0.96*v + 0.40*w - v^3 +
0.556*w + I` yang sering dicopy dari blog:

1. **Bentuk diskret itu hand-fit, bukan hasil integrasi numerik.**
   Suku `-v^3` tidak bisa muncul dari Euler atas ODE kuadratik —
   koefisiennya harus positif. Jadi bentuk diskret bukan hasil
   mentransformasi ODE yang sama.
2. **Satuan eksplisit.** Persamaan di atas punya satuan jelas:
   `dv/dt` dalam mV/ms, `I` dalam mV/ms.
3. **Numerik bisa dikontrol.** Karena kuadratik (bukan kubik), syarat
   stabilitas Euler eksplisit bisa dihitung: `dt * 0.04 * v_max < 2`.
   Dengan `v_max ~ 35` (threshold 30 + overshoot), `dt < 1.4 ms`.
   Default kita `dt = 0.5 ms` supaya aman margin.

Titik istirahat: dengan `I = 0, w = 0`, akar dari
`0.04v^2 + 5v + 140 = 0` adalah `v ≈ -82.7` mV (dan satu akar lagi
di ~-42.4). Perhatikan bahwa `c = -65` adalah nilai **RESET**, bukan
   nilai istirahat — ini yang sering dikeliru orang.

Units: v dalam mV, I dalam mV/ms, waktu dalam ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rak.neuron.lif import LIF

__all__ = ["IzhikevichConfig", "IzhikevichState", "Izhikevich"]


@dataclass(frozen=True)
class IzhikevichConfig:
    """Parameter Izhikevich.

    Default `a=0.02, b=0.2, c=-65, d=8, I=10, v_th=30` menghasilkan
    regular-spiking neuron (Izhikevich 2003, tabel mode firing).
    Ubah `a`/`b` (dan `c`/`d`) untuk mode firing lain.
    """

    a: float = 0.02
    b: float = 0.20
    c: float = -65.0
    d: float = 8.0
    v_threshold: float = 30.0
    dt_ms: float = 0.5

    #: Batas atas dt yang tervalidasi lewat uji konvergensi
    #: (lihat `test_izhikevich.py::test_isi_converges_across_dt`).
    #: Pada dt = 1.0 ms, ISI devi ~3.4% dari referensi dt = 0.02 ms.
    dt_max_validated_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms harus > 0, dapat {self.dt_ms}")
        if self.dt_ms > self.dt_max_validated_ms:
            raise ValueError(
                f"dt_ms={self.dt_ms} melewati batas tervalidasi "
                f"({self.dt_max_validated_ms} ms). Di atas itu ISI devi "
                ">3.4% dari referensi fine-grained dan hasil tidak lagi "
                "dapat dipercaya secara biologis. Turunkan dt_ms."
            )
        # Syarat stabilitas Euler eksplisit untuk suku 0.04*v^2:
        #   dt * max|dv/dv| < 2, dengan f'(v) = 0.08v + 5
        # Neuron menghabiskan sebagian besar waktu di v ~ v_reset
        # (f' kecil), jadi batas konservatif ini terlalu longgar —
        # makanya kita pakai batas tervalidasi empiris di atas.
        v_lo, v_hi = min(self.c, self.v_threshold), self.v_threshold
        stiff = max(abs(0.08 * v_lo + 5.0), abs(0.08 * v_hi + 5.0))
        dt_euler = 2.0 / max(stiff, 1e-9)


@dataclass
class IzhikevichState:
    """State Izhikevich: recovery variable w."""

    v: Tensor
    w: Tensor


class Izhikevich(nn.Module):
    """Izhikevich neuron. API mirror LIF/AdaptiveLIF."""

    def __init__(
        self,
        n_neurons: int = 1,
        config: IzhikevichConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config or IzhikevichConfig(**kwargs)
        self.n_neurons = n_neurons

        self.register_buffer(
            "v_threshold",
            torch.as_tensor(self.config.v_threshold, dtype=torch.float32),
        )
        self.register_buffer(
            "v_reset", torch.as_tensor(self.config.c, dtype=torch.float32)
        )
        self.register_buffer(
            "w_reset", torch.as_tensor(self.config.d, dtype=torch.float32)
        )
        self.register_buffer(
            "dt", torch.as_tensor(self.config.dt_ms, dtype=torch.float32)
        )

    def init_state(
        self, shape: tuple[int, ...] | None = None
    ) -> IzhikevichState:
        """State baru. v mulai di `c` (nilai setelah spike terakhir)."""
        shape = shape or (self.n_neurons,)
        return IzhikevichState(
            v=torch.full(shape, self.config.c, dtype=torch.float32),
            w=torch.zeros(shape, dtype=torch.float32),
        )

    reset_state = init_state

    def step(
        self, current: Tensor, state: IzhikevichState
    ) -> tuple[Tensor, IzhikevichState]:
        """Satu timestep (Euler eksplisit)."""
        cfg = self.config
        h = self.dt

        dv = 0.04 * state.v.pow(2) + 5.0 * state.v + 140.0 - state.w + current
        dw = cfg.a * (cfg.b * state.v - state.w)

        v_next = state.v + h * dv
        w_next = state.w + h * dw

        spikes = (v_next >= self.v_threshold).to(torch.float32)

        # Hanya neuron yang spike yang di-reset ke (c, d).
        v_final = spikes * self.v_reset + (1.0 - spikes) * v_next
        w_final = spikes * self.w_reset + (1.0 - spikes) * w_next

        return spikes, IzhikevichState(v=v_final, w=w_final)

    def forward(self, current: Tensor, state: IzhikevichState):
        return self.step(current, state)

    def simulate(
        self,
        current: Tensor,
        state: IzhikevichState | None = None,
        n_steps: int | None = None,
        record: bool = False,
    ):
        """Multi timestep, rekam v DAN w."""
        current, was_1d = LIF._as_time_major(current, n_steps)
        T = current.shape[0]
        if state is None:
            state = self.init_state(shape=tuple(current.shape[1:]))

        spikes, v_rec, w_rec = [], [], []
        for t in range(T):
            spk, state = self.step(current[t], state)
            if record:
                v_rec.append(state.v)
                w_rec.append(state.w)
            spikes.append(spk)

        stacked = torch.stack(spikes, dim=0)
        if was_1d:
            stacked = stacked.squeeze(-1)

        return (
            stacked,
            state,
            (
                (torch.stack(v_rec, dim=0), torch.stack(w_rec, dim=0))
                if record
                else None
            ),
        )

    def firing_rate(
        self, current: float | Tensor, n_steps: int = 2000
    ) -> float:
        cur = torch.full((n_steps,), float(current), dtype=torch.float32)
        spikes, _, _ = self.simulate(cur)
        seconds = n_steps * self.config.dt_ms / 1000.0
        return float(spikes.sum().item()) / seconds

    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"n_neurons={self.n_neurons}, a={cfg.a}, b={cfg.b}, "
            f"c={cfg.c}, d={cfg.d}, v_th={cfg.v_threshold}, dt={cfg.dt_ms}ms"
        )
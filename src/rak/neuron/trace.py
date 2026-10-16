"""Spike trace dan spike-train generator.

T1.5 — fondasi STDP. Spike trace `x[t] = gamma * x[t-1] + z[t]` adalah
satu-satunya state yang dibutuhkan STDP trace-based, jadi dibangun di
Phase 1 supaya SpikeLayer tidak perlu dibongkar di Phase 2.

Konvensi waktu:
- `dt` dalam milidetik (ms), konsisten dengan Izhikevich
- `gamma = exp(-dt / tau)` adalah faktor decay per timestep

Bentuk tensor:
    input  : (..., T)   — sumbu waktu selalu TERAKHIR
    output : (..., T)   — shape sama
"""

from __future__ import annotations

import torch

__all__ = [
    "decay_to_target",
    "spike_trace",
    "poisson_spike_train",
    "PoissonEncoder",
]


def decay_to_target(tau_ms: float, dt_ms: float = 1.0) -> float:
    """Faktor decay per timestep untuk time constant `tau_ms`.

    Kita pakai bentuk eksponensial eksak (`exp(-dt/tau)`) daripada
    aproksimasi Euler (`1 - dt/tau`), karena yang eksak tidak pernah
    negatif walau `dt` besar, dan steady-state-nya exact — sama dengan
    yang tertulis di rumus STDP paper.
    """
    if tau_ms <= 0:
        raise ValueError(f"tau_ms harus > 0, dapat {tau_ms}")
    return float(torch.exp(torch.tensor(-dt_ms / tau_ms)).item())


def spike_trace(
    spikes: torch.Tensor,
    tau_ms: float,
    dt_ms: float = 1.0,
    initial: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Trace eksponensial dari spike train.

    Trace di timestep t hanya bergantung spike sampai t (bukan spike
    masa depan) — ini yang bikin STDP bisa dihitung online tanpa
    menyimpan histori penuh.

    Args:
        spikes: tensor spike biner, shape (..., T) — sumbu waktu terakhir
        tau_ms: time constant trace
        dt_ms: timestep
        initial: nilai trace awal (scalar atau broadcast ke spikes[..., 0])

    Returns:
        Tensor dengan shape sama seperti `spikes`, dtype float32.

    Contoh:
        >>> s = torch.tensor([[0., 1., 0., 0.]])
        >>> x = spike_trace(s, tau_ms=50., dt_ms=1.)
        >>> float(x[0, 1])          # spike menaikkan trace ke 1
        1.0
        >>> bool((x[0, 1:] < 1.0).all())   # lalu monoton turun
        True
    """
    if spikes.shape[-1] == 0:
        return torch.zeros_like(spikes, dtype=torch.float32)

    gamma = decay_to_target(tau_ms, dt_ms)
    feature_shape = spikes.shape[:-1]

    if not isinstance(initial, torch.Tensor):
        initial = torch.as_tensor(float(initial))
    state = initial.expand(feature_shape).to(torch.float32).clone()

    out = torch.empty(spikes.shape, dtype=torch.float32)
    for t in range(spikes.shape[-1]):
        state = gamma * state + spikes[..., t].to(torch.float32)
        out[..., t] = state

    return out


def poisson_spike_train(
    rates_hz: torch.Tensor,
    duration_ms: float,
    dt_ms: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Poisson spike train dari firing rate (Hz).

    Digunakan untuk input encoder MNIST (Diehl & Cook 2015 pakai
    Poisson, bukan Bernoulli, supaya noise-nya realistis).

    Args:
        rates_hz: firing rate dalam Hz, shape (..., P)
        duration_ms: durasi presentasi
        dt_ms: timestep simulasi
        generator: torch generator untuk reproducibility

    Returns:
        Tensor biner shape (..., P, T) dengan T = round(duration_ms/dt_ms)

    Catatan: implementasi pakai Bernoulli per timestep, bukan sampler
    Poisson. Untuk `dt` kecil (1 ms) keduanya hampir identik secara
    statistik, dan Bernoulli jauh lebih cepat di CPU.
    """
    n_steps = max(1, round(duration_ms / dt_ms))
    p = rates_hz * dt_ms / 1000.0
    p = p.clamp(0.0, 1.0).unsqueeze(-1)          # (..., P, 1)
    p = p.expand(*p.shape[:-1], n_steps)

    return torch.rand(p.shape, generator=generator).lt(p).to(torch.float32)


class PoissonEncoder:
    """Encoder Poisson yang menyimpan generator supaya bisa di-seed."""

    def __init__(self, seed: int | None = None) -> None:
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset state generator. Panggil sebelum run baru."""
        if seed is not None:
            self.generator.manual_seed(seed)

    def encode(
        self,
        values: torch.Tensor,
        duration_ms: float = 350.0,
        dt_ms: float = 1.0,
        max_rate_hz: float = 63.75,
    ) -> torch.Tensor:
        """Nilai kontinu [0, 1] -> spike train.

        `max_rate_hz` default 63.75 Hz mengikuti Diehl & Cook 2015:
        max pixel intensity 255 dibagi 4 = 63.75 Hz.
        `duration_ms` default 350 ms juga mengikuti paper yang sama.

        Args:
            values: tensor dalam [0, 1], shape (..., P)

        Returns:
            Spike train shape (..., P, T)
        """
        rates = values.clamp(0.0, 1.0) * max_rate_hz
        return poisson_spike_train(
            rates,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            generator=self.generator,
        )
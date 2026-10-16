"""SpikeLayer — kumpulan neuron + spike train processing.

T1.4 — forward pass = simulasi T timestep.

Bentuk tensor (waktu di sumbu PERTAMA, konsisten dengan LIF.simulate):
    input  : (T, batch, n_in)
    output : (T, batch, n_out)
    state  : dict dengan `v`, `trace_pre`, `trace_post`, (theta)

Trace (`trace_pre`, `trace_post`) sudah ada di Phase 1 walaupun T1.5
mendasarkannya ke Phase 2 — SpikeLayer harus punya tempat untuk
menyimpannya sekarang, kalau tidak Phase 2 akan perlu mengubah
signature `forward()`.

Bobot dense di Phase 1. Ini disengaja: T1.4.5 mengukur throughput
baseline yang jujur (dense timestep simulation). Sparse connectivity
(FITUR 3.1) masuk Phase 4 dengan backend dan benchmark sendiri.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rak.neuron.adaptive_lif import AdaptiveLIFConfig
from rak.neuron.lif import LIF
from rak.neuron.trace import decay_to_target

__all__ = ["SpikeLayer"]


class SpikeLayer(nn.Module):
    """Lapisan neuron spiking.

    Args:
        n_in: jumlah input neuron
        n_out: jumlah output neuron
        neuron: "lif" | "adaptive"
        tau_mem_ms: time constant membran
        tau_trace_ms: time constant spike trace (untuk STDP Phase 2)
        w_init_std: std inisialisasi bobot
        seed: seed inisialisasi (NFR-05)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        neuron: str = "lif",
        tau_mem_ms: float = 20.0,
        tau_trace_ms: float = 50.0,
        w_init_std: float = 0.3,
        dt_ms: float = 1.0,
        max_rate_hz: float = 63.75,
        calib_value: float = 0.12,
        refrac_ms: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if neuron not in ("lif", "adaptive"):
            raise ValueError(
                f"neuron harus 'lif' atau 'adaptive', dapat {neuron!r}"
            )
        self.n_in = n_in
        self.n_out = n_out
        self.neuron_kind = neuron
        self.dt_ms = dt_ms
        self.tau_trace_ms = tau_trace_ms
        self.max_rate_hz = max_rate_hz

        self.is_adaptive = neuron == "adaptive"

        if self.is_adaptive:
            from rak.neuron.adaptive_lif import AdaptiveLIF

            self.neuron: LIF = AdaptiveLIF(
                n_out,
                tau_mem_ms=tau_mem_ms,
                v_th=1.0,
                v_reset=0.0,
                tau_adapt_ms=5_000.0,
                beta_theta=0.05,
                dt_ms=dt_ms,
                refrac_ms=refrac_ms,
            )
        else:
            self.neuron = LIF(
                n_out,
                tau_mem_ms=tau_mem_ms,
                v_th=1.0,
                v_reset=0.0,
                dt_ms=dt_ms,
                refrac_ms=refrac_ms,
            )

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        w = torch.randn(n_in, n_out, generator=gen) * w_init_std
        w = w.clamp(min=0.0)  # excitatory only di Phase 1
        # Kalibrasi supaya neuron benar-benar bisa fire, dan responsnya
        # *graded* terhadap intensitas input.
        #
        # Untuk bentuk implisit LIF, steady-state membrane = input current
        # persis (v_ss = beta*I / (1 - alpha) = I). Current itself ∝ nilai
        # input. Jadi v(input) = v_target * (input / calib_value).
        #
        # `calib_value` = intensitas RATA-RATA seluruh piksel (termasuk
        # yang nol), yang harusnya menghasilkan v sedikit di atas
        # threshold. Untuk MNIST angka itu ~0.13 — itu yang kita pakai
        # sebagai default.
        #
        # Kenapa bukan intensitas per-piksel yang menyala (~0.5) atau
        # full scale (1.0): current per timestep dijumlahkan dari SEMUA
        # piksel yang spike pada timestep itu. Yang menentukan apakah
        # neuron fire adalah mean, bukan piksel termbright. Kalau
        # dikalibrasi ke 0.5 atau 1.0, sebagian besar neuron akan diam
        # dan klaim sparsity jadi tidak berlaku.
        #
        # Tanpa kalibrasi sama sekali, bobot random menghasilkan v_ss
        # di bawah threshold dan layer diam total (density 0).
        self.v_target = 1.5 * float(self.neuron.v_th)
        p_max = self.max_rate_hz * dt_ms / 1000.0
        expected_w = self.v_target / max(p_max * n_in * calib_value, 1e-9)
        current_mean = float(w.mean().item())
        if current_mean > 1e-9:
            w = w * (expected_w / current_mean)
        self.weight = nn.Parameter(w)

        self.register_buffer(
            "trace_gamma",
            torch.as_tensor(decay_to_target(tau_trace_ms, dt_ms)),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int) -> dict[str, Tensor]:
        """State baru untuk batch tertentu."""
        zeros = torch.zeros(batch_size, self.n_out, dtype=torch.float32)
        state: dict[str, Tensor] = {
            "v": torch.full(
                (batch_size, self.n_out),
                self.neuron.config.v_leak,
                dtype=torch.float32,
            ),
            "trace_pre": torch.zeros(batch_size, self.n_in, dtype=torch.float32),
            "trace_post": zeros.clone(),
        }
        if self.neuron.config.refrac_ms > 0.0:
            state["timer"] = torch.full(
                (batch_size, self.n_out), float("inf"), dtype=torch.float32
            )
        if self.is_adaptive:
            state["theta"] = torch.full(
                (batch_size, self.n_out),
            self.neuron.config.theta_0,  # type: ignore[attr-defined]
                dtype=torch.float32,
            )
        return state

    reset_state = init_state

    def reset_state_(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """In-place reset state dict yang sudah ada."""
        state["v"].fill_(self.neuron.config.v_leak)
        state["trace_pre"].zero_()
        state["trace_post"].zero_()
        if "theta" in state:
            state["theta"].zero_()
        if "timer" in state:
            state["timer"].fill_(float("inf"))
        return state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        spikes_in: Tensor,
        state: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """Jalankan spike train input melalui layer.

        Args:
            spikes_in: (T, batch, n_in) spike biner
            state: dict state, atau None untuk inisialisasi otomatis

        Returns:
            (spikes_out, state, aux)
              spikes_out : (T, batch, n_out)
              state      : dict baru (input state tidak di-mutate)
              aux        : snapshot trace + synaptic current per timestep
        """
        if spikes_in.ndim != 3:
            raise ValueError(
                f"spikes_in harus 3D (T, batch, n_in), dapat {spikes_in.ndim}D "
                f"dengan shape {tuple(spikes_in.shape)}"
            )
        if spikes_in.shape[-1] != self.n_in:
            raise ValueError(
                f"n_in layer = {self.n_in}, tapi input punya "
                f"{spikes_in.shape[-1]} channel"
            )

        T, B, _ = spikes_in.shape
        if state is None:
            state = self.init_state(B)
        if state["v"].shape != (B, self.n_out):
            raise ValueError(
                f"state v punya shape {tuple(state['v'].shape)}, "
                f"harus (batch={B}, n_out={self.n_out})"
            )

        cfg = self.neuron.config
        gamma = float(self.trace_gamma)
        w = self.weight
        v_th = self.neuron.v_th
        v_reset = self.neuron.v_reset

        is_adaptive = self.is_adaptive
        if is_adaptive:
            acfg: AdaptiveLIFConfig = cfg  # type: ignore[assignment]
            alpha_th = acfg.alpha_theta
            theta_0 = acfg.theta_0
            beta_theta = acfg.beta_theta

        v = state["v"]
        trace_pre = state["trace_pre"]
        trace_post = state["trace_post"]
        theta = state.get("theta") if is_adaptive else None

        use_refrac = cfg.refrac_ms > 0.0
        refrac = cfg.refrac_ms
        timer = state.get("timer")
        if use_refrac and timer is None:
            timer = torch.full_like(v, float("inf"))

        spikes_out: list[Tensor] = []
        syn_current: list[Tensor] = []

        for t in range(T):
            # Sumbu waktu PERTAMA, jadi slice langsung di dim 0.
            pre = spikes_in[t]  # (B, n_in)

            # Trace pre SEBELUM propagate — ini yang bikin STDP bisa
            # menghitung kontribusi spike pre pada timestep yang sama.
            trace_pre = gamma * trace_pre + pre
            cur = pre @ w  # (B, n_out)

            # Neuron dalam periode refraktori: input diabaikan dan
            # spike diblokir. Dipakai torch.where, BUKAN masking
            # aritmetika — `a*x + b*y` dengan y = inf menghasilkan
            # 0*inf = nan dan neuron diam selamanya.
            if use_refrac and timer is not None:
                timer = timer + cfg.dt_ms
                cur = torch.where(timer < refrac, torch.zeros_like(cur), cur)
                can_spike = timer >= refrac
            else:
                can_spike = None

            if is_adaptive:
                theta = alpha_th * theta + (1.0 - alpha_th) * theta_0
                v_next = cfg.alpha * v + cfg.beta * (cur + cfg.v_leak)
                spikes = (v_next >= v_th + theta).float()
                theta = theta + beta_theta * spikes
            else:
                v_next = cfg.alpha * v + cfg.beta * (cur + cfg.v_leak)
                spikes = (v_next >= v_th).float()

            if can_spike is not None:
                spikes = torch.where(
                    can_spike, spikes, torch.zeros_like(spikes)
                )

            v = torch.where(spikes > 0, v_reset.expand_as(v_next), v_next)
            if timer is not None:
                timer = torch.where(
                    spikes > 0, torch.zeros_like(timer), timer
                )

            trace_post = gamma * trace_post + spikes
            spikes_out.append(spikes)
            syn_current.append(cur)

        new_state: dict[str, Tensor] = {
            "v": v,
            "trace_pre": trace_pre,
            "trace_post": trace_post,
        }
        if theta is not None:
            new_state["theta"] = theta
        if timer is not None:
            new_state["timer"] = timer

        aux = {
            "trace_pre_final": trace_pre,
            "trace_post_final": trace_post,
            "syn_current": torch.stack(syn_current, dim=0),
        }

        return torch.stack(spikes_out, dim=0), new_state, aux

    def simulate(self, spikes_in: Tensor, state=None):
        """Alias `forward` dengan return signature lebih eksplisit."""
        return self.forward(spikes_in, state)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        values: Tensor,
        n_steps: int,
        max_rate_hz: float = 63.75,
        seed: int | None = None,
    ) -> Tensor:
        """Nilai kontinu [0,1] -> spike train (T, batch, n_in).

        Rate coding. `n_steps` adalah sumber kebenaran jumlah timestep;
        durasi fisiknya = `n_steps * dt_ms`.

        Default `max_rate_hz = 63.75` mengikuti Diehl & Cook 2015
        (max pixel intensity 255 dibagi 4) — lihat ADR-0003.
        """
        from rak.neuron.trace import poisson_spike_train

        rates = values.clamp(0, 1) * max_rate_hz
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        spk = poisson_spike_train(
            rates,
            duration_ms=n_steps * self.dt_ms,
            dt_ms=self.dt_ms,
            generator=gen,
        )
        # (batch, n_in, T) -> (T, batch, n_in)
        return spk.movedim(-1, 0)

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"n_in={self.n_in}, n_out={self.n_out}, neuron={self.neuron_kind}, "
            f"tau_trace={self.tau_trace_ms}ms, "
            f"synapses={self.n_in * self.n_out:,}"
        )
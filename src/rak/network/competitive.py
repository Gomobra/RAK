"""Competitive network for MNIST (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rak.homeostasis.intrinsic import adaptive_threshold_step
from rak.homeostasis.scaling import hard_normalize
from rak.learning.stdp import make_trace, trace_step
from rak.neuron.conductance import ConductanceConfig, ConductanceLIF, ConductanceState
from rak.neuron.trace import make_trace as make_spike_trace


@dataclass
class NetworkConfig:
    n_in: int = 784
    n_e: int = 400
    n_i: int | None = None
    column_sum: float = 78.0
    soft_power: float = 0.01
    theta_plus: float = 0.05
    tau_theta_ms: float = 1e7
    theta_offset: float = 20.0
    theta_init: float = 20.0
    theta_max: float = 200.0
    use_reference_params: bool = True
    inhibition_strength: float = 17.0
    e_to_i_strength: float = 10.4
    inhib_connectivity: float = 0.5
    x_to_i_connectivity: float = 0.1
    x_to_i_weight: float = 0.2
    delay_max_in_ms: float = 10.0
    rest_ms: float = 150.0
    dt_ms: float = 0.5
    rule: str = "variant3"

    @property
    def n_inhibitory(self) -> int:
        return self.n_e if self.n_i is None else self.n_i


class CompetitiveNetwork:
    def __init__(self, config: NetworkConfig | None = None, seed: int = 0):
        self.config = config or NetworkConfig()
        cfg = self.config
        self.generator = torch.Generator().manual_seed(seed)

        # Excitatory neuron (tau_mem=100ms)
        self.exc = ConductanceLIF(
            cfg.n_e, strict_stability=False,
            config=ConductanceConfig(
                tau_mem_ms=100.0, tau_ge_ms=1.0, tau_gi_ms=2.0,
                v_rest=-65.0, e_syn_e=0.0, e_syn_i=-100.0,
                v_th=-52.0, v_reset=-65.0, refrac_ms=5.0,
                spike_refrac_ms=50.0 if cfg.use_reference_params else 5.0,
                dt_ms=cfg.dt_ms,
            ),
        )

        # Inhibitory neuron (tau_mem=10ms)
        self.inh = ConductanceLIF(
            cfg.n_inhibitory, strict_stability=False,
            config=ConductanceConfig(
                tau_mem_ms=10.0, tau_ge_ms=1.0, tau_gi_ms=2.0,
                v_rest=-60.0, e_syn_e=0.0, e_syn_i=-85.0,
                v_th=-40.0, v_reset=-45.0, refrac_ms=2.0,
                dt_ms=cfg.dt_ms,
            ),
        )

        # Weights
        if cfg.use_reference_params:
            self.w_e = (torch.rand(cfg.n_in, cfg.n_e, generator=self.generator) + 0.01) * 0.3
            self.w_ei = torch.eye(cfg.n_e, cfg.n_inhibitory) * cfg.e_to_i_strength
            mask = (torch.rand(cfg.n_inhibitory, cfg.n_e, generator=self.generator) < 0.9).to(torch.float32)
            empty = mask.sum(dim=0) == 0
            if bool(empty.any()):
                idx = torch.nonzero(empty).flatten()
                mask[idx, idx % cfg.n_inhibitory] = 1.0
            col = mask.sum(dim=0).clamp_min(1.0)
            self.w_ie = mask * (cfg.inhibition_strength / col).unsqueeze(0)
            mask_xi = (torch.rand(cfg.n_in, cfg.n_inhibitory, generator=self.generator) < cfg.x_to_i_connectivity).to(torch.float32)
            empty_xi = mask_xi.sum(dim=1) == 0
            if bool(empty_xi.any()):
                idx = torch.nonzero(empty_xi).flatten()
                mask_xi[idx, idx % cfg.n_inhibitory] = 1.0
            self.w_xi = mask_xi * cfg.x_to_i_weight
        else:
            scale = cfg.column_sum / cfg.n_in
            self.w_e = torch.rand(cfg.n_in, cfg.n_e, generator=self.generator) * scale + 0.5 * scale
            self.w_ei = torch.eye(cfg.n_e, cfg.n_inhibitory) * cfg.e_to_i_strength
            mask = (torch.rand(cfg.n_inhibitory, cfg.n_e, generator=self.generator) < cfg.inhib_connectivity).to(torch.float32)
            empty = mask.sum(dim=0) == 0
            if bool(empty.any()):
                idx = torch.nonzero(empty).flatten()
                mask[idx, idx % cfg.n_inhibitory] = 1.0
            col = mask.sum(dim=0).clamp_min(1.0)
            self.w_ie = mask * (cfg.inhibition_strength / col).unsqueeze(0)
            mask_xi = (torch.rand(cfg.n_in, cfg.n_inhibitory, generator=self.generator) < cfg.x_to_i_connectivity).to(torch.float32)
            empty_xi = mask_xi.sum(dim=1) == 0
            if bool(empty_xi.any()):
                idx = torch.nonzero(empty_xi).flatten()
                mask_xi[idx, idx % cfg.n_inhibitory] = 1.0
            self.w_xi = mask_xi * cfg.x_to_i_weight

        # State
        self.state_e = self.exc.init_state()
        self.state_i = self.inh.init_state()
        self.z_i_prev = None
        self.theta = torch.full((cfg.n_e,), cfg.theta_init)
        self.theta_batch = None
        self.n_presentations = 0

        # Delay
        cfg_delay = float(cfg.delay_max_in_ms)
        if cfg_delay > 0:
            D = max(1, int(round(cfg_delay / cfg.dt_ms)))
            self.delay_steps = torch.randint(0, D + 1, (cfg.n_e,), generator=self.generator).to(torch.long)
            self._delay_D = D
        else:
            self.delay_steps = torch.zeros(cfg.n_e, dtype=torch.long)
            self._delay_D = 0

        # STDP state (3-trace)
        gamma_pre = torch.exp(torch.tensor(-cfg.dt_ms / 20.0))
        gamma_post1 = torch.exp(torch.tensor(-cfg.dt_ms / 20.0))
        gamma_post2 = torch.exp(torch.tensor(-cfg.dt_ms / 40.0))
        self.stdp_state = {
            "pre": torch.zeros(cfg.n_in),
            "post1": torch.zeros(cfg.n_e),
            "post2": torch.zeros(cfg.n_e),
            "gamma_pre": float(gamma_pre),
            "gamma_post1": float(gamma_post1),
            "gamma_post2": float(gamma_post2),
        }
        self.nu_pre = 1e-4
        self.nu_post = 0.01
        self.w_min = 0.0
        self.w_max = 1.0

    def init_batch(self, batch_size):
        self.state_e = self.exc.init_state((batch_size, self.config.n_e))
        self.state_i = self.inh.init_state((batch_size, self.config.n_inhibitory))
        self.z_i_prev = None
        self.theta_batch = self.theta.unsqueeze(0).expand(batch_size, -1).clone()

    def present_batch(self, z_pre, learn=True, rest_steps=None, record_history=False):
        cfg = self.config
        n_steps, batch, _ = z_pre.shape
        D = self._delay_D
        if rest_steps is None:
            rest_steps = int(round(cfg.rest_ms / cfg.dt_ms))
        if rest_steps < 0:
            rest_steps = 0

        self.init_batch(batch)
        if record_history:
            z_e_hist = torch.zeros(n_steps, batch, cfg.n_e)
        else:
            z_e_hist = torch.empty(0)
        counts = torch.zeros(batch, cfg.n_e)

        gamma_theta = float(torch.exp(torch.tensor(-cfg.dt_ms / cfg.tau_theta_ms)))
        zero_i = torch.zeros(batch, cfg.n_inhibitory)
        zero_ge = torch.zeros(batch, cfg.n_e)

        # STDP state (per-batch)
        pre_trace = self.stdp_state["pre"].unsqueeze(0).expand(batch, -1).clone()
        post1 = self.stdp_state["post1"].unsqueeze(0).expand(batch, -1).clone()
        post2 = self.stdp_state["post2"].unsqueeze(0).expand(batch, -1).clone()
        gamma_pre = self.stdp_state["gamma_pre"]
        gamma_post1 = self.stdp_state["gamma_post1"]
        gamma_post2 = self.stdp_state["gamma_post2"]

        L = D + 1 if D > 0 else 1
        if D > 0:
            buf_zp = torch.zeros(L, batch, cfg.n_in, dtype=z_pre.dtype)
            buf_pre = torch.zeros(L, batch, cfg.n_in)

        for t in range(n_steps):
            zp = z_pre[t]
            if D > 0:
                buf_zp[t % L] = zp
                pre_d_cur = gamma_pre * pre_trace
                buf_pre[t % L] = pre_d_cur
                idx = (t - self.delay_steps) % L
                zp_d = buf_zp[idx].permute(1, 0, 2)
                pre_d_delayed = buf_pre[idx].permute(1, 0, 2)
                ge_drive = torch.einsum("bji,ij->bj", zp_d, self.w_e)
            else:
                ge_drive = zp @ self.w_e

            gi_drive = self.z_i_prev @ self.w_ie if (self.z_i_prev is not None and t > 0) else zero_ge
            # fix: z_i_prev is None initially
            if self.z_i_prev is None:
                gi_drive = zero_ge

            v_th_eff = self.theta_batch - cfg.theta_offset + (-52.0)
            z_e, self.state_e = self.exc.step(ge_drive, gi_drive, self.state_e, v_th=v_th_eff)
            z_i, self.state_i = self.inh.step(z_e @ self.w_ei + zp @ self.w_xi, zero_i, self.state_i)

            if learn:
                post1_d = gamma_post1 * post1
                post2_d = gamma_post2 * post2
                fired = (z_e > 0)
                if D > 0:
                    ltp = torch.einsum("bni,bn->in", pre_d_delayed, post2_d * fired.to(pre_d_delayed.dtype))
                    ltd = torch.einsum("bni,bn->in", zp_d, post1_d)
                else:
                    pre_d = gamma_pre * pre_trace
                    ltp = pre_d.T @ (post2_d * fired.to(pre_d.dtype))
                    ltd = zp.T @ post1_d
                dw = self.nu_post * ltp - self.nu_pre * ltd
                self.w_e = (self.w_e + dw).clamp(self.w_min, self.w_max)
                post1 = post1_d + z_e
                post2 = post2_d + z_e
                pre_trace = gamma_pre * pre_trace + zp

                gamma_t = float(torch.exp(torch.tensor(-cfg.dt_ms / cfg.tau_theta_ms)))
                self.theta_batch = gamma_t * self.theta_batch + cfg.theta_plus * z_e

            self.z_i_prev = z_i
            if record_history:
                z_e_hist[t] = z_e
            counts += z_e

        # rest steps
        for r in range(rest_steps):
            zp = torch.zeros(batch, cfg.n_in)
            if D > 0:
                ge_drive = zero_ge  # zp=0 -> no delayed spikes
            else:
                ge_drive = zp @ self.w_e
            gi_drive = self.z_i_prev @ self.w_ie if self.z_i_prev is not None else zero_ge
            v_th_eff = self.theta_batch - cfg.theta_offset + (-52.0)
            z_e, self.state_e = self.exc.step(ge_drive, gi_drive, self.state_e, v_th=v_th_eff)
            z_i, self.state_i = self.inh.step(z_e @ self.w_ei + zp @ self.w_xi, zero_i, self.state_i)
            if learn:
                post1_d = gamma_post1 * post1
                post2_d = gamma_post2 * post2
                fired = (z_e > 0)
                pre_d = gamma_pre * pre_trace
                ltp = pre_d.T @ (post2_d * fired.to(pre_d.dtype))
                ltd = zp.T @ post1_d
                dw = self.nu_post * ltp - self.nu_pre * ltd
                self.w_e = (self.w_e + dw).clamp(self.w_min, self.w_max)
                post1 = post1_d + z_e
                post2 = post2_d + z_e
                pre_trace = gamma_pre * pre_trace + zp
                gamma_t = float(torch.exp(torch.tensor(-cfg.dt_ms / cfg.tau_theta_ms)))
                self.theta_batch = gamma_t * self.theta_batch + cfg.theta_plus * z_e
            self.z_i_prev = z_i

        # theta cross-presentation
        if learn:
            gamma_total = float(torch.exp(torch.tensor(-cfg.dt_ms / cfg.tau_theta_ms))) ** float(n_steps + rest_steps)
            self.theta = self.theta_batch.sum(dim=0) - float(batch - 1) * gamma_total * self.theta
        self.theta = self.theta.clamp(max=cfg.theta_max)

        # column normalization
        if learn:
            scale = cfg.column_sum
            col_sums = self.w_e.sum(dim=0).clamp_min(1e-9)
            self.w_e = self.w_e * (scale / col_sums).unsqueeze(0)

        self.n_presentations += 1
        if not record_history:
            z_e_hist = torch.empty(0)
        return z_e_hist, counts

    def state_dict(self):
        return {
            "w_e": self.w_e.detach().clone(),
            "w_ie": self.w_ie.detach().clone(),
            "w_ei": self.w_ei.detach().clone(),
            "w_xi": self.w_xi.detach().clone(),
            "theta": self.theta.detach().clone(),
            "delay_steps": self.delay_steps.detach().clone(),
            "n_presentations": self.n_presentations,
        }

    def load_state_dict(self, state):
        self.w_e = state["w_e"]
        if "w_ie" in state:
            self.w_ie = state["w_ie"]
        if "w_ei" in state:
            self.w_ei = state["w_ei"]
        if "w_xi" in state:
            self.w_xi = state["w_xi"]
        self.theta = state["theta"]
        if "delay_steps" in state:
            self.delay_steps = state["delay_steps"].to(torch.long)
            D = int(self.delay_steps.max().item()) if self.delay_steps.numel() else 0
            self._delay_D = D
        self.n_presentations = state.get("n_presentations", 0)
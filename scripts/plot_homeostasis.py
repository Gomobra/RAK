"""Visualisasi homeostasis + STDP — Phase 2 deliverable."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rak.homeostasis.intrinsic import adaptive_threshold_step
from rak.learning.stdp import stdp_window


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Adaptive threshold: theta naik saat spike, decay saat diam
    ax = axes[0]
    theta = torch.zeros(1)
    thetas = []
    spikes = []
    for t in range(200):
        sp = 1.0 if (t % 20 == 0) else 0.0
        theta = adaptive_threshold_step(theta, torch.tensor([sp]), beta=0.5, alpha=0.01)
        thetas.append(float(theta[0]))
        spikes.append(sp)
    ax.plot(thetas, color="steelblue", lw=1.2, label="θ")
    t_sp = [i for i, s in enumerate(spikes) if s > 0]
    ax.vlines(t_sp, 0, max(thetas) * 1.1, color="darkorange", lw=0.8, alpha=0.5, label="spike")
    ax.set_title("Adaptive threshold (homeostasis)", fontsize=9)
    ax.set_xlabel("timestep"); ax.set_ylabel("θ")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # STDP window
    ax = axes[1]
    dts, win = stdp_window(1.0, max_ms=50.0)
    ax.plot(dts.numpy(), win.numpy(), color="steelblue", lw=1.2)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_title("STDP window (additive)", fontsize=9)
    ax.set_xlabel("Δt (ms)"); ax.set_ylabel("Δw")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = "experiments/exp_002_homeostasis/homeostasis_stdp.png"
    import os
    os.makedirs("experiments/exp_002_homeostasis", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
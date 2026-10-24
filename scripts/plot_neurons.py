"""Visualisasi semua neuron Phase 1 — LIF, Adaptive LIF, Izhikevich."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rak.neuron.adaptive_lif import AdaptiveLIF
from rak.neuron.izhikevich import IZHIKEVICH_MODES, Izhikevich
from rak.neuron.lif import LIF


def main():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    # LIF
    lif = LIF(1)
    I = torch.zeros(300); I[50:150] = 20.0
    sp, v = lif.simulate(I, record=True)
    ax = axes[0, 0]
    ax.plot(v.numpy(), color="steelblue", lw=1)
    ax.vlines(sp.nonzero(as_tuple=True)[0].numpy(), -90, -80, color="darkorange", lw=0.8)
    ax.set_title(f"LIF ({int(sp.sum())} spikes)", fontsize=9)
    ax.set_ylim(-90, -40); ax.grid(alpha=0.2)

    # Adaptive LIF
    alif = AdaptiveLIF(1)
    I = torch.full((500,), 15.0)
    sp, v = alif.simulate(I, record=True)
    ax = axes[0, 1]
    ax.plot(v.numpy()[:, 0], color="steelblue", lw=1)
    ax.plot(alif.theta.numpy() if hasattr(alif, "theta") else [0], color="red", lw=0.8, ls="--", label="θ")
    t_sp = sp.nonzero(as_tuple=True)[0].numpy()
    ax.vlines(t_sp, -90, -80, color="darkorange", lw=0.6)
    ax.set_title(f"Adaptive LIF ({int(sp.sum())} spikes)", fontsize=9)
    ax.set_ylim(-90, -40); ax.grid(alpha=0.2); ax.legend(fontsize=7)

    # 6 Izhikevich modes
    for ax, (name, cfg) in zip(axes.flat[2:], IZHIKEVICH_MODES.items()):
        neuron = Izhikevich(1, config=cfg)
        I = torch.full((300, 1), 10.0)
        sp, v = neuron.simulate(I, record=True)
        ax.plot(v.numpy()[:, 0], color="steelblue", lw=0.8)
        t_sp = sp.nonzero(as_tuple=True)[0].numpy()
        ax.vlines(t_sp, -90, -80, color="darkorange", lw=0.5)
        ax.set_title(f"{name} ({int(sp.sum())})", fontsize=8)
        ax.set_ylim(-90, 45); ax.grid(alpha=0.2)

    fig.suptitle("Phase 1 — Neuron Models", fontsize=12)
    fig.tight_layout()
    out = "experiments/exp_001_neuron/neurons_all.png"
    import os
    os.makedirs("experiments/exp_001_neuron", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
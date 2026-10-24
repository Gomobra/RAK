"""Visualisasi Izhikevich 6 mode firing — Phase 1 deliverable."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rak.neuron.izhikevich import IZHIKEVICH_MODES, Izhikevich


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (name, cfg) in zip(axes.flat, IZHIKEVICH_MODES.items()):
        neuron = Izhikevich(1, config=cfg)
        I = torch.full((300, 1), 10.0)
        sp, v = neuron.simulate(I, record=True)
        ax.plot(v.numpy()[:, 0], color="steelblue", lw=0.8)
        t_sp = sp.nonzero(as_tuple=True)[0].numpy()
        ax.vlines(t_sp, -90, -80, color="darkorange", lw=0.6)
        ax.set_title(f"{name} ({int(sp.sum())} spikes)", fontsize=9)
        ax.set_ylim(-90, 45)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    fig.suptitle("Izhikevich — 6 mode firing", fontsize=11)
    fig.tight_layout()
    out = "experiments/exp_001_neuron/izhikevich_modes.png"
    import os
    os.makedirs("experiments/exp_001_neuron", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
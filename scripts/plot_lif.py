"""Visualisasi LIF neuron — Phase 1 deliverable."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rak.neuron.lif import LIF


def main():
    lif = LIF(1)
    # Input: step current (naik di t=50, turun di t=150)
    I = torch.zeros(300)
    I[50:150] = 20.0
    sp, v = lif.simulate(I, record=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(v.numpy(), color="steelblue", lw=1.2)
    ax1.axhline(-52, color="red", ls="--", lw=0.8, label="v_th")
    ax1.axhline(-65, color="gray", ls=":", lw=0.8, label="v_rest")
    ax1.set_ylabel("V (mV)")
    ax1.set_title("LIF neuron — membran potential")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    t_spikes = sp.nonzero(as_tuple=True)[0].numpy()
    ax2.vlines(t_spikes, 0, 1, color="darkorange", lw=1.0)
    ax2.set_ylabel("spike")
    ax2.set_xlabel("timestep (ms)")
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_title(f"spike train ({len(t_spikes)} spikes)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = "experiments/exp_001_neuron/lif_membrane.png"
    import os
    os.makedirs("experiments/exp_001_neuron", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
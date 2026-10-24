"""Render receptive fields dari checkpoint RAK.

Mengurutkan neuron per kelas hasil assign, lalu per kekuatan
selektivitas (std bobot). Tujuannya: mata bisa memeriksa apakah
yang dipelajari memang bentuk digit, bukan blob.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rak.network.competitive import CompetitiveNetwork, NetworkConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="receptive_fields.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--cols", type=int, default=10)
    a = ap.parse_args()

    blob = torch.load(a.weights, map_location="cpu", weights_only=False)
    state = blob["net"]
    n_e = int(state["w_e"].shape[1])
    net = CompetitiveNetwork(NetworkConfig(n_e=n_e), seed=1)
    net.load_state_dict(state)
    w = net.w_e.detach().cpu().double().numpy()  # (784, n_e)
    assignments = np.asarray(blob.get("assignments", []))
    if assignments.size != n_e:
        assignments = np.zeros(n_e, dtype=int)

    sd = w.std(axis=0)
    per_class = a.per_class
    rows_per_class = (per_class + a.cols - 1) // a.cols
    n_rows = 10 * rows_per_class

    fig = plt.figure(figsize=(a.cols * 0.7 + 0.9, n_rows * 0.7 + 0.9))
    gs = fig.add_gridspec(n_rows, a.cols + 1, width_ratios=[0.55] + [1] * a.cols,
                          wspace=0.05, hspace=0.05)

    for c in range(10):
        idx = np.where(assignments == c)[0]
        idx = idx[np.argsort(-sd[idx])][:per_class]
        r0 = c * rows_per_class
        ax_lab = fig.add_subplot(gs[r0:r0 + rows_per_class, 0])
        ax_lab.axis("off")
        ax_lab.text(0.5, 0.5, str(c), ha="center", va="center",
                    fontsize=15, fontweight="bold")
        for k in range(per_class):
            rr = r0 + k // a.cols
            cc = k % a.cols
            ax = fig.add_subplot(gs[rr, cc + 1])
            if k < len(idx):
                ax.imshow(w[:, idx[k]].reshape(28, 28), cmap="hot_r",
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.2)

    title = a.title or f"RAK receptive fields — {n_e} neuron"
    fig.suptitle(title, fontsize=11)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"tersimpan: {a.out}")


if __name__ == "__main__":
    main()
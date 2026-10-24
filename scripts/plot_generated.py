"""Visualisasi generated images dari SNN weights."""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rak.generate import generate_samples, reconstruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="generated.png")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    blob = torch.load(a.weights, map_location="cpu", weights_only=False)
    state = blob["net"]
    w = state["w_e"]  # (784, n_e)

    imgs = generate_samples(w, n=a.n, seed=0)

    n_cols = min(a.n, 5)
    n_rows = (a.n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(-1)
    for i in range(a.n):
        axes[i].imshow(imgs[i].numpy(), cmap="hot_r", interpolation="nearest")
        axes[i].set_title(f"neuron {i}", fontsize=8)
        axes[i].axis("off")
    for i in range(a.n, len(axes)):
        axes[i].axis("off")
    fig.suptitle("Generated images (receptive field templates)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"tersimpan: {a.out}")


if __name__ == "__main__":
    main()
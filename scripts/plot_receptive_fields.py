r"""Render receptive fields dari checkpoint RAK.

Mengurutkan neuron per kelas hasil `assign_classes`, lalu per
kekuatan selektivitas (std bobot). Tujuannya satu: mata bisa
memeriksa apakah yang dipelajari memang bentuk digit, bukan blob.

Contoh:
    python scripts/plot_receptive_fields.py \
        --weights runs/mnist/weights.pt \
        --out experiments/exp_003_mnist/receptive_fields.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="receptive_fields.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--cols", type=int, default=20)
    a = ap.parse_args(argv)

    blob = torch.load(a.weights, map_location="cpu", weights_only=False)
    w = blob["net"]["w_e"].double().numpy()
    assignments = np.asarray(blob.get("assignments", []))
    if assignments.size != w.shape[1]:
        assignments = np.zeros(w.shape[1], dtype=int)

    sd = w.std(axis=0)
    per_class = a.per_class
    n_rows = 10 * (per_class // a.cols)
    if per_class % a.cols:
        n_rows += 10

    fig = plt.figure(figsize=(a.cols * 0.62 + 0.9, n_rows * 0.62 + 0.9))
    gs = fig.add_gridspec(n_rows, a.cols + 1, width_ratios=[0.55] + [1] * a.cols,
                          wspace=0.05, hspace=0.05)
    rows_per_class = per_class // a.cols

    for c in range(10):
        idx = np.where(assignments == c)[0]
        idx = idx[np.argsort(-sd[idx])][:per_class]
        # Baris pertama kelas c.
        r0 = c * rows_per_class
        ax_lab = fig.add_subplot(gs[r0 : r0 + rows_per_class, 0])
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
            for s in ax.spines.values():
                s.set_linewidth(0.2)

    title = a.title or (
        f"RAK receptive fields — {w.shape[1]} neuron, "
        f"urut kelas lalu selektivitas"
    )
    fig.suptitle(title, fontsize=11)
    fig.legend(
        [Line2D([0], [0], color="w")], [" "], frameon=False, loc="lower right"
    )
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"tersimpan: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

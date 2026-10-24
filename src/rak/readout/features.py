"""Ekstraksi fitur spike count, termasuk ensemble beberapa model."""

from __future__ import annotations

import torch
from torch import Tensor

from rak.data.encode import EncoderConfig
from rak.network.competitive import CompetitiveNetwork
from rak.network.train import present_adaptive

__all__ = ["collect_counts", "ensemble_counts"]


def collect_counts(
    net: CompetitiveNetwork,
    images: Tensor,
    *,
    trials: int = 1,
    batch: int = 64,
    seed: int = 0,
    enc: EncoderConfig | None = None,
) -> Tensor:
    """Spike count satu model untuk setiap gambar.

    `trials > 1` menjumlahkan beberapa presentasi Poisson untuk gambar
    yang sama. Encoder stokastik; rata-rata mengurangi varians sampling
    tanpa mengubah bobot (`learn=False`).
    """
    if trials < 1:
        raise ValueError(f"trials harus >= 1, dapat {trials}")
    enc = enc or EncoderConfig()
    gen = torch.Generator().manual_seed(seed)
    out: list[Tensor] = []
    n = int(images.shape[0])
    for start in range(0, n, batch):
        end = min(start + batch, n)
        total = None
        for _ in range(trials):
            counts, _ = present_adaptive(
                net, images[start:end], enc, gen, learn=False
            )
            total = counts if total is None else total + counts
        out.append(total)
    return torch.cat(out, dim=0)


def ensemble_counts(
    nets: list[CompetitiveNetwork],
    images: Tensor,
    *,
    trials: int = 1,
    batch: int = 64,
    seed: int = 0,
    enc: EncoderConfig | None = None,
) -> Tensor:
    """Gabungkan fitur beberapa model: (n_images, sum(n_e))."""
    if not nets:
        raise ValueError("ensemble butuh minimal satu model")
    parts = [
        collect_counts(
            net, images, trials=trials, batch=batch, seed=seed + i, enc=enc
        )
        for i, net in enumerate(nets)
    ]
    return torch.cat(parts, dim=1)

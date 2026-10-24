"""Feature extraction (spike counts) for readout."""

from __future__ import annotations

import torch
from torch import Tensor

from rak.data.encode import EncoderConfig, present_adaptive


def collect_counts(net, images, trials=1, batch=64, seed=0, enc=None):
    enc = enc or EncoderConfig()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, images.shape[0], batch):
        e = min(s + batch, images.shape[0])
        total = None
        for _ in range(trials):
            c, _ = present_adaptive(net, images[s:e], enc, gen, learn=False)
            total = c if total is None else total + c
        out.append(total)
    return torch.cat(out, dim=0)


def ensemble_counts(nets, images, trials=1, batch=64, seed=0, enc=None):
    parts = [collect_counts(net, images, trials=trials, batch=batch, seed=seed + i, enc=enc) for i, net in enumerate(nets)]
    return torch.cat(parts, dim=1)
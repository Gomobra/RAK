"""Training loop (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rak.data.encode import EncoderConfig, present_adaptive
from rak.data.mnist import MnistSplit, load_mnist
from rak.network.competitive import CompetitiveNetwork, NetworkConfig


@dataclass
class TrainConfig:
    n_e: int = 400
    epochs: int = 1
    batch_size: int = 128
    max_repeat: int = 10
    seed: int = 1
    inhibition_strength: float = 17.0
    e_to_i_strength: float = 10.4
    theta_plus: float = 0.05
    theta_max: float = 200.0
    delay_max_in_ms: float = 10.0
    rest_ms: float = 150.0
    use_reference_params: bool = True
    rule: str = "variant3"
    out_dir: str = "runs/mnist"


def train(data, cfg: TrainConfig, net=None):
    if net is None:
        net = CompetitiveNetwork(
            NetworkConfig(
                n_e=cfg.n_e,
                inhibition_strength=cfg.inhibition_strength,
                e_to_i_strength=cfg.e_to_i_strength,
                theta_plus=cfg.theta_plus,
                theta_max=cfg.theta_max,
                delay_max_in_ms=cfg.delay_max_in_ms,
                rest_ms=cfg.rest_ms,
                use_reference_params=cfg.use_reference_params,
                rule=cfg.rule,
            ),
            seed=cfg.seed,
        )
    enc = EncoderConfig(rest_ms=cfg.rest_ms)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    images = data.images
    n = images.shape[0]
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            batch = images[idx]
            present_adaptive(net, batch, enc, gen, learn=True, max_repeat=cfg.max_repeat)
        print(f"epoch {epoch+1}/{cfg.epochs} done")
    return net


def evaluate(net, data, cfg=None, trials=1, max_samples=None):
    enc = EncoderConfig(rest_ms=cfg.rest_ms if cfg else 0.0)
    gen = torch.Generator().manual_seed(42)
    images = data.images if max_samples is None else data.images[:max_samples]
    n = images.shape[0]
    total = torch.zeros(n, net.config.n_e)
    for start in range(0, n, 128):
        end = min(start + 128, n)
        batch_total = None
        for _ in range(trials):
            c, _ = present_adaptive(net, images[start:end], enc, gen, learn=False)
            batch_total = c if batch_total is None else batch_total + c
        total[start:end] = batch_total
    return total


def accuracy(counts, labels):
    # simple mean readout
    n_classes = 10
    n_e = counts.shape[1]
    # assign classes by mean count per class
    proto = torch.zeros(n_classes, n_e)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            proto[c] = counts[mask].mean(dim=0)
    pred = (counts @ proto.T).argmax(dim=1)
    return float((pred == labels).float().mean() * 100)
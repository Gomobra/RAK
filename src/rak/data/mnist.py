"""MNIST loader + Poisson encoder (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision import datasets, transforms


@dataclass
class MnistSplit:
    images: Tensor
    labels: Tensor


def load_mnist(split: str = "train", data_dir: str = "data/mnist") -> MnistSplit:
    ds = datasets.MNIST(data_dir, train=(split == "train"), download=True,
                        transform=transforms.ToTensor())
    images = ds.data.reshape(-1, 784).float() / 255.0
    labels = ds.targets
    return MnistSplit(images=images, labels=labels)


@dataclass
class EncoderConfig:
    present_ms: float = 350.0
    rest_ms: float = 0.0
    dt_ms: float = 0.5
    min_spikes: int = 5
    start_intensity: float = 2.0
    max_rate_per_intensity: float = 31.875

    @property
    def n_steps(self) -> int:
        return int(round(self.present_ms / self.dt_ms))

    @property
    def n_rest_steps(self) -> int:
        return int(round(self.rest_ms / self.dt_ms))


def encode_poisson(images: Tensor, intensity: float, cfg: EncoderConfig,
                   generator: torch.Generator | None = None) -> Tensor:
    if images.dim() == 1:
        images = images.unsqueeze(0)
    rates_hz = images * cfg.max_rate_per_intensity * intensity
    prob = (rates_hz * (cfg.dt_ms / 1000.0)).clamp(0, 1)
    steps = cfg.n_steps
    g = generator or torch.Generator()
    u = torch.rand(steps, images.shape[0], images.shape[1], generator=g)
    spikes = (u < prob.unsqueeze(0)).to(torch.float32)
    return spikes


class AdaptiveIntensity:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.intensity = cfg.start_intensity

    def escalate(self):
        self.intensity += 1.0


def present_adaptive(net, images, enc, gen, learn=True, max_repeat=10):
    n = int(images.shape[0])
    total = torch.zeros(n, net.config.n_e)
    intensity = torch.full((n,), float(enc.start_intensity))
    if n == 0:
        return total, intensity

    pending = torch.arange(n)
    for rep in range(max_repeat + 1):
        if pending.numel() == 0:
            break
        ctrl = AdaptiveIntensity(enc)
        for _ in range(rep):
            ctrl.escalate()
        spikes = encode_poisson(images[pending], ctrl.intensity, enc, generator=gen)
        rest = int(round(enc.rest_ms / enc.dt_ms))
        _, counts = net.present_batch(spikes, learn=learn, rest_steps=rest)
        total[pending] += counts
        intensity[pending] = ctrl.intensity
        pending = pending[total[pending].sum(dim=1) < enc.min_spikes]

    return total, intensity
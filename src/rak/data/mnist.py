"""MNIST loader (Phase 3) — uses torchvision."""

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
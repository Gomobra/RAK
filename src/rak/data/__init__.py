"""Data module (Phase 3)."""

from rak.data.encode import EncoderConfig, encode_poisson, present_adaptive
from rak.data.mnist import MnistSplit, load_mnist

__all__ = ["MnistSplit", "load_mnist", "EncoderConfig", "encode_poisson", "present_adaptive"]
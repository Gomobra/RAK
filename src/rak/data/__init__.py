"""Data: pembaca MNIST dan encoding spike.

Semua tanpa dependensi jaringan, kecuali `mnist.load_mnist` yang memang
perlu mengunduh file IDX kalau belum ada di cache.
"""

from rak.data.encode import (
    AdaptiveIntensity,
    EncoderConfig,
    encode_poisson,
    estimated_rate_hz,
    spike_counts,
)
from rak.data.mnist import (
    MIRRORS,
    SPLITS,
    MnistSplit,
    download_mnist,
    has_mnist,
    load_mnist,
    read_idx_images,
    read_idx_labels,
)

__all__ = [
    "AdaptiveIntensity",
    "EncoderConfig",
    "encode_poisson",
    "estimated_rate_hz",
    "spike_counts",
    "MnistSplit",
    "MIRRORS",
    "SPLITS",
    "load_mnist",
    "download_mnist",
    "has_mnist",
    "read_idx_images",
    "read_idx_labels",
]

"""Validasi argumen `trials` pada evaluate().

`trials > 1` menjumlahkan spike count dari beberapa presentasi
Poisson untuk gambar yang sama. Encoder-nya stokastik; satu sampel
memberi pola yang bervariasi. Menjumlahkan percobaan mengurangi
varians itu tanpa menyentuh bobot.

Yang diuji di sini hanya kontrak argumennya; efek akurasi diukur di
Actions, bukan di unit test.
"""

from __future__ import annotations

import pytest
import torch

from rak.data.mnist import MnistSplit
from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.network.train import evaluate


def _split(n: int = 4) -> MnistSplit:
    return MnistSplit(
        images=torch.rand(n, 784),
        labels=torch.zeros(n, dtype=torch.long),
    )


def test_trials_must_be_positive() -> None:
    net = CompetitiveNetwork(
        NetworkConfig(n_e=20, inhibition_strength=400.0), seed=1
    )
    with pytest.raises(ValueError, match="trials"):
        evaluate(
            net, _split(), assignments=torch.zeros(20, dtype=torch.long),
            trials=0,
        )


def test_trials_do_not_change_weights() -> None:
    """Evaluasi berkali-kali tidak boleh menyentuh bobot atau theta."""
    net = CompetitiveNetwork(
        NetworkConfig(n_e=20, inhibition_strength=400.0), seed=1
    )
    w0 = net.w_e.clone()
    th0 = net.theta.clone()
    evaluate(
        net,
        _split(4),
        assignments=torch.zeros(20, dtype=torch.long),
        trials=2,
        max_samples=4,
    )
    assert torch.equal(net.w_e, w0)
    assert torch.equal(net.theta, th0)

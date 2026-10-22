"""Readout matched filter: prototipe kelas + korelasi pola.

Yang diuji adalah properti matematis, bukan asumsi penulis:

1. `class_prototypes` menghasilkan rata-rata per kelas yang benar.
2. `classify_matched` memilih kelas yang polanya cocok, bukan kelas
   dengan jumlah spike terbanyak.
3. Hasilnya INVARIAN terhadap skala global: mengalikan semua count
   dengan 10 tidak mengubah prediksi (z-score membuang skala).
4. Neuron yang merespons semua kelas sama kuat tidak menyumbang
   diskriminasi (setelah z-score, kontribusinya nol).
"""

from __future__ import annotations

import torch

from rak.network.competitive import class_prototypes, classify_matched


def test_prototypes_are_class_means() -> None:
    counts = torch.tensor(
        [
            [10.0, 0.0, 2.0],
            [20.0, 0.0, 2.0],
            [0.0, 8.0, 2.0],
            [0.0, 12.0, 2.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    p = class_prototypes(counts, labels, n_classes=2)
    assert torch.allclose(p[0], torch.tensor([15.0, 0.0, 2.0], dtype=torch.float64))
    assert torch.allclose(p[1], torch.tensor([0.0, 10.0, 2.0], dtype=torch.float64))
    # Kelas tanpa contoh tetap nol, tidak NaN.
    p3 = class_prototypes(counts, labels, n_classes=3)
    assert torch.allclose(p3[2], torch.zeros(3, dtype=torch.float64))


def test_matched_uses_pattern_not_magnitude() -> None:
    """Gambar dengan total spike KECIL bisa menang kalau polanya cocok."""
    prototypes = torch.zeros(2, 4, dtype=torch.float64)
    prototypes[0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    prototypes[1] = torch.tensor([0.0, 0.0, 1.0, 1.0])

    # Pola kelas 0, total 4 spike.
    small = torch.tensor([1.0, 1.0, 0.0, 0.0])
    # Pola kelas 1, total 40 spike (10x lebih besar).
    big = torch.tensor([0.0, 0.0, 20.0, 20.0])
    assert int(classify_matched(small, prototypes)) == 0
    assert int(classify_matched(big, prototypes)) == 1


def test_matched_is_invariant_to_global_scale() -> None:
    prototypes = torch.zeros(2, 3, dtype=torch.float64)
    prototypes[0] = torch.tensor([2.0, 0.0, 0.0])
    prototypes[1] = torch.tensor([0.0, 2.0, 1.0])
    x = torch.tensor([5.0, 1.0, 1.0])
    a = int(classify_matched(x, prototypes))
    b = int(classify_matched(x * 10, prototypes))
    assert a == b


def test_non_selective_neuron_adds_nothing() -> None:
    """Neuron yang sama kuat di semua kelas hilang setelah z-score."""
    prototypes = torch.zeros(2, 3, dtype=torch.float64)
    prototypes[0] = torch.tensor([5.0, 0.0, 5.0])   # neuron 2 netral
    prototypes[1] = torch.tensor([0.0, 5.0, 5.0])   # neuron 2 netral
    x0 = torch.tensor([9.0, 0.0, 3.0])
    x1 = torch.tensor([0.0, 9.0, 3.0])
    assert int(classify_matched(x0, prototypes)) == 0
    assert int(classify_matched(x1, prototypes)) == 1
    # Menambah spike di neuron netral tidak mengubah keputusan.
    assert int(classify_matched(x0 + torch.tensor([0.0, 0.0, 100.0]), prototypes)) == 0

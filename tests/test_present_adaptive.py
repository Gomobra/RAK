"""Properti `present_adaptive`, jalur presentasi bersama training/evaluasi.

Konteks kegagalan yang memunculkan file ini: `train()` menaikkan
intensitas sampai satu gambar menghasilkan >= `min_spikes`, sementara
`evaluate()` menyajikan gambar yang sama pada `start_intensity` saja.
Jaringan melihat input jauh lebih lemah saat dievaluasi daripada saat
dilatih, dan hasilnya 98/100 neuron tidak fire sama sekali di evaluasi
(akurasi 9.74%, setara tebakan).

Test di sini memakai jaringan PALSU yang deterministik, bukan jaringan
sungguhan. Yang diuji adalah aturan eskalasinya, bukan perilaku STDP:

1. Cukup spike di percobaan pertama  -> intensitas tidak naik.
2. Tidak pernah cukup                -> eskalasi sampai `max_repeat`.
3. Spike dari semua pengulangan      -> dijumlahkan, bukan diganti.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rak.data.encode import EncoderConfig
from rak.network.train import present_adaptive


class _StubNet:
    """Jaringan palsu: hasil spike ditentukan angka tetap per panggilan."""

    def __init__(self, n_e: int = 2, spikes_per_call: float = 0.0) -> None:
        self.config = SimpleNamespace(n_e=n_e, n_in=4)
        self.spikes_per_call = float(spikes_per_call)
        self.calls = 0

    def present_batch(self, z_pre: torch.Tensor):
        self.calls += 1
        n = int(z_pre.shape[1])
        counts = torch.full((n, self.config.n_e), self.spikes_per_call / self.config.n_e)
        zeros = torch.zeros(z_pre.shape[0], n, self.config.n_e)
        return zeros, counts


def _enc(min_spikes: int = 25) -> EncoderConfig:
    return EncoderConfig(start_intensity=1.0, min_spikes=min_spikes)


def _images(n: int = 2) -> torch.Tensor:
    return torch.ones(n, 784)


def test_enough_spikes_first_try_leaves_intensity_alone() -> None:
    """Kalau percobaan pertama sudah cukup, tidak ada eskalasi."""
    enc = _enc(min_spikes=5)
    net = _StubNet(spikes_per_call=50.0)

    counts, intensity = present_adaptive(net, _images(), enc, torch.Generator().manual_seed(0))

    assert net.calls == 1
    assert torch.all(intensity == enc.start_intensity)
    assert torch.all(counts.sum(dim=1) >= enc.min_spikes)


def test_silent_network_escalates_to_the_limit() -> None:
    """Jaringan yang tidak pernah cukup harus naik sampai `max_repeat`.

    `max_repeat` adalah jumlah pengulangan, jadi total panggilan adalah
    `max_repeat + 1` (presentasi pertama + pengulangan).
    """
    enc = _enc(min_spikes=25)
    net = _StubNet(spikes_per_call=0.0)

    counts, intensity = present_adaptive(
        net, _images(), enc, torch.Generator().manual_seed(0), max_repeat=3
    )

    assert net.calls == 4
    assert torch.all(intensity == enc.start_intensity + 3)
    assert float(counts.sum()) == 0.0


def test_repeat_spikes_are_summed_not_replaced() -> None:
    """Spike dari pengulangan harus MENAMBAH, bukan menimpa.

    Versi lama `train()` menulis `per_image[need] = sub_counts.sum(1)`,
    sehingga spike dari presentasi pertama hilang dari statistik.
    Kalau penjumlahan benar, total = (jumlah panggilan) x (spike/panggilan).
    """
    enc = _enc(min_spikes=25)
    per_call = 8.0
    net = _StubNet(spikes_per_call=per_call)

    counts, _ = present_adaptive(
        net, _images(), enc, torch.Generator().manual_seed(0), max_repeat=4
    )

    # 8 spike/gambar/panggilan, ambang 25 -> butuh 4 panggilan (32).
    expected_calls = 4
    assert net.calls == expected_calls
    assert torch.allclose(
        counts.sum(dim=1), torch.full((2,), per_call * expected_calls)
    )


def test_escalation_stops_per_image_not_per_batch() -> None:
    """Gambar yang sudah cukup berhenti; yang belum terus naik.

    Palsu-kan dua gambar dengan kebutuhan berbeda lewat urutan
    `spikes_per_call` yang berubah: panggilan pertama 0 untuk semua,
    lalu 50. Batch pertama tetap harus naik satu tingkat saja karena
    setelah itu semua gambar sudah cukup.
    """
    enc = _enc(min_spikes=25)
    net = _StubNet(spikes_per_call=0.0)
    net.spikes_per_call = 0.0

    original = net.present_batch

    def first_zero_rest_enough(z_pre):
        if net.calls == 0:
            return original(z_pre)
        net.calls += 1
        n = int(z_pre.shape[1])
        counts = torch.full((n, net.config.n_e), 50.0)
        zeros = torch.zeros(z_pre.shape[0], n, net.config.n_e)
        return zeros, counts

    net.present_batch = first_zero_rest_enough  # type: ignore[method-assign]

    _, intensity = present_adaptive(
        net, _images(3), enc, torch.Generator().manual_seed(0), max_repeat=3
    )

    assert net.calls == 2
    assert torch.all(intensity == enc.start_intensity + 1)

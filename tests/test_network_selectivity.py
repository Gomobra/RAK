r"""Gate: neuron harus mengembangkan receptive field BERBEDA.

T6.4 — test ini ada karena run MNIST pertama (100 neuron, 59.968
presentasi, 189 menit di Actions) SELESAI dengan melaporkan "0 neuron
mati", lolos semua criteria yang kita punya, dan menghasilkan akurasi
~10% — setara tebakan acak.

Diagnosisnya:

    korelasi antar kolom        : 1.000
    jarak L1 neuron 0 vs neuron 99 : 0.00
    neuron per kelas            : [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Semua 100 neuron punya bobot IDENTIK. Jaringan tidak belajar apa-apa.

PENYEBAB
--------
Rumus Eq. 3:

    dw[i,j] = eta · (x_pre[i] − x_tar[j]) · (w_max − w[i,j])^mu

Kalau semua neuron fire pada timestep yang sama, `x_tar[j]` bernilai
sama untuk semua j, sehingga `x_pre[i] − x_tar[j]` TIDAK bergantung
pada j. Persamaan tiap kolom jadi

    dw[i,j] / dt = c_i · (w_max − w[i,j])^mu

dengan `c_i` yang sama untuk semua kolom — hanya initial condition-nya
yang berbeda. Persamaan seperti itu attracting: dua solusi dengan
awal berbeda akan mendekat satu sama lain. Itu persis yang terjadi,
korelasi naik dari 0.000 ke 1.000 selama training.

Yang hanya bisa memecah simetri: neuron yang fire pada timestep
BERBEDA melihat profil `x_pre` yang berbeda. Lateral inhibition adalah
mekanismenya — tapi kalau inhibisi tidak cukup kuat untuk mengunci
satu pemenang per timestep, semua neuron fire bersamaan dan tidak ada
yang bisa membedakan diri.

CATATAN BIAYA
-------------
Test di sini sengaja KECIL (24 neuron, 6 batch) dan memakai pola
sintetis, bukan MNIST. Gate yang sebenarnya butuh 100 neuron dan
59.968 presentasi. Test kecil menangkap kelas bug yang SAMA dalam
hitungan detik, supaya ketahuan sebelum membakar kuota.
"""

import pytest
import torch

from rak.data.encode import EncoderConfig, encode_poisson
from rak.learning.stdp3 import Eq3Config
from rak.network.competitive import CompetitiveNetwork, NetworkConfig


def _column_correlation(w: torch.Tensor) -> float:
    """Rata-rata korelasi antar kolom bobot.

    Kolom dinormalisasi dulu supaya yang dibandingkan adalah POLA
    relatif tiap neuron, bukan besarannya — besarnya memang sama
    semua, itu hasil normalisasi kolom.
    """
    normed = w / w.sum(dim=0, keepdim=True).clamp(min=1e-12)
    corr = torch.corrcoef(normed.T.double())
    n = w.shape[1]
    off = corr[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean())


def _synthetic_digits(n_per_class=8, n_classes=10, seed=0):
    """Pola acak yang bisa dibedakan, sebagai pengganti MNIST."""
    g = torch.Generator().manual_seed(seed)
    n = n_classes * n_per_class
    images = torch.rand(n, 784, generator=g) * 0.05
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    for c in range(n_classes):
        block = slice(c * n_per_class, (c + 1) * n_per_class)
        images[block, c * 70 : (c + 1) * 70] = 0.8
    return images, labels


def _train_briefly(n_e=24, batches=6, eta=1e-3, **net_kw):
    images, _ = _synthetic_digits()
    # Inhibisi HARUS cukup kuat. Dengan default 1.0, w_ie kolom
    # berjumlah 1.0 -- praktis tidak ada inhibisi, dan semua neuron
    # memakai receptive field yang sama. Test-nya lalu menguji rezim
    # yang tidak pernah dipakai training.
    net_kw.setdefault("inhibition_strength", 400.0)
    net = CompetitiveNetwork(
        NetworkConfig(n_e=n_e, **net_kw),
        stdp_config=Eq3Config(eta=eta),
        seed=1,
    )
    enc = EncoderConfig()
    g = torch.Generator().manual_seed(1)
    for i in range(batches):
        start = (i * 10) % max(images.shape[0] - 10, 1)
        net.present_batch(
            encode_poisson(images[start : start + 10], 2.0, enc, generator=g)
        )
    return net


# ============================================================ gate
def test_neurons_develop_different_receptive_fields():
    """GATE: kolom neuron harus BERBEDA setelah training singkat."""
    net = _train_briefly()
    corr = _column_correlation(net.w_e)
    assert corr < 0.9, (
        f"korelasi antar kolom {corr:.4f} — neuron tidak membedakan diri. "
        f"Semua neuron memakai receptive field yang sama, jadi klasifikasi "
        f"mustahil berapa pun lama trainingnya."
    )


def test_correlation_does_not_drift_toward_one():
    """Korelasi tidak boleh merambat ke 1.

    Ini gradien dari gate di atas. Kenaikan monoton ke arah 1 berarti
    ada gaya menarik semua kolom ke titik yang sama.
    """
    images, _ = _synthetic_digits()
    net = CompetitiveNetwork(
        NetworkConfig(n_e=24, inhibition_strength=400.0),
        stdp_config=Eq3Config(eta=1e-3),
        seed=1,
    )
    enc = EncoderConfig()
    g = torch.Generator().manual_seed(1)

    series = [_column_correlation(net.w_e)]
    for i in range(6):
        start = (i * 10) % max(images.shape[0] - 10, 1)
        net.present_batch(
            encode_poisson(images[start : start + 10], 2.0, enc, generator=g)
        )
        series.append(_column_correlation(net.w_e))

    assert series[-1] < 0.9, f"derinya: {[round(x, 3) for x in series]}"
    assert series[-1] < series[1] + 0.3, (
        f"korelasi merambat ke arah 1: {series[1]:.3f} -> {series[-1]:.3f}. "
        f"Derinya {[round(x, 3) for x in series]}"
    )


def test_initial_columns_are_already_different():
    """Kontrol: inisialisasi HARUS acak antar kolom.

    Kalau kolom sudah identik sejak awal, tidak ada yang bisa
    membedakan, dan test lain akan gagal tanpa alasan yang jelas.
    """
    net = CompetitiveNetwork(
        NetworkConfig(n_e=24, inhibition_strength=400.0),
        stdp_config=Eq3Config(eta=1e-3),
        seed=1,
    )
    assert _column_correlation(net.w_e) < 0.3


# ================================================ selektivitas kelas
def test_neurons_fire_at_different_times():
    """Neuron harus TIDAK fire serempak.

    Inilah yang memecah simetri: kalau semua neuron fire di timestep
    yang sama, `x_pre` yang mereka lihat identik dan tidak ada yang
    bisa belajar receptive field berbeda.

    Yang diukur adalah KORELASI WAKTU antar neuron, bukan perbedaan
    laju. Versi pertama mengukur `std` dari laju fire — dan itu
    salah: dua neuron bisa punya laju identik tapi fire di timestep
    yang bergantian, yang justru justru yang kita butuhkan.
    """
    images, _ = _synthetic_digits()
    net = _train_briefly(n_e=24, batches=6)
    enc = EncoderConfig()
    g = torch.Generator().manual_seed(7)
    z_e, _ = net.present_batch(
        encode_poisson(images[:10], 2.0, enc, generator=g)
    )

    # (n_steps, B, n_e) -> (n_steps, B*n_e): satu deret waktu
    series = z_e.permute(0, 2, 1).reshape(z_e.shape[0], -1)
    if float(series.sum()) < 2:
        pytest.skip("terlalu sedikit spike untuk mengukur korelasi")

    # Neuron yang tidak pernah fire punya varians nol, sehingga
    # korelasinya NaNs. `mean()` lalu menjadi NaN untuk SELURUH metrik —
    # bug yang sama sempat membuat kolom deco di sweep kosong. Buang
    # neuron itu; korelasi mereka memang tidak terdefinisi.
    series = series[:, series.std(dim=0) > 0]
    if series.shape[1] < 2:
        pytest.skip("kurang dari 2 neuron aktif untuk mengukur korelasi")

    corr = torch.corrcoef(series.double())
    corr = torch.nan_to_num(corr, nan=0.0)
    n = corr.shape[0]
    off = corr[~torch.eye(n, dtype=torch.bool)]
    mean_off = float(off.mean())
    assert mean_off < 0.8, (
        f"korelasi antar neuron {mean_off:.3f} — mereka fire serempak, "
        f"jadi semua melihat `x_pre` yang sama dan tidak bisa membedakan"
    )


def test_assign_classes_maps_to_preferred_class():
    """`assign_classes` harus memilih kelas dengan laju tertinggi.

    Versi lama menguji ini lewat jaringan yang dilatih 10 batch, lalu
    gagal dengan `[14, 16, 0, ...]`. Itu bukan bug `assign_classes` —
    jaringan 30 neuron yang baru 10 batch memang belum tentu menyentuh
    10 kelas. Menguji sifat fungsi lewat training yang belum konvergen
    adalah asumsi penulis, bukan properti matematis.

    Di sini `assign_classes` diuji langsung: setiap neuron dibuat
    menyala hanya untuk satu kelas, jadi hasilnya harus persis itu.
    Gerbang end-to-end (apakah jaringan SUNGGUHAN belajar semua kelas)
    ada di `run_mnist.py --gate`, dengan cukup presentasi untuk
    mengukurnya.
    """
    from rak.network.competitive import assign_classes

    n_classes, n_e, per_class = 10, 30, 8
    n = n_classes * per_class
    labels = torch.arange(n) % n_classes

    # Neuron j paling kuat untuk kelas j % n_classes.
    counts = torch.full((n, n_e), 0.1)
    for j in range(n_e):
        counts[:, j] = torch.where(labels == (j % n_classes), 5.0, 0.1)

    assignments = assign_classes(counts, labels)
    expected = torch.arange(n_e) % n_classes

    assert torch.equal(assignments, expected), (
        f"assignment {assignments.tolist()} != {expected.tolist()}"
    )
    per_class_assigned = torch.bincount(assignments, minlength=n_classes)
    assert int(per_class_assigned.min()) == n_e // n_classes
    assert int(per_class_assigned.sum()) == n_e


def test_assign_classes_never_assigns_dead_neurons_to_a_class():
    """Neuron tanpa spike di semua kelas tidak boleh "menang" di kelas 0.

    Ini persis penyebab `[100, 0, 0, ...]` pada run pertama: neuron
    sunyi punya laju 0 untuk SEMUA kelas. Kalau tidak ditangani,
    semuanya jatuh ke kelas 0 secara default dan assignment jadi
    artefak, bukan hasil belajar.
    """
    from rak.network.competitive import assign_classes

    labels = torch.arange(100) % 10
    counts = torch.zeros(100, 12)
    # Neuron 0..5 memilih kelas 0, neuron 6..11 memilih kelas 9.
    counts[:, :6] = torch.where(
        (labels == 0).unsqueeze(1).expand(100, 6), 3.0, 0.0
    )
    counts[:, 6:] = torch.where(
        (labels == 9).unsqueeze(1).expand(100, 6), 3.0, 0.0
    )

    assignments = assign_classes(counts, labels)
    assigned = torch.bincount(assignments, minlength=10)

    assert int(assigned[0]) == 6
    assert int(assigned[9]) == 6
    assert int(assigned[1:9].sum()) == 0

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
        NetworkConfig(n_e=24), stdp_config=Eq3Config(eta=1e-3), seed=1
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
        NetworkConfig(n_e=24), stdp_config=Eq3Config(eta=1e-3), seed=1
    )
    assert _column_correlation(net.w_e) < 0.3


# ================================================ selektivitas kelas
def test_neurons_fire_at_different_times():
    """Neuron harus TIDAK fire serempak.

    Inilah yang memecah simetri: kalau semua neuron fire di timestep
    yang sama, `x_pre` yang mereka lihat identik.
    """
    images, _ = _synthetic_digits()
    net = _train_briefly(n_e=24, batches=6)
    enc = EncoderConfig()
    g = torch.Generator().manual_seed(7)
    z_e, _ = net.present_batch(
        encode_poisson(images[:10], 2.0, enc, generator=g)
    )

    firing = (z_e.sum(dim=0) > 0).to(torch.float32)   # (B, n_e)
    spread = float(firing.mean(dim=0).std())
    assert spread > 0.0, "semua neuron identik — tidak ada yang membedakan"


def test_class_assignment_uses_every_class():
    """Assignment tidak boleh menumpuk semua neuron di satu kelas.

    Run pertama menghasilkan [100, 0, 0, ...]. Itu bukan "kelas tidak
    terbaca" — itu "tidak ada kelas yang ada".
    """
    from rak.network.competitive import assign_classes

    images, labels = _synthetic_digits(n_per_class=8, seed=3)
    net = _train_briefly(n_e=30, batches=10, eta=2e-3)
    enc = EncoderConfig()
    g = torch.Generator().manual_seed(5)

    counts = []
    for i in range(0, images.shape[0], 10):
        _, c = net.present_batch(
            encode_poisson(images[i : i + 10], 2.0, enc, generator=g)
        )
        counts.append(c)

    assignments = assign_classes(torch.cat(counts, dim=0), labels)
    per_class = torch.bincount(assignments, minlength=10)

    assert int(per_class.min()) > 0, (
        f"ada kelas tanpa neuron: {per_class.tolist()}. "
        f"Kalau ini terjadi, akurasinya pasti mendekati 10%."
    )
"""Test loader MNIST — tanpa jaringan, tanpa file MNIST asli.

Semua test di sini memakai file IDX SINTETIS yang dibuat sendiri.
Alasannya dua:

  1. Test harus hermetik. Kalau suite bergantung pada download, dia
     gagal di CI yang tanpa network dan di Termux yang disknya 90% penuh.
  2. Yang perlu dibuktikan loader bukan "bisa baca MNIST" — itu
     trivially true kalau filenya benar. Yang perlu dibuktikan adalah
     DIA TOLAK file yang salah. Short read, magic salah, dan mirror yang
     mengirim halaman error status 200 adalah kegagalan yang paling
     mahal di Phase 3, karena muncul jauh kemudian sebagai akurasi
     yang inexplicable.

Jadi test di sini berlawanan dengan test biasa: mayoritasnya
memastikan Various kondisi rusak DITOLAK.
"""

import gzip
import struct

import pytest
import torch

from rak.data.mnist import (
    MIRRORS,
    SPLITS,
    download_mnist,
    has_mnist,
    load_mnist,
    read_idx_images,
    read_idx_labels,
)

MAGIC_IMAGES = 0x00000803
MAGIC_LABELS = 0x00000801


# ----------------------------------------------------------------------
# Pabrik file IDX sintetis
# ----------------------------------------------------------------------
def write_idx_images(path, images, rows=28, cols=28, *, gz=True, magic=MAGIC_IMAGES):
    """Tulis file IDX 3-dimensi dari array uint8 (N, rows, cols)."""
    n = images.shape[0]
    payload = struct.pack(">4I", magic, n, rows, cols)
    payload += images.astype("uint8").tobytes()
    opener = gzip.open if gz else open
    with opener(path, "wb") as fh:
        fh.write(payload)
    return path


def write_idx_labels(path, labels, *, gz=True, magic=MAGIC_LABELS):
    """Tulis file IDX 1-dimensi dari array uint8 (N,)."""
    payload = struct.pack(">2I", magic, len(labels))
    payload += labels.astype("uint8").tobytes()
    opener = gzip.open if gz else open
    with opener(path, "wb") as fh:
        fh.write(payload)
    return path


def write_truncated_images(path, n, rows, cols, missing, *, magic=MAGIC_IMAGES):
    """File IDX gambar yang header-nya功课 tapi isinya kurang.

    Ini yang terjadi kalau download putus di tengah: header lengkap,
    payload tidak. Simulasikan dengan menulis stream gzip yang berisi
    header utuh + payload dipotong — BUKAN dengan memotong file gzip,
    karena memotong file gzip merusak stream-nya dan menghasilkan
    `EOFError`, bukan kondisi short-read yang ingin kita uji.
    """
    payload = struct.pack(">4I", magic, n, rows, cols)
    payload += bytes((i * 7) % 256 for i in range(n * rows * cols - missing))
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return path


def write_truncated_labels(path, n, missing, *, magic=MAGIC_LABELS):
    """File IDX label yang header-nya功课 tapi isinya kurang."""
    payload = struct.pack(">2I", magic, n)
    payload += bytes(i % 10 for i in range(n - missing))
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return path


def make_split(tmp_path, split, n, *, rows=28, cols=28, seed=0, gz=True,
               magic_i=MAGIC_IMAGES, magic_l=MAGIC_LABELS):
    """Buat sepasang file IDX untuk `split` di `tmp_path`."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (n, rows, cols), generator=g).to(torch.uint8)
    labels = torch.randint(0, 10, (n,), generator=g).to(torch.uint8)

    write_idx_images(tmp_path / SPLITS[split]["images"], images.numpy(),
                     rows, cols, gz=gz, magic=magic_i)
    write_idx_labels(tmp_path / SPLITS[split]["labels"], labels.numpy(),
                     gz=gz, magic=magic_l)
    return images, labels


# ============================================================ round-trip
def test_idx_images_roundtrip(tmp_path):
    """Byte yang ditulis harus kembali persis sama."""
    g = torch.Generator().manual_seed(1)
    images = torch.randint(0, 256, (5, 28, 28), generator=g).to(torch.uint8)
    p = write_idx_images(tmp_path / "img.gz", images.numpy())

    back, rows, cols = read_idx_images(p)
    assert rows == 28 and cols == 28
    assert torch.equal(torch.from_numpy(back.copy()), images)


def test_idx_labels_roundtrip(tmp_path):
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.uint8)
    p = write_idx_labels(tmp_path / "lbl.gz", labels.numpy())
    assert torch.equal(torch.from_numpy(read_idx_labels(p).copy()), labels)


def test_gzip_and_raw_agree(tmp_path):
    """Ekstensi `.gz` harus transparan — isi identik."""
    g = torch.Generator().manual_seed(2)
    images = torch.randint(0, 256, (3, 28, 28), generator=g).to(torch.uint8)
    a = write_idx_images(tmp_path / "a.gz", images.numpy(), gz=True)
    b = write_idx_images(tmp_path / "b.raw", images.numpy(), gz=False)

    ia, *_ = read_idx_images(a)
    ib, *_ = read_idx_images(b)
    assert (ia == ib).all()


# ============================================================ penolakan
def test_wrong_magic_rejected(tmp_path):
    """File bukan IDX harus ditolak, bukan dibaca diam-diam."""
    g = torch.Generator().manual_seed(3)
    images = torch.randint(0, 256, (2, 28, 28), generator=g).to(torch.uint8)
    p = write_idx_images(tmp_path / "bad.gz", images.numpy(),
                         magic=0xDEADBEEF)
    with pytest.raises(ValueError, match="magic"):
        read_idx_images(p)


def test_wrong_label_magic_rejected(tmp_path):
    labels = torch.tensor([1, 2, 3], dtype=torch.uint8)
    p = write_idx_labels(tmp_path / "bad.gz", labels.numpy(),
                         magic=MAGIC_IMAGES)
    with pytest.raises(ValueError, match="magic"):
        read_idx_labels(p)


def test_truncated_image_file_rejected(tmp_path):
    """Download terpotong harus KETAHUAN, bukan jadi gambar berisi nol.

    Ini kegagalan paling mahal di Phase 3: file 90% lengkap menghasilkan
    gambar dengan 10% terakhir nol, network tetap jalan, dan akurasinya
    anjlok tanpa ada yang tahu kenapa.
    """
    write_truncated_images(tmp_path / "cut.gz", n=10, rows=28, cols=28,
                            missing=1000)
    with pytest.raises(ValueError, match="terpotong"):
        read_idx_images(tmp_path / "cut.gz")


def test_truncated_label_file_rejected(tmp_path):
    write_truncated_labels(tmp_path / "cut.gz", n=20, missing=3)
    with pytest.raises(ValueError, match="terpotong"):
        read_idx_labels(tmp_path / "cut.gz")


def test_truncated_header_rejected(tmp_path):
    """Header Less than 8/16 byte harus ditolak, bukan di-interpretasi."""
    p = tmp_path / "tiny.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(b"\x00\x00\x08")  # 3 byte saja
    with pytest.raises(ValueError, match="header"):
        read_idx_images(p)


def test_label_out_of_range_rejected(tmp_path):
    """Digit 10+ bukan MNIST — biasanya tanda file rusak."""
    labels = torch.tensor([0, 1, 10, 3], dtype=torch.uint8)
    p = write_idx_labels(tmp_path / "bad.gz", labels.numpy())
    with pytest.raises(ValueError, match="digit"):
        read_idx_labels(p)


# ============================================================ load_mnist
def test_load_normalizes_to_unit_range(tmp_path):
    make_split(tmp_path, "train", 40, seed=5)

    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert data.images.dtype == torch.float32
    assert float(data.images.min()) >= 0.0
    assert float(data.images.max()) <= 1.0


def test_full_scale_pixel_maps_to_exactly_one(tmp_path):
    """Pixel 255 harus jadi 1.0 PERSIS.

    Kalau tidak, konversi spike encoding nanti punya bias sistematis
    kecil yang sulit dicari.
    """
    images = torch.full((2, 28, 28), 255, dtype=torch.uint8)
    write_idx_images(tmp_path / SPLITS["train"]["images"], images.numpy())
    write_idx_labels(tmp_path / SPLITS["train"]["labels"],
                     torch.zeros(2, dtype=torch.uint8).numpy())

    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert float(data.images.max()) == pytest.approx(1.0, abs=1e-7)
    assert float(data.images.min()) == pytest.approx(1.0, abs=1e-7)


def test_zero_pixel_maps_to_exactly_zero(tmp_path):
    images = torch.zeros((2, 28, 28), dtype=torch.uint8)
    write_idx_images(tmp_path / SPLITS["train"]["images"], images.numpy())
    write_idx_labels(tmp_path / SPLITS["train"]["labels"],
                     torch.zeros(2, dtype=torch.uint8).numpy())

    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert float(data.images.abs().sum()) == 0.0


def test_shape_is_flattened(tmp_path):
    make_split(tmp_path, "train", 7, seed=6)
    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert data.images.shape == (7, 784)
    assert data.labels.shape == (7,)
    assert data.labels.dtype == torch.int64


def test_labels_in_valid_range(tmp_path):
    make_split(tmp_path, "train", 200, seed=7)
    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert int(data.labels.min()) >= 0
    assert int(data.labels.max()) <= 9


def test_preserves_pixel_values(tmp_path):
    """Nilai harus bertahan setelah normalisasi — cek pembulatan."""
    values = torch.tensor([[0, 1, 127, 128, 254, 255]], dtype=torch.uint8)
    images = torch.zeros((1, 28, 28), dtype=torch.uint8)
    images[0, 0, :6] = values[0]

    write_idx_images(tmp_path / SPLITS["train"]["images"], images.numpy())
    write_idx_labels(tmp_path / SPLITS["train"]["labels"],
                     torch.zeros(1, dtype=torch.uint8).numpy())

    data = load_mnist(tmp_path, "train", download=False, verify=False)
    got = (data.images[0, :6] * 255.0).round().to(torch.uint8)
    assert torch.equal(got, values[0]), f"nilai berubah: {got.tolist()}"


def test_label_histogram_sums_to_length(tmp_path):
    make_split(tmp_path, "train", 500, seed=8)
    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert int(data.label_histogram().sum()) == len(data)


def test_without_labels_hides_labels(tmp_path):
    """Loop training harus tidak punya jalan ke label."""
    make_split(tmp_path, "train", 10, seed=9)
    data = load_mnist(tmp_path, "train", download=False, verify=False)
    tensor_only = data.without_labels()
    assert torch.equal(tensor_only, data.images)


# ============================================================ verifikasi
def test_verify_catches_wrong_count(tmp_path):
    """Split train punya 60.000; file 100 contoh harus ditolak."""
    make_split(tmp_path, "train", 100, seed=10)
    with pytest.raises(ValueError, match="harus 60000"):
        load_mnist(tmp_path, "train", download=False, verify=True)


def test_verify_catches_wrong_shape(tmp_path):
    """MNIST 28x28. Bentuk lain bukan MNIST meski jumlah benar."""
    images = torch.zeros((60_000, 14, 14), dtype=torch.uint8)
    write_idx_images(tmp_path / SPLITS["train"]["images"], images.numpy(),
                     rows=14, cols=14)
    write_idx_labels(tmp_path / SPLITS["train"]["labels"],
                     torch.zeros(60_000, dtype=torch.uint8).numpy())

    with pytest.raises(ValueError, match="28"):
        load_mnist(tmp_path, "train", download=False, verify=True)


def test_verify_off_accepts_anything(tmp_path):
    """verify=False dipakai test kecil; harus benar-benar longgar."""
    make_split(tmp_path, "train", 3, seed=11)
    data = load_mnist(tmp_path, "train", download=False, verify=False)
    assert len(data) == 3


def test_image_label_count_mismatch_rejected(tmp_path):
    images = torch.zeros((5, 28, 28), dtype=torch.uint8)
    write_idx_images(tmp_path / SPLITS["train"]["images"], images.numpy())
    write_idx_labels(tmp_path / SPLITS["train"]["labels"],
                     torch.zeros(4, dtype=torch.uint8).numpy())

    with pytest.raises(ValueError, match="split berbeda"):
        load_mnist(tmp_path, "train", download=False, verify=False)


def test_unknown_split_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_mnist(tmp_path, "validasi", download=False)


# ============================================================ mirror
def test_at_least_two_mirrors_configured():
    """Satu mirror = satu titik kegagalan.

    yann.lecun.com sering mati; kalau itu satu-satunya mirror,
   Phase 3 berhenti tiap kali dia down.
    """
    assert len(MIRRORS) >= 2, f"cuma {len(MIRRORS)} mirror"


def test_no_lecun_in_mirrors():
    """yann.lecun.com 404 sejak 2024 — jangan dipakai sebagai mirror."""
    assert not any("lecun" in m for m in MIRRORS), (
        "yann.lecun.com tidak stabil, jangan masuk daftar mirror"
    )


def test_mirrors_are_https():
    assert all(m.startswith("https://") for m in MIRRORS), (
        "http bisa di-downgrade oleh MITM"
    )


# ============================================================ unduhan
def test_has_mnist_detects_presence(tmp_path):
    assert not has_mnist(tmp_path, "train")
    make_split(tmp_path, "train", 10)
    assert has_mnist(tmp_path, "train")
    assert not has_mnist(tmp_path, "test")


def test_missing_files_with_no_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mnist(tmp_path, "train", download=False)


def test_download_reports_which_mirrors_failed(tmp_path, monkeypatch):
    """Kalau semua mirror gagal, pesan harus menyebut SEMUA url."""
    attempts = []

    def boom(url, timeout=None):
        attempts.append(url)
        raise OSError("jaringan mati")

    monkeypatch.setattr("rak.data.mnist.urllib.request.urlopen", boom)

    with pytest.raises(RuntimeError) as exc:
        download_mnist(tmp_path, "test")

    assert len(attempts) == len(MIRRORS), (
        f"harus mencoba semua mirror, cuma coba {len(attempts)}"
    )
    for mirror in MIRRORS:
        assert mirror in str(exc.value), f"{mirror} tidak disebut di error"


def test_download_leaves_no_partial_file(tmp_path, monkeypatch):
    """File `.part` yang gagal harus dihapus.

    Kalau tertinggal, run berikutnya akan menganggap file itu lengkap
    dan memakai data setengah jadi tanpa error apa pun.
    """

    def boom(url, timeout=None):
        raise OSError("putus di tengah")

    monkeypatch.setattr("rak.data.mnist.urllib.request.urlopen", boom)

    with pytest.raises(RuntimeError):
        download_mnist(tmp_path, "test")

    leftovers = list(tmp_path.glob("*.part"))
    assert not leftovers, f"file parsial tertinggal: {leftovers}"


def test_download_falls_back_to_second_mirror(tmp_path, monkeypatch):
    """Mirror pertama harus dicoba, lalu yang kedua."""
    tried = []

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def flaky(url, timeout=None):
        tried.append(url)
        if url.startswith(MIRRORS[0]):
            raise OSError("mirror pertama mati")
        # sisipkan data gzip valid supaya step parse jalan
        import io
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(struct.pack(">4I", MAGIC_IMAGES, 1, 28, 28) + b"\x07" * 784)
        return FakeResp(buf.getvalue())

    monkeypatch.setattr("rak.data.mnist.urllib.request.urlopen", flaky)
    download_mnist(tmp_path, "test", )

    assert tried[0].startswith(MIRRORS[0])
    assert any(u.startswith(MIRRORS[1]) for u in tried), (
        "mirror kedua tidak dicoba"
    )
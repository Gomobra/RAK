"""Pembaca MNIST format IDX — tanpa dependensi eksternal.

Kenapa parser sendiri dan bukan `torchvision.datasets.MNIST`:

  1. `torchvision` menarik rantai dependensi besar hanya untuk membaca
     empat file. Untuk batch $0 dan repo yang harus tetap ringan, itu
     tidak sepadan.
  2. Format IDX itu sederhana dan stabil sejak 1998. Parser-nya
     sekitar 40 baris dan bisa diuji langsung tanpa jaringan.
  3. Termux (lokasi dev) tidak punya wheel torchvision. Kalau modul
     mengimpornya di level atas, seluruh suite test gagal di sana —
     bukan karena kode salah, tapi karena environment.

Yang TIDAK boleh ada di sini: logika apa pun tentang spike encoding
atau jaringan. Loader ini hanya mengubah byte jadi tensor.

S_NORMALISASI
-------------
Gambar disimpan sebagai float32 dalam rentang [0, 1] (pixel uint8 / 255).
Konversi uint8 -> laju spike terjadi di `rak.neuron.trace`, bukan di sini,
supaya tidak ada dua tempat yang sama-sama melakukan penskalaan.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "MnistSplit",
    "MIRRORS",
    "SPLITS",
    "read_idx_images",
    "read_idx_labels",
    "load_mnist",
    "download_mnist",
    "has_mnist",
]

#: Magic IDX. 0x0000 08 03 = 3-dimensi (N, rows, cols) uint8.
#: 0x0000 08 01 = 1-dimensi (N,) uint8.
_MAGIC_IMAGES = 0x00000803
_MAGIC_LABELS = 0x00000801

#: Mirror yang dipakai berurutan kalau yang pertama gagal.
#:
#: yann.lecun.com sengaja TIDAK ada di sini — sudah sering 404 / timeout
#: sejak 2024. Dua mirror di bawah yang dipakai torchvision dan tetap hidup.
MIRRORS: tuple[str, ...] = (
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
)

#: Nama file per split, sesuai spesifikasi MNIST aslinya.
SPLITS: dict[str, dict[str, str]] = {
    "train": {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz",
    },
    "test": {
        "images": "t10k-images-idx3-ubyte.gz",
        "labels": "t10k-labels-idx1-ubyte.gz",
    },
}

#: Jumlah digit yang diharapkan per split. Dipakai sebagai sanity check
#: supaya file yang terpotong atau salah mirror tidak lolos diam-diam.
_EXPECTED_COUNT = {"train": 60_000, "test": 10_000}

_EXPECTED_SHAPE = (28, 28)
_N_CLASSES = 10


@dataclass(frozen=True)
class MnistSplit:
    """Satu split MNIST.

    Attributes:
        images: float32 (N, 784), nilai dalam [0, 1].
        labels: int64 (N,), nilai dalam [0, 9].

    PENTING soal pemakaian label: di RAK, label **hanya** dipakai
    untuk evaluasi — neuron ke kelas apa ditentukan SETELAH
    training selesai, bukan selama. Loop training tidak boleh menerima
    `labels` sama sekali; lihat `rak.eval.classify_from_spikes`.
    """

    images: Tensor
    labels: Tensor
    split: str = "train"

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.images.shape[1])

    def without_labels(self) -> Tensor:
        """Cuma gambar — bentuk yang boleh dipakai loop training."""
        return self.images

    def subset(self, n: int | None = None) -> "MnistSplit":
        """Takes `n` contoh pertama, label ikut.

        Dipakai untuk smoke test dan milestone kecil: tanpa ini
        `train()` akan gagal dengan `AttributeError` yang membingungkan
        kalau diteruskan `split.images` langsung.
        """
        if n is None:
            return self
        return MnistSplit(
            images=self.images[:n], labels=self.labels[:n], split=self.split
        )

    def label_histogram(self) -> Tensor:
        """Jumlah contoh per digit. Untuk mendeteksi split yang rusak."""
        return torch.bincount(self.labels, minlength=_N_CLASSES)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _read_header(
    fh, expected_magic: int, n_fields: int, path: Path
) -> tuple[int, ...]:
    """Baca dan validasi header IDX.

    `n_fields` wajib per jenis file: header gambar punya 4 field
    (magic, N, rows, cols) = 16 byte, header label cuma 2 (magic, N)
    = 8 byte. Mengambil selalu 8 byte membuat file gambar gagal
    di-unpack — dan itu persis yang terjadi sebelum test ini ada.
    """
    size = 4 * n_fields
    raw = fh.read(size)
    if len(raw) < size:
        raise ValueError(
            f"{path.name}: header tidak lengkap, dapat {len(raw)} byte "
            f"(butuh {size}). File kemungkinan terpotong."
        )
    magic, *dims = struct.unpack(f">{n_fields}I", raw)
    if magic != expected_magic:
        raise ValueError(
            f"{path.name}: magic salah, dapat 0x{magic:08x}, "
            f"harus 0x{expected_magic:08x}. Ini bukan file IDX "
            f"yang benar — cek apakah mirror mengembalikan halaman "
            f"error, bukan 404."
        )
    return tuple(dims)


def read_idx_images(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Baca file gambar IDX 3-dimensi.

    Returns:
        (images uint8 (N, rows, cols), rows, cols)

    Raises:
        ValueError: kalau magic salah atau file lebih pendek dari
            yang dijanjikan header. Short read yang tidak divalidasi tidak akan
            menghasilkan error karena gambar berisi nol akan lolos — dan
            itu akan muncul jauh kemudian sebagai akurasi inexplicable
            yang mahal dicari.
    """
    path = Path(path)
    with _open_maybe_gzip(path) as fh:
        n, rows, cols = _read_header(fh, _MAGIC_IMAGES, 4, path)
        raw = fh.read()

    expected = n * rows * cols
    if len(raw) < expected:
        raise ValueError(
            f"{path.name}: header menjanjikan {n}x{rows}x{cols} = "
            f"{expected} byte, tapi file cuma berisi {len(raw)} byte "
            f"(kurang {expected - len(raw)}). Download terpotong."
        )
    images = np.frombuffer(raw[:expected], dtype=np.uint8)
    return images.reshape(n, rows, cols), rows, cols


def read_idx_labels(path: str | Path) -> np.ndarray:
    """Baca file label IDX 1-dimensi. Returns uint8 (N,)."""
    path = Path(path)
    with _open_maybe_gzip(path) as fh:
        (n,) = _read_header(fh, _MAGIC_LABELS, 2, path)
        raw = fh.read()

    if len(raw) < n:
        raise ValueError(
            f"{path.name}: header menjanjikan {n} label, file cuma "
            f"berisi {len(raw)} byte. Download terpotong."
        )
    labels = np.frombuffer(raw[:n], dtype=np.uint8)
    if labels.max(initial=0) >= _N_CLASSES:
        raise ValueError(
            f"{path.name}: label terbesar {int(labels.max())} >= "
            f"{_N_CLASSES}. MNIST cuma punya digit 0-9."
        )
    return labels


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _target_path(data_dir: Path, filename: str) -> Path:
    return data_dir / filename


def download_mnist(
    data_dir: str | Path,
    split: str = "train",
    *,
    timeout: float = 60.0,
    force: bool = False,
) -> list[Path]:
    """Unduh file MNIST untuk satu split.

    Mencoba semua mirror di `MIRRORS` berurutan. Unduhan dilakukan ke
    file sementara dulu lalu di-`replace`, supaya file yang terpotong
    tidak pernah tertinggal sebagai `.gz` utuh dan menyesatkan run
    berikutnya.

    Returns:
        daftar path file yang berhasil ditulis.
    """
    if split not in SPLITS:
        raise ValueError(f"split harus salah satu dari {sorted(SPLITS)}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    errors: list[str] = []

    for filename in SPLITS[split].values():
        dest = _target_path(data_dir, filename)
        if dest.exists() and not force:
            written.append(dest)
            continue

        last_error: Exception | None = None
        for base in MIRRORS:
            url = f"{base}/{filename}"
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    payload = resp.read()
                if not payload:
                    raise ValueError("respons kosong")
                tmp.write_bytes(payload)
                tmp.replace(dest)
                written.append(dest)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                tmp.unlink(missing_ok=True)
                errors.append(f"{url}: {exc}")

        if last_error is not None:
            raise RuntimeError(
                f"Gagal mengunduh {filename} dari semua mirror "
                f"({len(MIRRORS)} dicoba):\n  " + "\n  ".join(errors)
            ) from last_error

    return written


def has_mnist(data_dir: str | Path, split: str = "train") -> bool:
    """True kalau semua file untuk `split` sudah ada."""
    data_dir = Path(data_dir)
    try:
        names = SPLITS[split].values()
    except KeyError:
        return False
    return all(_target_path(data_dir, n).exists() for n in names)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_mnist(
    data_dir: str | Path = "data/mnist",
    split: str = "train",
    *,
    download: bool = True,
    verify: bool = True,
) -> MnistSplit:
    """Muat satu split MNIST sebagai tensor.

    Args:
        data_dir: tempat file idx dan hasil unduhan.
        split: "train" atau "test".
        download: kalau True dan file belum ada, unduh dulu.
        verify: kalau True, cek jumlah contoh dan bentuk gambar
            terhadap spesifikasi. Ini yang menangkap mirror yang
            mengirim halaman error dengan status 200.

    Returns:
        `MnistSplit` dengan gambar float32 [0, 1] (N, 784) dan label
        int64 (N,).
    """
    if split not in SPLITS:
        raise ValueError(f"split harus salah satu dari {sorted(SPLITS)}")

    data_dir = Path(data_dir)
    if not has_mnist(data_dir, split):
        if not download:
            raise FileNotFoundError(
                f"File MNIST untuk split {split!r} tidak ada di "
                f"{data_dir} dan download=False."
            )
        download_mnist(data_dir, split)

    raw_images, rows, cols = read_idx_images(
        data_dir / SPLITS[split]["images"]
    )
    raw_labels = read_idx_labels(data_dir / SPLITS[split]["labels"])

    if raw_images.shape[0] != raw_labels.shape[0]:
        raise ValueError(
            f"Jumlah gambar ({raw_images.shape[0]}) != jumlah label "
            f"({raw_labels.shape[0]}). File berasal dari split berbeda?"
        )

    if verify:
        if (rows, cols) != _EXPECTED_SHAPE:
            raise ValueError(
                f"Bentuk gambar {(rows, cols)} != {_EXPECTED_SHAPE}. "
                f"Bukan MNIST 28x28."
            )
        expected_n = _EXPECTED_COUNT[split]
        if raw_images.shape[0] != expected_n:
            raise ValueError(
                f"Split {split!r} punya {raw_images.shape[0]} contoh, "
                f"harus {expected_n}."
            )

    images = torch.from_numpy(
        np.array(raw_images, dtype=np.uint8, copy=True).reshape(-1, rows * cols)
    ).to(torch.float32)
    # uint8 -> [0, 1]. Dibagi 255 tepat supaya pixel 255 jadi 1.0
    # persis, bukan 0.999999 — pembulatan di float32 bikin ini
    # perlu Scaling float32 langsung.
    images /= 255.0

    labels = torch.from_numpy(raw_labels.astype(np.int64))

    return MnistSplit(images=images, labels=labels, split=split)
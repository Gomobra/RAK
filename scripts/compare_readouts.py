r"""Bandingkan beberapa aturan baca (readout) pada bobot yang sudah dilatih.

Training tidak diubah sama sekali. Yang diuji hanya cara mengubah
spike count menjadi label, memakai bobot hasil satu kali training.

Kenapa ini penting: `assign_classes` memakai label TRAIN untuk
menentukan neuron mana milik kelas mana. Itu sah. Tapi selama ini
readout-nya hanya satu (rata-rata spike per anggota kelas), dan belum
pernah dibandingkan dengan alternatif yang sama-sama tanpa supervision
tambahan:

    mean    : rata-rata spike anggota kelas           (dipakai sekarang)
    max     : neuron paling aktif, ambil kelasnya
    margin  : rata-rata berbobot kemurnian neuron
    topk    : rata-rata k spike teratas anggota kelas
    binary  : proporsi anggota kelas yang fire

Pemilihan dilakukan di SPLIT VALIDASI dari data training, bukan di
test set. Test set hanya dipakai sekali, untuk angka final readout
yang terpilih.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rak.data.encode import EncoderConfig
from rak.data.mnist import load_mnist
from rak.network.competitive import (
    CompetitiveNetwork,
    NetworkConfig,
    assign_classes,
)
from rak.network.train import present_adaptive


def _counts(
    net: CompetitiveNetwork,
    images: torch.Tensor,
    enc: EncoderConfig,
    batch: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, int(images.shape[0]), batch):
        c, _ = present_adaptive(net, images[s : s + batch], enc, gen, learn=False)
        out.append(c)
    return torch.cat(out, dim=0)


def _class_stats(
    counts: torch.Tensor, labels: torch.Tensor, n_classes: int = 10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kembalikan (assignments, purity) dari data kalibrasi.

    `purity[j] = rate_kelas_terpilih[j] - rate_kelas_lain_tertinggi[j]`.
    Neuron yang merespons semua kelas sama kuat mendapat purity ~0.
    """
    n_e = counts.shape[1]
    assignments = torch.zeros(n_e, dtype=torch.long)
    best = torch.zeros(n_e, dtype=torch.float64)
    second = torch.zeros(n_e, dtype=torch.float64)
    rates = torch.zeros(n_classes, n_e, dtype=torch.float64)
    for c in range(n_classes):
        m = labels == c
        if int(m.sum()) == 0:
            continue
        rates[c] = counts[m].sum(dim=0).double() / int(m.sum())
    for c in range(n_classes):
        better = rates[c] > best
        second = torch.where(better, best, torch.maximum(second, rates[c]))
        best = torch.where(better, rates[c], best)
        assignments = torch.where(
            better, torch.full_like(assignments, c), assignments
        )
    return assignments, (best - second).clamp_min(0.0), rates


def _predict(
    counts: torch.Tensor,
    assignments: torch.Tensor,
    weights: torch.Tensor | None,
    kind: str,
    rates: torch.Tensor | None = None,
    n_classes: int = 10,
    topk: int = 3,
) -> torch.Tensor:
    counts = counts.to(torch.float64)
    totals = torch.zeros(counts.shape[0], n_classes, dtype=torch.float64)

    # --- matched filter: pakai SELURUH profil laju, bukan assignment
    # keras. Neuron yang mirip 4 dan 9 tetap menyumbang ke keduanya
    # dengan bobot berbeda; assignment keras membuang informasi itu.
    if kind in ("matched", "matched_cos", "matched_cosz"):
        assert rates is not None
        r = rates.to(torch.float64)
        if kind == "matched":
            return (counts @ r.T).argmax(dim=1)
        if kind == "matched_cos":
            # Pusatkan per neuron: blob (laju sama di semua kelas)
            # jadi bobot nol. Normalisasi baris = cosine per kelas.
            r = r - r.mean(dim=0, keepdim=True)
            r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-9)
            return (counts @ r.T).argmax(dim=1)
        # matched_cosz: tambahan z-score per gambar (pola murni).
        c = counts - counts.mean(dim=1, keepdim=True)
        c = c / c.std(dim=1, keepdim=True).clamp_min(1e-9)
        r = r - r.mean(dim=1, keepdim=True)
        r = r / r.std(dim=1, keepdim=True).clamp_min(1e-9)
        return (c @ r.T).argmax(dim=1)
    for c in range(n_classes):
        members = assignments == c
        n = int(members.sum())
        if n == 0:
            continue
        block = counts[:, members]
        if kind == "mean":
            totals[:, c] = block.sum(dim=1) / n
        elif kind == "max":
            totals[:, c] = block.max(dim=1).values
        elif kind == "topk":
            k = min(topk, n)
            totals[:, c] = block.topk(k, dim=1).values.mean(dim=1)
        elif kind == "binary":
            totals[:, c] = (block > 0).double().sum(dim=1) / n
        elif kind == "margin":
            w = weights[members]
            denom = float(w.sum())
            if denom < 1e-9:
                # Semua anggota kelas ini tidak selektif. Jangan biarkan
                # kelas itu mustahil menang hanya karena pembaginya nol.
                totals[:, c] = block.sum(dim=1) / n
            else:
                totals[:, c] = (block * w).sum(dim=1) / denom
        else:
            raise ValueError(kind)
    return totals.argmax(dim=1)


def _accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    return float((pred == labels).double().mean() * 100.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inhibition", type=float, default=1600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    blob = torch.load(a.weights, map_location="cpu", weights_only=False)
    state = blob["net"]
    n_e = int(state["w_e"].shape[1])
    net = CompetitiveNetwork(
        NetworkConfig(n_e=n_e, inhibition_strength=a.inhibition, rule="variant3"),
        seed=a.seed,
    )
    net.load_state_dict(state)
    enc = EncoderConfig()

    train = load_mnist(split="train")
    test = load_mnist(split="test")
    n_tr = min(a.train_samples, int(train.images.shape[0]))
    n_te = min(a.test_samples, int(test.images.shape[0]))

    tr_counts = _counts(net, train.images[:n_tr], enc, a.batch, a.seed + 1)
    te_counts = _counts(net, test.images[:n_te], enc, a.batch, a.seed + 2)
    tr_labels = train.labels[:n_tr]
    te_labels = test.labels[:n_te]

    # Kalibrasi di separuh pertama, pilih readout di separuh kedua.
    split = n_tr // 2
    cal_counts, cal_labels = tr_counts[:split], tr_labels[:split]
    val_counts, val_labels = tr_counts[split:], tr_labels[split:]
    assignments, purity, rates = _class_stats(cal_counts, cal_labels)

    rows = {}
    for kind in (
        "mean", "max", "margin", "topk", "binary",
        "matched", "matched_cos", "matched_cosz",
    ):
        pred_val = _predict(val_counts, assignments, purity, kind, rates)
        pred_te = _predict(te_counts, assignments, purity, kind, rates)
        rows[kind] = {
            "val": round(_accuracy(pred_val, val_labels), 2),
            "test": round(_accuracy(pred_te, te_labels), 2),
        }

    best = max(rows, key=lambda k: rows[k]["val"])
    print(f"neuron: {n_e}, kalibrasi {split}, validasi {n_tr - split}, test {n_te}")
    print(f"{'readout':<8} {'val':>7} {'test':>7}")
    for k, v in rows.items():
        mark = " <- dipilih" if k == best else ""
        print(f"{k:<8} {v['val']:>7.2f} {v['test']:>7.2f}{mark}")
    print(f"\nreadout terpilih (dari validasi): {best} -> test {rows[best]['test']:.2f}%")

    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

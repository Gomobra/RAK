r"""Sweep kekuatan inhibisi lateral.

T6.4 — mencari nilai `inhibition_strength` yang MEMECAH SIMETRI antar
neuron.

MASALAH YANG INI SELESAIKAN
---------------------------
Run MNIST pertama (100 neuron, 59.968 presentasi) menghasilkan
jaringan yang semua neuronnya identik:

    korelasi antar kolom bobot : 1.000
    neuron per kelas           : [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Penyebabnya sudah ditelusuri. Persamaan Eq. 3

    dw[i,j] = eta · (x_pre[i] − x_tar[j]) · (w_max − w[i,j])^mu

hanya bergantung pada `j` lewat `x_tar[j]` dan `w[i,j]`. Kalau semua
neuron fire pada timestep yang sama, `x_tar[j]` sama untuk semua, dan
persamaan tiap kolom menjadi

    dw[i,j] / dt = c_i · (w_max − w[i,j])^mu

dengan `c_i` yang sama. Dua kolom dengan initial condition berbeda
menjalankan ODE yang sama, dan ODE seperti itu attracting — mereka
mendekat. Di sinilah lateral inhibition masuk: kalau hanya SATU neuron
yang boleh fire per timestep, tiap neuron melihat profil `x_pre` yang
BERBEDA, dan itu yang membuat receptive field bisa berbeda.

METRIK YANG DIUKUR
------------------
Tiga angka per konfigurasi:

1. `deco` — korelasi WAKTU antar neuron (spike train). Makin rendah
   makin baik. Ini metrik langsung untuk "apakah ada pemenang
   tunggal".
2. `col_corr` — korelasi antar kolom bobot. Kalau NAIK ke arah 1,
   simetri sedang runtuh.
3. `classes` — jumlah kelas yang dapat minimal satu neuron.
   10 = bagus, 1 = degenerat.

Yang kita cari: `deco` rendah DAN `col_corr` tidak merambat ke 1.

KENAPA MNIST, BUKAN POLA SINTETIS
---------------------------------
Pertanyaan ini ("seberapa kuat inhibisi yang benar?") bergantung pada
distribusi input MNIST yang sesungguhnya. Pola sintetis diskrit akan
memberi jawaban yang tidak berlaku untuk MNIST.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from rak.data.encode import EncoderConfig, encode_poisson
from rak.data.mnist import load_mnist
from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.learning.stdp3 import Eq3Config
from rak.network.competitive import assign_classes


# ----------------------------------------------------------------------
def temporal_decorrelation(z_e: torch.Tensor) -> float:
    """Rata-rata korelasi antar neuron pada spike train.

    `z_e` bentuknya (n_steps, B, n_e). Semua neuron digabung jadi
    satu deret waktu panjang supaya korelasinya bermakna.

    Dua neuron dengan laju fire identik tapi JEDA waktunya bisa punya
    korelasi ~0 — dan justru itu yang kita butuhkan. Mengukur
    perbedaan laju (std) tidak akan menangkapnya.
    """
    if float(z_e.sum()) < 2:
        return 1.0
    series = z_e.permute(0, 2, 1).reshape(z_e.shape[0], -1).double()
    corr = torch.corrcoef(series)
    n = corr.shape[0]
    off = corr[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean())


def column_correlation(w: torch.Tensor) -> float:
    """Rata-rata korelasi antar kolom bobot (pola relatif)."""
    normed = w / w.sum(dim=0, keepdim=True).clamp(min=1e-12)
    corr = torch.corrcoef(normed.T.double())
    n = w.shape[1]
    off = corr[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean())


# ----------------------------------------------------------------------
def run_config(
    data_dir: Path,
    strength: float,
    *,
    n_e: int,
    batches: int,
    batch_size: int,
    eta: float,
    seed: int,
    n_classes: int,
) -> dict:
    train_set = load_mnist(data_dir, "train", download=True)
    images = train_set.images
    labels = train_set.labels
    enc = EncoderConfig()

    net = CompetitiveNetwork(
        NetworkConfig(n_e=n_e, inhibition_strength=strength),
        stdp_config=Eq3Config(eta=eta, dt_ms=enc.dt_ms),
        seed=seed,
    )
    gen = torch.Generator().manual_seed(seed)

    traj_col: list[float] = []
    traj_deco: list[float] = []
    spike_total = 0.0
    n_done = 0
    fired_any = torch.zeros(n_e)

    g0 = torch.randperm(images.shape[0], generator=gen)
    for b in range(batches):
        idx = g0[b * batch_size : (b + 1) * batch_size]
        z_pre = encode_poisson(
            images[idx], enc.start_intensity, enc, generator=gen
        )
        z_e, counts = net.present_batch(z_pre)

        spike_total += float(counts.sum())
        n_done += idx.shape[0]
        fired_any += (counts > 0).any(dim=0).to(torch.float32)

        if (b + 1) % max(batches // 8, 1) == 0 or b == batches - 1:
            traj_col.append(column_correlation(net.w_e))
            traj_deco.append(temporal_decorrelation(z_e))

    # jumlah kelas yang dapat neuron
    n_assign = min(images.shape[0], 4000)
    chunks = []
    gen2 = torch.Generator().manual_seed(seed + 99)
    for start in range(0, n_assign, batch_size):
        _, c = net.present_batch(
            encode_poisson(
                images[start : start + batch_size],
                enc.start_intensity, enc, generator=gen2,
            )
        )
        chunks.append(c)
    assignments = assign_classes(torch.cat(chunks, dim=0), labels[:n_assign])
    per_class = torch.bincount(assignments, minlength=n_classes)

    return {
        "inhibition_strength": strength,
        "n_e": n_e,
        "batches": batches,
        "presentations": n_done,
        "eta": eta,
        "final_col_corr": traj_col[-1] if traj_col else None,
        "final_deco": traj_deco[-1] if traj_deco else None,
        "col_corr_traj": [round(x, 4) for x in traj_col],
        "deco_traj": [round(x, 4) for x in traj_deco],
        "classes_used": int((per_class > 0).sum()),
        "min_neurons_per_class": int(per_class.min()),
        "dead_neurons": int((fired_any == 0).sum()),
        "spikes_per_presentation": spike_total / max(n_done, 1),
    }


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rak sweep-inhibition")
    p.add_argument("--data-dir", default="data/mnist")
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--strengths", default="0,1,2,4,8,16")
    p.add_argument("--neurons", type=int, default=100)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    strengths = [float(x) for x in args.strengths.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'strength':>9} {'deco':>7} {'colcorr':>8} {'kelas':>6} "
          f"{'mati':>5} {'spike/pres':>11}", flush=True)
    for s in strengths:
        t0 = time.perf_counter()
        row = run_config(
            Path(args.data_dir), s,
            n_e=args.neurons, batches=args.batches,
            batch_size=args.batch_size, eta=args.eta,
            seed=args.seed, n_classes=10,
        )
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(
            f"{s:>9.1f} {row['final_deco']:>7.3f} "
            f"{row['final_col_corr']:>8.3f} {row['classes_used']:>6d} "
            f"{row['dead_neurons']:>5d} "
            f"{row['spikes_per_presentation']:>11.1f}",
            flush=True,
        )
        (out_dir / "sweep.json").write_text(json.dumps(rows, indent=2))

    # ringkasan: konfigurasi terbaik = deco rendah, kelas banyak,
    # col_corr tidak merambat ke 1
    best = max(
        rows,
        key=lambda r: (r["classes_used"], -r["final_deco"], -r["final_col_corr"]),
    )
    print(
        f"\nTERBAIK: inhibition_strength={best['inhibition_strength']} "
        f"(deco {best['final_deco']:.3f}, "
        f"{best['classes_used']} kelas, "
        f"col_corr {best['final_col_corr']:.3f})",
        flush=True,
    )

    table = [
        "# Sweep `inhibition_strength`",
        "",
        f"Jaringan {args.neurons} neuron, {args.batches} batch "
        f"({args.batches * args.batch_size} presentasi), eta={args.eta}.",
        "",
        "| strength | deco (lower better) | col_corr | kelas | mati | spike/pres |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        table.append(
            f"| {r['inhibition_strength']} | {r['final_deco']:.3f} | "
            f"{r['final_col_corr']:.3f} | {r['classes_used']}/10 | "
            f"{r['dead_neurons']} | {r['spikes_per_presentation']:.0f} |"
        )
    table += ["", f"**Terbaik:** `inhibition_strength = {best['inhibition_strength']}`", ""]
    (out_dir / "sweep.md").write_text("\n".join(table))

    summary = "\n".join(table)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
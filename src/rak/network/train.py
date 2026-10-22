r"""Loop training dan evaluasi MNIST.

T6.2 — menyatukan loader, encoder, jaringan, dan readout jadi satu
proses yang bisa dijalankan dari CLI atau GitHub Actions.

KONTRUKSI YANG DIPUTUSKAN (dan alasannya)
------------------------------------------
1. **Mini-batch, bukan satu-satu.** Kode acuanpresents one image at a time. Itu korekt tapi lambat: di Termux satu gambar
   100 neuron butuh ~500 ms, jadi satu epoch 60.000 gambar = 8 jam.
   Dengan batch 64, matmul jadi efisien dan epoch turun ke ~2 jam.
   Bobot tetap SATU matriks yang dipakai bersama batch — bukan satu
   per gambar.

2. **Intensitas adaptif diterapkan per-gambar dalam batch.**
   Setelah satu batch dipresentasikan, gambar yang spike-nya kurang
   dari `min_spikes` diulang dengan intensitas lebih tinggi, satu
   per satu, sampai cukup atau batas iterasi tercapai. Ini padanan
   dari aturan acuan, bukan perilaku baru.

3. **Checkpoint setiap N presentasi.** Run 2 jam yang hilang karena runner
   ke-restart adalah kegagalan yang tidak bisa diterima. Bobot
   disimpan sebagai `.pt` biasa, bukan `.pth` custom.

4. **Label hanya menyentuh `assign_classes`, setelah training.**
   `evaluate()` menerima spike count, bukan gambar berlabel.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from rak.data.encode import AdaptiveIntensity, EncoderConfig, encode_poisson
from rak.data.mnist import MnistSplit
from rak.learning.stdp3 import Eq3Config
from rak.network.competitive import (
    CompetitiveNetwork,
    NetworkConfig,
    assign_classes,
    classify,
    classify_matched,
)

__all__ = ["TrainConfig", "TrainResult", "train", "evaluate", "accuracy"]


@dataclass
class TrainConfig:
    """Parameter training."""

    n_e: int = 100
    n_in: int = 784

    #: Jumlah pass atas training set. Acuan: 3 epoch untuk 400 neuron.
    epochs: int = 1

    #: Jumlah gambar per batch presentasi.
    batch_size: int = 64

    #: Batas pengulangan untuk intensitas adaptif per gambar.
    max_repeat: int = 6

    seed: int = 1

    #: Kekuatan inhibisi lateral. Dengan `scale_inhibition=True`
    #: (default), ini adalah TOTAL konduktansi inhibitor per neuron
    #: eksitasi -- konstan terhadap ukuran jaringan.
    inhibition_strength: float = 800.0

    #: Rule STDP: "variant3" (bentuk acuan) atau "eq3" (rumus paper).
    rule: str = "variant3"

    #: Homeostasis: kenaikan theta per spike, dan batas atasnya.
    #: `theta_eq = theta_plus * laju * tau_theta`, jadi `theta_plus`
    #: menentukan laju fire kesetimbangan. Acuan 0.05.
    theta_plus: float = 0.0125
    theta_max: float = 40.0

    #: Bobot E->I (satu-satu). Terpisah dari `inhibition_strength`.
    e_to_i_strength: float = 1600.0

    #: Learning rate STDP Eq.3.
    eta: float = 1e-3

    #: Kolom bobot (Diehl-Cook: 78 untuk 784 input).
    column_sum: float = 78.0

    #: Checkpoint setiap N presentasi. 0 = jangan checkpoint.
    checkpoint_every: int = 5000

    #: Jumlah presentasi sebelum Diehl-Cook mulai mengukur akurasi
    #: training (untuk monitoring, bukan untuk deciding).
    log_every: int = 500

    out_dir: str = "runs/mnist"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class TrainResult:
    """Hasil training."""

    presentations: int = 0
    epochs_done: int = 0
    wall_seconds: float = 0.0
    mean_spikes: float = 0.0
    dead_neurons: int = 0
    history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.presentations} presentasi, "
            f"{self.epochs_done} epoch, "
            f"{self.wall_seconds / 60:.1f} menit, "
            f"{self.mean_spikes:.1f} spike/presentasi, "
            f"{self.dead_neurons} neuron mati"
        )


# ----------------------------------------------------------------------
def _batches(n: int, size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n - size + 1, size):
        yield order[start : start + size]


def present_adaptive(
    net: CompetitiveNetwork,
    images: Tensor,
    enc: EncoderConfig,
    gen: torch.Generator,
    *,
    max_repeat: int = 10,
    learn: bool = True,
) -> tuple[Tensor, Tensor]:
    """Presentasi dengan intensitas adaptif per-gambar (Diehl & Cook).

    Gambar yang menghasilkan < `enc.min_spikes` dipresentasikan ULANG
    dengan intensitas +1. Spike dari semua pengulangan DIJUMLAHKAN.

    Kenapa ini harus dipakai di training DAN evaluasi:

    Sebelumnya `train()` menaikkan intensitas sampai gambar menghasilkan
    >= 25 spike, lalu `evaluate()` menyajikan gambar yang sama pada
    `start_intensity` saja. Jaringan jadi melihat input jauh lebih
    lemah saat dievaluasi daripada saat dilatih. Hasilnya, dari
    run 100 neuron: **98/100 neuron tidak fire sama sekali** saat
    evaluasi, dan akurasi 9.74% (setara tebakan).

    Itu bukan jaringan mati -- itu dua jalur kode yang tidak sinkron.
    Sekarang keduanya memakai fungsi ini.

    Args:
        images: (B, n_in) gambar ternormalisasi.
        max_repeat: batas eskalasi per gambar.

    Returns:
        (counts, intensity): `counts` (B, n_e) jumlah spike lintas
        pengulangan; `intensity` (B,) intensitas terakhir tiap gambar.
    """
    n = int(images.shape[0])
    total = torch.zeros(n, int(net.config.n_e))
    intensity = torch.full((n,), float(enc.start_intensity))
    if n == 0:
        return total, intensity

    pending = torch.arange(n)
    for rep in range(max_repeat + 1):
        if pending.numel() == 0:
            break
        ctrl = AdaptiveIntensity(enc)
        for _ in range(rep):
            ctrl.escalate()
        spikes = encode_poisson(
            images[pending], ctrl.intensity, enc, generator=gen
        )
        _, counts = net.present_batch(spikes, learn=learn)
        total[pending] += counts
        intensity[pending] = ctrl.intensity
        pending = pending[total[pending].sum(dim=1) < enc.min_spikes]

    return total, intensity


def train(
    data: MnistSplit,
    config: TrainConfig | None = None,
    *,
    encoder: EncoderConfig | None = None,
    progress: bool = True,
    net: CompetitiveNetwork | None = None,
) -> tuple[CompetitiveNetwork, TrainResult]:
    """Latih jaringan tanpa supervision.

    Args:
        data: split training. `data.labels` TIDAK pernah dibaca di
            sini — hanya `data.without_labels()`. Itu dijamin oleh
            kode, bukan konvensi: satu-satunya akses ke `.labels`
            ada di `evaluate`.
        config: parameter training.
        encoder: konfigurasi spike.
        progress: cetak progres ke stderr.
        net: jaringan yang sudah dilatih, untuk MELANJUTKAN training.
            Kalau None, jaringan baru dibuat dari `config`. Bobot dan
            theta ikut, trace STDP mulai dari nol -- itu wajar, trace
            adalah keadaan jangka pendek, bukan parameter.

    Returns:
        (jaringan, hasil)
    """
    cfg = config or TrainConfig()
    enc = encoder or EncoderConfig()
    gen = torch.Generator().manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if net is None:
        net = CompetitiveNetwork(
            NetworkConfig(
                n_in=cfg.n_in, n_e=cfg.n_e, column_sum=cfg.column_sum,
                inhibition_strength=cfg.inhibition_strength, rule=cfg.rule,
                theta_plus=cfg.theta_plus, theta_max=cfg.theta_max,
                e_to_i_strength=cfg.e_to_i_strength,
            ),
            stdp_config=Eq3Config(eta=cfg.eta, dt_ms=enc.dt_ms),
            seed=cfg.seed,
        )

    images = data.without_labels()
    n_samples = int(images.shape[0])
    result = TrainResult()
    spike_acc = 0.0
    spike_events = 0
    fire_counts = torch.zeros(cfg.n_e)

    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        for idx in _batches(n_samples, cfg.batch_size, gen):
            batch = images[idx]

            # --- presentasi adaptif --------------------------------
            # Versi lama mengganti `per_image[need]` dengan hitungan
            # pengulangan TERAKHIR, bukan menjumlahkan. Spike dari
            # presentasi pertama hilang dari statistik.
            counts, _ = present_adaptive(
                net, batch, enc, gen, max_repeat=cfg.max_repeat
            )
            per_image = counts.sum(dim=1)

            fire_counts += (counts > 0).any(dim=0).to(torch.float32)
            spike_acc += float(per_image.sum())
            spike_events += int(batch.shape[0])
            result.presentations += int(batch.shape[0])

            # --- logging -------------------------------------------
            if progress and result.presentations % cfg.log_every < cfg.batch_size:
                elapsed = time.perf_counter() - t_start
                rate = result.presentations / max(elapsed, 1e-9)
                remaining = (cfg.epochs * n_samples - result.presentations)
                eta_h = remaining / max(rate, 1e-9) / 3600
                row = {
                    "presentations": result.presentations,
                    "epoch": epoch + 1,
                    "rate_per_s": round(rate, 2),
                    "eta_hours": round(eta_h, 2),
                    "mean_spikes": round(spike_acc / max(spike_events, 1), 1),
                    "dead": int((fire_counts == 0).sum()),
                }
                result.history.append(row)
                print(
                    f"  pres {row['presentations']:>7d} "
                    f"({rate:5.1f}/s, ETA {eta_h:5.2f}j) "
                    f"spike/pres {row['mean_spikes']:6.1f} "
                    f"mati {row['dead']}",
                    flush=True,
                )

            if cfg.checkpoint_every and result.presentations % cfg.checkpoint_every < cfg.batch_size:
                torch.save(
                    {"net": net.state_dict(), "train_cfg": asdict(cfg),
                     "presentations": result.presentations},
                    out_dir / "checkpoint.pt",
                )

        result.epochs_done = epoch + 1

    result.wall_seconds = time.perf_counter() - t_start
    result.mean_spikes = spike_acc / max(spike_events, 1)
    result.dead_neurons = int((fire_counts == 0).sum())

    torch.save(
        {"net": net.state_dict(), "train_cfg": asdict(cfg),
         "presentations": result.presentations},
        out_dir / "weights.pt",
    )
    (out_dir / "history.json").write_text(
        json.dumps({"config": asdict(cfg), "history": result.history,
                    "summary": result.summary()}, indent=2)
    )
    return net, result


# ----------------------------------------------------------------------
def evaluate(
    net: CompetitiveNetwork,
    data: MnistSplit,
    *,
    encoder: EncoderConfig | None = None,
    batch_size: int = 64,
    assignments: Tensor | None = None,
    prototypes: Tensor | None = None,
    seed: int = 0,
    max_samples: int | None = None,
    trials: int = 1,
) -> dict:
    """Klasifikasi tanpa supervision.

    Args:
        net: jaringan yang sudah dilatih.
        data: split test.
        assignments: peta neuron -> kelas. Kalau None, dihitung dari
            split TRAIN (bukan test) — memakai label test untuk
            menentukan neuron mana milik kelas mana adalah kebocoran.
        prototypes: (n_classes, n_e) profil laju dari split TRAIN.
            Kalau diberikan, klasifikasi memakai `classify_matched`
            (korelasi pola), bukan assignment keras.

    Returns:
        dict dengan `accuracy`, `confusion`, dan `n`.
    """
    enc = encoder or EncoderConfig()
    gen = torch.Generator().manual_seed(seed)

    images = data.without_labels()
    labels = data.labels
    if max_samples is not None:
        images = images[:max_samples]
        labels = labels[:max_samples]

    n = int(images.shape[0])
    if n == 0:
        raise ValueError(
            f"evaluate() menerima 0 gambar (max_samples={max_samples}, "
            f"test set punya {int(data.images.shape[0])}). "
            f"`0 or None` menghasilkan None, tapi kalau `max_samples` "
            f"dikirim sebagai 0 eksplisit, `images[:0]` jadi kosong."
        )
    if trials < 1:
        raise ValueError(f"trials harus >= 1, dapat {trials}")
    all_counts = []
    all_intensity = []
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        # WAJIB sama dengan training: intensitas adaptif, bukan
        # `start_intensity` sekali. Lihat `present_adaptive`.
        #
        # `trials > 1`: encoder Poisson itu stokastik. Satu sampel
        # memberi pola spike yang bervariasi; menjumlahkan beberapa
        # percobaan mengurangi varians itu tanpa mengubah bobot.
        total = None
        last_intensity = None
        for _ in range(trials):
            counts, intensity = present_adaptive(
                net, images[start:end], enc, gen, learn=False
            )
            total = counts if total is None else total + counts
            last_intensity = intensity
        all_counts.append(total)
        all_intensity.append(last_intensity)
        start = end
    counts = torch.cat(all_counts, dim=0)
    used_intensity = torch.cat(all_intensity, dim=0)

    # Diagnostik: kalau jaringan terlalu sunyi, assignment GAGAL
    # karena neuron yang tidak pernah fire punya laju 0 untuk SEMUA
    # kelas, dan `better = rate > best` tidak pernah benar -- semua
    # neuron itu tertinggal di kelas 0.
    silent = int((counts.sum(dim=0) == 0).sum())
    print(
        f"    intensitas rata-rata {float(used_intensity.mean()):.1f} "
        f"(maks {float(used_intensity.max()):.0f}), "
        f"spike/gambar {float(counts.sum(dim=1).mean()):.1f}",
        flush=True,
    )
    if silent:
        print(
            f"    [PERINGATAN] {silent}/{counts.shape[1]} neuron tidak "
            f"pernah fire saat evaluasi.",
            flush=True,
        )

    if prototypes is None and assignments is None:
        raise ValueError(
            "`prototypes` atau `assignments` wajib diberikan. Hitung "
            "dari spike count TRAIN — bukan test, itu kebocoran."
        )

    if prototypes is not None:
        pred = torch.stack([
            classify_matched(counts[i], prototypes)
            for i in range(counts.shape[0])
        ])
    else:
        pred = torch.stack([
            classify(counts[i], assignments) for i in range(counts.shape[0])
        ])

    confusion = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(labels.tolist(), pred.tolist()):
        confusion[t, p] += 1

    return {
        "accuracy": float((pred == labels).float().mean()),
        "confusion": confusion.tolist(),
        "n": n,
    }


def accuracy(result: dict) -> float:
    """Ambil akurasi dari hasil `evaluate`, dalam persen."""
    return 100.0 * float(result["accuracy"])
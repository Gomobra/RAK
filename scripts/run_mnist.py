"""Entry point training + evaluasi MNIST untuk Phase 3.

Dipakai dari CLI (`rak mnist`) dan dari GitHub Actions.

TWO-STAGE SECARA DEFAULT
------------------------
Jalankan smoke test kecil DULU (beberapa ratus presentasi), baru
training penuh. Alasannya ekonomi, bukan kehati-hatian:

  Training penuh 100 neuron = 2 jam. Kalau ada bug yang baru muncul
  setelah 10 menit dan kita tidak menangkapnya di awal, kita
  membakar 2 jam untuk NOTHING. Smoke test memakan ~5 menit dan
  menangkap 90% kelas bug (shape salah, data salah, loop tidak
  jalan, assignment salah).

Bisa dimatikan dengan `--no-smoke` kalau smoke-nya sudah lulus di
run sebelumnya.

Exit code 1 kalau ada gate yang gagal, supaya Actions menandai job
gagal dan tidak diam-diam hijau dengan hasil tidak masuk akal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from rak.data.encode import EncoderConfig, encode_poisson
from rak.data.mnist import load_mnist
from rak.network import assign_classes
from rak.network.train import TrainConfig, accuracy, evaluate, train

#: Target akurasi perNetworkConfig().n_e, dari Diehl & Cook 2015 Table 1.
#: 100 neuron -> 82.9%. Angka ini jadi GATE: kalau jauh di bawah,
#: ada yang salah — hasilnya bukan "temuan ilmiah", cuma bug.
PAPER_ACCURACY = {100: 82.9, 400: 87.0, 1600: 91.9, 6400: 95.0}


def _fmt_gate(n_e: int, acc: float) -> str:
    ref = PAPER_ACCURACY.get(n_e)
    if ref is None:
        return f"  (paper tidak punya referensi untuk {n_e} neuron)"
    delta = acc - ref
    flag = "OK" if delta > -10.0 else "DI BAWAH HARAP"
    return (
        f"  paper (Diehl-Cook 2015, {n_e} neuron): {ref:.1f}%\n"
        f"  selisih: {delta:+.1f} poin  -> {flag}"
    )


def run(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] memuat MNIST dari {data_dir}", flush=True)
    train_set = load_mnist(data_dir, "train", download=not args.no_download)
    test_set = load_mnist(data_dir, "test", download=not args.no_download)
    print(
        f"  train {len(train_set)} gambar, test {len(test_set)} gambar",
        flush=True,
    )

    enc = EncoderConfig()

    # ---------------- stage 1: smoke ------------------------------
    if args.smoke:
        n_smoke = args.smoke_presents
        print(
            f"\n[2/4] SMOKE TEST: {n_smoke} presentasi, "
            f"{args.smoke_neurons} neuron (~{args.smoke_minutes} mnt)",
            flush=True,
        )
        smoke_set = train_set.subset(n_smoke)
        net, res = train(
            smoke_set,
            TrainConfig(
                n_e=args.smoke_neurons,
                epochs=1,
                batch_size=min(64, n_smoke),
                eta=args.eta,
                out_dir=str(out_dir / "smoke"),
                checkpoint_every=0,
                log_every=max(64, n_smoke // 8),
                seed=args.seed,
            ),
            encoder=enc,
        )
        print(f"  {res.summary()}", flush=True)

        ok = (
            torch.isfinite(net.w_e).all()
            and res.dead_neurons < args.smoke_neurons
            and res.mean_spikes > 0
        )
        if not ok:
            print(
                f"\nGATE SMOKE GAGAL: dead={res.dead_neurons}, "
                f"spike/pres={res.mean_spikes:.1f}, "
                f"finite={bool(torch.isfinite(net.w_e).all())}",
                flush=True,
            )
            return 1
        print("  GATE SMOKE LOLOS", flush=True)

    # ---------------- stage 2: training ----------------------------
    subset = train_set.subset(args.limit) if args.limit else train_set
    print(
        f"\n[3/4] TRAINING: {args.neurons} neuron, {args.epochs} epoch, "
        f"eta={args.eta}",
        flush=True,
    )
    cfg = TrainConfig(
        n_e=args.neurons,
        epochs=args.epochs,
        batch_size=args.batch,
        eta=args.eta,
        seed=args.seed,
        out_dir=str(out_dir),
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )
    net, res = train(subset, cfg, encoder=enc)
    print(f"\n  {res.summary()}", flush=True)

    # ---------------- stage 3: assignment -------------------------
    # Assignment dihitung dari TRAINING SET. Kalau dari test set,
    # kita membocorkan label dan angkanya jadi tidak bermakna.
    print("\n[4/4] EVALUASI (10.000 test set)", flush=True)
    gen = torch.Generator().manual_seed(args.seed + 1)
    n_assign = args.assign_samples or min(len(subset), 10_000)
    _, counts = net.present_batch(
        encode_poisson(
            subset.images[:n_assign], enc.start_intensity, enc, generator=gen
        )
    )
    assignments = assign_classes(counts, subset.labels[:n_assign])
    n_per_class = int(torch.bincount(assignments, minlength=10).min())
    print(
        f"  neuron per kelas: min {n_per_class}, "
        f"total {int(torch.bincount(assignments, minlength=10).sum())}",
        flush=True,
    )

    result = evaluate(
        net, test_set, encoder=enc, batch_size=args.batch,
        assignments=assignments, seed=args.seed + 2,
        max_samples=args.test_limit,
    )
    acc = accuracy(result)

    report = {
        "neurons": args.neurons,
        "epochs": args.epochs,
        "eta": args.eta,
        "seed": args.seed,
        "presentations": res.presentations,
        "mean_spikes_per_presentation": res.mean_spikes,
        "dead_neurons": res.dead_neurons,
        "test_accuracy_percent": acc,
        "paper_accuracy_percent": PAPER_ACCURACY.get(args.neurons),
        "wall_minutes": round((time.perf_counter() - t0) / 60, 1),
        "confusion_matrix": result["confusion"],
        "assignments": assignments.tolist(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(
        {"net": net.state_dict(), "assignments": assignments},
        out_dir / "weights.pt",
    )

    # ---------------- ringkasan -----------------------------------
    summary_lines = [
        "=" * 62,
        f"  RAK — MNIST unsupervised (STDP Eq.3, tanpa teaching signal)",
        "=" * 62,
        f"  neuron eksitasi      : {args.neurons}",
        f"  epoch / presentasi   : {args.epochs} / {res.presentations}",
        f"  spike per presentasi : {res.mean_spikes:.1f}",
        f"  neuron mati          : {res.dead_neurons}",
        f"  AKURASI TEST         : {acc:.2f}%",
        _fmt_gate(args.neurons, acc),
        f"  waktu jalan          : {report['wall_minutes']} menit",
        "=" * 62,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary, flush=True)

    # GitHub Actions: tulis ke step summary supaya terlihat di halaman run
    if "GITHUB_STEP_SUMMARY" in os.environ:
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
            f"```\n{summary}\n```\n"
        )

    if args.gate and acc < args.gate:
        print(f"\nGATE GAGAL: akurasi {acc:.2f}% < {args.gate}%", flush=True)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rak mnist", description="Training MNIST unsupervised"
    )
    p.add_argument("--data-dir", default="data/mnist")
    p.add_argument("--out", default="runs/mnist")
    p.add_argument("--neurons", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = pakai seluruh training set")
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--assign-samples", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--no-download", action="store_true")
    # smoke
    p.add_argument("--no-smoke", dest="smoke", action="store_false")
    p.add_argument("--smoke", action="store_true", default=True)
    p.add_argument("--smoke-presents", type=int, default=640)
    p.add_argument("--smoke-neurons", type=int, default=100)
    p.add_argument("--smoke-minutes", type=int, default=5)
    # gate
    p.add_argument("--gate", type=float, default=0.0,
                   help="gagal (exit 1) kalau akurasi di bawah ini")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ndibatalkan", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
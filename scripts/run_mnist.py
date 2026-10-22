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
from rak.network import CompetitiveNetwork, NetworkConfig, assign_classes
from rak.network.train import present_adaptive
from rak.learning.stdp3 import Eq3Config
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



def _report(acc, meta) -> str:
    n_e = meta["neurons"]
    lines = [
        "=" * 62,
        "  RAK — MNIST unsupervised (STDP Eq.3, tanpa teaching signal)",
        "=" * 62,
        f"  neuron eksitasi      : {n_e}",
        f"  epoch / presentasi   : {meta.get('epochs', '?')} / "
        f"{meta.get('presentations', '?')}",
        f"  spike per presentasi : {meta.get('mean_spikes', 0):.1f}",
        f"  neuron mati          : {meta.get('dead_neurons', '?')}",
        f"  AKURASI TEST         : {acc:.2f}%",
    ]
    lines.append(_fmt_gate(n_e, acc))
    lines.append(f"  waktu total          : {meta.get('wall_minutes', 0)} menit")
    lines.append("=" * 62)
    return "\n".join(lines)


def _evaluate_only(args, net, train_set, test_set, enc, saved_assignments):
    """Muat bobot yang sudah dilatih, lalu ukur akurasi saja.

    Dipakai setelah run yang selesai training tapi gagal di langkah
    evaluasi. Menghemat 3 jam training ulang karena masalahnya
    bukan di training.
    """
    gen = torch.Generator().manual_seed(args.seed + 1)

    if saved_assignments is not None:
        assignments = torch.as_tensor(saved_assignments)
    else:
        n_assign = args.assign_samples or min(len(train_set), 10_000)
        chunks = []
        start = 0
        while start < n_assign:
            end = min(start + args.batch, n_assign)
            # SAMA dengan training: intensitas adaptif. Tanpa ini,
            # assignment dihitung dari neuron yang tidak menyala --
            # persis bug yang membuat run sebelumnya 9.74%.
            c, _ = present_adaptive(
                net, train_set.images[start:end], enc, gen, learn=False
            )
            chunks.append(c)
            start = end
        counts = torch.cat(chunks, dim=0)
        assignments = assign_classes(
            counts, train_set.labels[:counts.shape[0]]
        )

    per_class = torch.bincount(assignments, minlength=10)
    print(
        f"  neuron per kelas: {sorted(per_class.tolist(), reverse=True)}",
        flush=True,
    )

    result = evaluate(
        net, test_set, encoder=enc, batch_size=args.batch,
        assignments=assignments, seed=args.seed + 2,
        max_samples=args.test_limit or None,
    )
    acc = accuracy(result)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps({
        "neurons": int(assignments.shape[0]),
        "test_accuracy_percent": acc,
        "paper_accuracy_percent": PAPER_ACCURACY.get(int(assignments.shape[0])),
        "assignments": assignments.tolist(),
        "confusion_matrix": result["confusion"],
        "n_test": result["n"],
        "resumed_from": args.resume,
    }, indent=2))
    torch.save({"net": net.state_dict(), "assignments": assignments},
               out_dir / "weights.pt")

    summary = _report(acc, {
        "neurons": int(assignments.shape[0]),
        "mean_spikes": float("nan"),
        "wall_minutes": round(
            (time.perf_counter() - _T0) / 60, 1),
    })
    print("\n" + summary, flush=True)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
            f"```\n{summary}\n```\n"
        )
    if args.gate and acc < args.gate:
        print(f"\nGATE GAGAL: {acc:.2f}% < {args.gate}%", flush=True)
        return 1
    return 0


_T0 = time.perf_counter()


def run(args: argparse.Namespace) -> int:
    global _T0
    t0 = time.perf_counter()
    _T0 = t0
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
                inhibition_strength=args.inhibition,
                rule=args.rule,
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

    net = None
    if args.resume:
        print(
            f"\n[2b/4] RESUME: muat bobot dari {args.resume}",
            flush=True,
        )
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        net = CompetitiveNetwork(
            NetworkConfig(
                n_in=train_set.n_features,
                n_e=len(blob["net"]["w_e"][0]),
                column_sum=args.column_sum,
                inhibition_strength=args.inhibition,
                rule=args.rule,
            ),
            stdp_config=Eq3Config(eta=args.eta),
            seed=args.seed,
        )
        net.load_state_dict(blob["net"])
        if args.continue_epochs > 0:
            print(
                f"\n[3/4] LANJUT TRAINING: {args.continue_epochs} epoch "
                f"dari bobot yang dimuat",
                flush=True,
            )
            cfg = TrainConfig(
                n_e=int(net.w_e.shape[1]),
                epochs=args.continue_epochs,
                batch_size=args.batch,
                eta=args.eta,
                seed=args.seed,
                inhibition_strength=args.inhibition,
                rule=args.rule,
                out_dir=str(out_dir),
                log_every=args.log_every,
                checkpoint_every=args.checkpoint_every,
            )
            net, res = train(subset, cfg, encoder=enc, net=net)
            print(f"\n  {res.summary()}", flush=True)
        elif "assignments" in blob and not args.reassign:
            print(
                "  bobot sudah punya assignments; hanya perlu tes",
                flush=True,
            )
            return _evaluate_only(
                args, net, train_set, test_set, enc, blob["assignments"]
            )
    if net is None:
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
            inhibition_strength=args.inhibition,
            rule=args.rule,
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

    # PENTING: batch per bagian. Versi pertama memanggil
    # `encode_poisson` untuk 10.000 gambar sekaligus, yang membuat
    # tensor (700, 10000, 784) = 21.9 GB -- runner punya 16 GB dan
    # crash. Run 3 jam itu hilang di langkah terakhir, setelah
    # training selesai dan bobot sudah aman.
    # Chunk tidak boleh MELAMPAUI n_assign. `range(0, 4000, 64)`
    # berakhir di 4032, jadi `labels[:4000]` tidak akan sama panjang
    # dengan counts -- persis bug yang menggagalkan sweep 18 menit lalu.
    train_counts = []
    start = 0
    while start < n_assign:
        end = min(start + args.batch, n_assign)
        # `learn=False` WAJIB, dan intensitas harus adaptif seperti
        # evaluasi. Versi lama membiarkan STDP menyala di sini:
        # assignment dihitung sambil bobot dan theta masih berubah,
        # jadi neuron dinilai pada jaringan yang berbeda dari yang
        # diuji -- dan 5000 presentasi ekstra itu bukan bagian dari
        # training yang dilaporkan.
        c, _ = present_adaptive(
            net, subset.images[start:end], enc, gen, learn=False
        )
        train_counts.append(c)
        start = end
    counts = torch.cat(train_counts, dim=0)
    assignments = assign_classes(counts, subset.labels[:counts.shape[0]])
    n_per_class = int(torch.bincount(assignments, minlength=10).min())
    print(
        f"  neuron per kelas: min {n_per_class}, "
        f"total {int(torch.bincount(assignments, minlength=10).sum())}",
        flush=True,
    )

    result = evaluate(
        net, test_set, encoder=enc, batch_size=args.batch,
        assignments=assignments, seed=args.seed + 2,
        max_samples=args.test_limit or None,
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
    p.add_argument("--inhibition", type=float, default=800.0,
                   help="total konduktansi inhibitor per neuron")
    p.add_argument("--rule", default="variant3", choices=["eq3", "variant3"])
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
    p.add_argument("--resume", default="",
                   help="path ke weights.pt/checkpoint.pt")
    p.add_argument("--continue-epochs", type=int, default=0,
                   help="dengan --resume: lanjutkan training N epoch "
                        "lagi, bukan hanya evaluasi")
    p.add_argument("--reassign", action="store_true",
                   help="dengan --resume: abaikan assignments tersimpan, "
                        "hitung ulang dari data train")
    p.add_argument("--column-sum", type=float, default=78.0)
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
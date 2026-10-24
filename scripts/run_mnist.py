"""MNIST CLI (Phase 3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rak.data.encode import EncoderConfig, present_adaptive
from rak.data.mnist import MnistSplit, load_mnist
from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.network.train import TrainConfig, accuracy, evaluate, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neurons", type=int, default=400)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="batas presentasi (0 = penuh)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--inhibition", type=float, default=17.0)
    p.add_argument("--e-to-i", type=float, default=10.4)
    p.add_argument("--theta-plus", type=float, default=0.05)
    p.add_argument("--theta-max", type=float, default=200.0)
    p.add_argument("--delay-max", type=float, default=10.0)
    p.add_argument("--rest-ms", type=float, default=150.0)
    p.add_argument("--legacy-params", action="store_true")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--assign", type=int, default=5000)
    p.add_argument("--test", type=int, default=0)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--assign-samples", type=int, default=5000)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--out-dir", default="runs/mnist")
    p.add_argument("--no-smoke", action="store_true")
    args = p.parse_args()

    use_ref = not args.legacy_params
    train_data = load_mnist("train")
    if args.limit and args.limit > 0:
        train_data = MnistSplit(images=train_data.images[:args.limit],
                                labels=train_data.labels[:args.limit])
    test_data = load_mnist("test")

    cfg = TrainConfig(
        n_e=args.neurons,
        epochs=args.epochs,
        batch_size=args.batch,
        seed=args.seed,
        inhibition_strength=args.inhibition,
        e_to_i_strength=args.e_to_i,
        theta_plus=args.theta_plus,
        theta_max=args.theta_max,
        delay_max_in_ms=args.delay_max,
        rest_ms=args.rest_ms,
        use_reference_params=use_ref,
        out_dir=args.out_dir,
    )

    net = train(train_data, cfg)

    # Assign classes
    n_assign = min(args.assign_samples, len(train_data.images))
    enc = __import__("rak.data.encode", fromlist=["EncoderConfig"]).EncoderConfig(rest_ms=args.rest_ms)
    gen = torch.Generator().manual_seed(args.seed + 1)
    from rak.data.encode import present_adaptive as pa
    chunks = []
    for s in range(0, n_assign, args.batch):
        c, _ = pa(net, train_data.images[s:s+args.batch], enc, gen, learn=False)
        chunks.append(c)
    counts = torch.cat(chunks)
    from rak.network.competitive import assign_classes
    assignments = assign_classes(counts, train_data.labels[:counts.shape[0]])

    # Evaluate
    n_test = len(test_data.images) if args.test_limit == 0 else min(args.test_limit, len(test_data.images))
    test_counts = torch.zeros(n_test, args.neurons)
    for s in range(0, n_test, args.batch):
        e = min(s + args.batch, n_test)
        total = None
        for _ in range(args.trials):
            c, _ = pa(net, test_data.images[s:e], enc, gen, learn=False)
            total = c if total is None else total + c
        test_counts[s:e] = total

    # Simple accuracy (mean readout)
    proto = torch.zeros(10, args.neurons)
    for c in range(10):
        mask = train_data.labels[:counts.shape[0]] == c
        if mask.sum() > 0:
            proto[c] = counts[mask].mean(dim=0)
    pred = (test_counts @ proto.T).argmax(dim=1)
    acc = float((pred == test_data.labels[:n_test]).float().mean() * 100)
    print(f"AKURASI TEST: {acc:.2f}%")

    # Save
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "assignments": assignments.tolist()}, out / "weights.pt")
    print(f"weights saved to {out / 'weights.pt'}")


if __name__ == "__main__":
    main()
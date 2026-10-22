r"""Readout linear (ridge) di atas spike count.

Jaringan spike tetap satu-satunya yang belajar tanpa label. Readout
ini hanya memetakan vektor spike count (n_e) ke label, dilatih pada
data TRAIN, dipilih hyperparameter-nya di split validasi TRAIN, dan
diukur sekali di TEST.

Kenapa bukan sekadar prototipe kelas: prototipe adalah classifier
linear dengan bobot yang ditentukan tangan (rata-rata kelas). Ridge
mencari bobot yang meminimalkan error, dengan regularisasi. Kalau
fitur neuron memang informatif, ridge harus mengalahkan prototipe --
dan kalau tidak, hasilnya jujur akan terlihat.

Dua transformasi fitur diuji (raw dan log1p); pilihan dilakukan di
validasi, bukan di test.
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
    class_prototypes,
    classify_matched,
)
from rak.network.train import present_adaptive


def _counts(
    net: CompetitiveNetwork,
    images: torch.Tensor,
    enc: EncoderConfig,
    batch: int,
    seed: int,
    trials: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, int(images.shape[0]), batch):
        total = None
        for _ in range(trials):
            c, _ = present_adaptive(
                net, images[s : s + batch], enc, gen, learn=False
            )
            total = c if total is None else total + c
        out.append(total)
    return torch.cat(out, dim=0).to(torch.float64)


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge tertutup: (X'X + alpha I)^-1 X'Y, bias tidak diregularisasi."""
    d = X.shape[1]
    reg = alpha * torch.eye(d, dtype=X.dtype)
    reg[-1, -1] = 0.0
    A = X.T @ X + reg
    return torch.linalg.solve(A, X.T @ Y)


def _standardize(
    Xtr: torch.Tensor, others: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    mu = Xtr.mean(dim=0)
    sd = Xtr.std(dim=0).clamp_min(1e-6)

    def _one(X: torch.Tensor) -> torch.Tensor:
        Z = (X - mu) / sd
        return torch.cat([Z, torch.ones(Z.shape[0], 1, dtype=Z.dtype)], dim=1)

    return _one(Xtr), [_one(o) for o in others]


def _one_hot(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    Y = torch.zeros(y.shape[0], n_classes, dtype=torch.float64)
    Y[torch.arange(y.shape[0]), y] = 1.0
    return Y


def _acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).double().mean() * 100.0)


def _recall(pred: torch.Tensor, y: torch.Tensor) -> list[float]:
    out = []
    for c in range(10):
        m = y == c
        out.append(round(_acc(pred[m], y[m]), 1) if int(m.sum()) else 0.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=10000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inhibition", type=float, default=1600.0)
    ap.add_argument("--e-to-i", type=float, default=1600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    blob = torch.load(a.weights, map_location="cpu", weights_only=False)
    state = blob["net"]
    n_e = int(state["w_e"].shape[1])
    net = CompetitiveNetwork(
        NetworkConfig(
            n_e=n_e,
            inhibition_strength=a.inhibition,
            e_to_i_strength=a.e_to_i,
            rule="variant3",
        ),
        seed=a.seed,
    )
    net.load_state_dict(state)
    enc = EncoderConfig()

    train = load_mnist(split="train")
    test = load_mnist(split="test")
    n_tr = min(a.train_samples, int(train.images.shape[0]))
    n_te = min(a.test_samples, int(test.images.shape[0]))

    tr = _counts(net, train.images[:n_tr], enc, a.batch, a.seed + 1, a.trials)
    te = _counts(net, test.images[:n_te], enc, a.batch, a.seed + 2, a.trials)
    tr_y = train.labels[:n_tr]
    te_y = test.labels[:n_te]

    # Split TRAIN: 80% fit, 20% validasi.
    g = torch.Generator().manual_seed(a.seed + 3)
    perm = torch.randperm(n_tr, generator=g)
    n_fit = int(0.8 * n_tr)
    fit_idx, val_idx = perm[:n_fit], perm[n_fit:]

    results: dict[str, dict[str, float]] = {}

    # --- baseline: prototipe kelas + matched filter (tanpa fit) ------
    proto = class_prototypes(tr[fit_idx], tr_y[fit_idx])
    pv = torch.tensor([classify_matched(x, proto) for x in tr[val_idx]])
    pt = torch.tensor([classify_matched(x, proto) for x in te])
    results["matched_cosz"] = {
        "val": round(_acc(pv, tr_y[val_idx]), 2),
        "test": round(_acc(pt, te_y), 2),
    }

    # --- ridge, dua transformasi fitur ------------------------------
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for name, fn in (
        ("ridge_raw", lambda x: x),
        ("ridge_log1p", lambda x: torch.log1p(x)),
    ):
        X = fn(tr)
        Xfit, (Xval, Xte) = _standardize(X[fit_idx], [X[val_idx], te])
        Yfit = _one_hot(tr_y[fit_idx])
        best_a, best_val = None, -1.0
        for alpha in alphas:
            W = _ridge_fit(Xfit, Yfit, alpha)
            acc = _acc((Xval @ W).argmax(1), tr_y[val_idx])
            if acc > best_val:
                best_a, best_val = alpha, acc
        # Refit pada SELURUH train dengan alpha terpilih.
        Xall, (Xte_all,) = _standardize(X, [te])
        W = _ridge_fit(Xall, _one_hot(tr_y), best_a)
        pred = (Xte_all @ W).argmax(1)
        results[name] = {
            "val": round(best_val, 2),
            "test": round(_acc(pred, te_y), 2),
            "alpha": best_a,
            "recall": _recall(pred, te_y),
        }

    best = max(
        (k for k in results if k != "matched_cosz"),
        key=lambda k: results[k]["val"],
    )
    print(
        f"neuron {n_e}, trials {a.trials}, fit {n_fit}, val {n_tr - n_fit}, "
        f"test {n_te}"
    )
    for k, v in results.items():
        extra = f" alpha={v['alpha']}" if "alpha" in v else ""
        print(f"{k:<14} val {v['val']:>6.2f}  test {v['test']:>6.2f}{extra}")
    print(f"terpilih (validasi): {best} -> test {results[best]['test']:.2f}%")
    print(f"recall per kelas: {results[best]['recall']}")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

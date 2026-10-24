"""CLI: fit readout linear di atas satu atau beberapa checkpoint RAK.

Contoh:
    python -m rak readout --weights a.pt b.pt \\
        --train-samples 5000 --test-samples 10000 --trials 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from rak.data.mnist import load_mnist
from rak.network.competitive import (
    CompetitiveNetwork,
    NetworkConfig,
    class_prototypes,
    classify_matched,
)
from rak.readout.features import ensemble_counts
from rak.readout.linear import RidgeReadout

__all__ = ["main", "load_net"]


def load_net(
    path: str | Path,
    *,
    inhibition: float = 1600.0,
    e_to_i: float = 1600.0,
    seed: int = 0,
) -> tuple[CompetitiveNetwork, int]:
    """Muat checkpoint menjadi jaringan siap pakai (learn=False nanti)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["net"]
    n_e = int(state["w_e"].shape[1])
    net = CompetitiveNetwork(
        NetworkConfig(
            n_e=n_e,
            inhibition_strength=inhibition,
            e_to_i_strength=e_to_i,
            rule="variant3",
        ),
        seed=seed,
    )
    net.load_state_dict(state)
    return net, n_e


def _acc(pred: Tensor, y: Tensor) -> float:
    return float((pred == y).double().mean() * 100.0)


def _recall(pred: Tensor, y: Tensor, n_classes: int = 10) -> list[float]:
    out = []
    for c in range(n_classes):
        m = y == c
        out.append(round(_acc(pred[m], y[m]), 1) if int(m.sum()) else 0.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rak readout",
        description="Fit readout linear (ridge) di atas spike count. "
        "Beberapa checkpoint digabung menjadi satu vektor fitur.",
    )
    ap.add_argument(
        "--weights",
        required=True,
        nargs="+",
        help="satu atau lebih checkpoint; fitur digabung (ensemble)",
    )
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=10000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inhibition", type=float, default=1600.0)
    ap.add_argument("--e-to-i", type=float, default=1600.0)
    ap.add_argument("--rest-ms", type=float, default=0.0,
                   help="jeda zero-input setelah presentasi (ms)")
    ap.add_argument("--legacy-params", action="store_true",
                   help="pakai parameter lama (sparse inhibisi)")
    ap.add_argument("--out", default="", help="tulis hasil JSON ke path ini")
    ap.add_argument(
        "--save-readout",
        default="",
        help="simpan readout terpilih (.pt) untuk dipakai ulang",
    )
    a = ap.parse_args(argv)

    # --- muat model ---------------------------------------------------
    nets: list[CompetitiveNetwork] = []
    for i, path in enumerate(a.weights):
        net, n_e = load_net(
            path,
            inhibition=a.inhibition,
            e_to_i=a.e_to_i,
            seed=a.seed,
        )
        nets.append(net)
        print(f"model {i + 1}: {n_e} neuron, {path}")

    # --- data ---------------------------------------------------------
    train = load_mnist(split="train")
    test = load_mnist(split="test")
    n_tr = min(a.train_samples, int(train.images.shape[0]))
    n_te = min(a.test_samples, int(test.images.shape[0]))

    tr = ensemble_counts(
        nets, train.images[:n_tr], trials=a.trials, batch=a.batch,
        seed=a.seed + 1,
    )
    te = ensemble_counts(
        nets, test.images[:n_te], trials=a.trials, batch=a.batch,
        seed=a.seed + 2,
    )
    tr_y = train.labels[:n_tr]
    te_y = test.labels[:n_te]

    # Split TRAIN: 80% fit, 20% validasi. Test tidak disentuh.
    g = torch.Generator().manual_seed(a.seed + 3)
    perm = torch.randperm(n_tr, generator=g)
    n_fit = int(0.8 * n_tr)
    fit_idx, val_idx = perm[:n_fit], perm[n_fit:]
    x_fit, y_fit = tr[fit_idx], tr_y[fit_idx]
    x_val, y_val = tr[val_idx], tr_y[val_idx]

    results: dict[str, dict] = {}

    # --- baseline: prototipe kelas + matched filter -------------------
    proto = class_prototypes(x_fit, y_fit)
    pv = torch.tensor([classify_matched(x, proto) for x in x_val])
    pt = torch.tensor([classify_matched(x, proto) for x in te])
    results["matched_cosz"] = {
        "val": round(_acc(pv, y_val), 2),
        "test": round(_acc(pt, te_y), 2),
    }

    # --- ridge: raw dan log1p, alpha dipilih di validasi -------------
    fitted: dict[str, RidgeReadout] = {}
    for transform in ("raw", "log1p"):
        model = RidgeReadout(transform=transform)
        model.fit(x_fit, y_fit, x_val=x_val, y_val=y_val)
        fitted[transform] = model
        pred = model.predict(te)
        results[f"ridge_{transform}"] = {
            "val": round(model.val_accuracy_ * 100.0, 2),
            "test": round(_acc(pred, te_y), 2),
            "alpha": model.alpha_,
            "recall": _recall(pred, te_y),
        }

    best_key = max(
        (k for k in results if k.startswith("ridge_")),
        key=lambda k: results[k]["val"],
    )
    best_transform = best_key.removeprefix("ridge_")
    best = fitted[best_transform]

    print(
        f"neuron {tr.shape[1]}, trials {a.trials}, fit {n_fit}, "
        f"val {n_tr - n_fit}, test {n_te}"
    )
    for k, v in results.items():
        extra = f" alpha={v['alpha']}" if "alpha" in v else ""
        print(f"{k:<14} val {v['val']:>6.2f}  test {v['test']:>6.2f}{extra}")
    print(
        f"terpilih (validasi): {best_key} -> "
        f"test {results[best_key]['test']:.2f}%"
    )
    print(f"recall per kelas: {results[best_key]['recall']}")

    if a.save_readout:
        best.save(a.save_readout)
        print(f"readout tersimpan: {a.save_readout}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

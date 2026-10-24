"""CLI: rak readout — fit ridge readout on spike counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rak.data.mnist import load_mnist
from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.readout.features import collect_counts, ensemble_counts
from rak.readout.linear import RidgeReadout


def load_net(path, inhibition=17.0, e_to_i=10.4, seed=0):
    import torch as t
    blob = t.load(path, map_location="cpu", weights_only=False)
    state = blob["net"]
    n_e = int(state["w_e"].shape[1])
    net = CompetitiveNetwork(NetworkConfig(n_e=n_e, inhibition_strength=inhibition, e_to_i_strength=e_to_i, rule="variant3"), seed=seed)
    net.load_state_dict(state)
    return net, n_e


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rak readout")
    ap.add_argument("--weights", required=True, nargs="+")
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=10000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inhibition", type=float, default=17.0)
    ap.add_argument("--e-to-i", type=float, default=10.4)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-readout", default="")
    a = ap.parse_args(argv)

    nets = []
    for path in a.weights:
        net, n_e = load_net(path, a.inhibition, a.e_to_i, a.seed)
        nets.append(net)
        print(f"model {len(nets)}: {n_e} neuron, {path}")

    train = load_mnist("train")
    test = load_mnist("test")
    n_tr = min(a.train_samples, len(train.images))
    n_te = min(a.test_samples, len(test.images))

    from rak.data.encode import EncoderConfig
    enc = EncoderConfig()
    tr = ensemble_counts(nets, train.images[:n_tr], trials=a.trials, batch=a.batch, seed=a.seed + 1, enc=enc)
    te = ensemble_counts(nets, test.images[:n_te], trials=a.trials, batch=a.batch, seed=a.seed + 2, enc=enc)
    tr_y = train.labels[:n_tr]
    te_y = test.labels[:n_te]

    g = torch.Generator().manual_seed(a.seed + 3)
    perm = torch.randperm(n_tr, generator=g)
    n_fit = int(0.8 * n_tr)
    fit_idx, val_idx = perm[:n_fit], perm[n_fit:]
    x_fit, y_fit = tr[fit_idx], tr_y[fit_idx]
    x_val, y_val = tr[val_idx], tr_y[val_idx]

    results = {}
    # matched baseline on first model
    from rak.network.competitive import class_prototypes, classify_matched
    proto = class_prototypes(tr[:n_tr] if False else tr[fit_idx][:, :nets[0].config.n_e] if False else None, y_fit) if False else None

    # Ridge
    for transform in ("raw", "log1p"):
        rr = RidgeReadout()
        # Use log1p by transforming features
        x_fit_t = torch.log1p(x_fit) if transform == "log1p" else x_fit
        x_val_t = torch.log1p(x_val) if transform == "log1p" else x_val
        te_t = torch.log1p(te) if transform == "log1p" else te
        rr.fit(x_fit_t, y_fit, x_val=x_val_t, y_val=y_val)
        pred = rr.predict(te_t)
        results[f"ridge_{transform}"] = {
            "val": round(rr.val_accuracy_ * 100, 2),
            "test": round(float((pred == te_y).float().mean() * 100), 2),
            "alpha": rr.alpha_,
        }
        print(f"ridge_{transform:<8} val {results[f'ridge_{transform}']['val']:>6.2f}  test {results[f'ridge_{transform}']['test']:>6.2f} alpha={rr.alpha_}")

    best = max((k for k in results if k.startswith("ridge_")), key=lambda k: results[k]["val"])
    print(f"terpilih: {best} -> test {results[best]['test']:.2f}%")
    print(f"recall: {results[best].get(chr(39)+chr(39), [])}")

    if a.save_readout:
        rr.save(a.save_readout)
        print(f"readout tersimpan: {a.save_readout}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
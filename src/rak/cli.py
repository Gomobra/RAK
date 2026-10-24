"""CLI entry point — Phase 1."""

from __future__ import annotations

import argparse
import sys

import torch

from rak import __version__


def _cmd_info(_args):
    print(f"RAK {__version__}")
    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}")
    print(f"cuda    {torch.cuda.is_available()}")
    return 0


def _cmd_selfcheck(_args):
    from rak.neuron import AdaptiveLIF, Izhikevich, LIF, SpikeLayer

    lif = LIF(1)
    rate = lif.firing_rate(current=15.0, n_steps=500)
    assert rate > 0, "LIF harus fire"

    alif = AdaptiveLIF(1)
    rate_a = alif.firing_rate(current=15.0, n_steps=500)
    assert rate_a > 0, "AdaptiveLIF harus fire"

    iz = Izhikevich(4)
    I = torch.full((100, 4), 10.0)
    sp = iz.simulate(I)
    assert sp.sum() > 0, "Izhikevich harus fire"

    layer = SpikeLayer(n_in=16, n_out=8, seed=1)
    x = torch.rand(20, 4, 16).lt(0.3).float()
    y, v = layer.forward(x, None)
    assert y.shape == (20, 4, 8)

    print(f"ok: lif={rate:.1f}Hz alif={rate_a:.1f}Hz izh={int(sp.sum())} layer={tuple(y.shape)}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "readout":
        from rak.readout.cli import main as readout_main
        return readout_main(argv[1:])
    if argv and argv[0] == "serve":
        try:
            import uvicorn
            from rak.serve import app
        except ImportError:
            print("pip install fastapi uvicorn python-multipart Pillow")
            return 1
        host, port = "0.0.0.0", 8000
        weights_path, readout_path = None, None
        i = 1
        while i < len(argv):
            if argv[i] == "--port":
                port = int(argv[i + 1]); i += 2; continue
            if argv[i] == "--weights":
                weights_path = argv[i + 1]; i += 2; continue
            if argv[i] == "--readout":
                readout_path = argv[i + 1]; i += 2; continue
            i += 1
        if weights_path:
            app.state.weights_path = weights_path
            app.state.readout_path = readout_path
        uvicorn.run(app, host=host, port=port)
        return 0
    if argv and argv[0] == "generate":
        import torch
        from rak.network.competitive import CompetitiveNetwork, NetworkConfig
        ap = argparse.ArgumentParser(prog="rak generate")
        ap.add_argument("--weights", required=True)
        ap.add_argument("--out", default="generated.png")
        ap.add_argument("--n", type=int, default=10)
        ap.add_argument("--class-idx", type=int, default=None)
        gargs = ap.parse_args(argv[1:])
        blob = torch.load(gargs.weights, map_location="cpu", weights_only=False)
        state = blob["net"]
        n_e = int(state["w_e"].shape[1])
        net = CompetitiveNetwork(NetworkConfig(n_e=n_e), seed=1)
        net.load_state_dict(state)
        if gargs.class_idx is not None:
            img = net.generate_class(blob.get("assignments", []), gargs.class_idx)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imshow(img.numpy(), cmap="hot_r")
            plt.axis("off")
            plt.savefig(gargs.out, dpi=120)
            print(f"tersimpan: {gargs.out}")
        else:
            imgs = net.generate_samples(n=gargs.n)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            ncols = min(gargs.n, 5)
            nrows = (gargs.n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
            axes = np.array(axes).reshape(-1)
            for i in range(gargs.n):
                axes[i].imshow(imgs[i].numpy(), cmap="hot_r")
                axes[i].axis("off")
            for i in range(gargs.n, len(axes)):
                axes[i].axis("off")
            fig.savefig(gargs.out, dpi=120)
            print(f"tersimpan: {gargs.out}")
        return 0
    if argv and argv[0] == "serve":
        import uvicorn
        from rak.serve import app
        host = "0.0.0.0"
        port = 8000
        weights_path = None
        readout_path = None
        for i, a in enumerate(argv[1:]):
            if a == "--port" and i + 2 <= len(argv):
                port = int(argv[i + 2])
            if a == "--weights" and i + 2 <= len(argv):
                weights_path = argv[i + 2]
            if a == "--readout" and i + 2 <= len(argv):
                readout_path = argv[i + 2]
        if weights_path:
            app.state.weights_path = weights_path
            app.state.readout_path = readout_path
        uvicorn.run(app, host=host, port=port)
        return 0

    parser = argparse.ArgumentParser(prog="rak")
    parser.add_argument("--version", action="version", version=f"RAK {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info").set_defaults(fn=_cmd_info)
    sub.add_parser("selfcheck").set_defaults(fn=_cmd_selfcheck)
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        args.fn = _cmd_info
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
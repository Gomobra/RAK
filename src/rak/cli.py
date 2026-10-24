"""CLI entry point — RAK (Recurrent Adaptive Knowledge)."""

from __future__ import annotations

import argparse
import sys

import torch

from rak import __version__

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _console = Console()
    _HAS_RICH = True
except ImportError:
    _console = None
    _HAS_RICH = False


def _cmd_info(_args):
    if _HAS_RICH:
        t = Table(title=f"RAK {__version__} — Recurrent Adaptive Knowledge")
        t.add_column("Key", style="cyan")
        t.add_column("Value", style="green")
        t.add_row("python", sys.version.split()[0])
        t.add_row("torch", torch.__version__)
        t.add_row("cuda", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            t.add_row("gpu", torch.cuda.get_device_name(0))
        _console.print(t)
    else:
        print(f"RAK {__version__} — Recurrent Adaptive Knowledge")
        print(f"python  {sys.version.split()[0]}")
        print(f"torch   {torch.__version__}")
        print(f"cuda    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu     {torch.cuda.get_device_name(0)}")
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

    msg = f"ok: lif={rate:.1f}Hz alif={rate_a:.1f}Hz izh={int(sp.sum())} layer={tuple(y.shape)}"
    if _HAS_RICH:
        _console.print(f"[green]{msg}[/green]")
    else:
        print(msg)
    return 0


def _cmd_train(args):
    """Train SNN on MNIST."""
    import subprocess
    cmd = ["python", "scripts/run_mnist.py",
           "--neurons", str(args.neurons),
           "--epochs", str(args.epochs),
           "--seed", str(args.seed),
           "--inhibition", str(args.inhibition),
           "--theta-plus", str(args.theta_plus),
           "--theta-max", str(args.theta_max),
           "--delay-max", str(args.delay_max),
           "--rest-ms", str(args.rest_ms),
           "--assign-samples", str(args.assign),
           "--test-limit", str(args.test),
           "--trials", str(args.trials),
           "--out-dir", args.out_dir]
    if args.legacy_params:
        cmd.append("--legacy-params")
    if args.device:
        cmd += ["--device", args.device]
    if args.no_smoke:
        cmd.append("--no-smoke")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def _cmd_eval(args):
    """Evaluate a checkpoint."""
    import subprocess
    cmd = ["python", "scripts/run_mnist.py",
           "--neurons", str(args.neurons),
           "--epochs", "1",
           "--seed", str(args.seed),
           "--inhibition", str(args.inhibition),
           "--theta-plus", str(args.theta_plus),
           "--theta-max", str(args.theta_max),
           "--delay-max", str(args.delay_max),
           "--rest-ms", str(args.rest_ms),
           "--assign-samples", str(args.assign),
           "--test-limit", str(args.test),
           "--trials", str(args.trials),
           "--out-dir", args.out_dir,
           "--resume", args.weights,
           "--reassign"]
    if args.legacy_params:
        cmd.append("--legacy-params")
    if args.device:
        cmd += ["--device", args.device]
    cmd.append("--no-smoke")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def _cmd_generate(args):
    """Generate images from a checkpoint."""
    import torch
    from rak.generate import generate_samples, generate_class

    blob = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = blob["net"]
    w = state["w_e"]
    assignments = blob.get("assignments", [])

    if args.class_idx is not None:
        img = generate_class(w, assignments, args.class_idx, seed=args.seed)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imshow(img.numpy(), cmap="hot_r")
        plt.axis("off")
        plt.title(f"Generated: class {args.class_idx}")
        plt.savefig(args.out, dpi=120)
        print(f"tersimpan: {args.out}")
    else:
        imgs = generate_samples(w, n=args.n, seed=args.seed)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        ncols = min(args.n, 5)
        nrows = (args.n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = np.array(axes).reshape(-1)
        for i in range(args.n):
            axes[i].imshow(imgs[i].numpy(), cmap="hot_r")
            axes[i].set_title(f"img {i}", fontsize=8)
            axes[i].axis("off")
        for i in range(args.n, len(axes)):
            axes[i].axis("off")
        fig.savefig(args.out, dpi=120)
        print(f"tersimpan: {args.out}")
    return 0


def _cmd_serve(args):
    """Start FastAPI serving."""
    try:
        import uvicorn
        from rak.serve import app
    except ImportError:
        print("pip install fastapi uvicorn python-multipart Pillow")
        return 1
    app.state.weights_path = args.weights
    app.state.readout_path = args.readout
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_readout(args):
    """Fit ridge readout (delegated)."""
    from rak.readout.cli import main as readout_main
    argv = []
    if args.weights:
        argv += ["--weights"] + args.weights
    if args.train_samples:
        argv += ["--train-samples", str(args.train_samples)]
    if args.test_samples:
        argv += ["--test-samples", str(args.test_samples)]
    if args.trials:
        argv += ["--trials", str(args.trials)]
    if args.out:
        argv += ["--out", args.out]
    if args.save_readout:
        argv += ["--save-readout", args.save_readout]
    return readout_main(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("readout", "serve", "generate"):
        # Delegate to sub-handlers
        cmd = argv[0]
        rest = argv[1:]
        if cmd == "readout":
            return _cmd_readout(_parse_readout(rest))
        if cmd == "serve":
            return _cmd_serve(_parse_serve(rest))
        if cmd == "generate":
            return _cmd_generate(_parse_generate(rest))

    parser = argparse.ArgumentParser(prog="rak", description="RAK — Recurrent Adaptive Knowledge")
    parser.add_argument("--version", action="version", version=f"RAK {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="versi + backend").set_defaults(fn=_cmd_info)
    sub.add_parser("selfcheck", help="smoke test").set_defaults(fn=_cmd_selfcheck)

    tr = sub.add_parser("train", help="train SNN on MNIST")
    tr.add_argument("--neurons", type=int, default=400)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument("--inhibition", type=float, default=17.0)
    tr.add_argument("--theta-plus", type=float, default=0.05)
    tr.add_argument("--theta-max", type=float, default=200.0)
    tr.add_argument("--delay-max", type=float, default=0.0)
    tr.add_argument("--rest-ms", type=float, default=0.0)
    tr.add_argument("--legacy-params", action="store_true")
    tr.add_argument("--assign", type=int, default=5000)
    tr.add_argument("--test", type=int, default=0)
    tr.add_argument("--trials", type=int, default=1)
    tr.add_argument("--out-dir", default="runs/mnist")
    tr.add_argument("--device", default=None)
    tr.add_argument("--no-smoke", action="store_true")
    tr.set_defaults(fn=_cmd_train)

    ev = sub.add_parser("eval", help="evaluate checkpoint")
    ev.add_argument("--weights", required=True)
    ev.add_argument("--neurons", type=int, default=400)
    ev.add_argument("--seed", type=int, default=1)
    ev.add_argument("--inhibition", type=float, default=17.0)
    ev.add_argument("--theta-plus", type=float, default=0.05)
    ev.add_argument("--theta-max", type=float, default=200.0)
    ev.add_argument("--delay-max", type=float, default=0.0)
    ev.add_argument("--rest-ms", type=float, default=0.0)
    ev.add_argument("--legacy-params", action="store_true")
    ev.add_argument("--assign", type=int, default=5000)
    ev.add_argument("--test", type=int, default=0)
    ev.add_argument("--trials", type=int, default=5)
    ev.add_argument("--out-dir", default="runs/mnist")
    ev.add_argument("--device", default=None)
    ev.set_defaults(fn=_cmd_eval)

    sv = sub.add_parser("serve", help="start FastAPI server")
    sv.add_argument("--weights", required=True)
    sv.add_argument("--readout", default=None)
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=_cmd_serve)

    gn = sub.add_parser("generate", help="generate images from checkpoint")
    gn.add_argument("--weights", required=True)
    gn.add_argument("--out", default="generated.png")
    gn.add_argument("--n", type=int, default=10)
    gn.add_argument("--class-idx", type=int, default=None)
    gn.add_argument("--seed", type=int, default=0)
    gn.set_defaults(fn=_cmd_generate)

    rd = sub.add_parser("readout", help="fit ridge readout")
    rd.add_argument("--weights", nargs="+", required=True)
    rd.add_argument("--train-samples", type=int, default=5000)
    rd.add_argument("--test-samples", type=int, default=10000)
    rd.add_argument("--trials", type=int, default=5)
    rd.add_argument("--out", default="")
    rd.add_argument("--save-readout", default="")
    rd.set_defaults(fn=_cmd_readout)

    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        args.fn = _cmd_info
    return args.fn(args)


def _parse_readout(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--train-samples", type=int, default=5000)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--out", default="")
    p.add_argument("--save-readout", default="")
    return p.parse_args(argv)


def _parse_serve(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--readout", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def _parse_generate(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="generated.png")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--class-idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)
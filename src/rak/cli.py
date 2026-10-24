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
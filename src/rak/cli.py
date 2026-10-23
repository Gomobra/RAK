"""CLI entry point.

Fase 0 deliverable: `python -m rak` harus jalan dan=self-check, supaya
scaffold bisa diverifikasi tanpa GPU dan tanpa data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from rak import __version__


def _cmd_info(_args: argparse.Namespace) -> int:
    """Tampilkan versi + backend — gate pertama reproducibility."""
    print(f"RAK {__version__}")
    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}")
    print(f"cuda    {torch.cuda.is_available()}")
    return 0


def _cmd_selfcheck(_args: argparse.Namespace) -> int:
    """Smoke test: LIF + AdaptiveLIF + SpikeLayer harus jalan."""
    from rak.neuron import AdaptiveLIF, LIF, SpikeLayer

    lif = LIF(1)
    rate = lif.firing_rate(current=1.5, n_steps=500)
    assert rate > 0, "LIF harus fire pada input di atas threshold"

    alif = AdaptiveLIF(1)
    rate_a = alif.firing_rate(current=1.5, n_steps=500)
    assert rate_a > 0, "AdaptiveLIF harus fire"

    layer = SpikeLayer(n_in=16, n_out=8, seed=1)
    x = torch.rand(20, 4, 16).lt(0.3).float()
    y, state, _ = layer.forward(x, None)
    assert y.shape == (20, 4, 8), f"shape salah: {tuple(y.shape)}"

    print(f"ok: lif_rate={rate:.1f}Hz alif_rate={rate_a:.1f}Hz layer={tuple(y.shape)}")
    return 0


def _mnist_like_input(
    n_steps: int, batch: int, n_in: int, seed: int
) -> torch.Tensor:
    """Input spike train dengan statistik mirip MNIST.

    Statistik MNIST (28x28, nilai dinormalisasi ke [0, 1]):
      - ~20% piksel non-zero, intensitas non-zero rata-rata ~0.5
      - mean per piksel (termasuk nol) ~0.13

    Kita tidak bisa mengunduh MNIST asli di lingkungan tanpa jaringan,
    jadi ini SINTETIK dan harus dibaca begitu — bukan benchmark MNIST
    sungguhan. Tujuannya cuma satu: memberi benchmark dengan spike
    density yang realistis, bukan noise acak yang selalu membuat neuron
    saturasi atau mati.

    Args:
        n_steps: jumlah timestep
        batch: ukuran batch
        n_in: jumlah input neuron
        seed: seed generator

    Returns:
        Tensor (n_steps, batch, n_in) biner
    """
    from rak.neuron.trace import poisson_spike_train

    gen = torch.Generator().manual_seed(seed)

    # ~20% piksel "menyala", intensitas tersebar di [0.05, 1.0]
    lit = torch.rand(batch, n_in, generator=gen) < 0.20
    intensity = torch.rand(batch, n_in, generator=gen) * 0.95 + 0.05
    values = torch.where(lit, intensity, torch.zeros_like(intensity))

    spk = poisson_spike_train(
        values * 63.75, duration_ms=n_steps * 1.0, dt_ms=1.0, generator=gen
    )
    return spk.movedim(-1, 0)


def _cmd_bench(args: argparse.Namespace) -> int:
    """T1.4.5 — throughput SpikeLayer."""
    import time

    from rak.neuron import SpikeLayer

    layer = SpikeLayer(n_in=args.n_in, n_out=args.n_out, seed=args.seed)

    if args.input == "mnist-like":
        x = _mnist_like_input(args.steps, args.batch, args.n_in, args.seed)
    else:
        x = torch.rand(
            args.steps, args.batch, args.n_in
        ).lt(args.rate).float()

    # warmup
    layer.forward(x, None)
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        y, _, _ = layer.forward(x, None)
    elapsed = time.perf_counter() - t0

    n_spikes = float(y.sum().item())
    neuron_updates = args.steps * args.batch * args.n_out * args.repeats
    cells = args.steps * args.batch * args.n_out
    density = n_spikes / cells

    with torch.no_grad():
        yn, _, _ = layer.forward(x, None)
    fired = int((yn.sum(0) > 0).sum())

    print(f"config       : {args.n_in}in -> {args.n_out}out, "
          f"{args.steps} steps, batch {args.batch}, input={args.input}")
    print(f"elapsed      : {elapsed:.3f}s over {args.repeats} repeats")
    print(f"neuron-upd/s : {neuron_updates / elapsed:,.0f}")
    print(f"spikes/s     : {n_spikes * args.repeats / elapsed:,.0f}")
    print(f"spike density: {density:.4f}  (SynOps per cell = {density:.4f})")
    print(f"neurons fired: {fired}/{args.n_out}")
    print(f"synapses     : {args.n_in * args.n_out:,}")

    if density == 0.0:
        print(
            "\nWARNING: density 0 — layer diam total. Cek kalibrasi "
            "bobot (lihat docs/theory.md §7)."
        )
    elif density > 0.5:
        print(
            f"\nWARNING: density {density:.2f} sangat tinggi — neuron "
            "mendekati saturasi, klaim sparsity tidak valid."
        )
    return 0


def _cmd_viz(args: argparse.Namespace) -> int:
    """T1.1.4 / T1.2.4 — plot V(t), spike train, dan perbandingan model."""
    import torch

    from rak.neuron import AdaptiveLIF, LIF, Izhikevich
    from rak.neuron.viz import (
        plot_adaptation,
        plot_comparison,
        plot_membrane,
        plot_spike_raster,
    )

    out = Path(args.out)
    T = args.steps
    cur = torch.full((T,), args.current)
    written = []

    lif = LIF(1, tau_mem_ms=args.tau_mem)
    s_lif, _, v_lif = lif.simulate(cur, record=True)
    written.append(
        plot_membrane(
            v_lif[:, 0], s_lif, out / "01_lif.png", title="LIF", v_th=1.0
        )
    )

    alif = AdaptiveLIF(1, tau_mem_ms=args.tau_mem, tau_adapt_ms=args.tau_adapt)
    s_alif, _, (v_al, th_al) = alif.simulate(cur, record=True)
    written.append(
        plot_membrane(
            v_al[:, 0], s_alif, out / "02_adaptive_lif.png",
            title="Adaptive LIF", v_th=1.0,
        )
    )
    written.append(
        plot_adaptation(
            v_al[:, 0], th_al[:, 0], s_alif,
            out / "03_adaptation_detail.png",
        )
    )
    written.append(
        plot_comparison(
            {"LIF": (v_lif[:, 0], s_lif), "Adaptive": (v_al[:, 0], s_alif)},
            out / "04_comparison.png",
        )
    )

    izh = Izhikevich(1)
    s_izh, _, (v_izh, _w_izh) = izh.simulate(
        torch.full((T * 4,), args.current * 10.0), record=True
    )
    written.append(
        plot_membrane(
            v_izh[:, 0], s_izh, out / "05_izhikevich.png",
            title="Izhikevich", v_th=30.0,
        )
    )

    from rak.neuron import SpikeLayer

    # Pakai input statistik-MNIST yang SAMA dengan `rak bench`, supaya
    # angka density di figure ini konsisten dengan benchmark. Kalau
    # pakai torch.rand (mean 0.5), neuron hampir saturasi dan klaim
    # sparsity jadi kelihatan palsu.
    layer = SpikeLayer(n_in=args.n_in, n_out=args.n_out, seed=args.seed)
    for val, tag in ((0.13, "dim"), (0.30, "sedang"), (1.00, "terang")):
        base = torch.full((1, args.n_in), val)
        x = layer.encode(
            base, n_steps=args.steps, max_rate_hz=layer.max_rate_hz,
            seed=args.seed,
        )
        with torch.no_grad():
            y, _, _ = layer.forward(x, None)
        written.append(
            plot_spike_raster(
                y,
                out / f"06_layer_raster_{tag}.png",
                title=f"SpikeLayer {args.n_in}->{args.n_out} "
                f"(input mean={val:.2f})",
                max_neurons=60,
            )
        )

    # ---- 7: GATE — kurva belajar asosiasi ---------------------------
    from rak.neuron import LIF as _LIF
    from rak.neuron.viz import plot_association_learning
    from rak.learning.stdp import make_trace as _make_trace

    _N = 2
    _W_MAX, _W_INIT, _DELTA, _TRIAL = 3.0, 0.05, 5, 40

    def _assoc_epoch_count(epoch):
        """Jalankan `epoch` epoch-diajar asosiasi A->B.

        Returns (w_ab, w_aa, respons A-ke-B %, respons B-ke-B %).
        Respons diukur sebagai spike PERTAMA per trial, bukan rate
        akumulatif — rate bisa tinggi hanya karena neuron tereksitasi
        berulang, yang bukan bukti asosiasi.
        """
        cfg = STDPConfig(eta_plus=2e-2, eta_minus=1e-4, w_max=_W_MAX)
        w = torch.full((_N, _N), _W_INIT)
        tp, tq = _make_trace(_N), _make_trace(_N)
        for _ in range(max(epoch, 1)):
            for t in range(_TRIAL):
                zp, zq = torch.zeros(_N), torch.zeros(_N)
                if t == 0:
                    zp[0] = 1.0
                if t == _DELTA:
                    zq[1] = 1.0
                w, tp, tq = stdp_step(zp, zq, w, tp, tq, cfg)

        neuron = _LIF(_N, tau_mem_ms=20.0, v_th=1.0, v_reset=0.0)
        res = []
        for stim, out_idx in ((0, 1), (1, 1)):
            hits = 0
            for _ in range(30):
                st = neuron.init_state()
                for t in range(80):
                    cur = torch.zeros(_N)
                    if t < 30:
                        cur[stim] = 1.0
                    z, st = neuron.step(cur @ w, st)
                    if float(z[out_idx]) > 0.0:
                        hits += 1
                        break
            res.append(100.0 * hits / 30)
        return float(w[0, 1]), float(w[0, 0]), res[0], res[1]

    grid = [5, 10, 20, 40, 60, 80, 120, 160, 200, 250, 300]
    rows = [_assoc_epoch_count(e) for e in grid]
    written.append(plot_association_learning(
        grid,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        out_path=out / "07_association_learning.png",
    ))

    for p in written:
        print(f"wrote {p}")
    return 0


def _cmd_viz_phase2(args: argparse.Namespace) -> int:
    """Figure Phase 2: learning rules + homeostasis."""
    from pathlib import Path

    import torch

    from rak.homeostasis.inhibition import (
        InhibitionConfig,
        build_inhibitory_weights,
        make_inhibitory_neuron,
    )
    from rak.homeostasis.intrinsic import (
        TIME_SCALES,
        HomeostaticConfig,
        make_homeostatic_neuron,
    )
    from rak.homeostasis.scaling import (
        ScalingConfig,
        hard_normalize,
        incoming_drive,
        soft_normalize,
    )
    from rak.learning.stdp import STDPConfig, make_trace, stdp_step, stdp_window
    from rak.neuron.viz import (
        plot_column_normalization,
        plot_drive_history,
        plot_stdp_window,
        plot_theta_adaptation,
    )

    out = Path(args.out)
    written = []

    # ---- 1-3: STDP window -------------------------------------------
    written.append(plot_stdp_window(
        STDPConfig(), out / "01_stdp_window.png"
    ))
    written.append(plot_stdp_window(
        STDPConfig(tau_pre_ms=50.0, tau_post_ms=10.0),
        out / "02_stdp_window_asym.png",
    ))
    # Figure 03 — inilah yang TIDAK bekerja di versi sebelumnya.
    # `STDPConfig(mu=1.0)` tanpa `algorithm=` jatuh ke default
    # "additive", jadi `mu` dibuang dan ketiga kurva w_ref keluar
    # identik — figure mengklaim menunjukkan multiplicative padahal
    # tidak. Sekarang `algorithm` dan `mu` dikirim BERSAMAAN.
    #
    # Hasilnya justru figure yang lebih jujur: dua panel atas (additive,
    # additive_step) memang TIDAK bergantung pada w — itu kebenaran,
    # bukan bug. Dua panel bawah baru benar-benar terpisah.
    written.append(plot_stdp_window(
        STDPConfig(mu=1.0, algorithm="multiplicative_pow"),
        out / "03_stdp_window_variants.png",
        w_refs=(0.1, 0.5, 0.9),
        title="STDP window vs bobot saat ini "
              "(mu=1) — additive TIDAK bergantung pada w, "
              "multiplicative YA",
    ))

    # ---- 4: theta adaptation -------------------------------------------
    # Dua eksperimen terpisah, karena mencampur keduanya bikin kurvanya
    # tumpang tindih dan tidak teach anything:
    #
    #   (a) DECAY  — theta mulai dari 1.0, input NOL (jadi tidak ada
    #       spike sama sekali). Kurva jadi murni laju peluruhan, dan
    #       perbedaan tau terlihat jelas di log scale.
    #   (b) RATE   — beta_theta divariasikan pada tau tetap. Ini yang
    #       membuktikan ADR-0005: beta besar -> lebih sedikit spike.
    decay_runs = {}
    for label, scale in (
        ("demo_seconds (5 s)", "demo_seconds"),
        ("realistic_hour (1 jam)", "realistic_hour"),
        ("diehl_cook (2.8 jam)", "diehl_cook_2.8h"),
    ):
        n = make_homeostatic_neuron(
            1,
            config=HomeostaticConfig(
                tau_theta_ms=TIME_SCALES[scale], tau_mem_ms=20.0
            ),
            strict=True,
        )
        state = n.init_state()
        state.theta.fill_(1.0)  # neuron "baru saja" terlalu aktif
        theta_series = []
        for _ in range(20000):
            # Arus nol: v membocor ke 0, jadi neuron tidak akan fire,
            # dan theta meluruh murni karena decay.
            _, state = n.step(torch.zeros(1), state)
            theta_series.append(float(state.theta[0]))
        decay_runs[label] = torch.tensor(theta_series)

    rate_runs = {}
    for label, beta in (
        ("beta=0 (LIF biasa)", 0.0),
        ("beta=0.05", 0.05),
        ("beta=0.2 (kuat)", 0.2),
    ):
        n = make_homeostatic_neuron(
            1,
            config=HomeostaticConfig(
                tau_theta_ms=TIME_SCALES["demo_seconds"],
                beta_theta=beta,
                tau_mem_ms=20.0,
            ),
            strict=True,
        )
        spikes, _, _ = n.simulate(torch.full((20000,), 1.2), record=True)
        rate_runs[label] = spikes

    written.append(plot_theta_adaptation(
        decay_runs, rate_runs, out / "04_theta_adaptation.png"
    ))

    # ---- 5: GATE — drive dengan vs tanpa homeostasis ------------------
    n_in, n_out = args.n_in, args.n_out

    # Dibuat di luar `run` supaya figure 6 (column normalization) juga
    # memakai rule yang sama persis dengan gate test.
    stdp_cfg = STDPConfig(eta_plus=2e-3, eta_minus=1e-4, w_max=1.0)

    def run(homeostasis: bool):
        g = torch.Generator().manual_seed(args.seed)
        cfg = stdp_cfg
        sc = ScalingConfig(target_sum=4.0, soft_power=0.02)
        w = torch.rand(n_in, n_out, generator=g) * 0.3 + 0.01
        tp, tq = make_trace(n_in), make_trace(n_out)
        exc = make_homeostatic_neuron(
            n_out,
            config=HomeostaticConfig(
                tau_theta_ms=TIME_SCALES["realistic_hour"],
                beta_theta=0.05 if homeostasis else 0.0,
                tau_mem_ms=20.0,
            ),
            strict=True,
        )
        st = exc.init_state()
        inh = make_inhibitory_neuron(n_out)
        ins = inh.init_state()
        iw = build_inhibitory_weights(
            n_out,
            InhibitionConfig(inhibition_strength=1.0, scale_per_neuron=False),
        )
        history, w_last = [], None
        for epoch in range(args.epochs):
            for _ in range(args.steps):
                zp = (torch.rand(n_in, generator=g) < 0.15).float()
                zq, st = exc.step(zp @ w, st)
                zi, ins = inh.step(zq @ iw.e_to_i, ins)
                inhib = zi @ iw.i_to_e
                st = type(st)(
                    v=(st.v + inhib).clamp(min=0.0),
                    theta=st.theta,
                    timer=st.timer,
                )
                w, tp, tq = stdp_step(zp, zq, w, tp, tq, cfg)
            if homeostasis and epoch % 5 == 0:
                w = soft_normalize(hard_normalize(w, sc), sc)
            history.append(float(w.sum()))
        return history, w.clone()

    plain_hist, w_plain = run(False)
    homeo_hist, w_homeo = run(True)
    written.append(plot_drive_history(
        plain_hist, homeo_hist, out / "05_homeostatic_gate.png",
        ceiling=float(n_in * n_out),
    ))

    # ---- 6: column normalization --------------------------------------
    # CATATAN METHODOLOGIS — ini yang bikin gw sempat hampir salah:
    #
    # Awalnya figure ini memakai bobot dari run STDP sungguhan. Hasilnya
    # membingungkan: CV-nya cuma 0.06, bahkan MAKIN KECIL kalau input
    # made heterogen (0.014). Setelah dicek, itu bukan salah di plotting.
    #
    #   1. STDP di jaringan 1-lapis dengan input homogen SANGAT seragam:
    #      tiap neuron postsynaptic dapat total input yang sama, jadi
    #      kolomnya memang sudah seimbang.
    #   2. Input heterogen justru "menyembuhkan": neuron yang dapat lebih
    #      banyak fire lebih sering, theta-nya naik, dan rate-nya
    #      ditekan balik — persis ADR-0005 (theta = pembatas daya).
    #
    # Imbalance yang realistis butuh struktur BERLAPIS: kolom layer
    # pertama tidak seimbang kalau bobot ke layer kedua tidak
    # seimbang. Itu belum dibangun di Phase 2.
    #
    # Jadi figure ini memakai matriks yang SENGAJA dibuat tidak
    # seimbang (faktor lognormal per kolom) untuk menunjukkan operasi
    # column normalization itu sendiri, bukan hasil jaringan.
    g = torch.Generator().manual_seed(args.seed + 1)
    w_base = torch.rand(n_in, n_out, generator=g) * 0.4 + 0.05
    # lognormal per kolom: sebagian neuron "menang" dan dapat drive
    # jauh lebih besar dari yang lain.
    col_scale = torch.exp(torch.randn(n_out, generator=g) * 0.6)
    w_rough = w_base * col_scale.unsqueeze(0)

    # Target = drive rata-rata SEBELUM normalisasi, supaya kedua panel
    # berada di skala sama dan perbandingannya jujur.
    mean_drive = float(incoming_drive(w_rough).mean())
    w_scaled = hard_normalize(w_rough, ScalingConfig(target_sum=mean_drive))

    written.append(plot_column_normalization(
        w_rough, w_scaled, out / "06_column_normalization.png",
        target_sum=mean_drive,
    ))

    # ---- 7: GATE — kurva belajar asosiasi ---------------------------
    from rak.neuron import LIF as _LIF
    from rak.neuron.viz import plot_association_learning
    from rak.learning.stdp import make_trace as _make_trace

    _N = 2
    _W_MAX, _W_INIT, _DELTA, _TRIAL = 3.0, 0.05, 5, 40

    def _assoc_epoch_count(epoch):
        """Jalankan `epoch` epoch-diajar asosiasi A->B.

        Returns (w_ab, w_aa, respons A-ke-B %, respons B-ke-B %).
        Respons diukur sebagai spike PERTAMA per trial, bukan rate
        akumulatif — rate bisa tinggi hanya karena neuron tereksitasi
        berulang, yang bukan bukti asosiasi.
        """
        cfg = STDPConfig(eta_plus=2e-2, eta_minus=1e-4, w_max=_W_MAX)
        w = torch.full((_N, _N), _W_INIT)
        tp, tq = _make_trace(_N), _make_trace(_N)
        for _ in range(max(epoch, 1)):
            for t in range(_TRIAL):
                zp, zq = torch.zeros(_N), torch.zeros(_N)
                if t == 0:
                    zp[0] = 1.0
                if t == _DELTA:
                    zq[1] = 1.0
                w, tp, tq = stdp_step(zp, zq, w, tp, tq, cfg)

        neuron = _LIF(_N, tau_mem_ms=20.0, v_th=1.0, v_reset=0.0)
        res = []
        for stim, out_idx in ((0, 1), (1, 1)):
            hits = 0
            for _ in range(30):
                st = neuron.init_state()
                for t in range(80):
                    cur = torch.zeros(_N)
                    if t < 30:
                        cur[stim] = 1.0
                    z, st = neuron.step(cur @ w, st)
                    if float(z[out_idx]) > 0.0:
                        hits += 1
                        break
            res.append(100.0 * hits / 30)
        return float(w[0, 1]), float(w[0, 0]), res[0], res[1]

    grid = [5, 10, 20, 40, 60, 80, 120, 160, 200, 250, 300]
    rows = [_assoc_epoch_count(e) for e in grid]
    written.append(plot_association_learning(
        grid,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        out_path=out / "07_association_learning.png",
    ))

    for p in written:
        print(f"wrote {p}")
    print(
        f"\nGATE: drive tanpa homeostasis {plain_hist[-1]:.1f} "
        f"(plafon {n_in * n_out:.0f}) vs dengan homeostasis {homeo_hist[-1]:.1f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `readout` punya parser sendiri. Delegasi di depan supaya opsi
    # seperti --weights (nargs="+") tidak bertabrakan dengan parser
    # utama.
    if argv and argv[0] == "readout":
        from rak.readout.cli import main as readout_main

        return readout_main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="rak", description="RAK — pure spiking architecture"
    )
    parser.add_argument(
        "--version", action="version", version=f"RAK {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="versi + backend").set_defaults(
        fn=_cmd_info
    )
    sub.add_parser(
        "selfcheck", help="smoke test neuron + layer"
    ).set_defaults(fn=_cmd_selfcheck)

    b = sub.add_parser("bench", help="benchmark SpikeLayer throughput")
    b.add_argument("--n-in", type=int, default=784)
    b.add_argument("--n-out", type=int, default=100)
    b.add_argument("--steps", type=int, default=350)
    b.add_argument("--batch", type=int, default=1)
    b.add_argument(
        "--input",
        choices=("mnist-like", "uniform"),
        default="mnist-like",
        help="mnist-like: statistik realistis (default). uniform: spike acak.",
    )
    b.add_argument("--rate", type=float, default=0.05, help="untuk --input uniform")
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--seed", type=int, default=1)
    b.set_defaults(fn=_cmd_bench)

    v = sub.add_parser("viz", help="plot V(t), spike train, perbandingan")
    v.add_argument("--out", default="experiments/exp_001_neuron/figures")
    v.add_argument("--steps", type=int, default=600)
    v.add_argument("--current", type=float, default=2.0)
    v.add_argument("--tau-mem", type=float, default=20.0)
    v.add_argument("--tau-adapt", type=float, default=500.0)
    v.add_argument("--n-in", type=int, default=784)
    v.add_argument("--n-out", type=int, default=400)
    v.add_argument("--seed", type=int, default=1)
    v.set_defaults(fn=_cmd_viz)

    p2 = sub.add_parser(
        "viz-phase2",
        help="figure learning rules + homeostasis (Phase 2)",
    )
    p2.add_argument(
        "--out", default="experiments/exp_002_learning/figures"
    )
    p2.add_argument("--seed", type=int, default=1)
    p2.add_argument("--n-in", type=int, default=64)
    p2.add_argument("--n-out", type=int, default=32)
    p2.add_argument("--epochs", type=int, default=60)
    p2.add_argument("--steps", type=int, default=20)
    p2.set_defaults(fn=_cmd_viz_phase2)

    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        args.fn = _cmd_info
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
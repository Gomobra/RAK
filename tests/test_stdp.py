"""T2.2 — Trace-based STDP tests.

Fokus test di sini adalah SATU hal yang paling rawan di STDP: **urutan
waktu**. Kalau trace di-bump sebelum weight update, spike di timestep
yang sama akan saling "melihat" dan kredit yang salah akan diberikan.
Test-test di bawah mengunci urutan itu.

Test window analitik juga membandingkan implementasi terhadap rumus
eksponensial langsung, jadi bug numerik ketahuan tanpa perlu baca kode.
"""

import pytest
import torch

from rak.learning.stdp import (
    ALGORITHMS,
    STDPConfig,
    A_minus,
    A_plus,
    make_trace,
    stdp_step,
    stdp_window,
)


N_IN, N_OUT = 3, 2


def _w(value=0.5):
    return torch.full((N_IN, N_OUT), value)


def _step(z_pre, z_post, w=None, tp=None, tq=None, cfg=None):
    w = _w() if w is None else w
    cfg = cfg or STDPConfig()
    tp = make_trace(N_IN) if tp is None else tp
    tq = make_trace(N_OUT) if tq is None else tq
    return stdp_step(
        torch.as_tensor(z_pre, dtype=torch.float32),
        torch.as_tensor(z_post, dtype=torch.float32),
        w, tp, tq, cfg,
    )


# ------------------------------------------------------------- config
def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="algorithm"):
        STDPConfig(algorithm="magic")


def test_rejects_bad_tau():
    with pytest.raises(ValueError, match="tau"):
        STDPConfig(tau_pre_ms=0.0)


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="w_min"):
        STDPConfig(w_min=1.0, w_max=0.0)


def test_gamma_is_exp_decay():
    cfg = STDPConfig(tau_pre_ms=50.0, dt_ms=1.0)
    assert cfg.gamma_pre == pytest.approx(0.9802, abs=1e-4)


# ------------------------------------------- ARAH (paling kritis)
def test_pre_then_post_potentiates():
    """Spike pre di t-1, spike post di t -> LTP (W naik)."""
    w = _w(0.5)
    # t-1: hanya pre neuron 0 yang fire
    _, tp, tq = _step([1.0, 0.0, 0.0], [0.0, 0.0], w=w)
    # t: hanya post neuron 0 yang fire
    w2, _, _ = _step([0.0, 0.0, 0.0], [1.0, 0.0], w=w, tp=tp, tq=tq)
    assert float(w2[0, 0]) > 0.5, "pre sebelum post harus LTP"


def test_post_then_pre_depresses():
    """Spike post di t-1, spike pre di t -> LTD (W turun)."""
    w = _w(0.5)
    _, tp, tq = _step([0.0, 0.0, 0.0], [1.0, 0.0], w=w)
    w2, _, _ = _step([1.0, 0.0, 0.0], [0.0, 0.0], w=w, tp=tp, tq=tq)
    assert float(w2[0, 0]) < 0.5, "post sebelum pre harus LTD"


def test_simultaneous_spike_is_noop():
    """Pre DAN post di timestep sama, tanpa histori -> dW = 0.

    Tidak ada urutan waktu, jadi tidak ada kredit yang bisa diberikan.
    Ini yang terjadi kalau trace di-bump SEBELUM weight update.
    """
    w2, _, _ = _step([1.0, 0.0, 0.0], [1.0, 0.0])
    assert float((w2 - 0.5).abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_update_is_causal():
    """Spike post tidak boleh masuk ke trace_pre (dan sebaliknya).

    trace_pre hanya boleh meluruh; kalau spike post ikut menambahnya,
    itu berarti pre "melihat" spike post yang belum terjadi.
    """
    cfg = STDPConfig(tau_pre_ms=100.0)
    w = _w()
    _, tp, tq = _step([1.0, 0.0, 0.0], [0.0, 0.0], w=w, cfg=cfg)
    _, tp2, tq2 = _step([0.0, 0.0, 0.0], [1.0, 0.0], w=w, tp=tp, tq=tq, cfg=cfg)
    # hanya decay
    assert float(tp2[0]) == pytest.approx(float(tp[0]) * cfg.gamma_pre, rel=1e-5)
    # pre neuron lain tetap nol
    assert float(tp2[1]) == pytest.approx(0.0, abs=1e-9)
    # trace_post justru bertambah karena post neuron 0 spike
    assert float(tq2[0]) == pytest.approx(1.0, rel=1e-5)


def test_long_delay_negligible():
    """Jeda jauh -> trace meluruh -> dW mendekati 0."""
    cfg = STDPConfig(tau_pre_ms=5.0, tau_post_ms=5.0)
    w = _w()
    _, tp, tq = _step([1.0, 0.0, 0.0], [0.0, 0.0], w=w, cfg=cfg)
    for _ in range(200):
        _, tp, tq = _step([0.0, 0.0, 0.0], [0.0, 0.0], w=w, tp=tp, tq=tq, cfg=cfg)
    w2, _, _ = _step([0.0, 0.0, 0.0], [1.0, 0.0], w=w, tp=tp, tq=tq, cfg=cfg)
    assert abs(float(w2[0, 0]) - 0.5) < 1e-6, "trace harus habis, dW ~ 0"


# ------------------------------------------------------ trace decay
def test_trace_decays_exponentially():
    cfg = STDPConfig(tau_pre_ms=10.0, dt_ms=1.0)
    _, tp, _ = _step([1.0, 0.0, 0.0], [0.0, 0.0], cfg=cfg)
    after_1 = float(tp[0])
    for _ in range(5):
        _, tp, _ = _step([0.0, 0.0, 0.0], [0.0, 0.0], tp=tp, cfg=cfg)
    # 1 bump + 5 decay
    expected = cfg.gamma_pre ** 5
    assert after_1 == pytest.approx(1.0, rel=1e-6)
    assert float(tp[0]) == pytest.approx(expected, rel=1e-4)


def test_trace_only_spikes_where_spike_occurred():
    _, tp, tq = _step([1.0, 0.0, 0.0], [0.0, 1.0])
    assert float(tp[0]) == pytest.approx(1.0)   # pre neuron 0
    assert float(tp[1]) == pytest.approx(0.0)   # pre neuron 1 tidak spike
    assert float(tq[1]) == pytest.approx(1.0)   # post neuron 1
    assert float(tq[0]) == pytest.approx(0.0)   # post neuron 0 tidak spike


# ------------------------------------------------------ hardbound
@pytest.mark.parametrize("algo", ALGORITHMS)
def test_weights_stay_bounded(algo):
    """Semua varian harus menjaga w di [w_min, w_max]."""
    cfg = STDPConfig(algorithm=algo, eta_plus=0.1, eta_minus=0.1)
    w = torch.full((N_IN, N_OUT), 0.5)
    tp = make_trace(N_IN)
    tq = make_trace(N_OUT)
    for i in range(3000):
        zp = [1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 0.0, 0.0]
        zq = [0.0, 0.0] if i % 2 == 0 else [1.0, 0.0]
        w, tp, tq = stdp_step(
            torch.tensor(zp), torch.tensor(zq), w, tp, tq, cfg
        )
        assert float(w.min()) >= cfg.w_min - 1e-7
        assert float(w.max()) <= cfg.w_max + 1e-7
    assert torch.isfinite(w).all()


@pytest.mark.parametrize("algo", ["additive", "additive_step"])
def test_hardbound_false_allows_overshoot(algo):
    """Varian additive bisa melewati batas kalau clipping dimatikan."""
    cfg = STDPConfig(algorithm=algo, eta_plus=0.5, hardbound=False)
    w = torch.full((N_IN, N_OUT), 0.99)
    tp = torch.ones(N_IN)
    tq = make_trace(N_OUT)
    w2, _, _ = stdp_step(
        torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0]),
        w, tp, tq, cfg,
    )
    assert float(w2.max()) > cfg.w_max


@pytest.mark.parametrize(
    "algo", ["multiplicative_pow", "multiplicative_relu"]
)
def test_multiplicative_stays_bounded_without_clipping(algo):
    """Varian multiplicative menjaga w TERBATAS tanpa perlu clipping.

    Ini klaim yang benar (dan alasan varian ini ada): `A_plus` meredup menuju 0
    saat `w -> w_max`, jadi pertumbuhan melambat secara quadratik alih-alih
    tetap linier seperti additive.

    Yang diklaim di sini **bukan** "tidak pernah melewati w_max" — satu
    langkah dengan eta besar masih bisa melewati. Yang dijamin: w
    BEROSILASI di sekitar w_max, bukan meledak. Trace yang jenuh membuat
    langkah update ikut membesar, jadi konvergensinya monotonic tidak.
    """
    cfg = STDPConfig(algorithm=algo, eta_plus=0.5, hardbound=False,
                     **_mu_for(algo))
    w = torch.full((N_IN, N_OUT), 0.9)
    tp = torch.ones(N_IN)
    tq = make_trace(N_OUT)
    for _ in range(200):
        w, tp, tq = stdp_step(
            torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0]),
            w, tp, tq, cfg,
        )
    assert torch.isfinite(w).all()
    assert float(w.max()) < cfg.w_max + 0.3, (
        f"w_max={cfg.w_max}, dapat max={float(w.max()):.4f} — "
        "harusnya berosilasi di sekitar w_max, bukan meledak"
    )
    assert float(w.mean()) > cfg.w_max - 0.3


def test_additive_diverges_where_multiplicative_stays_bounded():
    """Kontras langsung: pola LTP-murni, additive meledak, multiplicative tidak.

    Pola spike: pre di t=0, post di t=1, lalu diam sampai t=20, ulang.
    Selalu ada lag yang pendek (1 timestep), jadi yang terkumpul HANYA LTP.
    Ini skenario yang harus dipegang rem.

    Catatan penting: kalau pre dan post fire DENGAN FREKUENSI SAMA, LTP dan
    LTD saling menyeimbangkan dan w mencapai fixed point — bukan divergensi.
    Divergensi butuh asimetri, dan pola ini mengilustrasikannya.
    """
    add_cfg = STDPConfig(algorithm="additive", eta_plus=1e-3, hardbound=False)
    mul_cfg = STDPConfig(algorithm="multiplicative_relu", eta_plus=1e-3,
                         hardbound=False)
    zp_on = torch.tensor([1.0, 0.0, 0.0])
    zq_on = torch.tensor([1.0, 0.0])
    zero_p = torch.zeros(N_IN)
    zero_q = torch.zeros(N_OUT)
    period = 20

    results = {}
    for name, cfg in (("additive", add_cfg), ("multiplicative", mul_cfg)):
        w = torch.full((N_IN, N_OUT), 0.1)
        tp = torch.ones(N_IN)
        tq = make_trace(N_OUT)
        marks = []
        for t in range(8_000):
            phase = t % period
            if phase == 0:
                z_pre, z_post = zp_on, zero_q
            elif phase == 1:
                z_pre, z_post = zero_p, zq_on
            else:
                z_pre, z_post = zero_p, zero_q
            w, tp, tq = stdp_step(z_pre, z_post, w, tp, tq, cfg)
            if t in (3_999, 7_999):
                marks.append(float(w[0, 0]))
        results[name] = (marks, float(w[0, 0]))

    # Additive tumbuh LINIER: putaran kedua menambah sama besarnya dengan
    # putaran pertama -> tidak pernah berhenti.
    first, second = results["additive"][0]
    assert (second - first) == pytest.approx(first - 0.1, rel=0.25), (
        f"additive harus tumbuh tanpa henti: {0.1:.3f} -> "
        f"{first:.3f} -> {second:.3f}"
    )
    # Multiplicative melambat dan mendekati w_max
    mul_final = results["multiplicative"][1]
    assert mul_final < results["additive"][1], results
    assert mul_final < 1.5, results


def test_stdp_reaches_fixed_point_when_rates_are_balanced():
    """Pre dan post berfrekuensi sama -> LTP dan LTD menyeimbangkan.

    Ini lain dari divergensi, dan sering disalahpahami. Weight
    tidak meledak justru karena LTP dan LTD bisa saling meniadakan —
    di sinilah stabilizing role STDP muncul (bukan hanya clamping).
    """
    cfg = STDPConfig(algorithm="additive", eta_plus=1e-3, hardbound=False)
    w = torch.full((N_IN, N_OUT), 0.1)
    tp = torch.ones(N_IN)
    tq = make_trace(N_OUT)
    zp_on = torch.tensor([1.0, 0.0, 0.0])
    zq_on = torch.tensor([1.0, 0.0])
    zero_p = torch.zeros(N_IN)
    zero_q = torch.zeros(N_OUT)

    history = []
    for t in range(6000):
        z_pre, z_post = (zp_on, zero_q) if t % 2 == 0 else (zero_p, zq_on)
        w, tp, tq = stdp_step(z_pre, z_post, w, tp, tq, cfg)
        history.append(float(w[0, 0]))

    late_spread = max(history[-100:]) - min(history[-100:])
    assert torch.isfinite(w).all()
    assert late_spread < 0.05, (
        f"harus mencapai fixed point, tapi masih naik-turun {late_spread:.4f}"
    )


# --------------------------------------------------- A_plus/A_minus
def test_additive_is_independent_of_weight():
    w = torch.rand(4, 4) * 0.8 + 0.1
    cfg = STDPConfig(algorithm="additive", eta_plus=1e-3)
    assert torch.allclose(A_plus(w, cfg), torch.full_like(w, 1e-3))


def test_additive_step_blocks_at_bounds():
    cfg = STDPConfig(algorithm="additive_step", eta_plus=1e-3, eta_minus=1e-3)
    w = torch.tensor([[0.0, 0.5, 1.0]])
    ap = A_plus(w, cfg)
    am = A_minus(w, cfg)
    assert float(ap[0, 0]) == pytest.approx(1e-3)  # 0 < w_max
    assert float(ap[0, 1]) == pytest.approx(1e-3)
    assert float(ap[0, 2]) == pytest.approx(0.0)   # w == w_max, blok
    assert float(am[0, 0]) == pytest.approx(0.0)   # w == w_min, blok
    assert float(am[0, 1]) == pytest.approx(1e-3)
    assert float(am[0, 2]) == pytest.approx(1e-3)


def test_multiplicative_pow_uses_power():
    cfg = STDPConfig(algorithm="multiplicative_pow", eta_plus=1e-3, mu=0.5,
                     w_min=0.0, w_max=1.0)
    w = torch.tensor([[0.25, 0.75]])
    ap = A_plus(w, cfg)
    assert float(ap[0, 0]) == pytest.approx(1e-3 * 0.75 ** 0.5)
    assert float(ap[0, 1]) == pytest.approx(1e-3 * 0.25 ** 0.5)


def test_multiplicative_suppresses_near_bound():
    """Varian multiplicative harus menahan W dekat batas lebih ketat."""
    pre_cfg = STDPConfig(algorithm="additive", eta_plus=0.2)
    mul_cfg = STDPConfig(algorithm="multiplicative_relu", eta_plus=0.2)
    w = torch.full((N_IN, N_OUT), 0.99)
    tp = torch.ones(N_IN)
    tq = make_trace(N_OUT)
    zq = torch.tensor([1.0, 0.0])
    w_add, _, _ = stdp_step(
        torch.tensor([1.0, 0.0, 0.0]), zq, w, tp, tq, pre_cfg
    )
    w_mul, _, _ = stdp_step(
        torch.tensor([1.0, 0.0, 0.0]), zq, w, tp, tq, mul_cfg
    )
    assert float(w_mul.max()) < float(w_add.max()), (
        "multiplicative harus lebih lemah dekat batas"
    )


def test_multiplicative_relu_is_zero_at_bound():
    cfg = STDPConfig(algorithm="multiplicative_relu")
    w = torch.tensor([[cfg.w_max, cfg.w_min, 0.5]])
    assert float(A_plus(w, cfg)[0, 0]) == pytest.approx(0.0)
    assert float(A_minus(w, cfg)[0, 1]) == pytest.approx(0.0)


def test_multiplicative_pow_with_mu_zero_matches_additive():
    """mu = 0 -> (w_max - w)^0 = 1 -> sama dengan additive."""
    mul = STDPConfig(algorithm="multiplicative_pow", mu=0.0, eta_plus=1e-3)
    add = STDPConfig(algorithm="additive", eta_plus=1e-3)
    w = torch.rand(4, 4) * 0.9 + 0.05
    assert torch.allclose(A_plus(w, mul), A_plus(w, add), atol=1e-7)


# ------------------------------------------------ STDP window analitik
def _empirical_dw(dt: float, cfg: STDPConfig) -> float:
    """Hitung dW lewat `stdp_step` untuk dua spike tunggal berjarak dt.

    Spike pertama terjadi di timestep 0, spike kedua di timestep |dt|.
    Tepat SATU spike per neuron — kalau spike diulang tiap timestep,
    trace_post akan menjumlah banyak suku dan tidak lagi cocok rumus
    eksponensial tunggal.
    """
    n_in = n_out = 1
    w = torch.full((n_in, n_out), 0.5)
    tp = make_trace(n_in)
    tq = make_trace(n_out)
    zero = torch.zeros(1)
    one = torch.tensor([1.0])

    for t in range(abs(int(round(dt)))):
        if dt < 0:
            # post lebih awal: spike post di t=0, lalu pre di t=|dt|
            z_pre = one if t == 0 else zero
            z_post = one if t == 0 else zero
        else:
            # pre lebih awal: spike pre di t=0, lalu post di t=dt
            z_pre = one if t == 0 else zero
            z_post = one if t == 0 else zero
        w, tp, tq = stdp_step(z_pre, z_post, w, tp, tq, cfg)

    # timestep terakhir: spike kedua
    if dt < 0:
        z_pre, z_post = one, zero
    else:
        z_pre, z_post = zero, one
    w, _, _ = stdp_step(z_pre, z_post, w, tp, tq, cfg)
    return float(w[0, 0] - 0.5)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_stdp_window_matches_empirical(algo):
    """Window dari `stdp_step` harus cocok rumus eksponensial analitik."""
    cfg = STDPConfig(algorithm=algo, eta_plus=1e-2, eta_minus=1e-2,
                     tau_pre_ms=20.0, tau_post_ms=20.0, **_mu_for(algo))
    dts, analytic = stdp_window(1.0, cfg, w_ref=0.5, max_ms=60.0)

    worst = 0.0
    worst_dt = 0.0
    for dt, want in zip(dts.tolist(), analytic.tolist()):
        if dt == 0:
            continue
        got = _empirical_dw(dt, cfg)
        err = abs(got - want)
        if err > worst:
            worst, worst_dt = err, dt
    assert worst < 2e-4, (
        f"window menyimpang {worst:.2e} dari analitik (paling buruk di dt={worst_dt})"
    )


def test_window_is_zero_at_zero_lag():
    cfg = STDPConfig()
    dts, win = stdp_window(1.0, cfg)
    idx = int((dts.abs()).argmin())
    assert abs(float(win[idx])) < 1e-9


def test_window_is_bipolar():
    """Harus ada daerah LTP (positif) dan LTD (negatif)."""
    cfg = STDPConfig()
    _, win = stdp_window(1.0, cfg, max_ms=50.0)
    assert float(win.max()) > 0
    assert float(win.min()) < 0


def test_window_asymmetric_when_tau_differ():
    """tau_pre != tau_post -> window tidak simetris."""
    cfg = STDPConfig(tau_pre_ms=50.0, tau_post_ms=5.0)
    dts, win = stdp_window(1.0, cfg, max_ms=100.0)
    pos = float(win[dts > 0].abs().max())
    neg = float(win[dts < 0].abs().max())
    assert not (pos == pytest.approx(neg, rel=1e-3)), "window harus asimetris"


# ------------------------------------------------------------- shape
def test_update_preserves_weight_shape():
    n_in, n_out = 7, 5
    w = torch.rand(n_in, n_out)
    zp = torch.zeros(n_in)
    zp[0] = 1.0
    zq = torch.zeros(n_out)
    w, tp, tq = stdp_step(zp, zq, w, make_trace(n_in), make_trace(n_out), STDPConfig())
    zq[0] = 1.0
    w2, tp2, tq2 = stdp_step(zp, zq, w, tp, tq, STDPConfig())
    assert w2.shape == (n_in, n_out)
    assert tp2.shape == (n_in,) and tq2.shape == (n_out,)


def test_trace_shape_validation_is_strict():
    """Trace per-synapse harus DITOLAK dengan pesan yang jelas.

    Trace per-synapse boros (n_in x n_out memori) dan tidak dipakai rumus
    STDP mana pun — jadi accepting-nya diam-diam hanya menyembunyikan bug.
    """
    with pytest.raises(ValueError, match="per-neuron"):
        stdp_step(
            torch.ones(N_IN), torch.ones(N_OUT),
            _w(), torch.zeros(N_IN, N_OUT), torch.zeros(N_OUT),
            STDPConfig(),
        )


def test_does_not_mutate_inputs():
    w = _w()
    tp = make_trace(N_IN)
    tq = make_trace(N_OUT)
    w0, tp0, tq0 = w.clone(), tp.clone(), tq.clone()
    stdp_step(
        torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0]),
        w, tp, tq, STDPConfig(),
    )
    assert torch.equal(w, w0) and torch.equal(tp, tp0) and torch.equal(tq, tq0)


def test_only_affected_synapse_changes():
    """Synapse (pre 2, post 1) tidak boleh tersentuh."""
    w = torch.full((N_IN, N_OUT), 0.5)
    _, tp, tq = _step([0.0, 0.0, 1.0], [0.0, 0.0], w=w)
    w2, _, _ = _step([0.0, 0.0, 0.0], [0.0, 1.0], w=w, tp=tp, tq=tq)
    assert float(w2[2, 1]) != 0.5, "synapse yang bersangkutan harus berubah"
    others = w2.clone()
    others[2, 1] = 0.5
    assert torch.allclose(others, torch.full((N_IN, N_OUT), 0.5)), (
        f"synapse lain tidak boleh berubah:\n{w2}"
    )


def _mu_for(algo: str) -> dict:
    """`mu` hanya dibaca `multiplicative_pow`.

    Test yang diparametrisasi over `ALGORITHMS` tidak boleh menyetel
    `mu` secara buta — `STDPConfig` akan menolaknya dengan
    ValueError, karena menyetel mu di varian yang tidak membacanya
    adalah no-op senyap (persis bug yang pernah hampir masuk figure
    03). Helper ini menjaga pemanggilan tetap eksplisit per varian.
    """
    return {"mu": 1.0} if algo == "multiplicative_pow" else {}

# =====================================================================
# multiplicative: apakah `mu` benar-benar mengubah bentuk kurva?
# =====================================================================
# Test ini ada karena ketahuan bug di figure 03: CLI mengirim
# `STDPConfig(mu=1.0)` TANPA `algorithm=`, jadi default `"additive"`
# yang dipakai dan `mu` dibuang diam-diam. Tiga kurva w_ref jadi
# identik persis, dan figure ituriersebut "memperlihatkan perbedaan
# varian multiplicative" padahal tidak.

def test_additive_amplitude_does_not_depend_on_weight():
    """Kontrol: `additive` memang TIDAK boleh bergantung pada w."""
    w = torch.tensor([[0.1], [0.5], [0.9]])
    cfg = STDPConfig(algorithm="additive")
    assert torch.allclose(A_plus(w, cfg), torch.full((3, 1), cfg.eta_plus))
    assert torch.allclose(A_minus(w, cfg), torch.full((3, 1), cfg.eta_minus))


@pytest.mark.parametrize("algo", ["multiplicative_pow", "multiplicative_relu"])
def test_multiplicative_amplitude_depends_on_weight(algo):
    """Inilah yang harusnya dicecek figure 03 dan TIDAK dicek sebelumnya.

    Amplitudo LTP harus MENURUN seiring w mendekati w_max, dan amplitudo
    LTD harus MENINGKAT. Kalau keduanya datar, `mu`/w tidak ada
    pengaruhnya sama sekali dan varian ini identik dengan additive.
    """
    w = torch.tensor([[0.1], [0.5], [0.9]])
    cfg = STDPConfig(algorithm=algo, **_mu_for(algo))

    ap = A_plus(w, cfg).flatten()
    am = A_minus(w, cfg).flatten()

    assert float(ap[0]) > float(ap[1]) > float(ap[2]), (
        f"A_plus harus menurun menuju w_max, dapat {ap.tolist()}"
    )
    assert float(am[0]) < float(am[1]) < float(am[2]), (
        f"A_minus harus naik menjauhi w_min, dapat {am.tolist()}"
    )


def test_mu_zero_makes_pow_identical_to_additive():
    """mu = 0 → (w_max - w)^0 = 1 → ekuivalen additive. Sudah didokumentasikan."""
    w = torch.tensor([[0.1], [0.5], [0.9]])
    pow0 = STDPConfig(algorithm="multiplicative_pow", mu=0.0)
    add = STDPConfig(algorithm="additive")
    assert torch.allclose(A_plus(w, pow0), A_plus(w, add))
    assert torch.allclose(A_minus(w, pow0), A_minus(w, add))


def test_mu_one_equals_relu_variant():
    """mu = 1 persis sama dengan multiplicative_relu."""
    w = torch.tensor([[0.1], [0.5], [0.9]])
    assert torch.allclose(
        A_plus(w, STDPConfig(algorithm="multiplicative_pow", mu=1.0)),
        A_plus(w, STDPConfig(algorithm="multiplicative_relu")),
    )


def test_mu_above_one_flattens_growth():
    """mu > 1 membuat pertumbuhan melambat LEBIH TAJAM dekat w_max.

    Ini alasan varian ini ada (docstring stdp.py baris 46). Kalau tidak
    ada, multiplicative_pow tidak punya alasanDigestif dibanding relu.
    """
    w = torch.tensor([[0.1], [0.5], [0.9]])
    a1 = A_plus(w, STDPConfig(algorithm="multiplicative_pow", mu=1.0))
    a2 = A_plus(w, STDPConfig(algorithm="multiplicative_pow", mu=3.0))
    # w = 0.9 → w_max - w = 0.1 → 0.1^3 = 0.001 << 0.1^1
    assert float(a2[2]) < float(a1[2])
    # w = 0.1 → w_max - w = 0.9 → 0.9^3 = 0.729 < 0.9
    assert float(a2[0]) < float(a1[0])


@pytest.mark.parametrize("algo", ["multiplicative_pow", "multiplicative_relu"])
def test_window_peak_scales_with_w_ref(algo):
    """Figure 03 harus menunjukkan PEAK yang berbeda per w_ref.

    Ini test yang langsung menangkap bug CLI: kalau `algorithm` tidak
    ikut dikirim, ketiga kurvanya peak-nya identik dan figure tidak
    membuktikan apa pun.
    """
    peaks = []
    for wr in (0.1, 0.5, 0.9):
        cfg = STDPConfig(algorithm=algo, **_mu_for(algo))
        _, win = stdp_window(1.0, cfg, w_ref=wr)
        peaks.append(float(win.max()))

    assert peaks[0] > peaks[1] > peaks[2], (
        f"peak LTP harus menurun seiring w, dapat {peaks}"
    )
    # dan harus benar-benar berbeda secara numerik, bukan bedanya
    # satu orde besaran cuma karena pembulatan float
    assert peaks[0] - peaks[2] > 1e-6


def test_window_trough_magnitude_increases_with_w_ref():
    """Sisi LTD:|w yang besar → depresi besar, bukan kecil."""
    troughs = []
    for wr in (0.1, 0.5, 0.9):
        cfg = STDPConfig(algorithm="multiplicative_pow", mu=1.0)
        _, win = stdp_window(1.0, cfg, w_ref=wr)
        troughs.append(abs(float(win.min())))

    assert troughs[0] < troughs[1] < troughs[2], troughs


def test_additive_variants_window_ignores_w_ref():
    """Kontrol untuk dua test di atas: additive HARUS datar.

    Diletakkan berdekatan dengan test multiplicative supaya kalau
    satu keluarga test gagal, yang lain langsung jadi kontrol.
    """
    cfg = STDPConfig(algorithm="additive")
    peaks = [float(stdp_window(1.0, cfg, w_ref=wr)[1].max())
             for wr in (0.1, 0.5, 0.9)]
    assert peaks[0] == peaks[1] == peaks[2]


def test_end_to_end_weight_depends_on_current_weight():
    """Uji tingkat `stdp_step`, bukan cuma A_plus/A_minus.

    Kalau ada yang salah shortlisted di implementasi `stdp_step` —
    misalnya A_plus dipanggil dengan w yang salah bentuk — test
    A_plus di atas tetap hijau tapi STDP-nya salah.
    """
    cfg = STDPConfig(algorithm="multiplicative_pow", mu=1.0)
    deltas = []
    for w_val in (0.1, 0.5, 0.9):
        w = torch.full((N_IN, N_OUT), w_val)
        tp = make_trace(N_IN)
        tp[0] = 1.0  # jejakkan spike pre di input 0
        tq = make_trace(N_OUT)
        z_post = torch.zeros(N_OUT)
        z_post[0] = 1.0
        w_new, _, _ = stdp_step(
            torch.zeros(N_IN), z_post, w, tp, tq, cfg
        )
        deltas.append(float(w_new[0, 0] - w[0, 0]))

    assert deltas[0] > deltas[1] > deltas[2], (
        f"ΔW LTP harus mengecil seiring w, dapat {deltas}"
    )

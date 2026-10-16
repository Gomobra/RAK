"""Edge case tests — guardrail numerik dan stabilitas.

Test di sini bukan "fitur", tapibenteng supaya regresi numerik
ketahuan sebelum jadi bug diam-diam. Semua bug yang ditemukan di
Phase 1 punya test di file ini.

Lihat docs/theory.md §1 dan §5 untuk penjelasan teoretisnya.
"""

import pytest
import torch

from rak.neuron import AdaptiveLIF, Izhikevich, LIF, SpikeLayer
from rak.neuron.lif import LIFConfig


# ---------------------------------------------------------------- LIF
def test_lif_rejects_dt_greater_than_tau():
    """dt > tau membuat alpha < 0: membran BEROSILASI, bukan membocor.

    Gejalanya: neuron fire di SETIAP timestep tanpa memedulikan
    besar kecilnya input. Tidak meledak — jadi tidak ketahuan kalau
    kita tidak memeriksa alpha-nya secara eksplisit.
    """
    with pytest.raises(ValueError, match="dt_ms"):
        LIF(1, tau_mem_ms=5.0, dt_ms=10.0)
    # batasnya inklusif: dt == tau masih boleh (alpha == 0, jadi v beku)
    assert LIF(1, tau_mem_ms=5.0, dt_ms=5.0).config.alpha == 0.0


def test_lif_negative_input_hyperpolarizes():
    """Input negatif harus menurunkan V, bukan membuat spike."""
    lif = LIF(1)
    sp, state, rec = lif.simulate(torch.full((500,), -5.0), record=True)
    assert float(sp.sum()) == 0.0
    assert float(rec[-1, 0]) < 0.0
    # v_ss = v_leak + I, jadi input -5 harus memberi v_ss = -5
    assert float(rec[-1, 0]) == pytest.approx(-5.0, abs=1e-3)


def test_lif_never_firing_neuron_stays_at_leak():
    """Neuron tanpa input harus konverge ke v_leak.

    Ini yang membedakan `v_leak` sebagai potensial istirahat dari sekadar
    offset. Kalau v_leak hanya offset, v akan konverge ke 0 dan test ini
    gagal — persis bug yang ketahuan di review Phase 1.
    """
    lif = LIF(1, v_leak=0.25)
    sp, state, rec = lif.simulate(torch.zeros(1000), record=True)
    assert float(sp.sum()) == 0.0
    assert float(rec[-1, 0]) == pytest.approx(0.25, abs=1e-6)


def test_lif_alpha_stays_in_unit_interval():
    """Invarian: 0 <= alpha <= 1 untuk semua config yang valid."""
    for tau in (0.5, 1.0, 5.0, 20.0, 1000.0):
        dt = min(1.0, tau)
        cfg = LIFConfig(tau_mem_ms=tau, dt_ms=dt)
        assert 0.0 <= cfg.alpha <= 1.0
        assert cfg.alpha + cfg.beta == pytest.approx(1.0)


def test_lif_steady_state_equals_leak_plus_current():
    """v_ss = v_leak + I, selama tegangan itu di BAWAH threshold.

    Ini yang membuat kalibrasi bobot SpikeLayer bisa dihitung analitik
    (dengan v_leak default 0, jadi v_ss = I).

    Formulasi steady-state hanya berlaku kalau neuron tidak pernah
    spike — makanya test ini memakai I < v_th.
    """
    for v_leak, cur in ((0.0, 0.5), (0.0, 0.9), (0.25, 0.3)):
        lif = LIF(1, tau_mem_ms=10.0, v_leak=v_leak)
        assert v_leak + cur < lif.config.v_th, "test precondition"
        _, _, rec = lif.simulate(
            torch.full((2000,), cur), record=True
        )
        assert float(rec[-1, 0]) == pytest.approx(v_leak + cur, rel=0.02)


def test_lif_spiking_neuron_saturates_below_threshold():
    """Kalau neuron spike terus, v tidak pernah mencapai v_ss.

    Rekamannya berfluktuasi di bawah threshold, bukan konverge ke I.
    Ini yang bikin SpikeLayer perlu k-WTA / lateral inhibition
    (Phase 2) — kalau tidak, semua neuron fire tanpa informasi pembeda.
    """
    lif = LIF(1, tau_mem_ms=10.0)
    sp, _, rec = lif.simulate(torch.full((2000,), 2.0), record=True)
    assert float(sp.sum()) > 0.0
    assert float(rec[-1, 0]) < lif.config.v_th


# ------------------------------------------------------------ Adaptive
def test_adaptive_negative_input_reduces_theta():
    """Input negatif tidak boleh menaikkan theta (tidak ada spike)."""
    alif = AdaptiveLIF(1)
    state = alif.init_state()
    state.theta.fill_(0.5)
    _, state = alif.step(torch.tensor([-10.0]), state)
    assert float(state.theta[0]) < 0.5


def test_adaptive_theta_bounded_by_repetition():
    """Theta tidak boleh tumbuh tanpa batas walau input tak terbatas.

    Kalau theta tembus v_th, neuron berhenti fire selamanya — itu
    "dead neuron" yang harus dicegah homeostasis Phase 2.
    """
    alif = AdaptiveLIF(1, tau_adapt_ms=100.0, beta_theta=0.1, tau_mem_ms=10.0)
    sp, state, _ = alif.simulate(torch.full((20_000,), 100.0))
    assert torch.isfinite(state.theta).all()
    assert float(state.theta[0]) < 1e3, "theta meledak"


# ----------------------------------------------------------- Izhikevich
def test_izhikevich_v_squared_never_overflows_with_reset():
    """v^2 bisa besar, tapi reset memaksa v ke c setiap spike.

    Karena itu tidak ada overflow yang bisa terjadi pada operasi normal.
    Yang TIDAK aman: konfigurasi patologis yang membuat neuron tidak
    pernah spike — di situ v bisa merambat tanpa batas.
    """
    z = Izhikevich(1)
    sp, _, (v, w) = z.simulate(torch.full((3000,), 1e9), record=True)
    assert torch.isfinite(v).all()
    assert torch.isfinite(w).all()
    assert float(v.max()) <= z.config.v_threshold


def test_izhikevich_extreme_threshold_still_runs():
    """v_threshold ekstrem harus tetap bisa disimulasikan tanpa crash.

    Guard dt di __post_init__ sengaja TIDAK bergantung pada
    v_threshold. Sebelumnya keduanya terikat, sehingga v_threshold besar
    membuat batas dt jadi 0.00 dan config yang sah ikut ditolak.
    """
    z = Izhikevich(1, v_threshold=1e30)
    sp, state, (v, w) = z.simulate(torch.full((500,), 10.0), record=True)
    assert torch.isfinite(v).all()
    assert torch.isfinite(w).all()


# --------------------------------------------------------- SpikeLayer
def test_layer_rejects_all_zero_weight():
    """Bobot nol harus tetap valid (neuron diam), bukan crash."""
    layer = SpikeLayer(n_in=10, n_out=5, seed=1)
    with torch.no_grad():
        layer.weight.zero_()
    x = torch.rand(100, 1, 10).lt(0.5).float()
    y, _, _ = layer.forward(x, None)
    assert float(y.sum()) == 0.0


def test_layer_state_survives_all_zero_batch():
    """Batch kosong harus ditangani, bukan IndexError."""
    layer = SpikeLayer(n_in=10, n_out=5, seed=1)
    state = layer.init_state(0)
    assert state["v"].shape == (0, 5)
    x = torch.zeros(10, 0, 10)
    y, new_state, _ = layer.forward(x, state)
    assert y.shape == (10, 0, 5)


def test_layer_handles_single_timestep():
    """T=1 harus jalan (degenerate tapi mungkin dipakai saat debug)."""
    layer = SpikeLayer(n_in=10, n_out=5, seed=1)
    x = torch.rand(1, 1, 10).lt(0.5).float()
    y, _, _ = layer.forward(x, None)
    assert y.shape == (1, 1, 5)


# --------------------------------------------------- reproducibility (P4)
def test_weight_init_is_reproducible():
    """Bobot HARUS di-seed — tanpa ini, semua eksperimen tidak repeatable."""
    a = SpikeLayer(n_in=64, n_out=32, seed=7)
    b = SpikeLayer(n_in=64, n_out=32, seed=7)
    c = SpikeLayer(n_in=64, n_out=32, seed=8)
    assert torch.equal(a.weight, b.weight), "seed sama harus hasil sama"
    assert not torch.equal(a.weight, c.weight), "seed beda harus hasil beda"


def test_weight_init_uses_torch_generator_not_global():
    """Seed harus lewat Generator eksplisit, bukan seed global torch.

    Kalau pakai torch.randn tanpa generator,Coupling dengan state
    global berarti SpikeLayer(seed=7) akan menghasilkan bobot berbeda
    tergantung call sebelumnya.
    """
    from rak.neuron.layer import SpikeLayer as SL

    torch.manual_seed(0)
    a = SL(n_in=32, n_out=16, seed=5)
    torch.manual_seed(999)  # ganggu global state
    b = SL(n_in=32, n_out=16, seed=5)
    assert torch.equal(a.weight, b.weight), (
        "SpikeLayer(seed=5) tidak boleh bergantung pada global torch seed"
    )


def test_layer_forward_is_deterministic_given_seed():
    """forward + encode dengan seed sama harus bit-identical."""
    a = SpikeLayer(n_in=64, n_out=32, seed=3)
    b = SpikeLayer(n_in=64, n_out=32, seed=3)
    x = a.encode(torch.rand(1, 64), n_steps=100, seed=3)
    y1, _, _ = a.forward(x, None)
    y2, _, _ = b.forward(x, None)
    assert torch.equal(y1, y2)
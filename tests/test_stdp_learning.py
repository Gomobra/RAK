"""GATE — STDP bisa belajar asosiasi dari nol.

Selama ini Phase 2 baru membuktikan dua hal:

  1. STDP menghasilkan window yang benar secara matematis
     (`test_stdp.py`)
  2. STDP + homeostasis tidak ledak (`test_homeostatic_gate.py`)

Keduanya soal MEKANISME. Tidak ada satu pun test yang menyatakan
"jaringan dengan STDP bisa berubah karena pengalamannya". Tanpa itu,
seluruh Phase 2 hanya membuktikan bahwa rumusnya benar — bukan bahwa
rumbunya berguna.

Task yang diuji: asosiasi A -> B dan B -> A pada jaringan 2x2.

Mengapa butuh `eta_plus >> eta_minus`: kalau learning rate LTP dan
LTD sama besar, STDP punya FIXED POINT (lihat ADR-0005) dan bobot
berhenti di sekitar 0.83. Dengan satu kanal input, `v_ss = w`, jadi
`w = 0.83` tidak pernah bisa menembus `v_th = 1.0` — neuron tidak
pernah fire dan asosiasi yang "terpelajar" tidak teramati. Ini bukan
batas artifact test: di jaringan sungguhan banyak kanal input yang
berkembang summon, jadi bobot per synapse tetap kecil.
"""

import pytest
import torch

from rak.learning.stdp import STDPConfig, make_trace, stdp_step
from rak.neuron import LIF

N = 2
#: Asimetri wajib — lihat module docstring.
ETA_PLUS, ETA_MINUS = 5e-2, 1e-4
#: `w` untuk neuron target harus > 1.0 supaya `v_ss = w` menembus
#: `v_th = 1.0`. Default RAK `w_max = 1.0` tidak cukup untuk 1 kanal.
W_MAX = 3.0
W_INIT = 0.05
DELTA_MS = 5  # post fire 5 ms setelah pre — di dalam window LTP
TRIAL_MS = 40  # jeda antar trial


# ------------------------------------------------------------ helper
def _teach(pairs, *, epochs=200, delta=DELTA_MS, causal=True,
           eta_plus=ETA_PLUS, eta_minus=ETA_MINUS, reset_traces=False):
    """Ajarkan `(pre_idx, post_idx)` berulang.

    Args:
        pairs: daftar pasangan neuron yang dipresentasikan.
        causal: kalau True, pre fire LEBIH DAHULU (LTP). Kalau False
            dibalik — post lebih dulu (LTD).
        reset_traces: kalau True, trace di-nolkan di awal tiap trial.

    **Kenapa `reset_traces` itu parameter, bukan detail teknis:**

    Percobaan pertama — uji "anti-causal tidak mengajar" — GAGAL:
    bobot naik ke 2.03, bukan turun. Setelah ditelusuri, penyebabnya
    bukan bug STDP tapi sifat nyata trace:

        trial sebelumnya menyisakan `trace_pre[0] = 0.142`
        trial berikutnya, t=0, z_post[1] menyala
        -> LTP = 5e-2 * 0.142 = 7.1e-3
        sementara LTD di t=5 cuma 1e-4 * 0.78 = 7.8e-5

    Jadi LTP residual **9x lebih besar** dari LTD yang seharusnya
    menetralkannya. Trace yang bocor lintas trial sama sekali tidak melihat
    causality — dia cuma tahu "ada aktivitas pre yang baru saja
    terjadi".

    Ini perilaku yang benar secara biologis (justru itulah cara STDP
    menangkap pola temporal), tapi kalau dipakai sebagai kontrol
    causality, hasilnya kabur. Karena itu kontrol anti-causal di
    bawah memakai `reset_traces=True` supaya mengisolasi causality
    DALAM satu trial.
    """
    cfg = STDPConfig(
        eta_plus=eta_plus, eta_minus=eta_minus, w_max=W_MAX,
    )
    w = torch.full((N, N), W_INIT)
    trace_pre, trace_post = make_trace(N), make_trace(N)

    for _ in range(epochs):
        for pre_i, post_i in pairs:
            # Setiap pasangan = trial sendiri dengan jeda, supaya
            # trace pasangan sebelumnya tidak bocor ke pasangan ini.
            # Kalau semua dinyalakan bersamaan, z_pre dan z_post
            # sama-sama penuh -> semua synapse kena LTP dan
            # selektivitas hilang.
            if reset_traces:
                trace_pre = make_trace(N)
                trace_post = make_trace(N)
            for t in range(TRIAL_MS):
                z_pre = torch.zeros(N)
                z_post = torch.zeros(N)
                pre_at, post_at = (0, delta) if causal else (delta, 0)
                if t == pre_at:
                    z_pre[pre_i] = 1.0
                if t == post_at:
                    z_post[post_i] = 1.0
                w, trace_pre, trace_post = stdp_step(
                    z_pre, z_post, w, trace_pre, trace_post, cfg
                )
    return w


def _respond(w, stim, out_idx, *, trials=60, hold=30):
    """Berapa persen trial di mana input `stim` memicu `out_idx`.

    Neuron di-reset tiap trial, jadi yang diukur adalah spike PERTAMA —
    bukan rate akumulatif. Rate bisa tinggi hanya karena neuron
    tereksitasi berulang, yang bukan bukti asosiasi.
    """
    neuron = LIF(N, tau_mem_ms=20.0, v_th=1.0, v_reset=0.0)
    hits = 0
    for _ in range(trials):
        state = neuron.init_state()
        fired = False
        for t in range(80):
            current = torch.zeros(N)
            if t < hold:  # input ditahan beberapa ms
                current[stim] = 1.0
            spikes, state = neuron.step(current @ w, state)
            if float(spikes[out_idx]) > 0.0:
                fired = True
                break
        hits += fired
    return hits


# ============================================================ GATE
def test_association_is_learned_from_scratch():
    """GATE: A harus bisa memicu B setelah diajar, dan TIDAK sebelum.

    Ini klaim yang belum pernah diuji di Phase 2 — jaringan berubah
    karena pengalaman.
    """
    before = torch.full((N, N), W_INIT)
    after = _teach([(0, 1)])

    assert _respond(before, 0, 1) == 0, (
        "sebelum diajar TIDAK boleh ada respons — kalau sudah ada, "
        "test ini tidak mengukur apa pun"
    )
    assert _respond(after, 0, 1) == 60, "A harus memicu B di semua trial"


def test_only_the_taught_synapse_grows():
    """STDP harus mengubah HANYA synapse yang-diajarkan.

    Kalau semua bobot naik bersama, itu bukan credit assignment —
    itu cuma drift global.
    """
    w = _teach([(0, 1)])
    assert float(w[0, 1]) > W_INIT, "synapse yang diajarkan harus tumbuh"
    untouched = [
        (i, j) for i in range(N) for j in range(N) if (i, j) != (0, 1)
    ]
    for i, j in untouched:
        assert float(w[i, j]) == pytest.approx(W_INIT, abs=1e-5), (
            f"synapse ({i},{j}) tidak boleh berubah dari {W_INIT} "
            f"jadi {float(w[i, j]):.4f}"
        )


def test_association_is_selective():
    """Setelah A->BFormation, input B tidak boleh memicu B."""
    w = _teach([(0, 1)])
    assert _respond(w, 1, 1) == 0, (
        "input B tidak pernah diajarkan, jadi tidak boleh memicu B"
    )


# ==================================================== timing is causal
def test_anti_causal_teaching_teaches_nothing():
    """Kontrol kritis: urutan terbalik TIDAK boleh menghasilkan asosiasi.

    Kalau test ini gagal, yang kita punya bukan STDP — itu aturan
    "dua neuron yang sama-sama aktif jadi makin kuat", tanpa
    informasi waktu sama sekali.

    `reset_traces=True` wajib di sini — lihat catatan di `_teach`.
    """
    w = _teach([(0, 1)], causal=False, reset_traces=True)
    assert _respond(w, 0, 1) == 0, (
        "post yang fire lebih awal menghasilkan LTD, bukan asosiasi"
    )
    assert float(w[0, 1]) <= W_INIT + 1e-5, (
        f"bobot bahkan boleh turun, bukan naik: {float(w[0, 1]):.4f}"
    )


def test_causal_still_learns_with_trace_reset():
    """Kontrol:activeitas `reset_traces` bukan penyebab utama.

    Kalau ini gagal berarti `reset_traces` yang merusak, bukan
    residual trace.
    """
    w = _teach([(0, 1)], causal=True, reset_traces=True)
    assert float(w[0, 1]) > 1.0, f"kausal harus tetap belajar, dapat {float(w[0, 1]):.4f}"
    assert _respond(w, 0, 1) == 60


def test_equal_eta_cannot_teach_this_task():
    """`eta_plus == eta_minus` mandek di fixed point — asosiasi tak terbentuk.

    Konsekuensi dari ADR-0005 yang belum pernah diuji pada level
    task: learning rate simetris bukan hanya "lebih lambat", tapi
    benar-benar tidak bisa mengajar di sini.
    """
    w = _teach([(0, 1)], eta_plus=2e-3, eta_minus=2e-3)
    assert float(w[0, 1]) < 1.0, (
        f"fixed point harus di bawah v_th=1.0, dapat {float(w[0, 1]):.4f}"
    )
    assert _respond(w, 0, 1) == 0


# ================================================ bidirectional
def test_both_directions_can_be_learned():
    """A->B dan B->A sekaligus, dan tetap selektif.

    Ini yang 준비 structure berlapis di Phase 3 butuhkan: dua asosiasi
    yang hidup bersamaan tanpa saling menimpa.
    """
    w = _teach([(0, 1), (1, 0)])
    assert _respond(w, 0, 1) == 60, "A -> B"
    assert _respond(w, 1, 0) == 60, "B -> A"
    # pasangan yang tidak diajarkan harus tetap bungkam
    assert _respond(w, 0, 0) == 0, "A -> A tidak diajarkan"
    assert _respond(w, 1, 1) == 0, "B -> B tidak postrepair"


def test_more_training_strengthens_the_association():
    """Pembelajaran harus monoton dengan usaha — bukan acak."""
    weak = _teach([(0, 1)], epochs=50)
    strong = _teach([(0, 1)], epochs=200)
    assert float(strong[0, 1]) > float(weak[0, 1]), (
        f"200 epoch harus lebih besar dari 50: "
        f"{float(weak[0, 1]):.4f} -> {float(strong[0, 1]):.4f}"
    )


def test_tighter_weight_bound_weakens_response():
    """`w_max` yang lebih rendah harus membatasi quanto yang bisa dicapai.

    Ini yang membuat homeostasis punya tegangan: bounding Hebbian juga
    membatasi apa yang bisa dipelajari.
    """
    small = _respond(_teach([(0, 1)], epochs=200), 0, 1, hold=30)

    cfg_w = STDPConfig(eta_plus=ETA_PLUS, eta_minus=ETA_MINUS, w_max=1.0)
    w = torch.full((N, N), W_INIT)
    tp, tq = make_trace(N), make_trace(N)
    for _ in range(200):
        for t in range(TRIAL_MS):
            zp, zq = torch.zeros(N), torch.zeros(N)
            if t == 0:
                zp[0] = 1.0
            if t == DELTA_MS:
                zq[1] = 1.0
            w, tp, tq = stdp_step(zp, zq, w, tp, tq, cfg_w)
    big = _respond(w, 0, 1, hold=30)

    assert float(w[0, 1]) < W_MAX, "harus berhenti di w_max yang lebih kecil"
    assert big < small, (
        f"w_max=1.0 harus lebih lemah: {big} vs {small}"
    )


# ============================================================ seeds
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_untrained_network_never_responds(seed):
    """Kontrol untuk semua test di atas, di beberapa seed berbeda.

    Kalau neuron bisa fire tanpa belajar, maka setiap "berhasil
    belajar" di file ini bisa jadi noise.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(N, N, generator=g) * W_INIT
    assert _respond(w, 0, 1, trials=20) == 0
    assert _respond(w, 1, 0, trials=20) == 0
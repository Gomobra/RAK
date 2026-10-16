"""Lateral inhibition.

T1B.3 — kompetisi antar neuron eksitasi.

Arsitektur (Diehl & Cook 2015, satu neuron inhibitor per neuron eksitasi):

    E -> I : SATU-SATU (matriks identitas). Tiap spike di neuron
             eksitasi memicu spike di neuron inhibitor pasangannya.
    I -> E : ALL-TO-ALL. Tiap neuron inhibitor menghambat SEMUA
             neuron eksitasi.

Kenapa bukan inhibisi all-to-all langsung E->E:

- Sepadan dengan circuito basal ganglia di otak (engram -> interneuron
  -> striatum).
- Inhibitor jadi "unit pembanding" yang bernilai seragam untuk semua
  neuron eksitasi, sehingga kompetisinya adil — semua neuron dapat
  perlakuan yang sama.
- Lebih murah komputasi: instead `n_e * n_e` bobot inhibisi, cukup
  `n_e * n_e` bobot E->I (identitas, gratis) + `n_i * n_e` bobot I->E
  dengan `n_i = n_e`. Keduanya sama banyaknya, tapi jalurnya eksplisit dan
  bisa dimodulasi terpisah (inhibitory learning, STDP dg lr berbeda).

Populasi inhibitor adalah LIF terpisah dengan parameter berbeda:

    v_rest_i = -60 mV   (bukan -65 seperti eksitasi)
    v_reset_i = -45 mV  (lebih tinggi dari rest)
    v_thresh_i = -40 mV
    refrac_i = 2 ms

`inhibition_strength` (rho) harus diskalakan dengan ukuran jaringan:
kalau tidak, inhibisi jadi tidak berarti pada jaringan besar.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "InhibitionConfig",
    "InhibitoryWeights",
    "build_inhibitory_weights",
    "make_inhibitory_neuron",
    "make_excitatory_neuron",
]


@dataclass(frozen=True)
class InhibitionConfig:
    """Parameter lateral inhibition."""

    #: Kekuatan bobot inhibitor. Diskalakan oleh `n_exc`.
    #: Diehl-Cook memakai bobot inhibisi yang jauh lebih besar dari
    #: bobot eksitasi, tapi skalanya relatif terhadap `n_exc`.
    inhibition_strength: float = 1.0

    #: Kalau True, bobot inhibisi diskalakan `1/n_exc` supaya TOTAL
    #: inhibisi yang diterima tiap neuron eksitasi tetap konstan
    #: terlepas dari ukuran jaringan.
    #:
    #: Perhitungan: ada `n_inh = n_exc` neuron inhibitor, masing-masing
    #: konek ke semua neuron eksitasi, jadi total inhibisi per neuron
    #: eksitasi = `n_exc * w`. Supaya konstan, `w = strength / n_exc`.
    scale_per_neuron: bool = True

    refrac_ms: float = 2.0
    v_rest: float = -1.0
    v_reset: float = -0.45
    v_th: float = -0.4
    tau_mem_ms: float = 10.0


@dataclass
class InhibitoryWeights:
    """Bobot inhibisi, sudah diskalakan."""

    #: (n_exc, n_i) — biasanya matriks identitas, tapi disimpan penuh
    #: supaya neuron inhibisi bisa punya sifat sendiri nanti.
    e_to_i: Tensor
    #: (n_i, n_exc) — all-to-all, semua nilai negatif.
    i_to_e: Tensor

    @property
    def n_exc(self) -> int:
        return self.i_to_e.shape[1]

    @property
    def n_inh(self) -> int:
        return self.i_to_e.shape[0]


def build_inhibitory_weights(
    n_exc: int, cfg: InhibitionConfig | None = None
) -> InhibitoryWeights:
    """Bangun bobot inhibisi untuk jaringan `n_exc` neuron eksitasi.

    Args:
        n_exc: jumlah neuron eksitasi
        cfg: konfigurasi. `n_inh` selalu = `n_exc` (satu inhibitor per
            eksitasi, seperti Diehl-Cook).

    Returns:
        `InhibitoryWeights` dengan bobot yang sudah diskalakan.
    """
    cfg = cfg or InhibitionConfig()
    if n_exc <= 0:
        raise ValueError(f"n_exc harus > 0, dapat {n_exc}")

    n_inh = n_exc

    # E -> I: satu-satu (identitas)
    e_to_i = torch.eye(n_exc, dtype=torch.float32)

    # I -> E: all-to-all, semua negatif.
    # Skala supaya TOTAL inhibisi tidak tumbuh bareng ukuran jaringan:
    # kalau tidak, jaringan besar akan langsung ter-damp semua.
    strength = cfg.inhibition_strength
    if cfg.scale_per_neuron:
        strength = strength / n_exc
    i_to_e = torch.full((n_inh, n_exc), -abs(strength), dtype=torch.float32)

    return InhibitoryWeights(e_to_i=e_to_i, i_to_e=i_to_e)


def make_inhibitory_neuron(
    n_neurons: int, cfg: InhibitionConfig | None = None
):
    """Buat neuron inhibitor dengan parameter Diehl-Cook yang benar.

    PENTING: neuron inhibitor TIDAK boleh dibangun dengan default LIF.
    `LIFConfig.v_leak = 0.0` sedangkan `InhibitionConfig.v_th = -0.4`
    (yaitu -40 mV). Kalau dibiarkan, v dimulai di 0.0 yang sudah LEBIH
    dari threshold, jadi neuron inhibitor akan fire di SETIAP timestep
    dan mematikan seluruh jaringan.

    Parameter Diehl-Cook:

        v_rest_i   = -60 mV   (v_leak)
        v_reset_i  = -45 mV
        v_thresh_i = -40 mV
        tau_mem_i  = 10 ms
        refrac_i   = 2 ms
    """
    from rak.neuron.lif import LIF

    cfg = cfg or InhibitionConfig()
    return LIF(
        n_neurons,
        tau_mem_ms=cfg.tau_mem_ms,
        v_leak=cfg.v_rest,
        v_reset=cfg.v_reset,
        v_th=cfg.v_th,
        refrac_ms=cfg.refrac_ms,
    )


def make_excitatory_neuron(n_neurons: int, tau_mem_ms: float = 20.0):
    """Buat neuron eksitasi dengan parameter standar RAK.

    Diehl-Cook memakai `tau_mem_e = 100 ms` dengan skala tegangan berbeda.
    RAK memakai bentuk implisit dengan `v_th = 1.0` sebagai unit, jadi
    `tau_mem` 20 ms sudah caled dengan `SpikeLayer`.
    """
    from rak.neuron.lif import LIF

    return LIF(
        n_neurons,
        tau_mem_ms=tau_mem_ms,
        v_leak=0.0,
        v_reset=0.0,
        v_th=1.0,
    )
"""Mekanisme homeostatis — Phase 2B.

Tiga mekanisme yang menjaga aktivitas neuron tetap stabil. Waktu yang yang
memulai plastisitas dan selesai jauh sesudahnya:

    intrinsic.py  adaptive threshold (theta)
    scaling.py    column normalization (hard) + Oja-style (soft)
    inhibition.py lateral inhibition (kompetisi)

Kenapa tiga-duanya dibutuhkan: masing-masing menahan skala yang berbeda.
Intrinsic menahan *excitability*, scaling menahan *drive*, inhibition
menahan *relasi antar neuron*. Menghapus satu saja tidak otomatis
menggantikan yang lain.
"""

from rak.homeostasis.inhibition import (
    InhibitionConfig,
    InhibitoryWeights,
    build_inhibitory_weights,
    make_excitatory_neuron,
    make_inhibitory_neuron,
)
from rak.homeostasis.intrinsic import (
    TIME_SCALES,
    HomeostaticConfig,
    make_homeostatic_neuron,
    validate_timescales,
)
from rak.homeostasis.scaling import (
    ScalingConfig,
    hard_normalize,
    incoming_drive,
    rate_uniformity,
    soft_normalize,
    target_sum_for,
)

__all__ = [
    "TIME_SCALES",
    "HomeostaticConfig",
    "InhibitionConfig",
    "InhibitoryWeights",
    "ScalingConfig",
    "build_inhibitory_weights",
    "hard_normalize",
    "make_excitatory_neuron",
    "make_inhibitory_neuron",
    "incoming_drive",
    "make_homeostatic_neuron",
    "rate_uniformity",
    "soft_normalize",
    "target_sum_for",
    "validate_timescales",
]

"""Homeostasis (Phase 2)."""

from rak.homeostasis.inhibition import apply_inhibition, build_inhibition
from rak.homeostasis.intrinsic import adaptive_threshold_step
from rak.homeostasis.scaling import hard_normalize, soft_scale

__all__ = [
    "adaptive_threshold_step",
    "hard_normalize", "soft_scale",
    "build_inhibition", "apply_inhibition",
]
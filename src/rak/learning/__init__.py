"""Aturan belajar lokal — Phase 2A."""

from rak.learning.hebbian import (
    BCMConfig,
    BCMState,
    HebbianConfig,
    OjaConfig,
    bcm_theta,
    bcm_update,
    hebbian_update,
    oja_update,
)
from rak.learning.stdp3 import (
    Eq3Config,
    Eq3State,
    eq3_step,
    eq3_window,
    make_eq3_state,
)
from rak.learning.stdp import (
    ALGORITHMS,
    STDPConfig,
    A_minus,
    A_plus,
    make_trace,
    stdp_step,
    stdp_step_linear_layers,
    stdp_window,
)

__all__ = [
    "Eq3Config",
    "Eq3State",
    "eq3_step",
    "eq3_window",
    "make_eq3_state",
    "ALGORITHMS",
    "BCMConfig",
    "BCMState",
    "HebbianConfig",
    "OjaConfig",
    "STDPConfig",
    "A_minus",
    "A_plus",
    "bcm_theta",
    "bcm_update",
    "hebbian_update",
    "make_trace",
    "oja_update",
    "stdp_step",
    "stdp_step_linear_layers",
    "stdp_window",
]
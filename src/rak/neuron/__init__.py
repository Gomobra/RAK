"""Subpackage: neuron models (Phase 1)."""

from rak.neuron.trace import (
    PoissonEncoder,
    decay_to_target,
    poisson_spike_train,
    spike_trace,
)
from rak.neuron.lif import LIF, LIFConfig
from rak.neuron.conductance import (
    ConductanceConfig,
    ConductanceLIF,
    ConductanceState,
)
from rak.neuron.adaptive_lif import AdaptiveLIF
from rak.neuron.izhikevich import Izhikevich
from rak.neuron.layer import SpikeLayer

__all__ = [
    "ConductanceConfig",
    "ConductanceState",
    "ConductanceLIF",
    "PoissonEncoder",
    "decay_to_target",
    "poisson_spike_train",
    "spike_trace",
    "LIF",
    "LIFConfig",
    "AdaptiveLIF",
    "Izhikevich",
    "SpikeLayer",
]
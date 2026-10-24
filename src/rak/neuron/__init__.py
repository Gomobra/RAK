"""Neuron models (Phase 1)."""

from rak.neuron.adaptive_lif import AdaptiveLIF, AdaptiveLIFConfig
from rak.neuron.izhikevich import IZHIKEVICH_MODES, Izhikevich, IzhikevichConfig
from rak.neuron.layer import SpikeLayer
from rak.neuron.lif import LIF, LIFConfig
from rak.neuron.trace import make_trace, poisson_spike_train, trace_step

__all__ = [
    "LIF", "LIFConfig",
    "AdaptiveLIF", "AdaptiveLIFConfig",
    "Izhikevich", "IzhikevichConfig", "IZHIKEVICH_MODES",
    "SpikeLayer",
    "poisson_spike_train", "make_trace", "trace_step",
]
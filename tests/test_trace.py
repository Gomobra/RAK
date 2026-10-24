"""Test spike trace — properti, bukan asumsi."""

from __future__ import annotations

import math

import torch

from rak.neuron.trace import make_trace, poisson_spike_train, trace_step


def test_trace_monoton_turun():
    trace = make_trace(4, value=1.0)
    gamma = math.exp(-1.0 / 20.0)
    vals = [float(trace.mean())]
    for _ in range(20):
        trace = trace_step(trace, torch.zeros(4), gamma)
        vals.append(float(trace.mean()))
    assert vals[-1] < vals[0]


def test_trace_increments_on_spike():
    trace = make_trace(4, value=0.0)
    gamma = math.exp(-1.0 / 20.0)
    sp = torch.ones(4)
    new = trace_step(trace, sp, gamma)
    assert torch.allclose(new, sp)


def test_poisson_spike_train_shape():
    sp = poisson_spike_train(50.0, 100.0, dt_ms=1.0)
    assert sp.shape == (100,)
    assert set(sp.unique().tolist()) <= {0.0, 1.0}
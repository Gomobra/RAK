"""Test Izhikevich neuron — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.neuron.izhikevich import IZHIKEVICH_MODES, Izhikevich, IzhikevichConfig


def test_izhikevich_fires():
    neuron = Izhikevich(4)
    I = torch.full((300, 4), 10.0)
    sp = neuron.simulate(I)
    assert sp.sum() > 0, "Izhikevich harus fire"


def test_izhikevich_silent_on_zero():
    neuron = Izhikevich(4)
    I = torch.zeros(300, 4)
    sp = neuron.simulate(I)
    assert sp.sum() == 0, "Izhikevich tidak boleh fire tanpa input"


def test_all_six_modes_fire():
    for name, cfg in IZHIKEVICH_MODES.items():
        neuron = Izhikevich(4, config=cfg)
        I = torch.full((300, 4), 10.0)
        sp = neuron.simulate(I)
        assert sp.sum() > 0, f"mode {name} harus fire"


def test_izhikevich_config_validation():
    try:
        IzhikevichConfig(dt_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("dt_ms <= 0 harus ValueError")


def test_izhikevich_simulate_shape():
    neuron = Izhikevich(8)
    I = torch.rand(100, 8)
    sp = neuron.simulate(I)
    assert sp.shape == (100, 8)
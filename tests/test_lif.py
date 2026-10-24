"""Test LIF neuron — properti, bukan asumsi."""

from __future__ import annotations

import torch

from rak.neuron.lif import LIF, LIFConfig


def test_lif_fires_on_suprathreshold_input():
    lif = LIF(4)
    I = torch.full((200, 4), 15.0)
    sp = lif.simulate(I)
    assert sp.sum() > 0, "LIF harus fire pada input di atas threshold"


def test_lif_silent_on_zero_input():
    lif = LIF(4)
    I = torch.zeros(200, 4)
    sp = lif.simulate(I)
    assert sp.sum() == 0, "LIF tidak boleh fire tanpa input"


def test_lif_firing_rate_increases_with_current():
    r1 = LIF(4).firing_rate(current=5.0, n_steps=500)
    r2 = LIF(4).firing_rate(current=15.0, n_steps=500)
    assert r2 > r1, "laju fire harus naik dengan arus"


def test_lif_reset_after_spike():
    cfg = LIFConfig(v_rest=-65.0, v_th=-52.0, v_reset=-65.0)
    lif = LIF(1, config=cfg)
    I = torch.full((50, 1), 100.0)
    sp, v = lif.simulate(I, record=True)
    # Setelah spike pertama, V harus reset
    first_spike = int(sp.flatten().argmax())
    assert v[first_spike] == cfg.v_reset or v[first_spike] <= cfg.v_th


def test_lif_config_validation():
    try:
        LIFConfig(tau_mem_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("tau_mem_ms <= 0 harus ValueError")
    try:
        LIFConfig(v_reset=-40.0, v_th=-52.0)
    except ValueError:
        pass
    else:
        raise AssertionError("v_reset >= v_th harus ValueError")


def test_lif_simulate_shape():
    lif = LIF(8)
    I = torch.rand(100, 8)
    sp = lif.simulate(I)
    assert sp.shape == (100, 8)
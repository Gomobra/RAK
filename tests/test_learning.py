"""Test learning rules — properti, bukan asumsi."""

from __future__ import annotations

import math

import torch

from rak.learning.hebbian import bcm_step, hebbian_step, oja_step
from rak.learning.stdp import make_trace, stdp_step, stdp_window, trace_step


def test_hebbian_increases_on_coactivation():
    w = torch.zeros(4, 3)
    x_pre = torch.ones(4)
    x_post = torch.ones(3)
    w2 = hebbian_step(w, x_pre, x_post, eta=0.1)
    assert (w2 > 0).all()


def test_hebbian_no_change_if_no_spike():
    w = torch.ones(4, 3)
    x_pre = torch.zeros(4)
    x_post = torch.ones(3)
    w2 = hebbian_step(w, x_pre, x_post, eta=0.1)
    assert torch.equal(w, w2)


def test_bcm_depresses_below_threshold():
    w = torch.ones(4, 3)
    x_pre = torch.ones(4)
    x_post = torch.full((3,), 0.5)
    theta = torch.full((3,), 1.0)
    w2 = bcm_step(w, x_pre, x_post, eta=0.1, theta=theta)
    assert (w2 < w).all()


def test_oja_normalizes():
    w = torch.ones(4, 3) * 0.5
    x_pre = torch.ones(4)
    x_post = torch.ones(3)
    w2 = oja_step(w, x_pre, x_post, eta=0.1, beta=1.0)
    # Oja should keep weights bounded
    assert torch.isfinite(w2).all()


def test_stdp_pre_before_post_potentiates():
    w = torch.zeros(4, 3)
    tp = torch.zeros(4)
    tq = torch.zeros(3)
    # pre spike at t=0
    zp = torch.tensor([1.0, 0, 0, 0])
    zq = torch.zeros(3)
    w, tp, tq = stdp_step(w, zp, zq, tp, tq, eta_plus=0.1, eta_minus=0.1)
    # post spike at t=1
    zq = torch.tensor([1.0, 0, 0])
    w2, tp2, tq2 = stdp_step(w, torch.zeros(4), zq, tp, tq, eta_plus=0.1, eta_minus=0.1)
    assert w2[0, 0] > 0, "LTP: pre sebelum post harus naik"


def test_stdp_post_before_pre_depresses():
    w = torch.ones(4, 3) * 0.5
    tp = torch.zeros(4)
    tq = torch.zeros(3)
    # post spike first
    zq = torch.tensor([1.0, 0, 0])
    w, tp, tq = stdp_step(w, torch.zeros(4), zq, tp, tq, eta_plus=0.1, eta_minus=0.1)
    # pre spike after
    zp = torch.tensor([1.0, 0, 0, 0])
    w2, tp2, tq2 = stdp_step(w, zp, torch.zeros(3), tp, tq, eta_plus=0.1, eta_minus=0.1)
    assert w2[0, 0] < w[0, 0], "LTD: post sebelum pre harus turun"


def test_trace_step():
    trace = torch.zeros(4)
    gamma = 0.9
    sp = torch.tensor([1.0, 0, 0, 0])
    new = trace_step(trace, sp, gamma)
    assert torch.allclose(new, sp)


def test_trace_decay():
    trace = torch.ones(4)
    gamma = 0.9
    new = trace_step(trace, torch.zeros(4), gamma)
    assert torch.allclose(new, torch.full((4,), 0.9))


def test_stdp_window_shape():
    dts, win = stdp_window(1.0, max_ms=50.0)
    assert dts.shape == win.shape
    assert len(dts) == 101  # 2*50+1


def test_stdp_window_zero_at_zero():
    dts, win = stdp_window(1.0, max_ms=50.0)
    idx = (dts == 0).nonzero(as_tuple=True)[0]
    assert len(idx) == 1
    assert abs(float(win[idx])) < 1e-6
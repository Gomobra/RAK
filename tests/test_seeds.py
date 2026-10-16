"""Seed management tests (Fase 0 deliverable, NFR-05)."""

import random

import numpy as np
import torch

from rak.eval.seeds import DEFAULT_SEEDS, seed_everything, seed_iter


def test_default_seeds_is_1_to_10():
    assert DEFAULT_SEEDS == tuple(range(1, 11))


def test_seed_everything_is_reproducible():
    seed_everything(42)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(42)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_different_seeds_differ():
    seed_everything(1)
    a = (random.random(), float(np.random.rand()))
    seed_everything(2)
    b = (random.random(), float(np.random.rand()))
    assert a != b


def test_seed_iter_yields_all_seeds():
    specs = list(seed_iter("test_exp", {"lr": 0.1}, seeds=(1, 2, 3)))
    assert [s.seed for s in specs] == [1, 2, 3]
    assert all(s.experiment == "test_exp" for s in specs)
    assert all(s.config == {"lr": 0.1} for s in specs)


def test_seed_spec_is_jsonable():
    specs = list(seed_iter("exp", {"n": 5}, seeds=(7,)))
    d = specs[0].to_dict()
    assert d["seed"] == 7
    assert d["experiment"] == "exp"
    assert d["config"] == {"n": 5}


def test_torch_deterministic_flag_set():
    seed_everything(1, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    seed_everything(1, deterministic=False)  # restore-ish, tidak diassert
"""pytest fixtures bersama."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    """Semua test jalan deterministik. Lihat NFR-05."""
    torch.manual_seed(1234)
    yield


@pytest.fixture
def n_neurons() -> int:
    return 100


@pytest.fixture
def n_timesteps() -> int:
    return 100
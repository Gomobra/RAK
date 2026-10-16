"""Seed management — NFR-05.

Aturan validasi (JMLR 21(55), Li & Talwalkar 2020): SEMUA hasil harus
mean ± std over multiple seeds. STDP punya variance sangat tinggi, jadi
satu run bukan bukti apa pun.

Default RAK: seeds 1..10. Seed dicatat di setiap config hasil run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch

__all__ = ["DEFAULT_SEEDS", "SeedSpec", "seed_everything", "seed_iter"]

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1, 11))


@dataclass(frozen=True)
class SeedSpec:
    """Seed untuk satu eksperimen + metadata json-able."""

    seed: int
    experiment: str = "unnamed"
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "experiment": self.experiment,
            "config": self.config,
        }


def seed_everything(seed: int, deterministic: bool = True) -> SeedSpec:
    """Seed python, numpy, dan torch sekaligus."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Global, jadi panggilan berikutnya ikut terpengaruh. Ini yang
        # dicari untuk run-to-run reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return SeedSpec(seed=seed)


def seed_iter(
    experiment: str,
    config: dict | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Iterator[SeedSpec]:
    """Yield SeedSpec untuk setiap seed, sudah di-seed."""
    config = config or {}
    for s in seeds:
        spec = seed_everything(s)
        yield SeedSpec(seed=s, experiment=experiment, config=config)
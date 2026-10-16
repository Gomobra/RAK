"""Jaringan dan loop training MNIST."""

from rak.network.competitive import (
    CompetitiveNetwork,
    NetworkConfig,
    assign_classes,
    classify,
)

__all__ = [
    "NetworkConfig",
    "CompetitiveNetwork",
    "assign_classes",
    "classify",
]

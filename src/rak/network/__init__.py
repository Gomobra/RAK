"""Network module (Phase 3)."""

from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.network.train import TrainConfig, accuracy, evaluate, train

__all__ = ["CompetitiveNetwork", "NetworkConfig", "TrainConfig", "train", "evaluate", "accuracy"]
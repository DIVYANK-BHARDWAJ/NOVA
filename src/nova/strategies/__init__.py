"""NOVA Quantitative Strategy Subsystem (Person 3 Exclusive Domain)."""

from nova.strategies.base import BaseStrategy, StrategySignal
from nova.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy

__all__ = [
    "BaseStrategy",
    "StrategySignal",
    "TrendFollowingConfig",
    "TrendFollowingStrategy",
    "MeanReversionConfig",
    "MeanReversionStrategy",
]

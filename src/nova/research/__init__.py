"""NOVA Research subsystem package."""

from nova.research.adapters import (
    generate_synthetic_ohlcv,
    load_market_bars_from_csv,
    market_tick_to_market_bar,
    parse_timestamp_iso,
    validate_market_bar_sequence,
)
from nova.research.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)
from nova.research.evaluator import (
    BarEvaluationStep,
    EvaluationReport,
    FrictionModel,
    StrategyEvaluator,
)

__all__ = [
    "BarEvaluationStep",
    "EMACrossoverConfig",
    "EMACrossoverStrategy",
    "EvaluationReport",
    "FrictionModel",
    "StrategyEvaluator",
    "generate_synthetic_ohlcv",
    "load_market_bars_from_csv",
    "market_tick_to_market_bar",
    "parse_timestamp_iso",
    "validate_market_bar_sequence",
]


"""NOVA Research subsystem package."""

from nova.research.adapters import (
    generate_synthetic_ohlcv,
    load_market_bars_from_csv,
    market_tick_to_market_bar,
    parse_simulated_date_to_timestamp,
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
from nova.research.kafka_connector import (
    ConnectorMetrics,
    KafkaStrategyConnector,
)

__all__ = [
    "BarEvaluationStep",
    "ConnectorMetrics",
    "EMACrossoverConfig",
    "EMACrossoverStrategy",
    "EvaluationReport",
    "FrictionModel",
    "KafkaStrategyConnector",
    "StrategyEvaluator",
    "generate_synthetic_ohlcv",
    "load_market_bars_from_csv",
    "market_tick_to_market_bar",
    "parse_simulated_date_to_timestamp",
    "parse_timestamp_iso",
    "validate_market_bar_sequence",
]


"""Market Data Ingestion Package for MANTHAN 2026.

Provides strategy-agnostic Kafka consumption, normalization, validation,
and event dispatching for market ticks and corporate news.
"""

from market_data.config import ConfigValidationError, KafkaConfig
from market_data.consumer import MarketDataConsumer, PlaceholderBrokerError
from market_data.handlers import (
    DiagnosticLoggingNewsHandler,
    DiagnosticLoggingTickHandler,
    DispatcherMetrics,
    EventDispatcher,
    MarketNewsHandlerProtocol,
    MarketTickHandlerProtocol,
)
from market_data.models import (
    KafkaMetadata,
    MarketNewsEvent,
    MarketTickEvent,
    ValidationResult,
)
from market_data.normalizer import (
    NormalizationError,
    decode_json_payload,
    normalize_market_news,
    normalize_market_tick,
    parse_date,
    parse_timestamp,
)
from market_data.validators import (
    ValidationError,
    validate_market_news,
    validate_market_tick,
)

__all__ = [
    "ConfigValidationError",
    "DiagnosticLoggingNewsHandler",
    "DiagnosticLoggingTickHandler",
    "DispatcherMetrics",
    "EventDispatcher",
    "KafkaConfig",
    "KafkaMetadata",
    "MarketDataConsumer",
    "MarketNewsEvent",
    "MarketNewsHandlerProtocol",
    "MarketTickEvent",
    "MarketTickHandlerProtocol",
    "NormalizationError",
    "PlaceholderBrokerError",
    "ValidationError",
    "ValidationResult",
    "decode_json_payload",
    "normalize_market_news",
    "normalize_market_tick",
    "parse_date",
    "parse_timestamp",
    "validate_market_news",
    "validate_market_tick",
]

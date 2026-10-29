"""Strategy-independent domain models for market ticks and news events.

These models encapsulate normalized Kafka payloads and preserve metadata,
raw payloads, and unknown candidate fields for downstream consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional


@dataclass(frozen=True)
class KafkaMetadata:
    """Metadata extracted from the underlying Kafka record."""

    topic: str
    partition: int = -1
    offset: int = -1
    timestamp_ms: Optional[int] = None
    timestamp_type: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "timestamp_ms": self.timestamp_ms,
            "timestamp_type": self.timestamp_type,
        }


@dataclass(frozen=True)
class MarketTickEvent:
    """Normalized OHLCV market tick event.

    Strategy-agnostic representation of candlestick records received from
    the 'market.ticks' topic.

    Attributes:
        symbol: Ticker identifier (e.g. 'ROUT', 'TCS').
        open: Open price for the candle (> 0).
        high: High price for the candle (>= low, >= open, >= close).
        low: Low price for the candle (<= high, <= open, <= close).
        close: Close price for the candle (> 0).
        volume: Volume traded during the candle (>= 0).
        timestamp: Event emission or broker UTC timestamp.
        simulated_date: Historical market date being replayed in simulation (e.g. 2024-10-25).
        metadata: Kafka transport metadata (topic, partition, offset).
        raw_payload: Original parsed JSON dictionary.
        extra: Unknown or unmapped candidate fields.
    """

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Optional[datetime] = None
    simulated_date: Optional[date] = None
    metadata: Optional[KafkaMetadata] = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_bullish(self) -> bool:
        """Helper check if close >= open."""
        return self.close >= self.open

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary, suitable for downstream ingestion."""
        return {
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "simulated_date": self.simulated_date.isoformat() if self.simulated_date else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class MarketNewsEvent:
    """Normalized corporate news event.

    Strategy-agnostic representation of news announcements received from
    the 'market.news' topic.

    Attributes:
        headline: Main news headline or announcement text.
        symbol: Ticker or company identifier (None if macro/general market news).
        timestamp: Event emission or broker UTC timestamp.
        simulated_date: Historical market date being replayed in simulation.
        news_date: Publication date reported inside the news body.
        metadata: Kafka transport metadata (topic, partition, offset).
        raw_payload: Original parsed JSON dictionary.
        extra: Unmapped fields (description, urgency, source, etc.).
    """

    headline: str
    symbol: Optional[str] = None
    timestamp: Optional[datetime] = None
    simulated_date: Optional[date] = None
    news_date: Optional[date] = None
    metadata: Optional[KafkaMetadata] = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary, suitable for downstream ingestion."""
        return {
            "headline": self.headline,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "simulated_date": self.simulated_date.isoformat() if self.simulated_date else None,
            "news_date": self.news_date.isoformat() if self.news_date else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Encapsulates the outcome of validating a normalized event."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        """Formatted string of validation errors."""
        return "; ".join(self.errors)

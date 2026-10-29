"""NOVA Quantitative Research — Market News & Sentiment Interface (Person 2).

Provides an architectural interface and research contract for integrating genuine
corporate news events and NLP sentiment scores into quantitative feature sets.

POLICY ENFORCEMENT:
- Strictly prohibits generating or training on synthetic news text.
- If real news archives are unavailable, flags status as BLOCKED_INSUFFICIENT_REAL_DATA.
- Enforces point-in-time publication timestamp alignment (anti-lookahead protection).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from nova.quant.contracts import (
    ContractValidationError,
    NumericalValidationError,
    validate_finite,
    validate_no_secrets,
    validate_timezone_aware,
)


class NewsSentimentStatus(str, Enum):
    """Availability status of genuine market news data."""
    READY = "READY"
    BLOCKED_INSUFFICIENT_REAL_DATA = "BLOCKED_INSUFFICIENT_REAL_DATA"
    NO_EVENTS_FOR_WINDOW = "NO_EVENTS_FOR_WINDOW"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class MarketNewsRecord:
    """Canonical representation of an ingested market news event."""
    news_id: str
    headline: str
    published_at: datetime
    tickers: tuple[str, ...]
    sentiment_score: float | None = None  # Bounded in [-1.0, +1.0] if present
    confidence: float | None = None       # Bounded in [0.0, 1.0]
    provenance: str = "MANTHAN_KAFKA_MARKET_NEWS"

    def __post_init__(self) -> None:
        if not self.news_id or not isinstance(self.news_id, str):
            raise ContractValidationError("news_id must be a non-empty string")
        if not self.headline or not isinstance(self.headline, str):
            raise ContractValidationError("headline must be a non-empty string")
        validate_timezone_aware(self.published_at, "published_at")
        if not self.tickers:
            raise ContractValidationError("tickers tuple cannot be empty")
        if self.sentiment_score is not None:
            validate_finite(self.sentiment_score, "sentiment_score")
            if not (-1.0 <= self.sentiment_score <= 1.0):
                raise NumericalValidationError(f"sentiment_score must be in [-1.0, 1.0], got {self.sentiment_score}")
        if self.confidence is not None:
            validate_finite(self.confidence, "confidence")
            if not (0.0 <= self.confidence <= 1.0):
                raise NumericalValidationError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


class NewsSentimentInterface:
    """Research interface for extracting point-in-time sentiment features."""

    def __init__(self, lookback_window_hours: float = 24.0) -> None:
        self.lookback_window_hours = lookback_window_hours
        self._news_store: list[MarketNewsRecord] = []

    def ingest_news_events(self, events: Sequence[MarketNewsRecord]) -> int:
        """Store validated real news events in chronological order."""
        count = 0
        for ev in events:
            if not isinstance(ev, MarketNewsRecord):
                raise ContractValidationError(f"Expected MarketNewsRecord, got {type(ev)}")
            self._news_store.append(ev)
            count += 1
        self._news_store.sort(key=lambda x: x.published_at)
        return count

    def compute_sentiment_feature(
        self,
        ticker: str,
        as_of: datetime,
    ) -> tuple[NewsSentimentStatus, float]:
        """Compute point-in-time aggregated sentiment score for a ticker.

        Returns (status, score) where score is in [-1.0, 1.0] or 0.0 if unavailable.
        """
        validate_timezone_aware(as_of, "as_of")

        if not self._news_store:
            return NewsSentimentStatus.BLOCKED_INSUFFICIENT_REAL_DATA, 0.0

        window_start = as_of.timestamp() - (self.lookback_window_hours * 3600.0)
        as_of_ts = as_of.timestamp()

        # Strict point-in-time filtering: published_at <= as_of
        eligible = [
            ev for ev in self._news_store
            if ticker in ev.tickers
            and window_start <= ev.published_at.timestamp() <= as_of_ts
            and ev.sentiment_score is not None
        ]

        if not eligible:
            return NewsSentimentStatus.NO_EVENTS_FOR_WINDOW, 0.0

        scores = [ev.sentiment_score for ev in eligible if ev.sentiment_score is not None]
        avg_sentiment = sum(scores) / len(scores)
        return NewsSentimentStatus.READY, round(avg_sentiment, 4)


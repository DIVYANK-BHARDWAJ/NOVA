"""Strategy-agnostic event dispatcher and downstream handler interfaces.

Exposes clean subscription mechanisms for future trading strategies, risk management,
and portfolio systems without embedding any trading logic into the ingestion layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, Union, runtime_checkable

from market_data.models import MarketTickEvent, MarketNewsEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class MarketTickHandlerProtocol(Protocol):
    """Protocol for classes implementing tick event handling."""

    def handle_market_tick(self, event: MarketTickEvent) -> None:
        """Handle a validated OHLCV market tick event."""
        ...


@runtime_checkable
class MarketNewsHandlerProtocol(Protocol):
    """Protocol for classes implementing news event handling."""

    def handle_market_news(self, event: MarketNewsEvent) -> None:
        """Handle a validated corporate news announcement event."""
        ...


TickHandlerType = Union[Callable[[MarketTickEvent], None], MarketTickHandlerProtocol]
NewsHandlerType = Union[Callable[[MarketNewsEvent], None], MarketNewsHandlerProtocol]


class OverflowPolicy(str, Enum):
    """Policy for queue overflow handling if buffering is enabled."""
    REJECT_NEW = "reject_new"
    DROP_OLDEST = "drop_oldest"


@dataclass
class DispatcherMetrics:
    """Metrics tracking dispatched events and handler execution."""

    ticks_dispatched: int = 0
    news_dispatched: int = 0
    handler_errors: int = 0
    dropped_events: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "ticks_dispatched": self.ticks_dispatched,
            "news_dispatched": self.news_dispatched,
            "handler_errors": self.handler_errors,
            "dropped_events": self.dropped_events,
        }


class EventDispatcher:
    """Lightweight, strategy-agnostic event dispatcher.

    Maintains subscriber lists for market ticks and corporate news. Dispatches
    events synchronously with zero overhead by default to satisfy tight latency
    requirements (< 3-4s decision window).

    Catches and logs exceptions from downstream handlers to prevent a faulty
    strategy from terminating the Kafka ingestion loop.
    """

    def __init__(self) -> None:
        self._tick_handlers: list[TickHandlerType] = []
        self._news_handlers: list[NewsHandlerType] = []
        self._metrics = DispatcherMetrics()

    @property
    def metrics(self) -> DispatcherMetrics:
        """Return current dispatcher metrics."""
        return self._metrics

    @property
    def tick_handlers(self) -> list[TickHandlerType]:
        """Return copy of registered tick handlers."""
        return list(self._tick_handlers)

    @property
    def news_handlers(self) -> list[NewsHandlerType]:
        """Return copy of registered news handlers."""
        return list(self._news_handlers)

    def subscribe_tick(self, handler: TickHandlerType) -> None:
        """Register a handler for validated market tick events."""
        if handler not in self._tick_handlers:
            self._tick_handlers.append(handler)
            logger.debug("Registered tick handler: %s", getattr(handler, "__name__", repr(handler)))

    def unsubscribe_tick(self, handler: TickHandlerType) -> None:
        """Unregister a tick handler."""
        if handler in self._tick_handlers:
            self._tick_handlers.remove(handler)

    def subscribe_news(self, handler: NewsHandlerType) -> None:
        """Register a handler for validated corporate news events."""
        if handler not in self._news_handlers:
            self._news_handlers.append(handler)
            logger.debug("Registered news handler: %s", getattr(handler, "__name__", repr(handler)))

    def unsubscribe_news(self, handler: NewsHandlerType) -> None:
        """Unregister a news handler."""
        if handler in self._news_handlers:
            self._news_handlers.remove(handler)

    def dispatch_tick(self, event: MarketTickEvent) -> int:
        """Dispatch a validated MarketTickEvent to all registered handlers.

        Returns:
            Number of successfully executed handlers.
        """
        success_count = 0
        for handler in self._tick_handlers:
            try:
                if hasattr(handler, "handle_market_tick") and callable(getattr(handler, "handle_market_tick")):
                    handler.handle_market_tick(event)
                elif callable(handler):
                    handler(event)
                else:
                    logger.error("Handler %r is neither callable nor implements handle_market_tick", handler)
                    self._metrics.handler_errors += 1
                    continue
                success_count += 1
            except Exception as exc:
                self._metrics.handler_errors += 1
                logger.error(
                    "Error executing tick handler %r for symbol %s: %s",
                    handler,
                    event.symbol,
                    exc,
                    exc_info=True,
                )

        self._metrics.ticks_dispatched += 1
        return success_count

    def dispatch_news(self, event: MarketNewsEvent) -> int:
        """Dispatch a validated MarketNewsEvent to all registered handlers.

        Returns:
            Number of successfully executed handlers.
        """
        success_count = 0
        for handler in self._news_handlers:
            try:
                if hasattr(handler, "handle_market_news") and callable(getattr(handler, "handle_market_news")):
                    handler.handle_market_news(event)
                elif callable(handler):
                    handler(event)
                else:
                    logger.error("Handler %r is neither callable nor implements handle_market_news", handler)
                    self._metrics.handler_errors += 1
                    continue
                success_count += 1
            except Exception as exc:
                self._metrics.handler_errors += 1
                logger.error(
                    "Error executing news handler %r: %s",
                    handler,
                    exc,
                    exc_info=True,
                )

        self._metrics.news_dispatched += 1
        return success_count


# Example downstream handlers for logging and diagnostic purposes
class DiagnosticLoggingTickHandler:
    """Example strategy-agnostic handler that logs tick events without trading logic."""

    def handle_market_tick(self, event: MarketTickEvent) -> None:
        logger.info(
            "[TICK] Symbol=%s Close=%.2f (H=%.2f L=%.2f O=%.2f) Vol=%.0f",
            event.symbol,
            event.close,
            event.high,
            event.low,
            event.open,
            event.volume,
        )


class DiagnosticLoggingNewsHandler:
    """Example strategy-agnostic handler that logs news events without sentiment logic."""

    def handle_market_news(self, event: MarketNewsEvent) -> None:
        logger.info(
            "[NEWS] Symbol=%s Headline=%s",
            event.symbol or "GLOBAL",
            event.headline[:80],
        )

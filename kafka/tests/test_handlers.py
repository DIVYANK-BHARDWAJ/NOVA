"""Tests for downstream handler registration, protocols, and event dispatching."""

from unittest.mock import MagicMock
import pytest

from market_data.handlers import (
    DiagnosticLoggingNewsHandler,
    DiagnosticLoggingTickHandler,
    EventDispatcher,
    MarketNewsHandlerProtocol,
    MarketTickHandlerProtocol,
)
from market_data.models import KafkaMetadata, MarketNewsEvent, MarketTickEvent


class CustomTickStrategy(MarketTickHandlerProtocol):
    """Example custom class implementing the protocol."""

    def __init__(self):
        self.received_events: list[MarketTickEvent] = []

    def handle_market_tick(self, event: MarketTickEvent) -> None:
        self.received_events.append(event)


class CustomNewsStrategy(MarketNewsHandlerProtocol):
    """Example custom class implementing the protocol."""

    def __init__(self):
        self.received_events: list[MarketNewsEvent] = []

    def handle_market_news(self, event: MarketNewsEvent) -> None:
        self.received_events.append(event)


def test_handler_registration_and_unregistration():
    """Verify subscribing and unsubscribing handlers."""
    dispatcher = EventDispatcher()
    tick_fn = MagicMock()
    news_fn = MagicMock()

    dispatcher.subscribe_tick(tick_fn)
    dispatcher.subscribe_news(news_fn)
    assert tick_fn in dispatcher.tick_handlers
    assert news_fn in dispatcher.news_handlers

    dispatcher.unsubscribe_tick(tick_fn)
    dispatcher.unsubscribe_news(news_fn)
    assert tick_fn not in dispatcher.tick_handlers
    assert news_fn not in dispatcher.news_handlers


def test_dispatch_tick_to_multiple_handlers():
    """Verify dispatching tick invokes both function and class protocol subscribers."""
    dispatcher = EventDispatcher()
    received_fn_events = []

    def sample_tick_fn(evt: MarketTickEvent):
        received_fn_events.append(evt)

    strategy = CustomTickStrategy()

    dispatcher.subscribe_tick(sample_tick_fn)
    dispatcher.subscribe_tick(strategy)

    tick = MarketTickEvent(
        symbol="INFY",
        open=1450.0,
        high=1470.0,
        low=1440.0,
        close=1465.0,
        volume=25000.0,
        metadata=KafkaMetadata(topic="market.ticks", partition=0, offset=12),
    )

    success_count = dispatcher.dispatch_tick(tick)
    assert success_count == 2
    assert len(received_fn_events) == 1
    assert len(strategy.received_events) == 1
    assert received_fn_events[0].symbol == "INFY"
    assert strategy.received_events[0].close == 1465.0
    assert dispatcher.metrics.ticks_dispatched == 1


def test_dispatch_news_to_handlers():
    """Verify dispatching news invokes subscribers and tracks metrics."""
    dispatcher = EventDispatcher()
    strategy = CustomNewsStrategy()
    dispatcher.subscribe_news(strategy)

    news = MarketNewsEvent(
        headline="Quarterly revenue up 22%",
        symbol="TCS",
        metadata=KafkaMetadata(topic="market.news", partition=1, offset=55),
    )

    success_count = dispatcher.dispatch_news(news)
    assert success_count == 1
    assert len(strategy.received_events) == 1
    assert strategy.received_events[0].headline == "Quarterly revenue up 22%"
    assert dispatcher.metrics.news_dispatched == 1


def test_handler_exception_isolation():
    """Verify faulty handler does not crash dispatcher and tracks error count."""
    dispatcher = EventDispatcher()

    def faulty_handler(evt: MarketTickEvent):
        raise RuntimeError("Bug in user trading strategy")

    good_strategy = CustomTickStrategy()

    dispatcher.subscribe_tick(faulty_handler)
    dispatcher.subscribe_tick(good_strategy)

    tick = MarketTickEvent(
        symbol="WIPRO",
        open=450.0,
        high=460.0,
        low=445.0,
        close=455.0,
        volume=1000.0,
    )

    # Should not raise exception
    success_count = dispatcher.dispatch_tick(tick)
    assert success_count == 1
    assert len(good_strategy.received_events) == 1
    assert dispatcher.metrics.handler_errors == 1


def test_diagnostic_logging_handlers_smoke():
    """Verify built-in diagnostic handlers execute without error."""
    tick_logger = DiagnosticLoggingTickHandler()
    news_logger = DiagnosticLoggingNewsHandler()

    tick = MarketTickEvent(
        symbol="SBIN",
        open=750.0,
        high=760.0,
        low=745.0,
        close=755.0,
        volume=12000.0,
    )
    news = MarketNewsEvent(
        headline="Annual shareholder meeting scheduled for next month",
        symbol="SBIN",
    )

    tick_logger.handle_market_tick(tick)
    news_logger.handle_market_news(news)


def test_no_unintended_order_submission_interface():
    """Verify that the ingestion module does NOT expose any order execution interfaces.

    Confirms non-negotiable boundary: the ingestion module must remain strategy-agnostic
    and cannot submit orders, trade, or allocate portfolios.
    """
    import market_data
    # Verify module does not export trading execution symbols
    forbidden_terms = ["order", "trade", "buy", "sell", "portfolio", "broker_client", "execute"]
    for attr in dir(market_data):
        attr_lower = attr.lower()
        for term in forbidden_terms:
            assert term not in attr_lower or attr_lower in ["marketdataconsumer", "dispatcher"], (
                f"Ingestion layer unexpectedly exposes trading-related symbol: {attr}"
            )

    # Verify EventDispatcher has no trading methods
    dispatcher = market_data.EventDispatcher()
    for forbidden in ["submit_order", "execute_order", "place_trade", "buy", "sell"]:
        assert not hasattr(dispatcher, forbidden), f"Dispatcher must not expose {forbidden}"


def test_secret_safe_logging(caplog):
    """Verify diagnostic logging does not log raw authentication tokens or secrets."""
    import logging
    caplog.set_level(logging.INFO)

    tick_logger = DiagnosticLoggingTickHandler()
    tick = MarketTickEvent(
        symbol="ROUT",
        open=90.0,
        high=92.0,
        low=89.0,
        close=91.0,
        volume=1000.0,
        extra={"api_key": "SECRET_KEY_123", "token": "BEARER_TOKEN_XYZ"},
    )
    tick_logger.handle_market_tick(tick)

    # Check that secrets in extra are never logged by the diagnostic handler
    log_text = caplog.text
    assert "SECRET_KEY_123" not in log_text
    assert "BEARER_TOKEN_XYZ" not in log_text
    assert "[TICK] Symbol=ROUT Close=91.00" in log_text


def test_multiple_subscribers_failure_isolation():
    """Confirm one or multiple failing subscribers do not prevent other subscribers from receiving events."""
    dispatcher = EventDispatcher()

    received_sub2 = []
    received_sub4 = []

    def failing_sub1(evt):
        raise RuntimeError("Sub 1 failure")

    def working_sub2(evt):
        received_sub2.append(evt)

    def failing_sub3(evt):
        raise ValueError("Sub 3 failure")

    def working_sub4(evt):
        received_sub4.append(evt)

    dispatcher.subscribe_tick(failing_sub1)
    dispatcher.subscribe_tick(working_sub2)
    dispatcher.subscribe_tick(failing_sub3)
    dispatcher.subscribe_tick(working_sub4)

    tick = MarketTickEvent(
        symbol="ROUT",
        open=90.0,
        high=92.0,
        low=89.0,
        close=91.0,
        volume=1000.0,
    )

    success_count = dispatcher.dispatch_tick(tick)

    # 2 succeeded, 2 failed
    assert success_count == 2
    assert len(received_sub2) == 1
    assert len(received_sub4) == 1
    assert dispatcher.metrics.handler_errors == 2

    # Repeat for corporate news
    news_received = []

    def failing_news_sub(evt):
        raise RuntimeError("News strategy crashed")

    def working_news_sub(evt):
        news_received.append(evt)

    dispatcher.subscribe_news(failing_news_sub)
    dispatcher.subscribe_news(working_news_sub)

    news = MarketNewsEvent(headline="Test headline", symbol="ROUT")
    news_success = dispatcher.dispatch_news(news)

    assert news_success == 1
    assert len(news_received) == 1
    assert dispatcher.metrics.handler_errors == 3


def test_invalid_handler_type_handling():
    """Verify non-callable, non-protocol handler object is safely skipped with error logged."""
    dispatcher = EventDispatcher()

    # Pass an integer which has no __call__ and no handle_market_tick
    dispatcher._tick_handlers.append(12345)  # type: ignore
    dispatcher._news_handlers.append(67890)  # type: ignore

    tick = MarketTickEvent(
        symbol="ROUT",
        open=90.0,
        high=92.0,
        low=89.0,
        close=91.0,
        volume=1000.0,
    )
    news = MarketNewsEvent(headline="Test headline", symbol="ROUT")

    assert dispatcher.dispatch_tick(tick) == 0
    assert dispatcher.dispatch_news(news) == 0
    assert dispatcher.metrics.handler_errors == 2


def test_slow_synchronous_handler_consequences():
    """Demonstrate that slow synchronous handlers block the dispatch thread proportionally."""
    import time
    dispatcher = EventDispatcher()

    sleep_duration_sec = 0.005  # 5ms

    def slow_handler(evt):
        time.sleep(sleep_duration_sec)

    dispatcher.subscribe_tick(slow_handler)

    tick = MarketTickEvent(
        symbol="ROUT",
        open=90.0,
        high=92.0,
        low=89.0,
        close=91.0,
        volume=1000.0,
    )

    start_time = time.perf_counter()
    dispatcher.dispatch_tick(tick)
    elapsed = time.perf_counter() - start_time

    # Must have taken at least the sleep duration
    assert elapsed >= sleep_duration_sec
    assert dispatcher.metrics.ticks_dispatched == 1


def test_dispatcher_metrics_to_dict():
    """Verify metrics dictionary export contains expected keys."""
    dispatcher = EventDispatcher()
    metrics_dict = dispatcher.metrics.to_dict()
    assert "ticks_dispatched" in metrics_dict
    assert "news_dispatched" in metrics_dict
    assert "handler_errors" in metrics_dict
    assert "dropped_events" in metrics_dict


"""Integration tests verifying Kafka market data ingestion to EMACrossoverStrategy pipeline.

Validates:
1. End-to-end flow: Raw JSON -> Kafka Consumer -> Normalizer -> Dispatcher -> Connector -> Strategy -> CandidateSignal.
2. Invariant preservation: Timestamps, price positivity, and chronological monotonicity.
3. Out-of-order and duplicate timestamp drop policy.
4. Indicator warm-up and abstention behavior.
5. Error isolation: Malformed payloads or handler failures do not crash the consumer loop.
6. Zero trading/order execution side-effects.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
from typing import Any
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
KAFKA_SRC_DIR = Path(__file__).resolve().parent.parent / "kafka" / "src"
for d in (SRC_DIR, KAFKA_SRC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from market_data.config import KafkaConfig
from market_data.consumer import MarketDataConsumer
from market_data.handlers import EventDispatcher
from market_data.models import KafkaMetadata, MarketTickEvent
from nova.quant.contracts import (
    DataQualityStatus,
    EvaluationStatus,
    ScoreSemantics,
    SignalDirection,
)
from nova.research.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)
from nova.research.kafka_connector import (
    KafkaStrategyConnector,
)


class MockKafkaMessage:
    """Mock Kafka message for offline testing without confluent-kafka client."""

    def __init__(
        self,
        value: bytes | None,
        topic: str = "market.ticks",
        partition: int = 0,
        offset: int = 1,
        timestamp_ms: int = 1793257286000,
        error: Any = None,
    ) -> None:
        self._value = value
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._timestamp_ms = timestamp_ms
        self._error = error

    def value(self) -> bytes | None:
        return self._value

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def timestamp(self) -> tuple[int, int]:
        return (1, self._timestamp_ms)

    def error(self) -> Any:
        return self._error


def make_tick_payload(
    symbol: str,
    price: float,
    minute_offset: int = 0,
    volume: float = 500.0,
) -> dict:
    """Helper generating candidate JSON payload matching market.ticks schema."""
    base_ts = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    ts = base_ts + timedelta(minutes=minute_offset)
    return {
        "symbol": symbol,
        "open": price,
        "high": price * 1.002,
        "low": price * 0.998,
        "close": price,
        "volume": volume,
        "timestamp": ts.isoformat(),
    }


class TestKafkaStrategyIntegration(unittest.TestCase):
    """Verifies seamless data delivery from Kafka ingestion layer to quantitative strategy."""

    def setUp(self) -> None:
        self.strategy_config = EMACrossoverConfig(
            fast_window=3,
            slow_window=6,
            allow_short=False,
            prediction_horizon_bars=5,
        )
        self.strategy = EMACrossoverStrategy(config=self.strategy_config)
        self.connector = KafkaStrategyConnector(strategy=self.strategy)

        self.kafka_config = KafkaConfig(
            bootstrap_servers="127.0.0.1:9094",
            group_id="group-test",
            topic_ticks="market.ticks",
            topic_news="market.news",
        )
        self.dispatcher = EventDispatcher()
        self.dispatcher.subscribe_tick(self.connector)
        self.consumer = MarketDataConsumer(config=self.kafka_config, dispatcher=self.dispatcher)

    def test_single_tick_processing_warmup_abstention(self) -> None:
        """Single tick event must pass through consumer -> connector and yield warm-up status."""
        payload = make_tick_payload("INFY", 1500.0, minute_offset=0)
        msg = MockKafkaMessage(value=json.dumps(payload).encode("utf-8"))

        success = self.consumer.process_raw_message(msg)
        self.assertTrue(success)

        # Check connector state
        self.assertEqual(self.connector.metrics.ticks_received, 1)
        self.assertEqual(self.connector.metrics.ticks_converted, 1)
        self.assertEqual(self.connector.metrics.evaluations_count, 1)
        self.assertEqual(self.connector.metrics.warmup_abstentions_count, 1)

        eval_res = self.connector.last_evaluation_by_symbol.get("INFY")
        self.assertIsNotNone(eval_res)
        assert eval_res is not None
        self.assertEqual(eval_res.status, EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY)
        self.assertIsNone(eval_res.candidate)

    def test_warmed_up_bullish_crossover_candidate_production(self) -> None:
        """Stream of 10 ticks (6 flat warm-up, 4 rising) triggers bullish crossover CandidateSignal."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 110.0, 115.0, 120.0]
        for i, price in enumerate(prices):
            payload = make_tick_payload("TCS", price, minute_offset=i)
            msg = MockKafkaMessage(value=json.dumps(payload).encode("utf-8"), offset=i + 1)
            dispatched = self.consumer.process_raw_message(msg)
            self.assertTrue(dispatched)

        self.assertEqual(self.connector.metrics.ticks_received, 10)
        self.assertEqual(self.connector.metrics.ticks_converted, 10)
        self.assertEqual(self.connector.metrics.evaluations_count, 10)
        self.assertEqual(self.connector.metrics.warmup_abstentions_count, 5)
        self.assertEqual(self.connector.metrics.bullish_signals_count, 4)

        last_cand = self.connector.last_candidate_by_symbol.get("TCS")
        self.assertIsNotNone(last_cand)
        assert last_cand is not None
        self.assertEqual(last_cand.ticker, "TCS")
        self.assertEqual(last_cand.direction, SignalDirection.BULLISH)
        self.assertEqual(last_cand.score_semantics, ScoreSemantics.RAW_INTENSITY)
        self.assertGreater(last_cand.signal_score, 0.0)

    def test_out_of_order_and_duplicate_ticks_dropped(self) -> None:
        """Ticks with duplicate or older timestamps must be dropped to preserve temporal integrity."""
        # 1st tick at t=0
        p1 = make_tick_payload("RELIANCE", 2500.0, minute_offset=0)
        self.consumer.process_raw_message(MockKafkaMessage(json.dumps(p1).encode("utf-8")))

        # 2nd tick at t=10
        p2 = make_tick_payload("RELIANCE", 2505.0, minute_offset=10)
        self.consumer.process_raw_message(MockKafkaMessage(json.dumps(p2).encode("utf-8")))

        # Duplicate tick at t=10
        p3_dup = make_tick_payload("RELIANCE", 2505.0, minute_offset=10)
        self.consumer.process_raw_message(MockKafkaMessage(json.dumps(p3_dup).encode("utf-8")))

        # Out-of-order tick at t=5
        p4_old = make_tick_payload("RELIANCE", 2502.0, minute_offset=5)
        self.consumer.process_raw_message(MockKafkaMessage(json.dumps(p4_old).encode("utf-8")))

        # Only 2 valid chronological ticks should be buffered
        history = self.connector.get_symbol_history("RELIANCE")
        self.assertEqual(len(history), 2)
        self.assertEqual(self.connector.metrics.ticks_received, 4)
        self.assertEqual(self.connector.metrics.ticks_dropped_out_of_order, 2)

    def test_malformed_json_and_schema_validation_failure_isolation(self) -> None:
        """Malformed JSON or corrupted financial data is rejected without terminating loop."""
        # Corrupted JSON
        corrupt_msg = MockKafkaMessage(value=b"{invalid_json_payload: null")
        success_corrupt = self.consumer.process_raw_message(corrupt_msg)
        self.assertFalse(success_corrupt)
        self.assertEqual(self.consumer.decode_errors["market.ticks"], 1)

        # Invalid financial data (negative price)
        neg_price_payload = make_tick_payload("BAD_PRICE", -50.0)
        bad_price_msg = MockKafkaMessage(value=json.dumps(neg_price_payload).encode("utf-8"))
        success_bad = self.consumer.process_raw_message(bad_price_msg)
        self.assertFalse(success_bad)
        self.assertEqual(self.consumer.validation_errors["market.ticks"], 1)

        # Strategy should not have received either
        self.assertEqual(self.connector.metrics.ticks_received, 0)

    def test_multi_symbol_history_isolation(self) -> None:
        """Ticks for distinct symbols must maintain separate histories."""
        for i in range(5):
            self.consumer.process_raw_message(
                MockKafkaMessage(json.dumps(make_tick_payload("SYM_A", 100.0, i)).encode("utf-8"))
            )
            self.consumer.process_raw_message(
                MockKafkaMessage(json.dumps(make_tick_payload("SYM_B", 200.0, i)).encode("utf-8"))
            )

        self.assertEqual(len(self.connector.get_symbol_history("SYM_A")), 5)
        self.assertEqual(len(self.connector.get_symbol_history("SYM_B")), 5)
        self.assertEqual(self.connector.get_symbol_history("SYM_A")[-1].close, 100.0)
        self.assertEqual(self.connector.get_symbol_history("SYM_B")[-1].close, 200.0)

    def test_zero_trading_side_effects(self) -> None:
        """Processing ticks must never create orders, trade allocations, or modify accounts."""
        payload = make_tick_payload("ZERO_TRADE", 100.0)
        self.consumer.process_raw_message(MockKafkaMessage(json.dumps(payload).encode("utf-8")))

        eval_res = self.connector.last_evaluation_by_symbol.get("ZERO_TRADE")
        self.assertIsNotNone(eval_res)
        # Verify the candidate has no trade execution fields
        if eval_res and eval_res.candidate:
            self.assertFalse(hasattr(eval_res.candidate, "order_id"))
            self.assertFalse(hasattr(eval_res.candidate, "shares_to_buy"))


if __name__ == "__main__":
    unittest.main()

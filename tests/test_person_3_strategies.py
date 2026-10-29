"""Unit tests for Person 3 Strategies (TrendFollowing and MeanReversion)."""

from datetime import datetime, timedelta, timezone
import unittest

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    TimestampSemantics,
)
from nova.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy, compute_causal_rsi
from nova.strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy


def create_synthetic_bars(
    prices: list[float],
    ticker: str = "TEST_ASSET",
    quality: DataQualityStatus = DataQualityStatus.VALID,
) -> list[MarketBar]:
    base_time = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars: list[MarketBar] = []
    for i, p in enumerate(prices):
        t = base_time + timedelta(minutes=5 * i)
        bars.append(
            MarketBar(
                ticker=ticker,
                open=p * 0.999,
                high=p * 1.002,
                low=p * 0.998,
                close=p,
                volume=1000.0,
                source_timestamp=t,
                available_timestamp=t,
                timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                data_quality=quality,
            )
        )
    return bars


class TestTrendFollowingStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TrendFollowingConfig(
            fast_window=4,
            slow_window=10,
            momentum_window=3,
            momentum_threshold=0.001,
        )
        self.strategy = TrendFollowingStrategy(self.config)

    def test_warmup_abstention(self) -> None:
        bars = create_synthetic_bars([100.0 + i for i in range(5)])
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.NEUTRAL)
        self.assertEqual(signal.raw_score, 0.0)
        self.assertTrue(any("INSUFFICIENT_HISTORY" in r for r in signal.reason_codes))

    def test_bullish_trend_and_momentum(self) -> None:
        # Strong upward trajectory
        prices = [100.0 + (i * 2.0) for i in range(25)]
        bars = create_synthetic_bars(prices)
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.BULLISH)
        self.assertGreater(signal.raw_score, 0.2)
        self.assertGreater(signal.confidence, 0.5)
        self.assertIsNotNone(signal.expected_gross_return)
        self.assertGreater(signal.expected_gross_return, 0.0)
        self.assertTrue(any("BULLISH_TREND_AND_MOMENTUM_CONFIRMED" in r for r in signal.reason_codes))

    def test_bearish_trend_and_momentum(self) -> None:
        # Strong downward trajectory
        prices = [200.0 - (i * 2.5) for i in range(25)]
        bars = create_synthetic_bars(prices)
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.BEARISH)
        self.assertLess(signal.raw_score, -0.2)
        self.assertTrue(any("BEARISH_TREND_AND_MOMENTUM_CONFIRMED" in r for r in signal.reason_codes))

    def test_invalid_data_quality_handling(self) -> None:
        prices = [100.0 + i for i in range(20)]
        bars = create_synthetic_bars(prices, quality=DataQualityStatus.INVALID)
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.NEUTRAL)
        self.assertEqual(signal.raw_score, 0.0)
        self.assertTrue(any("DATA_QUALITY_NOT_VALID" in r for r in signal.reason_codes))


class TestMeanReversionStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MeanReversionConfig(
            lookback_window=10,
            z_score_threshold=1.5,
            rsi_period=7,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
        )
        self.strategy = MeanReversionStrategy(self.config)

    def test_warmup_abstention(self) -> None:
        bars = create_synthetic_bars([100.0 for _ in range(5)])
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.NEUTRAL)
        self.assertTrue(any("INSUFFICIENT_HISTORY" in r for r in signal.reason_codes))

    def test_oversold_dislocation_bullish(self) -> None:
        # Stable prices then sudden sharp drop to trigger oversold RSI and negative z-score
        prices = [100.0] * 15 + [90.0, 85.0, 80.0, 75.0]
        bars = create_synthetic_bars(prices)
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.BULLISH)
        self.assertGreater(signal.raw_score, 0.0)
        self.assertTrue(any("OVERSOLD_DISLOCATION" in r for r in signal.reason_codes))

    def test_overbought_dislocation_bearish(self) -> None:
        # Stable prices then sudden sharp surge to trigger overbought RSI and positive z-score
        prices = [100.0] * 15 + [115.0, 125.0, 135.0, 145.0]
        bars = create_synthetic_bars(prices)
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.BEARISH)
        self.assertLess(signal.raw_score, 0.0)
        self.assertTrue(any("OVERBOUGHT_DISLOCATION" in r for r in signal.reason_codes))

    def test_within_normal_bounds(self) -> None:
        # Small noise around 100
        prices = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(20)]
        bars = create_synthetic_bars(prices)
        signal = self.strategy.evaluate(bars)
        self.assertEqual(signal.direction, SignalDirection.NEUTRAL)
        self.assertEqual(signal.raw_score, 0.0)
        self.assertTrue(any("WITHIN_NORMAL_BOUNDS" in r for r in signal.reason_codes))


if __name__ == "__main__":
    unittest.main()

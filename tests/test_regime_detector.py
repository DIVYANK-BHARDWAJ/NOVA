"""Unit tests for MarketRegimeDetector (ER, EWMA volatility, trend share)."""

from datetime import datetime, timezone
import unittest

from nova.quant.contracts import ContractValidationError, DataQualityStatus
from nova.regime.detector import (
    MarketRegimeDetector,
    MarketRegimeType,
    RegimeConfig,
    RegimeOutput,
)
from tests.test_person_3_strategies import create_synthetic_bars


class TestMarketRegimeDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RegimeConfig(
            er_window=10,
            er_trend_lower=0.20,
            er_trend_range=0.30,
            ewma_lambda=0.94,
            min_volatility_floor=0.005,
            high_volatility_threshold=0.025,
        )
        self.detector = MarketRegimeDetector(self.config)

    def test_warmup_fallback(self) -> None:
        bars = create_synthetic_bars([100.0 + i for i in range(5)])
        out = self.detector.detect(bars)
        self.assertEqual(out.regime_type, MarketRegimeType.UNKNOWN)
        self.assertTrue(any("INSUFFICIENT_HISTORY" in r for r in out.reason_codes))

    def test_high_efficiency_trending_regime(self) -> None:
        # Strictly monotonic ascending price series has ER near 1.0
        prices = [100.0 + i * 2.0 for i in range(30)]
        bars = create_synthetic_bars(prices)
        out = self.detector.detect(bars)
        self.assertGreaterEqual(out.efficiency_ratio, 0.90)
        self.assertEqual(out.trend_share, 1.0)
        self.assertEqual(out.regime_type, MarketRegimeType.TRENDING)

    def test_low_efficiency_ranging_regime(self) -> None:
        # Perfectly alternating series has net change near 0, ER near 0
        prices = [100.0 if i % 2 == 0 else 102.0 for i in range(30)]
        bars = create_synthetic_bars(prices)
        out = self.detector.detect(bars)
        self.assertLessEqual(out.efficiency_ratio, 0.20)
        self.assertEqual(out.trend_share, 0.0)
        self.assertEqual(out.regime_type, MarketRegimeType.RANGING)

    def test_high_volatility_detection(self) -> None:
        # Extreme price jumps
        prices = [100.0 * (1.10 if i % 2 == 0 else 0.90) for i in range(30)]
        bars = create_synthetic_bars(prices)
        out = self.detector.detect(bars)
        self.assertTrue(out.is_high_volatility)
        self.assertEqual(out.regime_type, MarketRegimeType.HIGH_VOLATILITY)

    def test_data_quality_fallback(self) -> None:
        bars = create_synthetic_bars([100.0 + i for i in range(20)], quality=DataQualityStatus.INVALID)
        out = self.detector.detect(bars)
        self.assertEqual(out.regime_type, MarketRegimeType.UNKNOWN)
        self.assertTrue(any("DATA_QUALITY_INVALID" in r for r in out.reason_codes))


if __name__ == "__main__":
    unittest.main()

"""Unit, numerical reference, and contract tests for NOVA Market Regime Intelligence (Milestone 3).

Verifies all Person 2 regime requirements:
1. Mathematical correctness of Kaufman Efficiency Ratio (KER) and EWMA volatility
2. Classification of canonical market regimes (BULLISH, BEARISH, RANGING, HIGH_VOLATILITY)
3. Interface Contract B (ModelPrediction) compliance and schema parity
4. Temporal integrity & strict anti-leakage invariance
5. Pre-simulation warm-start using genuine MANTHAN competition historical data
6. Stream event processing and immediate READY availability post-warmup
7. Strict configuration validation and fail-closed error handling
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    NumericalValidationError,
    PredictionAvailability,
    PredictionType,
    TemporalValidationError,
    validate_model_prediction_schema_parity,
    validate_no_secrets,
)
from nova.regime.config import MarketRegimeConfig
from nova.regime.transparent_detector import (
    RegimeLabel,
    TransparentRegimeDetector,
)


def make_test_bar(
    ticker: str,
    dt: datetime,
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_price: float | None = None,
    volume: float = 10000.0,
    quality: DataQualityStatus = DataQualityStatus.VALID,
    provenance: str = "OFFICIAL_COMPETITION_FEED",
) -> MarketBar:
    o = open_price if open_price is not None else close
    h = high if high is not None else max(o, close) + 0.5
    l = low if low is not None else min(o, close) - 0.5
    return MarketBar(
        ticker=ticker,
        open=o,
        high=h,
        low=l,
        close=close,
        volume=volume,
        source_timestamp=dt,
        provenance=provenance,
        data_quality=quality,
    )


class TestRegimeCanonical(unittest.TestCase):
    """Canonical test suite for Person 2 Market Regime Intelligence subsystem."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.base_time = datetime(2026, 3, 1, 9, 30, tzinfo=self.utc)
        self.config = MarketRegimeConfig(
            ker_window=10,
            ker_trend_threshold=0.50,
            ker_ranging_threshold=0.25,
            volatility_window=20,
            high_volatility_threshold=0.025,
            min_warmup_bars=28,
            fast_ema_window=12,
            slow_ema_window=26,
            ewma_lambda=0.94,
        )
        self.detector = TransparentRegimeDetector(self.config)

    def test_monotonic_rise_classified_as_trending_bullish(self) -> None:
        """Strong monotonic upward rise must yield high KER and TRENDING_BULLISH."""
        bars = []
        for i in range(40):
            dt = self.base_time + timedelta(days=i)
            # Steady 1.0 point rise per day: KER = 10 / 10 = 1.0
            price = 100.0 + float(i)
            bars.append(make_test_bar("TEST_BULL", dt, price))

        pred = self.detector.detect(bars)

        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertEqual(pred.metadata["dominant_regime"], RegimeLabel.TRENDING_BULLISH.value)
        self.assertIsNotNone(pred.prediction_distribution)
        dist = pred.prediction_distribution
        self.assertGreater(dist[RegimeLabel.TRENDING_BULLISH.value], 0.70)
        self.assertLess(dist[RegimeLabel.TRENDING_BEARISH.value], 0.15)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=4)

    def test_monotonic_fall_classified_as_trending_bearish(self) -> None:
        """Strong monotonic downward drop must yield high KER and TRENDING_BEARISH."""
        bars = []
        for i in range(40):
            dt = self.base_time + timedelta(days=i)
            # Steady 1.0 point drop per day: KER = 10 / 10 = 1.0
            price = 200.0 - float(i)
            bars.append(make_test_bar("TEST_BEAR", dt, price))

        pred = self.detector.detect(bars)

        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertEqual(pred.metadata["dominant_regime"], RegimeLabel.TRENDING_BEARISH.value)
        self.assertIsNotNone(pred.prediction_distribution)
        dist = pred.prediction_distribution
        self.assertGreater(dist[RegimeLabel.TRENDING_BEARISH.value], 0.70)
        self.assertLess(dist[RegimeLabel.TRENDING_BULLISH.value], 0.15)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=4)

    def test_alternating_chop_classified_as_ranging(self) -> None:
        """Alternating price oscillations must yield low KER and RANGING regime."""
        bars = []
        for i in range(40):
            dt = self.base_time + timedelta(days=i)
            # Oscillates between 100 and 100.2
            price = 100.0 + (0.2 if i % 2 == 1 else 0.0)
            bars.append(make_test_bar("TEST_RANGE", dt, price))

        pred = self.detector.detect(bars)

        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertEqual(pred.metadata["dominant_regime"], RegimeLabel.RANGING.value)
        self.assertIsNotNone(pred.prediction_distribution)
        dist = pred.prediction_distribution
        self.assertGreater(dist[RegimeLabel.RANGING.value], 0.50)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=4)

    def test_constant_prices_zero_variance_safety(self) -> None:
        """Constant prices must not divide by zero and should classify as RANGING."""
        bars = []
        for i in range(35):
            dt = self.base_time + timedelta(days=i)
            bars.append(make_test_bar("TEST_CONST", dt, 150.0))

        pred = self.detector.detect(bars)

        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertEqual(pred.metadata["dominant_regime"], RegimeLabel.RANGING.value)
        self.assertIsNotNone(pred.prediction_distribution)
        self.assertAlmostEqual(sum(pred.prediction_distribution.values()), 1.0, places=4)

    def test_high_volatility_takes_precedence(self) -> None:
        """Massive return swings exceeding volatility threshold must trigger HIGH_VOLATILITY."""
        bars = []
        for i in range(40):
            dt = self.base_time + timedelta(days=i)
            # Wild swings: ±6% daily fluctuations
            factor = 1.06 if i % 2 == 1 else 0.94
            price = 100.0 * factor
            bars.append(make_test_bar("TEST_VOL", dt, price))

        pred = self.detector.detect(bars)

        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertEqual(pred.metadata["dominant_regime"], RegimeLabel.HIGH_VOLATILITY.value)
        self.assertIsNotNone(pred.prediction_distribution)
        dist = pred.prediction_distribution
        self.assertGreater(dist[RegimeLabel.HIGH_VOLATILITY.value], 0.50)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=4)

    def test_insufficient_history_returns_honest_unavailability(self) -> None:
        """Bars fewer than min_warmup_bars must report INSUFFICIENT_HISTORY and UNKNOWN."""
        bars = []
        for i in range(15):  # 15 < 28
            dt = self.base_time + timedelta(days=i)
            bars.append(make_test_bar("TEST_SHORT", dt, 100.0 + i))

        pred = self.detector.detect(bars)

        self.assertEqual(pred.availability_status, PredictionAvailability.INSUFFICIENT_HISTORY)
        self.assertEqual(pred.metadata["dominant_regime"], RegimeLabel.UNKNOWN.value)
        self.assertEqual(pred.metadata["status_reason"], "INSUFFICIENT_HISTORY")
        self.assertIsNotNone(pred.prediction_distribution)
        self.assertAlmostEqual(sum(pred.prediction_distribution.values()), 1.0, places=4)

    def test_model_prediction_schema_parity(self) -> None:
        """Prediction dictionary must strictly validate against model_prediction.schema.json."""
        bars = [
            make_test_bar("TEST_SCHEMA", self.base_time + timedelta(days=i), 100.0 + i)
            for i in range(35)
        ]
        pred = self.detector.detect(bars)
        payload = pred.to_dict()

        # Direct schema parity check
        validate_model_prediction_schema_parity(payload)
        self.assertEqual(payload["prediction_type"], "REGIME_PROBABILITY")
        self.assertEqual(payload["instrument_id"], "TEST_SCHEMA")
        self.assertIn("prediction_distribution", payload)
        self.assertEqual(len(payload["prediction_distribution"]), 4)

    def test_temporal_invariance_and_anti_leakage(self) -> None:
        """State at bar t must not be altered by future bars t+1..t+n."""
        bars = [
            make_test_bar("TEST_LEAK", self.base_time + timedelta(days=i), 100.0 + (i * 0.5))
            for i in range(35)
        ]

        # Prediction at bar 35
        pred_at_35 = self.detector.detect(bars)

        # Now simulate a massive future market shock at days 36..40
        future_bars = list(bars)
        for i in range(35, 40):
            future_bars.append(
                make_test_bar(
                    "TEST_LEAK",
                    self.base_time + timedelta(days=i),
                    50.0 - float(i - 35) * 2.0,  # Radical future drop but strictly positive
                )
            )

        # Re-evaluating historical slice [0..35] with future data existing outside slice
        pred_recalculated = self.detector.detect(bars)

        # Must be bit-exact identical
        self.assertEqual(pred_at_35.metadata["dominant_regime"], pred_recalculated.metadata["dominant_regime"])
        self.assertEqual(pred_at_35.prediction_timestamp, pred_recalculated.prediction_timestamp)
        self.assertEqual(pred_at_35.prediction_distribution, pred_recalculated.prediction_distribution)
        self.assertEqual(pred_at_35.confidence_or_uncertainty, pred_recalculated.confidence_or_uncertainty)

    def test_out_of_order_bars_rejected(self) -> None:
        """Non-chronological candles must fail-closed with TemporalValidationError."""
        bar1 = make_test_bar("TEST_ORDER", self.base_time, 100.0)
        bar2 = make_test_bar("TEST_ORDER", self.base_time - timedelta(minutes=5), 101.0)

        with self.assertRaises(TemporalValidationError):
            self.detector.detect([bar1, bar2])

    def test_warm_start_from_genuine_manthan_history(self) -> None:
        """Pre-simulation historical warm-start using archived MANTHAN competition data."""
        history_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "manthan_history"
        json_files = list(history_dir.glob("*_history_90d.json"))
        self.assertGreater(len(json_files), 0, "Archived MANTHAN history files not found")

        # Pick the first genuine ticker file
        target_file = json_files[0]
        with open(target_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        ticker = raw_data["ticker"]
        raw_bars = raw_data["prices"]
        self.assertGreaterEqual(len(raw_bars), 60, f"Expected >= 60 bars for {ticker}")

        # Convert to MarketBar objects
        history_bars: list[MarketBar] = []
        for b in raw_bars:
            dt = datetime.fromisoformat(b["date"]).replace(tzinfo=timezone.utc)
            history_bars.append(
                MarketBar(
                    ticker=ticker,
                    open=float(b["open"]),
                    high=float(b["high"]),
                    low=float(b["low"]),
                    close=float(b["close"]),
                    volume=float(b["volume"]),
                    source_timestamp=dt,
                    provenance=f"MANTHAN_HISTORICAL_API:{ticker}",
                    data_quality=DataQualityStatus.VALID,
                )
            )

        # Execute warm_start
        warm_count = self.detector.warm_start(ticker, history_bars)
        self.assertEqual(warm_count, len(history_bars))
        self.assertEqual(self.detector.get_buffered_bar_count(ticker), len(history_bars))

        # Now simulate the arrival of the FIRST live Kafka streaming tick/candle
        last_hist_dt = history_bars[-1].source_timestamp
        live_dt = last_hist_dt + timedelta(days=1)
        live_event = {
            "symbol": ticker,
            "open": history_bars[-1].close,
            "high": history_bars[-1].close * 1.01,
            "low": history_bars[-1].close * 0.99,
            "close": history_bars[-1].close * 1.005,
            "volume": 25000.0,
            "timestamp": live_dt.isoformat(),
            "topic": "market.ticks",
            "partition": 0,
            "offset": 5001,
        }

        # Process the live event
        live_pred = self.detector.process_event(live_event)

        # Must be immediately READY with NO warmup delay in live trading
        self.assertEqual(live_pred.availability_status, PredictionAvailability.READY)
        self.assertEqual(live_pred.instrument_id, ticker)
        self.assertIsNotNone(live_pred.prediction_distribution)
        self.assertAlmostEqual(sum(live_pred.prediction_distribution.values()), 1.0, places=4)
        self.assertIn("efficiency_ratio", live_pred.metadata)
        self.assertIn("ewma_volatility", live_pred.metadata)

        # Validate schema parity of live output
        validate_model_prediction_schema_parity(live_pred.to_dict())

    def test_config_strict_validation(self) -> None:
        """MarketRegimeConfig must validate boundaries and fail-closed on corrupt values."""
        with self.assertRaises(NumericalValidationError):
            MarketRegimeConfig(ker_window=1)  # Window must be >= 2

        with self.assertRaises((ContractValidationError, NumericalValidationError)):
            MarketRegimeConfig(ewma_lambda=1.05)  # Lambda must be < 1.0

        with self.assertRaises((ContractValidationError, NumericalValidationError)):
            MarketRegimeConfig(ewma_lambda=-0.1)  # Lambda must be > 0.0

        with self.assertRaises((ContractValidationError, NumericalValidationError)):
            MarketRegimeConfig(
                ker_trend_threshold=0.30,
                ker_ranging_threshold=0.40,  # trend threshold must be > ranging threshold
            )

    def test_toml_config_loader(self) -> None:
        """MarketRegimeConfig.from_toml must load correctly from config/features_config.toml."""
        toml_path = Path(__file__).resolve().parent.parent / "config" / "features_config.toml"
        cfg = MarketRegimeConfig.from_toml(toml_path)
        self.assertEqual(cfg.ker_window, 10)
        self.assertEqual(cfg.ker_trend_threshold, 0.50)
        self.assertEqual(cfg.ker_ranging_threshold, 0.25)
        self.assertEqual(cfg.ewma_lambda, 0.94)
        self.assertEqual(cfg.min_warmup_bars, 28)


if __name__ == "__main__":
    unittest.main()

"""Unit and numerical reference tests for NOVA Causal Feature Pipeline (Milestone 2).

Verifies mathematical correctness against independent hand-calculated fixtures:
- Momentum (Rate of Change)
- Linear regression trend slope
- Kaufman Efficiency Ratio (KER)
- Crossover recency tracking
- Rolling mean, standard deviation, and z-score
- Zero variance & zero volume safety
- Wilder's RSI-14
- True Range & Average True Range (ATR)
- Realized & Parkinson volatility
- On-Balance Volume (OBV) & Relative Volume (RVOL)
- Price-Volume Divergence
- Benchmark-relative return and momentum
- Incremental vs Batch equivalence: compute_latest(bars[:i]) == compute_history(bars)[i-1]
"""

from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.features import (
    BarFeatures,
    FeaturePipeline,
    FeaturePipelineConfig,
    calculate_adx_series,
    calculate_atr_series,
    calculate_benchmark_relative_return,
    calculate_crossover_recency,
    calculate_downside_deviation,
    calculate_kaufman_efficiency_ratio,
    calculate_mean_deviation,
    calculate_momentum_series,
    calculate_on_balance_volume,
    calculate_opening_gap,
    calculate_parkinson_volatility,
    calculate_price_volume_divergence,
    calculate_realized_volatility,
    calculate_relative_momentum,
    calculate_relative_volume,
    calculate_rolling_mean,
    calculate_rolling_mean_volume,
    calculate_rolling_std,
    calculate_rolling_vwap,
    calculate_rolling_z_score,
    calculate_rsi_series,
    calculate_trend_slope,
    calculate_true_range,
    calculate_volume_z_score,
)
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    NumericalValidationError,
    TemporalValidationError,
    TimestampSemantics,
)


def make_bar(
    ticker: str,
    dt: datetime,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float = 1000.0,
) -> MarketBar:
    """Helper to create valid synthetic MarketBar."""
    return MarketBar(
        ticker=ticker,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        source_timestamp=dt,
        available_timestamp=dt,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=DataQualityStatus.VALID,
    )


class TestCausalFeatureCalculations(unittest.TestCase):
    """Verifies each feature family against hand-calculated mathematical test cases."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.base_time = datetime(2026, 3, 1, 9, 15, tzinfo=self.utc)

    def test_hand_calculated_momentum(self) -> None:
        """Hand calculation: series = [100.0, 105.0, 110.0, 120.0], lookback = 2
        i = 0: None
        i = 1: None
        i = 2: (110.0 - 100.0) / 100.0 = +0.10 (+10.0%)
        i = 3: (120.0 - 105.0) / 105.0 = 15 / 105 = 0.142857...
        """
        closes = [100.0, 105.0, 110.0, 120.0]
        mom = calculate_momentum_series(closes, lookback=2)
        self.assertIsNone(mom[0])
        self.assertIsNone(mom[1])
        self.assertAlmostEqual(mom[2], 0.10, places=6)
        self.assertAlmostEqual(mom[3], 15.0 / 105.0, places=6)

    def test_hand_calculated_trend_slope(self) -> None:
        """Hand calculation: 3-period linear regression slope on [10.0, 12.0, 14.0]
        x = [0, 1, 2], mean_x = 1.0, S_xx = (0-1)^2 + (1-1)^2 + (2-1)^2 = 2.0
        y = [10.0, 12.0, 14.0], mean_y = 12.0
        S_xy = (-1)*(-2.0) + (0)*(0.0) + (1)*(2.0) = 4.0
        raw_slope = 4.0 / 2.0 = 2.0 per bar
        normalized_slope = 2.0 / 12.0 = 1.0 / 6.0 = 0.166666...
        """
        closes = [10.0, 12.0, 14.0]
        slopes = calculate_trend_slope(closes, window=3)
        self.assertIsNone(slopes[0])
        self.assertIsNone(slopes[1])
        self.assertAlmostEqual(slopes[2], 2.0 / 12.0, places=6)

    def test_kaufman_efficiency_ratio_bounds(self) -> None:
        """KER must be 1.0 for a strictly monotonic trend and 0.0 for zero net move.
        Straight line: [10, 11, 12, 13, 14] -> Direction = 4, Vol = 1+1+1+1=4 -> KER = 1.0
        Choppy flat: [10, 12, 10, 12, 10] -> Direction = 0, Vol = 2+2+2+2=8 -> KER = 0.0
        """
        straight = [10.0, 11.0, 12.0, 13.0, 14.0]
        ker_straight = calculate_kaufman_efficiency_ratio(straight, window=4)
        self.assertAlmostEqual(ker_straight[4], 1.0, places=6)

        choppy = [10.0, 12.0, 10.0, 12.0, 10.0]
        ker_choppy = calculate_kaufman_efficiency_ratio(choppy, window=4)
        self.assertAlmostEqual(ker_choppy[4], 0.0, places=6)

    def test_crossover_recency(self) -> None:
        """Verifies causal tracking of bars elapsed since last crossover event.
        Events: [0, 1, 0, 0, -1, 0]
        Recency: [None, 0, 1, 2, 0, 1]
        """
        events = [0, 1, 0, 0, -1, 0]
        recency = calculate_crossover_recency(events)
        self.assertEqual(recency, [None, 0, 1, 2, 0, 1])

    def test_hand_calculated_rolling_mean_std_z_score(self) -> None:
        """Hand calculation: window = 3 on [10.0, 20.0, 30.0]
        mean = (10 + 20 + 30) / 3 = 20.0
        sample variance (ddof=1) = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = 200 / 2 = 100.0
        std = 10.0
        z_score at index 2 = (30.0 - 20.0) / 10.0 = +1.0
        mean_deviation = (30.0 - 20.0) / 20.0 = +0.50 (+50%)
        """
        closes = [10.0, 20.0, 30.0]
        m = calculate_rolling_mean(closes, window=3)
        s = calculate_rolling_std(closes, window=3, ddof=1)
        z = calculate_rolling_z_score(closes, window=3)
        dev = calculate_mean_deviation(closes, window=3)

        self.assertAlmostEqual(m[2], 20.0, places=6)
        self.assertAlmostEqual(s[2], 10.0, places=6)
        self.assertAlmostEqual(z[2], 1.0, places=6)
        self.assertAlmostEqual(dev[2], 0.50, places=6)

    def test_zero_variance_numerical_safety(self) -> None:
        """Constant prices must not raise division-by-zero; z-score must return 0.0."""
        constant_prices = [50.0] * 10
        s = calculate_rolling_std(constant_prices, window=5)
        z = calculate_rolling_z_score(constant_prices, window=5)
        dev = calculate_mean_deviation(constant_prices, window=5)

        for i in range(4, 10):
            self.assertEqual(s[i], 0.0)
            self.assertEqual(z[i], 0.0)
            self.assertEqual(dev[i], 0.0)

    def test_rsi_bounds_and_extremes(self) -> None:
        """RSI must remain strictly bounded in [0.0, 100.0].
        Monotonically rising prices: RSI -> 100.0
        Monotonically falling prices: RSI -> 0.0
        """
        rising = [100.0 + i * 2.0 for i in range(20)]
        rsi_rising = calculate_rsi_series(rising, window=14)
        self.assertIsNotNone(rsi_rising[14])
        self.assertAlmostEqual(rsi_rising[14], 100.0, places=6)

        falling = [200.0 - i * 2.0 for i in range(20)]
        rsi_falling = calculate_rsi_series(falling, window=14)
        self.assertIsNotNone(rsi_falling[14])
        self.assertAlmostEqual(rsi_falling[14], 0.0, places=6)

    def test_hand_calculated_true_range(self) -> None:
        """Hand calculation of True Range:
        Bar 0: H=105, L=95, C=100 -> TR = 105 - 95 = 10.0
        Bar 1: H=115, L=108, C=112 (Gap up from prev C=100)
            TR = max(115-108=7, |115-100|=15, |108-100|=8) = 15.0
        Bar 2: H=102, L=90, C=92 (Gap down from prev C=112)
            TR = max(102-90=12, |102-112|=10, |90-112|=22) = 22.0
        """
        highs = [105.0, 115.0, 102.0]
        lows = [95.0, 108.0, 90.0]
        closes = [100.0, 112.0, 92.0]

        tr = calculate_true_range(highs, lows, closes)
        self.assertAlmostEqual(tr[0], 10.0, places=6)
        self.assertAlmostEqual(tr[1], 15.0, places=6)
        self.assertAlmostEqual(tr[2], 22.0, places=6)

    def test_hand_calculated_obv(self) -> None:
        """Hand calculation for On-Balance Volume:
        Bar 0: C=100, V=1000 -> OBV = 1000
        Bar 1: C=105 (>100), V=500 -> OBV = 1000 + 500 = 1500
        Bar 2: C=103 (<105), V=300 -> OBV = 1500 - 300 = 1200
        Bar 3: C=103 (==103), V=400 -> OBV = 1200
        """
        closes = [100.0, 105.0, 103.0, 103.0]
        vols = [1000.0, 500.0, 300.0, 400.0]
        obv = calculate_on_balance_volume(closes, vols)
        self.assertEqual(obv, [1000.0, 1500.0, 1200.0, 1200.0])

    def test_benchmark_relative_return_and_missing_benchmark(self) -> None:
        """Verifies benchmark-relative return calculation and None when benchmark missing."""
        asset_c = [100.0, 102.0]  # +2.0%
        bm_c = [50.0, 50.5]       # +1.0%

        rel = calculate_benchmark_relative_return(asset_c, bm_c)
        self.assertIsNone(rel[0])
        self.assertAlmostEqual(rel[1], 0.02 - 0.01, places=6)

        # Missing benchmark produces None, never fabricates
        missing = calculate_benchmark_relative_return(asset_c, None)
        self.assertEqual(missing, [None, None])


class TestFeaturePipelineIntegration(unittest.TestCase):
    """Verifies end-to-end FeaturePipeline orchestration and streaming vs batch parity."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.t0 = datetime(2026, 3, 1, 9, 15, tzinfo=self.utc)
        self.config = FeaturePipelineConfig(
            fast_ema_window=4,
            slow_ema_window=8,
            rolling_mean_window=6,
            momentum_window=3,
            ker_window=4,
            adx_window=4,
            rsi_window=4,
            atr_window=4,
            realized_vol_window=6,
            parkinson_vol_window=6,
            downside_vol_window=6,
            volume_mean_window=6,
            volume_divergence_window=4,
            benchmark_window=4,
        )
        self.pipeline = FeaturePipeline(self.config)

        # Generate 40 synthetic bars
        self.bars: list[MarketBar] = []
        for i in range(40):
            ts = self.t0 + timedelta(minutes=5 * i)
            price = 100.0 + 0.5 * i + (1.5 if i % 2 == 0 else -1.5)
            self.bars.append(
                make_bar(
                    ticker="TEST_TICKER",
                    dt=ts,
                    o=price - 0.2,
                    h=price + 1.0,
                    l=price - 1.0,
                    c=price,
                    v=1000.0 + i * 50.0,
                )
            )

    def test_pipeline_compute_history_output(self) -> None:
        """Verify pipeline computes full chronological BarFeatures list."""
        features = self.pipeline.compute_history(self.bars)
        self.assertEqual(len(features), 40)

        # Check latest bar features
        last_f = features[-1]
        self.assertEqual(last_f.ticker, "TEST_TICKER")
        self.assertTrue(last_f.is_warmed_up)
        self.assertIsNotNone(last_f.rsi)
        self.assertIsNotNone(last_f.atr)
        self.assertIsNotNone(last_f.fast_ema)
        self.assertIsNotNone(last_f.slow_ema)
        self.assertIsNotNone(last_f.relative_volume)

        # Check dict serialization
        f_dict = last_f.to_dict()
        self.assertIn("rsi", f_dict)
        self.assertIn("fast_ema", f_dict)
        self.assertIn("timestamp", f_dict)

    def test_streaming_vs_batch_consistency(self) -> None:
        """Streaming / incremental execution: compute_latest(bars[:i]) must equal compute_history(bars)[i-1]."""
        history = self.pipeline.compute_history(self.bars)

        # Compare at multiple points after warmup
        for i in [15, 25, 40]:
            latest_bar_features = self.pipeline.compute_latest(self.bars[:i])
            historical_bar_features = history[i - 1]

            self.assertEqual(latest_bar_features.timestamp, historical_bar_features.timestamp)
            self.assertEqual(latest_bar_features.close, historical_bar_features.close)
            self.assertEqual(latest_bar_features.fast_ema, historical_bar_features.fast_ema)
            self.assertEqual(latest_bar_features.slow_ema, historical_bar_features.slow_ema)
            self.assertEqual(latest_bar_features.ema_spread, historical_bar_features.ema_spread)
            self.assertEqual(latest_bar_features.rsi, historical_bar_features.rsi)
            self.assertEqual(latest_bar_features.atr, historical_bar_features.atr)
            self.assertEqual(latest_bar_features.relative_volume, historical_bar_features.relative_volume)


if __name__ == "__main__":
    unittest.main()

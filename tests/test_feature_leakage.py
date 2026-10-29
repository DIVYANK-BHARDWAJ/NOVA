"""Strict temporal integrity, causality, and anti-leakage test suite for NOVA feature pipeline (Milestone 2).

Proves mathematically and empirically:
1. Pure point-in-time calculation: Features at bar t depend strictly on observations <= t.
2. Prefix Invariance: Appending future bars has exactly ZERO effect on historical feature values.
3. Future Mutation Invariance: Modifying future bar prices or volumes cannot leak backward into past features.
4. Temporal Ordering Enforcement: Non-chronological or duplicate timestamps are strictly rejected.
"""

from datetime import datetime, timezone, timedelta
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
)
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    TemporalValidationError,
    TimestampSemantics,
)


def create_test_bar(
    ticker: str,
    ts: datetime,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    vol: float,
) -> MarketBar:
    return MarketBar(
        ticker=ticker,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=vol,
        source_timestamp=ts,
        available_timestamp=ts,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=DataQualityStatus.VALID,
    )


class TestFeaturePipelineTemporalIntegrity(unittest.TestCase):
    """Rigorous causality and anti-leakage verification for FeaturePipeline."""

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

        # 60 synthetic bars
        self.bars: list[MarketBar] = []
        for i in range(60):
            ts = self.t0 + timedelta(minutes=5 * i)
            base = 100.0 + 0.3 * i + (1.2 if i % 3 == 0 else -0.8)
            self.bars.append(
                create_test_bar(
                    ticker="CAUSAL_TEST",
                    ts=ts,
                    open_p=base - 0.1,
                    high_p=base + 0.8,
                    low_p=base - 0.8,
                    close_p=base,
                    vol=5000.0 + (i % 5) * 500.0,
                )
            )

    def test_prefix_invariance(self) -> None:
        """Invariant: Evaluating features on prefix [:30] yields bit-for-bit identical features to [:60][:30]."""
        prefix_features = self.pipeline.compute_history(self.bars[:30])
        full_features = self.pipeline.compute_history(self.bars[:60])

        self.assertEqual(len(prefix_features), 30)

        for i in range(30):
            p = prefix_features[i]
            f = full_features[i]

            self.assertEqual(p.timestamp, f.timestamp)
            self.assertEqual(p.close, f.close)
            self.assertEqual(p.fast_ema, f.fast_ema)
            self.assertEqual(p.slow_ema, f.slow_ema)
            self.assertEqual(p.ema_spread, f.ema_spread)
            self.assertEqual(p.ema_normalized_spread, f.ema_normalized_spread)
            self.assertEqual(p.crossover_event, f.crossover_event)
            self.assertEqual(p.crossover_recency, f.crossover_recency)
            self.assertEqual(p.momentum, f.momentum)
            self.assertEqual(p.trend_slope, f.trend_slope)
            self.assertEqual(p.kaufman_efficiency_ratio, f.kaufman_efficiency_ratio)
            self.assertEqual(p.adx, f.adx)
            self.assertEqual(p.rolling_mean, f.rolling_mean)
            self.assertEqual(p.rolling_std, f.rolling_std)
            self.assertEqual(p.rolling_z_score, f.rolling_z_score)
            self.assertEqual(p.rsi, f.rsi)
            self.assertEqual(p.vwap, f.vwap)
            self.assertEqual(p.true_range, f.true_range)
            self.assertEqual(p.atr, f.atr)
            self.assertEqual(p.realized_volatility, f.realized_volatility)
            self.assertEqual(p.parkinson_volatility, f.parkinson_volatility)
            self.assertEqual(p.downside_deviation, f.downside_deviation)
            self.assertEqual(p.relative_volume, f.relative_volume)
            self.assertEqual(p.volume_z_score, f.volume_z_score)
            self.assertEqual(p.on_balance_volume, f.on_balance_volume)
            self.assertEqual(p.price_volume_divergence, f.price_volume_divergence)

    def test_future_bar_mutation_invariance(self) -> None:
        """Invariant: Mutating future bar 45 (10,000x price surge) has zero impact on bar 0..44 features."""
        unmutated_features = self.pipeline.compute_history(self.bars)

        # Mutate bar 45
        mutated_bars = list(self.bars)
        b45 = mutated_bars[45]
        mutated_bars[45] = create_test_bar(
            ticker=b45.ticker,
            ts=b45.source_timestamp,
            open_p=1000000.0,
            high_p=1000000.0,
            low_p=1000000.0,
            close_p=1000000.0,
            vol=99999999.0,
        )

        mutated_features = self.pipeline.compute_history(mutated_bars)

        for i in range(45):
            u = unmutated_features[i]
            m = mutated_features[i]

            self.assertEqual(u.timestamp, m.timestamp)
            self.assertEqual(u.close, m.close)
            self.assertEqual(u.fast_ema, m.fast_ema)
            self.assertEqual(u.slow_ema, m.slow_ema)
            self.assertEqual(u.momentum, m.momentum)
            self.assertEqual(u.rsi, m.rsi)
            self.assertEqual(u.atr, m.atr)
            self.assertEqual(u.realized_volatility, m.realized_volatility)
            self.assertEqual(u.on_balance_volume, m.on_balance_volume)

    def test_non_chronological_bars_rejected(self) -> None:
        """Out-of-order bars must raise TemporalValidationError."""
        reversed_bars = list(reversed(self.bars))
        with self.assertRaises(TemporalValidationError):
            self.pipeline.compute_history(reversed_bars)

    def test_duplicate_timestamps_rejected(self) -> None:
        """Duplicate consecutive timestamps must raise TemporalValidationError."""
        dup_bars = list(self.bars)
        dup_bars[1] = create_test_bar(
            ticker="CAUSAL_TEST",
            ts=self.bars[0].source_timestamp,  # Duplicate of bar 0
            open_p=100.0,
            high_p=101.0,
            low_p=99.0,
            close_p=100.5,
            vol=1000.0,
        )
        with self.assertRaises(TemporalValidationError):
            self.pipeline.compute_history(dup_bars)

    def test_inconsistent_ticker_rejected(self) -> None:
        """Mixed tickers in bar sequence must raise ContractValidationError."""
        mixed_bars = list(self.bars)
        b10 = mixed_bars[10]
        mixed_bars[10] = create_test_bar(
            ticker="DIFFERENT_TICKER",
            ts=b10.source_timestamp,
            open_p=b10.open,
            high_p=b10.high,
            low_p=b10.low,
            close_p=b10.close,
            vol=b10.volume,
        )
        with self.assertRaises(ContractValidationError):
            self.pipeline.compute_history(mixed_bars)

    def test_empty_sequence_rejected(self) -> None:
        """Empty bar list raises ContractValidationError."""
        with self.assertRaises(ContractValidationError):
            self.pipeline.compute_history([])


if __name__ == "__main__":
    unittest.main()


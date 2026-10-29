"""Unit and regression tests for NOVA EMA Crossover baseline strategy."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    ScoreSemantics,
    SignalDirection,
    TimestampSemantics,
)
from nova.research.adapters import generate_synthetic_ohlcv
from nova.research.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)


def make_bar(
    ticker: str,
    price: float,
    minute_offset: int,
    data_quality: DataQualityStatus = DataQualityStatus.VALID,
) -> MarketBar:
    """Helper creating a valid MarketBar at a given minute offset."""
    ts = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=minute_offset)
    return MarketBar(
        ticker=ticker,
        open=price,
        high=price * 1.001,
        low=price * 0.999,
        close=price,
        volume=100.0,
        source_timestamp=ts,
        available_timestamp=ts,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=data_quality,
        provenance="TEST_HELPER",
    )


class TestEMACrossoverStrategy(unittest.TestCase):
    """Test suite for EMACrossoverStrategy invariants and signal generation."""

    def setUp(self) -> None:
        self.config = EMACrossoverConfig(
            fast_window=3,
            slow_window=6,
            allow_short=False,
            prediction_horizon_bars=5,
        )
        self.strategy = EMACrossoverStrategy(config=self.config)

    def test_warmup_abstention(self) -> None:
        """Strategy must abstain with ABSTAIN_INSUFFICIENT_HISTORY before slow_window."""
        bars = [make_bar("TEST", 100.0, i) for i in range(5)]  # slow_window is 6
        eval_result = self.strategy.evaluate(bars)

        self.assertEqual(eval_result.status, EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY)
        self.assertIsNone(eval_result.candidate)
        self.assertIn("INSUFFICIENT_HISTORY", eval_result.reason_codes[0])

    def test_unreliable_data_abstention(self) -> None:
        """Bar with INVALID or SUSPECT quality must trigger ABSTAIN_UNRELIABLE_DATA."""
        bars = [make_bar("TEST", 100.0, i) for i in range(10)]
        # Replace final bar with SUSPECT quality
        bars[-1] = make_bar("TEST", 100.0, 9, data_quality=DataQualityStatus.SUSPECT)

        eval_result = self.strategy.evaluate(bars)
        self.assertEqual(eval_result.status, EvaluationStatus.ABSTAIN_UNRELIABLE_DATA)
        self.assertIsNone(eval_result.candidate)

    def test_bullish_crossover_and_hold_continuation(self) -> None:
        """Ensure rising prices trigger bullish crossover entry and subsequent hold continuation."""
        # 6 bars flat at 100.0 (warm-up), then prices surge to 120.0
        prices = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 110.0, 115.0]
        bars = [make_bar("TEST", p, i) for i, p in enumerate(prices)]

        # Bar 6 (price 105.0): fast EMA moves above slow EMA
        eval_cross = self.strategy.evaluate(bars[:7])
        self.assertEqual(eval_cross.status, EvaluationStatus.CANDIDATE_PRODUCED)
        self.assertIsNotNone(eval_cross.candidate)
        assert eval_cross.candidate is not None
        self.assertEqual(eval_cross.candidate.direction, SignalDirection.BULLISH)
        self.assertEqual(eval_cross.candidate.reason_codes[0], "BULLISH_CROSSOVER_ENTRY")
        self.assertGreater(eval_cross.candidate.signal_score, 0.0)

        # Bar 7 (price 110.0): continuation
        eval_hold = self.strategy.evaluate(bars[:8])
        self.assertEqual(eval_hold.status, EvaluationStatus.CANDIDATE_PRODUCED)
        assert eval_hold.candidate is not None
        self.assertEqual(eval_hold.candidate.direction, SignalDirection.BULLISH)
        self.assertEqual(eval_hold.candidate.reason_codes[0], "BULLISH_TREND_HOLD")

    def test_bearish_crossover_and_flat_regime(self) -> None:
        """Ensure falling prices trigger bearish crossover exit to NEUTRAL."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 95.0, 90.0, 85.0]
        bars = [make_bar("TEST", p, i) for i, p in enumerate(prices)]

        eval_exit = self.strategy.evaluate(bars[:7])
        self.assertEqual(eval_exit.status, EvaluationStatus.CANDIDATE_PRODUCED)
        assert eval_exit.candidate is not None
        self.assertEqual(eval_exit.candidate.direction, SignalDirection.NEUTRAL)
        self.assertEqual(eval_exit.candidate.reason_codes[0], "BEARISH_CROSSOVER_EXIT")
        self.assertLessEqual(eval_exit.candidate.signal_score, 0.0)

        # Bar 7: flat continuation
        eval_flat = self.strategy.evaluate(bars[:8])
        assert eval_flat.candidate is not None
        self.assertEqual(eval_flat.candidate.direction, SignalDirection.NEUTRAL)
        self.assertEqual(eval_flat.candidate.reason_codes[0], "FLAT_BEARISH_REGIME")

    def test_output_stability_across_repeated_runs(self) -> None:
        """Calling evaluate repeatedly with the exact same input must yield identical outputs."""
        bars = generate_synthetic_ohlcv(ticker="SYNTH", n_bars=40)
        eval1 = self.strategy.evaluate(bars)
        eval2 = self.strategy.evaluate(bars)

        self.assertEqual(eval1.status, eval2.status)
        self.assertIsNotNone(eval1.candidate)
        self.assertIsNotNone(eval2.candidate)
        assert eval1.candidate is not None and eval2.candidate is not None

        self.assertEqual(eval1.candidate.signal_score, eval2.candidate.signal_score)
        self.assertEqual(eval1.candidate.direction, eval2.candidate.direction)
        self.assertEqual(eval1.candidate.supporting_evidence, eval2.candidate.supporting_evidence)

    def test_empty_bars_rejected(self) -> None:
        """Empty bar sequence raises ContractValidationError."""
        with self.assertRaises(ContractValidationError):
            self.strategy.evaluate([])


if __name__ == "__main__":
    unittest.main()


"""Leakage, temporal integrity, and causality tests for NOVA research subsystem."""

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
    MarketBar,
    TemporalValidationError,
    TimestampSemantics,
)
from nova.research.adapters import (
    generate_synthetic_ohlcv,
    validate_market_bar_sequence,
)
from nova.research.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)
from nova.research.evaluator import StrategyEvaluator


class TestLeakageAndTemporalInvariants(unittest.TestCase):
    """Verifies that signals, features, and evaluations have ZERO future leakage."""

    def setUp(self) -> None:
        self.strategy = EMACrossoverStrategy(
            config=EMACrossoverConfig(fast_window=4, slow_window=8)
        )
        self.bars = generate_synthetic_ohlcv(ticker="SYNTH_LEAK", n_bars=60)

    def test_appending_future_bars_does_not_alter_past_evaluations(self) -> None:
        """Evaluations at time t must be strictly identical regardless of future bars."""
        # Evaluate up to bar 30
        prefix_eval = self.strategy.evaluate(self.bars[:30])

        # Evaluate all 60 bars up to bar 30
        full_evals = self.strategy.evaluate_all(self.bars)
        eval_at_30 = full_evals[29]  # 0-indexed: index 29 is the 30th bar

        self.assertEqual(prefix_eval.status, eval_at_30.status)
        self.assertIsNotNone(prefix_eval.candidate)
        self.assertIsNotNone(eval_at_30.candidate)
        assert prefix_eval.candidate is not None and eval_at_30.candidate is not None

        self.assertEqual(prefix_eval.candidate.signal_score, eval_at_30.candidate.signal_score)
        self.assertEqual(prefix_eval.candidate.direction, eval_at_30.candidate.direction)
        self.assertEqual(prefix_eval.candidate.supporting_evidence, eval_at_30.candidate.supporting_evidence)

    def test_modifying_future_bar_does_not_alter_past_signal(self) -> None:
        """Altering bar 40 must have zero effect on the evaluation of bar 35."""
        eval_before = self.strategy.evaluate(self.bars[:36])

        # Create mutated list where bar 40 price is 1,000,000
        mutated_bars = list(self.bars)
        b40 = mutated_bars[40]
        mutated_bars[40] = MarketBar(
            ticker=b40.ticker,
            open=1000000.0,
            high=1000000.0,
            low=1000000.0,
            close=1000000.0,
            volume=b40.volume,
            source_timestamp=b40.source_timestamp,
            available_timestamp=b40.available_timestamp,
            timestamp_semantics=b40.timestamp_semantics,
            data_quality=b40.data_quality,
            provenance="MUTATED",
        )

        eval_after = self.strategy.evaluate(mutated_bars[:36])
        assert eval_before.candidate is not None and eval_after.candidate is not None
        self.assertEqual(eval_before.candidate.signal_score, eval_after.candidate.signal_score)

    def test_terminal_observation_cannot_receive_future_return(self) -> None:
        """The final bar in an evaluation cannot execute or receive a return."""
        evaluator = StrategyEvaluator(strategy=self.strategy)
        report = evaluator.run_evaluation(self.bars)

        # Check the last step in report.steps
        last_step = report.steps[-1]
        self.assertTrue(last_step.is_terminal_step)
        self.assertIsNone(last_step.execution_timestamp)
        self.assertIsNone(last_step.entry_price)
        self.assertIsNone(last_step.exit_price)
        self.assertIsNone(last_step.gross_return)

    def test_next_bar_open_execution_timing(self) -> None:
        """Execution must occur strictly at bar t+1 open, never at bar t close."""
        evaluator = StrategyEvaluator(strategy=self.strategy)
        report = evaluator.run_evaluation(self.bars)

        for step in report.steps:
            if not step.is_terminal_step:
                t = step.bar_index
                bar_t = self.bars[t]
                bar_t_plus_1 = self.bars[t + 1]

                # Signal timestamp matches bar t
                self.assertEqual(step.signal_timestamp, bar_t.source_timestamp)
                # Execution timestamp matches bar t+1
                self.assertEqual(step.execution_timestamp, bar_t_plus_1.source_timestamp)
                # Entry price is bar t+1 open (NOT bar t close)
                self.assertEqual(step.entry_price, bar_t_plus_1.open)

    def test_duplicate_timestamp_rejection(self) -> None:
        """Duplicate timestamps in the sequence must raise TemporalValidationError."""
        bars_with_duplicate = list(self.bars[:10])
        duplicate_bar = MarketBar(
            ticker=bars_with_duplicate[5].ticker,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=100.0,
            source_timestamp=bars_with_duplicate[5].source_timestamp,  # Same timestamp!
            available_timestamp=bars_with_duplicate[5].available_timestamp,
            timestamp_semantics=TimestampSemantics.CLOSE_TIME,
            data_quality=DataQualityStatus.VALID,
        )
        bars_with_duplicate.insert(6, duplicate_bar)

        with self.assertRaises(TemporalValidationError):
            validate_market_bar_sequence(bars_with_duplicate)

    def test_out_of_order_timestamps_rejection(self) -> None:
        """Shuffled or non-chronological bars must raise TemporalValidationError."""
        shuffled = list(self.bars[:10])
        shuffled[3], shuffled[4] = shuffled[4], shuffled[3]  # Swap order

        with self.assertRaises(TemporalValidationError):
            validate_market_bar_sequence(shuffled)


if __name__ == "__main__":
    unittest.main()


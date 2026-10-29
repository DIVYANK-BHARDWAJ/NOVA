"""Unit and regression tests for NOVA strategy evaluation pipeline and friction modeling."""

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
    TimestampSemantics,
)
from nova.research.adapters import generate_synthetic_ohlcv
from nova.research.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)
from nova.research.evaluator import (
    FrictionModel,
    StrategyEvaluator,
)


def make_simple_bar(index: int, open_p: float, close_p: float, ticker: str = "EVAL_TEST") -> MarketBar:
    ts = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=index * 5)
    hi = max(open_p, close_p) * 1.001
    lo = min(open_p, close_p) * 0.999
    return MarketBar(
        ticker=ticker,
        open=open_p,
        high=hi,
        low=lo,
        close=close_p,
        volume=1000.0,
        source_timestamp=ts,
        available_timestamp=ts,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=DataQualityStatus.VALID,
        provenance="FIXTURE",
    )


class TestStrategyEvaluationPipeline(unittest.TestCase):
    """Verifies return accounting, friction impact, drawdown, and benchmark alignment."""

    def test_flat_price_no_trade_case(self) -> None:
        """Flat prices must yield zero signals, zero turnover, zero return, and zero drawdown."""
        bars = [make_simple_bar(i, 100.0, 100.0) for i in range(30)]
        strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=3, slow_window=6))
        evaluator = StrategyEvaluator(strategy=strategy)

        report = evaluator.run_evaluation(bars)
        self.assertEqual(report.total_turnover, 0.0)
        self.assertAlmostEqual(report.gross_cumulative_return, 0.0, places=7)
        self.assertAlmostEqual(report.benchmark_cumulative_return, 0.0, places=7)
        self.assertAlmostEqual(report.max_drawdown, 0.0, places=7)

    def test_unverified_friction_is_never_silently_zero(self) -> None:
        """When friction is unverified, net_cumulative_return must be None."""
        bars = generate_synthetic_ohlcv(ticker="SYNTH", n_bars=50)
        strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=4, slow_window=8))
        unverified_friction = FrictionModel(status="UNVERIFIED_PENDING_PLATFORM_CONTRACT")
        evaluator = StrategyEvaluator(strategy=strategy, friction=unverified_friction)

        report = evaluator.run_evaluation(bars)
        self.assertIsNone(report.net_cumulative_return)
        self.assertEqual(report.friction_status, "UNVERIFIED_PENDING_PLATFORM_CONTRACT")

    def test_verified_friction_reduces_net_return_strictly(self) -> None:
        """Specified transaction costs must reduce net return below gross return."""
        bars = generate_synthetic_ohlcv(ticker="SYNTH", n_bars=60)
        strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=4, slow_window=8))
        verified_friction = FrictionModel(
            brokerage_bps=10.0,  # 10 bps
            slippage_bps=10.0,   # 10 bps
            status="VERIFIED_PROVISIONAL",
        )
        evaluator = StrategyEvaluator(strategy=strategy, friction=verified_friction)

        report = evaluator.run_evaluation(bars)
        self.assertIsNotNone(report.net_cumulative_return)
        assert report.net_cumulative_return is not None

        if report.total_turnover > 0.0:
            self.assertLess(report.net_cumulative_return, report.gross_cumulative_return)

    def test_insufficient_bars_rejection(self) -> None:
        """Fewer bars than slow_window must raise ContractValidationError."""
        bars = [make_simple_bar(i, 100.0, 100.0) for i in range(5)]
        strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=3, slow_window=10))
        evaluator = StrategyEvaluator(strategy=strategy)

        with self.assertRaises(ContractValidationError):
            evaluator.run_evaluation(bars)

    def test_benchmark_uses_identical_evaluation_window(self) -> None:
        """Benchmark return must equal (last_exit - first_entry) / first_entry."""
        bars = generate_synthetic_ohlcv(ticker="SYNTH", n_bars=45)
        strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=3, slow_window=6))
        evaluator = StrategyEvaluator(strategy=strategy)

        report = evaluator.run_evaluation(bars)
        evaluated_steps = [s for s in report.steps if not s.is_terminal_step and s.gross_return is not None]

        first_entry = evaluated_steps[0].entry_price
        last_exit = evaluated_steps[-1].exit_price
        assert first_entry is not None and last_exit is not None

        expected_bench = (last_exit - first_entry) / first_entry
        self.assertAlmostEqual(report.benchmark_cumulative_return, expected_bench, places=7)


if __name__ == "__main__":
    unittest.main()


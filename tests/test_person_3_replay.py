"""Unit tests for Person 3 Chronological Replay Evaluation Engine."""

from datetime import datetime, timedelta, timezone
import unittest

from nova.decision.contracts import ActionType
from nova.decision.engine import SentinelDecisionEngine
from nova.evaluation.replay import ChronologicalReplayEngine, ReplaySummary
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import ContractValidationError, DataQualityStatus, MarketBar, TimestampSemantics


def generate_market_scenario(pattern: str = "bullish", n_bars: int = 50) -> list[MarketBar]:
    base_time = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars: list[MarketBar] = []
    price = 100.0

    for i in range(n_bars):
        if pattern == "bullish":
            price *= 1.005  # Steady rise
        elif pattern == "bearish":
            price *= 0.995  # Steady fall
        elif pattern == "cyclical":
            # Oscillates
            price = 100.0 + 10.0 * (1.0 if (i // 5) % 2 == 0 else -1.0)
        else:
            price += (0.1 if i % 2 == 0 else -0.1)

        t = base_time + timedelta(minutes=5 * i)
        bars.append(
            MarketBar(
                ticker="SYNTH_REPLAY",
                open=round(price * 0.999, 2),
                high=round(price * 1.003, 2),
                low=round(price * 0.997, 2),
                close=round(price, 2),
                volume=5000.0,
                source_timestamp=t,
                available_timestamp=t,
                timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                data_quality=DataQualityStatus.VALID,
            )
        )
    return bars


class TestChronologicalReplayEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.friction = MarketFrictionEngine(
            FrictionParameters(
                spread_bps=2.0,
                slippage_bps=3.0,
                brokerage_bps=1.0,
                tax_bps=1.0,
            )
        )
        self.decision_engine = SentinelDecisionEngine(friction_engine=self.friction)
        self.replay = ChronologicalReplayEngine(
            decision_engine=self.decision_engine,
            friction_engine=self.friction,
        )

    def test_chronological_replay_bullish_run(self) -> None:
        bars = generate_market_scenario(pattern="bullish", n_bars=45)
        summary: ReplaySummary = self.replay.run(bars, warmup_bars=20)

        self.assertIsInstance(summary, ReplaySummary)
        self.assertEqual(summary.total_input_bars, 45)
        self.assertGreater(summary.evaluated_steps, 0)
        self.assertGreater(summary.total_turnover, 0.0)
        self.assertGreater(summary.total_friction_cost, 0.0)
        # Gross return must be strictly greater than net return due to friction
        self.assertGreater(summary.gross_cumulative_return, summary.net_cumulative_return)
        self.assertGreater(summary.net_cumulative_return, 0.0)
        self.assertLessEqual(summary.max_drawdown, 1.0)
        self.assertGreaterEqual(summary.max_drawdown, 0.0)

    def test_chronological_causality_guard(self) -> None:
        # Replay must fail if insufficient bars for warmup
        bars = generate_market_scenario(pattern="bullish", n_bars=15)
        with self.assertRaises(ContractValidationError):
            self.replay.run(bars, warmup_bars=20)

    def test_friction_cost_deduction(self) -> None:
        bars = generate_market_scenario(pattern="bullish", n_bars=40)
        summary = self.replay.run(bars, warmup_bars=15)

        # Confirm step-by-step math: net_return = gross_return - friction_cost
        for step in summary.steps:
            calc_net = step.step_gross_return - step.step_friction_cost
            self.assertAlmostEqual(step.step_net_return, calc_net, places=6)


if __name__ == "__main__":
    unittest.main()

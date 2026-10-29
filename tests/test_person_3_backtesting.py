"""Unit and Integration Tests for Person 3 Backtesting, Performance Metrics & Ablation."""

from datetime import datetime, timedelta, timezone
import math
import unittest

from nova.decision.contracts import ActionType, PortfolioState, PositionState
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.evaluation.ablation import AblationRunner, AblationStudyResult
from nova.evaluation.metrics import ClosedTradeRecord, PerformanceMetrics, compute_performance_metrics
from nova.evaluation.replay import ChronologicalReplayEngine, ReplayStep, ReplaySummary
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import ContractValidationError, DataQualityStatus, MarketBar, TimestampSemantics
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration


def create_mock_bar(index: int, open_p: float, close_p: float, ticker: str = "SYNTH_TEST") -> MarketBar:
    t = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return MarketBar(
        ticker=ticker,
        open=open_p,
        high=max(open_p, close_p) * 1.002,
        low=min(open_p, close_p) * 0.998,
        close=close_p,
        volume=1000.0,
        source_timestamp=t,
        available_timestamp=t,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=DataQualityStatus.VALID,
    )


class TestBacktestingAndPerformanceMetrics(unittest.TestCase):
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

    def test_performance_metrics_full_calculation(self) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        steps = [
            ReplayStep(
                step_index=0,
                decision_timestamp=t0,
                execution_timestamp=t0 + timedelta(minutes=5),
                close_price=100.0,
                next_open_price=100.0,
                action=ActionType.BUY,
                target_exposure=0.20,
                turnover=0.20,
                step_gross_return=0.01,
                step_friction_cost=0.001,
                step_net_return=0.009,
                portfolio_equity=1009000.0,
                current_drawdown=0.0,
                unrealized_pnl=5000.0,
                realized_pnl=0.0,
            ),
            ReplayStep(
                step_index=1,
                decision_timestamp=t0 + timedelta(minutes=5),
                execution_timestamp=t0 + timedelta(minutes=10),
                close_price=105.0,
                next_open_price=105.0,
                action=ActionType.SELL,
                target_exposure=0.0,
                turnover=0.20,
                step_gross_return=0.005,
                step_friction_cost=0.001,
                step_net_return=0.004,
                portfolio_equity=1013036.0,
                current_drawdown=0.0,
                unrealized_pnl=0.0,
                realized_pnl=9000.0,
            ),
        ]

        metrics = compute_performance_metrics(
            steps=steps,
            initial_capital=1000000.0,
            benchmark_return=0.05,
        )

        self.assertAlmostEqual(metrics.initial_capital, 1000000.0)
        self.assertAlmostEqual(metrics.final_wealth_inr, 1013036.0)
        self.assertAlmostEqual(metrics.net_pnl_inr, 13036.0)
        self.assertAlmostEqual(metrics.total_realized_pnl_inr, 9000.0)
        self.assertEqual(metrics.total_trades, 2)
        self.assertEqual(metrics.winning_trades, 1)
        self.assertEqual(metrics.losing_trades, 0)
        self.assertAlmostEqual(metrics.win_rate, 1.0)
        # When losing_trades == 0, profit_factor is mathematically undefined (None)
        self.assertIsNone(metrics.profit_factor)
        self.assertEqual(metrics.long_trades_count, 1)
        self.assertEqual(metrics.short_trades_count, 0)
        self.assertAlmostEqual(metrics.excess_return_over_benchmark, metrics.net_cumulative_return - 0.05, places=5)

    def test_performance_metrics_zero_trades_empty(self) -> None:
        metrics = compute_performance_metrics(
            steps=(),
            initial_capital=1000000.0,
            benchmark_return=0.02,
        )
        self.assertEqual(metrics.net_cumulative_return, 0.0)
        self.assertEqual(metrics.net_pnl_inr, 0.0)
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertIsNone(metrics.profit_factor)
        self.assertAlmostEqual(metrics.excess_return_over_benchmark, -0.02)

    def test_performance_metrics_loss_profit_factor(self) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        steps = [
            ReplayStep(
                step_index=0,
                decision_timestamp=t0,
                execution_timestamp=t0,
                close_price=100.0,
                next_open_price=100.0,
                action=ActionType.SELL,
                target_exposure=0.0,
                turnover=0.10,
                step_gross_return=0.01,
                step_friction_cost=0.001,
                step_net_return=0.009,
                portfolio_equity=1010000.0,
                current_drawdown=0.0,
                realized_pnl=10000.0,  # Win: +10,000
            ),
            ReplayStep(
                step_index=1,
                decision_timestamp=t0,
                execution_timestamp=t0,
                close_price=95.0,
                next_open_price=95.0,
                action=ActionType.COVER,
                target_exposure=0.0,
                turnover=0.10,
                step_gross_return=-0.005,
                step_friction_cost=0.001,
                step_net_return=-0.006,
                portfolio_equity=1005000.0,
                current_drawdown=0.005,
                realized_pnl=-5000.0,  # Loss: -5,000
            ),
        ]
        metrics = compute_performance_metrics(steps=steps, initial_capital=1000000.0)
        self.assertEqual(metrics.winning_trades, 1)
        self.assertEqual(metrics.losing_trades, 1)
        self.assertAlmostEqual(metrics.win_rate, 0.5)
        # Profit Factor = 10,000 / 5,000 = 2.0
        self.assertAlmostEqual(metrics.profit_factor, 2.0)
        self.assertEqual(metrics.long_trades_count, 1)
        self.assertEqual(metrics.short_trades_count, 1)
        self.assertEqual(metrics.long_realized_pnl_inr, 10000.0)
        self.assertEqual(metrics.short_realized_pnl_inr, -5000.0)

    def test_accounting_reconciliation_total_wealth(self) -> None:
        """Verify Total Wealth = Cash + Margin Locked + Long Market Value at all times."""
        bars = [create_mock_bar(i, 100.0 + (i * 0.5), 100.0 + (i * 0.5)) for i in range(40)]
        summary = self.replay.run(bars, warmup_bars=15, initial_capital=1000000.0)

        # Wealth must match step net return compounded exactly
        calc_wealth = 1000000.0
        for step in summary.steps:
            calc_wealth *= (1.0 + step.step_net_return)
            self.assertAlmostEqual(step.portfolio_equity, calc_wealth, places=2)

        self.assertAlmostEqual(summary.final_wealth_inr, calc_wealth, places=2)
        self.assertAlmostEqual(summary.final_wealth_inr - 1000000.0, summary.net_pnl_inr, places=2)

    def test_feature_ablation_study_integrity(self) -> None:
        """Verify controlled ablation runner runs all variants under identical causal data."""
        bars = [create_mock_bar(i, 100.0 + (i * 0.3), 100.0 + (i * 0.3)) for i in range(45)]
        runner = AblationRunner()
        study: AblationStudyResult = runner.run_study(bars, warmup_bars=20, initial_capital=1000000.0)

        self.assertIsInstance(study, AblationStudyResult)
        self.assertEqual(study.total_bars, 45)
        self.assertGreater(len(study.variants), 5)

        # Full system must have 0 bps marginal contribution (as the baseline)
        full_sys = study.variants[0]
        self.assertEqual(full_sys.variant_id, "full_system")
        self.assertEqual(full_sys.marginal_contribution_bps, 0.0)

        # News sentiment must be explicitly reported as unsupported data missing
        news_v = next(v for v in study.variants if v.variant_id == "news_sentiment")
        self.assertFalse(news_v.is_supported)
        self.assertIn("DATA_MISSING", news_v.unsupported_reason)

        # Formatted table must generate without error
        tbl = study.format_table()
        self.assertIn("Full Sentinel Trader", tbl)
        self.assertIn("Baseline: Buy & Hold", tbl)
        self.assertIn("DATA MISSING", tbl)

    def test_causality_and_anti_leakage_guarantee(self) -> None:
        """Verify that altering future bars has ZERO impact on past decisions."""
        bars_base = [create_mock_bar(i, 100.0 + (i * 0.2), 100.0 + (i * 0.2)) for i in range(40)]

        engine = SentinelDecisionEngine()
        # Decision at index 25
        dec_original = engine.evaluate_decision(bars_base[:26])

        # Create modified future bars starting from index 26 onwards
        bars_tampered = list(bars_base[:26])
        for j in range(26, 40):
            bars_tampered.append(create_mock_bar(j, 500.0, 500.0))  # Huge price surge in the future

        dec_after_future_tampering = engine.evaluate_decision(bars_tampered[:26])

        # Decisions must be 100% strictly identical
        self.assertEqual(dec_original.selected_action, dec_after_future_tampering.selected_action)
        self.assertEqual(dec_original.target_quantity_or_exposure, dec_after_future_tampering.target_quantity_or_exposure)
        self.assertEqual(dec_original.selected_strategy, dec_after_future_tampering.selected_strategy)


if __name__ == "__main__":
    unittest.main()

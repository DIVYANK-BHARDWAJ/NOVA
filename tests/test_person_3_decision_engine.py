"""Unit tests for Person 3 Decision Engine, Friction Models, and Risk Gates."""

from datetime import datetime, timedelta, timezone
import unittest

from nova.decision.contracts import ActionType, OrderType, TradeDecision
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    TimestampSemantics,
)
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from tests.test_person_3_strategies import create_synthetic_bars


class TestMarketFrictionEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.params = FrictionParameters(
            spread_bps=2.0,
            slippage_bps=3.0,
            brokerage_bps=1.0,
            tax_bps=1.0,
        )
        self.engine = MarketFrictionEngine(self.params)

    def test_cost_fractions(self) -> None:
        # One-way: 2 + 3 + 1 + 1 = 7 bps = 0.0007
        self.assertAlmostEqual(self.engine.one_way_friction_fraction, 0.0007, places=6)
        # Roundtrip: 14 bps = 0.0014
        self.assertAlmostEqual(self.engine.roundtrip_friction_fraction, 0.0014, places=6)

    def test_break_even_movement(self) -> None:
        # At price 1000, 14 bps break-even movement is 1.40
        be_move = self.engine.calculate_break_even_movement(1000.0, is_roundtrip=True)
        self.assertAlmostEqual(be_move, 1.40, places=4)

    def test_clearing_probability(self) -> None:
        cost = self.engine.calculate_all_in_cost(is_roundtrip=True)
        # Return equals cost -> exactly 50% probability
        prob_even = self.engine.estimate_probability_of_clearing_costs(
            expected_gross_return=cost,
            all_in_cost=cost,
            volatility=0.01,
        )
        self.assertAlmostEqual(prob_even, 0.50, places=2)

        # High return -> high probability
        prob_high = self.engine.estimate_probability_of_clearing_costs(
            expected_gross_return=cost + 0.03,
            all_in_cost=cost,
            volatility=0.01,
        )
        self.assertGreater(prob_high, 0.90)


class TestDeterministicRiskGate(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RiskConfiguration(
            max_position_exposure=0.30,
            max_drawdown_limit=0.10,
            allow_short=False,
            min_cash_buffer_fraction=0.05,
            max_staleness_seconds=300.0,
        )
        self.gate = DeterministicRiskGate(self.config)
        self.now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    def test_circuit_breaker_drawdown(self) -> None:
        # Drawdown is 12% (> 10% limit) -> must veto BUY to NO_TRADE
        result = self.gate.evaluate(
            proposed_action=ActionType.BUY,
            proposed_exposure=0.20,
            data_quality=DataQualityStatus.VALID,
            bar_timestamp=self.now,
            as_of_time=self.now,
            current_drawdown=0.12,
        )
        self.assertFalse(result.is_approved)
        self.assertEqual(result.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(result.circuit_breaker_active)

    def test_exposure_clamping(self) -> None:
        # Proposed 50% exceeds max 30% -> clamped to 30%
        result = self.gate.evaluate(
            proposed_action=ActionType.BUY,
            proposed_exposure=0.50,
            data_quality=DataQualityStatus.VALID,
            bar_timestamp=self.now,
            as_of_time=self.now,
            current_drawdown=0.02,
        )
        self.assertTrue(result.is_approved)
        self.assertAlmostEqual(result.authorized_exposure, 0.30)
        self.assertTrue(any("EXPOSURE_CLAMPED" in r for r in result.risk_reason_codes))

    def test_short_prohibition(self) -> None:
        result = self.gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=-0.20,
            data_quality=DataQualityStatus.VALID,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(result.is_approved)
        self.assertEqual(result.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("SHORT_POSITIONS_PROHIBITED" in r for r in result.risk_reason_codes))

    def test_stale_data_veto(self) -> None:
        old_bar = self.now - timedelta(seconds=400)  # > 300s limit
        result = self.gate.evaluate(
            proposed_action=ActionType.BUY,
            proposed_exposure=0.20,
            data_quality=DataQualityStatus.VALID,
            bar_timestamp=old_bar,
            as_of_time=self.now,
        )
        self.assertFalse(result.is_approved)
        self.assertEqual(result.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("MARKET_DATA_STALE" in r for r in result.risk_reason_codes))


class TestSentinelDecisionEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SentinelDecisionEngine()

    def test_decision_on_strong_bullish_uptrend(self) -> None:
        prices = [100.0 + (i * 2.0) for i in range(35)]
        bars = create_synthetic_bars(prices)
        decision: TradeDecision = self.engine.evaluate_decision(bars, current_position_exposure=0.0)

        self.assertIsInstance(decision, TradeDecision)
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertGreater(decision.target_quantity_or_exposure, 0.0)
        self.assertEqual(decision.order_type, OrderType.MARKET)
        self.assertGreater(decision.expected_net_return, 0.0)
        self.assertGreater(decision.probability_of_clearing_costs, 0.5)
        self.assertTrue(len(decision.selection_or_rejection_reasons) > 0)

        # JSON serialization check
        dct = decision.to_dict()
        self.assertEqual(dct["selected_action"], "BUY")
        self.assertIn("decision_id", dct)

    def test_abstention_on_insufficient_history(self) -> None:
        prices = [100.0 + i for i in range(5)]
        bars = create_synthetic_bars(prices)
        decision = self.engine.evaluate_decision(bars, current_position_exposure=0.0)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertEqual(decision.order_type, OrderType.NONE)


if __name__ == "__main__":
    unittest.main()

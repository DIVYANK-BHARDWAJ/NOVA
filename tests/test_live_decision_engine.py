"""Unit tests for LiveDecisionCoreEngine (14-step algorithm)."""

from datetime import datetime, timezone
import unittest

from nova.decision.contracts import ActionType, TradeDecision
from nova.decision.live_engine import LiveDecisionConfig, LiveDecisionCoreEngine
from tests.test_person_3_strategies import create_synthetic_bars


class TestLiveDecisionCoreEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LiveDecisionConfig(
            sma_fast_window=5,
            sma_slow_window=15,
            er_window=10,
            ema_long_filter_window=20,
            atr_period=10,
            risk_per_trade_fraction=0.0075,
            theta_edge_ratio=0.10,
            min_trade_notional=1000.0,
            score_to_mu_scale=1.5,
        )
        self.engine = LiveDecisionCoreEngine(self.config)

    def test_strong_uptrend_clears_edge_gate_and_buys(self) -> None:
        # Steady uptrend with low noise
        prices = [100.0 * (1.008 ** i) for i in range(40)]
        bars = create_synthetic_bars(prices)

        decision: TradeDecision = self.engine.evaluate_stock_day(
            bars=bars,
            total_equity=100000.0,
            available_cash=80000.0,
            current_drawdown=0.01,
        )

        self.assertIsInstance(decision, TradeDecision)
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertGreater(decision.target_quantity_or_exposure, 0.0)
        self.assertGreater(decision.expected_net_return, 0.0)
        self.assertTrue(any("LONG_CRITERIA_SATISFIED" in r for r in decision.selection_or_rejection_reasons))

    def test_news_explained_dip_buying(self) -> None:
        # Drop with NO bad news (news=0) gets bought as a dip
        prices = [100.0] * 25 + [98.0, 95.0, 92.0]
        bars = create_synthetic_bars(prices)
        # S2 reversion triggers
        decision = self.engine.evaluate_stock_day(
            bars=bars,
            total_equity=100000.0,
            available_cash=80000.0,
            news_sentiment_score=0.0,
        )
        # Check that S2 is computed and reasons reflect S2
        self.assertTrue(any("S2_REVERSION" in r for r in decision.selection_or_rejection_reasons))

    def test_atr_trailing_stop_exit(self) -> None:
        # Position is entered then price drops sharply below trailing stop
        prices = [100.0 + i for i in range(30)] + [130.0, 115.0]
        bars = create_synthetic_bars(prices)

        # Simulate existing position held with high water mark at 130
        self.engine.positions["TEST_ASSET"] = self.engine.positions.get("TEST_ASSET", None) or type("MockPos", (), {})()
        self.engine.positions["TEST_ASSET"].quantity = 100.0
        self.engine.positions["TEST_ASSET"].highest_price = 130.0
        self.engine.positions["TEST_ASSET"].cost_basis = 120.0
        self.engine.positions["TEST_ASSET"].days_held = 3

        decision = self.engine.evaluate_stock_day(
            bars=bars,
            total_equity=100000.0,
            available_cash=80000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.EXIT)
        self.assertTrue(any("EXIT_ATR_TRAILING_STOP" in r for r in decision.selection_or_rejection_reasons))

    def test_drawdown_guards(self) -> None:
        prices = [100.0 * (1.008 ** i) for i in range(40)]
        bars = create_synthetic_bars(prices)

        # Tier 2 drawdown >= 12% triggers exits only / NO_TRADE
        decision = self.engine.evaluate_stock_day(
            bars=bars,
            total_equity=100000.0,
            available_cash=80000.0,
            current_drawdown=0.14,
        )
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("DD_GUARD_TIER2_EXITS_ONLY" in r for r in decision.selection_or_rejection_reasons))

    def test_online_weights_update(self) -> None:
        initial_m1 = self.engine.m1
        # Positive performance update
        self.engine.update_online_weights(signals_at_t=(0.8, 0.0, 0.0), abnormal_return=0.03, sigma=0.01)
        self.assertGreater(self.engine.m1, initial_m1)

    def test_dynamic_friction_and_cash_reserve_harmony(self) -> None:
        from nova.friction.models import FrictionParameters
        custom_friction = FrictionParameters(flat_brokerage_inr=25.0, flat_dp_charges_inr=18.0)
        cfg = LiveDecisionConfig(friction_params=custom_friction)

        # Dynamic roundtrip costs: Long = 25*2 + 18 = 68.0; Short = 25*2 = 50.0
        self.assertAlmostEqual(cfg.long_roundtrip_cost_fixed, 68.0)
        self.assertAlmostEqual(cfg.short_roundtrip_cost_fixed, 50.0)

        # Harmonized cash cushion
        self.assertAlmostEqual(cfg.min_cash_reserve_fraction, 0.20)
        self.assertAlmostEqual(cfg.max_cash_reserve_fraction, 0.30)


if __name__ == "__main__":
    unittest.main()

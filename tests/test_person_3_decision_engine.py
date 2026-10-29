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

    def test_regime_weighting_integration(self) -> None:
        from nova.decision.regime import MarketRegimeDetector, RegimeConfig
        # Trending prices
        prices = [100.0 + (i * 2.0) for i in range(40)]
        bars = create_synthetic_bars(prices)

        regime_detector = MarketRegimeDetector(RegimeConfig(er_window=10))
        cfg = DecisionEngineConfig(use_regime_weighting=True)
        engine = SentinelDecisionEngine(config=cfg, regime_detector=regime_detector)

        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertTrue(any("REGIME:TRENDING" in r for r in decision.selection_or_rejection_reasons))
        self.assertTrue(any("regime_detector" == m[0] for m in decision.model_versions))

    def test_regime_fallback_on_insufficient_bars(self) -> None:
        from nova.decision.regime import MarketRegimeDetector
        prices = [100.0 + (i * 0.1) for i in range(5)]
        bars = create_synthetic_bars(prices)

        regime_detector = MarketRegimeDetector()
        cfg = DecisionEngineConfig(use_regime_weighting=True)
        engine = SentinelDecisionEngine(config=cfg, regime_detector=regime_detector)

        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)

    def test_ml_prediction_active_integration(self) -> None:
        from nova.decision.prediction import BasePredictor, PredictionResult
        prices = [100.0 + (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices)

        class MockPredictor(BasePredictor):
            def predict(self, b):
                return PredictionResult(
                    expected_return=0.02,
                    directional_probability=0.85,
                    confidence=0.90,
                    horizon_bars=1,
                    timestamp=b[-1].source_timestamp,
                    model_id="mock_model",
                    model_version="1.0.0",
                    is_fallback=False,
                )

        cfg = DecisionEngineConfig(predictor_weight=0.3, min_prediction_confidence=0.5)
        engine = SentinelDecisionEngine(config=cfg, predictor=MockPredictor())
        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)

        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertTrue(any("PREDICTOR_ACTIVE" in r for r in decision.selection_or_rejection_reasons))
        self.assertTrue(any("ml_predictor" == m[0] for m in decision.model_versions))

    def test_ml_prediction_fallback_on_low_confidence(self) -> None:
        from nova.decision.prediction import BasePredictor, PredictionResult
        prices = [100.0 + (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices)

        class LowConfPredictor(BasePredictor):
            def predict(self, b):
                return PredictionResult(
                    expected_return=-0.05,
                    directional_probability=0.20,
                    confidence=0.30,  # Below 0.50 threshold
                    horizon_bars=1,
                    timestamp=b[-1].source_timestamp,
                    model_id="mock_model",
                    model_version="1.0.0",
                    is_fallback=False,
                )

        cfg = DecisionEngineConfig(predictor_weight=0.5, min_prediction_confidence=0.50)
        engine = SentinelDecisionEngine(config=cfg, predictor=LowConfPredictor())
        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)

        # Still BUY because baseline strategy is bullish and low-confidence prediction was safely ignored
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertTrue(any("PREDICTOR_LOW_CONFIDENCE" in r for r in decision.selection_or_rejection_reasons))

    def test_ml_prediction_fallback_on_exception(self) -> None:
        from nova.decision.prediction import BasePredictor
        prices = [100.0 + (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices)

        class FailingPredictor(BasePredictor):
            def predict(self, b):
                raise RuntimeError("Predictor internal failure")

        cfg = DecisionEngineConfig(predictor_weight=0.5)
        engine = SentinelDecisionEngine(config=cfg, predictor=FailingPredictor())
        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)

        # Baseline trend following handles it safely
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertTrue(any("PREDICTOR_ERROR_FALLBACK" in r for r in decision.selection_or_rejection_reasons))

    def test_decision_engine_config_validation(self) -> None:
        with self.assertRaises(ContractValidationError):
            DecisionEngineConfig(trend_weight=-0.1)
        with self.assertRaises(ContractValidationError):
            DecisionEngineConfig(target_exposure=1.5)
        with self.assertRaises(ContractValidationError):
            DecisionEngineConfig(predictor_weight=1.5)
        with self.assertRaises(ContractValidationError):
            DecisionEngineConfig(min_prediction_confidence=1.2)

    def test_manthan_fee_calculations(self) -> None:
        from nova.friction.models import FrictionParameters, MarketFrictionEngine
        params = FrictionParameters(flat_brokerage_inr=20.0, flat_dp_charges_inr=15.34)
        engine = MarketFrictionEngine(params)

        # BUY order: Flat ₹20 + proportional
        buy_fee = engine.calculate_order_fee_inr("BUY", notional=100000.0)
        self.assertGreaterEqual(buy_fee, 20.0)

        # SELL order: Flat ₹20 + Flat ₹15.34 DP + proportional
        sell_fee = engine.calculate_order_fee_inr("SELL", notional=100000.0)
        self.assertAlmostEqual(sell_fee - buy_fee, 15.34, places=2)

        # SHORT order: Flat ₹20 (NO DP charges)
        short_fee = engine.calculate_order_fee_inr("SHORT", notional=100000.0)
        self.assertAlmostEqual(short_fee, buy_fee, places=2)

        # Break-even price calculation
        # Long break-even target price must be higher than entry price
        be_long = engine.calculate_break_even_price(entry_price=1000.0, quantity=50, is_short=False)
        self.assertGreater(be_long, 1000.0)

        # Short break-even target price must be lower than entry price
        be_short = engine.calculate_break_even_price(entry_price=1000.0, quantity=50, is_short=True)
        self.assertLess(be_short, 1000.0)

    def test_manthan_short_and_cover_flow(self) -> None:
        from nova.decision.contracts import PortfolioState, PositionState
        # Strong bearish price series
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices)

        risk_cfg = RiskConfiguration(allow_short=True)
        risk_gate = DeterministicRiskGate(risk_cfg)
        cfg = DecisionEngineConfig(allow_short=True)
        engine = SentinelDecisionEngine(config=cfg, risk_gate=risk_gate)

        # 1. From Flat, Bearish signal produces SHORT
        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)
        self.assertEqual(decision.selected_action, ActionType.SHORT)
        self.assertLess(decision.target_quantity_or_exposure, 0.0)

        # 2. Bullish reversal while holding Short produces COVER
        bullish_prices = [50.0 + (i * 2.0) for i in range(40)]
        bullish_bars = create_synthetic_bars(bullish_prices)
        short_pos = PositionState(
            ticker="SYNTH",
            quantity=-50.0,
            entry_price=100.0,
            current_market_price=130.0,
            holding_days=2,
            locked_margin=5000.0,
        )
        cover_decision = engine.evaluate_decision(
            bars=bullish_bars,
            current_position_exposure=-0.20,
            position_state=short_pos,
        )
        self.assertEqual(cover_decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(cover_decision.target_quantity_or_exposure, 0.0)

    def test_manthan_stop_loss_long(self) -> None:
        from nova.decision.contracts import PositionState
        prices = [100.0 - (i * 0.2) for i in range(40)]  # Current close ~ 92.2
        bars = create_synthetic_bars(prices)

        # Long position bought at 100.0; price is now 92.2 (-7.8% drop, exceeds 4% stop loss)
        long_pos = PositionState(
            ticker="SYNTH",
            quantity=100.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=3,
        )
        cfg = DecisionEngineConfig(stop_loss_pct=0.04)
        engine = SentinelDecisionEngine(config=cfg)
        decision = engine.evaluate_decision(bars, position_state=long_pos)

        self.assertEqual(decision.selected_action, ActionType.SELL)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)
        self.assertTrue(any("STOP_LOSS_LONG" in r for r in decision.selection_or_rejection_reasons))

    def test_manthan_stop_loss_short(self) -> None:
        from nova.decision.contracts import PositionState
        prices = [100.0 + (i * 0.3) for i in range(40)]  # Current close ~ 111.7
        bars = create_synthetic_bars(prices)

        # Short position entered at 100.0; price rose to 111.7 (+11.7% rise, exceeds 4% stop loss)
        short_pos = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=3,
            locked_margin=10000.0,
        )
        risk_cfg = RiskConfiguration(allow_short=True, stop_loss_pct=0.04)
        risk_gate = DeterministicRiskGate(risk_cfg)
        cfg = DecisionEngineConfig(allow_short=True, stop_loss_pct=0.04)
        engine = SentinelDecisionEngine(config=cfg, risk_gate=risk_gate)

        decision = engine.evaluate_decision(bars, position_state=short_pos)
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)
        self.assertTrue(any("STOP_LOSS_SHORT" in r for r in decision.selection_or_rejection_reasons))

    def test_manthan_short_10_day_auto_settlement(self) -> None:
        from nova.decision.contracts import PositionState
        prices = [100.0 for _ in range(40)]
        bars = create_synthetic_bars(prices)

        # Short position open for 10 trading days must be forcibly covered per MANTHAN bi-weekly rule
        short_pos = PositionState(
            ticker="SYNTH",
            quantity=-50.0,
            entry_price=100.0,
            current_market_price=100.0,
            holding_days=10,  # Exactly 10 days
            locked_margin=5000.0,
        )
        risk_cfg = RiskConfiguration(allow_short=True, max_short_holding_days=10)
        risk_gate = DeterministicRiskGate(risk_cfg)
        cfg = DecisionEngineConfig(allow_short=True, max_short_holding_days=10)
        engine = SentinelDecisionEngine(config=cfg, risk_gate=risk_gate)

        decision = engine.evaluate_decision(bars, position_state=short_pos)
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)
        self.assertTrue(any("SHORT_10_DAY_SETTLEMENT" in r for r in decision.selection_or_rejection_reasons))

    def test_manthan_churn_prevention_rebalance_band(self) -> None:
        from nova.decision.contracts import PositionState
        # Strong bullish series
        prices = [100.0 + (i * 2.0) for i in range(40)]
        bars = create_synthetic_bars(prices)

        # Target exposure is 0.20; current exposure is 0.19 (difference = 0.01 < 0.015 band)
        pos = PositionState(
            ticker="SYNTH",
            quantity=19.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=2,
        )
        cfg = DecisionEngineConfig(target_exposure=0.20, rebalance_band_fraction=0.015)
        engine = SentinelDecisionEngine(config=cfg)

        # Current exposure 0.19 should produce HOLD to avoid churn fees
        decision = engine.evaluate_decision(
            bars=bars,
            current_position_exposure=0.19,
            position_state=pos,
        )
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertTrue(any("POSITION_WITHIN_REBALANCE_BAND_HOLD" in r for r in decision.selection_or_rejection_reasons))

    def test_manthan_insufficient_margin_short(self) -> None:
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices)

        risk_cfg = RiskConfiguration(allow_short=True)
        risk_gate = DeterministicRiskGate(risk_cfg)
        cfg = DecisionEngineConfig(allow_short=True, target_exposure=0.20)
        engine = SentinelDecisionEngine(config=cfg, risk_gate=risk_gate)

        # Available cash is only ₹100, but required margin for 20% on ₹10,00,000 portfolio is ₹2,00,000
        decision = engine.evaluate_decision(
            bars=bars,
            current_position_exposure=0.0,
            available_cash=100.0,
            current_portfolio_value=1000000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("INSUFFICIENT_MARGIN" in r for r in decision.selection_or_rejection_reasons))

    def test_portfolio_state_total_wealth(self) -> None:
        from nova.decision.contracts import PortfolioState, PositionState
        # Total Wealth = Available Cash + Margin Locked + Market Value of All Stocks Held
        long_pos = PositionState(ticker="LONG_CO", quantity=100.0, entry_price=500.0, current_market_price=600.0)
        short_pos = PositionState(ticker="SHORT_CO", quantity=-50.0, entry_price=1000.0, current_market_price=900.0, locked_margin=50000.0)

        port = PortfolioState(
            available_cash=350000.0,
            locked_margin=50000.0,
            positions={"LONG_CO": long_pos, "SHORT_CO": short_pos},
            initial_capital=1000000.0,
            peak_wealth=1000000.0,
        )

        # Long market value: 100 * 600 = 60,000
        self.assertEqual(port.long_market_value, 60000.0)
        # Total wealth: 350,000 + 50,000 + 60,000 = 460,000
        self.assertEqual(port.total_wealth, 460000.0)
        # Drawdown: (1,000,000 - 460,000) / 1,000,000 = 54%
        self.assertAlmostEqual(port.current_drawdown, 0.54, places=4)


if __name__ == "__main__":
    unittest.main()

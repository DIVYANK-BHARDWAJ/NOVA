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

    def test_unconfirmed_portfolio_state_triggers_safe_abstain_hold(self) -> None:
        prices = [100.0 + (i * 1.0) for i in range(40)]
        bars = create_synthetic_bars(prices)
        engine = SentinelDecisionEngine()

        # Caller asserts active position exposure 0.20, but neither position_state nor portfolio_state is provided
        # System must fail safely to HOLD without inventing entry price or holding days
        decision = engine.evaluate_decision(bars, current_position_exposure=0.20)
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.20)
        self.assertTrue(any("PORTFOLIO_STATE_UNCONFIRMED_ABSTAIN_HOLD" in r for r in decision.selection_or_rejection_reasons))

    def test_invalid_entry_price_abstained(self) -> None:
        from nova.decision.contracts import PositionState
        prices = [100.0 + (i * 1.0) for i in range(40)]
        bars = create_synthetic_bars(prices)
        engine = SentinelDecisionEngine()

        # Quantity > 0 with entry_price <= 0.0 is corrupted/invalid data
        corrupted_pos = PositionState(ticker="SYNTH", quantity=50.0, entry_price=0.0, current_market_price=120.0)
        decision = engine.evaluate_decision(bars, position_state=corrupted_pos)
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertTrue(any("INVALID_POSITION_ENTRY_PRICE_ABSTAIN_HOLD" in r for r in decision.selection_or_rejection_reasons))

    # =========================================================================
    # Part 6: Profit Realization, Position Exits, Execution Safety, & Recycling
    # =========================================================================

    def test_profit_taking_partial_scale_out_50pct(self) -> None:
        from nova.decision.contracts import PositionState
        from nova.portfolio.profit_realization import ProfitRealizationConfig
        # Steady prices that rise +9% from entry
        prices = [100.0] * 30 + [102.0, 104.0, 106.0, 108.0, 109.0]
        bars = create_synthetic_bars(prices)

        pos = PositionState(
            ticker="SYNTH",
            quantity=100.0,
            entry_price=100.0,
            current_market_price=109.0,  # +9% gain > 8% threshold
            holding_days=5,
            highest_price=109.0,
            has_taken_partial_profit=False,
        )
        cfg = DecisionEngineConfig(
            enable_profit_realization=True,
            profit_realization_config=ProfitRealizationConfig(take_profit_threshold_pct=0.08, partial_exit_ratio=0.50),
            target_exposure=0.20,
        )
        engine = SentinelDecisionEngine(config=cfg)

        decision = engine.evaluate_decision(
            bars=bars,
            current_position_exposure=0.20,
            position_state=pos,
        )
        self.assertEqual(decision.selected_action, ActionType.SELL)
        # Should reduce exposure by 50%
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.10, places=4)
        self.assertTrue(any("TAKE_PROFIT_PARTIAL_50PCT_SCALE_OUT" in r for r in decision.selection_or_rejection_reasons))

    def test_trailing_stop_exit_long_and_short(self) -> None:
        from nova.decision.contracts import PositionState
        from nova.portfolio.profit_realization import ProfitRealizationConfig

        # 1. Long trailing stop: peak 120.0 (+20% > 4%), current price 115.0 (dropped > 3% from peak: 120 * 0.97 = 116.4)
        prices_long = [100.0] * 30 + [115.0]
        bars_long = create_synthetic_bars(prices_long)

        pos_long = PositionState(
            ticker="SYNTH",
            quantity=100.0,
            entry_price=100.0,
            current_market_price=115.0,
            highest_price=120.0,
            holding_days=10,
            has_taken_partial_profit=True,
        )
        cfg = DecisionEngineConfig(
            enable_profit_realization=True,
            profit_realization_config=ProfitRealizationConfig(trailing_stop_activation_pct=0.04, trailing_stop_distance_pct=0.03),
        )
        engine = SentinelDecisionEngine(config=cfg)

        decision_long = engine.evaluate_decision(
            bars=bars_long,
            current_position_exposure=0.20,
            position_state=pos_long,
        )
        self.assertEqual(decision_long.selected_action, ActionType.SELL)
        self.assertAlmostEqual(decision_long.target_quantity_or_exposure, 0.0)
        self.assertTrue(any("TRAILING_STOP_LONG_TRIGGERED" in r for r in decision_long.selection_or_rejection_reasons))

        # 2. Short trailing stop: trough 80.0 (-20% price > 4%), current price 84.0 (rebounded > 3%: 80 * 1.03 = 82.4)
        prices_short = [100.0] * 30 + [84.0]
        bars_short = create_synthetic_bars(prices_short)

        pos_short = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=100.0,
            current_market_price=84.0,
            lowest_price=80.0,
            holding_days=4,
            locked_margin=20000.0,
        )
        cfg_short = DecisionEngineConfig(
            allow_short=True,
            enable_profit_realization=True,
            profit_realization_config=ProfitRealizationConfig(trailing_stop_activation_pct=0.04, trailing_stop_distance_pct=0.03),
        )
        risk_short = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        engine_short = SentinelDecisionEngine(config=cfg_short, risk_gate=risk_short)

        decision_short = engine_short.evaluate_decision(
            bars=bars_short,
            current_position_exposure=-0.20,
            position_state=pos_short,
        )
        self.assertEqual(decision_short.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision_short.target_quantity_or_exposure, 0.0)
        self.assertTrue(any("TRAILING_STOP_SHORT_TRIGGERED" in r for r in decision_short.selection_or_rejection_reasons))

    def test_profit_giveback_protection_exit(self) -> None:
        from nova.decision.contracts import PositionState
        from nova.portfolio.profit_realization import ProfitRealizationConfig
        prices = [100.0] * 30 + [110.0]
        bars = create_synthetic_bars(prices)

        # Peak unrealized PnL was 20,000, current price 110 with 100 shares gives PnL 1,000.
        # Giveback = (20,000 - 1,000) / 20,000 = 95% > 40% maximum giveback
        pos = PositionState(
            ticker="SYNTH",
            quantity=100.0,
            entry_price=100.0,
            current_market_price=110.0,
            highest_price=130.0,
            peak_unrealized_pnl=20000.0,
            holding_days=15,
        )
        cfg = DecisionEngineConfig(
            enable_profit_realization=True,
            profit_realization_config=ProfitRealizationConfig(
                profit_giveback_max_pct=0.40,
                trailing_stop_activation_pct=0.50,  # Keep trailing stop dormant so giveback triggers
            ),
        )
        engine = SentinelDecisionEngine(config=cfg)

        decision = engine.evaluate_decision(
            bars=bars,
            current_position_exposure=0.20,
            position_state=pos,
        )
        self.assertEqual(decision.selected_action, ActionType.SELL)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)
        self.assertTrue(any("PROFIT_GIVEBACK_PROTECTION_EXIT" in r for r in decision.selection_or_rejection_reasons))

    def test_tax_deferral_window_holding(self) -> None:
        from nova.decision.contracts import PositionState
        from nova.portfolio.profit_realization import ProfitRealizationConfig
        # Flat prices (neutral signal decay)
        prices = [100.0] * 35
        bars = create_synthetic_bars(prices)

        # Holding days = 345 (within [340, 365] window)
        pos = PositionState(
            ticker="SYNTH",
            quantity=100.0,
            entry_price=90.0,
            current_market_price=100.0,
            holding_days=345,
            highest_price=101.0,
        )
        cfg = DecisionEngineConfig(
            enable_profit_realization=True,
            profit_realization_config=ProfitRealizationConfig(
                tax_deferral_buffer_days=25,
                take_profit_threshold_pct=0.20,
            ),
        )
        engine = SentinelDecisionEngine(config=cfg)

        decision = engine.evaluate_decision(
            bars=bars,
            current_position_exposure=0.20,
            position_state=pos,
        )
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertTrue(any("TAX_AWARE_HOLDING_DEFERRAL_LTCG" in r for r in decision.selection_or_rejection_reasons))

    def test_execution_safety_duplicate_exit_and_oversell_veto(self) -> None:
        from nova.decision.contracts import PortfolioState, PositionState
        from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
        risk_gate = DeterministicRiskGate(RiskConfiguration(allow_short=False))

        # 1. Duplicate exit order veto
        res_dup = risk_gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=0.0,
            current_position_exposure=0.20,
            has_pending_exit_order=True,
        )
        self.assertFalse(res_dup.is_approved)
        self.assertIn("EXECUTION_SAFETY_DUPLICATE_EXIT_PENDING_REJECT", res_dup.risk_reason_codes)

        # 2. Oversell: cannot sell without long position
        res_no_pos = risk_gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=0.0,
            current_position_exposure=0.0,
        )
        self.assertFalse(res_no_pos.is_approved)
        self.assertIn("EXECUTION_SAFETY_CANNOT_SELL_WITHOUT_LONG_POSITION_REJECT", res_no_pos.risk_reason_codes)

        # 3. Oversell: all shares already pending exit
        res_all_pending = risk_gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=0.0,
            current_position_exposure=0.20,
            pending_exit_quantity=100.0,
            current_market_price=100.0,
        )
        # Note: quantity = 0.20 * 1,000,000 / 100 = 2,000 shares. If pending is 2000:
        res_all_pending2 = risk_gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=0.0,
            current_position_exposure=0.20,
            pending_exit_quantity=2000.0,
            current_market_price=100.0,
        )
        self.assertFalse(res_all_pending2.is_approved)
        self.assertIn("EXECUTION_SAFETY_ALL_LONG_SHARES_PENDING_EXIT_REJECT", res_all_pending2.risk_reason_codes)

        # 4. Nonexistent cover: cannot cover without short position
        res_no_short = risk_gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=0.0,
            current_position_exposure=0.0,
        )
        self.assertFalse(res_no_short.is_approved)
        self.assertIn("EXECUTION_SAFETY_CANNOT_COVER_WITHOUT_SHORT_POSITION_REJECT", res_no_short.risk_reason_codes)

        # 5. Conflicting pending order reject
        res_conf = risk_gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=0.0,
            current_position_exposure=0.20,
            has_pending_entry_order=True,
        )
        self.assertFalse(res_conf.is_approved)
        self.assertIn("EXECUTION_SAFETY_CONFLICTING_ORDER_PENDING_REJECT", res_conf.risk_reason_codes)

    def test_capital_recycler_ranking_and_cushion_preservation(self) -> None:
        from nova.portfolio.profit_realization import CandidateOpportunity, CapitalRecycler, ProfitRealizationConfig
        recycler = CapitalRecycler(ProfitRealizationConfig(min_cash_reserve_fraction=0.20))

        candidates = [
            CandidateOpportunity(ticker="LOW_EDGE", expected_net_edge=0.001, return_volatility=0.02),     # 0.05 ratio < 0.15 threshold
            CandidateOpportunity(ticker="STAR_ALPHA", expected_net_edge=0.010, return_volatility=0.02),   # 0.50 ratio
            CandidateOpportunity(ticker="GOOD_RUNNER", expected_net_edge=0.006, return_volatility=0.02),  # 0.30 ratio
            CandidateOpportunity(ticker="SOLID_VALUE", expected_net_edge=0.004, return_volatility=0.02),  # 0.20 ratio
        ]

        # ₹10,00,000 total wealth, ₹5,00,000 available cash freed from exits
        plan = recycler.evaluate_reinvestment_plan(
            available_cash=500000.0,
            total_wealth=1000000.0,
            candidate_opportunities=candidates,
            current_holdings_count=1,
        )

        # Candidate LOW_EDGE is rejected (ratio 0.05 < 0.15)
        allocated_tickers = [alloc.ticker for alloc in plan.allocations]
        self.assertNotIn("LOW_EDGE", allocated_tickers)
        # STAR_ALPHA must be ranked first (highest ratio)
        self.assertEqual(allocated_tickers[0], "STAR_ALPHA")
        self.assertEqual(allocated_tickers[1], "GOOD_RUNNER")
        self.assertEqual(allocated_tickers[2], "SOLID_VALUE")

        # Cash cushion of 20% (₹2,00,000) must be preserved
        self.assertGreaterEqual(plan.preserved_cash_cushion, 200000.0)
        self.assertLessEqual(plan.total_allocated_capital, 500000.0 - 200000.0)


if __name__ == "__main__":
    unittest.main()

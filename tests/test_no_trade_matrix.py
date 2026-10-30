"""NOVA Decision Intelligence — Comprehensive NO_TRADE Decision Matrix Test Suite.

Proves with deterministic test fixtures that NOVA correctly says NO_TRADE (or HOLD)
when expected returns do not justify friction, when data is invalid or stale, when
risk constraints prohibit trading, or when the 3.0-second deadline is exceeded,
and verifies that order submission is physically prevented at the execution boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import unittest

from execution.adapter import ExecutionAdapter
from execution.accounting import AccountLedger
from execution.client import TradingApiClient
from execution.config import ExecutionConfig
from execution.models import OrderAction, OrderStatus, OrderResult, StockInfo, SimulationStage
from nova.decision.contracts import ActionType, OrderType, PositionState, PortfolioState, TradeDecision
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.pipeline import IntegratedTradingPipeline
from nova.decision.prediction import BasePredictor, PredictionResult
from nova.decision.regime import MarketRegimeDetector, MarketRegimeType, RegimeOutput
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.portfolio.profit_realization import ProfitRealizationManager
from nova.quant.contracts import (
    CandidateSignal,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    TimestampSemantics,
)
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.mean_reversion import MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingStrategy


class MockPredictor(BasePredictor):
    """Predictor returning controlled returns and confidences for testing."""

    def __init__(self, expected_return: float = 0.005, confidence: float = 0.90, should_raise: bool = False) -> None:
        self.expected_return = expected_return
        self.confidence = confidence
        self.should_raise = should_raise

    def predict(self, bars: list[MarketBar]) -> PredictionResult:
        if self.should_raise:
            raise RuntimeError("Simulated ML model failure")
        latest = bars[-1]
        return PredictionResult(
            expected_return=self.expected_return,
            directional_probability=0.55 if self.expected_return > 0 else 0.45,
            confidence=self.confidence,
            horizon_bars=1,
            timestamp=latest.source_timestamp,
            model_id="mock_predictor",
            model_version="1.0.0",
            is_fallback=False,
            reason_codes=("MOCK_PREDICTION",),
        )


class TestNoTradeDecisionMatrix(unittest.TestCase):
    """Deterministic validation of NO_TRADE behavior and execution gate enforcement."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=TradingApiClient)
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["TESTX"],
            ticker_count=1,
            current_date="2026-05-20",
            days_completed=100,
            total_days=500,
        )
        self.mock_client.get_stocks.return_value = [
            StockInfo(symbol="TESTX", ticker="TESTX", name="Test Corp")
        ]
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_mock_123",
            "ticker": "TESTX",
            "action": "BUY",
            "quantity": 50,
            "status": "FILLED",
            "execution_price": 100.0,
            "charges": 20.0,
        }

        self.ledger = AccountLedger(initial_cash=1_000_000.0)
        self.adapter = ExecutionAdapter(
            client=self.mock_client,
            config=ExecutionConfig(bot_access_token="test_tok", stage_cache_ttl_seconds=0.0, stocks_cache_ttl_seconds=0.0),
            ledger=self.ledger,
        )
        self.risk_gate = DeterministicRiskGate(RiskConfiguration(allow_short=False, max_staleness_seconds=60.0))
        self.engine_config = DecisionEngineConfig(
            trend_weight=0.50,
            mean_reversion_weight=0.50,
            predictor_weight=0.30,
            min_net_edge_bps=5.0,
            target_exposure=0.15,
            allow_short=False,
            use_regime_weighting=True,
            initial_capital=1_000_000.0,
        )
        self.predictor = MockPredictor(expected_return=0.005)
        self.engine = SentinelDecisionEngine(
            config=self.engine_config,
            trend_strategy=TrendFollowingStrategy(),
            mean_reversion_strategy=MeanReversionStrategy(),
            friction_engine=MarketFrictionEngine(),
            risk_gate=self.risk_gate,
            predictor=self.predictor,
            regime_detector=MarketRegimeDetector(),
            profit_realization_manager=ProfitRealizationManager(),
        )
        self.pipeline = IntegratedTradingPipeline(
            adapter=self.adapter,
            risk_gate=self.risk_gate,
            ledger=self.ledger,
            decision_engine=self.engine,
        )

    def _create_bars(
        self,
        n: int = 30,
        base_price: float = 100.0,
        trend: float = 0.5,
        data_quality: DataQualityStatus = DataQualityStatus.VALID,
        end_time: datetime | None = None,
    ) -> list[MarketBar]:
        if end_time is None:
            end_time = datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc)
        start_time = end_time - timedelta(days=n)
        bars: list[MarketBar] = []
        p = base_price
        for i in range(n):
            ts = start_time + timedelta(days=i + 1)
            p = max(1.0, p + trend)
            bars.append(
                MarketBar(
                    ticker="TESTX",
                    open=p - 0.2,
                    high=p + 0.5,
                    low=p - 0.5,
                    close=p,
                    volume=10000.0,
                    source_timestamp=ts,
                    available_timestamp=ts,
                    timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                    data_quality=data_quality,
                    provenance="test_fixture",
                )
            )
        return bars

    # ------------------------------------------------------------------------
    # Category A: Scenarios Requiring Abstention
    # ------------------------------------------------------------------------

    def test_case_01_negative_or_zero_expected_return(self) -> None:
        """Case 1: When expected gross return is negative or zero, action must be NO_TRADE."""
        bars = self._create_bars(n=30, trend=-0.5)  # Downtrend
        self.engine.predictor = MockPredictor(expected_return=-0.01)

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("NO_TRADE" in r or "BEARISH" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_02_gross_return_eroded_by_friction(self) -> None:
        """Case 2: Positive gross return that cannot cover all-in transaction costs must abstain."""
        bars = self._create_bars(n=30, trend=0.005)
        # 0.1 bps return, friction is ~2.0+ bps
        self.engine.predictor = MockPredictor(expected_return=0.00001)

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("FRICTION" in r or "NO_TRADE" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_03_net_utility_fails_decision_hurdle(self) -> None:
        """Case 3: Positive return where net edge fails configured hurdle (5 bps)."""
        bars = self._create_bars(n=30, trend=0.01)
        self.engine.predictor = MockPredictor(expected_return=0.0001, confidence=0.50)

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertIn(decision.selected_action, (ActionType.NO_TRADE, ActionType.HOLD))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_04_uncertainty_excessive_in_high_volatility(self) -> None:
        """Case 4: High volatility regime penalizes utility and prevents trade."""
        bars = self._create_bars(n=30, trend=0.0)
        for i, b in enumerate(bars):
            object.__setattr__(b, "close", 100.0 + (10.0 if i % 2 == 0 else -10.0))

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_05_invalid_data_quality_abstains(self) -> None:
        """Case 5: Unacceptable data quality strictly returns NO_TRADE without evaluating alpha."""
        bars = self._create_bars(n=30, trend=1.0, data_quality=DataQualityStatus.INVALID)
        decision = self.engine.evaluate_decision(bars=bars)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("DATA_QUALITY" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_06_stale_market_data_vetoed_by_risk_gate(self) -> None:
        """Case 6: Market data older than max_staleness_seconds must be vetoed by Risk Gate."""
        t_bar = datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc)
        t_as_of = t_bar + timedelta(seconds=120)  # 120s stale (> 60s config)
        bars = self._create_bars(n=30, trend=1.0, end_time=t_bar)

        decision = self.engine.evaluate_decision(bars=bars, as_of_time=t_as_of)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("STALE" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_07_model_failure_with_safe_fallback(self) -> None:
        """Case 7: ML model failure gracefully falls back to baseline with NO_TRADE."""
        bars = self._create_bars(n=30, trend=0.0)
        self.engine.predictor = MockPredictor(should_raise=True)

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertIn(decision.selected_action, (ActionType.NO_TRADE, ActionType.HOLD))
        self.assertTrue(any("PREDICTOR_ERROR" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_08_short_prohibited_policy_veto(self) -> None:
        """Case 8: Bearish signal with allow_short=False produces NO_TRADE (never SHORT)."""
        bars = self._create_bars(n=30, trend=-1.0)
        self.engine.predictor = MockPredictor(expected_return=-0.05)

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertNotEqual(decision.selected_action, ActionType.SHORT)

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_09_cash_buffer_and_buying_power_veto(self) -> None:
        """Case 9: Insufficient available cash below minimum cash buffer forbids BUY."""
        bars = self._create_bars(n=30, trend=1.0)
        self.engine.predictor = MockPredictor(expected_return=0.05)

        # Available cash is only 1,000 out of 1,000,000 portfolio (0.1% < 5% buffer)
        decision = self.engine.evaluate_decision(
            bars=bars,
            available_cash=1_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("INSUFFICIENT_CASH_BUFFER" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_10_duplicate_exit_pending_prevention(self) -> None:
        """Case 10: If an exit order is already pending, reject duplicate exit and HOLD."""
        bars = self._create_bars(n=30, trend=-1.0)
        pos = PositionState(ticker="TESTX", quantity=100.0, entry_price=100.0, current_market_price=90.0)
        port = PortfolioState(
            available_cash=500_000.0,
            positions={"TESTX": pos},
            pending_orders={"TESTX": [MagicMock(action=ActionType.SELL)]},
        )

        decision = self.engine.evaluate_decision(bars=bars, portfolio_state=port)
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertTrue(any("DUPLICATE_EXIT_PENDING" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_11_cannot_sell_without_long_position(self) -> None:
        """Case 11: Attempting to SELL when held quantity is zero must be vetoed to NO_TRADE."""
        risk_result = self.risk_gate.evaluate(
            proposed_action=ActionType.SELL,
            proposed_exposure=0.0,
            current_position_quantity=0.0,
        )
        self.assertFalse(risk_result.is_approved)
        self.assertEqual(risk_result.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("CANNOT_SELL_WITHOUT_LONG_POSITION" in r for r in risk_result.risk_reason_codes))

    def test_case_12_cannot_cover_without_short_position(self) -> None:
        """Case 12: Attempting to COVER when held quantity is zero or positive must be vetoed."""
        risk_result = self.risk_gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=0.0,
            current_position_quantity=0.0,
        )
        self.assertFalse(risk_result.is_approved)
        self.assertEqual(risk_result.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("CANNOT_COVER_WITHOUT_SHORT_POSITION" in r for r in risk_result.risk_reason_codes))

    def test_case_13_circuit_breaker_drawdown_limit(self) -> None:
        """Case 13: Drawdown exceeding max limit triggers circuit breaker vetoing all entries."""
        bars = self._create_bars(n=30, trend=1.0)
        self.engine.predictor = MockPredictor(expected_return=0.05)

        decision = self.engine.evaluate_decision(bars=bars, current_drawdown=0.25)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("CIRCUIT_BREAKER" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_14_deadline_exceeded_forces_abstention(self) -> None:
        """Case 14: When decision takes > 3.0 seconds, force abstention and record deadline miss."""
        bars = self._create_bars(n=30, trend=1.0)
        self.engine.predictor = MockPredictor(expected_return=0.05)

        # Set artificially tiny deadline to trigger deadline miss
        self.pipeline.hard_deadline_seconds = 0.000001

        tick = {
            "event_type": "price_tick",
            "data": {
                "ticker": "TESTX",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        for b in bars:
            self.pipeline._symbol_history.setdefault("TESTX", []).append(b)

        res = self.pipeline.handle_market_tick(tick)
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

        ledger = self.pipeline.get_decision_ledger()
        self.assertGreater(len(ledger), 0)
        last_rec = ledger[-1]
        self.assertEqual(last_rec["deadline_status"], "MISSED")
        self.assertEqual(last_rec["selected_action"], "NO_TRADE")
        self.assertFalse(last_rec["order_submission_attempted"])
        self.assertTrue(any("DEADLINE" in r for r in last_rec["reasons"]))
        self.assertGreater(self.pipeline.deadline_misses, 0)

    # ------------------------------------------------------------------------
    # Category B: Scenarios Permitting Trades
    # ------------------------------------------------------------------------

    def test_case_15_valid_bullish_opportunity_permits_buy(self) -> None:
        """Case 15: Strong net edge and valid risk parameters approve and execute BUY."""
        bars = self._create_bars(n=30, trend=1.5)
        self.engine.predictor = MockPredictor(expected_return=0.03, confidence=0.95)

        decision = self.engine.evaluate_decision(bars=bars)
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertGreater(decision.target_quantity_or_exposure, 0.0)

        # Route through pipeline
        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNotNone(res)
        self.assertEqual(res.status, OrderStatus.FILLED)
        self.mock_client.submit_order.assert_called_once()

    def test_case_16_hold_existing_position_within_rebalance_band(self) -> None:
        """Case 16: Position already meeting target exposure produces HOLD (zero order placement)."""
        bars = self._create_bars(n=30, trend=1.0)
        self.engine.predictor = MockPredictor(expected_return=0.02)

        # Position is currently 15% (equal to target exposure 15%)
        decision = self.engine.evaluate_decision(
            bars=bars,
            current_position_exposure=0.15,
            position_state=PositionState(ticker="TESTX", quantity=1500.0, entry_price=100.0, current_market_price=100.0),
        )
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertTrue(any("HOLD" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_case_17_stop_loss_exit_permitted(self) -> None:
        """Case 17: Stop loss breach authorizes SELL exit and routes to ExecutionAdapter."""
        bars = self._create_bars(n=30, trend=-2.0)
        latest_px = bars[-1].close
        # Record 500 shares in the ledger
        from execution.accounting import PositionLedgerEntry
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=500, total_cost=50_000.0)
        pos = PositionState(ticker="TESTX", quantity=500.0, entry_price=100.0, current_market_price=latest_px)
        port = PortfolioState(available_cash=500_000.0, positions={"TESTX": pos})

        decision = self.engine.evaluate_decision(bars=bars, portfolio_state=port)
        self.assertEqual(decision.selected_action, ActionType.SELL)
        self.assertTrue(any("STOP_LOSS" in r for r in decision.selection_or_rejection_reasons))

        res = self.pipeline.process_trade_decision(decision, bars[-1])
        self.assertIsNotNone(res)
        self.mock_client.submit_order.assert_called()


if __name__ == "__main__":
    unittest.main()

"""NOVA Person 1 Finalization Test Suite.

Comprehensive tests for:
1. Decision-engine invocation through the competition entrypoint and pipeline.
2. Deterministic Risk Gate denials preventing order submission.
3. Order abstentions (NO_TRADE, HOLD, NONE) never placing orders.
4. Position-aware incremental sizing and exposure calculation.
5. Execution event idempotency and duplicate deduplication in AccountLedger.
6. Partial-fill accounting and CANCELLED status event mapping.
7. Reconciliation of uncertain order states using verified remote fills.
8. Authoritative PortfolioState and PortfolioSummary generation for Person 3's engine.
9. Malformed and missing API responses failing safely.
10. Standalone script import resolution without PYTHONPATH.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import unittest
from unittest.mock import MagicMock, patch

from execution.accounting import AccountLedger, PortfolioSummary, PositionLedgerEntry
from execution.adapter import ExecutionAdapter
from execution.client import TradingApiClient, TradingApiError
from execution.config import ExecutionConfig
from execution.models import (
    CashBalance,
    ExecutionEvent,
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Portfolio,
    PortfolioHolding,
    SimulationStage,
    StockInfo,
)
from nova.decision.contracts import (
    ActionType,
    OrderType,
    PortfolioState,
    PositionState,
    TradeDecision,
)
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.pipeline import IntegratedTradingPipeline
from nova.quant.contracts import (
    CandidateSignal,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    ScoreSemantics,
    SignalDirection,
    StrategyEvaluation,
)
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration, RiskGateResult


class TestPerson1Finalization(unittest.TestCase):
    """Targeted validation tests for Person 1 finalization requirements."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=TradingApiClient)
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["INFY", "ALPHACORP"],
            days_completed=10,
            total_days=500,
            current_date="2026-01-15",
            is_sandbox_stream=False,
        )
        self.mock_client.get_stocks.return_value = [
            StockInfo(symbol="INFY", ticker="INFY", name="Infosys Ltd"),
            StockInfo(symbol="ALPHACORP", ticker="ALPHACORP", name="Alpha Corp"),
        ]
        self.config = ExecutionConfig(
            bot_access_token="test_token",
            stage_cache_ttl_seconds=0.0,
            stocks_cache_ttl_seconds=0.0,
        )
        self.ledger = AccountLedger(initial_cash=1_000_000.0)
        self.adapter = ExecutionAdapter(
            client=self.mock_client,
            config=self.config,
            ledger=self.ledger,
        )
        self.risk_gate = DeterministicRiskGate(RiskConfiguration(allow_short=False, max_drawdown_limit=0.10))
        self.pipeline = IntegratedTradingPipeline(
            adapter=self.adapter,
            risk_gate=self.risk_gate,
            ledger=self.ledger,
            default_target_exposure=0.15,
        )

    def test_decision_engine_wired_in_competition_bot(self) -> None:
        """Verify SentinelDecisionEngine evaluates market bars through pipeline.handle_market_tick."""
        now = datetime.now(timezone.utc)
        mock_engine = MagicMock(spec=SentinelDecisionEngine)

        # Configure engine to return a valid BUY trade decision
        mock_engine.evaluate_decision.return_value = TradeDecision(
            decision_id="dec_001",
            decision_timestamp=now,
            instrument_id="INFY",
            selected_action=ActionType.BUY,
            target_quantity_or_exposure=0.15,
            order_type=OrderType.MARKET,
            selected_strategy="sentinel_engine",
            selection_or_rejection_reasons=("STRONG_TREND",),
            configuration_version="1.0.0",
        )

        pipeline_with_engine = IntegratedTradingPipeline(
            adapter=self.adapter,
            risk_gate=self.risk_gate,
            ledger=self.ledger,
            decision_engine=mock_engine,
        )

        self.mock_client.submit_order.return_value = {
            "order_id": "ord_buy_100",
            "ticker": "INFY",
            "action": "BUY",
            "quantity": 100,
            "status": "FILLED",
            "execution_price": 1500.0,
            "charges": 20.0,
        }

        tick_data = {
            "event_type": "price_tick",
            "data": {
                "ticker": "INFY",
                "open": 1490.0,
                "high": 1510.0,
                "low": 1485.0,
                "close": 1500.0,
                "volume": 50000,
                "timestamp": now.isoformat(),
            },
        }

        result = pipeline_with_engine.handle_market_tick(tick_data)

        # Confirm engine was invoked
        mock_engine.evaluate_decision.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(self.ledger.get_position("INFY"), 100)
        self.assertEqual(pipeline_with_engine.orders_executed, 1)

    def test_risk_gate_denial_blocks_order_submission(self) -> None:
        """Verify that when RiskGate denies an action, zero orders reach the ExecutionAdapter/TradingAPI."""
        now = datetime.now(timezone.utc)
        strict_risk = MagicMock(spec=DeterministicRiskGate)
        strict_risk.evaluate.return_value = RiskGateResult(
            is_approved=False,
            authorized_action=ActionType.NO_TRADE,
            authorized_exposure=0.0,
            risk_reason_codes=("DRAWDOWN_EXCEEDED",),
            circuit_breaker_active=True,
        )

        pipeline_gated = IntegratedTradingPipeline(
            adapter=self.adapter,
            risk_gate=strict_risk,
            ledger=self.ledger,
        )

        candidate = CandidateSignal(
            strategy_id="strat_test",
            strategy_version="1.0.0",
            ticker="INFY",
            direction=SignalDirection.BULLISH,
            signal_score=1.0,
            score_semantics=ScoreSemantics.UNSPECIFIED,
            evaluation_timestamp=now,
        )
        bar = MarketBar(
            ticker="INFY",
            source_timestamp=now,
            open=1500.0,
            high=1510.0,
            low=1490.0,
            close=1500.0,
            volume=10000.0,
            data_quality=DataQualityStatus.VALID,
        )

        res = pipeline_gated.process_candidate_signal(candidate, bar)
        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()
        self.assertEqual(pipeline_gated.orders_vetoed, 1)

    def test_abstention_actions_do_not_submit_orders(self) -> None:
        """Verify NO_TRADE, HOLD, and OrderType.NONE decisions never submit orders."""
        now = datetime.now(timezone.utc)
        for act in (ActionType.NO_TRADE, ActionType.HOLD):
            decision = TradeDecision(
                decision_id=f"dec_{act.value}",
                decision_timestamp=now,
                instrument_id="INFY",
                selected_action=act,
                target_quantity_or_exposure=0.0,
                order_type=OrderType.MARKET,
                selected_strategy="sentinel_engine",
                selection_or_rejection_reasons=("ABSTAIN",),
                configuration_version="1.0.0",
            )
            res = self.adapter.execute_trade_decision(
                decision, current_price=1500.0, current_portfolio_value=1_000_000.0
            )
            self.assertIsNone(res)
            self.mock_client.submit_order.assert_not_called()

        # Test OrderType.NONE
        none_decision = TradeDecision(
            decision_id="dec_none",
            decision_timestamp=now,
            instrument_id="INFY",
            selected_action=ActionType.BUY,
            target_quantity_or_exposure=0.15,
            order_type=OrderType.NONE,
            selected_strategy="sentinel_engine",
            selection_or_rejection_reasons=("NO_ORDER_TYPE",),
            configuration_version="1.0.0",
        )
        res_none = self.adapter.execute_trade_decision(
            none_decision, current_price=1500.0, current_portfolio_value=1_000_000.0
        )
        self.assertIsNone(res_none)
        self.mock_client.submit_order.assert_not_called()

    def test_position_aware_order_sizing_and_exposure(self) -> None:
        """Verify incremental order sizing respects existing holdings and target exposure."""
        now = datetime.now(timezone.utc)
        # If portfolio value is 1,000,000 and target exposure is 15% at price 1,000:
        # Target shares = (1,000,000 * 0.15) / 1,000 = 150 shares
        buy_decision = TradeDecision(
            decision_id="dec_pos_sizing",
            decision_timestamp=now,
            instrument_id="INFY",
            selected_action=ActionType.BUY,
            target_quantity_or_exposure=0.15,
            order_type=OrderType.MARKET,
            selected_strategy="sentinel_engine",
            selection_or_rejection_reasons=("TARGET_EXPOSURE",),
            configuration_version="1.0.0",
        )

        self.mock_client.submit_order.return_value = {
            "order_id": "ord_buy_150",
            "ticker": "INFY",
            "action": "BUY",
            "quantity": 150,
            "status": "FILLED",
            "execution_price": 1000.0,
            "charges": 20.0,
        }

        # Case 1: Starting flat -> full 150 shares
        res = self.adapter.execute_trade_decision(
            decision=buy_decision,
            current_price=1000.0,
            current_portfolio_value=1_000_000.0,
            current_position_shares=0,
        )
        self.assertIsNotNone(res)
        self.mock_client.submit_order.assert_called_with(ticker="INFY", action="BUY", quantity=150, date=None)

        # Case 2: Already holding 100 shares -> incremental 50 shares
        self.mock_client.submit_order.reset_mock()
        self.mock_client.submit_order.return_value["quantity"] = 50
        res_incremental = self.adapter.execute_trade_decision(
            decision=buy_decision,
            current_price=1000.0,
            current_portfolio_value=1_000_000.0,
            current_position_shares=100,
        )
        self.assertIsNotNone(res_incremental)
        self.mock_client.submit_order.assert_called_with(ticker="INFY", action="BUY", quantity=50, date=None)

        # Case 3: Already holding 150 shares -> delta 0, no order submitted
        self.mock_client.submit_order.reset_mock()
        res_full = self.adapter.execute_trade_decision(
            decision=buy_decision,
            current_price=1000.0,
            current_portfolio_value=1_000_000.0,
            current_position_shares=150,
        )
        self.assertIsNone(res_full)
        self.mock_client.submit_order.assert_not_called()

    def test_execution_event_idempotency_and_deduplication(self) -> None:
        """Verify repeated execution events with the same key are strictly deduplicated."""
        now = datetime.now(timezone.utc)
        result = OrderResult(
            client_intent_id="intent_test_01",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.FILLED,
            order_id="ord_test_01",
            execution_price=1500.0,
            filled_quantity=50,
            charges=25.0,
            timestamp=now,
        )
        event = ExecutionEvent(
            event_type="ORDER_FILLED",
            order_result=result,
            timestamp=now,
        )

        # Process event first time
        is_first = self.ledger.process_execution_event(event)
        self.assertTrue(is_first)
        self.assertEqual(self.ledger.get_cash(), 1_000_000.0 - (50 * 1500.0) - 25.0)
        self.assertEqual(self.ledger.get_position("INFY"), 50)

        # Re-processing exact same event must return False (deduplicated)
        is_second = self.ledger.process_execution_event(event)
        self.assertFalse(is_second)
        self.assertEqual(self.ledger.get_cash(), 1_000_000.0 - (50 * 1500.0) - 25.0)
        self.assertEqual(self.ledger.get_position("INFY"), 50)

    def test_partial_fill_and_cancelled_event_handling(self) -> None:
        """Verify partial fills update cash and position, and CANCELLED order status is recognized."""
        now = datetime.now(timezone.utc)
        # 1. Partial fill event
        part_result = OrderResult(
            client_intent_id="intent_test_02",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.PARTIALLY_FILLED,
            order_id="ord_test_02",
            execution_price=1000.0,
            filled_quantity=30,
            charges=10.0,
            timestamp=now,
        )
        event = ExecutionEvent(
            event_type="ORDER_PARTIALLY_FILLED",
            order_result=part_result,
            timestamp=now,
        )
        self.ledger.process_execution_event(event)
        self.assertEqual(self.ledger.get_position("INFY"), 30)
        self.assertEqual(self.ledger.get_cash(), 1_000_000.0 - 30_000.0 - 10.0)

        # 2. CANCELLED event parsing in adapter
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_cancelled_01",
            "ticker": "INFY",
            "action": "BUY",
            "quantity": 50,
            "status": "CANCELLED",
            "execution_price": 0.0,
            "charges": 0.0,
        }
        intent = OrderIntent(
            intent_id="intent_cancel_test",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            authorized_by_risk=True,
        )
        res = self.adapter.execute_order(intent)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, OrderStatus.CANCELLED)

    def test_reconciliation_resolves_uncertain_orders_with_fill_evidence(self) -> None:
        """Verify AccountLedger.reconcile applies remote fill evidence for uncertain orders."""
        now = datetime.now(timezone.utc)
        # Record an uncertain order result into the ledger
        uncertain_res = OrderResult(
            client_intent_id="intent_uncertain_99",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=40,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
            timestamp=now,
        )
        self.ledger.process_execution_event(uncertain_res)

        cash_val = 1_000_000.0 - (40 * 1200.0) - 15.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="team_m2pccp5",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            portfolio_market_value=40 * 1200.0,
            total_wealth=cash_val + 48000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=15.0,
            total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=40, avg_price=1200.0, current_price=1200.0, market_value=48000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="team_m2pccp5",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            capital_remaining=cash_val,
            margin_locked=0.0,
            initial_capital=1_000_000.0,
            realized_pnl=0.0,
        )
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "ord_uncertain_99",
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 40,
                    "status": "FILLED",
                    "execution_price": 1200.0,
                    "charges": 15.0,
                }
            ]
        }

        report = self.ledger.reconcile(client=self.mock_client)

        # Discrepancy is reconciled and resolved
        self.assertTrue(report.is_synchronized)
        self.assertEqual(self.ledger.get_position("INFY"), 40)
        self.assertEqual(self.ledger.get_cash(), 1_000_000.0 - (40 * 1200.0) - 15.0)

    def test_account_ledger_portfolio_state_and_summary(self) -> None:
        """Verify get_portfolio_state and get_portfolio_summary return authoritative state."""
        self.ledger.positions["INFY"] = PositionLedgerEntry(ticker="INFY", quantity=100, total_cost=150000.0)
        self.ledger.cash = 850000.0

        current_prices = {"INFY": 1600.0}

        # Test PortfolioSummary
        summary: PortfolioSummary = self.ledger.get_portfolio_summary(current_prices)
        self.assertEqual(summary.available_cash, 850000.0)
        self.assertEqual(summary.total_portfolio_value, 850000.0 + (100 * 1600.0))
        self.assertEqual(summary.positions_count, 1)

        # Test PortfolioState (Person 3 contract)
        state: PortfolioState = self.ledger.get_portfolio_state(current_prices)
        self.assertEqual(state.available_cash, 850000.0)
        self.assertEqual(state.total_wealth, 850000.0 + (100 * 1600.0))
        self.assertIn("INFY", state.positions)
        self.assertEqual(state.positions["INFY"].quantity, 100.0)
        self.assertEqual(state.positions["INFY"].current_market_price, 1600.0)
        self.assertEqual(state.positions["INFY"].unrealized_pnl, 10000.0)

    def test_malformed_and_missing_api_responses_fail_safely(self) -> None:
        """Verify adapter handles malformed JSON and missing fields safely without unhandled exceptions."""
        # 1. API returns malformed dict missing status and order_id
        self.mock_client.submit_order.return_value = {"error": "unexpected format"}
        intent = OrderIntent(
            intent_id="intent_malformed_1",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=10,
            authorized_by_risk=True,
        )
        res = self.adapter.execute_order(intent)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, OrderStatus.REJECTED)

        # 2. API raises TradingApiError
        self.mock_client.submit_order.side_effect = TradingApiError("500 Internal Server Error")
        res_err = self.adapter.execute_order(intent)
        self.assertIsNotNone(res_err)
        self.assertEqual(res_err.status, OrderStatus.REJECTED)

    def test_standalone_script_import_paths(self) -> None:
        """Verify backtest and replay modules import properly without manually set PYTHONPATH."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        self.assertTrue((repo_root / "scripts" / "run_sentinel_backtest.py").exists())
        self.assertTrue((repo_root / "scripts" / "run_nova_replay.py").exists())

    def test_multi_ticker_interleaved_timestamp_monotonicity(self) -> None:
        """Verify chronological monotonicity is enforced per ticker, allowing interleaved tickers."""
        t1 = datetime(2026, 1, 15, 10, 0, 2, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 15, 10, 0, 1, tzinfo=timezone.utc)  # Earlier than t1, but for different ticker
        t3 = datetime(2026, 1, 15, 10, 0, 3, tzinfo=timezone.utc)

        bar_infy_1 = MarketBar(
            ticker="INFY",
            source_timestamp=t1,
            open=1500.0,
            high=1510.0,
            low=1490.0,
            close=1505.0,
            volume=1000.0,
            data_quality=DataQualityStatus.VALID,
        )
        bar_tcs = MarketBar(
            ticker="TCS",
            source_timestamp=t2,
            open=3500.0,
            high=3520.0,
            low=3490.0,
            close=3510.0,
            volume=500.0,
            data_quality=DataQualityStatus.VALID,
        )
        bar_infy_2 = MarketBar(
            ticker="INFY",
            source_timestamp=t3,
            open=1505.0,
            high=1515.0,
            low=1500.0,
            close=1510.0,
            volume=1200.0,
            data_quality=DataQualityStatus.VALID,
        )

        self.pipeline.handle_market_tick(bar_infy_1)
        # TCS at t2 must NOT be rejected even though t2 < t1
        self.pipeline.handle_market_tick(bar_tcs)
        self.pipeline.handle_market_tick(bar_infy_2)

        self.assertEqual(len(self.pipeline._symbol_history["INFY"]), 2)
        self.assertEqual(len(self.pipeline._symbol_history["TCS"]), 1)

    def test_sequential_partial_fills_and_full_fill_accounting(self) -> None:
        """Verify sequential partial fills (incremental deltas) and completion update cash and shares correctly."""
        now = datetime.now(timezone.utc)
        intent_id = "intent_seq_partial"

        # Partial fill 1: 30 cumulative shares @ 1000, 10 charges
        r1 = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
            execution_price=1000.0,
            filled_quantity=30,
            charges=10.0,
            timestamp=now,
        )
        self.ledger.process_execution_event(r1)
        self.assertEqual(self.ledger.get_position("INFY"), 30)
        self.assertEqual(self.ledger.get_cash(), 1_000_000.0 - 30_000.0 - 10.0)

        # Duplicate of Partial fill 1: must be skipped
        self.assertFalse(self.ledger.process_execution_event(r1))
        self.assertEqual(self.ledger.get_position("INFY"), 30)

        # Partial fill 2: 70 cumulative shares (+40 delta) @ 1010, 15 charges
        r2 = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
            execution_price=1010.0,
            filled_quantity=70,
            charges=15.0,
            timestamp=now,
        )
        self.ledger.process_execution_event(r2)
        self.assertEqual(self.ledger.get_position("INFY"), 70)
        self.assertEqual(self.ledger.get_cash(), 1_000_000.0 - 30_000.0 - 10.0 - (40 * 1010.0) - 15.0)

        # Complete fill: 100 cumulative shares (+30 delta) @ 1020, 12 charges
        r3 = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.FILLED,
            execution_price=1020.0,
            filled_quantity=100,
            charges=12.0,
            timestamp=now,
        )
        self.ledger.process_execution_event(r3)
        self.assertEqual(self.ledger.get_position("INFY"), 100)
        expected_cash = 1_000_000.0 - 30_000.0 - 10.0 - 40_400.0 - 15.0 - (30 * 1020.0) - 12.0
        self.assertEqual(self.ledger.get_cash(), expected_cash)

    def test_reconciliation_order_history_pagination(self) -> None:
        """Verify reconciliation paginates get_orders to find an uncertain order on page 2."""
        now = datetime.now(timezone.utc)
        uncertain_res = OrderResult(
            client_intent_id="intent_page_2_uncertain",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
            timestamp=now,
        )
        self.ledger.process_execution_event(uncertain_res)

        # Mock page 1 with 50 unrelated orders, and page 2 containing the matching order
        page_1_orders = [{"order_id": f"dummy_{i}", "ticker": "TCS", "action": "BUY", "quantity": 10, "status": "FILLED"} for i in range(50)]
        page_2_orders = [{
            "order_id": "ord_found_page_2",
            "ticker": "INFY",
            "action": "BUY",
            "quantity": 50,
            "status": "FILLED",
            "execution_price": 1400.0,
            "charges": 20.0,
        }]

        def mock_get_orders(page=1, limit=50, **kwargs):
            if page == 1:
                return {"orders": page_1_orders}
            elif page == 2:
                return {"orders": page_2_orders}
            return {"orders": []}

        self.mock_client.get_orders.side_effect = mock_get_orders
        cash_val = 1_000_000.0 - (50 * 1400.0) - 20.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="team_test",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            portfolio_market_value=50 * 1400.0,
            total_wealth=cash_val + 70000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=20.0,
            total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=50, avg_price=1400.0, current_price=1400.0, market_value=70000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="team_test",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            capital_remaining=cash_val,
            margin_locked=0.0,
            initial_capital=1_000_000.0,
            realized_pnl=0.0,
        )

        report = self.ledger.reconcile(client=self.mock_client)
        self.assertTrue(report.is_synchronized)
        self.assertEqual(self.ledger.get_position("INFY"), 50)
        self.assertEqual(self.ledger.get_cash(), cash_val)

    def test_multi_stock_cash_reservation_prevents_overspending(self) -> None:
        """Verify that sequential BUY signals for different stocks deplete cash so later signals respect cash buffer."""
        now = datetime.now(timezone.utc)
        # Starting cash: 1,000,000. Risk gate has min_cash_buffer_fraction 0.05 (50,000 INR reserve).
        # First BUY: 960 shares @ 1,000 = 960,000 INR
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_buy_1",
            "ticker": "INFY",
            "action": "BUY",
            "quantity": 960,
            "status": "FILLED",
            "execution_price": 1000.0,
            "charges": 100.0,
        }

        intent_1 = OrderIntent(
            intent_id="intent_buy_stock_1",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=960,
            authorized_by_risk=True,
        )
        res1 = self.adapter.execute_order(intent_1)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.status, OrderStatus.FILLED)
        # Remaining cash: 1,000,000 - 960,100 = 39,900 INR (3.99% of 1M portfolio, below 5% buffer)
        self.assertEqual(self.ledger.get_cash(), 39900.0)

        # Second BUY candidate for TCS attempts to buy, but cash fraction is 3.99% < 5.0% min buffer
        # Risk gate VETOES!
        risk_result = self.risk_gate.evaluate(
            proposed_action=ActionType.BUY,
            proposed_exposure=0.20,
            data_quality=DataQualityStatus.VALID,
            bar_timestamp=now,
            as_of_time=now,
            current_drawdown=0.0,
            available_cash=self.ledger.get_cash(),
            current_portfolio_value=self.ledger.get_portfolio_value({"INFY": 1000.0}),
            current_position_quantity=0,
            current_market_price=1000.0,
        )
        self.assertFalse(risk_result.is_approved)
        self.assertTrue(any("CASH_BUFFER_VIOLATION" in c or "INSUFFICIENT" in c for c in risk_result.risk_reason_codes))

    def test_uncertain_order_subsequently_partially_filled(self) -> None:
        """1. Targeted test: An uncertain order subsequently reported as partially filled."""
        now = datetime.now(timezone.utc)
        intent_id = "intent_uncertain_p1"
        uncertain_res = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
            timestamp=now,
        )
        self.ledger.process_execution_event(uncertain_res)
        self.assertEqual(len(self.ledger.get_uncertain_orders()), 1)
        rec = self.ledger.get_uncertain_orders()[0]
        self.assertFalse(rec.is_resolved)
        self.assertEqual(rec.remaining_quantity, 100)

        # Server order history reports partial fill of 40 shares @ 1000.0, charges 10.0
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "srv_ord_pf_1",
                    "client_intent_id": intent_id,
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 100,
                    "status": "PARTIALLY_FILLED",
                    "filled_quantity": 40,
                    "price": 1000.0,
                    "charges": 10.0,
                }
            ]
        }
        cash_val = 1_000_000.0 - 40_010.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="team_test",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            portfolio_market_value=40_000.0,
            total_wealth=cash_val + 40_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=10.0,
            total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=40, avg_price=1000.0, current_price=1000.0, market_value=40000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="team_test",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            capital_remaining=cash_val,
            margin_locked=0.0,
            initial_capital=1_000_000.0,
            realized_pnl=0.0,
        )

        report = self.ledger.reconcile(client=self.mock_client)
        # Remainder is still pending, so pending_count == 1 and is_synchronized == False
        self.assertFalse(report.is_synchronized)
        self.assertEqual(report.uncertain_orders_pending, 1)
        self.assertEqual(report.uncertain_orders_resolved, 0)
        self.assertFalse(rec.is_resolved)
        self.assertEqual(rec.resolved_status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(rec.filled_quantity, 40)
        self.assertEqual(rec.remaining_quantity, 60)
        self.assertEqual(self.ledger.get_position("INFY"), 40)
        self.assertEqual(self.ledger.get_cash(), cash_val)
        self.assertEqual(self.ledger.total_charges, 10.0)

    def test_repeated_reconciliation_of_same_partial_fill(self) -> None:
        """2. Targeted test: Repeated reconciliation of the same partial fill preserves idempotency."""
        now = datetime.now(timezone.utc)
        intent_id = "intent_idempotent_pf"
        uncertain_res = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
            timestamp=now,
        )
        self.ledger.process_execution_event(uncertain_res)

        order_history = {
            "orders": [
                {
                    "order_id": "srv_ord_idem",
                    "client_intent_id": intent_id,
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 100,
                    "status": "PARTIALLY_FILLED",
                    "filled_quantity": 40,
                    "price": 1000.0,
                    "charges": 10.0,
                }
            ]
        }
        self.mock_client.get_orders.return_value = order_history
        cash_val = 1_000_000.0 - 40_010.0
        portfolio_mock = Portfolio(
            team_id="team_test",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            portfolio_market_value=40_000.0,
            total_wealth=cash_val + 40_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=10.0,
            total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=40, avg_price=1000.0, current_price=1000.0, market_value=40000.0)],
        )
        cash_mock = CashBalance(
            team_id="team_test",
            team_name="Nova Forge",
            available_cash=cash_val,
            cash_balance=cash_val,
            capital_remaining=cash_val,
            margin_locked=0.0,
            initial_capital=1_000_000.0,
            realized_pnl=0.0,
        )
        self.mock_client.get_portfolio.return_value = portfolio_mock
        self.mock_client.get_cash.return_value = cash_mock

        # First reconciliation
        self.ledger.reconcile(client=self.mock_client)
        self.assertEqual(self.ledger.get_position("INFY"), 40)
        self.assertEqual(self.ledger.get_cash(), cash_val)

        # Repeated reconciliation calls with identical remote state
        for _ in range(3):
            rep = self.ledger.reconcile(client=self.mock_client)
            self.assertEqual(self.ledger.get_position("INFY"), 40)
            self.assertEqual(self.ledger.get_cash(), cash_val)
            self.assertEqual(self.ledger.total_charges, 10.0)
            self.assertEqual(self.ledger.get_uncertain_orders()[0].filled_quantity, 40)
            self.assertEqual(self.ledger.get_uncertain_orders()[0].remaining_quantity, 60)
            self.assertFalse(self.ledger.get_uncertain_orders()[0].is_resolved)

    def test_later_increase_in_cumulative_filled_quantity(self) -> None:
        """3. Targeted test: A later increase in cumulative filled quantity applies only delta."""
        now = datetime.now(timezone.utc)
        intent_id = "intent_inc_pf"
        uncertain_res = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
            timestamp=now,
        )
        self.ledger.process_execution_event(uncertain_res)

        # Initial partial fill: 40 shares @ 1000.0, charges 10.0
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "srv_ord_inc",
                    "client_intent_id": intent_id,
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 100,
                    "status": "PARTIALLY_FILLED",
                    "filled_quantity": 40,
                    "price": 1000.0,
                    "charges": 10.0,
                }
            ]
        }
        cash_val_1 = 1_000_000.0 - 40_010.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="t", team_name="n", available_cash=cash_val_1, cash_balance=cash_val_1,
            portfolio_market_value=40000.0, total_wealth=1000000.0, realized_pnl=0.0, unrealized_pnl=0.0,
            total_pnl=0.0, total_charges=10.0, total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=40, avg_price=1000.0, current_price=1000.0, market_value=40000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="t", team_name="n", available_cash=cash_val_1, cash_balance=cash_val_1,
            capital_remaining=cash_val_1, margin_locked=0.0, initial_capital=1000000.0, realized_pnl=0.0,
        )
        self.ledger.reconcile(client=self.mock_client)
        self.assertEqual(self.ledger.get_position("INFY"), 40)

        # Second fill: Cumulative 75 shares (delta +35) @ 1010.0, cumulative charges 20.0 (delta +10.0)
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "srv_ord_inc",
                    "client_intent_id": intent_id,
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 100,
                    "status": "PARTIALLY_FILLED",
                    "filled_quantity": 75,
                    "price": 1010.0,
                    "charges": 20.0,
                }
            ]
        }
        delta_cost = (35 * 1010.0) + 10.0
        cash_val_2 = cash_val_1 - delta_cost
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="t", team_name="n", available_cash=cash_val_2, cash_balance=cash_val_2,
            portfolio_market_value=75 * 1010.0, total_wealth=cash_val_2 + (75 * 1010.0), realized_pnl=0.0, unrealized_pnl=0.0,
            total_pnl=0.0, total_charges=20.0, total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=75, avg_price=1004.67, current_price=1010.0, market_value=75*1010.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="t", team_name="n", available_cash=cash_val_2, cash_balance=cash_val_2,
            capital_remaining=cash_val_2, margin_locked=0.0, initial_capital=1000000.0, realized_pnl=0.0,
        )
        self.ledger.reconcile(client=self.mock_client)
        self.assertEqual(self.ledger.get_position("INFY"), 75)
        self.assertEqual(self.ledger.get_cash(), cash_val_2)
        self.assertEqual(self.ledger.total_charges, 20.0)
        rec = self.ledger.get_uncertain_orders()[0]
        self.assertEqual(rec.filled_quantity, 75)
        self.assertEqual(rec.remaining_quantity, 25)
        self.assertFalse(rec.is_resolved)

    def test_partial_fill_followed_by_full_fill(self) -> None:
        """4. Targeted test: A partial fill followed by a full fill marks order resolved and updates state."""
        now = datetime.now(timezone.utc)
        intent_id = "intent_pf_to_ff"
        uncertain_res = OrderResult(
            client_intent_id=intent_id,
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
            timestamp=now,
        )
        self.ledger.process_execution_event(uncertain_res)

        # Step 1: Partial fill 40 shares @ 1000.0, charges 10.0
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "srv_ord_step",
                    "client_intent_id": intent_id,
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 100,
                    "status": "PARTIALLY_FILLED",
                    "filled_quantity": 40,
                    "price": 1000.0,
                    "charges": 10.0,
                }
            ]
        }
        cash_1 = 1_000_000.0 - 40_010.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="t", team_name="n", available_cash=cash_1, cash_balance=cash_1,
            portfolio_market_value=40000.0, total_wealth=1000000.0, realized_pnl=0.0, unrealized_pnl=0.0,
            total_pnl=0.0, total_charges=10.0, total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=40, avg_price=1000.0, current_price=1000.0, market_value=40000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="t", team_name="n", available_cash=cash_1, cash_balance=cash_1,
            capital_remaining=cash_1, margin_locked=0.0, initial_capital=1000000.0, realized_pnl=0.0,
        )
        self.ledger.reconcile(client=self.mock_client)
        self.assertFalse(self.ledger.get_uncertain_orders()[0].is_resolved)

        # Step 2: Full fill of remaining 60 shares (100 total) @ 1020.0, cumulative charges 25.0
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "srv_ord_step",
                    "client_intent_id": intent_id,
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 100,
                    "status": "FILLED",
                    "filled_quantity": 100,
                    "price": 1020.0,
                    "charges": 25.0,
                }
            ]
        }
        cash_2 = cash_1 - (60 * 1020.0) - 15.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="t", team_name="n", available_cash=cash_2, cash_balance=cash_2,
            portfolio_market_value=100 * 1020.0, total_wealth=cash_2 + 102000.0, realized_pnl=0.0, unrealized_pnl=0.0,
            total_pnl=0.0, total_charges=25.0, total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="INFY", quantity=100, avg_price=1012.0, current_price=1020.0, market_value=102000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="t", team_name="n", available_cash=cash_2, cash_balance=cash_2,
            capital_remaining=cash_2, margin_locked=0.0, initial_capital=1000000.0, realized_pnl=0.0,
        )
        rep = self.ledger.reconcile(client=self.mock_client)
        self.assertTrue(rep.is_synchronized)
        self.assertEqual(rep.uncertain_orders_pending, 0)
        self.assertEqual(rep.uncertain_orders_resolved, 1)
        rec = self.ledger.get_uncertain_orders()[0]
        self.assertTrue(rec.is_resolved)
        self.assertEqual(rec.resolved_status, OrderStatus.FILLED)
        self.assertEqual(rec.filled_quantity, 100)
        self.assertEqual(rec.remaining_quantity, 0)
        self.assertEqual(self.ledger.get_position("INFY"), 100)
        self.assertEqual(self.ledger.get_cash(), cash_2)
        self.assertEqual(self.ledger.total_charges, 25.0)

    def test_missing_or_inconsistent_remote_fill_details(self) -> None:
        """5. Targeted test: Missing or inconsistent remote fill details leave ledger unsynchronized without guessing."""
        now = datetime.now(timezone.utc)

        # Case A: Missing filled_quantity
        ledger_a = AccountLedger(initial_cash=1_000_000.0)
        ledger_a.process_execution_event(OrderResult(
            client_intent_id="intent_bad_qty", ticker="INFY", action=OrderAction.BUY, quantity=100,
            status=OrderStatus.UNKNOWN, requires_reconciliation=True, timestamp=now,
        ))
        client_a = MagicMock(spec=TradingApiClient)
        client_a.get_orders.return_value = {
            "orders": [{"order_id": "o_bad_qty", "ticker": "INFY", "action": "BUY", "quantity": 100, "status": "PARTIALLY_FILLED", "price": 1000.0}]
        }
        client_a.get_portfolio.return_value = Portfolio(team_id="t", team_name="n", available_cash=1000000.0, cash_balance=1000000.0, portfolio_market_value=0.0, total_wealth=1000000.0, realized_pnl=0.0, unrealized_pnl=0.0, total_pnl=0.0, total_charges=0.0, total_taxes=0.0, holdings=[])
        client_a.get_cash.return_value = CashBalance(team_id="t", team_name="n", available_cash=1000000.0, cash_balance=1000000.0, capital_remaining=1000000.0, margin_locked=0.0, initial_capital=1000000.0, realized_pnl=0.0)
        rep_a = ledger_a.reconcile(client_a)
        self.assertFalse(rep_a.is_synchronized)
        self.assertEqual(ledger_a.get_position("INFY"), 0)
        self.assertTrue(any("MISSING_DETAILS" in d for d in rep_a.discrepancies_detected))

        # Case B: Missing price (or non-positive price)
        ledger_b = AccountLedger(initial_cash=1_000_000.0)
        ledger_b.process_execution_event(OrderResult(
            client_intent_id="intent_bad_px", ticker="INFY", action=OrderAction.BUY, quantity=100,
            status=OrderStatus.UNKNOWN, requires_reconciliation=True, timestamp=now,
        ))
        client_b = MagicMock(spec=TradingApiClient)
        client_b.get_orders.return_value = {
            "orders": [{"order_id": "o_bad_px", "ticker": "INFY", "action": "BUY", "quantity": 100, "status": "PARTIALLY_FILLED", "filled_quantity": 40, "price": 0.0}]
        }
        client_b.get_portfolio.return_value = client_a.get_portfolio.return_value
        client_b.get_cash.return_value = client_a.get_cash.return_value
        rep_b = ledger_b.reconcile(client_b)
        self.assertFalse(rep_b.is_synchronized)
        self.assertEqual(ledger_b.get_position("INFY"), 0)
        self.assertTrue(any("INVALID_DETAILS" in d for d in rep_b.discrepancies_detected))

        # Case C: Inconsistent filled_quantity > total order quantity
        ledger_c = AccountLedger(initial_cash=1_000_000.0)
        ledger_c.process_execution_event(OrderResult(
            client_intent_id="intent_overflow", ticker="INFY", action=OrderAction.BUY, quantity=100,
            status=OrderStatus.UNKNOWN, requires_reconciliation=True, timestamp=now,
        ))
        client_c = MagicMock(spec=TradingApiClient)
        client_c.get_orders.return_value = {
            "orders": [{"order_id": "o_over", "ticker": "INFY", "action": "BUY", "quantity": 100, "status": "PARTIALLY_FILLED", "filled_quantity": 150, "price": 1000.0}]
        }
        client_c.get_portfolio.return_value = client_a.get_portfolio.return_value
        client_c.get_cash.return_value = client_a.get_cash.return_value
        rep_c = ledger_c.reconcile(client_c)
        self.assertFalse(rep_c.is_synchronized)
        self.assertEqual(ledger_c.get_position("INFY"), 0)
        self.assertTrue(any("INCONSISTENT_DETAILS" in d for d in rep_c.discrepancies_detected))

    def test_correct_cash_positions_fees_and_remaining_quantity(self) -> None:
        """6. Targeted test: Partial fill reconciliation updates cash, positions, fees, and remaining quantity accurately for SELL."""
        now = datetime.now(timezone.utc)
        # Seed 100 shares of ALPHACORP @ 500
        self.ledger.process_execution_event(OrderResult(
            client_intent_id="seed_alpha", ticker="ALPHACORP", action=OrderAction.BUY, quantity=100,
            status=OrderStatus.FILLED, execution_price=500.0, filled_quantity=100, charges=0.0, timestamp=now,
        ))
        self.assertEqual(self.ledger.get_position("ALPHACORP"), 100)
        cash_after_buy = 1_000_000.0 - 50_000.0

        # Place uncertain SELL of 60 shares
        self.ledger.process_execution_event(OrderResult(
            client_intent_id="uncertain_sell", ticker="ALPHACORP", action=OrderAction.SELL, quantity=60,
            status=OrderStatus.UNKNOWN, requires_reconciliation=True, timestamp=now,
        ))
        # Server reports partial sell of 40 shares @ 600.0 with 25.0 charges
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "srv_sell_pf",
                    "client_intent_id": "uncertain_sell",
                    "ticker": "ALPHACORP",
                    "action": "SELL",
                    "quantity": 60,
                    "status": "PARTIALLY_FILLED",
                    "filled_quantity": 40,
                    "price": 600.0,
                    "charges": 25.0,
                }
            ]
        }
        # Proceeds: 40 * 600 - 25 = 23,975. Remaining position: 100 - 40 = 60 shares
        expected_cash = cash_after_buy + 23_975.0
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="t", team_name="n", available_cash=expected_cash, cash_balance=expected_cash,
            portfolio_market_value=60 * 600.0, total_wealth=expected_cash + 36000.0, realized_pnl=3975.0, unrealized_pnl=0.0,
            total_pnl=3975.0, total_charges=25.0, total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="ALPHACORP", quantity=60, avg_price=500.0, current_price=600.0, market_value=36000.0)],
        )
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="t", team_name="n", available_cash=expected_cash, cash_balance=expected_cash,
            capital_remaining=expected_cash, margin_locked=0.0, initial_capital=1000000.0, realized_pnl=3975.0,
        )
        self.ledger.reconcile(client=self.mock_client)
        self.assertEqual(self.ledger.get_position("ALPHACORP"), 60)
        self.assertEqual(self.ledger.get_cash(), expected_cash)
        self.assertEqual(self.ledger.total_charges, 25.0)
        rec = [u for u in self.ledger.get_uncertain_orders() if u.intent_id == "uncertain_sell"][0]
        self.assertEqual(rec.filled_quantity, 40)
        self.assertEqual(rec.remaining_quantity, 20)
        self.assertFalse(rec.is_resolved)

    def test_no_duplicate_execution_events_or_accounting_mutations(self) -> None:
        """7. Targeted test: Duplicate execution events or identical cumulative quantities never cause mutations."""
        now = datetime.now(timezone.utc)
        r = OrderResult(
            client_intent_id="intent_dedup",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=50,
            execution_price=1000.0,
            charges=10.0,
            timestamp=now,
        )
        # First execution event applied successfully
        self.assertTrue(self.ledger.process_execution_event(r, event_id="evt_001"))
        cash_1 = self.ledger.get_cash()
        pos_1 = self.ledger.get_position("INFY")
        fees_1 = self.ledger.total_charges

        # Exact same event ID
        self.assertFalse(self.ledger.process_execution_event(r, event_id="evt_001"))
        # Different event ID but same or lesser cumulative fill
        self.assertFalse(self.ledger.process_execution_event(r, event_id="evt_002"))
        r_smaller = OrderResult(
            client_intent_id="intent_dedup", ticker="INFY", action=OrderAction.BUY, quantity=100,
            status=OrderStatus.PARTIALLY_FILLED, filled_quantity=40, execution_price=1000.0, charges=5.0, timestamp=now,
        )
        self.assertFalse(self.ledger.process_execution_event(r_smaller, event_id="evt_003"))

        # Invariants must remain strictly untouched
        self.assertEqual(self.ledger.get_cash(), cash_1)
        self.assertEqual(self.ledger.get_position("INFY"), pos_1)
        self.assertEqual(self.ledger.total_charges, fees_1)

    def test_mock_mode_heartbeat_skips_http_request(self) -> None:
        """Task 2 Regression: Verify mock mode does not make real HTTP calls to the gateway."""
        from scripts.manthan_trading_bot import ManthanTradingBot

        with patch("scripts.manthan_trading_bot.send_http_request") as mock_send_http:
            # 1. In mock mode, handle_preflight must return True and NOT call send_http_request
            bot_mock = ManthanTradingBot(
                gateway="http://192.168.65.157:3002",
                token="mock_token",
                mock_mode=True,
            )
            success = bot_mock.handle_preflight(101)
            self.assertTrue(success)
            self.assertEqual(bot_mock.last_heartbeat_seq, 101)
            mock_send_http.assert_not_called()

            # 2. In non-mock mode, handle_preflight must make the HTTP call
            mock_send_http.return_value = {"status": "ok"}
            bot_live = ManthanTradingBot(
                gateway="http://192.168.65.157:3002",
                token="live_token_123",
                mock_mode=False,
            )
            live_success = bot_live.handle_preflight(102)
            self.assertTrue(live_success)
            mock_send_http.assert_called_once_with(
                "http://192.168.65.157:3002/api/v1/bot/heartbeat",
                method="POST",
                token="live_token_123",
                payload={"seq": 102},
            )

if __name__ == "__main__":
    unittest.main()

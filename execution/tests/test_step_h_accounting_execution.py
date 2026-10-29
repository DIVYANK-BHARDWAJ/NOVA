"""NOVA Step H — Accounting, Execution & Reliability Hardening Tests.

Covers:
1. Accepted, rejected, full, and partial fills.
2. Idempotent processing of duplicate execution events in AccountLedger.
3. API timeouts, UNKNOWN outcomes, and duplicate submission prevention during uncertain states.
4. Accounting reconciliation against platform state with discrepancy detection (cash, positions, missing orders).
5. Lifecycle and reliability: preflight heartbeat, readiness invariants, Kafka disconnect recovery, worker idempotency.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from execution.adapter import ExecutionAdapter
from execution.accounting import AccountLedger, FullReconciliationReport, PositionLedgerEntry, UncertainOrderRecord
from execution.client import (
    AuthenticationError,
    RateLimitError,
    RequestTimeoutError,
    TradingApiClient,
    TradingApiError,
)
from execution.config import ExecutionConfig
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from execution.models import (
    CashBalance,
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Portfolio,
    PortfolioHolding,
    SimulationClock,
    SimulationStage,
    StockInfo,
)


class TestAccountingAndExecutionHardening(unittest.TestCase):
    """Verifies accounting correctness, idempotency, and timeout uncertainty handling."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=TradingApiClient)
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["INFY", "TCS"],
            days_completed=10,
            total_days=500,
            current_date="2026-01-15",
            is_sandbox_stream=False,
        )
        self.mock_client.get_stocks.return_value = [
            StockInfo(symbol="INFY", ticker="INFY", name="Infosys"),
            StockInfo(symbol="TCS", ticker="TCS", name="Tata Consultancy"),
        ]
        self.config = ExecutionConfig(
            api_base_url="http://mock-api:3002",
            bot_access_token="mock_token",
            permitted_stages=("competition",),
            allow_development_stage=True,
            stage_cache_ttl_seconds=60.0,
            stocks_cache_ttl_seconds=60.0,
        )
        self.ledger = AccountLedger(initial_cash=1_000_000.0)
        self.adapter = ExecutionAdapter(
            client=self.mock_client,
            config=self.config,
            ledger=self.ledger,
        )

    def test_full_and_partial_fill_accounting(self) -> None:
        """Full and partial fills update cash, positions, and average cost accurately."""
        # 1. Partial Fill: Order 100 shares, 40 filled @ 1000.0 with 10.0 charges
        partial_result = OrderResult(
            client_intent_id="intent_p1",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
            order_id="ord_part_1",
            execution_price=1000.0,
            filled_quantity=40,
            charges=10.0,
        )
        self.ledger.process_execution_event(partial_result)
        self.assertEqual(self.ledger.get_position("INFY"), 40)
        # Cash = 1,000,000 - (40 * 1000) - 10 = 959,990.0
        self.assertAlmostEqual(self.ledger.cash, 959_990.0)

        # 2. Second Fill: Complete remaining 60 shares @ 1050.0 with 15.0 charges
        complete_result = OrderResult(
            client_intent_id="intent_p2",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=60,
            status=OrderStatus.FILLED,
            order_id="ord_part_2",
            execution_price=1050.0,
            filled_quantity=60,
            charges=15.0,
        )
        self.ledger.process_execution_event(complete_result)
        self.assertEqual(self.ledger.get_position("INFY"), 100)
        # Cash = 959,990 - (60 * 1050) - 15 = 896,975.0
        self.assertAlmostEqual(self.ledger.cash, 896_975.0)

        # 3. Partial Sell: Sell 30 shares @ 1100.0 with 10.0 charges
        sell_result = OrderResult(
            client_intent_id="intent_s1",
            ticker="INFY",
            action=OrderAction.SELL,
            quantity=30,
            status=OrderStatus.FILLED,
            order_id="ord_sell_1",
            execution_price=1100.0,
            filled_quantity=30,
            charges=10.0,
        )
        self.ledger.process_execution_event(sell_result)
        self.assertEqual(self.ledger.get_position("INFY"), 70)
        # Cash = 896,975 + (30 * 1100) - 10 = 929,965.0
        self.assertAlmostEqual(self.ledger.cash, 929_965.0)
        # Realized PnL is tracked
        self.assertGreater(self.ledger.total_realized_pnl, 0.0)

    def test_partial_fill_followed_by_cancellation(self) -> None:
        """Verify partial fill followed by cancellation retains filled shares, cash deductions, and resolves order cleanly."""
        initial_cash = self.ledger.cash
        self.assertEqual(initial_cash, 1_000_000.0)

        # 1. Step 1: Initial Partial Fill (30 out of 100 shares @ 1000.0, charges 10.0)
        part1 = OrderResult(
            client_intent_id="intent_cancel_seq",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
            order_id="ord_pfc_1",
            execution_price=1000.0,
            filled_quantity=30,
            charges=10.0,
        )
        self.ledger.process_execution_event(part1)

        self.assertEqual(self.ledger.get_position("INFY"), 30)
        expected_cash_step1 = initial_cash - (30 * 1000.0) - 10.0
        self.assertAlmostEqual(self.ledger.cash, expected_cash_step1)
        self.assertEqual(self.ledger._cumulative_fills_by_intent["intent_cancel_seq"], 30)
        remaining_qty_step1 = 100 - self.ledger._cumulative_fills_by_intent["intent_cancel_seq"]
        self.assertEqual(remaining_qty_step1, 70)

        # 2. Step 2: Sequential Partial Fill (incremental 20 shares, cumulative 50 @ 1020.0, charges 8.0)
        part2 = OrderResult(
            client_intent_id="intent_cancel_seq",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
            order_id="ord_pfc_1",
            execution_price=1020.0,
            filled_quantity=50,  # Cumulative 50 shares
            charges=8.0,
        )
        self.ledger.process_execution_event(part2)

        self.assertEqual(self.ledger.get_position("INFY"), 50)
        expected_cash_step2 = expected_cash_step1 - (20 * 1020.0) - 8.0
        self.assertAlmostEqual(self.ledger.cash, expected_cash_step2)
        self.assertEqual(self.ledger._cumulative_fills_by_intent["intent_cancel_seq"], 50)
        remaining_qty_step2 = 100 - self.ledger._cumulative_fills_by_intent["intent_cancel_seq"]
        self.assertEqual(remaining_qty_step2, 50)

        # Register an uncertain order record to verify cancellation resolution
        self.ledger._uncertain_orders["intent_cancel_seq"] = UncertainOrderRecord(
            intent_id="intent_cancel_seq",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            timestamp=datetime.now(timezone.utc),
            rejection_reason="Network disconnect",
            filled_quantity=50,
        )
        self.assertTrue("intent_cancel_seq" in self.ledger._uncertain_orders)
        self.assertFalse(self.ledger._uncertain_orders["intent_cancel_seq"].is_resolved)

        # 3. Step 3: Cancellation event arrives for this order
        cancel_evt = OrderResult(
            client_intent_id="intent_cancel_seq",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.CANCELLED,
            order_id="ord_pfc_1",
            execution_price=0.0,
            filled_quantity=50,
            charges=0.0,
        )
        self.ledger.process_execution_event(cancel_evt)

        # Verify invariant preservation:
        # - Position remains exactly the 50 filled shares (no rollback)
        self.assertEqual(self.ledger.get_position("INFY"), 50)
        # - Cash remains exactly unchanged from step 2 (no double deduction, no phantom refund)
        self.assertAlmostEqual(self.ledger.cash, expected_cash_step2)
        # - Cumulative filled quantity remains 50
        self.assertEqual(self.ledger._cumulative_fills_by_intent["intent_cancel_seq"], 50)
        # - Remaining 50 shares are permanently cancelled and will never be filled
        remaining_cancelled = 100 - self.ledger._cumulative_fills_by_intent["intent_cancel_seq"]
        self.assertEqual(remaining_cancelled, 50)
        # - Uncertain order is resolved as CANCELLED
        uncertain_rec = self.ledger._uncertain_orders["intent_cancel_seq"]
        self.assertTrue(uncertain_rec.is_resolved)
        self.assertEqual(uncertain_rec.resolved_status, OrderStatus.CANCELLED)

    def test_duplicate_execution_events_are_idempotent(self) -> None:
        """Reprocessing the exact same execution event must not double-count cash or positions."""
        fill = OrderResult(
            client_intent_id="intent_dup_1",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.FILLED,
            order_id="ord_dup_100",
            execution_price=1500.0,
            filled_quantity=50,
            charges=20.0,
        )
        # First processing
        self.ledger.process_execution_event(fill)
        initial_cash = self.ledger.cash
        initial_pos = self.ledger.get_position("INFY")

        # Second processing of identical fill
        self.ledger.process_execution_event(fill)
        self.assertEqual(self.ledger.cash, initial_cash)
        self.assertEqual(self.ledger.get_position("INFY"), initial_pos)
        self.assertEqual(len(self.ledger._processed_event_ids), 1)

    def test_timeout_reconciliation_and_uncertainty_handling(self) -> None:
        """When submit_order times out, adapter queries order history for unambiguous match."""
        self.mock_client.submit_order.side_effect = RequestTimeoutError("Connection timed out after 5.0s")

        # Case A: Exactly 1 unambiguous match in order history -> safely reconciled to FILLED
        self.mock_client.get_orders.return_value = {
            "orders": [
                {
                    "order_id": "ord_timed_out_1",
                    "ticker": "INFY",
                    "action": "BUY",
                    "quantity": 50,
                    "status": "FILLED",
                    "price": 1500.0,
                    "charges": 20.0,
                }
            ]
        }

        intent1 = OrderIntent(
            intent_id="intent_timeout_reconciled",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            authorized_by_risk=True,
        )
        result1 = self.adapter.execute_order(intent1)
        self.assertEqual(result1.status, OrderStatus.FILLED)
        self.assertEqual(result1.order_id, "ord_timed_out_1")

        # Case B: Zero matching orders in history -> marked UNKNOWN with requires_reconciliation=True
        self.mock_client.get_orders.return_value = {"orders": []}
        intent2 = OrderIntent(
            intent_id="intent_timeout_unknown",
            ticker="TCS",
            action=OrderAction.BUY,
            quantity=25,
            authorized_by_risk=True,
        )
        result2 = self.adapter.execute_order(intent2)
        self.assertEqual(result2.status, OrderStatus.UNKNOWN)
        self.assertTrue(result2.requires_reconciliation)

        # Case C: Re-submitting the exact same completed intent is strictly rejected
        dup_result = self.adapter.execute_order(intent2)
        self.assertEqual(dup_result.status, OrderStatus.REJECTED)
        self.assertIn("DUPLICATE_INTENT", dup_result.rejection_reason or "")

    def test_accounting_reconciliation_discrepancy_reporting(self) -> None:
        """Reconciliation accurately detects cash, position, and order mismatches."""
        # Local ledger state: 50 INFY, cash = 900,000
        self.ledger.cash = 900_000.0
        self.ledger.positions["INFY"] = PositionLedgerEntry(ticker="INFY", quantity=50, total_cost=75000.0)

        # Platform state reports cash = 950,000 (discrepancy) and 60 INFY (discrepancy)
        self.mock_client.get_cash.return_value = CashBalance(
            team_id="team_1",
            team_name="Team One",
            available_cash=950_000.0,
            cash_balance=950_000.0,
            capital_remaining=950_000.0,
            margin_locked=0.0,
            initial_capital=1_000_000.0,
            realized_pnl=0.0,
            currency="INR",
        )
        self.mock_client.get_portfolio.return_value = Portfolio(
            team_id="team_1",
            team_name="Team One",
            available_cash=950_000.0,
            cash_balance=950_000.0,
            portfolio_market_value=90_000.0,
            total_wealth=1_040_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=0.0,
            total_taxes=0.0,
            holdings=[
                PortfolioHolding(ticker="INFY", quantity=60, avg_price=1500.0),
            ],
        )
        self.mock_client.get_orders.return_value = {"orders": []}

        report = self.ledger.reconcile(self.mock_client)
        self.assertFalse(report.is_synchronized)
        self.assertTrue(len(report.discrepancies_detected) > 0)
        self.assertFalse(report.positions.is_synchronized)
        self.assertFalse(report.cash.is_synchronized)


class TestLifecycleAndReliabilityHardening(unittest.TestCase):
    """Verifies preflight heartbeats, readiness transitions, and background worker lifecycle."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=TradingApiClient)
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["INFY", "TCS"],
            days_completed=1,
            total_days=500,
            current_date="2026-01-01",
            is_sandbox_stream=False,
        )
        self.mock_client.get_simulation_clock.return_value = SimulationClock(
            status="running",
            simulated_date="2026-01-01",
            current_day=1,
            days_completed=1,
            total_days=500,
            remaining_days=499,
            interval_seconds=10,
        )
        self.mock_client.send_heartbeat.return_value = True

    def test_heartbeat_worker_acknowledgement_and_worker_idempotency(self) -> None:
        """HeartbeatWorker processes sequence IDs and duplicate starts do not create extra threads."""
        worker = HeartbeatWorker(client=self.mock_client)
        worker.start()
        # Duplicate start call must be harmless
        worker.start()
        self.assertTrue(worker.is_running)

        # Enqueue preflight heartbeat
        worker.enqueue_heartbeat(42)
        time.sleep(0.05)
        worker.stop()
        self.assertFalse(worker.is_running)

        self.mock_client.send_heartbeat.assert_called_with("42")
        self.assertEqual(worker.acknowledged_count, 1)

    def test_readiness_invariants_require_heartbeat_acknowledgement(self) -> None:
        """LifecycleMonitor cannot report bot ready if preflight heartbeats were never acked."""
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="preflight",
            status="ready",
            tickers=["INFY", "TCS"],
        )
        worker = HeartbeatWorker(client=self.mock_client)
        mock_consumer = MagicMock()
        mock_consumer.is_running = True
        mock_consumer.assigned_partitions = [1]

        monitor = LifecycleMonitor(
            client=self.mock_client,
            poll_interval_seconds=0.1,
            heartbeat_worker=worker,
            consumer=mock_consumer,
        )
        worker.start()
        try:
            monitor.poll_once()

            # In preflight stage with 0 heartbeats acknowledged, bot must NOT be ready
            summary = monitor.get_readiness_summary()
            self.assertFalse(summary["bot_ready"])
            self.assertTrue(any("Heartbeat acknowledgements missing" in b for b in summary.get("readiness_blockers", [])))

            # Simulate acknowledging a heartbeat
            worker.acknowledged_count = 1
            summary_after = monitor.get_readiness_summary()
            self.assertTrue(summary_after["bot_ready"])
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()

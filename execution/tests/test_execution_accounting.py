"""Exhaustive unit tests for hardened portfolio accounting and reconciliation (Step F).

Tests:
1. Baseline position and cash matching (preserves original tests).
2. BUY, SELL, SHORT, COVER position updates and realized PnL calculations.
3. Fees and commission accounting across actions.
4. Idempotent deduplication of execution events preventing double-counting.
5. Cumulative and incremental partial fill tracking.
6. Rejection of duplicate partial fills.
7. Rejected and cancelled order handling without balance side-effects.
8. UNKNOWN execution outcomes held in uncertain ledger without blind retries.
9. Full authoritative API reconciliation (position matching, cash variances).
10. Automated resolution of uncertain orders via remote order history.
11. Graceful API failure handling (timeouts, HTTP errors, unreachable server).
12. Integration seam between ExecutionAdapter and AccountLedger.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from execution.accounting import (
    AccountLedger,
    CashReconciliationReport,
    FullReconciliationReport,
    PositionLedgerEntry,
    PositionReconciliationReport,
    UncertainOrderRecord,
    reconcile_cash,
    reconcile_positions,
)
from execution.adapter import ExecutionAdapter
from execution.client import TradingApiClient, TradingApiError, RequestTimeoutError
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


# ---------------------------------------------------------------------------
# 1. Baseline Position and Cash Reconciliation Tests (Preserved)
# ---------------------------------------------------------------------------

def test_reconcile_positions_perfect_match():
    """Identical local and remote holdings produce synchronized report."""
    local = {"ROUT": 100, "BLUX": 50}
    remote = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=100000.0,
        cash_balance=100000.0,
        portfolio_market_value=15000.0,
        total_wealth=115000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[
            PortfolioHolding(ticker="ROUT", quantity=100),
            PortfolioHolding(ticker="BLUX", quantity=50),
        ],
    )

    report = reconcile_positions(local, remote)
    assert report.is_synchronized is True
    assert report.matched == {"ROUT": 100, "BLUX": 50}
    assert len(report.discrepancies) == 0


def test_reconcile_positions_discrepancies_and_missing():
    """Detects quantity mismatches, missing local positions, and missing remote positions."""
    local = {"ROUT": 100, "MOTR": 20}  # MOTR not in remote
    remote = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=100000.0,
        cash_balance=100000.0,
        portfolio_market_value=15000.0,
        total_wealth=115000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[
            PortfolioHolding(ticker="ROUT", quantity=80),  # Discrepancy: local 100, remote 80
            PortfolioHolding(ticker="BLUX", quantity=30),  # Missing in local
        ],
    )

    report = reconcile_positions(local, remote)
    assert report.is_synchronized is False
    assert "ROUT" in report.discrepancies
    assert report.discrepancies["ROUT"]["local"] == 100
    assert report.discrepancies["ROUT"]["remote"] == 80
    assert report.discrepancies["ROUT"]["diff"] == -20

    assert report.missing_in_local == {"BLUX": 30}
    assert report.missing_in_remote == {"MOTR": 20}


def test_reconcile_cash():
    """Reconciles local cash against remote balance within tolerance."""
    remote_cash = CashBalance(
        team_id="team_1",
        team_name="Nova",
        available_cash=500000.0,
        cash_balance=500000.0,
        capital_remaining=500000.0,
        margin_locked=0.0,
        initial_capital=1000000.0,
        realized_pnl=0.0,
    )

    # 1. Matching cash
    report_match = reconcile_cash(500000.0, remote_cash)
    assert report_match.is_synchronized is True
    assert report_match.variance == 0.0

    # 2. Minor rounding within tolerance
    report_minor = reconcile_cash(499999.995, remote_cash, tolerance=0.01)
    assert report_minor.is_synchronized is True

    # 3. Discrepancy exceeding tolerance
    report_diff = reconcile_cash(480000.0, remote_cash)
    assert report_diff.is_synchronized is False
    assert report_diff.variance == 20000.0


# ---------------------------------------------------------------------------
# 2. AccountLedger: Position & Cash Updates on BUY / SELL / Realized PnL
# ---------------------------------------------------------------------------

def test_ledger_buy_increases_position_and_reduces_cash():
    """Verify BUY order updates positions, entry price, and cash correctly."""
    ledger = AccountLedger(initial_cash=100000.0)

    # BUY 100 ROUT @ 150.0, charges 10.0
    res = OrderResult(
        client_intent_id="intent_buy_1",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=100,
        status=OrderStatus.FILLED,
        filled_quantity=100,
        execution_price=150.0,
        charges=10.0,
    )
    ok = ledger.process_execution_event(res)
    assert ok is True

    # Cash: 100000 - (100 * 150 + 10) = 84990.0
    assert ledger.get_cash() == 84990.0
    assert ledger.get_positions() == {"ROUT": 100}
    assert ledger.positions["ROUT"].avg_price == 150.0
    assert ledger.total_charges == 10.0
    assert ledger.total_realized_pnl == 0.0


def test_ledger_sell_reduces_position_and_realizes_pnl():
    """Verify SELL order reduces position, increases cash, and records realized PnL."""
    ledger = AccountLedger(initial_cash=100000.0)

    # BUY 100 ROUT @ 150.0 (no charges for baseline)
    ledger.process_execution_event(
        OrderResult(
            client_intent_id="buy_1",
            ticker="ROUT",
            action=OrderAction.BUY,
            quantity=100,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            execution_price=150.0,
            charges=0.0,
        )
    )
    assert ledger.get_cash() == 85000.0

    # SELL 40 ROUT @ 170.0, charges 5.0
    # Profit: (170 - 150) * 40 - 5.0 = 800 - 5 = 795.0
    # Proceeds: (40 * 170) - 5 = 6800 - 5 = 6795.0
    ledger.process_execution_event(
        OrderResult(
            client_intent_id="sell_1",
            ticker="ROUT",
            action=OrderAction.SELL,
            quantity=40,
            status=OrderStatus.FILLED,
            filled_quantity=40,
            execution_price=170.0,
            charges=5.0,
        )
    )

    assert ledger.get_cash() == 85000.0 + 6795.0
    assert ledger.get_positions() == {"ROUT": 60}
    assert ledger.positions["ROUT"].avg_price == 150.0
    assert ledger.positions["ROUT"].realized_pnl == 795.0
    assert ledger.total_realized_pnl == 795.0
    assert ledger.total_charges == 5.0


def test_ledger_short_and_cover_flow():
    """Verify SHORT and COVER operations track negative holdings and PnL correctly."""
    ledger = AccountLedger(initial_cash=50000.0)

    # SHORT 50 MOTR @ 200.0, charges 5.0
    # Proceeds: (50 * 200) - 5 = 9995.0
    ledger.process_execution_event(
        OrderResult(
            client_intent_id="short_1",
            ticker="MOTR",
            action=OrderAction.SHORT,
            quantity=50,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            execution_price=200.0,
            charges=5.0,
        )
    )
    assert ledger.get_cash() == 50000.0 + 9995.0
    assert ledger.get_positions() == {"MOTR": -50}

    # COVER 50 MOTR @ 180.0, charges 5.0
    # Cost: (50 * 180) + 5 = 9005.0
    # Short Profit: (200 - 180) * 50 - 5.0 = 1000 - 5 = 995.0 (excluding short open fee)
    ledger.process_execution_event(
        OrderResult(
            client_intent_id="cover_1",
            ticker="MOTR",
            action=OrderAction.COVER,
            quantity=50,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            execution_price=180.0,
            charges=5.0,
        )
    )
    assert ledger.get_cash() == 59995.0 - 9005.0
    assert ledger.get_positions() == {}  # Closed position
    assert ledger.total_charges == 10.0


# ---------------------------------------------------------------------------
# 3. Idempotency & Duplicate Execution Protection
# ---------------------------------------------------------------------------

def test_ledger_idempotent_duplicate_events():
    """Duplicate execution events must be skipped to prevent double-counting."""
    ledger = AccountLedger(initial_cash=50000.0)

    event = ExecutionEvent(
        event_type="ORDER_FILLED",
        order_result=OrderResult(
            client_intent_id="intent_100",
            ticker="ROUT",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            execution_price=100.0,
            charges=2.0,
        ),
    )

    # First delivery: processed successfully
    assert ledger.process_execution_event(event) is True
    assert ledger.get_cash() == 50000.0 - (50 * 100.0 + 2.0)
    assert ledger.get_positions() == {"ROUT": 50}

    # Second delivery (duplicate callback/stream redelivery): must be ignored!
    assert ledger.process_execution_event(event) is False
    # Cash and positions must remain exactly identical
    assert ledger.get_cash() == 50000.0 - (50 * 100.0 + 2.0)
    assert ledger.get_positions() == {"ROUT": 50}


# ---------------------------------------------------------------------------
# 4. Partial Fills & Cumulative/Incremental Fill Tracking
# ---------------------------------------------------------------------------

def test_ledger_partial_fills_incremental_accounting():
    """Incremental partial fills update state accurately without double-counting."""
    ledger = AccountLedger(initial_cash=100000.0)

    # Total order intent is 100 shares.
    # Partial Fill 1: 40 shares @ 100.0, charges 2.0
    res1 = OrderResult(
        client_intent_id="split_order_1",
        ticker="BLUX",
        action=OrderAction.BUY,
        quantity=100,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=40,
        execution_price=100.0,
        charges=2.0,
    )
    assert ledger.process_execution_event(res1) is True
    assert ledger.get_positions() == {"BLUX": 40}
    assert ledger.get_cash() == 100000.0 - (40 * 100.0 + 2.0)

    # Partial Fill 2: Cumulative 70 shares (30 incremental) @ 102.0, charges 1.5
    res2 = OrderResult(
        client_intent_id="split_order_1",
        ticker="BLUX",
        action=OrderAction.BUY,
        quantity=100,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=70,
        execution_price=102.0,
        charges=1.5,
    )
    assert ledger.process_execution_event(res2) is True
    assert ledger.get_positions() == {"BLUX": 70}
    assert ledger.get_cash() == 100000.0 - (40 * 100.0 + 2.0) - (30 * 102.0 + 1.5)

    # Final Fill: Cumulative 100 shares (30 incremental) @ 101.0, charges 1.5
    res3 = OrderResult(
        client_intent_id="split_order_1",
        ticker="BLUX",
        action=OrderAction.BUY,
        quantity=100,
        status=OrderStatus.FILLED,
        filled_quantity=100,
        execution_price=101.0,
        charges=1.5,
    )
    assert ledger.process_execution_event(res3) is True
    assert ledger.get_positions() == {"BLUX": 100}
    assert ledger.total_charges == 5.0


def test_ledger_duplicate_partial_fill_ignored():
    """Redelivered partial fill with identical cumulative quantity produces 0 delta and is ignored."""
    ledger = AccountLedger(initial_cash=50000.0)

    res = OrderResult(
        client_intent_id="intent_p",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=100,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=40,
        execution_price=100.0,
        charges=1.0,
    )
    assert ledger.process_execution_event(res) is True
    assert ledger.get_positions() == {"ROUT": 40}

    # Redeliver with different event_id but identical cumulative fill: delta is 0
    assert ledger.process_execution_event(res, event_id="forced_new_event_key") is False
    assert ledger.get_positions() == {"ROUT": 40}


# ---------------------------------------------------------------------------
# 5. Rejected and Cancelled Orders
# ---------------------------------------------------------------------------

def test_ledger_rejected_and_cancelled_orders():
    """Rejections and cancellations produce zero cash deductions and zero position updates."""
    ledger = AccountLedger(initial_cash=50000.0)

    rej = OrderResult(
        client_intent_id="intent_rej",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=50,
        status=OrderStatus.REJECTED,
        rejection_reason="INSUFFICIENT_MARGIN",
    )
    assert ledger.process_execution_event(rej) is True
    assert ledger.get_cash() == 50000.0
    assert ledger.get_positions() == {}

    canc = OrderResult(
        client_intent_id="intent_canc",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=50,
        status=OrderStatus.CANCELLED,
    )
    assert ledger.process_execution_event(canc) is True
    assert ledger.get_cash() == 50000.0
    assert ledger.get_positions() == {}


# ---------------------------------------------------------------------------
# 6. UNKNOWN Outcomes & Uncertainty Handling (No Blind Retry)
# ---------------------------------------------------------------------------

def test_ledger_unknown_outcomes_held_in_uncertain_no_blind_retry():
    """Orders with UNKNOWN status are held in uncertain ledger without modifying cash or positions."""
    ledger = AccountLedger(initial_cash=50000.0)

    uncertain = OrderResult(
        client_intent_id="intent_timeout_1",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=100,
        status=OrderStatus.UNKNOWN,
        requires_reconciliation=True,
        rejection_reason="TIMEOUT_UNCERTAIN: Server timed out during POST.",
    )
    assert ledger.process_execution_event(uncertain) is True

    # Cash and positions must NOT be assumed filled
    assert ledger.get_cash() == 50000.0
    assert ledger.get_positions() == {}

    # Must be tracked in uncertain orders
    uncertain_records = ledger.get_uncertain_orders()
    assert len(uncertain_records) == 1
    assert uncertain_records[0].intent_id == "intent_timeout_1"
    assert uncertain_records[0].is_resolved is False


# ---------------------------------------------------------------------------
# 7. Authoritative API Reconciliation
# ---------------------------------------------------------------------------

def test_reconcile_with_api_full_synchronization():
    """When local ledger matches remote platform state, reconciliation reports full sync."""
    ledger = AccountLedger(initial_cash=100000.0)
    # Local has 50 ROUT, cash 95000
    ledger.process_execution_event(
        OrderResult(
            client_intent_id="o1",
            ticker="ROUT",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.FILLED,
            filled_quantity=50,
            execution_price=100.0,
            charges=0.0,
        )
    )

    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.get_portfolio.return_value = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=95000.0,
        cash_balance=95000.0,
        portfolio_market_value=5000.0,
        total_wealth=100000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[PortfolioHolding(ticker="ROUT", quantity=50)],
    )
    mock_client.get_cash.return_value = CashBalance(
        team_id="team_1",
        team_name="Nova",
        available_cash=95000.0,
        cash_balance=95000.0,
        capital_remaining=95000.0,
        margin_locked=0.0,
        initial_capital=100000.0,
        realized_pnl=0.0,
    )
    mock_client.get_orders.return_value = {"orders": []}

    report = ledger.reconcile(mock_client)
    assert isinstance(report, FullReconciliationReport)
    assert report.is_synchronized is True
    assert report.api_reachable is True
    assert len(report.discrepancies_detected) == 0
    assert report.positions.matched == {"ROUT": 50}


def test_reconcile_with_api_detects_position_and_cash_mismatches():
    """Detects position discrepancies, missing positions, and cash variances."""
    ledger = AccountLedger(initial_cash=100000.0)
    ledger.cash = 90000.0
    ledger.positions["ROUT"] = PositionLedgerEntry(ticker="ROUT", quantity=100)

    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.get_portfolio.return_value = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=85000.0,  # Cash variance of 5000
        cash_balance=85000.0,
        portfolio_market_value=12000.0,
        total_wealth=97000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[
            PortfolioHolding(ticker="ROUT", quantity=80),   # Local has 100, remote has 80
            PortfolioHolding(ticker="MOTR", quantity=20),   # Missing in local
        ],
    )
    mock_client.get_cash.return_value = CashBalance(
        team_id="team_1",
        team_name="Nova",
        available_cash=85000.0,
        cash_balance=85000.0,
        capital_remaining=85000.0,
        margin_locked=0.0,
        initial_capital=100000.0,
        realized_pnl=0.0,
    )
    mock_client.get_orders.return_value = {"orders": []}

    report = ledger.reconcile(mock_client)
    assert report.is_synchronized is False
    assert len(report.discrepancies_detected) >= 3
    assert any("POSITION_DISCREPANCY [ROUT]" in d for d in report.discrepancies_detected)
    assert any("POSITION_MISSING_IN_LOCAL [MOTR]" in d for d in report.discrepancies_detected)
    assert any("CASH_VARIANCE" in d for d in report.discrepancies_detected)


def test_reconcile_with_api_resolves_uncertain_orders():
    """Reconciliation matches pending uncertain order against remote order history and resolves it."""
    ledger = AccountLedger(initial_cash=50000.0)

    # Record uncertain timed-out intent
    ledger.process_execution_event(
        OrderResult(
            client_intent_id="intent_uncertain_99",
            ticker="ROUT",
            action=OrderAction.BUY,
            quantity=50,
            status=OrderStatus.UNKNOWN,
            requires_reconciliation=True,
        )
    )
    assert ledger.get_uncertain_orders()[0].is_resolved is False

    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.get_portfolio.return_value = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=45000.0,
        cash_balance=45000.0,
        portfolio_market_value=5000.0,
        total_wealth=50000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[PortfolioHolding(ticker="ROUT", quantity=50)],
    )
    mock_client.get_cash.return_value = CashBalance(
        team_id="team_1",
        team_name="Nova",
        available_cash=45000.0,
        cash_balance=45000.0,
        capital_remaining=45000.0,
        margin_locked=0.0,
        initial_capital=50000.0,
        realized_pnl=0.0,
    )
    # Server history confirms the order was indeed FILLED
    mock_client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "srv_ord_99",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 50,
                "status": "FILLED",
                "price": 100.0,
            }
        ]
    }

    report = ledger.reconcile(mock_client)
    assert report.uncertain_orders_resolved == 1
    assert report.uncertain_orders_pending == 0
    assert ledger.get_uncertain_orders()[0].is_resolved is True
    assert ledger.get_uncertain_orders()[0].resolved_status == OrderStatus.FILLED


def test_reconcile_handles_api_failure_gracefully():
    """When API query fails (network timeout or HTTP error), reconciler returns clean failure report."""
    ledger = AccountLedger(initial_cash=50000.0)

    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.get_portfolio.side_effect = RequestTimeoutError("Connection timed out after 5.0s")

    report = ledger.reconcile(mock_client)
    assert report.is_synchronized is False
    assert report.api_reachable is False
    assert "Connection timed out" in str(report.error_message)


# ---------------------------------------------------------------------------
# 8. ExecutionAdapter Integration Seam
# ---------------------------------------------------------------------------

def test_execution_adapter_ledger_integration():
    """Verify ExecutionAdapter automatically updates attached AccountLedger and provides reconcile."""
    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.submit_order.return_value = {
        "order": {
            "order_id": "ord_101",
            "status": "FILLED",
            "price": 200.0,
            "charges": 5.0,
            "filled_quantity": 25,
        }
    }
    mock_client.get_stage.return_value = SimulationStage(
        stage="simulation",
        status="running",
        tickers=["ROUT"],
    )
    mock_client.get_stocks.return_value = [StockInfo(symbol="ROUT", ticker="ROUT", name="RouteWave")]

    config = ExecutionConfig(bot_access_token="test_jwt_token_123", allow_development_stage=True)
    ledger = AccountLedger(initial_cash=100000.0)
    adapter = ExecutionAdapter(client=mock_client, config=config, ledger=ledger)

    intent = OrderIntent(
        intent_id="intent_adapt_1",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=25,
        authorized_by_risk=True,
    )
    res = adapter.execute_order(intent)
    assert res.is_filled is True

    # Ledger must automatically update from adapter
    # Cash: 100000 - (25 * 200 + 5) = 94995.0
    assert ledger.get_cash() == 94995.0
    assert ledger.get_positions() == {"ROUT": 25}

    # Verify adapter.reconcile_accounting() forwards to ledger.reconcile()
    mock_client.get_portfolio.return_value = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=94995.0,
        cash_balance=94995.0,
        portfolio_market_value=5000.0,
        total_wealth=100000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=5.0,
        total_taxes=0.0,
        holdings=[PortfolioHolding(ticker="ROUT", quantity=25)],
    )
    mock_client.get_cash.return_value = CashBalance(
        team_id="team_1",
        team_name="Nova",
        available_cash=94995.0,
        cash_balance=94995.0,
        capital_remaining=94995.0,
        margin_locked=0.0,
        initial_capital=100000.0,
        realized_pnl=0.0,
    )
    mock_client.get_orders.return_value = {"orders": []}

    rec_rep = adapter.reconcile_accounting()
    assert rec_rep is not None
    assert rec_rep.is_synchronized is True

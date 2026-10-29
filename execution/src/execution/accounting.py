"""Portfolio accounting and reconciliation helpers for NOVA.

Provides:
1. Stateful AccountLedger tracking local cash, positions, fees/charges, and realized PnL.
2. Idempotent execution event processing preventing double-counting on duplicated events.
3. Accurate incremental partial fill tracking.
4. Non-retrying uncertain order handling for UNKNOWN outcomes.
5. End-to-end reconciliation against authoritative Trading API endpoints.
6. Robust detection and reporting of position variances, cash discrepancies, and malformed responses.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from execution.client import (
    RequestTimeoutError,
    TradingApiClient,
    TradingApiError,
)
from execution.models import (
    CashBalance,
    ExecutionEvent,
    OrderAction,
    OrderResult,
    OrderStatus,
    Portfolio,
    PortfolioHolding,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionReconciliationReport:
    """Report summarizing comparison between local and remote positions."""
    matched: dict[str, int]
    discrepancies: dict[str, dict[str, int]]  # ticker -> {'local': X, 'remote': Y, 'diff': Z}
    missing_in_local: dict[str, int]
    missing_in_remote: dict[str, int]
    is_synchronized: bool


@dataclass(frozen=True)
class CashReconciliationReport:
    """Report summarizing comparison between local and remote cash."""
    local_cash: float
    remote_cash: float
    variance: float
    is_synchronized: bool


@dataclass
class PositionLedgerEntry:
    """Local accounting record for an individual stock position."""
    ticker: str
    quantity: int = 0
    total_cost: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    last_update_time: Optional[datetime] = None

    @property
    def avg_price(self) -> float:
        """Average execution entry price for the position."""
        return self.total_cost / abs(self.quantity) if self.quantity != 0 else 0.0


@dataclass
class UncertainOrderRecord:
    """Record of an order with an UNKNOWN outcome requiring manual or API reconciliation."""
    intent_id: str
    ticker: str
    action: OrderAction
    quantity: int
    timestamp: datetime
    rejection_reason: str
    is_resolved: bool = False
    resolved_status: Optional[OrderStatus] = None


@dataclass(frozen=True)
class FullReconciliationReport:
    """Comprehensive reconciliation report across positions, cash, and uncertain orders."""
    timestamp: datetime
    positions: PositionReconciliationReport
    cash: CashReconciliationReport
    uncertain_orders_pending: int
    uncertain_orders_resolved: int
    discrepancies_detected: list[str]
    is_synchronized: bool
    api_reachable: bool
    error_message: Optional[str] = None


def reconcile_positions(
    local_positions: dict[str, int],
    remote_portfolio: Portfolio,
) -> PositionReconciliationReport:
    """Reconcile local position tracker against remote exchange portfolio.

    Args:
        local_positions: Mapping of ticker to quantity from internal ledger.
        remote_portfolio: Official Portfolio instance from competition API.

    Returns:
        PositionReconciliationReport instance.
    """
    remote_dict: dict[str, int] = {}
    for h in remote_portfolio.holdings:
        remote_dict[h.ticker.upper()] = h.quantity

    local_dict: dict[str, int] = {k.upper(): v for k, v in local_positions.items()}

    matched: dict[str, int] = {}
    discrepancies: dict[str, dict[str, int]] = {}
    missing_in_local: dict[str, int] = {}
    missing_in_remote: dict[str, int] = {}

    all_tickers = set(local_dict.keys()) | set(remote_dict.keys())

    for ticker in all_tickers:
        loc_qty = local_dict.get(ticker)
        rem_qty = remote_dict.get(ticker)

        if loc_qty is not None and rem_qty is not None:
            if loc_qty == rem_qty:
                matched[ticker] = loc_qty
            else:
                discrepancies[ticker] = {
                    "local": loc_qty,
                    "remote": rem_qty,
                    "diff": rem_qty - loc_qty,
                }
        elif loc_qty is None and rem_qty is not None:
            if rem_qty != 0:
                missing_in_local[ticker] = rem_qty
        elif loc_qty is not None and rem_qty is None:
            if loc_qty != 0:
                missing_in_remote[ticker] = loc_qty

    is_sync = len(discrepancies) == 0 and len(missing_in_local) == 0 and len(missing_in_remote) == 0

    return PositionReconciliationReport(
        matched=matched,
        discrepancies=discrepancies,
        missing_in_local=missing_in_local,
        missing_in_remote=missing_in_remote,
        is_synchronized=is_sync,
    )


def reconcile_cash(
    local_cash: float,
    remote_cash: CashBalance,
    tolerance: float = 0.01,
) -> CashReconciliationReport:
    """Reconcile local cash tracker against remote available cash balance.

    Args:
        local_cash: Internal tracked cash amount.
        remote_cash: CashBalance from GET /api/v1/portfolio/cash.
        tolerance: Allowed decimal variance for rounding.

    Returns:
        CashReconciliationReport instance.
    """
    variance = remote_cash.available_cash - local_cash
    is_sync = abs(variance) <= tolerance

    return CashReconciliationReport(
        local_cash=local_cash,
        remote_cash=remote_cash.available_cash,
        variance=variance,
        is_synchronized=is_sync,
    )


class AccountLedger:
    """Thread-safe portfolio accounting ledger with idempotent event processing and reconciliation."""

    def __init__(self, initial_cash: float = 0.0) -> None:
        """Initialize AccountLedger.

        Args:
            initial_cash: Starting cash balance in currency units.
        """
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, PositionLedgerEntry] = {}
        self.total_charges: float = 0.0
        self.total_realized_pnl: float = 0.0
        self._peak_portfolio_value: float = float(initial_cash)

        # Idempotency and fill tracking
        self._processed_event_ids: set[str] = set()
        self._cumulative_fills_by_intent: dict[str, int] = {}
        self._uncertain_orders: dict[str, UncertainOrderRecord] = {}
        self._lock = threading.RLock()

    def reset(self, initial_cash: float = 0.0) -> None:
        """Reset ledger state to initial conditions."""
        with self._lock:
            self.initial_cash = float(initial_cash)
            self.cash = float(initial_cash)
            self._peak_portfolio_value = float(initial_cash)
            self.positions.clear()
            self.total_charges = 0.0
            self.total_realized_pnl = 0.0
            self._processed_event_ids.clear()
            self._cumulative_fills_by_intent.clear()
            self._uncertain_orders.clear()

    def get_positions(self) -> dict[str, int]:
        """Return non-zero positions mapping ticker to quantity."""
        with self._lock:
            return {
                ticker: entry.quantity
                for ticker, entry in self.positions.items()
                if entry.quantity != 0
            }

    def get_position(self, ticker: str) -> int:
        """Return held share quantity for a specific ticker symbol."""
        with self._lock:
            entry = self.positions.get(ticker.upper())
            return entry.quantity if entry is not None else 0

    def get_cash(self) -> float:
        """Return current tracked cash balance."""
        with self._lock:
            return self.cash

    def get_portfolio_value(self, current_prices: Optional[dict[str, float]] = None) -> float:
        """Calculate total portfolio value based on cash and position market values."""
        with self._lock:
            val = self.cash
            prices = current_prices or {}
            for ticker, entry in self.positions.items():
                if entry.quantity != 0:
                    px = prices.get(ticker.upper(), entry.avg_price)
                    val += entry.quantity * px
            return val

    def get_drawdown(self, current_prices: Optional[dict[str, float]] = None) -> float:
        """Return current drawdown fraction relative to peak portfolio value."""
        with self._lock:
            current_val = self.get_portfolio_value(current_prices)
            if self._peak_portfolio_value <= 0.0:
                return 0.0
            if current_val >= self._peak_portfolio_value:
                self._peak_portfolio_value = current_val
                return 0.0
            return (self._peak_portfolio_value - current_val) / self._peak_portfolio_value

    def set_peak_portfolio_value(self, peak_value: float) -> None:
        """Explicitly set peak portfolio value for stress-testing drawdown circuit breakers."""
        with self._lock:
            self._peak_portfolio_value = float(peak_value)

    def get_uncertain_orders(self) -> list[UncertainOrderRecord]:
        """Return all uncertain order records."""
        with self._lock:
            return list(self._uncertain_orders.values())

    def process_execution_event(
        self,
        event: ExecutionEvent | OrderResult,
        event_id: Optional[str] = None,
    ) -> bool:
        """Process an execution event or order result into the local accounting ledger.

        Guarantees:
        1. Idempotency: Duplicate event IDs or duplicate fill quantities are never double-counted.
        2. Strict PnL & Cash update: Handles BUY, SELL, SHORT, COVER and charges correctly.
        3. Uncertainty Tracking: UNKNOWN outcomes are held in uncertain ledger, no blind retries.
        4. Partial Fills: Incremental quantity delta is tracked accurately.

        Returns:
            bool: True if state was updated or newly recorded, False if rejected as duplicate.
        """
        result = event.order_result if isinstance(event, ExecutionEvent) else event

        # Generate deterministic deduplication key
        event_key = event_id or f"{result.client_intent_id}:{result.status.value}:{result.filled_quantity}"

        with self._lock:
            # 1. Idempotency check: Reject duplicate event IDs
            if event_key in self._processed_event_ids:
                logger.info("[ACCOUNTING:DUPLICATE] Skipping duplicate execution event '%s'", event_key)
                return False

            # 2. Handle UNKNOWN outcomes: Hold in uncertain ledger without blind retries
            if result.status == OrderStatus.UNKNOWN or result.requires_reconciliation:
                logger.warning(
                    "[ACCOUNTING:UNKNOWN] Order intent %s marked UNKNOWN. Retaining in uncertain ledger for reconciliation.",
                    result.client_intent_id,
                )
                self._uncertain_orders[result.client_intent_id] = UncertainOrderRecord(
                    intent_id=result.client_intent_id,
                    ticker=result.ticker.upper(),
                    action=result.action,
                    quantity=result.quantity,
                    timestamp=result.timestamp,
                    rejection_reason=result.rejection_reason or "UNKNOWN_EXECUTION_OUTCOME",
                )
                self._processed_event_ids.add(event_key)
                return True

            # 3. Handle Fills (FILLED and PARTIALLY_FILLED)
            if result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                total_filled = result.filled_quantity
                prev_filled = self._cumulative_fills_by_intent.get(result.client_intent_id, 0)
                delta_qty = total_filled - prev_filled

                # If no new shares filled, skip to prevent double-counting
                if delta_qty <= 0:
                    logger.info(
                        "[ACCOUNTING:FILL] Intent %s fill quantity %d <= previously processed %d. Ignoring.",
                        result.client_intent_id,
                        total_filled,
                        prev_filled,
                    )
                    self._processed_event_ids.add(event_key)
                    return False

                price = float(result.execution_price or 0.0)
                charges = float(result.charges or 0.0)
                ticker = result.ticker.upper()

                if ticker not in self.positions:
                    self.positions[ticker] = PositionLedgerEntry(ticker=ticker)
                pos = self.positions[ticker]

                # Update cash, position, and realized PnL per order action
                if result.action == OrderAction.BUY:
                    cost = delta_qty * price
                    self.cash -= (cost + charges)
                    pos.total_cost += cost
                    pos.quantity += delta_qty

                elif result.action == OrderAction.SELL:
                    proceeds = delta_qty * price
                    self.cash += (proceeds - charges)
                    entry_avg = pos.avg_price
                    realized = (price - entry_avg) * delta_qty - charges
                    pos.realized_pnl += realized
                    self.total_realized_pnl += realized
                    pos.quantity -= delta_qty
                    pos.total_cost = max(0.0, pos.quantity * entry_avg)

                elif result.action == OrderAction.SHORT:
                    proceeds = delta_qty * price
                    self.cash += (proceeds - charges)
                    pos.quantity -= delta_qty
                    pos.total_cost += (delta_qty * price)

                elif result.action == OrderAction.COVER:
                    cost = delta_qty * price
                    self.cash -= (cost + charges)
                    entry_avg = pos.avg_price
                    realized = (entry_avg - price) * delta_qty - charges
                    pos.realized_pnl += realized
                    self.total_realized_pnl += realized
                    pos.quantity += delta_qty
                    pos.total_cost = max(0.0, abs(pos.quantity) * entry_avg)

                pos.total_fees += charges
                pos.last_update_time = result.timestamp
                self.total_charges += charges
                self._cumulative_fills_by_intent[result.client_intent_id] = total_filled
                self._processed_event_ids.add(event_key)

                # Resolve uncertain record if previously marked unknown
                if result.client_intent_id in self._uncertain_orders:
                    self._uncertain_orders[result.client_intent_id].is_resolved = True
                    self._uncertain_orders[result.client_intent_id].resolved_status = result.status

                return True

            # 4. Handle Rejected or Cancelled outcomes
            if result.status in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                self._processed_event_ids.add(event_key)
                if result.client_intent_id in self._uncertain_orders:
                    self._uncertain_orders[result.client_intent_id].is_resolved = True
                    self._uncertain_orders[result.client_intent_id].resolved_status = result.status
                return True

            # Other statuses (PENDING, SUBMITTED)
            self._processed_event_ids.add(event_key)
            return True

    def reconcile(
        self,
        client: TradingApiClient,
        tolerance: float = 0.01,
        auto_resolve_uncertain: bool = True,
    ) -> FullReconciliationReport:
        """Perform end-to-end reconciliation against authoritative platform Trading API.

        Queries:
        1. GET /api/v1/portfolio (authoritative stock positions)
        2. GET /api/v1/portfolio/cash (authoritative cash balance)
        3. GET /api/v1/orders (order history to resolve uncertain timeouts)

        Args:
            client: TradingApiClient instance.
            tolerance: Allowed decimal cash rounding variance (default: 0.01).
            auto_resolve_uncertain: If True, resolves pending uncertain orders found in history.

        Returns:
            FullReconciliationReport detailing matched positions, discrepancies, and variances.
        """
        now = datetime.now(timezone.utc)
        discrepancies: list[str] = []

        # 1. Authoritative API Queries
        try:
            remote_portfolio = client.get_portfolio()
            remote_cash = client.get_cash()
            remote_orders_resp = client.get_orders(limit=50)
            remote_orders = remote_orders_resp.get("orders", []) if isinstance(remote_orders_resp, dict) else []
        except (TradingApiError, RequestTimeoutError, Exception) as exc:
            logger.error("Reconciliation failed: Error contacting platform Trading API: %s", exc)
            return FullReconciliationReport(
                timestamp=now,
                positions=PositionReconciliationReport(
                    matched={},
                    discrepancies={},
                    missing_in_local={},
                    missing_in_remote={},
                    is_synchronized=False,
                ),
                cash=CashReconciliationReport(
                    local_cash=self.cash,
                    remote_cash=0.0,
                    variance=0.0,
                    is_synchronized=False,
                ),
                uncertain_orders_pending=len([u for u in self._uncertain_orders.values() if not u.is_resolved]),
                uncertain_orders_resolved=len([u for u in self._uncertain_orders.values() if u.is_resolved]),
                discrepancies_detected=[f"API_QUERY_FAILED: {exc}"],
                is_synchronized=False,
                api_reachable=False,
                error_message=str(exc),
            )

        # 2. Reconcile Positions
        pos_report = reconcile_positions(self.get_positions(), remote_portfolio)
        for ticker, disc in pos_report.discrepancies.items():
            discrepancies.append(
                f"POSITION_DISCREPANCY [{ticker}]: local={disc['local']}, remote={disc['remote']} (diff={disc['diff']})"
            )
        for ticker, qty in pos_report.missing_in_local.items():
            discrepancies.append(f"POSITION_MISSING_IN_LOCAL [{ticker}]: remote has {qty} shares unaccounted in local")
        for ticker, qty in pos_report.missing_in_remote.items():
            discrepancies.append(f"POSITION_MISSING_IN_REMOTE [{ticker}]: local has {qty} shares not found in remote")

        # 3. Reconcile Cash
        cash_report = reconcile_cash(self.cash, remote_cash, tolerance=tolerance)
        if not cash_report.is_synchronized:
            discrepancies.append(
                f"CASH_VARIANCE: local={cash_report.local_cash:.2f}, remote={cash_report.remote_cash:.2f} "
                f"(variance={cash_report.variance:.2f} > tolerance={tolerance})"
            )

        # 4. Resolve Uncertain Orders via Authoritative Order History
        resolved_count = 0
        pending_count = 0

        with self._lock:
            for intent_id, rec in self._uncertain_orders.items():
                if rec.is_resolved:
                    resolved_count += 1
                    continue

                if auto_resolve_uncertain and remote_orders:
                    # Look for matching order in remote history
                    matched_order = None
                    for o in remote_orders:
                        if not isinstance(o, dict):
                            continue
                        o_ticker = str(o.get("ticker", "")).upper()
                        o_action = str(o.get("action", "")).upper()
                        o_qty = int(o.get("quantity", 0))

                        if o_ticker == rec.ticker and o_action == rec.action.value and o_qty == rec.quantity:
                            matched_order = o
                            break

                    if matched_order is not None:
                        raw_status = str(matched_order.get("status", "")).upper()
                        if raw_status in ("FILLED", "SUCCESS"):
                            rec.is_resolved = True
                            rec.resolved_status = OrderStatus.FILLED
                            resolved_count += 1
                            logger.info("Reconciliation resolved uncertain order %s as FILLED.", intent_id)
                        elif raw_status in ("REJECTED", "FAILED", "CANCELLED"):
                            rec.is_resolved = True
                            rec.resolved_status = OrderStatus.REJECTED
                            resolved_count += 1
                            logger.info("Reconciliation resolved uncertain order %s as REJECTED.", intent_id)
                        else:
                            pending_count += 1
                            discrepancies.append(f"UNCERTAIN_ORDER_PENDING [{intent_id}]: Server status is '{raw_status}'")
                    else:
                        pending_count += 1
                        discrepancies.append(f"UNCERTAIN_ORDER_UNRESOLVED [{intent_id}]: Not found in remote order history")
                else:
                    pending_count += 1
                    discrepancies.append(f"UNCERTAIN_ORDER_PENDING [{intent_id}]")

        is_sync = pos_report.is_synchronized and cash_report.is_synchronized and (pending_count == 0)

        for d in discrepancies:
            logger.warning("[ACCOUNTING:DISCREPANCY] %s", d)

        logger.info(
            "[ACCOUNTING:RECONCILE] Completed: synchronized=%s positions_sync=%s cash_sync=%s pending_uncertain=%d",
            is_sync,
            pos_report.is_synchronized,
            cash_report.is_synchronized,
            pending_count,
        )

        return FullReconciliationReport(
            timestamp=now,
            positions=pos_report,
            cash=cash_report,
            uncertain_orders_pending=pending_count,
            uncertain_orders_resolved=resolved_count,
            discrepancies_detected=discrepancies,
            is_synchronized=is_sync,
            api_reachable=True,
        )

    def get_health_status(self) -> dict[str, Any]:
        """Return structured accounting health and position diagnostics."""
        with self._lock:
            active_positions = {k: v.quantity for k, v in self.positions.items() if v.quantity != 0}
            pending_uncertain = len([u for u in self._uncertain_orders.values() if not u.is_resolved])
            return {
                "cash": self.cash,
                "active_positions_count": len(active_positions),
                "total_charges": self.total_charges,
                "total_realized_pnl": self.total_realized_pnl,
                "processed_events_count": len(self._processed_event_ids),
                "uncertain_orders_pending": pending_uncertain,
                "is_healthy": pending_uncertain == 0,
            }

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


@dataclass(frozen=True)
class PortfolioSummary:
    """Point-in-time portfolio summary metrics."""
    available_cash: float
    total_portfolio_value: float
    drawdown_fraction: float
    positions_count: int
    total_realized_pnl: float
    total_charges: float


@dataclass
class PositionLedgerEntry:
    """Local accounting record for an individual stock position."""
    ticker: str
    quantity: int = 0
    total_cost: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    last_update_time: Optional[datetime] = None
    locked_margin: float = 0.0
    holding_days: int = 0

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
    filled_quantity: int = 0

    @property
    def remaining_quantity(self) -> int:
        """Unfilled quantity remaining for this order."""
        return max(0, self.quantity - self.filled_quantity)


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
        side = getattr(h, "side", None)
        if side is None and hasattr(h, "raw_response") and isinstance(h.raw_response, dict):
            side = h.raw_response.get("side") or h.raw_response.get("position_type")
        qty = h.quantity
        if side and str(side).upper() in ("SHORT", "SELL") and qty > 0:
            qty = -qty
        remote_dict[h.ticker.upper()] = qty

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
        self._cumulative_charges_by_intent: dict[str, float] = {}
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
            self._cumulative_charges_by_intent.clear()
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

    @property
    def total_locked_margin(self) -> float:
        """Total locked margin collateral across all active short positions."""
        with self._lock:
            return sum(pos.locked_margin for pos in self.positions.values() if pos.quantity < 0)

    @property
    def available_cash(self) -> float:
        """Spendable available cash after deducting locked short margin collateral."""
        with self._lock:
            return max(0.0, self.cash - sum(pos.locked_margin for pos in self.positions.values() if pos.quantity < 0))

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

    def get_portfolio_summary(self, current_prices: Optional[dict[str, float]] = None) -> PortfolioSummary:
        """Return point-in-time portfolio summary."""
        with self._lock:
            val = self.get_portfolio_value(current_prices)
            dd = self.get_drawdown(current_prices)
            avail_c = max(0.0, self.cash - sum(pos.locked_margin for pos in self.positions.values() if pos.quantity < 0))
            return PortfolioSummary(
                available_cash=avail_c,
                total_portfolio_value=val,
                drawdown_fraction=dd,
                positions_count=len(self.get_positions()),
                total_realized_pnl=self.total_realized_pnl,
                total_charges=self.total_charges,
            )

    def get_portfolio_state(self, current_prices: Optional[dict[str, float]] = None) -> Any:
        """Construct authoritative Person 3 PortfolioState from local accounting records.

        Bridges Person 1 AccountLedger to Person 3 Decision/Risk subsystems without
        creating parallel state or duplicating tracking.
        """
        with self._lock:
            try:
                from nova.decision.contracts import PortfolioState, PositionState
            except ImportError:
                return None

            prices = current_prices or {}
            pos_states: dict[str, PositionState] = {}
            total_locked = 0.0

            for ticker, entry in self.positions.items():
                if entry.quantity == 0:
                    continue
                px = prices.get(ticker.upper(), entry.avg_price)
                if entry.quantity > 0:
                    unrealized = (px - entry.avg_price) * entry.quantity
                    pos_locked = 0.0
                else:
                    unrealized = (entry.avg_price - px) * abs(entry.quantity)
                    pos_locked = max(0.0, entry.locked_margin)
                    total_locked += pos_locked

                pos_states[ticker.upper()] = PositionState(
                    ticker=ticker.upper(),
                    quantity=float(entry.quantity),
                    entry_price=entry.avg_price,
                    current_market_price=px,
                    holding_days=entry.holding_days,
                    unrealized_pnl=unrealized,
                    realized_pnl=entry.realized_pnl,
                    locked_margin=pos_locked,
                    highest_price=px,
                    lowest_price=px,
                    peak_unrealized_pnl=max(0.0, unrealized),
                )

            return PortfolioState(
                available_cash=max(0.0, self.cash - total_locked),
                locked_margin=total_locked,
                positions=pos_states,
                initial_capital=self.initial_cash,
                peak_wealth=self._peak_portfolio_value,
                realized_pnl_total=self.total_realized_pnl,
                fees_paid_total=self.total_charges,
            )

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
                    trade_val = delta_qty * price
                    self.cash += (trade_val - charges)
                    pos.locked_margin += trade_val
                    pos.quantity -= delta_qty
                    pos.total_cost += trade_val

                elif result.action == OrderAction.COVER:
                    trade_val = delta_qty * price
                    self.cash -= (trade_val + charges)
                    prior_short = abs(pos.quantity) if pos.quantity < 0 else 0
                    entry_avg = pos.avg_price
                    realized = (entry_avg - price) * delta_qty - charges
                    pos.realized_pnl += realized
                    self.total_realized_pnl += realized

                    # Proportional or complete margin release
                    if prior_short > 0:
                        if delta_qty >= prior_short:
                            # Full cover: release all remaining locked margin
                            pos.locked_margin = 0.0
                            pos.quantity = 0
                            pos.total_cost = 0.0
                        else:
                            # Partial cover: release margin proportionally
                            release_ratio = delta_qty / prior_short
                            margin_released = pos.locked_margin * release_ratio
                            pos.locked_margin = max(0.0, pos.locked_margin - margin_released)
                            pos.quantity += delta_qty
                            pos.total_cost = max(0.0, abs(pos.quantity) * entry_avg)
                    else:
                        pos.quantity = 0
                        pos.locked_margin = 0.0
                        pos.total_cost = 0.0

                    pos.last_update_time = result.timestamp

                pos.total_fees += charges
                pos.last_update_time = result.timestamp
                self.total_charges += charges
                self._cumulative_fills_by_intent[result.client_intent_id] = total_filled
                self._cumulative_charges_by_intent[result.client_intent_id] = (
                    self._cumulative_charges_by_intent.get(result.client_intent_id, 0.0) + charges
                )
                self._processed_event_ids.add(event_key)

                # Resolve uncertain record if previously marked unknown
                if result.client_intent_id in self._uncertain_orders:
                    rec = self._uncertain_orders[result.client_intent_id]
                    if result.status == OrderStatus.FILLED:
                        rec.is_resolved = True
                        rec.resolved_status = OrderStatus.FILLED
                        rec.filled_quantity = total_filled
                    elif result.status == OrderStatus.PARTIALLY_FILLED:
                        rec.is_resolved = False
                        rec.resolved_status = OrderStatus.PARTIALLY_FILLED
                        rec.filled_quantity = total_filled

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
            remote_orders: list[dict[str, Any]] = []

            # Fetch order history with pagination support
            page = 1
            max_pages = 5
            while page <= max_pages:
                try:
                    resp = client.get_orders(page=page, limit=50)
                except TypeError:
                    # Fallback for test mocks that do not accept page argument
                    resp = client.get_orders(limit=50)
                    orders_page = resp.get("orders", []) if isinstance(resp, dict) else []
                    remote_orders.extend(orders_page)
                    break

                orders_page = resp.get("orders", []) if isinstance(resp, dict) else []
                remote_orders.extend(orders_page)
                if len(orders_page) < 50:
                    break
                page += 1
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

        # 2. Resolve Uncertain Orders via Authoritative Order History FIRST
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
                        # Match by client_intent_id / client_order_id if available, else by ticker, action, quantity
                        o_client_id = str(o.get("client_intent_id") or o.get("client_order_id") or o.get("tag") or "")
                        if o_client_id and o_client_id == intent_id:
                            matched_order = o
                            break

                        o_ticker = str(o.get("ticker", "")).upper()
                        o_action = str(o.get("action", "")).upper()
                        o_qty = int(o.get("quantity", 0))

                        if o_ticker == rec.ticker and o_action == rec.action.value and o_qty == rec.quantity:
                            matched_order = o
                            break

                    if matched_order is not None:
                        raw_status = str(matched_order.get("status", "")).upper()
                        if raw_status in ("FILLED", "SUCCESS"):
                            # Validate price and quantity before applying
                            px_val = matched_order.get("price") if matched_order.get("price") is not None else matched_order.get("execution_price")
                            try:
                                fill_px = float(px_val) if px_val is not None else 0.0
                            except (TypeError, ValueError):
                                fill_px = 0.0

                            qty_val = matched_order.get("filled_quantity") if matched_order.get("filled_quantity") is not None else matched_order.get("quantity", rec.quantity)
                            try:
                                remote_filled = int(qty_val) if qty_val is not None else rec.quantity
                            except (TypeError, ValueError):
                                remote_filled = 0

                            if fill_px <= 0.0 or remote_filled <= 0:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INVALID_DETAILS [{intent_id}]: Cannot confirm FILLED details (px={px_val}, qty={qty_val})"
                                )
                                continue

                            prev_filled = self._cumulative_fills_by_intent.get(intent_id, rec.filled_quantity)
                            delta_qty = remote_filled - prev_filled

                            if delta_qty < 0:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INCONSISTENT_DETAILS [{intent_id}]: remote filled {remote_filled} < local recorded {prev_filled}"
                                )
                                continue

                            # Calculate delta charges
                            charges_val = matched_order.get("charges") if matched_order.get("charges") is not None else matched_order.get("commission", 0.0)
                            try:
                                total_remote_charges = float(charges_val or 0.0)
                            except (TypeError, ValueError):
                                total_remote_charges = 0.0

                            prev_charges = self._cumulative_charges_by_intent.get(intent_id, 0.0)
                            delta_charges = max(0.0, total_remote_charges - prev_charges)

                            if delta_qty > 0:
                                fill_res = OrderResult(
                                    client_intent_id=intent_id,
                                    ticker=rec.ticker,
                                    action=rec.action,
                                    quantity=rec.quantity,
                                    status=OrderStatus.FILLED,
                                    order_id=str(matched_order.get("order_id") or matched_order.get("id") or ""),
                                    execution_price=fill_px,
                                    filled_quantity=remote_filled,
                                    charges=delta_charges,
                                    raw_response=matched_order,
                                )
                                self.process_execution_event(fill_res)

                            rec.is_resolved = True
                            rec.resolved_status = OrderStatus.FILLED
                            rec.filled_quantity = remote_filled
                            resolved_count += 1
                            logger.info(
                                "Reconciliation resolved uncertain order %s as FILLED (total=%d, delta=%d).",
                                intent_id,
                                remote_filled,
                                delta_qty,
                            )

                        elif raw_status in ("PARTIALLY_FILLED", "PARTIAL_FILL", "PARTIAL"):
                            # Validate remote partial fill details: must not infer price or quantity without sufficient evidence
                            qty_val = matched_order.get("filled_quantity") if matched_order.get("filled_quantity") is not None else matched_order.get("filled_qty")
                            if qty_val is None:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_MISSING_DETAILS [{intent_id}]: PARTIALLY_FILLED order missing filled_quantity"
                                )
                                continue

                            try:
                                remote_filled = int(qty_val)
                            except (TypeError, ValueError):
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INVALID_DETAILS [{intent_id}]: Non-numeric filled_quantity '{qty_val}'"
                                )
                                continue

                            if remote_filled <= 0 or remote_filled > rec.quantity:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INCONSISTENT_DETAILS [{intent_id}]: filled_quantity {remote_filled} invalid for total quantity {rec.quantity}"
                                )
                                continue

                            px_val = matched_order.get("price") if matched_order.get("price") is not None else matched_order.get("execution_price")
                            if px_val is None:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_MISSING_DETAILS [{intent_id}]: Missing execution price for partial fill"
                                )
                                continue

                            try:
                                fill_px = float(px_val)
                            except (TypeError, ValueError):
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INVALID_DETAILS [{intent_id}]: Non-numeric execution price '{px_val}'"
                                )
                                continue

                            if fill_px <= 0.0:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INVALID_DETAILS [{intent_id}]: Non-positive execution price {fill_px}"
                                )
                                continue

                            prev_filled = self._cumulative_fills_by_intent.get(intent_id, rec.filled_quantity)
                            if remote_filled < prev_filled:
                                pending_count += 1
                                discrepancies.append(
                                    f"UNCERTAIN_ORDER_INCONSISTENT_DETAILS [{intent_id}]: remote filled {remote_filled} < local recorded {prev_filled}"
                                )
                                continue

                            delta_qty = remote_filled - prev_filled
                            charges_val = matched_order.get("charges") if matched_order.get("charges") is not None else matched_order.get("commission", 0.0)
                            try:
                                total_remote_charges = float(charges_val or 0.0)
                            except (TypeError, ValueError):
                                total_remote_charges = 0.0

                            prev_charges = self._cumulative_charges_by_intent.get(intent_id, 0.0)
                            delta_charges = max(0.0, total_remote_charges - prev_charges)

                            # Apply newly confirmed execution delta if delta_qty > 0
                            if delta_qty > 0:
                                fill_res = OrderResult(
                                    client_intent_id=intent_id,
                                    ticker=rec.ticker,
                                    action=rec.action,
                                    quantity=rec.quantity,
                                    status=OrderStatus.PARTIALLY_FILLED,
                                    order_id=str(matched_order.get("order_id") or matched_order.get("id") or ""),
                                    execution_price=fill_px,
                                    filled_quantity=remote_filled,
                                    charges=delta_charges,
                                    raw_response=matched_order,
                                )
                                self.process_execution_event(fill_res)

                            # Idempotently update uncertain record state: remaining quantity stays outstanding
                            rec.is_resolved = False
                            rec.resolved_status = OrderStatus.PARTIALLY_FILLED
                            rec.filled_quantity = remote_filled
                            pending_count += 1
                            logger.info(
                                "Reconciliation tracked partial fill for uncertain order %s: %d/%d filled (%d remaining, delta=%d).",
                                intent_id,
                                remote_filled,
                                rec.quantity,
                                rec.remaining_quantity,
                                delta_qty,
                            )

                        elif raw_status in ("REJECTED", "FAILED", "CANCELLED"):
                            rec.is_resolved = True
                            rec.resolved_status = OrderStatus.REJECTED if raw_status != "CANCELLED" else OrderStatus.CANCELLED
                            resolved_count += 1
                            logger.info("Reconciliation resolved uncertain order %s as %s.", intent_id, rec.resolved_status.value)
                        else:
                            pending_count += 1
                            discrepancies.append(f"UNCERTAIN_ORDER_PENDING [{intent_id}]: Server status is '{raw_status}'")
                    else:
                        pending_count += 1
                        discrepancies.append(f"UNCERTAIN_ORDER_UNRESOLVED [{intent_id}]: Not found in remote order history")
                else:
                    pending_count += 1
                    discrepancies.append(f"UNCERTAIN_ORDER_PENDING [{intent_id}]")

        # 3. Reconcile Positions against updated local ledger
        pos_report = reconcile_positions(self.get_positions(), remote_portfolio)
        for ticker, disc in pos_report.discrepancies.items():
            discrepancies.append(
                f"POSITION_DISCREPANCY [{ticker}]: local={disc['local']}, remote={disc['remote']} (diff={disc['diff']})"
            )
        for ticker, qty in pos_report.missing_in_local.items():
            discrepancies.append(f"POSITION_MISSING_IN_LOCAL [{ticker}]: remote has {qty} shares unaccounted in local")
        for ticker, qty in pos_report.missing_in_remote.items():
            discrepancies.append(f"POSITION_MISSING_IN_REMOTE [{ticker}]: local has {qty} shares not found in remote")

        # 4. Reconcile Cash against updated local ledger
        cash_report = reconcile_cash(self.cash, remote_cash, tolerance=tolerance)
        if not cash_report.is_synchronized:
            discrepancies.append(
                f"CASH_VARIANCE: local={cash_report.local_cash:.2f}, remote={cash_report.remote_cash:.2f} "
                f"(variance={cash_report.variance:.2f} > tolerance={tolerance})"
            )

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

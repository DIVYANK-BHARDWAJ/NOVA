"""Execution Adapter for NOVA.

Bridges authorized trading decisions from Person 3's risk gateway to the official
competition REST API. Enforces risk authorization, validates simulation stage and
universe, protects against duplicate execution on timeouts, and emits execution events.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from execution.client import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    TradingApiClient,
    TradingApiError,
)
from execution.config import ExecutionConfig
from execution.models import (
    ExecutionEvent,
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    SimulationStage,
    StockInfo,
)

logger = logging.getLogger(__name__)


class ExecutionAdapter:
    """Manages order submission, stage gating, universe validation, and failure reconciliation."""

    def __init__(
        self,
        client: TradingApiClient,
        config: ExecutionConfig,
        on_execution_event: Optional[Callable[[ExecutionEvent], None]] = None,
        ledger: Optional[Any] = None,
    ) -> None:
        """Initialize ExecutionAdapter.

        Args:
            client: TradingApiClient instance.
            config: ExecutionConfig instance.
            on_execution_event: Optional callback for execution event dispatch.
            ledger: Optional AccountLedger instance for stateful position/cash tracking.
        """
        self.client = client
        self.config = config
        self.on_execution_event = on_execution_event
        self.ledger = ledger

        # Caching for simulation stage and stock universe
        self._cached_stage: Optional[SimulationStage] = None
        self._cached_stage_time: float = 0.0

        self._cached_universe: Optional[set[str]] = None
        self._cached_universe_time: float = 0.0

        # Concurrency protection and duplicate-intent tracking
        self._intent_lock = threading.Lock()
        self._active_intents: set[str] = set()
        self._completed_intents: dict[str, OrderResult] = {}
        self._known_order_ids: set[str] = set()

    def _emit_event(self, event_type: str, result: OrderResult) -> None:
        """Dispatch execution event to registered listener and account ledger."""
        if self.ledger is not None:
            try:
                self.ledger.process_execution_event(result)
            except Exception as exc:
                logger.error("Exception in ledger event processing: %s", exc)

        if self.on_execution_event is not None:
            try:
                event = ExecutionEvent(event_type=event_type, order_result=result)
                self.on_execution_event(event)
            except Exception as exc:
                logger.error("Exception in execution event callback: %s", exc)

    def reconcile_accounting(self, tolerance: float = 0.01) -> Optional[Any]:
        """Perform end-to-end reconciliation between local ledger and platform Trading API.

        Args:
            tolerance: Allowed decimal variance for cash rounding.

        Returns:
            FullReconciliationReport if ledger is attached, else None.
        """
        if self.ledger is None:
            logger.warning("reconcile_accounting called, but no AccountLedger is attached to adapter.")
            return None
        return self.ledger.reconcile(client=self.client, tolerance=tolerance)

    def invalidate_stage_cache(self) -> None:
        """Explicitly invalidate the simulation stage cache.

        Ensures subsequent queries hit the server immediately, preventing
        silent order authorization across stage transitions.
        """
        self._cached_stage = None
        self._cached_stage_time = 0.0
        logger.debug("Simulation stage cache invalidated.")

    def invalidate_universe_cache(self) -> None:
        """Explicitly invalidate the tradable stock universe cache."""
        self._cached_universe = None
        self._cached_universe_time = 0.0
        logger.debug("Tradable stock universe cache invalidated.")

    def notify_stage_change(self, stage: Optional[SimulationStage] = None) -> None:
        """Notify the adapter of an external or server-signaled stage transition."""
        self.invalidate_stage_cache()
        if stage is not None:
            self._cached_stage = stage
            self._cached_stage_time = time.time()

    def get_simulation_stage(self, force_refresh: bool = False) -> SimulationStage:
        """Fetch current simulation stage with TTL caching.

        Args:
            force_refresh: If True, bypasses cache.

        Returns:
            SimulationStage instance.
        """
        now = time.time()
        if (
            not force_refresh
            and self.config.stage_cache_ttl_seconds > 0
            and self._cached_stage is not None
            and (now - self._cached_stage_time < self.config.stage_cache_ttl_seconds)
        ):
            return self._cached_stage

        stage = self.client.get_stage()
        self._cached_stage = stage
        self._cached_stage_time = now
        return stage

    def get_tradable_universe(self, force_refresh: bool = False) -> set[str]:
        """Fetch active tradable stock symbols with TTL caching.

        Args:
            force_refresh: If True, bypasses cache.

        Returns:
            Set of active uppercase ticker symbols.
        """
        now = time.time()
        if (
            not force_refresh
            and self.config.stocks_cache_ttl_seconds > 0
            and self._cached_universe is not None
            and (now - self._cached_universe_time < self.config.stocks_cache_ttl_seconds)
        ):
            return self._cached_universe

        stocks: list[StockInfo] = self.client.get_stocks()
        universe: set[str] = {s.ticker.upper() for s in stocks if s.ticker}
        self._cached_universe = universe
        self._cached_universe_time = now
        return universe

    def is_intent_active(self, intent_id: str) -> bool:
        """Check if an intent is currently in-flight."""
        with self._intent_lock:
            return intent_id in self._active_intents

    def is_intent_completed(self, intent_id: str) -> bool:
        """Check if an intent has already been completed."""
        with self._intent_lock:
            return intent_id in self._completed_intents

    def get_completed_order(self, intent_id: str) -> Optional[OrderResult]:
        """Retrieve completed result for a previous intent if present."""
        with self._intent_lock:
            return self._completed_intents.get(intent_id)

    def record_known_order_id(self, order_id: str) -> None:
        """Manually register an external or historical order ID into the known set."""
        if order_id:
            with self._intent_lock:
                self._known_order_ids.add(str(order_id))

    def execute_order(self, intent: OrderIntent) -> OrderResult:
        """Execute an authorized OrderIntent on the competition platform.

        Enforces risk authorization, concurrency uniqueness, stage eligibility,
        universe membership, and quantity validity before sending request. Protects
        against duplicate execution if timeouts occur.

        Args:
            intent: OrderIntent instance from Person 3's risk gateway.

        Returns:
            OrderResult detailing the execution outcome.
        """
        # 1. Concurrent and Duplicate Submission Protection
        with self._intent_lock:
            if intent.intent_id in self._active_intents:
                logger.warning(
                    "[EXECUTION:REJECT] Concurrent duplicate order submission rejected for intent_id %s.", intent.intent_id
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=(
                        f"CONCURRENT_DUPLICATE_INTENT: Intent '{intent.intent_id}' is currently in-flight."
                    ),
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            if intent.intent_id in self._completed_intents:
                prev_result = self._completed_intents[intent.intent_id]
                logger.warning(
                    "[EXECUTION:REJECT] Duplicate order submission rejected for already-completed intent_id %s (status=%s).",
                    intent.intent_id,
                    prev_result.status.value,
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=(
                        f"DUPLICATE_INTENT: Intent '{intent.intent_id}' has already been executed "
                        f"(status={prev_result.status.value})."
                    ),
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            # Register as active intent
            self._active_intents.add(intent.intent_id)
            # Take snapshot of order IDs known before submission
            pre_existing_order_ids = set(self._known_order_ids)
            submission_start_time = time.time()

        result: Optional[OrderResult] = None
        try:
            self._emit_event("ORDER_SUBMISSION_ATTEMPT", OrderResult(
                client_intent_id=intent.intent_id,
                ticker=intent.ticker,
                action=intent.action,
                quantity=intent.quantity,
                status=OrderStatus.PENDING,
            ))

            # 2. Enforce Risk Gateway Authorization
            if not intent.authorized_by_risk:
                logger.warning(
                    "Order intent %s for %s %d %s REJECTED: Not authorized by risk gateway.",
                    intent.intent_id,
                    intent.action.value,
                    intent.quantity,
                    intent.ticker,
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason="UNAUTHORIZED_BY_RISK_GATEWAY: Order intent lacks risk authorization.",
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            # 3. Validate Quantity Invariants (Positive Integer, No Silent Rounding)
            if type(intent.quantity) is not int or intent.quantity <= 0:
                logger.warning(
                    "Order intent %s REJECTED: Invalid quantity %r (type=%s). Must be integer > 0.",
                    intent.intent_id,
                    intent.quantity,
                    type(intent.quantity).__name__,
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=int(intent.quantity) if isinstance(intent.quantity, (int, float)) else 0,
                    status=OrderStatus.REJECTED,
                    rejection_reason=f"INVALID_QUANTITY: Quantity must be a positive integer, got {intent.quantity!r}.",
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            # 4. Check Simulation Stage
            try:
                stage = self.get_simulation_stage()
            except Exception as exc:
                logger.error("Failed to query simulation stage: %s. Blocking order submission.", exc)
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=f"STAGE_QUERY_FAILED: Cannot verify platform stage ({exc}).",
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            if stage.stage.lower() not in [s.lower() for s in self.config.permitted_stages] or not stage.is_trading_active:
                logger.warning(
                    "Trading blocked: Stage '%s' (status '%s') does not permit active trading.",
                    stage.stage,
                    stage.status,
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=(
                        f"STAGE_NOT_PERMITTED: Platform stage is '{stage.stage}' (status: '{stage.status}'). "
                        f"Trading allowed only in {self.config.permitted_stages} with status 'running'."
                    ),
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            # Check development vs live competition mode
            if not self.config.allow_development_stage and (
                stage.stage.lower() == "development" or stage.is_sandbox_stream
            ):
                logger.warning(
                    "Trading blocked: Stage '%s' is sandbox/development, but allow_development_stage is False.",
                    stage.stage,
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=intent.ticker,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=(
                        f"STAGE_DEVELOPMENT_BLOCKED: Platform stage is '{stage.stage}' (sandbox={stage.is_sandbox_stream}), "
                        f"but allow_development_stage is configured to False."
                    ),
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            # 5. Validate Tradable Stock Universe
            ticker_clean = intent.ticker.strip().upper()
            try:
                universe = self.get_tradable_universe()
            except Exception as exc:
                logger.warning("Could not refresh stock universe: %s. Using cached universe if present.", exc)
                universe = self._cached_universe or set()

            if universe and ticker_clean not in universe:
                logger.warning(
                    "Order intent %s REJECTED: Symbol '%s' is not in the active tradable universe.",
                    intent.intent_id,
                    ticker_clean,
                )
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=ticker_clean,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=f"INVALID_TICKER: '{ticker_clean}' is not in active stock universe.",
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            # 6. Order Submission to Official Platform REST API
            logger.info(
                "[EXECUTION:SUBMIT] Submitting order intent %s: %s %d %s",
                intent.intent_id,
                intent.action.value,
                intent.quantity,
                ticker_clean,
            )
            try:
                resp = self.client.submit_order(
                    ticker=ticker_clean,
                    action=intent.action.value,
                    quantity=intent.quantity,
                    date=intent.date,
                )
                result = self._parse_order_submission_response(intent, resp)
                return result

            except RequestTimeoutError as exc:
                # Timeout does not prove rejection! Mark UNKNOWN, do NOT blind retry.
                logger.critical(
                    "[EXECUTION:TIMEOUT] Request timeout during submission for %s %d %s (Intent: %s). "
                    "Order status is UNKNOWN. Withholding automatic retry to prevent duplicate execution.",
                    intent.action.value,
                    intent.quantity,
                    ticker_clean,
                    intent.intent_id,
                )
                result = self._reconcile_timeout_uncertainty(
                    intent=intent,
                    error_msg=str(exc),
                    submission_start_time=submission_start_time,
                    pre_existing_order_ids=pre_existing_order_ids,
                )
                return result

            except (AuthenticationError, AuthorizationError, RateLimitError, TradingApiError) as exc:
                logger.error("[EXECUTION:API_ERROR] API error during order submission for %s: %s", ticker_clean, exc)
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=ticker_clean,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=str(exc),
                    raw_response=getattr(exc, "response_body", None),
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

            except Exception as exc:
                logger.error("[EXECUTION:ERROR] Unexpected error submitting order for %s: %s", ticker_clean, exc, exc_info=True)
                result = OrderResult(
                    client_intent_id=intent.intent_id,
                    ticker=ticker_clean,
                    action=intent.action,
                    quantity=intent.quantity,
                    status=OrderStatus.REJECTED,
                    rejection_reason=f"UNEXPECTED_ERROR: {exc}",
                )
                self._emit_event("ORDER_REJECTED", result)
                return result

        finally:
            with self._intent_lock:
                self._active_intents.discard(intent.intent_id)
                if result is not None:
                    self._completed_intents[intent.intent_id] = result
                    if result.order_id:
                        self._known_order_ids.add(str(result.order_id))

    def _parse_order_submission_response(self, intent: OrderIntent, resp: dict[str, Any]) -> OrderResult:
        """Parse server JSON response into an OrderResult."""
        # Check standard error payloads
        if resp.get("error") or resp.get("status") == "error":
            err_msg = resp.get("message") or resp.get("error") or "Unknown error"
            result = OrderResult(
                client_intent_id=intent.intent_id,
                ticker=intent.ticker,
                action=intent.action,
                quantity=intent.quantity,
                status=OrderStatus.REJECTED,
                rejection_reason=str(err_msg),
                raw_response=resp,
            )
            self._emit_event("ORDER_REJECTED", result)
            return result

        # Extract order ID and status
        order_obj = resp.get("order") if isinstance(resp.get("order"), dict) else resp
        order_id = str(order_obj.get("order_id") or order_obj.get("id") or "")
        raw_status = str(order_obj.get("status", "FILLED")).upper()

        if raw_status in ("FILLED", "SUCCESS", "COMPLETED"):
            status = OrderStatus.FILLED
        elif raw_status in ("PENDING", "SUBMITTED", "ACCEPTED"):
            status = OrderStatus.PENDING
        elif raw_status in ("PARTIAL", "PARTIALLY_FILLED"):
            status = OrderStatus.PARTIALLY_FILLED
        elif raw_status in ("REJECTED", "FAILED"):
            status = OrderStatus.REJECTED
        else:
            logger.warning("Unrecognized order status '%s' in API response. Marking PENDING.", raw_status)
            status = OrderStatus.PENDING

        exec_price = order_obj.get("price") or order_obj.get("execution_price")
        if exec_price is not None:
            try:
                exec_price = float(exec_price)
            except (ValueError, TypeError):
                exec_price = None

        charges = float(order_obj.get("charges") or order_obj.get("commission") or 0.0)
        filled_qty = int(order_obj.get("filled_quantity", intent.quantity if status == OrderStatus.FILLED else 0))

        result = OrderResult(
            client_intent_id=intent.intent_id,
            ticker=intent.ticker,
            action=intent.action,
            quantity=intent.quantity,
            status=status,
            order_id=order_id if order_id else None,
            execution_price=exec_price,
            filled_quantity=filled_qty,
            charges=charges,
            raw_response=resp,
        )

        event_name = "ORDER_FILLED" if status == OrderStatus.FILLED else "ORDER_ACCEPTED"
        self._emit_event(event_name, result)
        return result

    @staticmethod
    def _parse_timestamp(raw: Any) -> Optional[float]:
        """Parse server timestamp into epoch seconds float."""
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                clean_str = raw.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_str)
                return dt.timestamp()
            except Exception:
                pass
        return None

    def _reconcile_timeout_uncertainty(
        self,
        intent: OrderIntent,
        error_msg: str,
        submission_start_time: float,
        pre_existing_order_ids: set[str],
    ) -> OrderResult:
        """Attempt safe reconciliation after a submission timeout.

        Inspects order history for an unambiguous matching order.
        CRITICAL SAFETY RULES:
        1. Ambiguous outcomes remain UNKNOWN with requires_reconciliation=True.
        2. No blind automatic retry occurs under any circumstances.
        3. Orders with IDs known prior to submission are strictly excluded.
        4. Orders with timestamps earlier than the submission window are excluded.
        5. If multiple matching candidates exist, do NOT guess; remain UNKNOWN.
        6. Does NOT claim guaranteed idempotency because server lacks idempotency keys.
        """
        candidates: list[dict[str, Any]] = []
        try:
            history = self.client.get_orders(limit=10, ticker=intent.ticker, action=intent.action.value)
            orders = history.get("orders", []) if isinstance(history, dict) else []

            for o in orders:
                if not isinstance(o, dict):
                    continue

                o_id = str(o.get("order_id") or o.get("id") or "")
                # Exclude orders already known prior to this submission
                if o_id and o_id in pre_existing_order_ids:
                    continue

                o_ticker = str(o.get("ticker", "")).upper()
                o_action = str(o.get("action", "")).upper()
                try:
                    o_qty = int(o.get("quantity", 0))
                except (ValueError, TypeError):
                    o_qty = 0

                if o_ticker != intent.ticker.upper() or o_action != intent.action.value or o_qty != intent.quantity:
                    continue

                # Verify timestamp if present in server response
                raw_ts = o.get("timestamp") or o.get("created_at") or o.get("time") or o.get("date")
                if raw_ts is not None:
                    parsed_ts = self._parse_timestamp(raw_ts)
                    # If timestamp is distinctly before submission window (e.g. > 5s prior), exclude
                    if parsed_ts is not None and parsed_ts < (submission_start_time - 5.0):
                        continue

                candidates.append(o)

        except Exception as exc:
            logger.warning("Failed to query order history during timeout reconciliation: %s", exc)

        # Rule: Only reconcile if exactly one unambiguous candidate is found
        if len(candidates) == 1:
            matched = candidates[0]
            matched_id = str(matched.get("order_id") or matched.get("id") or "")
            logger.info(
                "Unambiguously reconciled timed-out intent %s with order %s in history.",
                intent.intent_id,
                matched_id,
            )
            with self._intent_lock:
                if matched_id:
                    self._known_order_ids.add(matched_id)
            return self._parse_order_submission_response(intent, matched)

        if len(candidates) > 1:
            logger.warning(
                "Timeout reconciliation found %d ambiguous matching orders for intent %s. "
                "Cannot determine identity without server idempotency support. Marking UNKNOWN.",
                len(candidates),
                intent.intent_id,
            )
        else:
            logger.warning(
                "Timeout reconciliation found no matching order in history for intent %s. "
                "Order outcome is UNKNOWN. Withholding automatic retry.",
                intent.intent_id,
            )

        result = OrderResult(
            client_intent_id=intent.intent_id,
            ticker=intent.ticker,
            action=intent.action,
            quantity=intent.quantity,
            status=OrderStatus.UNKNOWN,
            rejection_reason=(
                f"TIMEOUT_UNCERTAIN: {error_msg} "
                f"({len(candidates)} matching candidate orders found). Withholding retry."
            ),
            requires_reconciliation=True,
        )
        self._emit_event("ORDER_UNCERTAIN", result)
        return result

    def get_health_status(self) -> dict[str, Any]:
        """Return health diagnostics for ExecutionAdapter."""
        with self._intent_lock:
            active_count = len(self._active_intents)
            completed_count = len(self._completed_intents)
            known_orders_count = len(self._known_order_ids)

        ledger_health = self.ledger.get_health_status() if self.ledger and hasattr(self.ledger, "get_health_status") else None

        return {
            "active_intents": active_count,
            "completed_intents": completed_count,
            "known_orders": known_orders_count,
            "has_ledger": self.ledger is not None,
            "ledger_diagnostics": ledger_health,
        }

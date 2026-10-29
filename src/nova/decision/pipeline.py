"""NOVA Integrated Trading Pipeline (Step H Interface Bridge).

Unifies:
Kafka Market Data -> EventDispatcher -> Person 2 Alpha Strategy -> Person 3 Decision/Risk Gateway -> ExecutionAdapter -> Trading API -> AccountLedger -> Reconciliation.

Strict Safety Invariants:
1. Every candidate order MUST pass through DeterministicRiskGate before reaching ExecutionAdapter.
2. If Risk Gate vetoes an action (drawdown, stale data, short forbidden, cash buffer), ZERO orders are sent.
3. ExecutionAdapter additionally enforces simulation stage gating, universe membership, and idempotency.
4. All execution outcomes (fills, partials, rejections, unknowns) feed the AccountLedger.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import TYPE_CHECKING, Any, Optional

from execution.adapter import ExecutionAdapter
from execution.accounting import AccountLedger
from execution.models import OrderAction, OrderIntent, OrderResult
from nova.decision.contracts import ActionType, TradeDecision
from nova.quant.contracts import (
    CandidateSignal,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
)
if TYPE_CHECKING:
    from nova.risk.gates import DeterministicRiskGate, RiskConfiguration, RiskGateResult

logger = logging.getLogger(__name__)


class IntegratedTradingPipeline:
    """End-to-end integration pipeline coordinating strategy, risk gates, and execution."""

    def __init__(
        self,
        adapter: ExecutionAdapter,
        risk_gate: Optional[DeterministicRiskGate] = None,
        ledger: Optional[AccountLedger] = None,
        default_target_exposure: float = 0.15,
        decision_engine: Optional[Any] = None,
    ) -> None:
        self.adapter = adapter
        if risk_gate is None:
            from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
            risk_gate = DeterministicRiskGate(RiskConfiguration())
        self.risk_gate = risk_gate
        self.ledger = ledger or getattr(adapter, "ledger", None)
        self.default_target_exposure = default_target_exposure
        self.decision_engine = decision_engine
        self._symbol_history: dict[str, list[MarketBar]] = {}
        self._max_history: int = 500

        # Performance & audit counters
        self.signals_processed: int = 0
        self.orders_authorized: int = 0
        self.orders_vetoed: int = 0
        self.orders_executed: int = 0

    def process_candidate_signal(
        self,
        candidate: CandidateSignal,
        bar: MarketBar,
        as_of_time: Optional[datetime] = None,
    ) -> Optional[OrderResult]:
        """Process candidate signal from Person 2 strategy through Person 3 Risk Gate into ExecutionAdapter."""
        self.signals_processed += 1

        # 1. Map Candidate Direction to Proposed Action
        if candidate.direction == SignalDirection.BULLISH:
            proposed_action = ActionType.BUY
            proposed_exposure = self.default_target_exposure
        elif candidate.direction == SignalDirection.BEARISH:
            proposed_action = ActionType.SELL
            proposed_exposure = 0.0
        else:
            logger.debug("[PIPELINE:NEUTRAL] Candidate signal for %s is NEUTRAL. Abstaining.", bar.ticker)
            return None

        # 2. Extract Portfolio State from AccountLedger
        avail_cash = 1_000_000.0
        portfolio_val = 1_000_000.0
        current_dd = 0.0
        current_pos_shares = 0

        if self.ledger is not None:
            avail_cash = self.ledger.get_cash()
            current_pos_shares = self.ledger.get_position(bar.ticker)
            portfolio_val = (
                self.ledger.get_portfolio_value({bar.ticker: bar.close})
                if hasattr(self.ledger, "get_portfolio_value")
                else self.ledger.get_cash()
            )
            current_dd = (
                self.ledger.get_drawdown({bar.ticker: bar.close})
                if hasattr(self.ledger, "get_drawdown")
                else 0.0
            )

        # For SELL, ensure we only sell if we actually hold shares
        if proposed_action == ActionType.SELL and current_pos_shares <= 0:
            logger.debug("[PIPELINE:SELL_FLAT] Sell signal for %s, but position is 0. Abstaining.", bar.ticker)
            return None

        # For BUY, if current position is already at or above target exposure or max limit, HOLD
        if proposed_action == ActionType.BUY and current_pos_shares > 0:
            current_pos_val = current_pos_shares * bar.close
            current_exposure = current_pos_val / max(portfolio_val, 1e-6)
            if current_exposure >= self.default_target_exposure or current_exposure >= self.risk_gate.config.max_position_exposure:
                logger.debug(
                    "[PIPELINE:HOLD] Position for %s already meets target exposure (%.2f >= %.2f). Holding.",
                    bar.ticker,
                    current_exposure,
                    self.default_target_exposure,
                )
                return None

        # 3. Authoritative Risk Gate Evaluation (Person 3 Domain)
        eval_as_of = as_of_time
        if eval_as_of is None:
            if bar.source_timestamp.tzinfo is not None:
                eval_as_of = datetime.now(bar.source_timestamp.tzinfo)
            else:
                eval_as_of = datetime.now(timezone.utc)

        risk_result: RiskGateResult = self.risk_gate.evaluate(
            proposed_action=proposed_action,
            proposed_exposure=proposed_exposure,
            data_quality=bar.data_quality,
            bar_timestamp=bar.source_timestamp,
            as_of_time=eval_as_of,
            current_drawdown=current_dd,
            available_cash=avail_cash,
            current_portfolio_value=portfolio_val,
            current_position_quantity=float(current_pos_shares),
            current_market_price=float(bar.close),
        )

        if not risk_result.is_approved or risk_result.authorized_action in (ActionType.NO_TRADE, ActionType.HOLD):
            self.orders_vetoed += 1
            logger.warning(
                "[PIPELINE:RISK_VETO] Signal for %s %s VETOED by Risk Gate: %s",
                proposed_action.value,
                bar.ticker,
                risk_result.risk_reason_codes,
            )
            return None

        self.orders_authorized += 1

        # 4. Sizing Calculation (Target Exposure to Incremental Shares)
        if proposed_action == ActionType.BUY:
            notional = risk_result.authorized_exposure * portfolio_val
            target_shares = int(notional / max(bar.close, 1e-4))
            quantity = max(0, target_shares - current_pos_shares)
        else:
            quantity = current_pos_shares

        if quantity <= 0:
            logger.warning(
                "[PIPELINE:ZERO_QTY] Sizing computed %d shares for %s (%s). Skipping order.",
                quantity,
                bar.ticker,
                proposed_action.value,
            )
            return None

        # 5. Construct Authorized OrderIntent and Route through ExecutionAdapter
        import uuid
        intent = OrderIntent(
            intent_id=f"intent_{uuid.uuid4().hex[:12]}",
            ticker=bar.ticker.upper(),
            action=OrderAction.BUY if risk_result.authorized_action == ActionType.BUY else OrderAction.SELL,
            quantity=quantity,
            authorized_by_risk=True,
            date=str(bar.source_timestamp.date()),
            strategy_id=candidate.strategy_id,
            risk_authorization_id=f"risk_auth_{candidate.ticker}_{int(bar.source_timestamp.timestamp())}",
        )

        result = self.adapter.execute_order(intent)
        if result and result.status.value in ("FILLED", "PARTIALLY_FILLED", "PENDING", "SUBMITTED"):
            self.orders_executed += 1
        return result

    def handle_market_tick(
        self,
        event: Any,
        as_of_time: Optional[datetime] = None,
    ) -> Optional[OrderResult]:
        """Process incoming MarketTickEvent through the full Sentinel decision and execution pipeline.

        Converts raw tick to canonical MarketBar, maintains chronological monotonicity,
        evaluates SentinelDecisionEngine (incorporating multi-strategy signals, causal ML prediction,
        regime detection, friction economics, profit realization, and deterministic risk gating),
        and routes the authorized decision through the ExecutionAdapter.
        """
        from nova.quant.contracts import MarketBar, TimestampSemantics, DataQualityStatus
        from nova.research.adapters import market_tick_to_market_bar

        bar: Optional[MarketBar] = None
        if isinstance(event, MarketBar):
            bar = event
        elif isinstance(event, dict):
            try:
                data = event.get("data") if isinstance(event.get("data"), dict) else event
                ticker = str(data.get("ticker") or data.get("symbol") or "").upper()
                op = float(data.get("open", data.get("close", 0.0)))
                hi = float(data.get("high", max(op, float(data.get("close", 0.0)))))
                lo = float(data.get("low", min(op, float(data.get("close", 0.0)))))
                cl = float(data.get("close", 0.0))
                vol = float(data.get("volume", 1000.0))

                ts_val = data.get("timestamp") or event.get("timestamp") or event.get("simulated_date")
                if isinstance(ts_val, str):
                    if len(ts_val) == 10:
                        ts = datetime.strptime(ts_val, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    else:
                        ts = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                elif isinstance(ts_val, datetime):
                    ts = ts_val if ts_val.tzinfo else ts_val.replace(tzinfo=timezone.utc)
                else:
                    ts = datetime.now(timezone.utc)

                bar = MarketBar(
                    ticker=ticker,
                    open=op,
                    high=hi,
                    low=lo,
                    close=cl,
                    volume=vol,
                    source_timestamp=ts,
                    available_timestamp=ts,
                    timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                    data_quality=DataQualityStatus.VALID,
                    provenance="dict_event",
                )
            except Exception as exc:
                logger.warning("[PIPELINE:ADAPT_ERROR] Failed to convert tick dict to MarketBar: %s", exc)
                return None
        else:
            try:
                bar = market_tick_to_market_bar(event)
            except Exception as exc:
                logger.warning("[PIPELINE:ADAPT_ERROR] Failed to convert tick event to MarketBar: %s", exc)
                return None

        ticker = bar.ticker.upper()
        if ticker not in self._symbol_history:
            self._symbol_history[ticker] = []

        history = self._symbol_history[ticker]

        # Enforce chronological monotonicity: drop duplicates / out-of-order ticks
        if history and bar.source_timestamp <= history[-1].source_timestamp:
            logger.warning(
                "[PIPELINE:DROP_TICK] Dropping out-of-order or duplicate tick for %s (%s <= %s)",
                ticker,
                bar.source_timestamp.isoformat(),
                history[-1].source_timestamp.isoformat(),
            )
            return None

        history.append(bar)
        if len(history) > self._max_history:
            self._symbol_history[ticker] = history[-self._max_history :]
            history = self._symbol_history[ticker]

        # If decision engine is wired, evaluate authoritative SentinelDecisionEngine
        if self.decision_engine is not None:
            portfolio_state = None
            pos_state = None
            avail_cash = 1_000_000.0
            port_val = 1_000_000.0
            dd = 0.0
            pos_exposure = 0.0

            if self.ledger is not None:
                avail_cash = self.ledger.get_cash()
                port_val = (
                    self.ledger.get_portfolio_value({ticker: bar.close})
                    if hasattr(self.ledger, "get_portfolio_value")
                    else avail_cash
                )
                dd = (
                    self.ledger.get_drawdown({ticker: bar.close})
                    if hasattr(self.ledger, "get_drawdown")
                    else 0.0
                )
                cur_qty = self.ledger.get_position(ticker)
                pos_exposure = (cur_qty * bar.close) / max(port_val, 1e-4)

                if hasattr(self.ledger, "get_portfolio_state"):
                    portfolio_state = self.ledger.get_portfolio_state({ticker: bar.close})
                    if portfolio_state and portfolio_state.positions:
                        pos_state = portfolio_state.positions.get(ticker)

            eval_time = as_of_time or bar.source_timestamp

            try:
                decision = self.decision_engine.evaluate_decision(
                    bars=history,
                    current_position_exposure=pos_exposure,
                    current_drawdown=dd,
                    available_cash=avail_cash,
                    current_portfolio_value=port_val,
                    as_of_time=eval_time,
                    position_state=pos_state,
                    portfolio_state=portfolio_state,
                )
                return self.process_trade_decision(decision, bar)
            except Exception as exc:
                logger.error("[PIPELINE:DECISION_ERROR] Exception in SentinelDecisionEngine for %s: %s", ticker, exc, exc_info=True)
                return None

        return None

    def __call__(self, event: Any) -> Optional[OrderResult]:
        """Callable interface allowing direct subscription to EventDispatcher."""
        return self.handle_market_tick(event)

    def process_trade_decision(
        self,
        decision: TradeDecision,
        bar: MarketBar,
    ) -> Optional[OrderResult]:
        """Route an authoritative Person 3 TradeDecision directly through ExecutionAdapter."""
        self.signals_processed += 1

        portfolio_val = 1_000_000.0
        current_pos_shares = 0
        if self.ledger is not None:
            current_pos_shares = self.ledger.get_position(bar.ticker)
            portfolio_val = (
                self.ledger.get_portfolio_value({bar.ticker: bar.close})
                if hasattr(self.ledger, "get_portfolio_value")
                else self.ledger.get_cash()
            )

        result = self.adapter.execute_trade_decision(
            decision=decision,
            current_price=bar.close,
            current_portfolio_value=portfolio_val,
            current_position_shares=current_pos_shares,
            date=str(bar.source_timestamp.date()),
        )
        if result and result.status.value in ("FILLED", "PARTIALLY_FILLED", "PENDING", "SUBMITTED"):
            self.orders_executed += 1
            self.orders_authorized += 1
        elif result is None:
            self.orders_vetoed += 1
        return result

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
from typing import Any, Optional

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
    ) -> None:
        self.adapter = adapter
        self.risk_gate = risk_gate or DeterministicRiskGate(RiskConfiguration())
        self.ledger = ledger or getattr(adapter, "ledger", None)
        self.default_target_exposure = default_target_exposure

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
        )

        if not risk_result.is_approved:
            self.orders_vetoed += 1
            logger.warning(
                "[PIPELINE:RISK_VETO] Signal for %s %s VETOED by Risk Gate: %s",
                proposed_action.value,
                bar.ticker,
                risk_result.risk_reason_codes,
            )
            return None

        self.orders_authorized += 1

        # 4. Sizing Calculation
        if proposed_action == ActionType.BUY:
            notional = risk_result.authorized_exposure * portfolio_val
            quantity = int(notional / max(bar.close, 1e-4))
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
        return result

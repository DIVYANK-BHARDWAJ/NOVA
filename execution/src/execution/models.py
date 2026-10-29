"""Data models and contract definitions for NOVA Trading Execution.

Defines schemas for authorized order intents, order submission payloads, execution
results, order lifecycle statuses, simulation metadata, and portfolio positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class OrderAction(str, Enum):
    """Documented order actions accepted by the competition platform."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderStatus(str, Enum):
    """Execution lifecycle status classifications."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"  # Assigned on timeout or ambiguous network error requiring reconciliation


@dataclass(frozen=True)
class OrderIntent:
    """Trading decision authorized by Person 3's risk gateway.

    The execution adapter receives OrderIntent instances, NOT raw candidate signals.

    CRITICAL ARCHITECTURAL BOUNDARY & LIMITATION:
    The boolean attribute `authorized_by_risk: bool = False` is an in-memory flag.
    In a shared Python runtime, a boolean does NOT provide cryptographic proof of
    provenance or tamper-resistance; any caller can instantiate `OrderIntent(..., authorized_by_risk=True)`.

    Concrete Integration Recommendations for Person 3:
    1. Cryptographic HMAC Token: Person 3's Risk Gateway should sign authorized intents
       with a private session key (`risk_authorization_id = hmac.new(secret, ...).hexdigest()`),
       verified by the adapter before order submission.
    2. Private Factory Pattern: Prevent strategies from directly instantiating `OrderIntent`.
       Only `RiskGateway.authorize()` should act as the factory producing `OrderIntent` objects.
    3. Monotonic Timestamp Verification: Check `created_at` against the current clock to reject
       stale risk authorizations (e.g. older than 5 seconds) to prevent replay attacks.
    """
    intent_id: str
    ticker: str
    action: OrderAction
    quantity: int
    authorized_by_risk: bool = False
    date: Optional[str] = None
    strategy_id: Optional[str] = None
    risk_authorization_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate basic field types and constraints."""
        if not self.intent_id or not isinstance(self.intent_id, str):
            raise ValueError("intent_id must be a non-empty string.")

        if not self.ticker or not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("ticker must be a non-empty string.")

        # Ensure action is an OrderAction enum
        if isinstance(self.action, str):
            try:
                object.__setattr__(self, "action", OrderAction(self.action.upper()))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid action '{self.action}'. Must be one of {[a.value for a in OrderAction]}."
                ) from exc
        elif not isinstance(self.action, OrderAction):
            raise ValueError(
                f"action must be an OrderAction enum, got {type(self.action).__name__}."
            )

        # Quantity must be a strictly positive integer (reject bool, float, negative, zero)
        if type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError(
                f"quantity must be a positive integer > 0, got {repr(self.quantity)} "
                f"(type={type(self.quantity).__name__})."
            )


@dataclass(frozen=True)
class OrderResult:
    """Outcome of an order execution attempt."""
    client_intent_id: str
    ticker: str
    action: OrderAction
    quantity: int
    status: OrderStatus
    order_id: Optional[str] = None
    execution_price: Optional[float] = None
    filled_quantity: int = 0
    charges: float = 0.0
    rejection_reason: Optional[str] = None
    requires_reconciliation: bool = False
    raw_response: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_filled(self) -> bool:
        """Return True if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_rejected(self) -> bool:
        """Return True if order was rejected by risk gateway or exchange."""
        return self.status == OrderStatus.REJECTED

    @property
    def is_uncertain(self) -> bool:
        """Return True if execution outcome is uncertain and requires reconciliation."""
        return self.status == OrderStatus.UNKNOWN or self.requires_reconciliation

    def to_dict(self) -> dict[str, Any]:
        """Convert OrderResult to dictionary representation."""
        return {
            "client_intent_id": self.client_intent_id,
            "ticker": self.ticker,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "quantity": self.quantity,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "order_id": self.order_id,
            "execution_price": self.execution_price,
            "filled_quantity": self.filled_quantity,
            "charges": self.charges,
            "rejection_reason": self.rejection_reason,
            "requires_reconciliation": self.requires_reconciliation,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ExecutionEvent:
    """Event emitted when order execution state changes."""
    event_type: str
    order_result: OrderResult
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ExecutionEvent to schema-compliant dictionary."""
        return {
            "event_type": self.event_type,
            "order_result": self.order_result.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "extra": self.extra,
        }


@dataclass(frozen=True)
class StockInfo:
    """Active simulation tradable stock metadata."""
    symbol: str
    ticker: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class SimulationStage:
    """Platform simulation stage metadata."""
    stage: str
    status: str
    tickers: list[str]
    ticker_count: int = 0
    current_date: Optional[str] = None
    days_completed: int = 0
    total_days: int = 0
    interval_seconds: int = 10
    is_sandbox_stream: bool = False
    sandbox_loop: int = 1
    raw_response: Optional[dict[str, Any]] = None

    @property
    def is_trading_active(self) -> bool:
        """Check if stage and status permit active trading operations."""
        # Status 'running' indicates active clock and simulation loop
        return self.status.lower() == "running"


@dataclass(frozen=True)
class SimulationClock:
    """Simulation clock information."""
    status: str
    simulated_date: str
    current_day: int
    days_completed: int
    total_days: int
    remaining_days: int
    interval_seconds: int
    raw_response: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class CashBalance:
    """Cash and capital status from GET /api/v1/portfolio/cash."""
    team_id: str
    team_name: str
    available_cash: float
    cash_balance: float
    capital_remaining: float
    margin_locked: float
    initial_capital: float
    realized_pnl: float
    currency: str = "INR"
    is_blocked: bool = False
    simulated_date: Optional[str] = None
    raw_response: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class PortfolioHolding:
    """Individual stock position holding."""
    ticker: str
    quantity: int
    avg_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class Portfolio:
    """Overall account equity and position data from GET /api/v1/portfolio."""
    team_id: str
    team_name: str
    available_cash: float
    cash_balance: float
    portfolio_market_value: float
    total_wealth: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_charges: float
    total_taxes: float
    holdings: list[PortfolioHolding] = field(default_factory=list)
    total_orders: int = 0
    rejected_orders_count: int = 0
    error_count: int = 0
    is_blocked: bool = False
    is_verified: bool = True
    simulated_date: Optional[str] = None
    raw_response: Optional[dict[str, Any]] = None

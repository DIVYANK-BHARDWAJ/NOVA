"""NOVA Decision Intelligence — Interface Contract C (TradeDecision).

STATUS: AUTHORITATIVE — PERSON 3 EXCLUSIVE OWNERSHIP.
Implements the TradeDecision schema defined in Sentinel Trader Master Project Specification (Page 10, Section 20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Mapping, Sequence

from nova.quant.contracts import (
    ContractValidationError,
    NumericalValidationError,
    TemporalValidationError,
    validate_finite,
    validate_timezone_aware,
)


class ActionType(str, Enum):
    """Supported actions evaluated by the decision engine."""
    BUY = "BUY"          # Open new long or add to long position
    SELL = "SELL"        # Close or reduce owned long shares (incurs ₹15.34 DP + ₹20 brokerage)
    SHORT = "SHORT"      # Open new short or add to short position (locks margin, ₹20 brokerage)
    COVER = "COVER"      # Close or reduce short position (releases margin, ₹20 brokerage)
    HOLD = "HOLD"        # Maintain current position
    REDUCE = "REDUCE"    # Generic position scale-down
    EXIT = "EXIT"        # Generic position closure
    NO_TRADE = "NO_TRADE" # Abstain from trading


@dataclass(frozen=True, slots=True)
class PositionState:
    """Point-in-time state of an individual instrument holding.

    Consumed from Person 1's portfolio ledger or simulation tracker.
    """
    ticker: str
    quantity: float                        # > 0 for Long, < 0 for Short, == 0 for Flat
    entry_price: float
    current_market_price: float
    holding_days: int = 0                  # Simulated trading days position has been open
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    locked_margin: float = 0.0             # Margin locked for short positions

    @property
    def is_long(self) -> bool:
        return self.quantity > 1e-6

    @property
    def is_short(self) -> bool:
        return self.quantity < -1e-6

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) <= 1e-6

    @property
    def market_value(self) -> float:
        """Current market value of position (positive for long, negative for short)."""
        return self.quantity * self.current_market_price


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Point-in-time account and portfolio state consumed by the decision engine.

    Represents MANTHAN 2026 account state:
    Total Wealth = Available Cash + Margin Locked + Market Value of All Stocks Held
    """
    available_cash: float                  # Liquid capital available (starts at 10,00,000)
    locked_margin: float = 0.0             # Collateral locked for active short positions
    positions: Mapping[str, PositionState] = field(default_factory=dict)
    initial_capital: float = 1000000.0     # Default MANTHAN 2026 ₹10,00,000
    peak_wealth: float = 1000000.0
    realized_pnl_total: float = 0.0
    fees_paid_total: float = 0.0

    @property
    def long_market_value(self) -> float:
        """Sum of market values of all owned long shares."""
        return sum(pos.market_value for pos in self.positions.values() if pos.is_long)

    @property
    def total_wealth(self) -> float:
        """Authoritative Total Wealth definition from MANTHAN guide:
        Total Wealth = Available Cash + Margin Locked + Market Value of All Stocks Held
        """
        return self.available_cash + self.locked_margin + self.long_market_value

    @property
    def current_drawdown(self) -> float:
        """Current drawdown percentage relative to peak wealth."""
        if self.peak_wealth <= 0.0:
            return 0.0
        return max(0.0, (self.peak_wealth - self.total_wealth) / self.peak_wealth)

    @property
    def cash_fraction(self) -> float:
        """Fraction of total wealth held in liquid cash."""
        wealth = self.total_wealth
        return self.available_cash / wealth if wealth > 0.0 else 0.0

    def get_position(self, ticker: str) -> PositionState:
        """Get position state for ticker or return a flat position."""
        return self.positions.get(
            ticker,
            PositionState(
                ticker=ticker,
                quantity=0.0,
                entry_price=0.0,
                current_market_price=0.0,
            ),
        )


class OrderType(str, Enum):
    """Order type for execution."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class TradeDecision:
    """Interface Contract C: Authoritative output of the Person 3 Decision Engine.

    This object is consumed by Person 1's Execution and Accounting subsystems.
    It contains complete cost-awareness, risk metrics, and audit trail reason codes.
    """
    decision_id: str
    decision_timestamp: datetime
    instrument_id: str
    selected_action: ActionType
    target_quantity_or_exposure: float
    order_type: OrderType
    selected_strategy: str
    selection_or_rejection_reasons: tuple[str, ...]
    configuration_version: str
    expected_gross_return: float | None = None
    estimated_all_in_cost: float | None = None
    expected_net_return: float | None = None
    probability_of_clearing_costs: float | None = None
    uncertainty_estimate: float | None = None
    estimated_risk_contribution: float | None = None
    model_versions: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ContractValidationError("TradeDecision 'decision_id' cannot be blank")

        validate_timezone_aware(self.decision_timestamp, "decision_timestamp")

        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise ContractValidationError("TradeDecision 'instrument_id' cannot be blank")

        if not isinstance(self.selected_action, ActionType):
            raise ContractValidationError(f"Invalid selected_action: {self.selected_action}")

        validate_finite(self.target_quantity_or_exposure, "target_quantity_or_exposure")

        if not isinstance(self.order_type, OrderType):
            raise ContractValidationError(f"Invalid order_type: {self.order_type}")

        if not isinstance(self.selected_strategy, str) or not self.selected_strategy.strip():
            raise ContractValidationError("TradeDecision 'selected_strategy' cannot be blank")

        if not isinstance(self.selection_or_rejection_reasons, tuple):
            raise ContractValidationError("selection_or_rejection_reasons must be a tuple")
        if not self.selection_or_rejection_reasons:
            raise ContractValidationError("TradeDecision must provide at least one reason code")

        for reason in self.selection_or_rejection_reasons:
            if not isinstance(reason, str) or not reason.strip():
                raise ContractValidationError("Reason codes must be non-empty strings")

        if not isinstance(self.configuration_version, str) or not self.configuration_version.strip():
            raise ContractValidationError("configuration_version cannot be blank")

        # Validate optional finite floats
        for name, val in [
            ("expected_gross_return", self.expected_gross_return),
            ("estimated_all_in_cost", self.estimated_all_in_cost),
            ("expected_net_return", self.expected_net_return),
            ("probability_of_clearing_costs", self.probability_of_clearing_costs),
            ("uncertainty_estimate", self.uncertainty_estimate),
            ("estimated_risk_contribution", self.estimated_risk_contribution),
        ]:
            if val is not None:
                validate_finite(val, name)

        if self.estimated_all_in_cost is not None and self.estimated_all_in_cost < 0.0:
            raise NumericalValidationError(f"estimated_all_in_cost cannot be negative: {self.estimated_all_in_cost}")

        if self.probability_of_clearing_costs is not None and not (0.0 <= self.probability_of_clearing_costs <= 1.0):
            raise NumericalValidationError(
                f"probability_of_clearing_costs must be in [0, 1], got {self.probability_of_clearing_costs}"
            )

        if self.uncertainty_estimate is not None and self.uncertainty_estimate < 0.0:
            raise NumericalValidationError(f"uncertainty_estimate cannot be negative: {self.uncertainty_estimate}")

        if self.expected_net_return is not None:
            if self.expected_gross_return is not None and self.estimated_all_in_cost is not None:
                calc_net = self.expected_gross_return - self.estimated_all_in_cost
                if not math.isclose(self.expected_net_return, calc_net, rel_tol=1e-5, abs_tol=1e-6):
                    raise ContractValidationError(
                        f"expected_net_return ({self.expected_net_return}) does not match gross ({self.expected_gross_return}) - cost ({self.estimated_all_in_cost})"
                    )

    def to_dict(self) -> dict:
        """Serialize to contract dictionary compatible with JSON schema."""
        return {
            "decision_id": self.decision_id,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "instrument_id": self.instrument_id,
            "selected_action": self.selected_action.value,
            "target_quantity_or_exposure": self.target_quantity_or_exposure,
            "order_type": self.order_type.value,
            "expected_gross_return": self.expected_gross_return,
            "estimated_all_in_cost": self.estimated_all_in_cost,
            "expected_net_return": self.expected_net_return,
            "probability_of_clearing_costs": self.probability_of_clearing_costs,
            "uncertainty_estimate": self.uncertainty_estimate,
            "estimated_risk_contribution": self.estimated_risk_contribution,
            "selected_strategy": self.selected_strategy,
            "selection_or_rejection_reasons": list(self.selection_or_rejection_reasons),
            "model_versions": dict(self.model_versions),
            "configuration_version": self.configuration_version,
        }

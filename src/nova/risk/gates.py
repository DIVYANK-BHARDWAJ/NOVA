"""NOVA Pre-Trade Risk Gate (Person 3 Exclusive Domain).

Deterministic risk enforcement independent of predictive models.
No ML model can override these constraints. Limits are strictly configuration-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from nova.decision.contracts import ActionType
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    NumericalValidationError,
    TemporalValidationError,
    validate_finite,
    validate_timezone_aware,
)


@dataclass(frozen=True, slots=True)
class RiskConfiguration:
    """Configurable risk limits for deterministic risk control."""
    max_position_exposure: float = 0.25      # Max exposure fraction per instrument (e.g., 25% for 4-8 stocks)
    max_gross_exposure: float = 1.00         # Max total portfolio gross exposure
    max_drawdown_limit: float = 0.15         # Kill-switch threshold (e.g., 15% drawdown)
    min_cash_buffer_fraction: float = 0.05   # Absolute minimum cash buffer (default 5%)
    min_cash_reserve_fraction: float = 0.20  # MANTHAN 20-30% cash cushion reserve (default 20%)
    stop_loss_pct: float = 0.04              # MANTHAN stop-loss rule (3%-4%, default 4%)
    max_short_holding_days: int = 10         # MANTHAN 10-day bi-weekly short closure rule
    target_diversification_stocks: int = 6   # Target 4-8 stocks diversification
    initial_capital: float = 1000000.0       # MANTHAN ₹10,00,000 initial capital
    allow_short: bool = False                # False by default for safety/back-compat, True for MANTHAN
    max_staleness_seconds: float = 300.0     # Max age of market bar before rejection (5 mins)
    config_version: str = "1.2.0"

    def __post_init__(self) -> None:
        for name, val in [
            ("max_position_exposure", self.max_position_exposure),
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_drawdown_limit", self.max_drawdown_limit),
            ("min_cash_buffer_fraction", self.min_cash_buffer_fraction),
            ("min_cash_reserve_fraction", self.min_cash_reserve_fraction),
            ("stop_loss_pct", self.stop_loss_pct),
            ("max_staleness_seconds", self.max_staleness_seconds),
            ("initial_capital", self.initial_capital),
        ]:
            validate_finite(val, name)
            if val <= 0.0 and name != "max_staleness_seconds":
                raise NumericalValidationError(f"Risk configuration '{name}' must be positive, got {val}")

        if not (0.0 < self.max_position_exposure <= 1.0):
            raise NumericalValidationError(f"max_position_exposure must be in (0, 1], got {self.max_position_exposure}")

        if self.max_short_holding_days <= 0:
            raise NumericalValidationError(f"max_short_holding_days must be positive, got {self.max_short_holding_days}")


@dataclass(frozen=True, slots=True)
class RiskGateResult:
    """Result of deterministic risk evaluation."""
    is_approved: bool
    authorized_action: ActionType
    authorized_exposure: float
    risk_reason_codes: tuple[str, ...]
    circuit_breaker_active: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.is_approved, bool):
            raise ContractValidationError("is_approved must be a boolean")
        if not isinstance(self.authorized_action, ActionType):
            raise ContractValidationError(f"Invalid authorized_action: {self.authorized_action}")
        validate_finite(self.authorized_exposure, "authorized_exposure")


class DeterministicRiskGate:
    """Evaluates candidate trade actions against immutable portfolio risk constraints."""

    def __init__(self, config: RiskConfiguration | None = None) -> None:
        self.config = config or RiskConfiguration()

    def evaluate(
        self,
        proposed_action: ActionType,
        proposed_exposure: float,
        data_quality: DataQualityStatus = DataQualityStatus.VALID,
        bar_timestamp: datetime | None = None,
        as_of_time: datetime | None = None,
        current_drawdown: float = 0.0,
        available_cash: float = 1.0,
        current_portfolio_value: float = 1.0,
        current_position_quantity: float = 0.0,
        position_entry_price: float = 0.0,
        current_market_price: float = 0.0,
        short_holding_days: int = 0,
        locked_margin: float = 0.0,
        has_pending_exit_order: bool = False,
        has_pending_entry_order: bool = False,
        pending_exit_quantity: float = 0.0,
        current_position_exposure: float | None = None,
    ) -> RiskGateResult:
        """Deterministic veto / clamp logic.

        Returns approved action and exposure with explicit audit reasons.
        """
        if bar_timestamp is None:
            bar_timestamp = datetime.now(timezone.utc)

        if current_position_exposure is not None and abs(current_position_quantity) <= 1e-6:
            px = current_market_price if current_market_price > 0 else 1.0
            current_position_quantity = (current_position_exposure * current_portfolio_value) / px

        validate_finite(proposed_exposure, "proposed_exposure")
        validate_finite(current_drawdown, "current_drawdown")
        validate_timezone_aware(bar_timestamp, "bar_timestamp")

        reasons: list[str] = []

        # 0. Short Prohibition Check (prioritized for negative exposure or SHORT action)
        is_short_action = (proposed_action == ActionType.SHORT or proposed_exposure < -1e-6)
        if is_short_action and not self.config.allow_short:
            reasons.append("SHORT_POSITIONS_PROHIBITED_BY_POLICY")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 0b. Execution Safety Checks: Pending Orders, Overselling & Nonexistent Covers
        is_exit_action = proposed_action in (ActionType.SELL, ActionType.COVER, ActionType.EXIT, ActionType.REDUCE)
        if has_pending_exit_order and is_exit_action:
            reasons.append("EXECUTION_SAFETY_DUPLICATE_EXIT_PENDING_REJECT")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.HOLD,
                authorized_exposure=proposed_exposure,
                risk_reason_codes=tuple(reasons),
            )

        if has_pending_entry_order and (is_exit_action or proposed_action in (ActionType.BUY, ActionType.SHORT)):
            reasons.append("EXECUTION_SAFETY_CONFLICTING_ORDER_PENDING_REJECT")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.HOLD,
                authorized_exposure=proposed_exposure,
                risk_reason_codes=tuple(reasons),
            )

        if has_pending_exit_order and proposed_action in (ActionType.BUY, ActionType.SHORT):
            reasons.append("EXECUTION_SAFETY_CONFLICTING_ORDER_PENDING_REJECT")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        if proposed_action == ActionType.SELL and proposed_exposure >= 0.0:
            if current_position_quantity <= 1e-6:
                reasons.append("EXECUTION_SAFETY_CANNOT_SELL_WITHOUT_LONG_POSITION_REJECT")
                return RiskGateResult(
                    is_approved=False,
                    authorized_action=ActionType.NO_TRADE,
                    authorized_exposure=0.0,
                    risk_reason_codes=tuple(reasons),
                )
            if pending_exit_quantity > 0:
                available_to_sell = current_position_quantity - pending_exit_quantity
                if available_to_sell <= 1e-6:
                    reasons.append("EXECUTION_SAFETY_ALL_LONG_SHARES_PENDING_EXIT_REJECT")
                    return RiskGateResult(
                        is_approved=False,
                        authorized_action=ActionType.HOLD,
                        authorized_exposure=proposed_exposure,
                        risk_reason_codes=tuple(reasons),
                    )

        if proposed_action == ActionType.COVER:
            if current_position_quantity >= -1e-6:
                reasons.append("EXECUTION_SAFETY_CANNOT_COVER_WITHOUT_SHORT_POSITION_REJECT")
                return RiskGateResult(
                    is_approved=False,
                    authorized_action=ActionType.NO_TRADE,
                    authorized_exposure=0.0,
                    risk_reason_codes=tuple(reasons),
                )
            if pending_exit_quantity > 0:
                available_to_cover = abs(current_position_quantity) - pending_exit_quantity
                if available_to_cover <= 1e-6:
                    reasons.append("EXECUTION_SAFETY_ALL_SHORT_SHARES_PENDING_COVER_REJECT")
                    return RiskGateResult(
                        is_approved=False,
                        authorized_action=ActionType.HOLD,
                        authorized_exposure=proposed_exposure,
                        risk_reason_codes=tuple(reasons),
                    )

        # 1. Circuit Breaker / Drawdown Limit Check
        if current_drawdown >= self.config.max_drawdown_limit:
            reasons.append(f"CIRCUIT_BREAKER_TRIGGERED_DRAWDOWN_{current_drawdown:.4f}_EXCEEDS_{self.config.max_drawdown_limit:.4f}")
            # If in drawdown, veto all new BUY / SHORT positions; allow EXIT, SELL, COVER, or NO_TRADE
            if proposed_action in (ActionType.BUY, ActionType.SHORT, ActionType.SELL):
                # Only allow reducing/closing positions
                if proposed_action == ActionType.SELL and current_position_quantity > 0:
                    pass
                else:
                    return RiskGateResult(
                        is_approved=False,
                        authorized_action=ActionType.NO_TRADE,
                        authorized_exposure=0.0,
                        risk_reason_codes=tuple(reasons),
                        circuit_breaker_active=True,
                    )

        # 2. Data Quality Check
        if data_quality != DataQualityStatus.VALID:
            reasons.append(f"DATA_QUALITY_UNACCEPTABLE_{data_quality.value}")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 3. Staleness Check
        if as_of_time is not None:
            validate_timezone_aware(as_of_time, "as_of_time")
            age_seconds = (as_of_time - bar_timestamp).total_seconds()
            if age_seconds > self.config.max_staleness_seconds:
                reasons.append(f"MARKET_DATA_STALE_{age_seconds:.1f}S_EXCEEDS_{self.config.max_staleness_seconds}S")
                return RiskGateResult(
                    is_approved=False,
                    authorized_action=ActionType.NO_TRADE,
                    authorized_exposure=0.0,
                    risk_reason_codes=tuple(reasons),
                )

        # 4. MANTHAN Rule: Short 10-day auto-settlement enforcement
        if current_position_quantity < -1e-6 and short_holding_days >= self.config.max_short_holding_days:
            reasons.append(f"SHORT_10_DAY_SETTLEMENT_AUTO_CLOSE_DAYS_{short_holding_days}")
            return RiskGateResult(
                is_approved=True,
                authorized_action=ActionType.COVER,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 5. MANTHAN Rule: Configurable Stop-Loss logic (3%-4% default)
        if position_entry_price > 0.0 and current_market_price > 0.0:
            if current_position_quantity > 1e-6:
                drop_pct = (position_entry_price - current_market_price) / position_entry_price
                if drop_pct >= self.config.stop_loss_pct:
                    reasons.append(f"STOP_LOSS_LONG_TRIGGERED_DROP_{drop_pct:.3%}_EXCEEDS_{self.config.stop_loss_pct:.3%}")
                    return RiskGateResult(
                        is_approved=True,
                        authorized_action=ActionType.SELL,
                        authorized_exposure=0.0,
                        risk_reason_codes=tuple(reasons),
                    )
            elif current_position_quantity < -1e-6:
                rise_pct = (current_market_price - position_entry_price) / position_entry_price
                if rise_pct >= self.config.stop_loss_pct:
                    reasons.append(f"STOP_LOSS_SHORT_TRIGGERED_RISE_{rise_pct:.3%}_EXCEEDS_{self.config.stop_loss_pct:.3%}")
                    return RiskGateResult(
                        is_approved=True,
                        authorized_action=ActionType.COVER,
                        authorized_exposure=0.0,
                        risk_reason_codes=tuple(reasons),
                    )

        # 6. Short Prohibition Check
        is_short_action = (proposed_action == ActionType.SHORT or proposed_exposure < 0.0)
        if is_short_action and not self.config.allow_short:
            reasons.append("SHORT_POSITIONS_PROHIBITED_BY_POLICY")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 7. MANTHAN Rule: Margin constraint check for SHORT
        if proposed_action == ActionType.SHORT:
            margin_required = abs(proposed_exposure) * current_portfolio_value
            if available_cash < margin_required:
                reasons.append(f"INSUFFICIENT_MARGIN_CASH_{available_cash:.2f}_REQUIRED_{margin_required:.2f}")
                return RiskGateResult(
                    is_approved=False,
                    authorized_action=ActionType.NO_TRADE,
                    authorized_exposure=0.0,
                    risk_reason_codes=tuple(reasons),
                )

        # 8. Position Exposure Cap Clamping
        capped_exposure = proposed_exposure
        if abs(proposed_exposure) > self.config.max_position_exposure:
            sign = 1.0 if proposed_exposure > 0 else -1.0
            capped_exposure = sign * self.config.max_position_exposure
            reasons.append(
                f"EXPOSURE_CLAMPED_FROM_{proposed_exposure:.2f}_TO_MAX_{capped_exposure:.2f}"
            )

        # 9. Cash Buffer & Reserve Check
        cash_fraction = available_cash / max(1e-6, current_portfolio_value)
        if proposed_action == ActionType.BUY and cash_fraction < self.config.min_cash_buffer_fraction:
            reasons.append(f"INSUFFICIENT_CASH_BUFFER_{cash_fraction:.4f}_BELOW_{self.config.min_cash_buffer_fraction:.4f}")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        if not reasons:
            reasons.append("RISK_GATES_PASSED_ALL_CONSTRAINTS_SATISFIED")

        return RiskGateResult(
            is_approved=True,
            authorized_action=proposed_action,
            authorized_exposure=capped_exposure,
            risk_reason_codes=tuple(reasons),
            circuit_breaker_active=False,
        )

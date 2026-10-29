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
    max_position_exposure: float = 0.25      # Max exposure fraction per instrument (e.g., 25%)
    max_gross_exposure: float = 1.00         # Max total portfolio gross exposure
    max_drawdown_limit: float = 0.15         # Kill-switch threshold (e.g., 15% drawdown)
    min_cash_buffer_fraction: float = 0.05   # Mandatory cash buffer (e.g., 5% liquid cash)
    allow_short: bool = False                # Whether short exposure is permitted
    max_staleness_seconds: float = 300.0     # Max age of market bar before rejection (5 mins)
    config_version: str = "1.0.0"

    def __post_init__(self) -> None:
        for name, val in [
            ("max_position_exposure", self.max_position_exposure),
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_drawdown_limit", self.max_drawdown_limit),
            ("min_cash_buffer_fraction", self.min_cash_buffer_fraction),
            ("max_staleness_seconds", self.max_staleness_seconds),
        ]:
            validate_finite(val, name)
            if val <= 0.0 and name != "max_staleness_seconds":
                raise NumericalValidationError(f"Risk configuration '{name}' must be positive, got {val}")

        if not (0.0 < self.max_position_exposure <= 1.0):
            raise NumericalValidationError(f"max_position_exposure must be in (0, 1], got {self.max_position_exposure}")


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
        data_quality: DataQualityStatus,
        bar_timestamp: datetime,
        as_of_time: datetime | None = None,
        current_drawdown: float = 0.0,
        available_cash: float = 1.0,
        current_portfolio_value: float = 1.0,
    ) -> RiskGateResult:
        """Deterministic veto / clamp logic.

        Returns approved action and exposure with explicit audit reasons.
        """
        validate_finite(proposed_exposure, "proposed_exposure")
        validate_finite(current_drawdown, "current_drawdown")
        validate_timezone_aware(bar_timestamp, "bar_timestamp")

        reasons: list[str] = []

        # 1. Circuit Breaker / Drawdown Limit Check
        if current_drawdown >= self.config.max_drawdown_limit:
            reasons.append(f"CIRCUIT_BREAKER_TRIGGERED_DRAWDOWN_{current_drawdown:.4f}_EXCEEDS_{self.config.max_drawdown_limit:.4f}")
            # If in drawdown, veto all new BUY / short positions; allow EXIT or NO_TRADE
            if proposed_action in (ActionType.BUY, ActionType.SELL):
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

        # 4. Short Prohibition Check
        if proposed_exposure < 0.0 and not self.config.allow_short:
            reasons.append("SHORT_POSITIONS_PROHIBITED_BY_POLICY")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 5. Position Exposure Cap Clamping
        capped_exposure = proposed_exposure
        if abs(proposed_exposure) > self.config.max_position_exposure:
            sign = 1.0 if proposed_exposure > 0 else -1.0
            capped_exposure = sign * self.config.max_position_exposure
            reasons.append(
                f"EXPOSURE_CLAMPED_FROM_{proposed_exposure:.2f}_TO_MAX_{capped_exposure:.2f}"
            )

        # 6. Cash Buffer Check
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

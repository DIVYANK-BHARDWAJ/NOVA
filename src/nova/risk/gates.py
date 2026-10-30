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
    order_brokerage: float = 20.0            # Flat brokerage per executed order (₹20 per competition rules)
    allow_short: bool = False                # False by default for safety/back-compat, True for MANTHAN
    enable_asymmetric_short: bool = True     # When True, enables conditional shorting in verified BEAR regimes
    short_max_holding_days: int = 4          # Max days to hold short under asymmetric protocol (auto-exit before 10-day exchange rule)
    short_stop_loss_pct: float = 0.025       # Tight 2.5% stop-loss for short positions
    short_exposure_scale: float = 0.50       # 50% sizing scale for short positions
    cooldown_days_after_stop: int = 5        # Anti-churn quarantine days after a stop-loss exit
    trend_filter_ema_period: int = 50        # EMA period for falling knife trend filter
    market_breadth_buy_threshold: float = 0.40   # Minimum universe breadth (% > EMA20) required to authorize BUY
    market_breadth_short_threshold: float = 0.30 # Maximum universe breadth (% > EMA20) permitted to authorize SHORT
    caution_drawdown_threshold: float = 0.04     # Zone 1 Caution Mode threshold (4% DD -> 70% sizing)
    recovery_drawdown_threshold: float = 0.07    # Zone 2 Recovery Mode threshold (7% DD -> 50% sizing)
    sniper_drawdown_threshold: float = 0.10      # Zone 3 Sniper Mode threshold (10% DD -> 20% sizing, micro-risk)
    max_staleness_seconds: float = 300.0     # Max age of market bar before rejection (5 mins)
    config_version: str = "1.3.0"

    def __post_init__(self) -> None:
        for name, val in [
            ("max_position_exposure", self.max_position_exposure),
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_drawdown_limit", self.max_drawdown_limit),
            ("min_cash_buffer_fraction", self.min_cash_buffer_fraction),
            ("min_cash_reserve_fraction", self.min_cash_reserve_fraction),
            ("stop_loss_pct", self.stop_loss_pct),
            ("short_stop_loss_pct", self.short_stop_loss_pct),
            ("short_exposure_scale", self.short_exposure_scale),
            ("market_breadth_buy_threshold", self.market_breadth_buy_threshold),
            ("market_breadth_short_threshold", self.market_breadth_short_threshold),
            ("caution_drawdown_threshold", self.caution_drawdown_threshold),
            ("recovery_drawdown_threshold", self.recovery_drawdown_threshold),
            ("sniper_drawdown_threshold", self.sniper_drawdown_threshold),
            ("max_staleness_seconds", self.max_staleness_seconds),
            ("initial_capital", self.initial_capital),
            ("order_brokerage", self.order_brokerage),
        ]:
            validate_finite(val, name)
            if val < 0.0:
                raise NumericalValidationError(f"Risk configuration '{name}' cannot be negative, got {val}")
            if val == 0.0 and name not in ("max_staleness_seconds", "order_brokerage"):
                raise NumericalValidationError(f"Risk configuration '{name}' must be positive, got {val}")

        if not (0.0 < self.max_position_exposure <= 1.0):
            raise NumericalValidationError(f"max_position_exposure must be in (0, 1], got {self.max_position_exposure}")

        if self.max_short_holding_days <= 0 or self.short_max_holding_days <= 0:
            raise NumericalValidationError("Short holding day limits must be positive")

        if self.cooldown_days_after_stop < 0:
            raise NumericalValidationError("cooldown_days_after_stop cannot be negative")


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
        self._ticker_cooldowns: dict[str, datetime] = {}

    def register_stop_loss_exit(
        self,
        ticker: str,
        exit_time: datetime,
        cooldown_days: int | None = None,
    ) -> None:
        """Register a stop-loss exit to enforce the anti-churn post-stop quarantine."""
        from datetime import timedelta
        days = cooldown_days if cooldown_days is not None else self.config.cooldown_days_after_stop
        t_key = ticker.upper()
        self._ticker_cooldowns[t_key] = exit_time + timedelta(days=days)

    def is_ticker_in_cooldown(self, ticker: str, current_time: datetime) -> bool:
        """Check if a ticker is currently in the post-stop anti-churn quarantine."""
        t_key = ticker.upper()
        if t_key not in self._ticker_cooldowns:
            return False
        return current_time < self._ticker_cooldowns[t_key]

    def clear_cooldowns(self) -> None:
        """Reset all active ticker cooldowns (useful for test isolation)."""
        self._ticker_cooldowns.clear()

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
        ticker: str = "",
        market_breadth: float | None = None,
        market_regime: str | None = None,
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

        # 0. Short Prohibition Check: new SHORT action or negative target exposure
        # Exception: risk-reducing exits (COVER, EXIT, REDUCE) are never blocked by allow_short=False
        is_short_intent = (
            (proposed_action == ActionType.SHORT or proposed_exposure < -1e-6)
            and proposed_action not in (ActionType.COVER, ActionType.EXIT, ActionType.REDUCE)
        )
        if is_short_intent and not self.config.allow_short:
            reasons.append("SHORT_POSITIONS_PROHIBITED_BY_POLICY")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 0d. Execution Safety Checks: Pending Orders, Overselling & Nonexistent Covers
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

        # 1. Tiered Resuscitation Ladder & Emergency Circuit Breaker Check
        drawdown_exposure_multiplier = 1.0
        if current_drawdown >= self.config.max_drawdown_limit:
            reasons.append(f"CIRCUIT_BREAKER_TRIGGERED_DRAWDOWN_{current_drawdown:.4f}_EXCEEDS_{self.config.max_drawdown_limit:.4f}")
            # Emergency circuit breaker: veto all new BUY / SHORT positions; allow EXIT, SELL, COVER, or NO_TRADE
            if proposed_action in (ActionType.BUY, ActionType.SHORT, ActionType.SELL):
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
        elif current_drawdown >= self.config.sniper_drawdown_threshold:
            drawdown_exposure_multiplier = 0.20
            reasons.append(f"TIERED_RESUSCITATION_ZONE_3_SNIPER_ACTIVE_DD_{current_drawdown:.3f}")
        elif current_drawdown >= self.config.recovery_drawdown_threshold:
            drawdown_exposure_multiplier = 0.50
            reasons.append(f"TIERED_RESUSCITATION_ZONE_2_RECOVERY_ACTIVE_DD_{current_drawdown:.3f}")
        elif current_drawdown >= self.config.caution_drawdown_threshold:
            drawdown_exposure_multiplier = 0.70
            reasons.append(f"TIERED_RESUSCITATION_ZONE_1_CAUTION_ACTIVE_DD_{current_drawdown:.3f}")

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

        # 4. Short Auto-Settlement Enforcement (Asymmetric Short 4-day or MANTHAN 10-day)
        effective_short_max_days = (
            self.config.short_max_holding_days
            if self.config.enable_asymmetric_short
            else self.config.max_short_holding_days
        )
        if current_position_quantity < -1e-6 and short_holding_days >= effective_short_max_days:
            reasons.append(f"SHORT_AUTO_SETTLEMENT_CLOSE_DAYS_{short_holding_days}_LIMIT_{effective_short_max_days}")
            return RiskGateResult(
                is_approved=True,
                authorized_action=ActionType.COVER,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 5. Configurable Stop-Loss logic (Long vs Short)
        if position_entry_price > 0.0 and current_market_price > 0.0:
            if current_position_quantity > 1e-6:
                drop_pct = (position_entry_price - current_market_price) / position_entry_price
                if drop_pct >= self.config.stop_loss_pct:
                    if t_sym:
                        self.register_stop_loss_exit(t_sym, bar_timestamp)
                    reasons.append(f"STOP_LOSS_LONG_TRIGGERED_DROP_{drop_pct:.3%}_EXCEEDS_{self.config.stop_loss_pct:.3%}")
                    return RiskGateResult(
                        is_approved=True,
                        authorized_action=ActionType.SELL,
                        authorized_exposure=0.0,
                        risk_reason_codes=tuple(reasons),
                    )
            elif current_position_quantity < -1e-6:
                rise_pct = (current_market_price - position_entry_price) / position_entry_price
                effective_short_stop = (
                    self.config.short_stop_loss_pct
                    if self.config.enable_asymmetric_short
                    else self.config.stop_loss_pct
                )
                if rise_pct >= effective_short_stop:
                    if t_sym:
                        self.register_stop_loss_exit(t_sym, bar_timestamp)
                    reasons.append(f"STOP_LOSS_SHORT_TRIGGERED_RISE_{rise_pct:.3%}_EXCEEDS_{effective_short_stop:.3%}")
                    return RiskGateResult(
                        is_approved=True,
                        authorized_action=ActionType.COVER,
                        authorized_exposure=0.0,
                        risk_reason_codes=tuple(reasons),
                    )

        # 6. Position Exposure Cap Clamping
        capped_exposure = proposed_exposure
        if abs(proposed_exposure) > self.config.max_position_exposure:
            sign = 1.0 if proposed_exposure > 0 else -1.0
            capped_exposure = sign * self.config.max_position_exposure
            reasons.append(
                f"EXPOSURE_CLAMPED_FROM_{proposed_exposure:.2f}_TO_MAX_{capped_exposure:.2f}"
            )

        # 7. Short Prohibition Check
        if proposed_action == ActionType.SHORT and not self.config.allow_short:
            reasons.append("SHORT_POSITIONS_PROHIBITED_BY_POLICY")
            return RiskGateResult(
                is_approved=False,
                authorized_action=ActionType.NO_TRADE,
                authorized_exposure=0.0,
                risk_reason_codes=tuple(reasons),
            )

        # 8. MANTHAN Rule: 100% Margin + Brokerage Check for SHORT
        if proposed_action == ActionType.SHORT:
            margin_required = abs(capped_exposure) * current_portfolio_value
            # In normalized fractional mode (portfolio <= 1.0), scale order brokerage by initial capital
            effective_brokerage = (
                (self.config.order_brokerage / max(1.0, self.config.initial_capital))
                if current_portfolio_value <= 1.0
                else self.config.order_brokerage
            )
            total_required_capital = margin_required + effective_brokerage
            if available_cash < total_required_capital:
                reasons.append(
                    f"INSUFFICIENT_MARGIN_CASH_{available_cash:.2f}_REQUIRED_{total_required_capital:.2f}"
                )
                return RiskGateResult(
                    is_approved=False,
                    authorized_action=ActionType.NO_TRADE,
                    authorized_exposure=0.0,
                    risk_reason_codes=tuple(reasons),
                )
            remaining_cash = available_cash - total_required_capital
            min_buffer = self.config.min_cash_buffer_fraction * current_portfolio_value
            if remaining_cash < min_buffer:
                reasons.append(
                    f"INSUFFICIENT_CASH_BUFFER_AFTER_SHORT_{remaining_cash:.2f}_BELOW_{min_buffer:.2f}"
                )
                return RiskGateResult(
                    is_approved=False,
                    authorized_action=ActionType.NO_TRADE,
                    authorized_exposure=0.0,
                    risk_reason_codes=tuple(reasons),
                )

        # 9. Cash Buffer & Reserve Check for BUY
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

"""NOVA Profit Realization, Position Exits & Capital Recycling (Person 3 Exclusive Domain).

Implements Section 11 of the Sentinel Trader Master Project Specification:
- Multi-tier exit evaluation: stop-loss, bi-weekly short closure, trailing stop,
  profit giveback protection, partial profit scale-outs, and tax-aware deferral.
- Capital recycling: allocates freed capital to candidate opportunities ranked by
  risk-adjusted expected net utility (Edge / sigma) while preserving the cash cushion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Mapping, Sequence

from nova.decision.contracts import ActionType, PortfolioState, PositionState
from nova.quant.contracts import (
    ContractValidationError,
    MarketBar,
    NumericalValidationError,
    validate_finite,
)


@dataclass(frozen=True, slots=True)
class ProfitRealizationConfig:
    """Configurable thresholds for profit taking, loss control, and capital recycling."""
    # Profit taking & scale-out
    take_profit_threshold_pct: float = 0.08       # 8% initial profit target
    partial_exit_ratio: float = 0.50               # 50% scale-out ratio on profit target
    trailing_stop_activation_pct: float = 0.04    # 4% gain activates trailing stop
    trailing_stop_distance_pct: float = 0.03      # 3% trailing distance from peak price
    profit_giveback_max_pct: float = 0.40         # Close if >40% of peak unrealized profit erodes

    # Risk & competition limits
    stop_loss_pct: float = 0.04                   # 4% stop-loss threshold
    max_short_holding_days: int = 10              # 10 trading days short auto-settlement
    long_term_tax_threshold_days: int = 365       # 365 days for LTCG qualification
    tax_deferral_buffer_days: int = 25            # Defer non-emergency exit if within 25 days of 365
    tax_deferral_min_score: float = -0.10         # Do not defer if score is severely bearish (<= -0.10)

    # Capital recycling parameters
    min_recycle_edge_ratio: float = 0.15          # Minimum Edge / sigma for recycling candidate
    target_diversification_min: int = 4           # Minimum target active stock holdings
    target_diversification_max: int = 8           # Maximum target active stock holdings
    max_position_allocation: float = 0.25         # Max 25% equity per position
    min_cash_reserve_fraction: float = 0.20       # Maintain 20% liquid cash cushion
    config_version: str = "1.0.0"

    def __post_init__(self) -> None:
        for name, val in [
            ("take_profit_threshold_pct", self.take_profit_threshold_pct),
            ("partial_exit_ratio", self.partial_exit_ratio),
            ("trailing_stop_activation_pct", self.trailing_stop_activation_pct),
            ("trailing_stop_distance_pct", self.trailing_stop_distance_pct),
            ("profit_giveback_max_pct", self.profit_giveback_max_pct),
            ("stop_loss_pct", self.stop_loss_pct),
            ("min_recycle_edge_ratio", self.min_recycle_edge_ratio),
            ("max_position_allocation", self.max_position_allocation),
            ("min_cash_reserve_fraction", self.min_cash_reserve_fraction),
        ]:
            validate_finite(val, name)
            if val < 0.0:
                raise NumericalValidationError(f"Profit realization parameter '{name}' cannot be negative")

        if not (0.0 < self.partial_exit_ratio <= 1.0):
            raise NumericalValidationError(f"partial_exit_ratio must be in (0, 1], got {self.partial_exit_ratio}")

        if not (0.0 < self.max_position_allocation <= 1.0):
            raise NumericalValidationError(f"max_position_allocation must be in (0, 1], got {self.max_position_allocation}")

        if self.max_short_holding_days <= 0:
            raise NumericalValidationError("max_short_holding_days must be positive")


@dataclass(frozen=True, slots=True)
class ExitRecommendation:
    """Outcome of position exit evaluation."""
    should_exit: bool
    is_partial: bool
    recommended_action: ActionType
    target_exposure: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.should_exit, bool):
            raise ContractValidationError("should_exit must be a boolean")
        if not isinstance(self.is_partial, bool):
            raise ContractValidationError("is_partial must be a boolean")
        if not isinstance(self.recommended_action, ActionType):
            raise ContractValidationError(f"Invalid recommended_action: {self.recommended_action}")
        validate_finite(self.target_exposure, "target_exposure")


class ProfitRealizationManager:
    """Evaluates positions against profit-taking, trailing stop, giveback, and loss controls."""

    def __init__(self, config: ProfitRealizationConfig | None = None) -> None:
        self.config = config or ProfitRealizationConfig()

    def evaluate_position_exit(
        self,
        position: PositionState,
        current_bar: MarketBar,
        combined_score: float = 0.0,
        expected_utility: float = 0.0,
        min_net_edge: float = 0.0001,
        current_exposure: float = 0.0,
    ) -> ExitRecommendation:
        """Evaluate open position and return an actionable ExitRecommendation."""
        if position.is_flat:
            return ExitRecommendation(
                should_exit=False,
                is_partial=False,
                recommended_action=ActionType.NO_TRADE,
                target_exposure=0.0,
                reasons=("POSITION_IS_FLAT_NO_EXIT_NEEDED",),
            )

        current_price = current_bar.close
        entry_price = position.entry_price
        if entry_price <= 0.0:
            return ExitRecommendation(
                should_exit=False,
                is_partial=False,
                recommended_action=ActionType.HOLD,
                target_exposure=current_exposure,
                reasons=("INVALID_ENTRY_PRICE_ABSTAIN_HOLD",),
            )

        reasons: list[str] = []
        is_long = position.is_long
        is_short = position.is_short

        # Track effective peak/trough price for trailing stop
        peak_px = max(position.highest_price, current_price) if is_long else (position.highest_price or current_price)
        lowest_px = min(position.lowest_price, current_price) if is_short else (position.lowest_price or current_price)
        if position.lowest_price <= 0.0 and is_short:
            lowest_px = current_price

        # 1. Hard Stop-Loss Check (Highest Priority)
        if is_long:
            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct >= self.config.stop_loss_pct:
                reasons.append(f"STOP_LOSS_LONG_TRIGGERED_LOSS_{loss_pct:.2%}_EXCEEDS_{self.config.stop_loss_pct:.2%}")
                return ExitRecommendation(
                    should_exit=True,
                    is_partial=False,
                    recommended_action=ActionType.SELL,
                    target_exposure=0.0,
                    reasons=tuple(reasons),
                )
        elif is_short:
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct >= self.config.stop_loss_pct:
                reasons.append(f"STOP_LOSS_SHORT_TRIGGERED_LOSS_{loss_pct:.2%}_EXCEEDS_{self.config.stop_loss_pct:.2%}")
                return ExitRecommendation(
                    should_exit=True,
                    is_partial=False,
                    recommended_action=ActionType.COVER,
                    target_exposure=0.0,
                    reasons=tuple(reasons),
                )

        # 2. MANTHAN Rule: Short 10-Day Auto-Settlement Check
        if is_short and position.holding_days >= self.config.max_short_holding_days:
            reasons.append(f"SHORT_10_DAY_SETTLEMENT_AUTO_CLOSE_DAYS_{position.holding_days}")
            return ExitRecommendation(
                should_exit=True,
                is_partial=False,
                recommended_action=ActionType.COVER,
                target_exposure=0.0,
                reasons=tuple(reasons),
            )

        # 3. Dynamic Trailing-Stop Check
        gain_from_entry_pct = position.unrealized_pnl_pct
        if gain_from_entry_pct >= self.config.trailing_stop_activation_pct:
            if is_long and peak_px > 0.0:
                drop_from_peak = (peak_px - current_price) / peak_px
                if drop_from_peak >= self.config.trailing_stop_distance_pct:
                    reasons.append(
                        f"TRAILING_STOP_LONG_TRIGGERED_PEAK_{peak_px:.2f}_DROP_{drop_from_peak:.2%}_EXCEEDS_{self.config.trailing_stop_distance_pct:.2%}"
                    )
                    return ExitRecommendation(
                        should_exit=True,
                        is_partial=False,
                        recommended_action=ActionType.SELL,
                        target_exposure=0.0,
                        reasons=tuple(reasons),
                    )
            elif is_short and lowest_px > 0.0:
                rise_from_lowest = (current_price - lowest_px) / lowest_px
                if rise_from_lowest >= self.config.trailing_stop_distance_pct:
                    reasons.append(
                        f"TRAILING_STOP_SHORT_TRIGGERED_TROUGH_{lowest_px:.2f}_RISE_{rise_from_lowest:.2%}_EXCEEDS_{self.config.trailing_stop_distance_pct:.2%}"
                    )
                    return ExitRecommendation(
                        should_exit=True,
                        is_partial=False,
                        recommended_action=ActionType.COVER,
                        target_exposure=0.0,
                        reasons=tuple(reasons),
                    )

        # 4. Profit Giveback Protection Check
        if position.peak_unrealized_pnl > 0.0:
            giveback_pct = position.profit_giveback_pct
            if gain_from_entry_pct >= 0.03 and giveback_pct >= self.config.profit_giveback_max_pct:
                reasons.append("PROFIT_GIVEBACK_PROTECTION_EXIT")
                reasons.append(
                    f"PROFIT_GIVEBACK_PROTECTION_TRIGGERED_{giveback_pct:.2%}_EXCEEDS_{self.config.profit_giveback_max_pct:.2%}"
                )
                return ExitRecommendation(
                    should_exit=True,
                    is_partial=False,
                    recommended_action=ActionType.SELL if is_long else ActionType.COVER,
                    target_exposure=0.0,
                    reasons=tuple(reasons),
                )

        # 5. Take-Profit Target & Partial Exit Scale-Out Check
        if gain_from_entry_pct >= self.config.take_profit_threshold_pct and not position.has_taken_partial_profit:
            remaining_exposure = current_exposure * (1.0 - self.config.partial_exit_ratio)
            action = ActionType.SELL if is_long else ActionType.COVER
            reasons.append("TAKE_PROFIT_PARTIAL_50PCT_SCALE_OUT")
            reasons.append(
                f"PROFIT_TAKING_PARTIAL_EXIT_SCALE_OUT_{self.config.partial_exit_ratio:.0%}_GAIN_{gain_from_entry_pct:.2%}"
            )
            return ExitRecommendation(
                should_exit=True,
                is_partial=True,
                recommended_action=action,
                target_exposure=remaining_exposure,
                reasons=tuple(reasons),
            )

        # 6. Tax Deferral Window Check (STCG -> LTCG Optimization)
        # If held for 340-365 days, defer non-emergency discretionary exits to cross 365 days
        if is_long and (self.config.long_term_tax_threshold_days - self.config.tax_deferral_buffer_days) <= position.holding_days < self.config.long_term_tax_threshold_days:
            if combined_score > self.config.tax_deferral_min_score:
                reasons.append("TAX_AWARE_HOLDING_DEFERRAL_LTCG")
                reasons.append(
                    f"TAX_DEFERRAL_HOLD_LTCG_WINDOW_DAYS_{position.holding_days}_REMAINING_{self.config.long_term_tax_threshold_days - position.holding_days}"
                )
                return ExitRecommendation(
                    should_exit=False,
                    is_partial=False,
                    recommended_action=ActionType.HOLD,
                    target_exposure=current_exposure,
                    reasons=tuple(reasons),
                )

        # 7. Quantitative Signal Reversal Check
        if is_long and combined_score < -0.15:
            reasons.append(f"STRATEGY_BEARISH_REVERSAL_CLOSE_LONG_SCORE_{combined_score:.3f}")
            return ExitRecommendation(
                should_exit=True,
                is_partial=False,
                recommended_action=ActionType.SELL,
                target_exposure=0.0,
                reasons=tuple(reasons),
            )
        elif is_short and combined_score > 0.15:
            reasons.append(f"STRATEGY_BULLISH_REVERSAL_CLOSE_SHORT_SCORE_{combined_score:.3f}")
            return ExitRecommendation(
                should_exit=True,
                is_partial=False,
                recommended_action=ActionType.COVER,
                target_exposure=0.0,
                reasons=tuple(reasons),
            )

        # 8. Mild Signal Decay Exit (if position is not within tax deferral)
        if is_long and combined_score < -0.05 and expected_utility < min_net_edge:
            reasons.append(f"SIGNAL_DECAY_DISCRETIONARY_SELL_SCORE_{combined_score:.3f}")
            return ExitRecommendation(
                should_exit=True,
                is_partial=False,
                recommended_action=ActionType.SELL,
                target_exposure=0.0,
                reasons=tuple(reasons),
            )
        elif is_short and combined_score > 0.05 and expected_utility < min_net_edge:
            reasons.append(f"SIGNAL_DECAY_DISCRETIONARY_COVER_SCORE_{combined_score:.3f}")
            return ExitRecommendation(
                should_exit=True,
                is_partial=False,
                recommended_action=ActionType.COVER,
                target_exposure=0.0,
                reasons=tuple(reasons),
            )

        # 9. Default: Hold Position
        reasons.append("PROFIT_REALIZATION_HOLD_CRITERIA_MAINTAINED")
        return ExitRecommendation(
            should_exit=False,
            is_partial=False,
            recommended_action=ActionType.HOLD,
            target_exposure=current_exposure,
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class OpportunityCandidate:
    """Standardized candidate opportunity evaluated for capital recycling."""
    ticker: str
    expected_net_edge: float
    volatility: float = 0.02
    confidence: float = 1.0
    return_volatility: float = 0.02

    def __post_init__(self) -> None:
        if self.volatility == 0.02 and self.return_volatility != 0.02:
            object.__setattr__(self, "volatility", self.return_volatility)
        elif self.return_volatility == 0.02 and self.volatility != 0.02:
            object.__setattr__(self, "return_volatility", self.volatility)

    @property
    def edge_ratio(self) -> float:
        """Risk-adjusted edge ratio: Edge / sigma."""
        v = self.return_volatility if self.return_volatility > 0 else self.volatility
        return self.expected_net_edge / max(1e-4, v)


CandidateOpportunity = OpportunityCandidate


@dataclass(frozen=True, slots=True)
class ReinvestmentAllocation:
    """Allocated investment parameters for a cleared candidate opportunity."""
    ticker: str
    allocated_exposure: float
    allocated_capital: float
    risk_adjusted_ratio: float


@dataclass(frozen=True, slots=True)
class ReinvestmentPlan:
    """Deterministic capital recycling and deployment plan."""
    allocations: tuple[ReinvestmentAllocation, ...]
    total_allocated_capital: float
    preserved_cash_cushion: float


class CapitalRecycler:
    """Ranks and allocates freed exit capital to highest risk-adjusted utility opportunities."""

    def __init__(self, config: ProfitRealizationConfig | None = None) -> None:
        self.config = config or ProfitRealizationConfig()

    def evaluate_reinvestment_plan(
        self,
        available_cash: float,
        total_wealth: float,
        candidate_opportunities: Sequence[OpportunityCandidate],
        current_holdings_count: int = 0,
    ) -> ReinvestmentPlan:
        """Compute structured capital reinvestment plan while preserving 20% liquid cash cushion."""
        if total_wealth <= 0.0 or not candidate_opportunities:
            return ReinvestmentPlan(allocations=(), total_allocated_capital=0.0, preserved_cash_cushion=available_cash)

        required_cushion = total_wealth * self.config.min_cash_reserve_fraction
        deployable_cash = max(0.0, available_cash - required_cushion)

        slots_available = max(0, self.config.target_diversification_max - current_holdings_count)
        if slots_available <= 0 or deployable_cash <= 0.0:
            return ReinvestmentPlan(allocations=(), total_allocated_capital=0.0, preserved_cash_cushion=available_cash)

        cleared = [
            c for c in candidate_opportunities
            if c.expected_net_edge > 0.0 and c.edge_ratio >= self.config.min_recycle_edge_ratio
        ]
        ranked = sorted(cleared, key=lambda c: c.edge_ratio, reverse=True)
        chosen = ranked[:slots_available]
        if not chosen:
            return ReinvestmentPlan(allocations=(), total_allocated_capital=0.0, preserved_cash_cushion=available_cash)

        allocs: list[ReinvestmentAllocation] = []
        alloc_per_candidate = deployable_cash / len(chosen)
        max_capital_per_pos = total_wealth * self.config.max_position_allocation

        total_alloc = 0.0
        for cand in chosen:
            capital = min(alloc_per_candidate, max_capital_per_pos)
            exp = capital / total_wealth
            allocs.append(
                ReinvestmentAllocation(
                    ticker=cand.ticker,
                    allocated_exposure=round(exp, 4),
                    allocated_capital=round(capital, 2),
                    risk_adjusted_ratio=round(cand.edge_ratio, 4),
                )
            )
            total_alloc += capital

        return ReinvestmentPlan(
            allocations=tuple(allocs),
            total_allocated_capital=round(total_alloc, 2),
            preserved_cash_cushion=round(available_cash - total_alloc, 2),
        )

    def allocate_recycled_capital(
        self,
        portfolio_state: PortfolioState,
        candidates: Sequence[OpportunityCandidate],
    ) -> Mapping[str, float]:
        """Rank candidates and compute target exposures for high-utility opportunities."""
        if not candidates:
            return {}

        total_wealth = portfolio_state.total_wealth
        if total_wealth <= 0.0:
            return {}

        available_cash = portfolio_state.available_cash
        required_reserve = total_wealth * self.config.min_cash_reserve_fraction
        deployable_cash = max(0.0, available_cash - required_reserve)

        # Current active holdings count
        active_holdings = sum(1 for pos in portfolio_state.positions.values() if not pos.is_flat)
        slots_available = max(0, self.config.target_diversification_max - active_holdings)
        if slots_available <= 0 or deployable_cash <= 0.0:
            return {}

        # Filter candidates clearing the edge threshold
        cleared_candidates = [
            c for c in candidates
            if c.expected_net_edge > 0.0 and c.edge_ratio >= self.config.min_recycle_edge_ratio
        ]
        # Rank descending by edge_ratio
        ranked = sorted(cleared_candidates, key=lambda c: c.edge_ratio, reverse=True)

        allocations: dict[str, float] = {}
        deployable_fraction = deployable_cash / total_wealth

        for candidate in ranked[:slots_available]:
            # Scale exposure by confidence and volatility
            base_size = min(self.config.max_position_allocation, deployable_fraction / slots_available)
            target_exposure = base_size * min(1.0, max(0.2, candidate.confidence))
            allocations[candidate.ticker] = round(target_exposure, 4)

        return allocations

"""NOVA Sentinel Decision Engine (Person 3 Exclusive Domain).

Integrates multi-strategy quantitative signals (trend-following & mean-reversion),
evaluates all-in market friction and break-even thresholds, scores expected net utility,
and routes proposed actions through deterministic risk controls to produce authoritative
TradeDecision objects (Interface Contract C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import uuid
from typing import Sequence

from nova.decision.contracts import ActionType, OrderType, TradeDecision
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    validate_finite,
    validate_timezone_aware,
)
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.base import StrategySignal
from nova.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy


@dataclass(frozen=True, slots=True)
class DecisionEngineConfig:
    """Configurable parameters for the multi-strategy decision engine."""
    trend_weight: float = 0.6
    mean_reversion_weight: float = 0.4
    min_net_edge_bps: float = 1.0             # 1 bp minimum required net edge
    uncertainty_multiplier: float = 0.5       # Uncertainty penalty scaling
    target_exposure: float = 0.20             # Default target exposure fraction (20%)
    engine_id: str = "sentinel_decision_engine"
    engine_version: str = "1.0.0"

    def __post_init__(self) -> None:
        validate_finite(self.trend_weight, "trend_weight")
        validate_finite(self.mean_reversion_weight, "mean_reversion_weight")
        if self.trend_weight < 0.0 or self.mean_reversion_weight < 0.0:
            raise ContractValidationError("Strategy weights cannot be negative")
        if math.isclose(self.trend_weight + self.mean_reversion_weight, 0.0):
            raise ContractValidationError("Total strategy weights cannot sum to zero")

        validate_finite(self.min_net_edge_bps, "min_net_edge_bps")
        validate_finite(self.uncertainty_multiplier, "uncertainty_multiplier")
        validate_finite(self.target_exposure, "target_exposure")
        if not (0.0 < self.target_exposure <= 1.0):
            raise ContractValidationError(f"target_exposure must be in (0, 1], got {self.target_exposure}")


class SentinelDecisionEngine:
    """Core decision intelligence combining signals, friction economics, and risk gates."""

    def __init__(
        self,
        config: DecisionEngineConfig | None = None,
        trend_strategy: TrendFollowingStrategy | None = None,
        mean_reversion_strategy: MeanReversionStrategy | None = None,
        friction_engine: MarketFrictionEngine | None = None,
        risk_gate: DeterministicRiskGate | None = None,
    ) -> None:
        self.config = config or DecisionEngineConfig()
        self.trend_strategy = trend_strategy or TrendFollowingStrategy()
        self.mean_reversion_strategy = mean_reversion_strategy or MeanReversionStrategy()
        self.friction_engine = friction_engine or MarketFrictionEngine()
        self.risk_gate = risk_gate or DeterministicRiskGate()

    def evaluate_decision(
        self,
        bars: Sequence[MarketBar],
        current_position_exposure: float = 0.0,
        current_drawdown: float = 0.0,
        available_cash: float = 1.0,
        current_portfolio_value: float = 1.0,
        as_of_time: datetime | None = None,
    ) -> TradeDecision:
        """Evaluate market state causally and produce an authoritative TradeDecision."""
        if not bars:
            raise ContractValidationError("Cannot evaluate decision on empty bar sequence")

        latest_bar = bars[-1]
        as_of = as_of_time or latest_bar.source_timestamp
        ticker = latest_bar.ticker

        reasons: list[str] = []

        # 1. Evaluate Individual Strategies Causally
        trend_sig: StrategySignal = self.trend_strategy.evaluate(bars)
        mr_sig: StrategySignal = self.mean_reversion_strategy.evaluate(bars)

        # Collect component reasons
        for r in trend_sig.reason_codes:
            reasons.append(f"TREND:{r}")
        for r in mr_sig.reason_codes:
            reasons.append(f"MEAN_REV:{r}")

        # Check for immediate data quality or warmup abstention
        if latest_bar.data_quality != DataQualityStatus.VALID:
            reasons.append("ABSTAIN_DUE_TO_INVALID_DATA_QUALITY")
            return self._build_abstain_decision(
                ticker=ticker,
                timestamp=as_of,
                reasons=reasons,
                current_position=current_position_exposure,
            )

        # 2. Combine Quantitative Signals
        tot_weight = self.config.trend_weight + self.config.mean_reversion_weight
        norm_w_trend = self.config.trend_weight / tot_weight
        norm_w_mr = self.config.mean_reversion_weight / tot_weight

        combined_score = (trend_sig.raw_score * norm_w_trend) + (mr_sig.raw_score * norm_w_mr)

        gross_trend = trend_sig.expected_gross_return or 0.0
        gross_mr = mr_sig.expected_gross_return or 0.0
        expected_gross = (gross_trend * norm_w_trend) + (gross_mr * norm_w_mr)

        # Volatility & Uncertainty Penalty
        vol_est = max(
            trend_sig.volatility_estimate or 0.01,
            mr_sig.volatility_estimate or 0.01,
        )
        uncertainty_penalty = self.config.uncertainty_multiplier * vol_est

        # 3. Market Friction & Break-Even Calculation
        all_in_cost = self.friction_engine.calculate_all_in_cost(is_roundtrip=True)
        prob_clearing = self.friction_engine.estimate_probability_of_clearing_costs(
            expected_gross_return=expected_gross,
            all_in_cost=all_in_cost,
            volatility=vol_est,
        )
        expected_net = expected_gross - all_in_cost
        expected_utility = expected_net - uncertainty_penalty
        min_net_edge = self.config.min_net_edge_bps / 10000.0

        reasons.append(f"COMBINED_SCORE_{combined_score:.3f}")
        reasons.append(f"EXPECTED_GROSS_{expected_gross:.5f}_COST_{all_in_cost:.5f}_NET_{expected_net:.5f}_UTILITY_{expected_utility:.5f}")

        # 4. Action Selection Logic
        candidate_action: ActionType
        candidate_exposure: float

        is_currently_invested = abs(current_position_exposure) > 1e-4

        if combined_score > 0.15 and expected_utility >= min_net_edge:
            if not is_currently_invested:
                candidate_action = ActionType.BUY
                candidate_exposure = self.config.target_exposure
                reasons.append("SIGNAL_BULLISH_NET_EDGE_EXCEEDS_FRICTION_BUY")
            else:
                candidate_action = ActionType.HOLD
                candidate_exposure = current_position_exposure
                reasons.append("SIGNAL_BULLISH_POSITION_MAINTAINED_HOLD")

        elif combined_score < -0.15:
            if is_currently_invested and current_position_exposure > 0:
                candidate_action = ActionType.EXIT
                candidate_exposure = 0.0
                reasons.append("SIGNAL_BEARISH_CLOSING_LONG_POSITION_EXIT")
            else:
                candidate_action = ActionType.NO_TRADE
                candidate_exposure = 0.0
                reasons.append("SIGNAL_BEARISH_FLAT_PORTFOLIO_NO_TRADE")

        else:
            # Score near zero or net edge does not survive friction
            if expected_net < min_net_edge and combined_score > 0.15:
                reasons.append("POSITIVE_GROSS_EDGE_ERODED_BY_FRICTION_ABSTAIN")

            if is_currently_invested:
                # Neutral signal while holding -> maintain or exit based on score sign
                if combined_score < -0.05:
                    candidate_action = ActionType.EXIT
                    candidate_exposure = 0.0
                    reasons.append("NEUTRAL_WEAK_SIGNAL_EXIT_LONG")
                else:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("NEUTRAL_SIGNAL_HOLD_EXISTING")
            else:
                candidate_action = ActionType.NO_TRADE
                candidate_exposure = 0.0
                reasons.append("NO_SUFFICIENT_NET_EDGE_NO_TRADE")

        # 5. Deterministic Risk Gate Evaluation
        risk_result = self.risk_gate.evaluate(
            proposed_action=candidate_action,
            proposed_exposure=candidate_exposure,
            data_quality=latest_bar.data_quality,
            bar_timestamp=latest_bar.source_timestamp,
            as_of_time=as_of,
            current_drawdown=current_drawdown,
            available_cash=available_cash,
            current_portfolio_value=current_portfolio_value,
        )

        for r_code in risk_result.risk_reason_codes:
            reasons.append(f"RISK:{r_code}")

        final_action = risk_result.authorized_action
        final_exposure = risk_result.authorized_exposure
        order_type = OrderType.MARKET if final_action in (ActionType.BUY, ActionType.SELL, ActionType.EXIT, ActionType.REDUCE) else OrderType.NONE

        return TradeDecision(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            decision_timestamp=as_of,
            instrument_id=ticker,
            selected_action=final_action,
            target_quantity_or_exposure=final_exposure,
            order_type=order_type,
            selected_strategy="ensemble_trend_and_mean_reversion",
            selection_or_rejection_reasons=tuple(reasons),
            configuration_version=self.config.engine_version,
            expected_gross_return=expected_gross,
            estimated_all_in_cost=all_in_cost,
            expected_net_return=expected_net,
            probability_of_clearing_costs=prob_clearing,
            uncertainty_estimate=uncertainty_penalty,
            estimated_risk_contribution=abs(final_exposure) * vol_est,
            model_versions=(
                ("trend_following", self.trend_strategy.config.strategy_id),
                ("mean_reversion", self.mean_reversion_strategy.config.strategy_id),
                ("risk_gate", self.risk_gate.config.config_version),
            ),
        )

    def _build_abstain_decision(
        self,
        ticker: str,
        timestamp: datetime,
        reasons: list[str],
        current_position: float,
    ) -> TradeDecision:
        """Construct a safe abstention / hold decision."""
        return TradeDecision(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            decision_timestamp=timestamp,
            instrument_id=ticker,
            selected_action=ActionType.NO_TRADE if abs(current_position) < 1e-4 else ActionType.HOLD,
            target_quantity_or_exposure=current_position,
            order_type=OrderType.NONE,
            selected_strategy="ensemble_abstain",
            selection_or_rejection_reasons=tuple(reasons),
            configuration_version=self.config.engine_version,
            expected_gross_return=0.0,
            estimated_all_in_cost=0.0,
            expected_net_return=0.0,
            probability_of_clearing_costs=0.5,
            uncertainty_estimate=0.0,
            estimated_risk_contribution=0.0,
        )

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

from nova.decision.contracts import (
    ActionType,
    OrderType,
    PortfolioState,
    PositionState,
    TradeDecision,
)
from nova.decision.prediction import BasePredictor, PredictionResult
from nova.decision.regime import MarketRegimeDetector, MarketRegimeType
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
    use_regime_weighting: bool = False        # Whether to dynamically weight by regime efficiency ratio
    predictor_weight: float = 0.0             # Weight of ML return prediction in [0.0, 1.0]
    min_prediction_confidence: float = 0.5    # Minimum confidence required to incorporate ML prediction
    allow_short: bool = False                # False by default for compatibility, True for MANTHAN
    rebalance_band_fraction: float = 0.015    # 1.5% deadband to reject repeated churn orders
    stop_loss_pct: float = 0.04              # 4% stop-loss threshold
    max_short_holding_days: int = 10         # 10 trading days short auto-settlement
    initial_capital: float = 1000000.0       # MANTHAN initial capital ₹10,00,000
    engine_id: str = "sentinel_decision_engine"
    engine_version: str = "1.2.0"

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

        validate_finite(self.predictor_weight, "predictor_weight")
        if not (0.0 <= self.predictor_weight <= 1.0):
            raise ContractValidationError(f"predictor_weight must be in [0, 1], got {self.predictor_weight}")
        validate_finite(self.min_prediction_confidence, "min_prediction_confidence")
        if not (0.0 <= self.min_prediction_confidence <= 1.0):
            raise ContractValidationError(f"min_prediction_confidence must be in [0, 1], got {self.min_prediction_confidence}")

        validate_finite(self.rebalance_band_fraction, "rebalance_band_fraction")
        if self.rebalance_band_fraction < 0.0:
            raise ContractValidationError("rebalance_band_fraction cannot be negative")
        validate_finite(self.stop_loss_pct, "stop_loss_pct")
        if self.stop_loss_pct <= 0.0:
            raise ContractValidationError("stop_loss_pct must be positive")
        validate_finite(self.initial_capital, "initial_capital")
        if self.initial_capital <= 0.0:
            raise ContractValidationError("initial_capital must be positive")


class SentinelDecisionEngine:
    """Core decision intelligence combining signals, friction economics, and risk gates."""

    def __init__(
        self,
        config: DecisionEngineConfig | None = None,
        trend_strategy: TrendFollowingStrategy | None = None,
        mean_reversion_strategy: MeanReversionStrategy | None = None,
        friction_engine: MarketFrictionEngine | None = None,
        risk_gate: DeterministicRiskGate | None = None,
        predictor: BasePredictor | None = None,
        regime_detector: MarketRegimeDetector | None = None,
    ) -> None:
        self.config = config or DecisionEngineConfig()
        self.trend_strategy = trend_strategy or TrendFollowingStrategy()
        self.mean_reversion_strategy = mean_reversion_strategy or MeanReversionStrategy()
        self.friction_engine = friction_engine or MarketFrictionEngine()
        self.risk_gate = risk_gate or DeterministicRiskGate()
        self.predictor = predictor
        self.regime_detector = regime_detector

    def evaluate_decision(
        self,
        bars: Sequence[MarketBar],
        current_position_exposure: float = 0.0,
        current_drawdown: float = 0.0,
        available_cash: float = 1.0,
        current_portfolio_value: float = 1.0,
        as_of_time: datetime | None = None,
        position_state: PositionState | None = None,
        portfolio_state: PortfolioState | None = None,
    ) -> TradeDecision:
        """Evaluate market state causally and produce an authoritative TradeDecision."""
        if not bars:
            raise ContractValidationError("Cannot evaluate decision on empty bar sequence")

        latest_bar = bars[-1]
        as_of = as_of_time or latest_bar.source_timestamp
        ticker = latest_bar.ticker

        # Consume position and portfolio state if provided
        if portfolio_state is not None:
            available_cash = portfolio_state.available_cash
            current_portfolio_value = portfolio_state.total_wealth
            current_drawdown = portfolio_state.current_drawdown
            pos = position_state or portfolio_state.get_position(ticker)
            current_quantity = pos.quantity
            entry_price = pos.entry_price
            holding_days = pos.holding_days
            locked_margin = pos.locked_margin
            current_position_exposure = (pos.market_value / current_portfolio_value) if current_portfolio_value > 0 else 0.0
        elif position_state is not None:
            current_quantity = position_state.quantity
            entry_price = position_state.entry_price
            holding_days = position_state.holding_days
            locked_margin = position_state.locked_margin
            if abs(current_position_exposure) <= 1e-4 and current_portfolio_value > 0.0:
                current_position_exposure = position_state.market_value / current_portfolio_value
        else:
            current_quantity = (current_position_exposure * current_portfolio_value) / max(1e-6, latest_bar.close)
            entry_price = latest_bar.close
            holding_days = 0
            locked_margin = 0.0

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

        # 2. Combine Quantitative Signals with Optional Regime Adaptation
        tot_weight = self.config.trend_weight + self.config.mean_reversion_weight
        norm_w_trend = self.config.trend_weight / tot_weight
        norm_w_mr = self.config.mean_reversion_weight / tot_weight
        regime_high_vol = False

        if self.config.use_regime_weighting and self.regime_detector is not None:
            try:
                regime_out = self.regime_detector.detect_regime(bars)
                reasons.append(f"REGIME:{regime_out.regime_type.value}_ER_{regime_out.efficiency_ratio:.3f}_TREND_SHARE_{regime_out.trend_share:.3f}")
                if regime_out.regime_type != MarketRegimeType.UNKNOWN:
                    norm_w_trend = regime_out.trend_share
                    norm_w_mr = 1.0 - regime_out.trend_share
                    if regime_out.is_high_volatility or regime_out.regime_type == MarketRegimeType.HIGH_VOLATILITY:
                        regime_high_vol = True
                        reasons.append("REGIME_HIGH_VOLATILITY_UNCERTAINTY_BOOST")
                else:
                    reasons.append("REGIME_UNKNOWN_FALLBACK_DEFAULT_WEIGHTS")
            except Exception as e:
                reasons.append(f"REGIME_ERROR_FALLBACK_DEFAULT_WEIGHTS:{str(e)}")

        base_score = (trend_sig.raw_score * norm_w_trend) + (mr_sig.raw_score * norm_w_mr)
        gross_trend = trend_sig.expected_gross_return or 0.0
        gross_mr = mr_sig.expected_gross_return or 0.0
        base_gross = (gross_trend * norm_w_trend) + (gross_mr * norm_w_mr)

        # 3. Integrate Causal Return Prediction with Safe Fallback
        combined_score = base_score
        expected_gross = base_gross

        if self.predictor is not None and self.config.predictor_weight > 0.0:
            try:
                pred = self.predictor.predict(bars)
                if pred.is_fallback:
                    reasons.append("PREDICTOR_FALLBACK_TO_BASELINE")
                elif pred.confidence < self.config.min_prediction_confidence:
                    reasons.append(f"PREDICTOR_LOW_CONFIDENCE_{pred.confidence:.2f}_FALLBACK_TO_BASELINE")
                else:
                    pw = self.config.predictor_weight
                    pred_score = math.tanh(pred.expected_return * 50.0)
                    combined_score = ((1.0 - pw) * base_score) + (pw * pred_score)
                    expected_gross = ((1.0 - pw) * base_gross) + (pw * pred.expected_return)
                    reasons.append(f"PREDICTOR_ACTIVE_PW_{pw:.2f}_RET_{pred.expected_return:.5f}_CONF_{pred.confidence:.2f}")
            except Exception as e:
                reasons.append(f"PREDICTOR_ERROR_FALLBACK_TO_BASELINE:{str(e)}")

        # Volatility & Uncertainty Penalty
        vol_est = max(
            trend_sig.volatility_estimate or 0.01,
            mr_sig.volatility_estimate or 0.01,
        )
        unc_mult = self.config.uncertainty_multiplier * (1.5 if regime_high_vol else 1.0)
        uncertainty_penalty = unc_mult * vol_est

        # 4. Market Friction & Break-Even Calculation
        all_in_cost = self.friction_engine.calculate_all_in_cost(is_roundtrip=True)
        prob_clearing = self.friction_engine.estimate_probability_of_clearing_costs(
            expected_gross_return=expected_gross,
            all_in_cost=all_in_cost,
            volatility=vol_est,
        )
        expected_net = expected_gross - all_in_cost
        expected_utility = expected_net - uncertainty_penalty
        min_net_edge = self.config.min_net_edge_bps / 10000.0

        # Break-Even Price Calculation
        be_price = self.friction_engine.calculate_break_even_price(
            entry_price=latest_bar.close,
            quantity=max(1.0, abs(current_quantity)),
            is_short=(current_position_exposure < -1e-4),
        )
        reasons.append(f"BREAK_EVEN_PRICE_{be_price:.2f}")
        reasons.append(f"COMBINED_SCORE_{combined_score:.3f}")
        reasons.append(f"EXPECTED_GROSS_{expected_gross:.5f}_COST_{all_in_cost:.5f}_NET_{expected_net:.5f}_UTILITY_{expected_utility:.5f}")

        # 5. Position-Aware Action Selection Logic (MANTHAN 2026 Rules)
        candidate_action: ActionType
        candidate_exposure: float

        is_long = current_position_exposure > 1e-4
        is_short = current_position_exposure < -1e-4
        is_flat = not is_long and not is_short

        if is_long:
            # Check stop loss
            if latest_bar.close <= entry_price * (1.0 - self.config.stop_loss_pct):
                candidate_action = ActionType.SELL
                candidate_exposure = 0.0
                reasons.append("STOP_LOSS_LONG_TRIGGERED_SELL")
            elif combined_score > 0.15 and expected_utility >= min_net_edge:
                target_exp = self.config.target_exposure
                # Churn prevention: check rebalance band
                if abs(target_exp - current_position_exposure) < self.config.rebalance_band_fraction:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("POSITION_WITHIN_REBALANCE_BAND_HOLD")
                else:
                    candidate_action = ActionType.BUY
                    candidate_exposure = target_exp
                    reasons.append("ADD_TO_LONG_POSITION_BUY")
            elif combined_score < -0.15:
                # Bearish reversal: exit owned long shares
                candidate_action = ActionType.SELL
                candidate_exposure = 0.0
                reasons.append("BEARISH_REVERSAL_CLOSE_LONG_SELL")
            else:
                if combined_score < -0.05:
                    candidate_action = ActionType.SELL
                    candidate_exposure = 0.0
                    reasons.append("NEUTRAL_WEAK_SIGNAL_SELL_LONG")
                else:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("NEUTRAL_SIGNAL_HOLD_LONG")

        elif is_short:
            # Check 10-day settlement
            if holding_days >= self.config.max_short_holding_days:
                candidate_action = ActionType.COVER
                candidate_exposure = 0.0
                reasons.append(f"SHORT_10_DAY_SETTLEMENT_AUTO_CLOSE_DAYS_{holding_days}")
            # Check stop loss
            elif latest_bar.close >= entry_price * (1.0 + self.config.stop_loss_pct):
                candidate_action = ActionType.COVER
                candidate_exposure = 0.0
                reasons.append("STOP_LOSS_SHORT_TRIGGERED_COVER")
            elif combined_score < -0.15:
                target_exp = -self.config.target_exposure
                if abs(target_exp - current_position_exposure) < self.config.rebalance_band_fraction:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("POSITION_WITHIN_REBALANCE_BAND_HOLD")
                else:
                    candidate_action = ActionType.SHORT
                    candidate_exposure = target_exp
                    reasons.append("ADD_TO_SHORT_POSITION_SHORT")
            elif combined_score > 0.15:
                # Bullish reversal: close short position
                candidate_action = ActionType.COVER
                candidate_exposure = 0.0
                reasons.append("BULLISH_REVERSAL_CLOSE_SHORT_COVER")
            else:
                if combined_score > 0.05:
                    candidate_action = ActionType.COVER
                    candidate_exposure = 0.0
                    reasons.append("NEUTRAL_WEAK_SIGNAL_COVER_SHORT")
                else:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("NEUTRAL_SIGNAL_HOLD_SHORT")

        else:
            # Currently flat portfolio
            if combined_score > 0.15 and expected_utility >= min_net_edge:
                candidate_action = ActionType.BUY
                candidate_exposure = self.config.target_exposure
                reasons.append("SIGNAL_BULLISH_NET_EDGE_EXCEEDS_FRICTION_BUY")
            elif combined_score < -0.15:
                can_short = self.config.allow_short or self.risk_gate.config.allow_short
                if can_short and expected_net < -min_net_edge:
                    candidate_action = ActionType.SHORT
                    candidate_exposure = -self.config.target_exposure
                    reasons.append("SIGNAL_BEARISH_NET_EDGE_EXCEEDS_FRICTION_SHORT")
                else:
                    candidate_action = ActionType.NO_TRADE
                    candidate_exposure = 0.0
                    reasons.append("SIGNAL_BEARISH_FLAT_PORTFOLIO_NO_TRADE")
            else:
                if expected_net < min_net_edge and combined_score > 0.15:
                    reasons.append("POSITIVE_GROSS_EDGE_ERODED_BY_FRICTION_ABSTAIN")
                candidate_action = ActionType.NO_TRADE
                candidate_exposure = 0.0
                reasons.append("NO_SUFFICIENT_NET_EDGE_NO_TRADE")

        # 6. Deterministic Risk Gate Evaluation
        risk_result = self.risk_gate.evaluate(
            proposed_action=candidate_action,
            proposed_exposure=candidate_exposure,
            data_quality=latest_bar.data_quality,
            bar_timestamp=latest_bar.source_timestamp,
            as_of_time=as_of,
            current_drawdown=current_drawdown,
            available_cash=available_cash,
            current_portfolio_value=current_portfolio_value,
            current_position_quantity=current_quantity,
            position_entry_price=entry_price,
            current_market_price=latest_bar.close,
            short_holding_days=holding_days,
            locked_margin=locked_margin,
        )

        for r_code in risk_result.risk_reason_codes:
            reasons.append(f"RISK:{r_code}")

        final_action = risk_result.authorized_action
        final_exposure = risk_result.authorized_exposure
        order_type = OrderType.MARKET if final_action in (
            ActionType.BUY,
            ActionType.SELL,
            ActionType.SHORT,
            ActionType.COVER,
            ActionType.EXIT,
            ActionType.REDUCE,
        ) else OrderType.NONE

        model_versions_list = [
            ("trend_following", self.trend_strategy.config.strategy_id),
            ("mean_reversion", self.mean_reversion_strategy.config.strategy_id),
            ("risk_gate", self.risk_gate.config.config_version),
        ]
        if self.regime_detector is not None:
            r_cfg = getattr(self.regime_detector, "config", None)
            r_ver = getattr(r_cfg, "config_version", "1.0.0") if r_cfg else "1.0.0"
            model_versions_list.append(("regime_detector", r_ver))
        if self.predictor is not None:
            p_cfg = getattr(self.predictor, "config", None)
            p_ver = getattr(p_cfg, "model_version", "1.0.0") if p_cfg else getattr(self.predictor, "model_version", "1.0.0")
            model_versions_list.append(("ml_predictor", p_ver))

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
            model_versions=tuple(model_versions_list),
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

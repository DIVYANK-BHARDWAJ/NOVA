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
from typing import Any, TYPE_CHECKING, Sequence

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
from nova.portfolio.profit_realization import ProfitRealizationConfig, ProfitRealizationManager
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    validate_finite,
    validate_timezone_aware,
)
if TYPE_CHECKING:
    from nova.risk.gates import DeterministicRiskGate
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
    enable_asymmetric_short: bool = True     # Enable conditional shorting under strict defensive rules
    enable_contradiction_veto: bool = True   # Veto entry when trend and mean-reversion strongly disagree
    contradiction_threshold: float = 0.35    # Conviction magnitude threshold to trigger contradiction veto
    enable_trend_filter: bool = True         # Never buy when Close < EMA50 (kills falling knife)
    trend_filter_ema_period: int = 50        # Period for EMA trend filter
    market_breadth_buy_threshold: float = 0.40   # Min market breadth to allow BUY
    market_breadth_short_threshold: float = 0.30 # Max market breadth to allow SHORT
    rebalance_band_fraction: float = 0.015    # 1.5% deadband to reject repeated churn orders
    stop_loss_pct: float = 0.04              # 4% stop-loss threshold
    max_short_holding_days: int = 10         # 10 trading days short auto-settlement
    initial_capital: float = 1000000.0       # MANTHAN initial capital ₹10,00,000
    enable_profit_realization: bool = False   # Enable profit-taking, trailing stop, giveback protection
    profit_realization_config: ProfitRealizationConfig = field(default_factory=ProfitRealizationConfig)
    engine_id: str = "sentinel_decision_engine"
    engine_version: str = "1.4.0"

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
        regime_detector: Any = None,
        profit_realization_manager: ProfitRealizationManager | None = None,
    ) -> None:
        self.config = config or DecisionEngineConfig()
        self.trend_strategy = trend_strategy or TrendFollowingStrategy()
        self.mean_reversion_strategy = mean_reversion_strategy or MeanReversionStrategy()
        self.friction_engine = friction_engine or MarketFrictionEngine()
        if risk_gate is None:
            from nova.risk.gates import DeterministicRiskGate
            risk_gate = DeterministicRiskGate()
        self.risk_gate = risk_gate
        self.predictor = predictor
        self.regime_detector = regime_detector
        self.profit_realization_manager = profit_realization_manager or ProfitRealizationManager(
            self.config.profit_realization_config
        )

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
        market_breadth: float | None = None,
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
            pos = position_state
            current_quantity = position_state.quantity
            entry_price = position_state.entry_price
            holding_days = position_state.holding_days
            locked_margin = position_state.locked_margin
            if abs(current_position_exposure) <= 1e-4 and current_portfolio_value > 0.0:
                current_position_exposure = position_state.market_value / current_portfolio_value
        else:
            pos = None
            if abs(current_position_exposure) > 1e-4:
                # Active position claimed but no authoritative position/portfolio state provided
                # Fail safely: do NOT invent entry_price = latest_bar.close or guess holding_days = 0
                return self._build_abstain_decision(
                    ticker=ticker,
                    timestamp=as_of,
                    reasons=["PORTFOLIO_STATE_UNCONFIRMED_ABSTAIN_HOLD"],
                    current_position=current_position_exposure,
                )
            current_quantity = 0.0
            entry_price = 0.0
            holding_days = 0
            locked_margin = 0.0

        if abs(current_quantity) > 1e-6 and entry_price <= 0.0:
            # Active shares reported with zero or negative entry price is invalid state
            return self._build_abstain_decision(
                ticker=ticker,
                timestamp=as_of,
                reasons=["INVALID_POSITION_ENTRY_PRICE_ABSTAIN_HOLD"],
                current_position=current_position_exposure,
            )

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
        regime_type_str: str | None = None

        if self.config.use_regime_weighting and self.regime_detector is not None:
            try:
                regime_out = self.regime_detector.detect_regime(bars)
                regime_type_str = regime_out.regime_type.value
                reasons.append(f"REGIME:{regime_type_str}_ER_{regime_out.efficiency_ratio:.3f}_TREND_SHARE_{regime_out.trend_share:.3f}")
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

        # 4. Market Friction & Side-Aware Economics Calculation
        all_in_cost = self.friction_engine.calculate_all_in_cost(is_roundtrip=True)
        min_net_edge = self.config.min_net_edge_bps / 10000.0

        # Long-side return economics
        long_expected_gross = expected_gross
        long_expected_net = long_expected_gross - all_in_cost
        long_expected_utility = long_expected_net - uncertainty_penalty
        long_prob_clearing = self.friction_engine.estimate_probability_of_clearing_costs(
            expected_gross_return=long_expected_gross,
            all_in_cost=all_in_cost,
            volatility=vol_est,
        )

        # Short-side return economics (short positions profit from underlying decline)
        short_expected_gross = -expected_gross
        short_expected_net = short_expected_gross - all_in_cost
        short_expected_utility = short_expected_net - uncertainty_penalty
        short_prob_clearing = self.friction_engine.estimate_probability_of_clearing_costs(
            expected_gross_return=short_expected_gross,
            all_in_cost=all_in_cost,
            volatility=vol_est,
        )

        # Break-Even Price Calculation (side-aware)
        is_short_direction = (current_position_exposure < -1e-4) or (
            abs(current_position_exposure) <= 1e-4 and combined_score < -0.15
        )
        be_price = self.friction_engine.calculate_break_even_price(
            entry_price=latest_bar.close,
            quantity=max(1.0, abs(current_quantity)),
            is_short=is_short_direction,
        )
        reasons.append(f"BREAK_EVEN_PRICE_{be_price:.2f}")
        reasons.append(f"COMBINED_SCORE_{combined_score:.3f}")
        reasons.append(
            f"LONG_ECONOMICS:GROSS_{long_expected_gross:.5f}_NET_{long_expected_net:.5f}_UTILITY_{long_expected_utility:.5f}"
        )
        reasons.append(
            f"SHORT_ECONOMICS:GROSS_{short_expected_gross:.5f}_NET_{short_expected_net:.5f}_UTILITY_{short_expected_utility:.5f}"
        )

        # 5. Position-Aware Action Selection Logic (MANTHAN 2026 Rules)
        candidate_action: ActionType
        candidate_exposure: float

        is_long = current_position_exposure > 1e-4
        is_short = current_position_exposure < -1e-4
        is_flat = not is_long and not is_short

        active_pos = pos if pos is not None else PositionState(
            ticker=ticker,
            quantity=current_quantity,
            entry_price=entry_price,
            current_market_price=latest_bar.close,
            holding_days=holding_days,
            locked_margin=locked_margin,
            highest_price=latest_bar.close if is_long else 0.0,
            lowest_price=latest_bar.close if is_short else 0.0,
        )

        if (is_long or is_short) and self.config.enable_profit_realization:
            active_utility = short_expected_utility if is_short else long_expected_utility
            exit_rec = self.profit_realization_manager.evaluate_position_exit(
                position=active_pos,
                current_bar=latest_bar,
                combined_score=combined_score,
                expected_utility=active_utility,
                min_net_edge=min_net_edge,
                current_exposure=current_position_exposure,
            )
            for r in exit_rec.reasons:
                reasons.append(r)

            if exit_rec.should_exit:
                candidate_action = exit_rec.recommended_action
                candidate_exposure = exit_rec.target_exposure
            elif is_long:
                if combined_score > 0.15 and long_expected_utility >= min_net_edge:
                    target_exp = self.config.target_exposure
                    if abs(target_exp - current_position_exposure) < self.config.rebalance_band_fraction:
                        candidate_action = ActionType.HOLD
                        candidate_exposure = current_position_exposure
                        reasons.append("POSITION_WITHIN_REBALANCE_BAND_HOLD")
                    else:
                        candidate_action = ActionType.BUY
                        candidate_exposure = target_exp
                        reasons.append("ADD_TO_LONG_POSITION_BUY")
                else:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("NEUTRAL_SIGNAL_HOLD_LONG")
            else:  # is_short
                if combined_score < -0.15 and short_expected_utility >= min_net_edge:
                    target_exp = -self.config.target_exposure
                    if abs(target_exp - current_position_exposure) < self.config.rebalance_band_fraction:
                        candidate_action = ActionType.HOLD
                        candidate_exposure = current_position_exposure
                        reasons.append("POSITION_WITHIN_REBALANCE_BAND_HOLD")
                    elif target_exp > current_position_exposure:
                        # Target is less negative than current: reducing short exposure requires COVER
                        candidate_action = ActionType.COVER
                        candidate_exposure = target_exp
                        reasons.append("REDUCE_SHORT_POSITION_COVER")
                    else:
                        candidate_action = ActionType.SHORT
                        candidate_exposure = target_exp
                        reasons.append("ADD_TO_SHORT_POSITION_SHORT")
                else:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("NEUTRAL_SIGNAL_HOLD_SHORT")

        elif is_long:
            # Fallback legacy long logic if profit realization disabled
            if latest_bar.close <= entry_price * (1.0 - self.config.stop_loss_pct):
                candidate_action = ActionType.SELL
                candidate_exposure = 0.0
                reasons.append("STOP_LOSS_LONG_TRIGGERED_SELL")
            elif combined_score > 0.15 and long_expected_utility >= min_net_edge:
                target_exp = self.config.target_exposure
                if abs(target_exp - current_position_exposure) < self.config.rebalance_band_fraction:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("POSITION_WITHIN_REBALANCE_BAND_HOLD")
                else:
                    candidate_action = ActionType.BUY
                    candidate_exposure = target_exp
                    reasons.append("ADD_TO_LONG_POSITION_BUY")
            elif combined_score < -0.15:
                candidate_action = ActionType.SELL
                candidate_exposure = 0.0
                reasons.append("BEARISH_REVERSAL_CLOSE_LONG_SELL")
            else:
                candidate_action = ActionType.HOLD
                candidate_exposure = current_position_exposure
                reasons.append("NEUTRAL_SIGNAL_HOLD_LONG")

        elif is_short:
            # Fallback legacy short logic if profit realization disabled
            if holding_days >= self.config.max_short_holding_days:
                candidate_action = ActionType.COVER
                candidate_exposure = 0.0
                reasons.append(f"SHORT_10_DAY_SETTLEMENT_AUTO_CLOSE_DAYS_{holding_days}")
            elif latest_bar.close >= entry_price * (1.0 + self.config.stop_loss_pct):
                candidate_action = ActionType.COVER
                candidate_exposure = 0.0
                reasons.append("STOP_LOSS_SHORT_TRIGGERED_COVER")
            elif combined_score < -0.15 and short_expected_utility >= min_net_edge:
                target_exp = -self.config.target_exposure
                if abs(target_exp - current_position_exposure) < self.config.rebalance_band_fraction:
                    candidate_action = ActionType.HOLD
                    candidate_exposure = current_position_exposure
                    reasons.append("POSITION_WITHIN_REBALANCE_BAND_HOLD")
                elif target_exp > current_position_exposure:
                    candidate_action = ActionType.COVER
                    candidate_exposure = target_exp
                    reasons.append("REDUCE_SHORT_POSITION_COVER")
                else:
                    candidate_action = ActionType.SHORT
                    candidate_exposure = target_exp
                    reasons.append("ADD_TO_SHORT_POSITION_SHORT")
            elif combined_score > 0.15:
                candidate_action = ActionType.COVER
                candidate_exposure = 0.0
                reasons.append("BULLISH_REVERSAL_CLOSE_SHORT_COVER")
            else:
                candidate_action = ActionType.HOLD
                candidate_exposure = current_position_exposure
                reasons.append("NEUTRAL_SIGNAL_HOLD_SHORT")

        else:
            # Currently flat portfolio
            if combined_score > 0.15 and long_expected_utility >= min_net_edge:
                candidate_action = ActionType.BUY
                candidate_exposure = self.config.target_exposure
                reasons.append("SIGNAL_BULLISH_NET_EDGE_EXCEEDS_FRICTION_BUY")
            elif combined_score < -0.15:
                can_short = self.config.allow_short or self.risk_gate.config.allow_short
                if can_short and short_expected_utility >= min_net_edge:
                    candidate_action = ActionType.SHORT
                    candidate_exposure = -self.config.target_exposure
                    reasons.append("SIGNAL_BEARISH_NET_EDGE_EXCEEDS_FRICTION_SHORT")
                elif can_short:
                    candidate_action = ActionType.NO_TRADE
                    candidate_exposure = 0.0
                    reasons.append("BEARISH_EDGE_ERODED_BY_FRICTION_ABSTAIN_NO_TRADE")
                else:
                    candidate_action = ActionType.NO_TRADE
                    candidate_exposure = 0.0
                    reasons.append("SIGNAL_BEARISH_SHORTING_PROHIBITED_NO_TRADE")
            else:
                if long_expected_utility < min_net_edge and combined_score > 0.15:
                    reasons.append("POSITIVE_GROSS_EDGE_ERODED_BY_FRICTION_ABSTAIN")
                candidate_action = ActionType.NO_TRADE
                candidate_exposure = 0.0
                reasons.append("NO_SUFFICIENT_NET_EDGE_NO_TRADE")

        # 5b. Falling Knife Trend Filter (Culprit 1: Never buy falling knives below EMA50)
        if self.config.enable_trend_filter and candidate_action == ActionType.BUY:
            if len(bars) >= self.config.trend_filter_ema_period:
                from nova.features.moving_averages import calculate_ema_series
                closes = [b.close for b in bars]
                ema_series = calculate_ema_series(closes, self.config.trend_filter_ema_period)
                ema_val = ema_series[-1]
                if latest_bar.close < ema_val:
                    reasons.append(
                        f"FALLING_KNIFE_VETO_CLOSE_{latest_bar.close:.2f}_BELOW_EMA{self.config.trend_filter_ema_period}_{ema_val:.2f}"
                    )
                    if is_flat:
                        candidate_action = ActionType.NO_TRADE
                        candidate_exposure = 0.0
                    elif is_long:
                        candidate_action = ActionType.HOLD
                        candidate_exposure = current_position_exposure

        # 6. Deterministic Risk Gate Evaluation (with Execution Safety)
        has_pending_exit = portfolio_state.has_pending_exit_order(ticker) if portfolio_state is not None else False
        has_pending_entry = portfolio_state.has_pending_entry_order(ticker) if portfolio_state is not None else False
        pending_exit_qty = active_pos.pending_exit_quantity if active_pos is not None else 0.0

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
            has_pending_exit_order=has_pending_exit,
            has_pending_entry_order=has_pending_entry,
            pending_exit_quantity=pending_exit_qty,
            ticker=ticker,
            market_breadth=market_breadth,
            market_regime=regime_type_str,
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

        is_short_selected = final_action in (ActionType.SHORT, ActionType.COVER) or (
            final_action == ActionType.HOLD and is_short
        )
        rep_gross = short_expected_gross if is_short_selected else long_expected_gross
        rep_net = short_expected_net if is_short_selected else long_expected_net
        rep_prob = short_prob_clearing if is_short_selected else long_prob_clearing

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
            expected_gross_return=rep_gross,
            estimated_all_in_cost=all_in_cost,
            expected_net_return=rep_net,
            probability_of_clearing_costs=rep_prob,
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

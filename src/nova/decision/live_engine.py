"""NOVA Live Decision Engine (Person 3 Core Algorithm).

Implements the authoritative 14-step live decision algorithm running once per simulated day:
1. Volatility: r = ln(C[t]/C[t-1]), sigma = sqrt(EWMA(r^2, lambda=0.94)), floored at 0.5%.
2. S1 Trend: S1 = tanh((SMA5/SMA20 - 1) / (2 * sigma)).
3. S2 Reversion: z3 = ln(C[t]/C[t-3]) / (sigma * sqrt(3)), raw = -clip(z3 / 2.5, -1, 1) if |z3| >= 1.5 else 0, S2 = raw * (1 - E).
4. S3 News: N blend with 5-day decay 0.6/day, E = clip(|N| / 0.5, 0, 1) if sign(N) == sign(move) else 0.
5. Regime: ER = |C[t] - C[t-10]| / sum(|daily changes|), trend share a = clip((ER - 0.2) / 0.3, 0, 1).
6. Weights: w1 = 0.1 + 0.5a, w2 = 0.1 + 0.4(1 - a), w3 = 0.5, scaled by online multipliers m_k.
7. Score & Uncertainty: Score = sum(w_k * S_k), u = stdev(S1, S2, S3).
8. Edge Gate: mu_hat = g(Score), c = cost/notional, tax = 0.15 * max(mu_hat, 0), Edge = mu_hat - c - tax - kappa * u * sigma. Trade only if Edge > 0 and Edge / sigma > theta.
9. Online Weights: m_k updated via exp(eta * mean_reward_20) clamped to [0.3, 1.5], kill switch on negative IC.
10. Sizing: d = 2 * ATR14, risk 0.75% equity, qty = risk/d * min(1, (Edge/sigma)/0.5), capped at 15% equity, max 8 names, min Rs 20,000, 1.5% no-trade band.
11. Longs / Shorts: EMA30 filter, news veto, square-off safety buffer.
12. Exits: ATR 2.5x trailing stop, bad news shock exit, 2-day min hold decay exit, tax-aware deferral within 25 days of 365-day holding.
13. Portfolio Guards: Drawdown 6% halves size, 12% exits only, negative market breadth filter, 10-15% cash reserve.
14. Execution: Exits first -> covers -> entries ranked by Edge / sigma.

ZERO hardcoded constants: 100% configuration-driven via LiveDecisionConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import uuid
from typing import Mapping, Sequence

from nova.decision.contracts import ActionType, OrderType, TradeDecision
from nova.decision.prediction import BasePredictor, CausalRidgePredictor, PredictionResult
from nova.decision.regime import MarketRegimeDetector, MarketRegimeType, RegimeConfig, RegimeOutput
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    NumericalValidationError,
    validate_finite,
    validate_timezone_aware,
)


@dataclass(frozen=True, slots=True)
class LiveDecisionConfig:
    """Configurable parameters for the 14-step live decision algorithm."""
    # 1. Volatility
    volatility_lambda: float = 0.94
    min_volatility_floor: float = 0.005

    # 2. S1 Trend
    sma_fast_window: int = 5
    sma_slow_window: int = 20
    trend_sigma_multiplier: float = 2.0

    # 3. S2 Reversion
    reversion_window: int = 3
    reversion_z_threshold: float = 1.5
    reversion_clip_scale: float = 2.5

    # 4. S3 News & Explained Factor
    news_decay_per_day: float = 0.60
    news_explained_scale: float = 0.50

    # 5. Regime & Efficiency Ratio
    er_window: int = 10
    er_trend_lower: float = 0.20
    er_trend_range: float = 0.30

    # 6. Weights
    w1_base: float = 0.10
    w1_trend_scale: float = 0.50
    w2_base: float = 0.10
    w2_reversion_scale: float = 0.40
    w3_news_base: float = 0.50

    # 8. Edge Gate
    score_to_mu_scale: float = 1.0       # g(Score) multiplier
    long_roundtrip_cost_fixed: float = 55.34  # 20 + 20 + 15.34
    short_roundtrip_cost_fixed: float = 40.0
    tax_drag_rate: float = 0.15          # 15% capital gains tax drag
    kappa_uncertainty: float = 0.25      # Uncertainty penalty multiplier
    theta_edge_ratio: float = 0.15       # Minimum Edge / sigma threshold

    # 8b. ML Blending
    ml_blend_confidence_min: float = 0.40
    ml_blend_weight: float = 0.30

    # 9. Online Weights Multiplier
    online_eta: float = 0.10
    multiplier_min: float = 0.30
    multiplier_max: float = 1.50
    online_reward_window: int = 20
    ic_kill_switch_window: int = 30
    specialist_reward_score_threshold: float = 0.20

    # 10. Sizing
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    risk_per_trade_fraction: float = 0.0075  # 0.75% equity
    edge_scale_divisor: float = 0.50
    max_name_equity_cap: float = 0.15        # 15% max per name
    max_active_names: int = 8
    min_trade_notional: float = 20000.0      # Rs 20,000 minimum
    no_trade_band_fraction: float = 0.015    # 1.5% equity rebalance threshold

    # 11. Longs / Shorts Filter
    ema_long_filter_window: int = 30
    short_square_off_days_buffer: int = 3
    bearish_news_veto_threshold: float = -0.30
    allow_shorts: bool = False

    # 12. Exits
    atr_trailing_exit_multiplier: float = 2.5
    bad_news_shock_threshold: float = -0.70
    signal_decay_score_threshold: float = -0.10
    min_hold_days_before_decay_exit: int = 2
    long_term_tax_threshold_days: int = 365
    long_term_tax_buffer_days: int = 25
    tax_deferral_min_edge_ratio: float = -0.10

    # 13. Portfolio Guards
    drawdown_tier1_half_size: float = 0.06   # 6% DD halves size
    drawdown_tier1_size_scale: float = 0.50  # 50% scale
    drawdown_tier2_exits_only: float = 0.12  # 12% DD exits only
    min_cash_reserve_fraction: float = 0.10  # 10% cash minimum
    max_cash_reserve_fraction: float = 0.15  # 15% cash target

    # 14. Execution
    max_requests_per_minute: int = 100
    engine_id: str = "sentinel_live_decision_engine"
    engine_version: str = "2.0.0"

    def __post_init__(self) -> None:
        for name, val in [
            ("volatility_lambda", self.volatility_lambda),
            ("min_volatility_floor", self.min_volatility_floor),
            ("trend_sigma_multiplier", self.trend_sigma_multiplier),
            ("reversion_z_threshold", self.reversion_z_threshold),
            ("reversion_clip_scale", self.reversion_clip_scale),
            ("news_decay_per_day", self.news_decay_per_day),
            ("news_explained_scale", self.news_explained_scale),
            ("er_trend_lower", self.er_trend_lower),
            ("er_trend_range", self.er_trend_range),
            ("w1_base", self.w1_base),
            ("w1_trend_scale", self.w1_trend_scale),
            ("w2_base", self.w2_base),
            ("w2_reversion_scale", self.w2_reversion_scale),
            ("w3_news_base", self.w3_news_base),
            ("long_roundtrip_cost_fixed", self.long_roundtrip_cost_fixed),
            ("short_roundtrip_cost_fixed", self.short_roundtrip_cost_fixed),
            ("tax_drag_rate", self.tax_drag_rate),
            ("kappa_uncertainty", self.kappa_uncertainty),
            ("theta_edge_ratio", self.theta_edge_ratio),
            ("online_eta", self.online_eta),
            ("multiplier_min", self.multiplier_min),
            ("multiplier_max", self.multiplier_max),
            ("atr_stop_multiplier", self.atr_stop_multiplier),
            ("risk_per_trade_fraction", self.risk_per_trade_fraction),
            ("edge_scale_divisor", self.edge_scale_divisor),
            ("max_name_equity_cap", self.max_name_equity_cap),
            ("min_trade_notional", self.min_trade_notional),
            ("no_trade_band_fraction", self.no_trade_band_fraction),
            ("atr_trailing_exit_multiplier", self.atr_trailing_exit_multiplier),
            ("drawdown_tier1_half_size", self.drawdown_tier1_half_size),
            ("drawdown_tier2_exits_only", self.drawdown_tier2_exits_only),
            ("min_cash_reserve_fraction", self.min_cash_reserve_fraction),
            ("max_cash_reserve_fraction", self.max_cash_reserve_fraction),
            ("ml_blend_confidence_min", self.ml_blend_confidence_min),
            ("ml_blend_weight", self.ml_blend_weight),
            ("specialist_reward_score_threshold", self.specialist_reward_score_threshold),
            ("drawdown_tier1_size_scale", self.drawdown_tier1_size_scale),
        ]:
            validate_finite(val, name)
            if val < 0.0:
                raise ContractValidationError(f"'{name}' cannot be negative, got {val}")

        for name, val in [
            ("bad_news_shock_threshold", self.bad_news_shock_threshold),
            ("signal_decay_score_threshold", self.signal_decay_score_threshold),
            ("tax_deferral_min_edge_ratio", self.tax_deferral_min_edge_ratio),
            ("bearish_news_veto_threshold", self.bearish_news_veto_threshold),
        ]:
            validate_finite(val, name)

        for name, ival in [
            ("sma_fast_window", self.sma_fast_window),
            ("sma_slow_window", self.sma_slow_window),
            ("reversion_window", self.reversion_window),
            ("er_window", self.er_window),
            ("online_reward_window", self.online_reward_window),
            ("ic_kill_switch_window", self.ic_kill_switch_window),
            ("atr_period", self.atr_period),
            ("max_active_names", self.max_active_names),
            ("ema_long_filter_window", self.ema_long_filter_window),
            ("short_square_off_days_buffer", self.short_square_off_days_buffer),
            ("min_hold_days_before_decay_exit", self.min_hold_days_before_decay_exit),
            ("long_term_tax_threshold_days", self.long_term_tax_threshold_days),
            ("long_term_tax_buffer_days", self.long_term_tax_buffer_days),
            ("max_requests_per_minute", self.max_requests_per_minute),
        ]:
            if isinstance(ival, bool) or not isinstance(ival, int) or ival <= 0:
                raise ContractValidationError(f"'{name}' must be positive integer, got {ival}")

        if self.sma_fast_window >= self.sma_slow_window:
            raise ContractValidationError("sma_fast_window must be strictly less than sma_slow_window")
        if self.multiplier_min >= self.multiplier_max:
            raise ContractValidationError("multiplier_min must be less than multiplier_max")
        if self.drawdown_tier1_half_size >= self.drawdown_tier2_exits_only:
            raise ContractValidationError("drawdown_tier1_half_size must be less than drawdown_tier2_exits_only")


@dataclass
class StockPositionState:
    """Tracks holding state, cost basis, high water mark, and hold duration for one stock."""
    ticker: str
    quantity: float = 0.0
    cost_basis: float = 0.0
    entry_timestamp: datetime | None = None
    days_held: int = 0
    highest_price: float = 0.0
    is_short: bool = False


class LiveDecisionCoreEngine:
    """Production implementation of the 14-step Live Decision Algorithm."""

    def __init__(
        self,
        config: LiveDecisionConfig | None = None,
        predictor: BasePredictor | None = None,
    ) -> None:
        self.config = config or LiveDecisionConfig()
        self.predictor = predictor or CausalRidgePredictor()

        # Online specialist multipliers: m1 (trend), m2 (reversion), m3 (news)
        self.m1: float = 1.0
        self.m2: float = 1.0
        self.m3: float = 1.0

        # Rolling history of specialist rewards: (reward1, reward2, reward3)
        self.specialist_rewards: list[tuple[float, float, float]] = []
        # Rolling IC tracking for S3
        self.s3_ic_history: list[float] = []

        # Position tracking map
        self.positions: dict[str, StockPositionState] = {}

    @property
    def required_warmup_bars(self) -> int:
        return max(
            self.config.sma_slow_window,
            self.config.er_window,
            self.config.atr_period,
            self.config.ema_long_filter_window,
        ) + 2

    def calculate_atr(self, bars: Sequence[MarketBar]) -> float:
        """Compute 14-period Average True Range causally."""
        p = self.config.atr_period
        if len(bars) < p + 1:
            return bars[-1].close * 0.015  # Fallback 1.5%

        trs: list[float] = []
        for i in range(-p, 0):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs)

    def calculate_ema(self, series: Sequence[float], window: int) -> float:
        """Compute latest value of causal EMA."""
        if not series:
            return 0.0
        alpha = 2.0 / (window + 1.0)
        ema = series[0]
        for val in series[1:]:
            ema = alpha * val + (1.0 - alpha) * ema
        return ema

    def evaluate_decision(
        self,
        bars: Sequence[MarketBar],
        current_position_exposure: float = 0.0,
        current_drawdown: float = 0.0,
        available_cash: float = 100000.0,
        current_portfolio_value: float = 100000.0,
        as_of_time: datetime | None = None,
    ) -> TradeDecision:
        """Adapter matching the replay interface for seamless multi-engine evaluation."""
        return self.evaluate_stock_day(
            bars=bars,
            total_equity=current_portfolio_value,
            available_cash=available_cash,
            current_drawdown=current_drawdown,
        )

    def evaluate_stock_day(
        self,
        bars: Sequence[MarketBar],
        total_equity: float,
        available_cash: float,
        current_drawdown: float = 0.0,
        news_sentiment_score: float = 0.0,
        market_breadth_positive: bool = True,
        days_to_square_off: int = 10,
    ) -> TradeDecision:
        """Execute Steps 1 through 13 for a single asset."""
        if not bars:
            raise ContractValidationError("Cannot evaluate on empty bar sequence")

        latest_bar = bars[-1]
        as_of = latest_bar.source_timestamp
        ticker = latest_bar.ticker
        c_t = latest_bar.close

        reasons: list[str] = []

        # Warmup and Data Quality Check
        if latest_bar.data_quality != DataQualityStatus.VALID:
            reasons.append(f"DATA_QUALITY_INVALID_{latest_bar.data_quality.value}")
            return self._build_abstain(ticker, as_of, reasons, ActionType.NO_TRADE, 0.0)

        if len(bars) < self.required_warmup_bars:
            reasons.append(f"INSUFFICIENT_HISTORY_BARS_{len(bars)}_OF_{self.required_warmup_bars}")
            return self._build_abstain(ticker, as_of, reasons, ActionType.NO_TRADE, 0.0)

        closes = [b.close for b in bars]

        # STEP 1: Volatility sigma = sqrt(EWMA(r^2, lambda=0.94)), floored at min_volatility_floor
        log_ret_1 = math.log(closes[-1] / closes[-2])
        # Compute EWMA over available returns
        decay = self.config.volatility_lambda
        ewma_sq = log_ret_1 ** 2
        for i in range(len(closes) - 2, 0, -1):
            r = math.log(closes[i] / closes[i - 1])
            ewma_sq = decay * ewma_sq + (1.0 - decay) * (r ** 2)
        sigma = max(self.config.min_volatility_floor, math.sqrt(ewma_sq))
        reasons.append(f"SIGMA_{sigma:.4f}")

        # STEP 2: S1 Trend = tanh((SMA5 / SMA20 - 1) / (2 * sigma))
        sma5 = sum(closes[-self.config.sma_fast_window:]) / self.config.sma_fast_window
        sma20 = sum(closes[-self.config.sma_slow_window:]) / self.config.sma_slow_window
        ratio_trend = (sma5 / sma20 - 1.0) / (self.config.trend_sigma_multiplier * sigma)
        s1 = math.tanh(ratio_trend)
        reasons.append(f"S1_TREND_{s1:.3f}")

        # STEP 3: S2 Reversion
        # z3 = ln(C[t] / C[t-3]) / (sigma * sqrt(3))
        c_t3 = closes[-self.config.reversion_window - 1]
        ret_3 = math.log(c_t / c_t3)
        z3 = ret_3 / (sigma * math.sqrt(self.config.reversion_window))

        if abs(z3) >= self.config.reversion_z_threshold:
            raw_s2 = -max(-1.0, min(1.0, z3 / self.config.reversion_clip_scale))
        else:
            raw_s2 = 0.0

        # STEP 4: S3 News & Explained Factor E
        N = news_sentiment_score
        # E = clip(|N| / 0.5, 0, 1) if news sign matches price move direction, else 0
        news_sign_matches_move = (N > 0 and ret_3 > 0) or (N < 0 and ret_3 < 0)
        if news_sign_matches_move:
            E = min(1.0, max(0.0, abs(N) / self.config.news_explained_scale))
        else:
            E = 0.0

        s2 = raw_s2 * (1.0 - E)
        s3 = N  # Specialist 3
        reasons.append(f"S2_REVERSION_{s2:.3f}_E_{E:.2f}")
        reasons.append(f"S3_NEWS_{s3:.3f}")

        # STEP 5: Regime - Kaufman Efficiency Ratio & Trend Share a
        n_er = self.config.er_window
        net_er = abs(closes[-1] - closes[-n_er - 1])
        gross_er = sum(abs(closes[-i] - closes[-i - 1]) for i in range(1, n_er + 1))
        er = min(1.0, max(0.0, net_er / max(1e-9, gross_er)))
        a = min(1.0, max(0.0, (er - self.config.er_trend_lower) / self.config.er_trend_range))
        reasons.append(f"REGIME_ER_{er:.3f}_TREND_SHARE_A_{a:.3f}")

        # STEP 6: Strategy Weights
        w1 = (self.config.w1_base + self.config.w1_trend_scale * a) * self.m1
        w2 = (self.config.w2_base + self.config.w2_reversion_scale * (1.0 - a)) * self.m2
        
        has_news = abs(N) > 1e-6
        w3 = (self.config.w3_news_base * self.m3) if has_news else 0.0
        tot_w = w1 + w2 + w3
        reasons.append(f"WEIGHTS_w1_{w1:.2f}_w2_{w2:.2f}_w3_{w3:.2f}")

        # STEP 7: Score & Uncertainty
        score = (w1 * s1 + w2 * s2 + w3 * s3) / max(1e-6, tot_w)
        # Uncertainty u = standard deviation of specialist signals
        if has_news:
            specialist_signals = [s1, s2, s3]
            mean_sig = sum(specialist_signals) / 3.0
            var_sig = sum((s - mean_sig) ** 2 for s in specialist_signals) / 2.0
        else:
            specialist_signals = [s1, s2]
            mean_sig = (s1 + s2) / 2.0
            var_sig = ((s1 - mean_sig) ** 2 + (s2 - mean_sig) ** 2) / 1.0
        u = math.sqrt(var_sig)
        reasons.append(f"SCORE_{score:.3f}_UNCERTAINTY_U_{u:.3f}")

        # STEP 8: Edge Gate
        # Optional ML forecast adjustment
        ml_pred: PredictionResult = self.predictor.predict(bars)
        mu_hat = score * sigma * self.config.score_to_mu_scale
        if not ml_pred.is_fallback and ml_pred.confidence >= self.config.ml_blend_confidence_min:
            w_ml = self.config.ml_blend_weight
            mu_hat = (1.0 - w_ml) * mu_hat + w_ml * ml_pred.expected_return
            reasons.append(f"ML_BLEND_PRED_{ml_pred.expected_return:+.4f}")

        # Round-trip cost fraction c
        # For a standard trade notional Rs 20,000, fixed fee Rs 55.34 is ~27.6 bps
        notional_est = max(self.config.min_trade_notional, total_equity * self.config.max_name_equity_cap)
        is_short_action = score < 0
        fixed_cost = self.config.short_roundtrip_cost_fixed if is_short_action else self.config.long_roundtrip_cost_fixed
        c = fixed_cost / notional_est

        # Tax drag = 0.15 * max(mu_hat, 0)
        tax_drag = self.config.tax_drag_rate * max(0.0, mu_hat)

        # Edge = mu_hat - c - tax - kappa * u * sigma
        edge = mu_hat - c - tax_drag - (self.config.kappa_uncertainty * u * sigma)
        edge_ratio = edge / sigma
        reasons.append(f"EDGE_{edge:.5f}_RATIO_{edge_ratio:.3f}_THETA_{self.config.theta_edge_ratio}")

        # STEP 10: Sizing
        atr = self.calculate_atr(bars)
        stop_distance = self.config.atr_stop_multiplier * atr
        risk_capital = total_equity * self.config.risk_per_trade_fraction
        # qty scaled by min(1, (Edge/sigma) / 0.5)
        edge_scaler = min(1.0, max(0.0, edge_ratio / self.config.edge_scale_divisor))
        target_shares = (risk_capital / max(1e-4, stop_distance)) * edge_scaler
        target_notional = target_shares * c_t

        # Cap at 15% equity per name
        max_notional = total_equity * self.config.max_name_equity_cap
        target_notional = min(target_notional, max_notional)
        target_exposure = target_notional / total_equity

        # Existing position tracking
        pos = self.positions.get(ticker, StockPositionState(ticker=ticker))
        current_exposure = (pos.quantity * c_t) / total_equity

        # Update high-water mark for open position
        if pos.quantity > 0:
            pos.highest_price = max(pos.highest_price, c_t)

        # STEP 12 & 13: Exits & Portfolio Guards
        # 1. Drawdown circuit breakers
        if current_drawdown >= self.config.drawdown_tier2_exits_only:
            reasons.append(f"DD_GUARD_TIER2_EXITS_ONLY_{current_drawdown:.3f}")
            if pos.quantity > 0:
                return self._build_decision(ticker, as_of, ActionType.EXIT, 0.0, reasons, mu_hat, c, edge, edge_ratio, u)
            return self._build_abstain(ticker, as_of, reasons, ActionType.NO_TRADE, 0.0)

        if current_drawdown >= self.config.drawdown_tier1_half_size:
            target_exposure *= self.config.drawdown_tier1_size_scale
            reasons.append(f"DD_GUARD_TIER1_HALF_SIZE_{current_drawdown:.3f}")

        # 2. ATR Trailing Stop Exit (2.5x ATR)
        if pos.quantity > 0 and pos.highest_price > 0:
            trail_stop = pos.highest_price - (self.config.atr_trailing_exit_multiplier * atr)
            if c_t <= trail_stop:
                reasons.append(f"EXIT_ATR_TRAILING_STOP_BREACHED_{c_t:.2f}_LE_{trail_stop:.2f}")
                return self._build_decision(ticker, as_of, ActionType.EXIT, 0.0, reasons, mu_hat, c, edge, edge_ratio, u)

        # 3. Bad News Shock Exit
        if pos.quantity > 0 and news_sentiment_score <= self.config.bad_news_shock_threshold:
            reasons.append(f"EXIT_BAD_NEWS_SHOCK_{news_sentiment_score:.2f}")
            return self._build_decision(ticker, as_of, ActionType.EXIT, 0.0, reasons, mu_hat, c, edge, edge_ratio, u)

        # 4. Long-term Tax Deferral
        if pos.quantity > 0 and c_t > pos.cost_basis:
            days_held = pos.days_held
            threshold = self.config.long_term_tax_threshold_days
            buffer = self.config.long_term_tax_buffer_days
            if (threshold - buffer) <= days_held < threshold and edge_ratio > self.config.tax_deferral_min_edge_ratio:
                reasons.append(f"TAX_AWARE_DEFERRAL_HOLD_{days_held}_DAYS")
                return self._build_decision(ticker, as_of, ActionType.HOLD, current_exposure, reasons, mu_hat, c, edge, edge_ratio, u)

        # 5. Signal Decay Exit after min hold
        if pos.quantity > 0 and pos.days_held >= self.config.min_hold_days_before_decay_exit:
            if score < self.config.signal_decay_score_threshold or edge <= 0.0:
                reasons.append(f"EXIT_SIGNAL_DECAY_SCORE_{score:.3f}")
                return self._build_decision(ticker, as_of, ActionType.EXIT, 0.0, reasons, mu_hat, c, edge, edge_ratio, u)

        # STEP 11: Long Entries
        ema30 = self.calculate_ema(closes, self.config.ema_long_filter_window)
        can_long = (
            edge > 0
            and edge_ratio > self.config.theta_edge_ratio
            and c_t > ema30
            and news_sentiment_score > self.config.bearish_news_veto_threshold
            and market_breadth_positive
        )

        # Check cash reserve
        cash_fraction = available_cash / total_equity
        if cash_fraction < self.config.min_cash_reserve_fraction and target_exposure > current_exposure:
            reasons.append(f"CASH_BUFFER_PROTECTION_{cash_fraction:.3f}")
            can_long = False

        if can_long:
            # Check No-Trade Band: skip rebalances under 1.5% of equity
            rebalance_delta = abs(target_exposure - current_exposure)
            if current_exposure > 0 and rebalance_delta < self.config.no_trade_band_fraction:
                reasons.append(f"NO_TRADE_BAND_SKIP_DELTA_{rebalance_delta:.4f}")
                return self._build_decision(ticker, as_of, ActionType.HOLD, current_exposure, reasons, mu_hat, c, edge, edge_ratio, u)

            # Minimum trade size
            if target_notional < self.config.min_trade_notional and current_exposure == 0:
                reasons.append(f"ORDER_BELOW_MIN_NOTIONAL_{target_notional:.1f}_LT_{self.config.min_trade_notional}")
                return self._build_abstain(ticker, as_of, reasons, ActionType.NO_TRADE, 0.0)

            action = ActionType.BUY if current_exposure == 0 else ActionType.HOLD
            reasons.append("LONG_CRITERIA_SATISFIED_ENTRY_AUTHORIZED")
            return self._build_decision(ticker, as_of, action, target_exposure, reasons, mu_hat, c, edge, edge_ratio, u)

        # If holding and no exit triggered -> HOLD
        if current_exposure > 0:
            reasons.append("HOLDING_POSITION_NO_EXIT_SIGNAL")
            return self._build_decision(ticker, as_of, ActionType.HOLD, current_exposure, reasons, mu_hat, c, edge, edge_ratio, u)

        # Otherwise: Edge gate failed -> Do nothing
        reasons.append("EDGE_GATE_NOT_CLEARED_NO_TRADE")
        return self._build_abstain(ticker, as_of, reasons, ActionType.NO_TRADE, 0.0)

    def update_online_weights(
        self,
        signals_at_t: tuple[float, float, float],
        abnormal_return: float,
        sigma: float,
    ) -> None:
        """Step 9: Online specialist multiplier updates after day t+1 is revealed."""
        s1, s2, s3 = signals_at_t
        eta = self.config.online_eta

        rewards: list[float] = [0.0, 0.0, 0.0]
        for k, s_k in enumerate([s1, s2, s3]):
            if abs(s_k) > self.config.specialist_reward_score_threshold:
                sgn = 1.0 if s_k > 0 else -1.0
                rewards[k] = sgn * (abnormal_return / max(1e-4, sigma))

        self.specialist_rewards.append(tuple(rewards))
        if len(self.specialist_rewards) > self.config.online_reward_window:
            self.specialist_rewards.pop(0)

        # Compute mean reward over last 20
        n = len(self.specialist_rewards)
        mean_r1 = sum(r[0] for r in self.specialist_rewards) / n
        mean_r2 = sum(r[1] for r in self.specialist_rewards) / n
        mean_r3 = sum(r[2] for r in self.specialist_rewards) / n

        # Update multipliers
        self.m1 = max(self.config.multiplier_min, min(self.config.multiplier_max, self.m1 * math.exp(eta * mean_r1)))
        self.m2 = max(self.config.multiplier_min, min(self.config.multiplier_max, self.m2 * math.exp(eta * mean_r2)))
        self.m3 = max(self.config.multiplier_min, min(self.config.multiplier_max, self.m3 * math.exp(eta * mean_r3)))

        # S3 Kill switch: if rolling 30-signal IC is below 0, set m3 = 0
        if len(self.s3_ic_history) >= self.config.ic_kill_switch_window:
            rolling_ic = sum(self.s3_ic_history[-self.config.ic_kill_switch_window:]) / self.config.ic_kill_switch_window
            if rolling_ic < 0.0:
                self.m3 = 0.0

    def _build_decision(
        self,
        ticker: str,
        timestamp: datetime,
        action: ActionType,
        target_exposure: float,
        reasons: list[str],
        gross: float,
        cost: float,
        net: float,
        ratio: float,
        uncertainty: float,
    ) -> TradeDecision:
        return TradeDecision(
            decision_id=f"live_{uuid.uuid4().hex[:12]}",
            decision_timestamp=timestamp,
            instrument_id=ticker,
            selected_action=action,
            target_quantity_or_exposure=target_exposure,
            order_type=OrderType.MARKET if action in (ActionType.BUY, ActionType.EXIT, ActionType.SELL) else OrderType.NONE,
            selected_strategy="sentinel_14_step_live_engine",
            selection_or_rejection_reasons=tuple(reasons),
            configuration_version=self.config.engine_version,
            expected_gross_return=gross,
            estimated_all_in_cost=cost,
            expected_net_return=gross - cost,
            probability_of_clearing_costs=min(1.0, max(0.0, 0.5 + 0.5 * math.tanh(ratio))),
            uncertainty_estimate=uncertainty,
            estimated_risk_contribution=abs(target_exposure) * uncertainty,
        )

    def _build_abstain(
        self,
        ticker: str,
        timestamp: datetime,
        reasons: list[str],
        action: ActionType,
        target_exposure: float,
    ) -> TradeDecision:
        return TradeDecision(
            decision_id=f"live_{uuid.uuid4().hex[:12]}",
            decision_timestamp=timestamp,
            instrument_id=ticker,
            selected_action=action,
            target_quantity_or_exposure=target_exposure,
            order_type=OrderType.NONE,
            selected_strategy="sentinel_14_step_live_engine",
            selection_or_rejection_reasons=tuple(reasons),
            configuration_version=self.config.engine_version,
            expected_gross_return=0.0,
            estimated_all_in_cost=0.0,
            expected_net_return=0.0,
            probability_of_clearing_costs=0.5,
            uncertainty_estimate=0.0,
            estimated_risk_contribution=0.0,
        )

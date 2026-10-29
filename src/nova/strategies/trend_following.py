"""NOVA Trend-Following Strategy (Person 3 Exclusive Domain).

Uses fast/slow moving-average relationship combined with price momentum (Rate of Change)
to identify persistent directional trends with zero look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from nova.features.moving_averages import calculate_ema_series, validate_window
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    validate_finite,
)
from nova.strategies.base import BaseStrategy, StrategySignal


@dataclass(frozen=True, slots=True)
class TrendFollowingConfig:
    """Configuration for trend-following strategy."""
    fast_window: int = 12
    slow_window: int = 26
    momentum_window: int = 9
    momentum_threshold: float = 0.001   # 10 bps minimum momentum
    strategy_id: str = "trend_following_v1"

    def __post_init__(self) -> None:
        validate_window(self.fast_window, "fast_window")
        validate_window(self.slow_window, "slow_window")
        validate_window(self.momentum_window, "momentum_window")
        if self.fast_window >= self.slow_window:
            raise ContractValidationError(
                f"fast_window ({self.fast_window}) must be strictly less than slow_window ({self.slow_window})"
            )
        validate_finite(self.momentum_threshold, "momentum_threshold")


class TrendFollowingStrategy(BaseStrategy):
    """Causal trend-following strategy using EMA crossover and price momentum."""

    def __init__(self, config: TrendFollowingConfig | None = None) -> None:
        self.config = config or TrendFollowingConfig()

    @property
    def required_warmup_bars(self) -> int:
        return max(self.config.slow_window, self.config.momentum_window) + 2

    def evaluate(self, bars: Sequence[MarketBar]) -> StrategySignal:
        if not bars:
            raise ContractValidationError("Cannot evaluate on empty bar sequence")

        latest_bar = bars[-1]
        as_of = latest_bar.source_timestamp

        # Check data quality
        if latest_bar.data_quality != DataQualityStatus.VALID:
            return StrategySignal(
                strategy_id=self.config.strategy_id,
                direction=SignalDirection.NEUTRAL,
                raw_score=0.0,
                confidence=0.0,
                timestamp=as_of,
                expected_gross_return=0.0,
                volatility_estimate=None,
                reason_codes=("DATA_QUALITY_NOT_VALID", f"STATUS_{latest_bar.data_quality.value}"),
            )

        # Warmup check
        if len(bars) < self.required_warmup_bars:
            return StrategySignal(
                strategy_id=self.config.strategy_id,
                direction=SignalDirection.NEUTRAL,
                raw_score=0.0,
                confidence=0.0,
                timestamp=as_of,
                expected_gross_return=0.0,
                volatility_estimate=None,
                reason_codes=(
                    "INSUFFICIENT_HISTORY",
                    f"BARS_{len(bars)}_OF_REQUIRED_{self.required_warmup_bars}",
                ),
            )

        closes = [b.close for b in bars]
        fast_emas = calculate_ema_series(closes, self.config.fast_window)
        slow_emas = calculate_ema_series(closes, self.config.slow_window)

        curr_fast = fast_emas[-1]
        curr_slow = slow_emas[-1]
        prev_fast = fast_emas[-2]
        prev_slow = slow_emas[-2]

        # Momentum calculation: (Close_t - Close_{t-m}) / Close_{t-m}
        m_idx = -(self.config.momentum_window + 1)
        momentum = (closes[-1] - closes[m_idx]) / closes[m_idx]

        # Rolling volatility estimate (standard deviation of log returns over slow_window)
        recent_closes = closes[-self.config.slow_window:]
        returns = [math.log(recent_closes[i] / recent_closes[i - 1]) for i in range(1, len(recent_closes))]
        mean_ret = sum(returns) / len(returns)
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, len(returns) - 1)
        volatility = math.sqrt(var_ret)

        # Crossover & trend checks
        bullish_crossover = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_crossover = prev_fast >= prev_slow and curr_fast < curr_slow
        trend_spread = (curr_fast - curr_slow) / curr_slow

        reasons: list[str] = []

        if curr_fast > curr_slow:
            # Uptrend
            if momentum > self.config.momentum_threshold:
                direction = SignalDirection.BULLISH
                # Score combines spread and momentum, scaled between 0 and 1
                raw_score = min(1.0, max(0.2, (trend_spread / max(1e-4, volatility)) * 0.5 + momentum * 10))
                confidence = min(0.95, 0.5 + abs(trend_spread) * 10)
                expected_gross = max(0.002, abs(momentum) * 0.5)
                reasons.append("BULLISH_TREND_AND_MOMENTUM_CONFIRMED")
                if bullish_crossover:
                    reasons.append("FRESH_BULLISH_EMA_CROSSOVER")
            else:
                # Uptrend with decelerating momentum
                direction = SignalDirection.NEUTRAL
                raw_score = 0.1
                confidence = 0.3
                expected_gross = 0.0005
                reasons.append("UPTREND_PRESENT_BUT_MOMENTUM_WEAK")
        else:
            # Downtrend
            if momentum < -self.config.momentum_threshold:
                direction = SignalDirection.BEARISH
                raw_score = max(-1.0, min(-0.2, (trend_spread / max(1e-4, volatility)) * 0.5 + momentum * 10))
                confidence = min(0.95, 0.5 + abs(trend_spread) * 10)
                expected_gross = -max(0.002, abs(momentum) * 0.5)
                reasons.append("BEARISH_TREND_AND_MOMENTUM_CONFIRMED")
                if bearish_crossover:
                    reasons.append("FRESH_BEARISH_EMA_CROSSOVER")
            else:
                direction = SignalDirection.NEUTRAL
                raw_score = -0.1
                confidence = 0.3
                expected_gross = -0.0005
                reasons.append("DOWNTREND_PRESENT_BUT_MOMENTUM_WANING")

        return StrategySignal(
            strategy_id=self.config.strategy_id,
            direction=direction,
            raw_score=raw_score,
            confidence=confidence,
            timestamp=as_of,
            expected_gross_return=expected_gross,
            volatility_estimate=volatility,
            reason_codes=tuple(reasons),
        )

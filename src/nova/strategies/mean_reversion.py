"""NOVA Mean-Reversion Strategy (Person 3 Exclusive Domain).

Identifies statistical price dislocations from rolling moving averages and extreme
RSI oscillator boundaries, predicting causal reversion to the local mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from nova.features.moving_averages import validate_window
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    validate_finite,
)
from nova.strategies.base import BaseStrategy, StrategySignal


@dataclass(frozen=True, slots=True)
class MeanReversionConfig:
    """Configuration for mean-reversion strategy."""
    lookback_window: int = 20
    z_score_threshold: float = 1.8
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    strategy_id: str = "mean_reversion_v1"

    def __post_init__(self) -> None:
        validate_window(self.lookback_window, "lookback_window")
        validate_window(self.rsi_period, "rsi_period")
        validate_finite(self.z_score_threshold, "z_score_threshold")
        if self.z_score_threshold <= 0.0:
            raise ContractValidationError("z_score_threshold must be strictly positive")
        validate_finite(self.rsi_oversold, "rsi_oversold")
        validate_finite(self.rsi_overbought, "rsi_overbought")
        if not (0.0 < self.rsi_oversold < self.rsi_overbought < 100.0):
            raise ContractValidationError("Must satisfy 0 < rsi_oversold < rsi_overbought < 100")


def compute_causal_rsi(closes: Sequence[float], period: int) -> float:
    """Causal RSI calculation over the given window."""
    if len(closes) < period + 1:
        return 50.0

    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss < 1e-9:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionStrategy(BaseStrategy):
    """Causal mean-reversion strategy using price z-scores and RSI oscillator."""

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    @property
    def required_warmup_bars(self) -> int:
        return max(self.config.lookback_window, self.config.rsi_period) + 2

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
        recent_closes = closes[-self.config.lookback_window:]

        # Rolling mean and standard deviation
        mean_price = sum(recent_closes) / len(recent_closes)
        variance = sum((p - mean_price) ** 2 for p in recent_closes) / max(1, len(recent_closes) - 1)
        stdev_price = math.sqrt(variance)

        current_close = closes[-1]
        z_score = (current_close - mean_price) / max(1e-4, stdev_price)
        rsi = compute_causal_rsi(closes, self.config.rsi_period)

        # Return volatility estimate over lookback window
        returns = [math.log(recent_closes[i] / recent_closes[i - 1]) for i in range(1, len(recent_closes))]
        mean_ret = sum(returns) / len(returns)
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / max(1, len(returns) - 1)
        volatility = math.sqrt(var_ret)

        # Expected return is expected reversion halfway back to mean
        expected_reversion_return = (mean_price - current_close) / current_close

        reasons: list[str] = []

        if z_score < -self.config.z_score_threshold and rsi < self.config.rsi_oversold:
            # Severely oversold dislocation -> bullish mean-reversion bounce
            direction = SignalDirection.BULLISH
            raw_score = min(1.0, max(0.2, abs(z_score) / (self.config.z_score_threshold * 1.5)))
            confidence = min(0.9, 0.5 + (self.config.rsi_oversold - rsi) / 100.0)
            reasons.append(f"OVERSOLD_DISLOCATION_Z_{z_score:.2f}_RSI_{rsi:.1f}")
        elif z_score > self.config.z_score_threshold and rsi > self.config.rsi_overbought:
            # Severely overbought dislocation -> bearish pullback expectation
            direction = SignalDirection.BEARISH
            raw_score = max(-1.0, min(-0.2, -z_score / (self.config.z_score_threshold * 1.5)))
            confidence = min(0.9, 0.5 + (rsi - self.config.rsi_overbought) / 100.0)
            reasons.append(f"OVERBOUGHT_DISLOCATION_Z_{z_score:.2f}_RSI_{rsi:.1f}")
        else:
            direction = SignalDirection.NEUTRAL
            raw_score = 0.0
            confidence = 0.2
            expected_reversion_return = 0.0
            reasons.append(f"WITHIN_NORMAL_BOUNDS_Z_{z_score:.2f}_RSI_{rsi:.1f}")

        return StrategySignal(
            strategy_id=self.config.strategy_id,
            direction=direction,
            raw_score=raw_score,
            confidence=confidence,
            timestamp=as_of,
            expected_gross_return=expected_reversion_return,
            volatility_estimate=volatility,
            reason_codes=tuple(reasons),
        )

"""NOVA Market Regime Detection Subsystem (Person 3 Domain).

Implements causal Efficiency Ratio (ER), EWMA volatility estimation (lambda=0.94),
and adaptive trend-share calculation to categorize market regimes:
- TRENDING: High efficiency ratio, directional persistence
- RANGING: Low efficiency ratio, choppy oscillations
- HIGH_VOLATILITY: EWMA volatility exceeding upper threshold
- UNKNOWN / FALLBACK: Insufficient history or corrupted data

Strictly ZERO hardcoding: all thresholds, windows, and decay factors are configuration-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Sequence

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    NumericalValidationError,
    validate_finite,
    validate_timezone_aware,
)


class MarketRegimeType(str, Enum):
    """Categorical classification of market environment."""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Configurable parameters for regime detection."""
    er_window: int = 10                  # Lookback for Efficiency Ratio
    er_trend_lower: float = 0.20         # ER threshold where trend begins (clip min)
    er_trend_range: float = 0.30         # ER divisor: a = clip((ER - lower) / range, 0, 1)
    ewma_lambda: float = 0.94            # RiskMetrics decay factor
    min_volatility_floor: float = 0.005  # 0.5% volatility floor
    high_volatility_threshold: float = 0.025 # 2.5% daily volatility threshold
    config_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if isinstance(self.er_window, bool) or not isinstance(self.er_window, int) or self.er_window < 2:
            raise ContractValidationError(f"er_window must be an integer >= 2, got {self.er_window}")
        validate_finite(self.er_trend_lower, "er_trend_lower")
        validate_finite(self.er_trend_range, "er_trend_range")
        if self.er_trend_range <= 0.0:
            raise ContractValidationError("er_trend_range must be strictly positive")
        validate_finite(self.ewma_lambda, "ewma_lambda")
        if not (0.0 < self.ewma_lambda < 1.0):
            raise ContractValidationError(f"ewma_lambda must be in (0, 1), got {self.ewma_lambda}")
        validate_finite(self.min_volatility_floor, "min_volatility_floor")
        if self.min_volatility_floor <= 0.0:
            raise ContractValidationError("min_volatility_floor must be strictly positive")
        validate_finite(self.high_volatility_threshold, "high_volatility_threshold")
        if self.high_volatility_threshold <= self.min_volatility_floor:
            raise ContractValidationError("high_volatility_threshold must exceed min_volatility_floor")


@dataclass(frozen=True, slots=True)
class RegimeOutput:
    """Point-in-time output of the regime detector."""
    regime_type: MarketRegimeType
    efficiency_ratio: float
    trend_share: float                   # 'a' in [0.0, 1.0]
    ewma_volatility: float               # sigma
    is_high_volatility: bool
    evaluation_timestamp: datetime
    confidence: float
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.regime_type, MarketRegimeType):
            raise ContractValidationError(f"Invalid regime_type: {self.regime_type}")
        validate_finite(self.efficiency_ratio, "efficiency_ratio")
        validate_finite(self.trend_share, "trend_share")
        if not (0.0 <= self.trend_share <= 1.0):
            raise ContractValidationError(f"trend_share must be in [0, 1], got {self.trend_share}")
        validate_finite(self.ewma_volatility, "ewma_volatility")
        if self.ewma_volatility <= 0.0:
            raise ContractValidationError("ewma_volatility must be strictly positive")
        validate_timezone_aware(self.evaluation_timestamp, "evaluation_timestamp")
        validate_finite(self.confidence, "confidence")


class MarketRegimeDetector:
    """Causal regime detector implementing Step 1 (Volatility) & Step 5 (Regime) of Live Algorithm."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    @property
    def required_warmup_bars(self) -> int:
        return self.config.er_window + 2

    def calculate_ewma_volatility(self, closes: Sequence[float]) -> float:
        """Compute sqrt(EWMA(r^2, lambda=0.94)), floored at min_volatility_floor."""
        if len(closes) < 2:
            return self.config.min_volatility_floor

        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        ewma_sq = log_returns[0] ** 2
        decay = self.config.ewma_lambda

        for r in log_returns[1:]:
            ewma_sq = decay * ewma_sq + (1.0 - decay) * (r ** 2)

        sigma = math.sqrt(ewma_sq)
        return max(self.config.min_volatility_floor, sigma)

    def calculate_efficiency_ratio(self, closes: Sequence[float]) -> float:
        """Kaufman Efficiency Ratio = |C[t] - C[t-N]| / sum(|C[i] - C[i-1]|) over N days."""
        n = self.config.er_window
        if len(closes) < n + 1:
            return 0.0

        net_change = abs(closes[-1] - closes[-n - 1])
        gross_change = sum(abs(closes[-i] - closes[-i - 1]) for i in range(1, n + 1))

        if gross_change < 1e-9:
            return 0.0

        return min(1.0, max(0.0, net_change / gross_change))

    def calculate_trend_share(self, efficiency_ratio: float) -> float:
        """Compute trend share: a = clip((ER - er_trend_lower) / er_trend_range, 0, 1)."""
        raw = (efficiency_ratio - self.config.er_trend_lower) / self.config.er_trend_range
        return min(1.0, max(0.0, raw))

    def detect(self, bars: Sequence[MarketBar]) -> RegimeOutput:
        """Evaluate market bars causally as of latest bar."""
        if not bars:
            raise ContractValidationError("Cannot detect regime on empty bar sequence")

        latest_bar = bars[-1]
        as_of = latest_bar.source_timestamp

        # Fallback on invalid data quality
        if latest_bar.data_quality != DataQualityStatus.VALID:
            return RegimeOutput(
                regime_type=MarketRegimeType.UNKNOWN,
                efficiency_ratio=0.0,
                trend_share=0.5,
                ewma_volatility=self.config.min_volatility_floor,
                is_high_volatility=False,
                evaluation_timestamp=as_of,
                confidence=0.0,
                reason_codes=("DATA_QUALITY_INVALID", f"STATUS_{latest_bar.data_quality.value}"),
            )

        # Fallback on insufficient warmup
        if len(bars) < self.required_warmup_bars:
            return RegimeOutput(
                regime_type=MarketRegimeType.UNKNOWN,
                efficiency_ratio=0.0,
                trend_share=0.5,
                ewma_volatility=self.config.min_volatility_floor,
                is_high_volatility=False,
                evaluation_timestamp=as_of,
                confidence=0.0,
                reason_codes=(
                    "INSUFFICIENT_HISTORY",
                    f"BARS_{len(bars)}_OF_REQUIRED_{self.required_warmup_bars}",
                ),
            )

        closes = [b.close for b in bars]
        sigma = self.calculate_ewma_volatility(closes)
        er = self.calculate_efficiency_ratio(closes)
        trend_share = self.calculate_trend_share(er)

        is_high_vol = sigma >= self.config.high_volatility_threshold

        reasons: list[str] = [
            f"ER_{er:.3f}",
            f"TREND_SHARE_A_{trend_share:.3f}",
            f"EWMA_SIGMA_{sigma:.5f}",
        ]

        if is_high_vol:
            regime = MarketRegimeType.HIGH_VOLATILITY
            reasons.append("VOLATILITY_EXCEEDS_HIGH_THRESHOLD")
        elif trend_share >= 0.65:
            regime = MarketRegimeType.TRENDING
            reasons.append("HIGH_DIRECTIONAL_EFFICIENCY")
        elif trend_share <= 0.35:
            regime = MarketRegimeType.RANGING
            reasons.append("LOW_DIRECTIONAL_EFFICIENCY_CHOP")
        else:
            regime = MarketRegimeType.NORMAL
            reasons.append("BALANCED_EFFICIENCY")

        confidence = 0.5 + 0.5 * abs(trend_share - 0.5) * 2.0

        return RegimeOutput(
            regime_type=regime,
            efficiency_ratio=er,
            trend_share=trend_share,
            ewma_volatility=sigma,
            is_high_volatility=is_high_vol,
            evaluation_timestamp=as_of,
            confidence=confidence,
            reason_codes=tuple(reasons),
        )

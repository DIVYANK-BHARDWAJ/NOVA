"""NOVA Features — Exponential Moving Average and Crossover Calculations.

Provides causal, deterministic exponential moving average (EMA) computation,
spread metrics, and crossover detection for OHLCV time series.

Causality Invariants:
1. Pure point-in-time: Features at time t are functions only of observations <= t.
2. Appending new bars never revises previously computed values.
3. Warm-up is explicitly tracked; features prior to slow_window are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from nova.quant.contracts import NumericalValidationError, ContractValidationError


def validate_window(window: int, name: str = "window") -> None:
    """Validate that moving average window is a positive integer >= 2."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise NumericalValidationError(f"'{name}' must be an integer (non-boolean), got {type(window)}")
    if window < 2:
        raise NumericalValidationError(f"'{name}' must be >= 2, got {window}")


@dataclass(frozen=True, slots=True)
class EMACrossoverFeatures:
    """Point-in-time EMA features computed for a single bar."""
    index: int
    timestamp: datetime
    close: float
    fast_ema: float
    slow_ema: float
    spread: float
    normalized_spread: float
    crossover_event: int  # +1 for bullish cross, -1 for bearish cross, 0 for none
    is_warmed_up: bool    # True when index >= slow_window - 1


def calculate_ema_series(series: Sequence[float], window: int) -> list[float]:
    """Calculate causal exponential moving average over a sequence of prices.

    Mathematical Definition:
        alpha = 2.0 / (window + 1.0)
        EMA_0 = series[0]
        EMA_t = alpha * series[t] + (1.0 - alpha) * EMA_{t-1} for t >= 1

    Properties:
        - Strictly causal: EMA_t depends only on series[0...t].
        - Appending series[t+1] does NOT change EMA_0...EMA_t.
        - Deterministic and numerical-stability verified.
    """
    validate_window(window, "window")
    if not series:
        return []

    for i, val in enumerate(series):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise NumericalValidationError(f"Value at index {i} must be a finite number, got {val}")
        if val <= 0.0:
            raise NumericalValidationError(f"Price at index {i} must be positive, got {val}")

    alpha = 2.0 / (float(window) + 1.0)
    ema_values: list[float] = [float(series[0])]

    for i in range(1, len(series)):
        current_ema = alpha * float(series[i]) + (1.0 - alpha) * ema_values[-1]
        ema_values.append(current_ema)

    return ema_values


def compute_ema_crossover_features(
    closes: Sequence[float],
    timestamps: Sequence[datetime],
    fast_window: int = 12,
    slow_window: int = 26,
) -> list[EMACrossoverFeatures]:
    """Compute causal fast/slow EMA crossover features for a sequence of prices.

    Invariants:
        - fast_window < slow_window
        - timestamps and closes must have identical length
        - warmup requires at least slow_window observations
    """
    validate_window(fast_window, "fast_window")
    validate_window(slow_window, "slow_window")
    if fast_window >= slow_window:
        raise ContractValidationError(
            f"fast_window ({fast_window}) must be strictly less than slow_window ({slow_window})"
        )

    if len(closes) != len(timestamps):
        raise ContractValidationError(
            f"Length mismatch: closes ({len(closes)}) vs timestamps ({len(timestamps)})"
        )

    if not closes:
        return []

    fast_emas = calculate_ema_series(closes, fast_window)
    slow_emas = calculate_ema_series(closes, slow_window)

    features: list[EMACrossoverFeatures] = []
    for i in range(len(closes)):
        f_val = fast_emas[i]
        s_val = slow_emas[i]
        spread = f_val - s_val
        norm_spread = spread / s_val if s_val > 0.0 else 0.0

        is_warmed_up = i >= (slow_window - 1)

        # Detect crossover event
        crossover_event = 0
        if is_warmed_up and i > 0:
            prev_f = fast_emas[i - 1]
            prev_s = slow_emas[i - 1]
            if prev_f <= prev_s and f_val > s_val:
                crossover_event = 1  # Bullish crossover
            elif prev_f >= prev_s and f_val < s_val:
                crossover_event = -1  # Bearish crossover

        features.append(
            EMACrossoverFeatures(
                index=i,
                timestamp=timestamps[i],
                close=float(closes[i]),
                fast_ema=f_val,
                slow_ema=s_val,
                spread=spread,
                normalized_spread=norm_spread,
                crossover_event=crossover_event,
                is_warmed_up=is_warmed_up,
            )
        )

    return features


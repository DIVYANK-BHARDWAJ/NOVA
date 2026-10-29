"""NOVA Features — Causal Mean Reversion Indicators and Oscillators.

Provides deterministic, point-in-time mean reversion metrics:
- Rolling mean and rolling standard deviation
- Rolling price z-score
- Percentage price deviation from rolling mean
- Relative Strength Index (RSI-14) with Wilder's smoothing
- Rolling Volume-Weighted Average Price (VWAP) and VWAP dislocation

Causality Invariants:
1. Pure point-in-time: Features at bar t use strictly bars <= t.
2. Zero-variance and zero-volume conditions are handled without division-by-zero or NaNs.
3. Warmup periods return None until full window length is satisfied.
"""

from __future__ import annotations

import math
from typing import Sequence

from nova.features.moving_averages import validate_window
from nova.quant.contracts import ContractValidationError, NumericalValidationError


def calculate_rolling_mean(series: Sequence[float], window: int) -> list[float | None]:
    """Calculate causal rolling arithmetic mean over window bars.

    Returns None for warmup bars (t < window - 1).
    """
    validate_window(window, "window")
    n = len(series)
    result: list[float | None] = [None] * n
    if n < window:
        return result

    running_sum = sum(series[:window])
    result[window - 1] = running_sum / float(window)

    for i in range(window, n):
        running_sum += series[i] - series[i - window]
        result[i] = running_sum / float(window)

    return result


def calculate_rolling_std(series: Sequence[float], window: int, ddof: int = 1) -> list[float | None]:
    """Calculate causal rolling sample standard deviation over window bars.

    Uses two-pass numerical stability over each window.
    Guards against negative variances caused by floating-point inaccuracy.
    Returns None for warmup bars (t < window - 1).
    """
    validate_window(window, "window")
    if ddof < 0 or ddof >= window:
        raise NumericalValidationError(f"ddof must be in [0, {window-1}], got {ddof}")

    n = len(series)
    result: list[float | None] = [None] * n
    if n < window:
        return result

    w_denom = float(window - ddof)

    for i in range(window - 1, n):
        window_vals = series[i - window + 1 : i + 1]
        mean_val = sum(window_vals) / float(window)
        variance = sum((x - mean_val) ** 2 for x in window_vals) / w_denom
        if variance <= 1e-15:
            result[i] = 0.0
        else:
            result[i] = math.sqrt(variance)

    return result


def calculate_rolling_z_score(series: Sequence[float], window: int) -> list[float | None]:
    """Calculate causal rolling z-score: (Close_t - mean_t) / std_t.

    If rolling standard deviation is zero (constant prices), returns 0.0.
    Returns None for warmup bars (t < window - 1).
    """
    validate_window(window, "window")
    n = len(series)
    means = calculate_rolling_mean(series, window)
    stds = calculate_rolling_std(series, window, ddof=1)

    result: list[float | None] = [None] * n
    for i in range(n):
        m = means[i]
        s = stds[i]
        if m is not None and s is not None:
            if s > 1e-12:
                result[i] = (series[i] - m) / s
            else:
                result[i] = 0.0

    return result


def calculate_mean_deviation(series: Sequence[float], window: int) -> list[float | None]:
    """Calculate percentage price deviation from rolling mean: (Close_t - mean_t) / mean_t.

    Returns dimensionless decimal fraction (e.g. +0.02 = +2.0% above mean).
    Returns None for warmup bars (t < window - 1).
    """
    means = calculate_rolling_mean(series, window)
    result: list[float | None] = [None] * len(series)

    for i in range(len(series)):
        m = means[i]
        if m is not None and m > 0.0:
            result[i] = (series[i] - m) / m

    return result


def calculate_rsi_series(closes: Sequence[float], window: int = 14) -> list[float | None]:
    """Calculate causal Relative Strength Index (RSI) using Wilder's smoothing.

    Properties:
    - Bounded strictly in [0.0, 100.0].
    - Returns None for warmup bars (t < window).
    - If average loss is 0 and average gain is 0 (constant prices), returns 50.0.
    - If average loss is 0 and average gain > 0, returns 100.0.
    """
    validate_window(window, "window")
    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= window:
        return result

    gains = [0.0] * n
    losses = [0.0] * n

    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        if delta > 0.0:
            gains[i] = delta
        elif delta < 0.0:
            losses[i] = -delta

    # First average is simple arithmetic mean over window
    avg_gain = sum(gains[1 : window + 1]) / float(window)
    avg_loss = sum(losses[1 : window + 1]) / float(window)

    def compute_rsi(ag: float, al: float) -> float:
        if al < 1e-14:
            return 100.0 if ag > 1e-14 else 50.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    result[window] = compute_rsi(avg_gain, avg_loss)

    w_float = float(window)
    for i in range(window + 1, n):
        avg_gain = (avg_gain * (w_float - 1.0) + gains[i]) / w_float
        avg_loss = (avg_loss * (w_float - 1.0) + losses[i]) / w_float
        result[i] = compute_rsi(avg_gain, avg_loss)

    return result


def calculate_rolling_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    window: int = 20,
) -> tuple[list[float | None], list[float | None]]:
    """Calculate causal rolling Volume-Weighted Average Price (VWAP) and VWAP deviation.

    Typical Price = (High + Low + Close) / 3.0
    Rolling VWAP = Sum_{k=0..window-1}(TP_k * Volume_k) / Sum_{k=0..window-1}(Volume_k)
    Deviation = (Close - VWAP) / VWAP

    Returns:
        (vwap_series, vwap_deviation_series)

    If volume over the window is zero or unavailable, falls back to rolling mean TP.
    Returns None for warmup bars (t < window - 1).
    """
    validate_window(window, "window")
    n = len(closes)
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ContractValidationError("Length mismatch between highs, lows, closes, and volumes")

    vwaps: list[float | None] = [None] * n
    deviations: list[float | None] = [None] * n
    if n < window:
        return vwaps, deviations

    tp_series = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]

    for i in range(window - 1, n):
        w_tp = tp_series[i - window + 1 : i + 1]
        w_vol = volumes[i - window + 1 : i + 1]

        total_vol = sum(w_vol)
        if total_vol > 1e-6:
            cum_pv = sum(p * v for p, v in zip(w_tp, w_vol))
            vwap_val = cum_pv / total_vol
        else:
            # Fallback to rolling typical price
            vwap_val = sum(w_tp) / float(window)

        vwaps[i] = vwap_val
        if vwap_val > 0.0:
            deviations[i] = (closes[i] - vwap_val) / vwap_val
        else:
            deviations[i] = 0.0

    return vwaps, deviations


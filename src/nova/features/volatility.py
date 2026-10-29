"""NOVA Features — Causal Volatility and Downside Risk Metrics.

Provides deterministic, point-in-time volatility measures:
- True Range (TR)
- Average True Range (ATR-14) using Wilder's smoothing
- Normalized ATR (NATR = ATR / Close)
- Rolling realized close-to-close return volatility
- Parkinson High-Low range volatility estimator
- Downside deviation (negative-return semi-variance)
- Inter-bar opening gap characteristics

Causality Invariants:
1. Pure point-in-time: Features at bar t use strictly bars <= t.
2. Returns and volatility are expressed as dimensionless decimal fractions.
3. Zero-volatility conditions return 0.0 with full numerical safety.
"""

from __future__ import annotations

import math
from typing import Sequence

from nova.features.moving_averages import validate_window
from nova.quant.contracts import ContractValidationError, NumericalValidationError


def calculate_true_range(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    """Calculate causal True Range for each bar.

    Formula:
        TR_0 = High_0 - Low_0
        TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|) for t >= 1
    """
    n = len(closes)
    if not (len(highs) == len(lows) == len(closes)):
        raise ContractValidationError("Length mismatch between highs, lows, and closes")
    if n == 0:
        return []

    tr = [0.0] * n
    tr[0] = max(highs[0] - lows[0], 0.0)

    for i in range(1, n):
        h, l, prev_c = highs[i], lows[i], closes[i - 1]
        tr[i] = max(h - l, abs(h - prev_c), abs(l - prev_c))

    return tr


def calculate_atr_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int = 14,
) -> tuple[list[float | None], list[float | None]]:
    """Calculate causal Average True Range (ATR) and Normalized ATR (NATR).

    Uses classical Wilder's smoothing.
    NATR = ATR_t / Close_t (dimensionless decimal fraction).

    Returns:
        (atr_series, natr_series)
    Returns None for warmup bars (t < window).
    """
    validate_window(window, "window")
    n = len(closes)
    atr: list[float | None] = [None] * n
    natr: list[float | None] = [None] * n
    if n <= window:
        return atr, natr

    tr = calculate_true_range(highs, lows, closes)

    # First ATR is simple arithmetic mean over window
    current_atr = sum(tr[1 : window + 1]) / float(window)
    atr[window] = current_atr
    natr[window] = current_atr / closes[window] if closes[window] > 0.0 else 0.0

    w_float = float(window)
    for i in range(window + 1, n):
        current_atr = (current_atr * (w_float - 1.0) + tr[i]) / w_float
        atr[i] = current_atr
        natr[i] = current_atr / closes[i] if closes[i] > 0.0 else 0.0

    return atr, natr


def calculate_realized_volatility(
    closes: Sequence[float],
    window: int = 20,
    annualization_factor: float = 1.0,
) -> list[float | None]:
    """Calculate causal rolling close-to-close realized return volatility.

    Uses log returns: r_t = ln(Close_t / Close_{t-1}).
    Standard deviation over rolling window * sqrt(annualization_factor).
    If annualization_factor == 1.0, returns unannualized per-bar volatility.

    Returns None for warmup bars (t < window).
    """
    validate_window(window, "window")
    if annualization_factor <= 0.0 or not math.isfinite(annualization_factor):
        raise NumericalValidationError(f"annualization_factor must be positive finite, got {annualization_factor}")

    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= window:
        return result

    # Log returns
    log_returns = [0.0] * n
    for i in range(1, n):
        if closes[i - 1] > 0.0 and closes[i] > 0.0:
            log_returns[i] = math.log(closes[i] / closes[i - 1])
        else:
            log_returns[i] = 0.0

    scale = math.sqrt(annualization_factor)
    w_denom = float(window - 1)

    for i in range(window, n):
        w_ret = log_returns[i - window + 1 : i + 1]
        mean_r = sum(w_ret) / float(window)
        variance = sum((r - mean_r) ** 2 for r in w_ret) / w_denom
        if variance > 1e-15:
            result[i] = math.sqrt(variance) * scale
        else:
            result[i] = 0.0

    return result


def calculate_parkinson_volatility(
    highs: Sequence[float],
    lows: Sequence[float],
    window: int = 20,
) -> list[float | None]:
    """Calculate causal Parkinson High-Low volatility estimator over rolling window.

    Formula:
        sigma_parkinson = sqrt( (1 / (4 * ln(2) * window)) * sum(ln(High_k / Low_k)^2) )

    Returns None for warmup bars (t < window - 1).
    """
    validate_window(window, "window")
    n = len(highs)
    if len(highs) != len(lows):
        raise ContractValidationError("Length mismatch between highs and lows")

    result: list[float | None] = [None] * n
    if n < window:
        return result

    factor = 1.0 / (4.0 * math.log(2.0) * float(window))

    for i in range(window - 1, n):
        sum_sq = 0.0
        for k in range(i - window + 1, i + 1):
            h, l = highs[k], lows[k]
            if h > 0.0 and l > 0.0 and h >= l:
                ratio = math.log(h / l)
                sum_sq += ratio * ratio
        var = sum_sq * factor
        result[i] = math.sqrt(var) if var > 1e-15 else 0.0

    return result


def calculate_downside_deviation(
    closes: Sequence[float],
    window: int = 20,
    target_return: float = 0.0,
) -> list[float | None]:
    """Calculate causal downside deviation (semi-variance of returns below target_return).

    Returns None for warmup bars (t < window).
    """
    validate_window(window, "window")
    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= window:
        return result

    returns = [0.0] * n
    for i in range(1, n):
        returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] > 0.0 else 0.0

    for i in range(window, n):
        w_ret = returns[i - window + 1 : i + 1]
        downside_sum_sq = sum(min(r - target_return, 0.0) ** 2 for r in w_ret)
        variance = downside_sum_sq / float(window)
        result[i] = math.sqrt(variance) if variance > 1e-15 else 0.0

    return result


def calculate_opening_gap(
    opens: Sequence[float],
    closes: Sequence[float],
) -> list[float | None]:
    """Calculate causal percentage opening gap relative to previous bar close.

    Formula:
        Gap_t = (Open_t - Close_{t-1}) / Close_{t-1} for t >= 1
    First bar returns None.
    """
    n = len(opens)
    if len(opens) != len(closes):
        raise ContractValidationError("Length mismatch between opens and closes")

    result: list[float | None] = [None] * n
    for i in range(1, n):
        prev_c = closes[i - 1]
        if prev_c > 0.0:
            result[i] = (opens[i] - prev_c) / prev_c
        else:
            result[i] = 0.0

    return result


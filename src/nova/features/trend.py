"""NOVA Features — Causal Trend and Crossover Indicators.

Provides deterministic, point-in-time trend indicators:
- Fast/Slow EMA, EMA spreads, and crossover events
- Momentum (Rate of Change)
- Linear regression trend slope
- Kaufman Efficiency Ratio (KER)
- Crossover recency tracking
- Average Directional Index (ADX)

Causality Invariants:
1. Pure point-in-time: Features at bar t are functions strictly of bars <= t.
2. Appending future bars never revises previously computed historical features.
3. Warmup periods are explicitly tracked; insufficient-history bars return None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from nova.features.moving_averages import (
    EMACrossoverFeatures,
    calculate_ema_series,
    compute_ema_crossover_features,
    validate_window,
)
from nova.quant.contracts import ContractValidationError, NumericalValidationError


def calculate_momentum_series(closes: Sequence[float], lookback: int = 5) -> list[float | None]:
    """Calculate causal price Rate-of-Change (momentum) over lookback bars.

    Formula:
        ROC_t = (Close_t - Close_{t - lookback}) / Close_{t - lookback}

    Returns None for warmup bars (t < lookback).
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise NumericalValidationError(f"lookback must be an integer >= 1, got {lookback}")

    n = len(closes)
    result: list[float | None] = [None] * n

    for i in range(n):
        c = closes[i]
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0.0:
            raise NumericalValidationError(f"Price at index {i} must be positive finite float, got {c}")

        if i >= lookback:
            past_c = closes[i - lookback]
            result[i] = (c - past_c) / past_c

    return result


def calculate_trend_slope(closes: Sequence[float], window: int = 10) -> list[float | None]:
    """Calculate causal linear regression slope over rolling window of prices.

    The slope is normalized by the mean price over the window to produce
    a dimensionless percentage return slope per bar.

    Returns None for warmup bars (t < window - 1).
    """
    validate_window(window, "window")
    n = len(closes)
    result: list[float | None] = [None] * n
    if n < window:
        return result

    # Precompute sum of x and sum of x^2 for x in 0..window-1
    # x_mean = (window - 1) / 2
    # S_xx = sum((x - x_mean)^2) = window * (window^2 - 1) / 12
    w = float(window)
    s_xx = w * (w * w - 1.0) / 12.0
    x_mean = (w - 1.0) / 2.0

    for i in range(window - 1, n):
        window_closes = closes[i - window + 1 : i + 1]
        y_mean = sum(window_closes) / w
        if y_mean <= 0.0:
            result[i] = 0.0
            continue

        s_xy = 0.0
        for j in range(window):
            s_xy += (float(j) - x_mean) * (window_closes[j] - y_mean)

        raw_slope = s_xy / s_xx
        norm_slope = raw_slope / y_mean
        result[i] = norm_slope

    return result


def calculate_kaufman_efficiency_ratio(closes: Sequence[float], window: int = 10) -> list[float | None]:
    """Calculate Kaufman Efficiency Ratio (KER) measuring trend strength.

    Formula:
        Direction = |Close_t - Close_{t - window}|
        Volatility = Sum_{k=1..window} |Close_{t-k+1} - Close_{t-k}|
        KER = Direction / Volatility

    KER is bounded in [0.0, 1.0], where 1.0 is a pure straight trend and 0.0 is pure noise.
    Returns None for warmup bars (t < window).
    """
    validate_window(window, "window")
    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= window:
        return result

    for i in range(window, n):
        direction = abs(closes[i] - closes[i - window])
        volatility = 0.0
        for k in range(window):
            volatility += abs(closes[i - k] - closes[i - k - 1])

        if volatility > 0.0:
            ker = direction / volatility
            result[i] = min(max(ker, 0.0), 1.0)
        else:
            result[i] = 0.0

    return result


def calculate_crossover_recency(crossover_events: Sequence[int]) -> list[int | None]:
    """Calculate bars elapsed since the most recent crossover event (+1 or -1).

    Returns None for bars before any crossover has occurred.
    Returns 0 at the crossover bar, 1 at the subsequent bar, etc.
    """
    n = len(crossover_events)
    result: list[int | None] = [None] * n
    last_cross_idx: int | None = None

    for i in range(n):
        ev = crossover_events[i]
        if ev in (1, -1):
            last_cross_idx = i

        if last_cross_idx is not None:
            result[i] = i - last_cross_idx

    return result


def calculate_adx_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int = 14,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Calculate causal Average Directional Index (ADX) with +DI and -DI.

    Uses classical Wilder's smoothing.
    Returns: (plus_di, minus_di, adx)

    All components are bounded in [0.0, 100.0].
    Warmup requires 2 * window bars for ADX.
    """
    validate_window(window, "window")
    n = len(closes)
    if not (len(highs) == len(lows) == len(closes)):
        raise ContractValidationError(
            f"Length mismatch: highs={len(highs)}, lows={len(lows)}, closes={len(closes)}"
        )

    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n
    adx: list[float | None] = [None] * n

    if n < window + 1:
        return plus_di, minus_di, adx

    # Step 1: Compute True Range, +DM, -DM
    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        h, l, prev_c = highs[i], lows[i], closes[i - 1]
        tr[i] = max(h - l, abs(h - prev_c), abs(l - prev_c))

        up_move = h - highs[i - 1]
        down_move = lows[i - 1] - l

        if up_move > down_move and up_move > 0.0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0.0:
            minus_dm[i] = down_move

    # Step 2: Wilder's smoothing of TR, +DM, -DM
    # First smoothed value at index = window is the sum of window bars
    smooth_tr = sum(tr[1 : window + 1])
    smooth_pdm = sum(plus_dm[1 : window + 1])
    smooth_mdm = sum(minus_dm[1 : window + 1])

    dx_values: list[float | None] = [None] * n

    def calc_di(p_val: float, m_val: float, tr_val: float) -> tuple[float, float, float]:
        if tr_val > 0.0:
            p_di = 100.0 * p_val / tr_val
            m_di = 100.0 * m_val / tr_val
        else:
            p_di, m_di = 0.0, 0.0
        di_sum = p_di + m_di
        dx_val = 100.0 * abs(p_di - m_di) / di_sum if di_sum > 0.0 else 0.0
        return p_di, m_di, dx_val

    p_di, m_di, dx = calc_di(smooth_pdm, smooth_mdm, smooth_tr)
    plus_di[window] = p_di
    minus_di[window] = m_di
    dx_values[window] = dx

    w_float = float(window)
    for i in range(window + 1, n):
        smooth_tr = smooth_tr - (smooth_tr / w_float) + tr[i]
        smooth_pdm = smooth_pdm - (smooth_pdm / w_float) + plus_dm[i]
        smooth_mdm = smooth_mdm - (smooth_mdm / w_float) + minus_dm[i]

        p_di, m_di, dx = calc_di(smooth_pdm, smooth_mdm, smooth_tr)
        plus_di[i] = p_di
        minus_di[i] = m_di
        dx_values[i] = dx

    # Step 3: ADX is Wilder's smoothing of DX (requires window valid DX values)
    # The first ADX is available at index = 2 * window - 1
    first_adx_idx = 2 * window - 1
    if n > first_adx_idx:
        valid_dxs = [dx_values[j] for j in range(window, first_adx_idx + 1) if dx_values[j] is not None]
        if len(valid_dxs) == window:
            current_adx = sum(valid_dxs) / w_float
            adx[first_adx_idx] = current_adx

            for i in range(first_adx_idx + 1, n):
                dx_val = dx_values[i]
                if dx_val is not None:
                    current_adx = (current_adx * (w_float - 1.0) + dx_val) / w_float
                    adx[i] = current_adx

    return plus_di, minus_di, adx


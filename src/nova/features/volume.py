"""NOVA Features — Causal Volume and Liquidity Proxies.

Provides deterministic, point-in-time volume indicators:
- Rolling mean volume
- Relative Volume (RVOL)
- Volume z-score
- On-Balance Volume (OBV)
- Price-volume divergence indicator

Causality Invariants:
1. Pure point-in-time: Features at bar t use strictly bars <= t.
2. Missing or zero volume is handled explicitly without division-by-zero.
"""

from __future__ import annotations

import math
from typing import Sequence

from nova.features.mean_reversion import calculate_rolling_mean, calculate_rolling_std
from nova.features.moving_averages import validate_window
from nova.quant.contracts import ContractValidationError, NumericalValidationError


def calculate_rolling_mean_volume(volumes: Sequence[float], window: int = 20) -> list[float | None]:
    """Calculate causal rolling average volume over window bars.

    Returns None for warmup bars (t < window - 1).
    """
    return calculate_rolling_mean(volumes, window)


def calculate_relative_volume(volumes: Sequence[float], window: int = 20) -> list[float | None]:
    """Calculate Relative Volume: RVOL_t = Volume_t / MeanVolume_t.

    RVOL > 1.0 indicates volume surge above rolling baseline.
    Returns None for warmup bars (t < window - 1).
    """
    means = calculate_rolling_mean(volumes, window)
    result: list[float | None] = [None] * len(volumes)

    for i in range(len(volumes)):
        m = means[i]
        if m is not None:
            if m > 1e-6:
                result[i] = volumes[i] / m
            else:
                result[i] = 1.0 if volumes[i] <= 1e-6 else 0.0

    return result


def calculate_volume_z_score(volumes: Sequence[float], window: int = 20) -> list[float | None]:
    """Calculate causal rolling volume z-score: (Volume_t - MeanVol) / StdVol.

    Returns None for warmup bars (t < window - 1).
    """
    means = calculate_rolling_mean(volumes, window)
    stds = calculate_rolling_std(volumes, window, ddof=1)
    result: list[float | None] = [None] * len(volumes)

    for i in range(len(volumes)):
        m = means[i]
        s = stds[i]
        if m is not None and s is not None:
            if s > 1e-6:
                result[i] = (volumes[i] - m) / s
            else:
                result[i] = 0.0

    return result


def calculate_on_balance_volume(
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Calculate causal On-Balance Volume (OBV).

    Formula:
        OBV_0 = Volume_0
        OBV_t = OBV_{t-1} + Volume_t if Close_t > Close_{t-1}
        OBV_t = OBV_{t-1} - Volume_t if Close_t < Close_{t-1}
        OBV_t = OBV_{t-1}             if Close_t == Close_{t-1}
    """
    n = len(closes)
    if len(closes) != len(volumes):
        raise ContractValidationError("Length mismatch between closes and volumes")
    if n == 0:
        return []

    obv = [0.0] * n
    obv[0] = float(volumes[0])

    for i in range(1, n):
        c, prev_c = closes[i], closes[i - 1]
        v = float(volumes[i])
        if c > prev_c:
            obv[i] = obv[i - 1] + v
        elif c < prev_c:
            obv[i] = obv[i - 1] - v
        else:
            obv[i] = obv[i - 1]

    return obv


def calculate_price_volume_divergence(
    closes: Sequence[float],
    volumes: Sequence[float],
    window: int = 10,
) -> list[float | None]:
    """Calculate causal price-volume divergence score over rolling window.

    Compares price Rate of Change with normalized volume trend.
    Returns:
        +1.0: Bullish divergence (price flat or falling while OBV/volume surges)
        -1.0: Bearish divergence (price rising while OBV/volume declines)
         0.0: Agreement / no significant divergence
    Returns None for warmup bars (t < window).
    """
    validate_window(window, "window")
    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= window:
        return result

    obv = calculate_on_balance_volume(closes, volumes)

    for i in range(window, n):
        price_ret = (closes[i] - closes[i - window]) / closes[i - window] if closes[i - window] > 0.0 else 0.0
        obv_diff = obv[i] - obv[i - window]

        # Significant threshold for divergence detection (e.g. 0.5% price movement)
        if price_ret > 0.005 and obv_diff < 0.0:
            result[i] = -1.0  # Bearish divergence (price pushing up on negative volume flow)
        elif price_ret < -0.005 and obv_diff > 0.0:
            result[i] = 1.0   # Bullish divergence (price dipping on positive volume accumulation)
        else:
            result[i] = 0.0

    return result


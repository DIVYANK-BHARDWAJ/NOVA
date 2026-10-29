"""NOVA Features — Causal Benchmark-Relative Market Context Features.

Provides point-in-time cross-asset and benchmark-relative metrics:
- Benchmark-relative single-bar returns
- Benchmark-relative rolling momentum spread

Causality & Data Integrity Invariants:
1. Strict alignment: Asset bars and benchmark bars must align on timestamp.
2. If benchmark data is unavailable or missing, features return None (NO fabricated data).
3. Benchmark bar availability must be verified at or before asset decision timestamp.
"""

from __future__ import annotations

from typing import Sequence

from nova.features.moving_averages import validate_window
from nova.quant.contracts import ContractValidationError


def calculate_benchmark_relative_return(
    asset_closes: Sequence[float],
    benchmark_closes: Sequence[float] | None,
) -> list[float | None]:
    """Calculate causal benchmark-relative return: r_{asset, t} - r_{benchmark, t}.

    Returns None if benchmark_closes is None or for bar 0.
    """
    n = len(asset_closes)
    result: list[float | None] = [None] * n
    if benchmark_closes is None:
        return result

    if len(benchmark_closes) != n:
        raise ContractValidationError(
            f"Length mismatch: asset_closes={n} vs benchmark_closes={len(benchmark_closes)}"
        )

    for i in range(1, n):
        a_prev, a_curr = asset_closes[i - 1], asset_closes[i]
        b_prev, b_curr = benchmark_closes[i - 1], benchmark_closes[i]
        if a_prev > 0.0 and b_prev > 0.0:
            r_asset = (a_curr - a_prev) / a_prev
            r_bench = (b_curr - b_prev) / b_prev
            result[i] = r_asset - r_bench
        else:
            result[i] = 0.0

    return result


def calculate_relative_momentum(
    asset_closes: Sequence[float],
    benchmark_closes: Sequence[float] | None,
    window: int = 10,
) -> list[float | None]:
    """Calculate causal rolling momentum relative to benchmark over window bars.

    Formula:
        Momentum_asset(W) - Momentum_benchmark(W)

    Returns None if benchmark is unavailable or during warmup (t < window).
    """
    validate_window(window, "window")
    n = len(asset_closes)
    result: list[float | None] = [None] * n
    if benchmark_closes is None:
        return result

    if len(benchmark_closes) != n:
        raise ContractValidationError(
            f"Length mismatch: asset_closes={n} vs benchmark_closes={len(benchmark_closes)}"
        )

    for i in range(window, n):
        a_prev, a_curr = asset_closes[i - window], asset_closes[i]
        b_prev, b_curr = benchmark_closes[i - window], benchmark_closes[i]
        if a_prev > 0.0 and b_prev > 0.0:
            mom_asset = (a_curr - a_prev) / a_prev
            mom_bench = (b_curr - b_prev) / b_prev
            result[i] = mom_asset - mom_bench
        else:
            result[i] = 0.0

    return result


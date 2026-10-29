"""NOVA Features — Production-Quality Causal Quantitative Feature Library.

Provides point-in-time, deterministic feature extractors for research and ML modeling:
- Moving Averages and Crossover: calculate_ema_series, compute_ema_crossover_features
- Trend & Crossover: calculate_momentum_series, calculate_trend_slope, calculate_kaufman_efficiency_ratio, calculate_adx_series, calculate_crossover_recency
- Mean Reversion: calculate_rolling_mean, calculate_rolling_std, calculate_rolling_z_score, calculate_mean_deviation, calculate_rsi_series, calculate_rolling_vwap
- Volatility: calculate_true_range, calculate_atr_series, calculate_realized_volatility, calculate_parkinson_volatility, calculate_downside_deviation, calculate_opening_gap
- Volume: calculate_rolling_mean_volume, calculate_relative_volume, calculate_volume_z_score, calculate_on_balance_volume, calculate_price_volume_divergence
- Benchmark Context: calculate_benchmark_relative_return, calculate_relative_momentum
- Pipeline Orchestration: FeaturePipeline, FeaturePipelineConfig, BarFeatures
"""

from nova.features.context import (
    calculate_benchmark_relative_return,
    calculate_relative_momentum,
)
from nova.features.mean_reversion import (
    calculate_mean_deviation,
    calculate_rolling_mean,
    calculate_rolling_std,
    calculate_rolling_vwap,
    calculate_rolling_z_score,
    calculate_rsi_series,
)
from nova.features.moving_averages import (
    EMACrossoverFeatures,
    calculate_ema_series,
    compute_ema_crossover_features,
    validate_window,
)
from nova.features.pipeline import (
    BarFeatures,
    FeaturePipeline,
    FeaturePipelineConfig,
)
from nova.features.trend import (
    calculate_adx_series,
    calculate_crossover_recency,
    calculate_kaufman_efficiency_ratio,
    calculate_momentum_series,
    calculate_trend_slope,
)
from nova.features.volatility import (
    calculate_atr_series,
    calculate_downside_deviation,
    calculate_opening_gap,
    calculate_parkinson_volatility,
    calculate_realized_volatility,
    calculate_true_range,
)
from nova.features.volume import (
    calculate_on_balance_volume,
    calculate_price_volume_divergence,
    calculate_relative_volume,
    calculate_rolling_mean_volume,
    calculate_volume_z_score,
)

__all__ = [
    # Baseline EMA & Crossover (preserved for backwards-compatibility)
    "EMACrossoverFeatures",
    "calculate_ema_series",
    "compute_ema_crossover_features",
    "validate_window",
    # Trend
    "calculate_momentum_series",
    "calculate_trend_slope",
    "calculate_kaufman_efficiency_ratio",
    "calculate_crossover_recency",
    "calculate_adx_series",
    # Mean Reversion
    "calculate_rolling_mean",
    "calculate_rolling_std",
    "calculate_rolling_z_score",
    "calculate_mean_deviation",
    "calculate_rsi_series",
    "calculate_rolling_vwap",
    # Volatility
    "calculate_true_range",
    "calculate_atr_series",
    "calculate_realized_volatility",
    "calculate_parkinson_volatility",
    "calculate_downside_deviation",
    "calculate_opening_gap",
    # Volume
    "calculate_rolling_mean_volume",
    "calculate_relative_volume",
    "calculate_volume_z_score",
    "calculate_on_balance_volume",
    "calculate_price_volume_divergence",
    # Benchmark Context
    "calculate_benchmark_relative_return",
    "calculate_relative_momentum",
    # Pipeline Orchestration
    "BarFeatures",
    "FeaturePipeline",
    "FeaturePipelineConfig",
]

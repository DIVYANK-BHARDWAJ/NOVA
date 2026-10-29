"""NOVA Features package."""

from nova.features.moving_averages import (
    EMACrossoverFeatures,
    calculate_ema_series,
    compute_ema_crossover_features,
    validate_window,
)

__all__ = [
    "EMACrossoverFeatures",
    "calculate_ema_series",
    "compute_ema_crossover_features",
    "validate_window",
]


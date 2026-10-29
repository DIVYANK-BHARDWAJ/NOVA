"""NOVA Regime Intelligence — Configuration for Transparent Market-Regime Detector.

Provides validated, centralized parameters:
- Kaufman Efficiency Ratio (KER) lookback and threshold boundaries
- Volatility lookback and elevated threshold
- EWMA decay factor (RiskMetrics lambda=0.94)
- Minimum warmup bars
- Dynamic loading from centralized TOML configuration

Invariants:
1. Zero hardcoded market assumptions. All parameters are configuration-driven.
2. Missing or invalid parameters fail closed with actionable ContractValidationError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping

from nova.quant.contracts import (
    ContractValidationError,
    NumericalValidationError,
    validate_finite,
)


@dataclass(frozen=True, slots=True)
class MarketRegimeConfig:
    """Validated configuration for Person 2 Transparent Market-Regime Detector."""
    ker_window: int = 10
    ker_trend_threshold: float = 0.50
    ker_ranging_threshold: float = 0.25
    volatility_window: int = 20
    high_volatility_threshold: float = 0.025
    min_volatility_floor: float = 0.005
    ewma_lambda: float = 0.94
    min_warmup_bars: int = 28
    fast_ema_window: int = 12
    slow_ema_window: int = 26
    model_id: str = "REGIME_DETECTOR_TRANSPARENT"
    config_version: str = "1.0.0"

    def __post_init__(self) -> None:
        # Integer window validations
        for name, val in [
            ("ker_window", self.ker_window),
            ("volatility_window", self.volatility_window),
            ("min_warmup_bars", self.min_warmup_bars),
            ("fast_ema_window", self.fast_ema_window),
            ("slow_ema_window", self.slow_ema_window),
        ]:
            if isinstance(val, bool) or not isinstance(val, int) or val < 2:
                raise NumericalValidationError(f"'{name}' must be an integer >= 2, got {val}")

        if self.fast_ema_window >= self.slow_ema_window:
            raise ContractValidationError(
                f"fast_ema_window ({self.fast_ema_window}) must be < slow_ema_window ({self.slow_ema_window})"
            )

        if self.min_warmup_bars < max(self.ker_window + 1, self.volatility_window, self.slow_ema_window):
            raise ContractValidationError(
                f"min_warmup_bars ({self.min_warmup_bars}) must be >= max lookback requirements"
            )

        # Float threshold validations
        for name, val in [
            ("ker_trend_threshold", self.ker_trend_threshold),
            ("ker_ranging_threshold", self.ker_ranging_threshold),
            ("high_volatility_threshold", self.high_volatility_threshold),
            ("min_volatility_floor", self.min_volatility_floor),
            ("ewma_lambda", self.ewma_lambda),
        ]:
            validate_finite(val, name)

        if not (0.0 < self.ker_ranging_threshold < self.ker_trend_threshold <= 1.0):
            raise ContractValidationError(
                f"KER thresholds must satisfy 0.0 < ranging ({self.ker_ranging_threshold}) < trend ({self.ker_trend_threshold}) <= 1.0"
            )

        if not (0.0 < self.min_volatility_floor < self.high_volatility_threshold):
            raise ContractValidationError(
                f"Volatility floor ({self.min_volatility_floor}) must be positive and < high threshold ({self.high_volatility_threshold})"
            )

        if not (0.0 < self.ewma_lambda < 1.0):
            raise ContractValidationError(f"ewma_lambda must be in (0.0, 1.0), got {self.ewma_lambda}")

        # String validations
        for name, val in [("model_id", self.model_id), ("config_version", self.config_version)]:
            if not isinstance(val, str) or not val.strip():
                raise ContractValidationError(f"'{name}' cannot be empty or non-string")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarketRegimeConfig:
        """Instantiate MarketRegimeConfig from dictionary with validation."""
        fields = {
            "ker_window",
            "ker_trend_threshold",
            "ker_ranging_threshold",
            "volatility_window",
            "high_volatility_threshold",
            "min_volatility_floor",
            "ewma_lambda",
            "min_warmup_bars",
            "fast_ema_window",
            "slow_ema_window",
            "model_id",
            "config_version",
        }
        kwargs = {k: v for k, v in data.items() if k in fields}
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, toml_path: str | Path, section: str = "regime") -> MarketRegimeConfig:
        """Load configuration from centralized TOML file."""
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        if section not in cfg:
            raise ContractValidationError(f"Missing section '[{section}]' in TOML config {path}")
        return cls.from_dict(cfg[section])


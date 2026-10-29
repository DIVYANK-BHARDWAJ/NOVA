"""NOVA Quantitative Machine Learning — Model Configuration (Person 2).

Provides validated, immutable configuration for Person 2's GBDT/XGBoost prediction model:
- Target definition and horizon
- Feature subset selection and ordering
- Tree hyperparameters (depth, estimators, learning rate, regularization)
- Chronological train/val/test split ratios and embargo width
- Early stopping parameters
- Contract metadata (model_id, model_version)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any, Mapping

from nova.quant.contracts import (
    ContractValidationError,
    NumericalValidationError,
    validate_finite,
    validate_no_secrets,
)

# Standard causal features selected from nova.features multi-family pipeline
DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "momentum",
    "trend_slope",
    "kaufman_efficiency_ratio",
    "ema_normalized_spread",
    "rolling_z_score",
    "rsi",
    "normalized_atr",
    "realized_volatility",
    "parkinson_volatility",
    "downside_deviation",
    "relative_volume",
    "volume_z_score",
)


@dataclass(frozen=True, slots=True)
class XGBoostModelConfig:
    """Validated configuration for XGBoost / GBDT Forward Return Predictor."""
    model_id: str = "GBDT_OHLCV_BASELINE"
    model_version: str = "1.0.0"
    horizon_bars: int = 1
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    feature_warmup_bars: int = 28
    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 40
    min_child_samples: int = 20
    l2_reg: float = 1.0
    early_stopping_rounds: int = 5
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    tree_method: str = "hist"
    booster: str = "gbtree"
    objective: str = "reg:squarederror"
    random_state: int = 42

    def __post_init__(self) -> None:
        # String validations
        for name, val in [("model_id", self.model_id), ("model_version", self.model_version)]:
            if not isinstance(val, str) or not val.strip():
                raise ContractValidationError(f"'{name}' cannot be blank")

        # Integer parameter validations
        for name, val in [
            ("horizon_bars", self.horizon_bars),
            ("feature_warmup_bars", self.feature_warmup_bars),
            ("max_depth", self.max_depth),
            ("n_estimators", self.n_estimators),
            ("min_child_samples", self.min_child_samples),
            ("early_stopping_rounds", self.early_stopping_rounds),
        ]:
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise NumericalValidationError(f"'{name}' must be a positive integer, got {val}")

        # Float validations
        for name, val in [
            ("learning_rate", self.learning_rate),
            ("l2_reg", self.l2_reg),
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
        ]:
            validate_finite(val, name)

        if not (0.0 < self.learning_rate <= 1.0):
            raise NumericalValidationError(f"learning_rate must be in (0.0, 1.0], got {self.learning_rate}")

        if self.l2_reg < 0.0:
            raise NumericalValidationError(f"l2_reg must be >= 0.0, got {self.l2_reg}")

        if not (0.0 < self.train_ratio < 1.0) or not (0.0 < self.val_ratio < 1.0):
            raise NumericalValidationError("train_ratio and val_ratio must be between 0.0 and 1.0")

        if self.train_ratio + self.val_ratio >= 1.0:
            raise NumericalValidationError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) must be < 1.0 to leave test data"
            )

        if not self.feature_names:
            raise ContractValidationError("feature_names cannot be empty")

        for f in self.feature_names:
            if not isinstance(f, str) or not f.strip():
                raise ContractValidationError("feature_names elements must be non-empty strings")

        if self.objective not in ("reg:squarederror", "binary:logistic"):
            raise ContractValidationError(f"Unsupported objective: '{self.objective}'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "horizon_bars": self.horizon_bars,
            "feature_names": list(self.feature_names),
            "feature_warmup_bars": self.feature_warmup_bars,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "l2_reg": self.l2_reg,
            "early_stopping_rounds": self.early_stopping_rounds,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "tree_method": self.tree_method,
            "booster": self.booster,
            "objective": self.objective,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> XGBoostModelConfig:
        """Construct configuration from dictionary with strict parsing."""
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k == "feature_names" and isinstance(v, (list, tuple)):
                kwargs[k] = tuple(v)
            elif hasattr(cls, k):
                kwargs[k] = v
        return cls(**kwargs)

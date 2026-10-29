"""NOVA Causal Machine Learning Prediction Interface (Person 3 Domain).

Implements point-in-time statistical return prediction using chronological / walk-forward
training, zero future-data leakage, and graceful fallback mechanics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Sequence

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    NumericalValidationError,
    validate_finite,
    validate_timezone_aware,
)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Standardized prediction output produced causally."""
    expected_return: float               # Forward return decimal estimate
    directional_probability: float      # P(return > 0) in [0.0, 1.0]
    confidence: float                   # Model conviction in [0.0, 1.0]
    horizon_bars: int
    timestamp: datetime
    model_id: str
    model_version: str
    is_fallback: bool
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_finite(self.expected_return, "expected_return")
        validate_finite(self.directional_probability, "directional_probability")
        if not (0.0 <= self.directional_probability <= 1.0):
            raise ContractValidationError(f"directional_probability must be in [0, 1], got {self.directional_probability}")
        validate_finite(self.confidence, "confidence")
        if not (0.0 <= self.confidence <= 1.0):
            raise ContractValidationError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.horizon_bars <= 0:
            raise ContractValidationError("horizon_bars must be positive")
        validate_timezone_aware(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Configurable predictor parameters."""
    horizon_bars: int = 1
    l2_regularization: float = 1.0       # Ridge alpha
    min_train_bars: int = 30
    lookback_features: tuple[int, ...] = (1, 3, 5, 10)
    model_id: str = "causal_ridge_predictor"
    model_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ContractValidationError("horizon_bars must be positive")
        validate_finite(self.l2_regularization, "l2_regularization")
        if self.l2_regularization < 0.0:
            raise ContractValidationError("l2_regularization cannot be negative")
        if self.min_train_bars < 10:
            raise ContractValidationError("min_train_bars must be at least 10")


class BasePredictor(ABC):
    """Abstract interface for all causal return predictors."""

    @abstractmethod
    def predict(self, bars: Sequence[MarketBar]) -> PredictionResult:
        pass


class CausalRidgePredictor(BasePredictor):
    """Chronologically trained linear ridge regression model for forward return prediction."""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        self.config = config or PredictorConfig()
        self.weights: list[float] | None = None
        self.bias: float = 0.0
        self.is_fitted: bool = False

    def _extract_feature_vector(self, closes: Sequence[float], idx: int) -> list[float] | None:
        """Extract causal normalized return features as of index idx."""
        max_lookback = max(self.config.lookback_features)
        if idx < max_lookback:
            return None

        current = closes[idx]
        features: list[float] = []
        for lb in self.config.lookback_features:
            past = closes[idx - lb]
            ret = (current - past) / past
            features.append(ret)
        return features

    def fit_chronological(self, bars: Sequence[MarketBar], train_fraction: float = 0.70) -> None:
        """Fit strictly on the historical chronological split [0 : int(len(bars) * train_fraction)].

        Never uses observations after the cutoff timestamp.
        """
        if len(bars) < self.config.min_train_bars:
            return

        cutoff_idx = int(len(bars) * train_fraction)
        train_bars = bars[:cutoff_idx]
        closes = [b.close for b in train_bars]

        h = self.config.horizon_bars
        X: list[list[float]] = []
        y: list[float] = []

        max_lb = max(self.config.lookback_features)
        for i in range(max_lb, len(closes) - h):
            feat = self._extract_feature_vector(closes, i)
            if feat is not None:
                future_return = (closes[i + h] - closes[i]) / closes[i]
                X.append(feat)
                y.append(future_return)

        if len(X) < 10:
            return

        # Solve Ridge Regression: (X^T X + alpha * I)^{-1} X^T y
        n_features = len(self.config.lookback_features)
        # Add bias column
        X_design = [[1.0] + row for row in X]
        n_cols = n_features + 1

        # Compute X^T X
        XtX = [[0.0] * n_cols for _ in range(n_cols)]
        Xty = [0.0] * n_cols

        for row, target in zip(X_design, y):
            for r in range(n_cols):
                Xty[r] += row[r] * target
                for c in range(n_cols):
                    XtX[r][c] += row[r] * row[c]

        # Add Ridge L2 penalty (do not penalize bias at index 0)
        alpha = self.config.l2_regularization
        for i in range(1, n_cols):
            XtX[i][i] += alpha

        # Simple Gaussian elimination solver for small matrix (e.g. 5x5)
        try:
            coeffs = self._solve_linear_system(XtX, Xty)
            self.bias = coeffs[0]
            self.weights = coeffs[1:]
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def _solve_linear_system(self, A: list[list[float]], b: list[float]) -> list[float]:
        """Gauss-Jordan elimination with partial pivoting."""
        n = len(b)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]

        for i in range(n):
            # Pivot
            max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
            M[i], M[max_row] = M[max_row], M[i]

            pivot = M[i][i]
            if abs(pivot) < 1e-12:
                raise ValueError("Singular matrix in ridge regression")

            for j in range(i, n + 1):
                M[i][j] /= pivot

            for r in range(n):
                if r != i:
                    factor = M[r][i]
                    for c in range(i, n + 1):
                        M[r][c] -= factor * M[i][c]

        return [M[i][n] for i in range(n)]

    def predict(self, bars: Sequence[MarketBar]) -> PredictionResult:
        """Causal point-in-time prediction with fallback logic."""
        if not bars:
            raise ContractValidationError("Cannot predict on empty bar sequence")

        latest_bar = bars[-1]
        as_of = latest_bar.source_timestamp

        # Fallback if invalid data
        if latest_bar.data_quality != DataQualityStatus.VALID:
            return self._build_fallback(as_of, reason="DATA_QUALITY_INVALID")

        closes = [b.close for b in bars]
        feat = self._extract_feature_vector(closes, len(closes) - 1)

        if feat is None:
            return self._build_fallback(as_of, reason="INSUFFICIENT_LOOKBACK_BARS")

        # Fallback if model not fitted yet -> use simple 3-bar momentum heuristic
        if not self.is_fitted or self.weights is None:
            return self._build_heuristic_fallback(closes, as_of)

        # Compute dot product
        raw_pred = self.bias + sum(w * f for w, f in zip(self.weights, feat))
        # Logistic probability estimate
        prob = 1.0 / (1.0 + math.exp(-raw_pred / 0.005))
        confidence = min(0.90, max(0.20, abs(prob - 0.5) * 2.0))

        return PredictionResult(
            expected_return=raw_pred,
            directional_probability=prob,
            confidence=confidence,
            horizon_bars=self.config.horizon_bars,
            timestamp=as_of,
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            is_fallback=False,
            reason_codes=(f"RIDGE_FORECAST_{raw_pred:+.4f}", f"PROB_{prob:.3f}"),
        )

    def _build_fallback(self, timestamp: datetime, reason: str) -> PredictionResult:
        return PredictionResult(
            expected_return=0.0,
            directional_probability=0.50,
            confidence=0.0,
            horizon_bars=self.config.horizon_bars,
            timestamp=timestamp,
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            is_fallback=True,
            reason_codes=(f"FALLBACK_{reason}",),
        )

    def _build_heuristic_fallback(self, closes: Sequence[float], timestamp: datetime) -> PredictionResult:
        # Heuristic 3-bar return
        ret3 = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0.0
        prob = 1.0 / (1.0 + math.exp(-ret3 / 0.01))
        return PredictionResult(
            expected_return=ret3 * 0.2, # Conservative dampen
            directional_probability=prob,
            confidence=0.35,
            horizon_bars=self.config.horizon_bars,
            timestamp=timestamp,
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            is_fallback=True,
            reason_codes=("FALLBACK_UNFITTED_MOMENTUM_HEURISTIC",),
        )

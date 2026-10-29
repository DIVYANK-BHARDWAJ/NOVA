"""NOVA Quantitative Strategies — Base Interfaces (Person 3 Exclusive Domain)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from nova.quant.contracts import (
    ContractValidationError,
    SignalDirection,
    validate_finite,
    validate_timezone_aware,
    MarketBar,
)


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Standardized output from an individual strategy evaluation."""
    strategy_id: str
    direction: SignalDirection
    raw_score: float             # Intensity bounded in [-1.0, 1.0], 0.0 is neutral
    confidence: float            # Heuristic conviction in [0.0, 1.0]
    timestamp: datetime
    expected_gross_return: float | None = None  # Expected price return decimal fraction
    volatility_estimate: float | None = None    # Estimated volatility (standard deviation of returns)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
            raise ContractValidationError("StrategySignal 'strategy_id' cannot be blank")

        if not isinstance(self.direction, SignalDirection):
            raise ContractValidationError(f"Invalid SignalDirection: {self.direction}")

        validate_finite(self.raw_score, "raw_score")
        if not (-1.0 <= self.raw_score <= 1.0):
            raise ContractValidationError(f"raw_score must be in [-1.0, 1.0], got {self.raw_score}")

        validate_finite(self.confidence, "confidence")
        if not (0.0 <= self.confidence <= 1.0):
            raise ContractValidationError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        validate_timezone_aware(self.timestamp, "timestamp")

        if self.expected_gross_return is not None:
            validate_finite(self.expected_gross_return, "expected_gross_return")

        if self.volatility_estimate is not None:
            validate_finite(self.volatility_estimate, "volatility_estimate")
            if self.volatility_estimate < 0.0:
                raise ContractValidationError("volatility_estimate cannot be negative")


class BaseStrategy(ABC):
    """Abstract base class for all quantitative strategies."""

    @abstractmethod
    def evaluate(self, bars: Sequence[MarketBar]) -> StrategySignal:
        """Evaluate market bars up to latest timestamp causally.

        Must NOT access future bars.
        Must handle warm-up and data quality anomalies gracefully.
        """
        pass

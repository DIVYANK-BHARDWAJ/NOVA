"""NOVA Portfolio Utility & Capital Allocation (Person 3 Exclusive Domain).

Implements volatility-adjusted position sizing, partial exit management,
and capital recycling into high-utility trade opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from nova.quant.contracts import ContractValidationError, validate_finite


@dataclass(frozen=True, slots=True)
class PortfolioAllocationConfig:
    """Configurable sizing and allocation limits."""
    max_capital_per_position: float = 0.25      # 25% max position allocation
    risk_target_volatility: float = 0.02        # 2% volatility target
    partial_exit_ratio: float = 0.50            # 50% scale-out ratio on profit realization
    min_order_value: float = 100.0              # Minimum currency order size
    config_version: str = "1.0.0"

    def __post_init__(self) -> None:
        for name, val in [
            ("max_capital_per_position", self.max_capital_per_position),
            ("risk_target_volatility", self.risk_target_volatility),
            ("partial_exit_ratio", self.partial_exit_ratio),
            ("min_order_value", self.min_order_value),
        ]:
            validate_finite(val, name)
            if val <= 0.0:
                raise ContractValidationError(f"Allocation parameter '{name}' must be positive")


class PortfolioAllocator:
    """Calculates risk-adjusted capital allocations and handles capital recycling."""

    def __init__(self, config: PortfolioAllocationConfig | None = None) -> None:
        self.config = config or PortfolioAllocationConfig()

    def calculate_target_exposure(
        self,
        signal_confidence: float,
        asset_volatility: float | None,
        available_cash_fraction: float,
    ) -> float:
        """Volatility-parity position sizing with confidence scaling."""
        validate_finite(signal_confidence, "signal_confidence")
        validate_finite(available_cash_fraction, "available_cash_fraction")

        if asset_volatility is None or asset_volatility < 1e-4:
            base_size = self.config.max_capital_per_position
        else:
            # Scale inversely with volatility
            vol_scalar = self.config.risk_target_volatility / asset_volatility
            base_size = min(self.config.max_capital_per_position, vol_scalar * self.config.max_capital_per_position)

        confidence_adjusted = base_size * max(0.2, min(1.0, signal_confidence))
        # Cash constrained
        capped_by_cash = min(confidence_adjusted, max(0.0, available_cash_fraction))
        return min(self.config.max_capital_per_position, capped_by_cash)

    def calculate_partial_exit_exposure(self, current_exposure: float) -> float:
        """Compute remaining exposure after a partial profit-taking exit."""
        validate_finite(current_exposure, "current_exposure")
        return current_exposure * (1.0 - self.config.partial_exit_ratio)

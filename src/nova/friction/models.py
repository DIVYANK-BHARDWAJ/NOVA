"""NOVA Friction & Execution Cost Models (Person 3 Exclusive Domain).

Implements realistic market friction: bid-ask spread, slippage, brokerage commissions,
taxes (STT), market impact, break-even movement calculation, and cost-clearing probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from nova.quant.contracts import ContractValidationError, NumericalValidationError, validate_finite


@dataclass(frozen=True, slots=True)
class FrictionParameters:
    """Configurable friction parameters in basis points (1 bp = 0.0001 = 0.01%)."""
    spread_bps: float = 2.0       # Half-spread or crossing spread
    slippage_bps: float = 3.0     # Expected execution slippage
    brokerage_bps: float = 1.0    # Broker commission / exchange fees
    tax_bps: float = 1.0          # Regulatory / turnover tax (e.g., STT)
    impact_coefficient: float = 0.0 # Market impact coefficient (bps per unit relative volume)

    def __post_init__(self) -> None:
        for name, val in [
            ("spread_bps", self.spread_bps),
            ("slippage_bps", self.slippage_bps),
            ("brokerage_bps", self.brokerage_bps),
            ("tax_bps", self.tax_bps),
            ("impact_coefficient", self.impact_coefficient),
        ]:
            validate_finite(val, name)
            if val < 0.0:
                raise NumericalValidationError(f"Friction parameter '{name}' cannot be negative, got {val}")


class MarketFrictionEngine:
    """Calculates all-in execution costs, break-even price movement, and probability of clearing friction."""

    def __init__(self, params: FrictionParameters | None = None) -> None:
        self.params = params or FrictionParameters()

    @property
    def one_way_friction_fraction(self) -> float:
        """Total one-way friction as a dimensionless decimal return fraction."""
        total_bps = (
            self.params.spread_bps
            + self.params.slippage_bps
            + self.params.brokerage_bps
            + self.params.tax_bps
        )
        return total_bps / 10000.0

    @property
    def roundtrip_friction_fraction(self) -> float:
        """Total roundtrip (entry + exit) friction as a dimensionless decimal return fraction."""
        return self.one_way_friction_fraction * 2.0

    def calculate_all_in_cost(
        self,
        is_roundtrip: bool = True,
        order_size_relative: float = 0.0,
    ) -> float:
        """Estimate total cost decimal fraction, including volume market impact."""
        base_fraction = self.roundtrip_friction_fraction if is_roundtrip else self.one_way_friction_fraction
        impact_bps = self.params.impact_coefficient * max(0.0, order_size_relative)
        return base_fraction + (impact_bps / 10000.0)

    def calculate_break_even_movement(
        self,
        entry_price: float,
        is_roundtrip: bool = True,
        order_size_relative: float = 0.0,
    ) -> float:
        """Calculate the absolute price movement required above entry price to break even after friction."""
        if entry_price <= 0.0:
            raise NumericalValidationError(f"Entry price must be positive, got {entry_price}")
        cost_frac = self.calculate_all_in_cost(is_roundtrip=is_roundtrip, order_size_relative=order_size_relative)
        return entry_price * cost_frac

    def estimate_probability_of_clearing_costs(
        self,
        expected_gross_return: float,
        all_in_cost: float,
        volatility: float | None = None,
    ) -> float:
        """Estimate P(Gross return > All-in cost).

        Uses Gaussian CDF if volatility is provided; otherwise uses a calibrated logistic sigmoid.
        Returns a probability bounded strictly in [0.0, 1.0].
        """
        validate_finite(expected_gross_return, "expected_gross_return")
        validate_finite(all_in_cost, "all_in_cost")

        net_edge = expected_gross_return - all_in_cost

        if volatility is not None and volatility > 1e-6:
            validate_finite(volatility, "volatility")
            # Standardized z-score for exceeding cost
            z = net_edge / volatility
            # Gaussian standard normal CDF: 0.5 * (1 + erf(z / sqrt(2)))
            prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            return max(0.0, min(1.0, prob))

        # Heuristic sigmoid if volatility is unavailable (steepness tuned so edge=0 gives 50%)
        # scaled by typical daily return scale ~ 100 bps (0.01)
        scale = 0.005
        z = net_edge / scale
        prob = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, prob))

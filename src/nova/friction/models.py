"""NOVA Friction & Execution Cost Models (Person 3 Exclusive Domain).

Implements realistic market friction: bid-ask spread, slippage, brokerage commissions,
taxes (STT), market impact, break-even movement calculation, and cost-clearing probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from nova.decision.contracts import ActionType
from nova.quant.contracts import ContractValidationError, NumericalValidationError, validate_finite


@dataclass(frozen=True, slots=True)
class FrictionParameters:
    """Configurable friction parameters: basis points and flat exchange/depository fees."""
    spread_bps: float = 2.0           # Half-spread or crossing spread (bps)
    slippage_bps: float = 3.0         # Expected execution slippage (bps)
    brokerage_bps: float = 1.0        # Proportional broker commission / exchange fees (bps)
    tax_bps: float = 1.0              # Regulatory / turnover tax (e.g., STT bps)
    impact_coefficient: float = 0.0   # Market impact coefficient (bps per unit relative volume)
    flat_brokerage_inr: float = 20.0  # Flat ₹20 per executed order (MANTHAN 2026 rule)
    flat_dp_charges_inr: float = 15.34 # Flat ₹15.34 DP charge on sales of owned shares (MANTHAN 2026 rule)
    stcg_tax_rate: float = 0.15       # Short-term capital gains tax rate (15% for <= 365 days)
    ltcg_tax_rate: float = 0.10       # Long-term capital gains tax rate (10% for > 365 days)
    ltcg_exemption_inr: float = 100000.0 # First ₹1,00,000 LTCG is tax-exempt per financial year

    def __post_init__(self) -> None:
        for name, val in [
            ("spread_bps", self.spread_bps),
            ("slippage_bps", self.slippage_bps),
            ("brokerage_bps", self.brokerage_bps),
            ("tax_bps", self.tax_bps),
            ("impact_coefficient", self.impact_coefficient),
            ("flat_brokerage_inr", self.flat_brokerage_inr),
            ("flat_dp_charges_inr", self.flat_dp_charges_inr),
            ("stcg_tax_rate", self.stcg_tax_rate),
            ("ltcg_tax_rate", self.ltcg_tax_rate),
            ("ltcg_exemption_inr", self.ltcg_exemption_inr),
        ]:
            validate_finite(val, name)
            if val < 0.0:
                raise NumericalValidationError(f"Friction parameter '{name}' cannot be negative, got {val}")


class MarketFrictionEngine:
    """Calculates all-in execution costs, flat/DP fees, break-even price movement, and clearing probabilities."""

    def __init__(self, params: FrictionParameters | None = None) -> None:
        self.params = params or FrictionParameters()

    def calculate_order_fee_inr(
        self,
        action: ActionType | str,
        notional: float,
    ) -> float:
        """Calculate total fee in INR for an individual order.

        MANTHAN 2026 fee rules:
        - Flat ₹20 brokerage on every executed order (BUY, SELL, SHORT, COVER).
        - Flat ₹15.34 DP charges on sales of owned shares (SELL only).
        - Proportional turnover taxes / STT and brokerage in bps.
        """
        validate_finite(notional, "notional")
        action_str = action.value if isinstance(action, ActionType) else str(action).upper()

        fee = self.params.flat_brokerage_inr
        if action_str == "SELL":
            fee += self.params.flat_dp_charges_inr

        # Proportional tax / fee component
        proportional_bps = self.params.brokerage_bps + self.params.tax_bps
        proportional_fee = notional * (proportional_bps / 10000.0)
        return fee + proportional_fee

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

    def calculate_break_even_price(
        self,
        entry_price: float,
        quantity: float = 1.0,
        is_short: bool = False,
        order_size_relative: float = 0.0,
    ) -> float:
        """Calculate the exact price at which an exit order breaks even after all costs.

        Includes flat ₹20 brokerage per order, ₹15.34 DP charges (for long positions),
        and proportional spread, slippage, and taxes.
        """
        if entry_price <= 0.0:
            raise NumericalValidationError(f"Entry price must be positive, got {entry_price}")
        if quantity <= 0.0:
            raise NumericalValidationError(f"Quantity must be positive, got {quantity}")

        # Flat fees: Entry ₹20 + Exit ₹20 (+ ₹15.34 if Long)
        flat_fee_roundtrip = (self.params.flat_brokerage_inr * 2.0) + (0.0 if is_short else self.params.flat_dp_charges_inr)
        flat_fee_per_share = flat_fee_roundtrip / quantity

        # Proportional friction: spread, slippage, taxes, impact
        prop_fraction = self.calculate_all_in_cost(is_roundtrip=True, order_size_relative=order_size_relative)
        prop_move_per_share = entry_price * prop_fraction

        total_move_required = flat_fee_per_share + prop_move_per_share

        if is_short:
            return max(0.01, entry_price - total_move_required)
        return entry_price + total_move_required

    def calculate_net_profit_and_costs(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        is_short: bool = False,
        holding_days: int = 0,
    ) -> dict[str, float]:
        """Compute complete trade breakdown: gross PnL, friction fees, estimated tax drag, net PnL."""
        validate_finite(entry_price, "entry_price")
        validate_finite(exit_price, "exit_price")
        validate_finite(quantity, "quantity")

        entry_notional = entry_price * quantity
        exit_notional = exit_price * quantity

        # Gross PnL
        if is_short:
            gross_pnl = (entry_price - exit_price) * quantity
        else:
            gross_pnl = (exit_price - entry_price) * quantity

        entry_fee = self.calculate_order_fee_inr("SHORT" if is_short else "BUY", entry_notional)
        exit_fee = self.calculate_order_fee_inr("COVER" if is_short else "SELL", exit_notional)

        # Proportional slippage & spread
        spread_slip_bps = self.params.spread_bps + self.params.slippage_bps
        slippage_cost = (entry_notional + exit_notional) * (spread_slip_bps / 10000.0)

        total_friction = entry_fee + exit_fee + slippage_cost
        pnl_after_friction = gross_pnl - total_friction

        # Estimated tax drag for decision evaluation
        tax_rate = self.params.ltcg_tax_rate if holding_days > 365 else self.params.stcg_tax_rate
        estimated_tax = max(0.0, pnl_after_friction * tax_rate)

        net_pnl = pnl_after_friction - estimated_tax

        return {
            "gross_pnl": gross_pnl,
            "total_friction": total_friction,
            "brokerage_fees": entry_fee + exit_fee,
            "slippage_and_spread": slippage_cost,
            "pnl_after_friction": pnl_after_friction,
            "estimated_tax_drag": estimated_tax,
            "net_pnl": net_pnl,
        }

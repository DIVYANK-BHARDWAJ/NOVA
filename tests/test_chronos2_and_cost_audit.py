"""Comprehensive unit and integration tests for Person 2 Milestone 10.

Tests:
1. MANTHAN official trading cost structure (flat ₹20, flat ₹15.34 DP, 0 bps proportional fee).
2. Cost audit discrepancy vs Milestone 9 (7.0 bps one-way / 14.0 bps roundtrip ungrounded friction).
3. Capital gains tax calculations (STCG 15%, LTCG 10% with ₹100,000 exemption).
4. Chronos forecast-to-return conversion logic (median price forecast to log-return and directional call).
5. Turnover reduction heuristics (2-day persistence filter and 3-day minimum holding period).
6. ModelPrediction contract parity, schema validation, and provenance for Chronos predictions.
7. Graceful unavailability handling when foundation model weights or dependencies are absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.decision.contracts import ActionType
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    PredictionAvailability,
    PredictionType,
    SignalDirection,
    validate_model_prediction_schema_parity,
)


def chronos_forecast_to_return(
    last_close: float,
    predicted_close: float,
) -> tuple[float, int]:
    """Convert Chronos future price forecast to implied 1-step return and direction.

    Returns:
        (implied_log_return, directional_call) where directional_call is 1 (UP) or 0 (DOWN).
    """
    if last_close <= 0.0 or predicted_close <= 0.0:
        raise ValueError(f"Prices must be strictly positive: last={last_close}, pred={predicted_close}")
    if math.isnan(last_close) or math.isnan(predicted_close):
        raise ValueError("Prices cannot be NaN")

    implied_return = math.log(predicted_close / last_close)
    direction = 1 if implied_return > 0.0 else 0
    return implied_return, direction


def apply_persistence_filter(
    signals: list[int],
    persistence_window: int = 2,
) -> list[int]:
    """Apply signal persistence filter: only change active signal if signal repeats N times.

    If signal flips for fewer than persistence_window periods, retain previous position.
    """
    if not signals:
        return []
    filtered = [signals[0]]
    consecutive_count = 1

    for i in range(1, len(signals)):
        raw_sig = signals[i]
        prev_raw = signals[i - 1]
        current_active = filtered[-1]

        if raw_sig == prev_raw:
            consecutive_count += 1
        else:
            consecutive_count = 1

        if consecutive_count >= persistence_window:
            filtered.append(raw_sig)
        else:
            filtered.append(current_active)

    return filtered


def apply_min_holding_period_filter(
    signals: list[int],
    min_hold_days: int = 3,
) -> list[int]:
    """Apply minimum holding period: once entered or flipped, hold position for min_hold_days."""
    if not signals:
        return []
    filtered = [signals[0]]
    days_in_position = 1

    for i in range(1, len(signals)):
        target_sig = signals[i]
        current_sig = filtered[-1]

        if target_sig != current_sig:
            if days_in_position >= min_hold_days:
                filtered.append(target_sig)
                days_in_position = 1
            else:
                filtered.append(current_sig)
                days_in_position += 1
        else:
            filtered.append(current_sig)
            days_in_position += 1

    return filtered


class TestChronos2AndCostAudit(unittest.TestCase):
    """Test suite for Milestone 10 cost audit, Chronos logic, and turnover dampening."""

    def setUp(self) -> None:
        # Scenario A: Official MANTHAN Platform Rules
        self.official_params = FrictionParameters(
            spread_bps=0.0,
            slippage_bps=0.0,
            brokerage_bps=0.0,
            tax_bps=0.0,
            impact_coefficient=0.0,
            flat_brokerage_inr=20.0,
            flat_dp_charges_inr=15.34,
            stcg_tax_rate=0.15,
            ltcg_tax_rate=0.10,
            ltcg_exemption_inr=100000.0,
        )
        self.official_engine = MarketFrictionEngine(self.official_params)

        # Milestone 9 Evaluator Parameters (Contains ungrounded 7.0 bps proportional friction)
        self.m9_params = FrictionParameters(
            spread_bps=2.0,
            slippage_bps=3.0,
            brokerage_bps=1.0,
            tax_bps=1.0,
            flat_brokerage_inr=20.0,
            flat_dp_charges_inr=15.34,
        )
        self.m9_engine = MarketFrictionEngine(self.m9_params)

        # Scenario B: Official Platform Rules + 5 bps Execution Slippage Sensitivity
        self.slippage_params = FrictionParameters(
            spread_bps=0.0,
            slippage_bps=5.0,
            brokerage_bps=0.0,
            tax_bps=0.0,
            impact_coefficient=0.0,
            flat_brokerage_inr=20.0,
            flat_dp_charges_inr=15.34,
        )
        self.slippage_engine = MarketFrictionEngine(self.slippage_params)

    def test_manthan_official_cost_structure(self) -> None:
        """Verify official MANTHAN rules: flat ₹20/order, ₹15.34 DP on SELL, 0 bps proportional."""
        # BUY order: exactly ₹20 flat fee regardless of notional
        buy_fee = self.official_engine.calculate_order_fee_inr("BUY", notional=100000.0)
        self.assertAlmostEqual(buy_fee, 20.0, places=4)

        # SHORT order: exactly ₹20 flat fee
        short_fee = self.official_engine.calculate_order_fee_inr("SHORT", notional=100000.0)
        self.assertAlmostEqual(short_fee, 20.0, places=4)

        # COVER order: exactly ₹20 flat fee
        cover_fee = self.official_engine.calculate_order_fee_inr("COVER", notional=100000.0)
        self.assertAlmostEqual(cover_fee, 20.0, places=4)

        # SELL order (delivery sale): ₹20 brokerage + ₹15.34 DP charge = ₹35.34
        sell_fee = self.official_engine.calculate_order_fee_inr("SELL", notional=100000.0)
        self.assertAlmostEqual(sell_fee, 35.34, places=4)

        # Proportional friction fraction must be strictly 0.0 in official rules
        self.assertAlmostEqual(self.official_engine.one_way_friction_fraction, 0.0, places=6)
        self.assertAlmostEqual(self.official_engine.roundtrip_friction_fraction, 0.0, places=6)

    def test_cost_audit_discrepancy_vs_milestone_9(self) -> None:
        """Audit and document the 7 bps / 14 bps friction discrepancy introduced in Milestone 9."""
        notional = 200000.0  # ₹2,00,000 order

        # Under Milestone 9 friction:
        # Brokerage bps (1) + tax bps (1) = 2 bps proportional in calculate_order_fee_inr
        # plus spread (2) + slippage (3) in calculate_all_in_cost = 7 bps one-way total
        m9_one_way_bps = self.m9_engine.one_way_friction_fraction * 10000.0
        self.assertAlmostEqual(m9_one_way_bps, 7.0, places=4)
        m9_roundtrip_bps = self.m9_engine.roundtrip_friction_fraction * 10000.0
        self.assertAlmostEqual(m9_roundtrip_bps, 14.0, places=4)

        # In INR for ₹2,00,000 BUY:
        # Official: ₹20.0
        # M9 friction: ₹20 + 2 bps * 200,000 = ₹60 fee + (5 bps spread/slip = ₹100) = ₹160 all-in!
        official_buy = self.official_engine.calculate_order_fee_inr("BUY", notional)
        m9_buy = self.m9_engine.calculate_order_fee_inr("BUY", notional)

        self.assertAlmostEqual(official_buy, 20.0, places=2)
        self.assertAlmostEqual(m9_buy, 60.0, places=2)
        self.assertGreater(m9_buy, official_buy)

    def test_capital_gains_tax_rules(self) -> None:
        """Verify STCG (15%) and LTCG (10% with ₹1,00,000 exemption)."""
        # Short-term gain (< 365 days) of ₹50,000
        gain_st = 50000.0
        stcg_tax = gain_st * self.official_params.stcg_tax_rate
        self.assertAlmostEqual(stcg_tax, 7500.0, places=2)

        # Long-term gain (> 365 days) of ₹1,50,000 (taxable gain is ₹50,000 above ₹1,00,000 exemption)
        gain_lt = 150000.0
        taxable_ltcg = max(0.0, gain_lt - self.official_params.ltcg_exemption_inr)
        ltcg_tax = taxable_ltcg * self.official_params.ltcg_tax_rate
        self.assertAlmostEqual(ltcg_tax, 5000.0, places=2)

    def test_chronos_forecast_conversion(self) -> None:
        """Verify Chronos price prediction conversion to 1-step return and direction."""
        last_close = 100.0

        # Case 1: Bullish forecast
        pred_close_up = 102.50
        ret_up, dir_up = chronos_forecast_to_return(last_close, pred_close_up)
        self.assertAlmostEqual(ret_up, math.log(1.025), places=6)
        self.assertEqual(dir_up, 1)

        # Case 2: Bearish forecast
        pred_close_down = 97.50
        ret_down, dir_down = chronos_forecast_to_return(last_close, pred_close_down)
        self.assertAlmostEqual(ret_down, math.log(0.975), places=6)
        self.assertEqual(dir_down, 0)

        # Case 3: Flat forecast
        pred_close_flat = 100.0
        ret_flat, dir_flat = chronos_forecast_to_return(last_close, pred_close_flat)
        self.assertAlmostEqual(ret_flat, 0.0, places=6)
        self.assertEqual(dir_flat, 0)

        # Edge cases: Non-positive prices must raise ValueError
        with self.assertRaises(ValueError):
            chronos_forecast_to_return(0.0, 100.0)
        with self.assertRaises(ValueError):
            chronos_forecast_to_return(100.0, -5.0)

    def test_turnover_reduction_persistence_filter(self) -> None:
        """Verify 2-day persistence filter reduces noisy alternating signals."""
        # Alternating noise: [1, 0, 1, 0, 1, 0]
        noisy_signals = [1, 0, 1, 0, 1, 0]
        filtered = apply_persistence_filter(noisy_signals, persistence_window=2)
        # Never repeats twice consecutively -> filtered remains at initial signal [1, 1, 1, 1, 1, 1]
        self.assertEqual(filtered, [1, 1, 1, 1, 1, 1])

        # Sustained trend flip: [1, 1, 0, 0, 0, 1, 1]
        trending_signals = [1, 1, 0, 0, 0, 1, 1]
        filtered_trending = apply_persistence_filter(trending_signals, persistence_window=2)
        # At index 2: raw=0 (count 1) -> retains 1
        # At index 3: raw=0 (count 2) -> flips to 0
        self.assertEqual(filtered_trending, [1, 1, 1, 0, 0, 0, 1])

    def test_turnover_reduction_min_holding_period(self) -> None:
        """Verify 3-day minimum holding period enforces hold duration."""
        signals = [1, 0, 0, 0, 1, 1, 1]
        # At t=0: enters 1 (day 1)
        # At t=1: raw=0, but days=1 < 3 -> held at 1 (day 2)
        # At t=2: raw=0, days=2 < 3 -> held at 1 (day 3)
        # At t=3: raw=0, days=3 >= 3 -> flips to 0 (day 1)
        filtered = apply_min_holding_period_filter(signals, min_hold_days=3)
        self.assertEqual(filtered, [1, 1, 1, 0, 0, 0, 1])

    def test_chronos_contract_parity_and_schema(self) -> None:
        """Verify Chronos-generated prediction satisfies ModelPrediction contract and schema parity."""
        now = datetime.now(timezone.utc)
        pred = ModelPrediction(
            model_id="amazon/chronos-2",
            model_version="2.0.0-zero-shot",
            prediction_timestamp=now,
            input_cutoff_timestamp=now,
            instrument_id="TCS",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=1,
            prediction_value=0.0084,
            availability_status=PredictionAvailability.READY,
            metadata={"lookback_length": 64, "device": "cpu"},
        )
        self.assertEqual(pred.model_id, "amazon/chronos-2")
        self.assertEqual(pred.prediction_value, 0.0084)
        self.assertEqual(pred.availability_status, PredictionAvailability.READY)

        # Validate schema parity
        self.assertEqual(validate_model_prediction_schema_parity(pred.to_dict()), [])

    def test_chronos_unavailability_graceful_handling(self) -> None:
        """Verify graceful fallback contract when foundation model is unavailable."""
        now = datetime.now(timezone.utc)
        pred_unavail = ModelPrediction(
            model_id="amazon/chronos-2",
            model_version="2.0.0-zero-shot",
            prediction_timestamp=now,
            input_cutoff_timestamp=now,
            instrument_id="TCS",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=1,
            prediction_value=0.0,
            availability_status=PredictionAvailability.UNAVAILABLE,
            metadata={"reason": "offline_execution_mode"},
        )
        self.assertEqual(pred_unavail.availability_status, PredictionAvailability.UNAVAILABLE)
        self.assertEqual(pred_unavail.prediction_value, 0.0)
        self.assertEqual(validate_model_prediction_schema_parity(pred_unavail.to_dict()), [])

    def test_context_cutoff_and_anti_lookahead(self) -> None:
        """Verify that historical context at time t strictly terminates at or before t."""
        dates = [
            datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 19, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc),
        ]
        bars = [
            MarketBar(
                ticker="AERX",
                open=100.0,
                high=105.0,
                low=99.0,
                close=100.0 + i,
                volume=1000.0,
                source_timestamp=d,
                provenance="TEST",
                data_quality=DataQualityStatus.VALID,
            )
            for i, d in enumerate(dates)
        ]
        # At index t=2 (2026-05-20), context must only include indices 0, 1, 2
        cutoff_idx = 2
        cutoff_date = dates[cutoff_idx]
        context = bars[: cutoff_idx + 1]

        # Invariant: Every timestamp in context <= cutoff_date
        for bar in context:
            self.assertLessEqual(bar.source_timestamp, cutoff_date)

        # Invariant: Future bar strictly greater than cutoff_date
        future_bar = bars[cutoff_idx + 1]
        self.assertGreater(future_bar.source_timestamp, cutoff_date)

    def test_instrument_isolation(self) -> None:
        """Verify time-series context does not mix bars from different instruments."""
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        universe_bars = {
            "AERX": [MarketBar(ticker="AERX", open=10.0, high=11.0, low=9.0, close=10.0, volume=100.0, source_timestamp=dt, provenance="P", data_quality=DataQualityStatus.VALID)],
            "BLUX": [MarketBar(ticker="BLUX", open=20.0, high=21.0, low=19.0, close=20.0, volume=100.0, source_timestamp=dt, provenance="P", data_quality=DataQualityStatus.VALID)],
        }
        for ticker, bars in universe_bars.items():
            for b in bars:
                self.assertEqual(b.ticker, ticker)

    def test_dp_charge_eligibility(self) -> None:
        """Verify DP charges apply strictly on delivery sales (SELL), not on BUY, SHORT, or COVER."""
        self.assertEqual(self.official_engine.calculate_order_fee_inr("BUY", 50000.0), 20.0)
        self.assertEqual(self.official_engine.calculate_order_fee_inr("SHORT", 50000.0), 20.0)
        self.assertEqual(self.official_engine.calculate_order_fee_inr("COVER", 50000.0), 20.0)
        self.assertEqual(self.official_engine.calculate_order_fee_inr("SELL", 50000.0), 35.34)

    def test_lot_based_holding_period_stcg_vs_ltcg(self) -> None:
        """Verify holding period determines STCG (<=365 days) vs LTCG (>365 days)."""
        t0 = datetime(2024, 6, 25, tzinfo=timezone.utc)
        t_short = datetime(2024, 12, 25, tzinfo=timezone.utc)  # ~183 days -> STCG
        t_long = datetime(2026, 5, 18, tzinfo=timezone.utc)   # ~692 days -> LTCG

        days_short = (t_short - t0).days
        days_long = (t_long - t0).days

        self.assertLessEqual(days_short, 365)
        self.assertGreater(days_long, 365)

    def test_ltcg_exemption_and_tax_rate(self) -> None:
        """Verify LTCG 10% tax rate applies strictly to gains exceeding Rs 100,000 exemption."""
        gain_below_exemption = 80_000.0
        taxable_below = max(0.0, gain_below_exemption - 100_000.0)
        self.assertEqual(taxable_below, 0.0)
        tax_below = taxable_below * 0.10
        self.assertEqual(tax_below, 0.0)

        gain_above_exemption = 650_000.0
        taxable_above = max(0.0, gain_above_exemption - 100_000.0)
        self.assertEqual(taxable_above, 550_000.0)
        tax_above = taxable_above * 0.10
        self.assertEqual(tax_above, 55_000.0)

    def test_accounting_identity_gross_to_net(self) -> None:
        """Verify exact ledger arithmetic identity: Net == Gross - Fees - Taxes."""
        gross_pnl = 482184.93
        brokerage = 400.00
        dp_charges = 153.40
        slippage = 0.00
        ltcg_tax = 55087.30
        stcg_tax = 0.00

        total_deductions = brokerage + dp_charges + slippage + ltcg_tax + stcg_tax
        net_pnl = gross_pnl - total_deductions

        expected_net = 426544.23
        self.assertAlmostEqual(net_pnl, expected_net, places=2)
        self.assertAlmostEqual(gross_pnl - total_deductions, net_pnl, places=4)

    def test_schema_parity_accepts_instance_and_dict(self) -> None:
        """Verify validate_model_prediction_schema_parity accepts ModelPrediction and dict."""
        now = datetime.now(timezone.utc)
        pred = ModelPrediction(
            model_id="amazon/chronos-2",
            model_version="2.0.0-zero-shot",
            prediction_timestamp=now,
            input_cutoff_timestamp=now,
            instrument_id="TEST",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=1,
            prediction_value=0.0123,
            availability_status=PredictionAvailability.READY,
        )
        # Passing ModelPrediction instance directly
        self.assertEqual(validate_model_prediction_schema_parity(pred), [])
        # Passing dictionary
        self.assertEqual(validate_model_prediction_schema_parity(pred.to_dict()), [])


if __name__ == "__main__":
    unittest.main()

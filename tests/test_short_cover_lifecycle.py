"""Comprehensive verification matrix for NOVA SHORT & COVER complete lifecycle.

Validates:
1. Configuration consistency and gating (allow_short=True in MANTHAN).
2. Decision engine side-aware expected return economics and short/cover candidate logic.
3. Deterministic risk gate margin collateral, brokerage charges, circuit breaker, and COVER safety.
4. Execution adapter action mapping and signed-position order sizing.
5. Account ledger 100% margin locking, proportional release on cover, and available cash accounting.
6. Profit realization stop-loss (4%), trailing stop (3% from trough), and 10-day settlement auto-closure.
"""

from datetime import datetime, timedelta, timezone
import math
from typing import Any, cast
import unittest

from execution.accounting import (
    AccountLedger,
    reconcile_cash,
    reconcile_positions,
)
from execution.adapter import ExecutionAdapter
from execution.config import ExecutionConfig
from execution.models import (
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Portfolio,
    PortfolioHolding,
    SimulationStage,
)
from nova.decision.contracts import (
    ActionType,
    OrderType,
    PendingOrder,
    PortfolioState,
    PositionState,
    TradeDecision,
)
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.pipeline import IntegratedTradingPipeline
from nova.quant.contracts import (
    CandidateSignal,
    DataQualityStatus,
    MarketBar,
    SignalDirection,
    TimestampSemantics,
)
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration


def create_synthetic_bars(
    prices: list[float],
    ticker: str = "SYNTH",
    quality: DataQualityStatus = DataQualityStatus.VALID,
    base_time: datetime | None = None,
) -> list[MarketBar]:
    """Helper to generate chronological sequence of synthetic bars."""
    start = base_time or datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars: list[MarketBar] = []
    for i, p in enumerate(prices):
        t = start + timedelta(minutes=5 * i)
        bars.append(
            MarketBar(
                ticker=ticker,
                open=p * 0.999,
                high=p * 1.002,
                low=p * 0.998,
                close=p,
                volume=5000.0,
                source_timestamp=t,
                available_timestamp=t,
                timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                data_quality=quality,
            )
        )
    return bars


class MockTradingClient:
    """Mock platform API client for testing execution adapter and reconciliation."""

    def __init__(self, stage: str = "competition", status: str = "running") -> None:
        self.stage = stage
        self.status = status
        self.submitted_orders: list[dict] = []

    def get_stage(self) -> SimulationStage:
        return SimulationStage(
            stage=self.stage,
            status=self.status,
            tickers=["SYNTH", "MOTR"],
            ticker_count=2,
            current_date="2026-01-01",
            days_completed=1,
            total_days=100,
        )

    def get_stocks(self) -> list:
        from execution.models import StockInfo
        return [
            StockInfo(symbol="SYNTH", ticker="SYNTH", name="Synthetic Corp"),
            StockInfo(symbol="MOTR", ticker="MOTR", name="Motor Corp"),
        ]

    def place_order(self, ticker: str, action: str, quantity: int, date: str | None = None) -> dict:
        self.submitted_orders.append({"ticker": ticker, "action": action, "quantity": quantity, "date": date})
        return {
            "order_id": f"ord_{len(self.submitted_orders)}",
            "status": "FILLED",
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "filled_quantity": quantity,
            "execution_price": 100.0,
            "charges": 20.0,
        }


class TestShortCoverLifecycle(unittest.TestCase):
    """Exhaustive test suite covering all 40 verification scenarios."""

    def setUp(self) -> None:
        self.now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        self.client = MockTradingClient()
        self.exec_config = ExecutionConfig(
            bot_access_token="mock_token",
            permitted_stages=("competition",),
            stage_cache_ttl_seconds=0,
        )
        self.ledger = AccountLedger(initial_cash=1_000_000.0)
        self.adapter = ExecutionAdapter(client=cast(Any, self.client), config=self.exec_config, ledger=self.ledger)

    # =========================================================================
    # Group 1: Configuration (Tests 1 - 5)
    # =========================================================================

    def test_01_risk_config_disallow_short_blocks_short_entry(self) -> None:
        """1. RiskConfiguration(allow_short=False) blocks SHORT entry."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=False))
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.NO_TRADE)
        self.assertIn("SHORT_POSITIONS_PROHIBITED_BY_POLICY", res.risk_reason_codes)

    def test_02_risk_config_allow_short_permits_short_entry(self) -> None:
        """2. RiskConfiguration(allow_short=True) permits SHORT entry when checks pass."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertTrue(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.SHORT)
        self.assertAlmostEqual(res.authorized_exposure, -0.20)

    def test_03_decision_engine_disallow_short_prevents_flat_short_generation(self) -> None:
        """3. DecisionEngineConfig(allow_short=False) prevents short candidate generation from flat state."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=False),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=False)),
        )
        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)
        self.assertNotEqual(decision.selected_action, ActionType.SHORT)
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)

    def test_04_decision_engine_allow_short_generates_short_candidate(self) -> None:
        """4. DecisionEngineConfig(allow_short=True) generates short candidate on valid bearish opportunity."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.0,
            available_cash=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.SHORT)
        self.assertLess(decision.target_quantity_or_exposure, 0.0)

    def test_05_configuration_mismatch_fails_or_reconciles_safely(self) -> None:
        """5. Incompatible configuration between risk and decision components fails safely."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        # Decision engine allows short, but risk gate prohibits short
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=False)),
        )
        decision = engine.evaluate_decision(bars, current_position_exposure=0.0)
        # Risk gate must veto the decision safely to NO_TRADE
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)
        self.assertTrue(any("SHORT_POSITIONS_PROHIBITED_BY_POLICY" in r for r in decision.selection_or_rejection_reasons))

    # =========================================================================
    # Group 2: Decision Logic (Tests 6 - 19)
    # =========================================================================

    def test_06_flat_portfolio_strong_bearish_produces_short(self) -> None:
        """6. Flat portfolio + strong bearish signal -> SHORT."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.0,
            available_cash=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.SHORT)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, -0.20)

    def test_07_flat_portfolio_weak_bearish_produces_no_trade(self) -> None:
        """7. Flat portfolio + weak bearish signal -> NO_TRADE."""
        # Mild drift that does not clear score hurdle / net friction
        prices = [100.0 - (i * 0.01) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.0,
            available_cash=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.NO_TRADE)

    def test_08_flat_portfolio_strong_bullish_produces_buy(self) -> None:
        """8. Flat portfolio + strong bullish signal -> BUY."""
        prices = [50.0 + (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.0,
            available_cash=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.BUY)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.20)

    def test_09_long_portfolio_bearish_produces_sell_exit(self) -> None:
        """9. Long portfolio + bearish signal -> SELL / EXIT."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=100.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=3,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.20,
            position_state=pos,
            available_cash=800_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertIn(decision.selected_action, (ActionType.SELL, ActionType.EXIT))
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)

    def test_10_short_portfolio_stronger_bearish_increases_short_or_holds(self) -> None:
        """10. Short portfolio + stronger bearish signal -> SHORT (increase) or HOLD if in rebalance band."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True, target_exposure=0.20, rebalance_band_fraction=0.03),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        # Small current short position (-0.05), target is -0.20
        pos = PositionState(
            ticker="SYNTH",
            quantity=-25.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=1,
            locked_margin=2500.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.05,
            position_state=pos,
            available_cash=900_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.SHORT)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, -0.20)

    def test_11_short_portfolio_target_exposure_closer_to_zero_produces_cover(self) -> None:
        """11. Short portfolio + target exposure closer to zero -> COVER."""
        prices = [100.0 - (i * 0.1) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        # Current exposure is heavily short (-0.25), target is -0.10
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True, target_exposure=0.10),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=-250.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=2,
            locked_margin=25000.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.25,
            position_state=pos,
            available_cash=750_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, -0.10)

    def test_12_short_portfolio_bullish_reversal_produces_cover(self) -> None:
        """12. Short portfolio + bullish reversal -> COVER."""
        prices = [50.0 + (i * 2.0) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=60.0,
            current_market_price=bars[-1].close,
            holding_days=2,
            locked_margin=6000.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.20,
            position_state=pos,
            available_cash=900_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)

    def test_13_short_portfolio_neutral_signal_produces_hold(self) -> None:
        """13. Short portfolio + neutral signal -> HOLD."""
        prices = [100.0 + (math.sin(i) * 0.2) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=2,
            locked_margin=10000.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.20,
            position_state=pos,
            available_cash=900_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.HOLD)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, -0.20)

    def test_14_short_portfolio_stop_loss_hit_produces_cover(self) -> None:
        """14. Short portfolio + stop-loss hit (price rises >= 4%) -> COVER."""
        # Entry price 100.0, current price rises to 105.0 (+5% rise on short)
        prices = [100.0 + (i * 0.15) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True, stop_loss_pct=0.04),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True, stop_loss_pct=0.04)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,  # ~105.85
            holding_days=2,
            locked_margin=10000.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.20,
            position_state=pos,
            available_cash=900_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)

    def test_15_short_portfolio_trailing_stop_hit_produces_cover(self) -> None:
        """15. Short portfolio + trailing-stop hit (3% rise from trough) -> COVER."""
        # Short entered at 100.0. Price dropped to 90.0 (trough, +10% gain for short, activates trailing stop)
        # Now price rises to 93.0 (+3.3% rise from trough, exceeds 3% trailing stop distance)
        prices = [90.0 + (i * 0.1) for i in range(35)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=100.0,
            current_market_price=93.0,
            holding_days=4,
            locked_margin=10000.0,
            lowest_price=90.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.20,
            position_state=pos,
            available_cash=900_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)

    def test_16_short_portfolio_10_day_settlement_produces_cover(self) -> None:
        """16. Short portfolio + 10-day settlement due -> COVER."""
        prices = [100.0 - (i * 0.05) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True, max_short_holding_days=10),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True, max_short_holding_days=10)),
        )
        pos = PositionState(
            ticker="SYNTH",
            quantity=-100.0,
            entry_price=100.0,
            current_market_price=bars[-1].close,
            holding_days=10,  # Exactly 10 days reached
            locked_margin=10000.0,
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=-0.20,
            position_state=pos,
            available_cash=900_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.COVER)
        self.assertAlmostEqual(decision.target_quantity_or_exposure, 0.0)

    def test_17_short_expected_net_return_is_positive_on_negative_underlying(self) -> None:
        """17. Expected net return on SHORT is positive when underlying asset return is negative."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.0,
            available_cash=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.SHORT)
        # Expected gross and net return must be strictly positive for the short position
        self.assertIsNotNone(decision.expected_gross_return)
        self.assertIsNotNone(decision.expected_net_return)
        self.assertGreater(float(decision.expected_gross_return or 0.0), 0.0)
        self.assertGreater(float(decision.expected_net_return or 0.0), 0.0)

    def test_18_short_probability_of_clearing_costs_reflects_short_distribution(self) -> None:
        """18. Reported probability of clearing cost for SHORT reflects short trade's return distribution."""
        prices = [100.0 - (i * 1.5) for i in range(40)]
        bars = create_synthetic_bars(prices, base_time=self.now)
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        decision = engine.evaluate_decision(
            bars,
            current_position_exposure=0.0,
            available_cash=1_000_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertEqual(decision.selected_action, ActionType.SHORT)
        self.assertIsNotNone(decision.probability_of_clearing_costs)
        self.assertGreater(float(decision.probability_of_clearing_costs or 0.0), 0.50)

    def test_19_short_break_even_price_is_lower_than_current(self) -> None:
        """19. Break-even calculation for SHORT is lower than current entry price."""
        engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(allow_short=True),
            risk_gate=DeterministicRiskGate(RiskConfiguration(allow_short=True)),
        )
        be_short = engine.friction_engine.calculate_break_even_price(entry_price=1000.0, quantity=50, is_short=True)
        self.assertLess(be_short, 1000.0)

    # =========================================================================
    # Group 3: Risk Gate (Tests 20 - 29)
    # =========================================================================

    def test_20_short_rejected_insufficient_cash_for_margin_plus_brokerage(self) -> None:
        """20. SHORT rejected when available cash is less than required 100% margin plus flat brokerage."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True, order_brokerage=20.0))
        # Proposed 20% of 1,000,000 = 200,000 margin + 20 brokerage = 200,020 required
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            available_cash=200_010.0,  # ₹10 short of 200,020
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("INSUFFICIENT_MARGIN_CASH" in r for r in res.risk_reason_codes))

    def test_21_short_clamped_or_rejected_on_position_exposure_limit(self) -> None:
        """21. SHORT clamped when proposed exposure exceeds configured maximum position limit."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True, max_position_exposure=0.25))
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.40,  # Exceeds 25% max
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertTrue(res.is_approved)
        self.assertAlmostEqual(res.authorized_exposure, -0.25)
        self.assertTrue(any("EXPOSURE_CLAMPED" in r for r in res.risk_reason_codes))

    def test_22_short_rejected_on_drawdown_circuit_breaker(self) -> None:
        """22. SHORT rejected when account drawdown exceeds circuit breaker limit."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True, max_drawdown_limit=0.15))
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            current_drawdown=0.16,  # 16% drawdown > 15% limit
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(res.circuit_breaker_active)

    def test_23_short_rejected_on_stale_market_data(self) -> None:
        """23. SHORT rejected when market data is stale."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True, max_staleness_seconds=300.0))
        stale_bar_time = self.now - timedelta(seconds=600)  # 10 minutes old
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            bar_timestamp=stale_bar_time,
            as_of_time=self.now,
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertFalse(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("MARKET_DATA_STALE" in r for r in res.risk_reason_codes))

    def test_24_short_rejected_on_invalid_data_quality(self) -> None:
        """24. SHORT rejected when data quality is invalid."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            data_quality=DataQualityStatus.INVALID,
            bar_timestamp=self.now,
            as_of_time=self.now,
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertFalse(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.NO_TRADE)
        self.assertTrue(any("DATA_QUALITY_UNACCEPTABLE" in r for r in res.risk_reason_codes))

    def test_25_short_rejected_when_conflicting_order_pending(self) -> None:
        """25. SHORT rejected when conflicting order is already pending."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        res = gate.evaluate(
            proposed_action=ActionType.SHORT,
            proposed_exposure=-0.20,
            has_pending_entry_order=True,
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.HOLD)
        self.assertTrue(any("EXECUTION_SAFETY_CONFLICTING_ORDER_PENDING" in r for r in res.risk_reason_codes))

    def test_26_cover_accepted_even_if_allow_short_is_false(self) -> None:
        """26. COVER accepted when valid negative position exists, even if allow_short=False."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=False))
        res = gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=0.0,
            current_position_quantity=-100.0,
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertTrue(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.COVER)

    def test_27_cover_rejected_when_position_is_flat_or_positive(self) -> None:
        """27. COVER rejected when position is flat or positive."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        # Flat position
        res_flat = gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=0.0,
            current_position_quantity=0.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res_flat.is_approved)
        self.assertTrue(any("EXECUTION_SAFETY_CANNOT_COVER_WITHOUT_SHORT_POSITION" in r for r in res_flat.risk_reason_codes))

        # Long position
        res_long = gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=0.0,
            current_position_quantity=100.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res_long.is_approved)
        self.assertTrue(any("EXECUTION_SAFETY_CANNOT_COVER_WITHOUT_SHORT_POSITION" in r for r in res_long.risk_reason_codes))

    def test_28_cover_rejected_when_all_shares_pending_cover(self) -> None:
        """28. COVER rejected when all short shares are already committed to pending cover orders."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        res = gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=0.0,
            current_position_quantity=-50.0,
            pending_exit_quantity=50.0,  # All 50 shares pending
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertFalse(res.is_approved)
        self.assertTrue(any("EXECUTION_SAFETY_ALL_SHORT_SHARES_PENDING_COVER" in r for r in res.risk_reason_codes))

    def test_29_partial_cover_authorized_without_exceeding_position(self) -> None:
        """29. Partial COVER authorized correctly without exceeding position size."""
        gate = DeterministicRiskGate(RiskConfiguration(allow_short=True))
        res = gate.evaluate(
            proposed_action=ActionType.COVER,
            proposed_exposure=-0.10,  # Target leaves -0.10 exposure (partial cover)
            current_position_quantity=-200.0,
            available_cash=500_000.0,
            current_portfolio_value=1_000_000.0,
            bar_timestamp=self.now,
            as_of_time=self.now,
        )
        self.assertTrue(res.is_approved)
        self.assertEqual(res.authorized_action, ActionType.COVER)
        self.assertAlmostEqual(res.authorized_exposure, -0.10)

    # =========================================================================
    # Group 4: Execution & Accounting (Tests 30 - 40)
    # =========================================================================

    def test_30_adapter_maps_short_action(self) -> None:
        """30. ActionType.SHORT maps to OrderAction.SHORT in ExecutionAdapter."""
        dec = TradeDecision(
            decision_id="dec_short_1",
            decision_timestamp=self.now,
            instrument_id="SYNTH",
            selected_action=ActionType.SHORT,
            target_quantity_or_exposure=-0.20,
            order_type=OrderType.MARKET,
            selected_strategy="test",
            selection_or_rejection_reasons=("MANTHAN_SHORT",),
            configuration_version="1.0",
        )
        res = self.adapter.execute_trade_decision(
            decision=dec,
            current_price=100.0,
            current_portfolio_value=1_000_000.0,
            current_position_shares=0,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.action, OrderAction.SHORT)
        self.assertEqual(res.quantity, 2000)  # 200,000 / 100 = 2000 shares

    def test_31_adapter_maps_cover_action(self) -> None:
        """31. ActionType.COVER maps to OrderAction.COVER in ExecutionAdapter."""
        dec = TradeDecision(
            decision_id="dec_cover_1",
            decision_timestamp=self.now,
            instrument_id="SYNTH",
            selected_action=ActionType.COVER,
            target_quantity_or_exposure=0.0,
            order_type=OrderType.MARKET,
            selected_strategy="test",
            selection_or_rejection_reasons=("MANTHAN_COVER",),
            configuration_version="1.0",
        )
        res = self.adapter.execute_trade_decision(
            decision=dec,
            current_price=100.0,
            current_portfolio_value=1_000_000.0,
            current_position_shares=-500,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.action, OrderAction.COVER)
        self.assertEqual(res.quantity, 500)

    def test_32_short_fill_locks_100pct_margin_collateral(self) -> None:
        """32. SHORT fill locks exactly 100% margin collateral."""
        ledger = AccountLedger(initial_cash=500_000.0)
        # Short 100 shares @ 200.0 = 20,000 trade value, 20.0 charges
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_evt_1",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        self.assertEqual(ledger.get_position("SYNTH"), -100)
        self.assertEqual(ledger.positions["SYNTH"].locked_margin, 20_000.0)
        self.assertEqual(ledger.total_locked_margin, 20_000.0)

    def test_33_available_cash_decreases_by_margin_and_charges(self) -> None:
        """33. Available cash decreases by margin plus brokerage charges on SHORT fill."""
        ledger = AccountLedger(initial_cash=500_000.0)
        # Short 100 shares @ 200.0 = 20,000 trade value, 20.0 charges
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_evt_2",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        # Total cash reflects short proceeds: 500,000 + 20,000 - 20 = 519,980
        self.assertEqual(ledger.get_cash(), 519_980.0)
        # Spendable available cash excludes locked margin: 519,980 - 20,000 = 499,980
        self.assertEqual(ledger.available_cash, 499_980.0)

    def test_34_portfolio_wealth_accounts_for_short_unrealized_pnl(self) -> None:
        """34. Total portfolio wealth properly accounts for short unrealized P&L."""
        ledger = AccountLedger(initial_cash=500_000.0)
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_evt_3",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        # Price drops from 200 to 180: short gains +20/sh * 100 = +2000
        val_gain = ledger.get_portfolio_value({"SYNTH": 180.0})
        # 519,980 + (-100 * 180) = 519,980 - 18,000 = 501,980 (initial 500k - 20 fee + 2000 pnl)
        self.assertEqual(val_gain, 501_980.0)

        # Price rises from 200 to 220: short loses -20/sh * 100 = -2000
        val_loss = ledger.get_portfolio_value({"SYNTH": 220.0})
        # 519,980 + (-100 * 220) = 519,980 - 22,000 = 497,980 (initial 500k - 20 fee - 2000 pnl)
        self.assertEqual(val_loss, 497_980.0)

    def test_35_partial_cover_releases_margin_proportionally(self) -> None:
        """35. COVER fill releases margin proportionally for partial fill."""
        ledger = AccountLedger(initial_cash=500_000.0)
        # Short 100 shares @ 200.0 (locked margin = 20,000)
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_evt_4",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        # Partial cover: 50 shares @ 190.0 (50% cover, releases 10,000 margin)
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="cover_evt_1",
                ticker="SYNTH",
                action=OrderAction.COVER,
                quantity=50,
                status=OrderStatus.FILLED,
                filled_quantity=50,
                execution_price=190.0,
                charges=20.0,
            )
        )
        self.assertEqual(ledger.get_position("SYNTH"), -50)
        self.assertEqual(ledger.positions["SYNTH"].locked_margin, 10_000.0)
        self.assertEqual(ledger.total_locked_margin, 10_000.0)

    def test_36_full_cover_releases_all_locked_margin(self) -> None:
        """36. Full COVER fill releases 100% of remaining locked margin."""
        ledger = AccountLedger(initial_cash=500_000.0)
        # Short 100 shares @ 200.0
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_evt_5",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        # Full cover: 100 shares @ 190.0
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="cover_evt_2",
                ticker="SYNTH",
                action=OrderAction.COVER,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=190.0,
                charges=20.0,
            )
        )
        self.assertEqual(ledger.get_position("SYNTH"), 0)
        self.assertEqual(ledger.total_locked_margin, 0.0)
        self.assertEqual(ledger.available_cash, ledger.get_cash())

    def test_37_profitable_cover_increases_realized_pnl(self) -> None:
        """37. Realized P&L on profitable cover increases cash and total wealth."""
        ledger = AccountLedger(initial_cash=100_000.0)
        # Short 100 @ 200.0, charges 20.0
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_pnl_1",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        # Cover 100 @ 180.0, charges 20.0
        # PnL = (200 - 180) * 100 - 20 = 2000 - 20 = +1980
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="cover_pnl_1",
                ticker="SYNTH",
                action=OrderAction.COVER,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=180.0,
                charges=20.0,
            )
        )
        self.assertEqual(ledger.total_realized_pnl, 1980.0)
        self.assertEqual(ledger.get_cash(), 100_000.0 + 1980.0 - 20.0)  # minus short entry fee

    def test_38_losing_cover_records_loss_and_deducts_brokerage(self) -> None:
        """38. Realized P&L on losing cover correctly records loss and deducts brokerage charges."""
        ledger = AccountLedger(initial_cash=100_000.0)
        # Short 100 @ 200.0, charges 20.0
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="short_pnl_2",
                ticker="SYNTH",
                action=OrderAction.SHORT,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=200.0,
                charges=20.0,
            )
        )
        # Cover 100 @ 220.0, charges 20.0
        # PnL = (200 - 220) * 100 - 20 = -2000 - 20 = -2020
        ledger.process_execution_event(
            OrderResult(
                client_intent_id="cover_pnl_2",
                ticker="SYNTH",
                action=OrderAction.COVER,
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100,
                execution_price=220.0,
                charges=20.0,
            )
        )
        self.assertEqual(ledger.total_realized_pnl, -2020.0)
        self.assertEqual(ledger.get_cash(), 100_000.0 - 2020.0 - 20.0)

    def test_39_position_reconciliation_syncs_short_holdings(self) -> None:
        """39. Position reconciliation correctly syncs short holdings from remote response."""
        # 1. Signed negative remote quantity
        remote_port_signed = Portfolio(
            team_id="team_1",
            team_name="Team 1",
            available_cash=800_000.0,
            cash_balance=1_000_000.0,
            portfolio_market_value=-20_000.0,
            total_wealth=980_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=40.0,
            total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="SYNTH", quantity=-100)],
        )
        local_positions = {"SYNTH": -100}
        report_signed = reconcile_positions(local_positions, remote_port_signed)
        self.assertTrue(report_signed.is_synchronized)
        self.assertEqual(report_signed.matched, {"SYNTH": -100})

        # 2. Tagged side='SHORT' with positive quantity
        remote_port_tagged = Portfolio(
            team_id="team_1",
            team_name="Team 1",
            available_cash=800_000.0,
            cash_balance=1_000_000.0,
            portfolio_market_value=-20_000.0,
            total_wealth=980_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            total_charges=40.0,
            total_taxes=0.0,
            holdings=[PortfolioHolding(ticker="SYNTH", quantity=100)],  # positive qty, side='SHORT' attribute
        )
        object.__setattr__(remote_port_tagged.holdings[0], "side", "SHORT")
        report_tagged = reconcile_positions(local_positions, remote_port_tagged)
        self.assertTrue(report_tagged.is_synchronized)
        self.assertEqual(report_tagged.matched, {"SYNTH": -100})

    def test_40_duplicate_execution_event_is_idempotent(self) -> None:
        """40. Duplicate execution event does not double-lock margin or double-count realized P&L."""
        ledger = AccountLedger(initial_cash=500_000.0)
        short_evt = OrderResult(
            client_intent_id="short_idem_1",
            ticker="SYNTH",
            action=OrderAction.SHORT,
            quantity=100,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            execution_price=200.0,
            charges=20.0,
        )
        first_proc = ledger.process_execution_event(short_evt)
        self.assertTrue(first_proc)
        self.assertEqual(ledger.total_locked_margin, 20_000.0)
        self.assertEqual(ledger.get_cash(), 519_980.0)

        # Replay identical short fill event
        second_proc = ledger.process_execution_event(short_evt)
        self.assertFalse(second_proc)
        self.assertEqual(ledger.total_locked_margin, 20_000.0)
        self.assertEqual(ledger.get_cash(), 519_980.0)
        self.assertEqual(ledger.get_position("SYNTH"), -100)


if __name__ == "__main__":
    unittest.main()

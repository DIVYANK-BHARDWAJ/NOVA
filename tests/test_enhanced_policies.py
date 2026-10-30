"""NOVA Enhanced Policies & Quantitative Resilience Tests (Person 3 Exclusive Domain).

Tests the enhanced policies designed to eliminate drawdowns and maximize realized profits:
1. Ambiguous Contradiction Veto (Trend vs Mean Reversion strong disagreement).
2. Falling Knife Trend Filter (Close < EMA50 blocks buy).
3. Market Breadth Gatekeeper (% universe > EMA20).
4. 5-Day Anti-Churn Post-Stop Quarantine.
5. 3-Tier Profit Preservation Ratchet (Tier 1 Break-Even, Tier 2 50% Lock, Tier 3 Tight Trail).
6. Tiered Resuscitation Ladder (Zone 1 Caution, Zone 2 Recovery, Zone 3 Sniper vs. Permanent Coma).
7. Asymmetric Defensive Short Protocol (BEAR regime gating, 50% sizing, 4-day hold, 2.5% stop, +4% profit target).
8. Volatility-Parity Risk Sizing (Risk / 2x ATR14).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pytest

from nova.decision.contracts import ActionType, PortfolioState, PositionState, TradeDecision
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.portfolio.profit_realization import (
    ExitRecommendation,
    ProfitRealizationConfig,
    ProfitRealizationManager,
)
from nova.quant.contracts import DataQualityStatus, MarketBar, SignalDirection, TimestampSemantics
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.base import StrategySignal


def make_bar(
    ticker: str = "TICKER_A",
    close: float = 1000.0,
    days_ago: int = 0,
    volume: float = 500000.0,
    high: float | None = None,
    low: float | None = None,
    open_px: float | None = None,
) -> MarketBar:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ts = base + timedelta(days=days_ago)
    op = open_px if open_px is not None else close - 2.0
    hi = high if high is not None else close + 5.0
    lo = low if low is not None else close - 5.0
    return MarketBar(
        ticker=ticker,
        open=op,
        high=hi,
        low=lo,
        close=close,
        volume=volume,
        source_timestamp=ts,
        available_timestamp=ts,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=DataQualityStatus.VALID,
    )


# ============================================================================
# 1. Ambiguity Handling: Contradiction Veto Tests
# ============================================================================

def test_contradiction_veto_suppresses_conflicting_speculative_entry():
    """When trend and mean reversion strongly contradict (e.g. +0.70 vs -0.50), veto entry."""
    cfg = DecisionEngineConfig(
        enable_contradiction_veto=True,
        contradiction_threshold=0.35,
        target_exposure=0.15,
    )
    engine = SentinelDecisionEngine(config=cfg)

    # Mock strategies to simulate a severe contradiction
    bars = [make_bar(days_ago=i, close=1000.0 + i) for i in range(30)]

    from unittest.mock import MagicMock
    engine.trend_strategy.evaluate = MagicMock(return_value=StrategySignal(
        strategy_id="trend",
        direction=SignalDirection.BULLISH,
        raw_score=0.75,
        confidence=0.8,
        timestamp=bars[-1].available_timestamp,
        expected_gross_return=0.015,
        volatility_estimate=0.012,
        reason_codes=("STRONG_UPTREND",),
    ))
    engine.mean_reversion_strategy.evaluate = MagicMock(return_value=StrategySignal(
        strategy_id="mean_rev",
        direction=SignalDirection.BEARISH,
        raw_score=-0.65,
        confidence=0.8,
        timestamp=bars[-1].available_timestamp,
        expected_gross_return=-0.012,
        volatility_estimate=0.012,
        reason_codes=("OVERBOUGHT_REVERSION",),
    ))

    dec = engine.evaluate_decision(bars)
    assert dec.selected_action == ActionType.NO_TRADE
    assert any("CONTRADICTION_VETO" in r for r in dec.selection_or_rejection_reasons)


# ============================================================================
# 2. Falling Knife Veto Tests (Close < EMA50)
# ============================================================================

def test_falling_knife_filter_blocks_buy_when_below_ema50():
    """Buying is vetoed when price is below its 50-day EMA, even with positive signals."""
    cfg = DecisionEngineConfig(
        enable_trend_filter=True,
        trend_filter_ema_period=50,
        enable_contradiction_veto=False,
    )
    engine = SentinelDecisionEngine(config=cfg)

    # Construct 60 bars in a steep downtrend from 1500 down to 800 (well below EMA50)
    bars = [make_bar(days_ago=i, close=1500.0 - (i * 12.0)) for i in range(60)]

    # Mock signals to propose a BUY
    from unittest.mock import MagicMock
    engine.trend_strategy.evaluate = MagicMock(return_value=StrategySignal(
        strategy_id="trend",
        direction=SignalDirection.BULLISH,
        raw_score=0.60,
        confidence=0.8,
        timestamp=bars[-1].available_timestamp,
        expected_gross_return=0.02,
        volatility_estimate=0.01,
        reason_codes=("BULLISH",),
    ))
    engine.mean_reversion_strategy.evaluate = MagicMock(return_value=StrategySignal(
        strategy_id="mean_rev",
        direction=SignalDirection.BULLISH,
        raw_score=0.60,
        confidence=0.8,
        timestamp=bars[-1].available_timestamp,
        expected_gross_return=0.02,
        volatility_estimate=0.01,
        reason_codes=("OVERSOLD_BOUNCE",),
    ))

    dec = engine.evaluate_decision(bars)
    assert dec.selected_action == ActionType.NO_TRADE
    assert any("FALLING_KNIFE_VETO" in r for r in dec.selection_or_rejection_reasons)


# ============================================================================
# 3. Market Breadth Gatekeeper Tests
# ============================================================================

def test_market_breadth_blocks_buy_during_market_collapse():
    """When market breadth is below threshold (e.g. 25% < 40%), all buys are blocked."""
    gate = DeterministicRiskGate(RiskConfiguration(
        market_breadth_buy_threshold=0.40,
        market_breadth_short_threshold=0.30,
    ))
    now_utc = datetime.now(timezone.utc)

    # 1. Low breadth (0.25) -> BUY rejected
    res_buy = gate.evaluate(
        proposed_action=ActionType.BUY,
        proposed_exposure=0.15,
        market_breadth=0.25,
        bar_timestamp=now_utc,
    )
    assert not res_buy.is_approved
    assert res_buy.authorized_action == ActionType.NO_TRADE
    assert any("REJECT_MARKET_BREADTH_INSUFFICIENT" in r for r in res_buy.risk_reason_codes)

    # 2. Healthy breadth (0.65) -> BUY approved
    res_healthy = gate.evaluate(
        proposed_action=ActionType.BUY,
        proposed_exposure=0.15,
        market_breadth=0.65,
        bar_timestamp=now_utc,
    )
    assert res_healthy.is_approved
    assert res_healthy.authorized_action == ActionType.BUY


def test_market_breadth_blocks_short_in_bull_market():
    """When market breadth is high (e.g. 70% > 30%), shorting is blocked."""
    gate = DeterministicRiskGate(RiskConfiguration(
        enable_asymmetric_short=True,
        market_breadth_short_threshold=0.30,
    ))
    now_utc = datetime.now(timezone.utc)

    res_short = gate.evaluate(
        proposed_action=ActionType.SHORT,
        proposed_exposure=-0.10,
        market_breadth=0.70,
        market_regime="BEAR_STRONG",
        bar_timestamp=now_utc,
    )
    assert not res_short.is_approved
    assert res_short.authorized_action == ActionType.NO_TRADE
    assert any("REJECT_MARKET_BREADTH_TOO_HIGH_FOR_SHORT" in r for r in res_short.risk_reason_codes)


# ============================================================================
# 4. Anti-Churn Post-Stop Quarantine Tests
# ============================================================================

def test_ticker_post_stop_quarantine_enforced():
    """A ticker that hit stop loss cannot be re-bought within the 5-day quarantine."""
    gate = DeterministicRiskGate(RiskConfiguration(cooldown_days_after_stop=5))
    stop_time = datetime(2025, 3, 10, 10, 0, tzinfo=timezone.utc)

    # Register stop loss exit for ALPHACORP
    gate.register_stop_loss_exit("ALPHACORP", stop_time)

    # Day 2 after stop -> REJECT
    day2 = stop_time + timedelta(days=2)
    res_day2 = gate.evaluate(
        proposed_action=ActionType.BUY,
        proposed_exposure=0.10,
        bar_timestamp=day2,
        ticker="ALPHACORP",
    )
    assert not res_day2.is_approved
    assert res_day2.authorized_action == ActionType.NO_TRADE
    assert any("REJECT_TICKER_IN_POST_STOP_QUARANTINE_ALPHACORP" in r for r in res_day2.risk_reason_codes)

    # Different ticker (BETACORP) on day 2 -> APPROVED
    res_beta = gate.evaluate(
        proposed_action=ActionType.BUY,
        proposed_exposure=0.10,
        bar_timestamp=day2,
        ticker="BETACORP",
    )
    assert res_beta.is_approved

    # Day 6 after stop -> APPROVED (quarantine expired)
    day6 = stop_time + timedelta(days=6)
    res_day6 = gate.evaluate(
        proposed_action=ActionType.BUY,
        proposed_exposure=0.10,
        bar_timestamp=day6,
        ticker="ALPHACORP",
    )
    assert res_day6.is_approved


# ============================================================================
# 5. 3-Tier Profit Preservation Ratchet Tests (Unrealized -> Realized)
# ============================================================================

def test_tier1_breakeven_ratchet():
    """At +2% gain, stop moves to entry + 0.3% friction floor; drops below trigger SELL."""
    mgr = ProfitRealizationManager(ProfitRealizationConfig(
        tier1_breakeven_gain_pct=0.02,
        tier1_friction_cushion_pct=0.003,
    ))
    # Entry at 1000.0, current price drops to 1002.0 (below 1003.0 break-even floor)
    # Peak was 1025.0 (+2.5% gain, triggering Tier 1)
    pos = PositionState(
        ticker="ABC",
        quantity=50,
        entry_price=1000.0,
        current_market_price=1002.0,
        highest_price=1025.0,
        unrealized_pnl=100.0,
    )
    bar = make_bar(close=1002.0)
    rec = mgr.evaluate_position_exit(pos, bar, current_exposure=0.10)
    assert rec.should_exit
    assert rec.recommended_action == ActionType.SELL
    assert any("TIER1_BREAKEVEN_PROTECTION_EXIT" in r for r in rec.reasons)


def test_tier2_lock_50pct_partial_exit():
    """At +4% gain, locks 50% scale-out into cash."""
    mgr = ProfitRealizationManager(ProfitRealizationConfig(
        tier2_lock_gain_pct=0.04,
        partial_exit_ratio=0.50,
    ))
    pos = PositionState(
        ticker="ABC",
        quantity=100,
        entry_price=1000.0,
        current_market_price=1045.0,  # +4.5% gain
        highest_price=1045.0,
        unrealized_pnl=4500.0,
        has_taken_partial_profit=False,
    )
    bar = make_bar(close=1045.0)
    rec = mgr.evaluate_position_exit(pos, bar, current_exposure=0.10)
    assert rec.should_exit
    assert rec.is_partial
    assert rec.recommended_action == ActionType.SELL
    assert rec.target_exposure == pytest.approx(0.05)
    assert any("TIER2_PROFIT_LOCK_50PCT_SCALE_OUT" in r for r in rec.reasons)


def test_tier3_tight_trailing_stop():
    """At +7% gain, trailing stop tightens to 1.2% distance."""
    mgr = ProfitRealizationManager(ProfitRealizationConfig(
        tier3_tighten_gain_pct=0.07,
        tier3_trailing_distance_pct=0.012,
    ))
    # Entry 1000, peak was 1090 (+9%), current is 1075 (dropped 1.37% from peak)
    pos = PositionState(
        ticker="ABC",
        quantity=50,
        entry_price=1000.0,
        current_market_price=1075.0,
        highest_price=1090.0,
        unrealized_pnl=3750.0,
    )
    bar = make_bar(close=1075.0)
    rec = mgr.evaluate_position_exit(pos, bar, current_exposure=0.10)
    assert rec.should_exit
    assert rec.recommended_action == ActionType.SELL
    assert any("TIER3_TIGHT_TRAILING_STOP_TRIGGERED" in r for r in rec.reasons)


# ============================================================================
# 6. Tiered Resuscitation Ladder Tests (No Permanent Coma)
# ============================================================================

def test_tiered_resuscitation_ladder_zones():
    """Drawdown scales position sizes without permanently freezing in a coma."""
    gate = DeterministicRiskGate(RiskConfiguration(
        caution_drawdown_threshold=0.04,
        recovery_drawdown_threshold=0.07,
        sniper_drawdown_threshold=0.10,
        max_drawdown_limit=0.15,
        max_position_exposure=0.20,
    ))
    now = datetime.now(timezone.utc)

    # Zone 0 (Normal: 2% DD) -> 100% size (0.20)
    r0 = gate.evaluate(ActionType.BUY, proposed_exposure=0.20, current_drawdown=0.02, bar_timestamp=now)
    assert r0.is_approved
    assert r0.authorized_exposure == pytest.approx(0.20)

    # Zone 1 (Caution: 5% DD) -> 70% size (0.14)
    r1 = gate.evaluate(ActionType.BUY, proposed_exposure=0.20, current_drawdown=0.05, bar_timestamp=now)
    assert r1.is_approved
    assert r1.authorized_exposure == pytest.approx(0.14)
    assert any("TIERED_RESUSCITATION_ZONE_1_CAUTION" in r for r in r1.risk_reason_codes)

    # Zone 2 (Recovery: 8% DD) -> 50% size (0.10)
    r2 = gate.evaluate(ActionType.BUY, proposed_exposure=0.20, current_drawdown=0.08, bar_timestamp=now)
    assert r2.is_approved
    assert r2.authorized_exposure == pytest.approx(0.10)
    assert any("TIERED_RESUSCITATION_ZONE_2_RECOVERY" in r for r in r2.risk_reason_codes)

    # Zone 3 (Sniper: 12% DD) -> 20% size (0.04) -> STILL APPROVED, NOT A COMA!
    r3 = gate.evaluate(ActionType.BUY, proposed_exposure=0.20, current_drawdown=0.12, bar_timestamp=now)
    assert r3.is_approved
    assert r3.authorized_exposure == pytest.approx(0.04)
    assert any("TIERED_RESUSCITATION_ZONE_3_SNIPER" in r for r in r3.risk_reason_codes)

    # Emergency Kill Switch (16% DD >= 15%) -> REJECTED
    r4 = gate.evaluate(ActionType.BUY, proposed_exposure=0.20, current_drawdown=0.16, bar_timestamp=now)
    assert not r4.is_approved
    assert r4.circuit_breaker_active


# ============================================================================
# 7. Asymmetric Short Protocol Tests
# ============================================================================

def test_asymmetric_short_protocol_gating():
    """Short is only authorized in BEAR regimes and scales to 50% size."""
    gate = DeterministicRiskGate(RiskConfiguration(
        enable_asymmetric_short=True,
        short_exposure_scale=0.50,
        max_position_exposure=0.20,
    ))
    now = datetime.now(timezone.utc)

    # 1. Short in BULL regime -> REJECTED
    r_bull = gate.evaluate(
        proposed_action=ActionType.SHORT,
        proposed_exposure=-0.15,
        market_regime="BULL_STRONG",
        bar_timestamp=now,
    )
    assert not r_bull.is_approved
    assert any("REJECT_ASYMMETRIC_SHORT_REQUIRES_BEAR_REGIME" in r for r in r_bull.risk_reason_codes)

    # 2. Short in BEAR regime -> APPROVED with 50% sizing
    r_bear = gate.evaluate(
        proposed_action=ActionType.SHORT,
        proposed_exposure=-0.15,
        market_regime="BEAR_STRONG",
        bar_timestamp=now,
    )
    assert r_bear.is_approved
    assert r_bear.authorized_exposure == pytest.approx(-0.075)  # 0.15 * 0.50
    assert any("ASYMMETRIC_SHORT_SCALED" in r for r in r_bear.risk_reason_codes)


def test_asymmetric_short_defensive_exits():
    """Defensive short covers at 4 days, or at +4% profit target, or at 2.5% stop loss."""
    mgr = ProfitRealizationManager(ProfitRealizationConfig(
        short_max_defensive_days=4,
        short_profit_target_pct=0.04,
        short_stop_loss_pct=0.025,
    ))

    # 1. 4-day time exit
    pos_time = PositionState(
        ticker="SHORT_A",
        quantity=-50,
        entry_price=1000.0,
        current_market_price=990.0,
        holding_days=4,
    )
    r_time = mgr.evaluate_position_exit(pos_time, make_bar(close=990.0), current_exposure=-0.08)
    assert r_time.should_exit
    assert r_time.recommended_action == ActionType.COVER
    assert any("DEFENSIVE_SHORT_TIME_EXIT" in r for r in r_time.reasons)

    # 2. +4.5% profit target cover
    pos_profit = PositionState(
        ticker="SHORT_B",
        quantity=-50,
        entry_price=1000.0,
        current_market_price=950.0,  # 5% gain for short
        holding_days=2,
    )
    r_profit = mgr.evaluate_position_exit(pos_profit, make_bar(close=950.0), current_exposure=-0.08)
    assert r_profit.should_exit
    assert r_profit.recommended_action == ActionType.COVER
    assert any("DEFENSIVE_SHORT_PROFIT_TARGET_REALIZED" in r for r in r_profit.reasons)

    # 3. 2.5% stop loss cover
    pos_stop = PositionState(
        ticker="SHORT_C",
        quantity=-50,
        entry_price=1000.0,
        current_market_price=1028.0,  # 2.8% loss for short
        holding_days=1,
    )
    r_stop = mgr.evaluate_position_exit(pos_stop, make_bar(close=1028.0), current_exposure=-0.08)
    assert r_stop.should_exit
    assert r_stop.recommended_action == ActionType.COVER
    assert any("STOP_LOSS_SHORT_TRIGGERED" in r for r in r_stop.reasons)


# ============================================================================
# 8. Volatility-Parity Risk Sizing Tests
# ============================================================================

def test_volatility_parity_risk_sizing():
    """Higher volatility results in proportionally smaller position sizing."""
    mgr = ProfitRealizationManager()
    equity = 1_000_000.0  # ₹10,00,000

    # Low volatility: ATR = ₹10 on ₹1000 stock (1% daily vol)
    # Stop distance = 2x ATR = ₹20 (2% stop)
    # Target risk = 0.75% = ₹7,500
    # Exposure = 0.75% / 2.0% = 37.5% -> capped at max 15%
    exp_low_vol = mgr.calculate_volatility_parity_exposure(
        current_price=1000.0,
        atr_14=10.0,
        portfolio_equity=equity,
        target_risk_fraction=0.0075,
        max_exposure_fraction=0.15,
    )
    assert exp_low_vol == 0.15

    # High volatility: ATR = ₹50 on ₹1000 stock (5% daily vol)
    # Stop distance = 2x ATR = ₹100 (10% stop)
    # Exposure = 0.75% / 10% = 7.5%
    exp_high_vol = mgr.calculate_volatility_parity_exposure(
        current_price=1000.0,
        atr_14=50.0,
        portfolio_equity=equity,
        target_risk_fraction=0.0075,
        max_exposure_fraction=0.15,
    )
    assert exp_high_vol == pytest.approx(0.075)

    # Short position on high volatility: 50% scaling of 7.5% = 3.75%
    exp_short = mgr.calculate_volatility_parity_exposure(
        current_price=1000.0,
        atr_14=50.0,
        portfolio_equity=equity,
        target_risk_fraction=0.0075,
        is_short=True,
        max_exposure_fraction=0.15,
    )
    assert exp_short == pytest.approx(0.0375)

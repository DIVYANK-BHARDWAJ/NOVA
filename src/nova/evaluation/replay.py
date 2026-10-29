"""NOVA Chronological Historical Replay Engine (Person 3 Exclusive Domain).

Simulates trading execution causally bar-by-bar:
- Signals generated at bar t close strictly using data <= t
- Orders executed at bar t+1 open
- Realistic friction (spread, slippage, brokerage) deducted on every portfolio rebalance
- Zero look-ahead bias and zero future-data leakage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Mapping, Sequence

from nova.decision.contracts import ActionType, TradeDecision
from nova.decision.engine import SentinelDecisionEngine
from nova.friction.models import MarketFrictionEngine
from nova.quant.contracts import (
    ContractValidationError,
    MarketBar,
    validate_finite,
    validate_timezone_aware,
)
from nova.research.adapters import validate_market_bar_sequence


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """Detailed record of a single chronological replay step."""
    step_index: int
    decision_timestamp: datetime
    execution_timestamp: datetime | None
    close_price: float
    next_open_price: float | None
    action: ActionType
    target_exposure: float
    turnover: float
    step_gross_return: float
    step_friction_cost: float
    step_net_return: float
    portfolio_equity: float
    current_drawdown: float


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Machine-readable and printable summary of chronological replay results."""
    strategy_label: str
    ticker: str
    total_input_bars: int
    evaluated_steps: int
    action_counts: Mapping[str, int]
    total_turnover: float
    total_friction_cost: float
    gross_cumulative_return: float
    net_cumulative_return: float
    benchmark_buy_and_hold_return: float
    max_drawdown: float
    win_rate: float
    steps: tuple[ReplayStep, ...] = field(default_factory=tuple)

    def print_summary(self) -> None:
        """Formatted terminal presentation."""
        print("=" * 78)
        print(f"SENTINEL TRADER -- Chronological Replay Evaluation ({self.strategy_label})")
        print("=" * 78)
        print(f"Instrument / Ticker:        {self.ticker}")
        print(f"Total Input Bars:           {self.total_input_bars}")
        print(f"Evaluated Decisions:        {self.evaluated_steps}")
        print(f"Action Counts:              {dict(self.action_counts)}")
        print(f"Total Portfolio Turnover:   {self.total_turnover:.2f}x")
        print(f"Total Friction Incurred:    {self.total_friction_cost * 100:.3f}%")
        print(f"Gross Cumulative Return:    {self.gross_cumulative_return * 100:+.2f}%")
        print(f"Net Cumulative Return:      {self.net_cumulative_return * 100:+.2f}%")
        print(f"Benchmark (Buy & Hold):     {self.benchmark_buy_and_hold_return * 100:+.2f}%")
        print(f"Maximum Drawdown:           {self.max_drawdown * 100:.2f}%")
        print(f"Trade Win Rate:             {self.win_rate * 100:.1f}%")
        print("=" * 78)


class ChronologicalReplayEngine:
    """Strict point-in-time chronological simulation engine."""

    def __init__(
        self,
        decision_engine: SentinelDecisionEngine | None = None,
        friction_engine: MarketFrictionEngine | None = None,
    ) -> None:
        self.decision_engine = decision_engine or SentinelDecisionEngine()
        self.friction_engine = friction_engine or MarketFrictionEngine()

    def run(
        self,
        bars: Sequence[MarketBar],
        warmup_bars: int = 30,
        initial_capital: float = 100000.0,
    ) -> ReplaySummary:
        """Execute chronological simulation across all bars."""
        if not bars:
            raise ContractValidationError("Cannot run replay on empty bar sequence")
        if len(bars) <= warmup_bars + 2:
            raise ContractValidationError(
                f"Need at least {warmup_bars + 3} bars for replay, got {len(bars)}"
            )

        validate_market_bar_sequence(bars)
        ticker = bars[0].ticker

        equity = initial_capital
        peak_equity = initial_capital
        current_exposure = 0.0
        max_dd = 0.0
        total_turnover = 0.0
        total_friction = 0.0
        winning_steps = 0
        active_steps = 0

        action_counts: dict[str, int] = {
            ActionType.BUY.value: 0,
            ActionType.SELL.value: 0,
            ActionType.HOLD.value: 0,
            ActionType.EXIT.value: 0,
            ActionType.REDUCE.value: 0,
            ActionType.NO_TRADE.value: 0,
        }

        recorded_steps: list[ReplayStep] = []

        # Chronological loop: from warmup_bars up to len(bars) - 2
        # Decision at index t (close of bar t)
        # Execution at index t + 1 (open of bar t + 1)
        # Position held until open of bar t + 2
        for t in range(warmup_bars, len(bars) - 1):
            history_slice = bars[: t + 1]  # STRICT CAUSALITY: only data up to bar t
            current_bar = bars[t]
            next_bar = bars[t + 1]
            future_bar = bars[t + 2] if (t + 2) < len(bars) else None

            current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

            # Evaluate decision causally
            decision: TradeDecision = self.decision_engine.evaluate_decision(
                bars=history_slice,
                current_position_exposure=current_exposure,
                current_drawdown=current_dd,
                available_cash=equity * (1.0 - abs(current_exposure)),
                current_portfolio_value=equity,
            )

            act_str = decision.selected_action.value
            action_counts[act_str] = action_counts.get(act_str, 0) + 1

            new_exposure = decision.target_quantity_or_exposure
            turnover = abs(new_exposure - current_exposure)
            total_turnover += turnover

            # Friction incurred on turnover
            step_friction = turnover * self.friction_engine.one_way_friction_fraction
            total_friction += step_friction

            # Price movement from next_bar.open to future_bar.open (or next_bar.close if terminal)
            entry_px = next_bar.open
            exit_px = future_bar.open if future_bar is not None else next_bar.close
            raw_asset_ret = (exit_px - entry_px) / entry_px

            step_gross_ret = new_exposure * raw_asset_ret
            step_net_ret = step_gross_ret - step_friction

            # Update equity
            equity *= (1.0 + step_net_ret)
            if equity > peak_equity:
                peak_equity = equity

            step_dd = (peak_equity - equity) / peak_equity
            if step_dd > max_dd:
                max_dd = step_dd

            if abs(new_exposure) > 1e-4:
                active_steps += 1
                if step_net_ret > 0:
                    winning_steps += 1

            current_exposure = new_exposure

            recorded_steps.append(
                ReplayStep(
                    step_index=t,
                    decision_timestamp=current_bar.source_timestamp,
                    execution_timestamp=next_bar.source_timestamp,
                    close_price=current_bar.close,
                    next_open_price=entry_px,
                    action=decision.selected_action,
                    target_exposure=new_exposure,
                    turnover=turnover,
                    step_gross_return=step_gross_ret,
                    step_friction_cost=step_friction,
                    step_net_return=step_net_ret,
                    portfolio_equity=equity,
                    current_drawdown=step_dd,
                )
            )

        # Cumulative metrics
        gross_cum = math.prod(1.0 + s.step_gross_return for s in recorded_steps) - 1.0
        net_cum = (equity - initial_capital) / initial_capital

        # Benchmark: buy and hold from first evaluation execution price to last exit price
        first_open = recorded_steps[0].next_open_price or bars[warmup_bars + 1].open
        last_close = bars[-1].close
        benchmark_ret = (last_close - first_open) / first_open

        win_rate = winning_steps / max(1, active_steps)

        return ReplaySummary(
            strategy_label=self.decision_engine.config.engine_id,
            ticker=ticker,
            total_input_bars=len(bars),
            evaluated_steps=len(recorded_steps),
            action_counts=action_counts,
            total_turnover=total_turnover,
            total_friction_cost=total_friction,
            gross_cumulative_return=gross_cum,
            net_cumulative_return=net_cum,
            benchmark_buy_and_hold_return=benchmark_ret,
            max_drawdown=max_dd,
            win_rate=win_rate,
            steps=tuple(recorded_steps),
        )

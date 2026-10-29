"""NOVA Research Evaluator — Leakage-Free Chronological Strategy Backtesting Pipeline.

Implements strict point-in-time signal-to-execution mechanics:
1. Signal at bar t is computed strictly from data <= bar t close.
2. Order execution occurs at bar t+1 open.
3. Return across step t+1 is (Open_{t+2} - Open_{t+1}) / Open_{t+1}.
4. Final bar has no future open, preventing fabricated returns.
5. Incomplete or unverified friction metrics are marked UNAVAILABLE (never assumed zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Sequence

from nova.quant.contracts import (
    ContractValidationError,
    EvaluationStatus,
    MarketBar,
    NumericalValidationError,
    SignalDirection,
    StrategyEvaluation,
    validate_finite,
)
from nova.research.adapters import validate_market_bar_sequence
from nova.research.ema_crossover import EMACrossoverStrategy


@dataclass(frozen=True, slots=True)
class FrictionModel:
    """Transaction cost and execution friction parameters."""
    brokerage_bps: float | None = None  # Basis points (e.g. 2.0 bps = 0.0002)
    slippage_bps: float | None = None   # Basis points (e.g. 5.0 bps = 0.0005)
    status: str = "UNVERIFIED_PENDING_PLATFORM_CONTRACT"

    def is_fully_specified(self) -> bool:
        """Returns True only when both brokerage and slippage are valid finite numbers."""
        return (
            self.brokerage_bps is not None
            and self.slippage_bps is not None
            and not isinstance(self.brokerage_bps, bool)
            and not isinstance(self.slippage_bps, bool)
            and math.isfinite(self.brokerage_bps)
            and math.isfinite(self.slippage_bps)
            and self.brokerage_bps >= 0.0
            and self.slippage_bps >= 0.0
        )

    @property
    def total_cost_fraction(self) -> float | None:
        """Total roundtrip/one-way cost as a dimensionless fraction, or None if unverified."""
        if not self.is_fully_specified():
            return None
        return (self.brokerage_bps + self.slippage_bps) / 10000.0


@dataclass(frozen=True, slots=True)
class BarEvaluationStep:
    """Point-in-time record of a single evaluation step."""
    bar_index: int
    signal_timestamp: datetime
    execution_timestamp: datetime | None
    signal_direction: SignalDirection
    position: float               # Target position held during holding step (1.0 = Long, 0.0 = Flat)
    position_change: float        # Absolute turnover |pos_curr - pos_prev|
    entry_price: float | None     # Open_{t+1}
    exit_price: float | None      # Open_{t+2}
    gross_return: float | None    # pos * (exit - entry) / entry
    friction_cost: float | None   # position_change * total_cost_fraction (or None)
    net_return: float | None      # gross_return - friction_cost (or None)
    is_terminal_step: bool        # True for final observation with no future execution


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Comprehensive, machine-readable strategy evaluation results."""
    strategy_id: str
    strategy_version: str
    ticker: str
    timing_convention: str
    total_bars_input: int
    warmup_bars: int
    evaluated_steps: int
    bullish_signals_count: int
    neutral_signals_count: int
    crossover_entries_count: int
    crossover_exits_count: int
    total_turnover: float
    gross_cumulative_return: float
    benchmark_cumulative_return: float
    max_drawdown: float
    friction_status: str
    net_cumulative_return: float | None
    is_synthetic_data: bool
    evaluation_start_time: datetime
    evaluation_end_time: datetime
    steps: tuple[BarEvaluationStep, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Convert report to serializable dictionary for JSON persistence."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "ticker": self.ticker,
            "timing_convention": self.timing_convention,
            "provenance": "SYNTHETIC_OFFLINE_VALIDATION" if self.is_synthetic_data else "CANONICAL_HISTORICAL_DATA",
            "is_synthetic_data": self.is_synthetic_data,
            "evaluation_start_time": self.evaluation_start_time.isoformat(),
            "evaluation_end_time": self.evaluation_end_time.isoformat(),
            "sample_counts": {
                "total_bars_input": self.total_bars_input,
                "warmup_bars": self.warmup_bars,
                "evaluated_steps": self.evaluated_steps,
            },
            "signals": {
                "bullish_signals_count": self.bullish_signals_count,
                "neutral_signals_count": self.neutral_signals_count,
                "crossover_entries_count": self.crossover_entries_count,
                "crossover_exits_count": self.crossover_exits_count,
            },
            "performance": {
                "total_turnover": round(self.total_turnover, 4),
                "gross_cumulative_return": round(self.gross_cumulative_return, 6),
                "benchmark_cumulative_return": round(self.benchmark_cumulative_return, 6),
                "max_drawdown": round(self.max_drawdown, 6),
                "friction_status": self.friction_status,
                "net_cumulative_return": round(self.net_cumulative_return, 6) if self.net_cumulative_return is not None else None,
            },
            "warnings": [
                "Real-market performance remains unverified; results based on offline execution.",
            ] if self.is_synthetic_data else [],
        }


class StrategyEvaluator:
    """Leakage-free evaluation pipeline for NOVA quantitative strategies."""

    def __init__(
        self,
        strategy: EMACrossoverStrategy,
        friction: FrictionModel | None = None,
    ) -> None:
        self.strategy = strategy
        self.friction = friction or FrictionModel()

    def run_evaluation(
        self,
        bars: Sequence[MarketBar],
        out_of_sample_start_index: int | None = None,
    ) -> EvaluationReport:
        """Run chronological evaluation across bars.

        Timing Convention:
        - At bar t: Signal S_t is produced using bars[0...t].
        - At bar t+1: Trade enters at Open_{t+1}.
        - At bar t+2: Trade exits/rebalances at Open_{t+2}.
        - For bar N-1: S_{N-1} is generated but no Open_N exists -> terminal step (return unavailable).
        """
        validate_market_bar_sequence(bars)
        n_bars = len(bars)
        slow_window = self.strategy.config.slow_window

        if n_bars <= slow_window:
            raise ContractValidationError(
                f"Insufficient bars for evaluation: {n_bars} provided, slow_window={slow_window}"
            )

        # Determine start index (must be at least slow_window to respect warm-up)
        eval_start = max(slow_window, out_of_sample_start_index or slow_window)

        # 1. Causally evaluate signals for each bar in sequence
        evaluations: list[StrategyEvaluation] = []
        for t in range(n_bars):
            eval_result = self.strategy.evaluate(bars[: t + 1])
            evaluations.append(eval_result)

        # 2. Simulate next-bar open execution and step returns
        steps: list[BarEvaluationStep] = []
        prev_pos = 0.0
        bullish_count = 0
        neutral_count = 0
        entries_count = 0
        exits_count = 0
        total_turnover = 0.0

        for t in range(eval_start - 1, n_bars):
            eval_res = evaluations[t]
            bar_t = bars[t]
            sig_ts = bar_t.source_timestamp

            # Derive position from signal
            if eval_res.status == EvaluationStatus.CANDIDATE_PRODUCED and eval_res.candidate is not None:
                if eval_res.candidate.direction == SignalDirection.BULLISH:
                    target_pos = 1.0
                    bullish_count += 1
                else:
                    target_pos = 0.0
                    neutral_count += 1
            else:
                target_pos = 0.0
                neutral_count += 1

            turnover = abs(target_pos - prev_pos)
            total_turnover += turnover

            if prev_pos == 0.0 and target_pos == 1.0:
                entries_count += 1
            elif prev_pos == 1.0 and target_pos == 0.0:
                exits_count += 1

            # Next bar open execution
            if t + 1 < n_bars:
                bar_t_plus_1 = bars[t + 1]
                exec_ts = bar_t_plus_1.source_timestamp
                entry_px = bar_t_plus_1.open

                # Exit at bar t+2 open if available
                if t + 2 < n_bars:
                    bar_t_plus_2 = bars[t + 2]
                    exit_px = bar_t_plus_2.open
                    gross_ret = target_pos * ((exit_px - entry_px) / entry_px)
                    is_terminal = False
                else:
                    exit_px = None
                    gross_ret = None
                    is_terminal = True
            else:
                # Terminal observation: bar N-1 has no next open
                exec_ts = None
                entry_px = None
                exit_px = None
                gross_ret = None
                is_terminal = True

            # Calculate friction
            friction_cost = None
            net_ret = None
            if gross_ret is not None and self.friction.is_fully_specified():
                cost_frac = self.friction.total_cost_fraction
                assert cost_frac is not None
                friction_cost = turnover * cost_frac
                net_ret = gross_ret - friction_cost

            step = BarEvaluationStep(
                bar_index=t,
                signal_timestamp=sig_ts,
                execution_timestamp=exec_ts,
                signal_direction=eval_res.candidate.direction if eval_res.candidate else SignalDirection.NEUTRAL,
                position=target_pos,
                position_change=turnover,
                entry_price=entry_px,
                exit_price=exit_px,
                gross_return=gross_ret,
                friction_cost=friction_cost,
                net_return=net_ret,
                is_terminal_step=is_terminal,
            )
            steps.append(step)
            prev_pos = target_pos

        # 3. Aggregate cumulative performance across evaluated steps
        evaluated_steps = [s for s in steps if not s.is_terminal_step and s.gross_return is not None]

        gross_equity = 1.0
        peak_equity = 1.0
        max_drawdown = 0.0

        for s in evaluated_steps:
            assert s.gross_return is not None
            gross_equity *= (1.0 + s.gross_return)
            if gross_equity > peak_equity:
                peak_equity = gross_equity
            dd = (peak_equity - gross_equity) / peak_equity if peak_equity > 0.0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd

        gross_cumulative_return = gross_equity - 1.0

        # Net performance
        net_cumulative_return = None
        if self.friction.is_fully_specified() and evaluated_steps:
            net_equity = 1.0
            for s in evaluated_steps:
                assert s.net_return is not None
                net_equity *= (1.0 + s.net_return)
            net_cumulative_return = net_equity - 1.0

        # Benchmark: Buy and Hold across the exact same holding window
        if evaluated_steps:
            first_entry_px = evaluated_steps[0].entry_price
            last_exit_px = evaluated_steps[-1].exit_price
            if first_entry_px is not None and last_exit_px is not None and first_entry_px > 0.0:
                benchmark_cumulative_return = (last_exit_px - first_entry_px) / first_entry_px
            else:
                benchmark_cumulative_return = 0.0
        else:
            benchmark_cumulative_return = 0.0

        is_synthetic = any(b.provenance == "SYNTHETIC_GENERATOR_DETERMINISTIC" for b in bars)

        return EvaluationReport(
            strategy_id=self.strategy.config.strategy_id,
            strategy_version=self.strategy.config.strategy_version,
            ticker=bars[0].ticker,
            timing_convention="SIGNAL_AT_T_CLOSE__EXECUTE_AT_T_PLUS_1_OPEN",
            total_bars_input=n_bars,
            warmup_bars=slow_window,
            evaluated_steps=len(evaluated_steps),
            bullish_signals_count=bullish_count,
            neutral_signals_count=neutral_count,
            crossover_entries_count=entries_count,
            crossover_exits_count=exits_count,
            total_turnover=total_turnover,
            gross_cumulative_return=gross_cumulative_return,
            benchmark_cumulative_return=benchmark_cumulative_return,
            max_drawdown=max_drawdown,
            friction_status=self.friction.status if not self.friction.is_fully_specified() else "VERIFIED_PROVISIONAL",
            net_cumulative_return=net_cumulative_return,
            is_synthetic_data=is_synthetic,
            evaluation_start_time=bars[eval_start].source_timestamp,
            evaluation_end_time=bars[-1].source_timestamp,
            steps=tuple(steps),
        )


"""NOVA Performance Evaluation & Metrics Subsystem (Person 3 Exclusive Domain).

Computes point-in-time, mathematically rigorous performance metrics:
- Net and cumulative return after friction
- Realized and unrealized P&L
- Maximum drawdown and drawdown series
- Win rate, average win, average loss, and profit factor
- Turnover, transaction costs, fees, and holding durations
- Cash utilization, exposure, and margin usage
- Long versus short performance breakdown
- Risk violations and order rejections
- Benchmark-relative performance (alpha, excess return)

Zero fabricated numbers. Undefined metrics (e.g. division by zero when no losses)
are handled cleanly with None or mathematical definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from nova.decision.contracts import ActionType
from nova.quant.contracts import validate_finite


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Standardized performance metrics suite for Sentinel Trader evaluation."""
    # Return & Wealth
    net_cumulative_return: float
    gross_cumulative_return: float
    net_pnl_inr: float
    total_realized_pnl_inr: float
    final_unrealized_pnl_inr: float
    final_wealth_inr: float
    initial_capital: float

    # Drawdown & Risk
    max_drawdown: float
    drawdown_series: tuple[float, ...] = field(default_factory=tuple)
    risk_violations_count: int = 0
    rejected_orders_count: int = 0

    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_winning_trade_inr: float | None = None
    avg_losing_trade_inr: float | None = None
    profit_factor: float | None = None

    # Costs & Execution
    total_turnover: float = 0.0
    total_friction_cost: float = 0.0
    total_fees_inr: float = 0.0
    partial_exits_count: int = 0

    # Holding & Utilization
    avg_holding_duration_bars: float = 0.0
    avg_cash_utilization: float = 0.0
    avg_exposure: float = 0.0
    max_margin_usage_inr: float = 0.0

    # Long vs Short Breakdown
    long_trades_count: int = 0
    short_trades_count: int = 0
    long_realized_pnl_inr: float = 0.0
    short_realized_pnl_inr: float = 0.0
    long_win_rate: float | None = None
    short_win_rate: float | None = None

    # Benchmark Comparison
    benchmark_return: float | None = None
    excess_return_over_benchmark: float | None = None

    def format_summary_table(self) -> str:
        """Return formatted string summarizing all performance metrics."""
        lines = [
            "=" * 78,
            "SENTINEL TRADER -- PERFORMANCE EVALUATION METRICS REPORT",
            "=" * 78,
            f"Initial Capital (INR):         INR {self.initial_capital:>16,.2f}",
            f"Final Account Wealth (INR):    INR {self.final_wealth_inr:>16,.2f}",
            f"Net Realized P&L (INR):        INR {self.total_realized_pnl_inr:>+16,.2f}",
            f"Final Unrealized P&L (INR):    INR {self.final_unrealized_pnl_inr:>+16,.2f}",
            f"Net Account P&L (INR):         INR {self.net_pnl_inr:>+16,.2f}",
            "-" * 78,
            f"Gross Cumulative Return:       {self.gross_cumulative_return * 100:>+20.2f}%",
            f"Net Cumulative Return:         {self.net_cumulative_return * 100:>+20.2f}%",
            f"Maximum Drawdown:              {self.max_drawdown * 100:>20.2f}%",
            "-" * 78,
            f"Executed Trades Count:         {self.total_trades:>20}",
            f"Winning / Losing Trades:       {f'{self.winning_trades} / {self.losing_trades}':>20}",
            f"Trade Win Rate:                {self.win_rate * 100:>20.1f}%",
            f"Profit Factor:                 {f'{self.profit_factor:.2f}' if self.profit_factor is not None else 'N/A (No Losses)':>20}",
            f"Avg Winning Trade (INR):       {f'INR {self.avg_winning_trade_inr:+,.2f}' if self.avg_winning_trade_inr is not None else 'N/A':>20}",
            f"Avg Losing Trade (INR):        {f'INR {self.avg_losing_trade_inr:+,.2f}' if self.avg_losing_trade_inr is not None else 'N/A':>20}",
            "-" * 78,
            f"Total Turnover:                {self.total_turnover:>20.2f}x",
            f"Total Friction Incurred:       {self.total_friction_cost * 100:>20.3f}%",
            f"Total Trading Fees (INR):      INR {self.total_fees_inr:>16,.2f}",
            f"Partial Profit Exits:          {self.partial_exits_count:>20}",
            f"Avg Holding Duration:          {self.avg_holding_duration_bars:>19.1f} bars",
            f"Avg Capital Exposure:          {self.avg_exposure * 100:>20.1f}%",
            f"Peak Margin Locked (INR):      INR {self.max_margin_usage_inr:>16,.2f}",
            "-" * 78,
            f"Long Trades / Short Trades:    {f'{self.long_trades_count} / {self.short_trades_count}':>20}",
            f"Long Realized P&L (INR):       INR {self.long_realized_pnl_inr:>+16,.2f}",
            f"Short Realized P&L (INR):      INR {self.short_realized_pnl_inr:>+16,.2f}",
            f"Long Win Rate:                 {f'{self.long_win_rate * 100:.1f}%' if self.long_win_rate is not None else 'N/A':>20}",
            f"Short Win Rate:                {f'{self.short_win_rate * 100:.1f}%' if self.short_win_rate is not None else 'N/A':>20}",
            "-" * 78,
            f"Risk Gate Violations:          {self.risk_violations_count:>20}",
            f"Rejected Orders:               {self.rejected_orders_count:>20}",
            f"Benchmark Return:              {f'{self.benchmark_return * 100:+.2f}%' if self.benchmark_return is not None else 'N/A':>20}",
            f"Excess Return vs Benchmark:    {f'{self.excess_return_over_benchmark * 100:+.2f}%' if self.excess_return_over_benchmark is not None else 'N/A':>20}",
            "=" * 78,
        ]
        return "\n".join(lines)


@dataclass(slots=True)
class ClosedTradeRecord:
    """Record of a closed or partially closed trading roundtrip."""
    action: ActionType
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl_inr: float
    holding_bars: int
    is_long: bool


def compute_performance_metrics(
    steps: Sequence[any],
    initial_capital: float,
    benchmark_return: float | None = None,
    risk_violations: int = 0,
    rejected_orders: int = 0,
) -> PerformanceMetrics:
    """Calculate comprehensive, un-manipulated metrics from chronological replay steps."""
    validate_finite(initial_capital, "initial_capital")
    if initial_capital <= 0.0:
        raise ValueError(f"initial_capital must be strictly positive, got {initial_capital}")

    if not steps:
        return PerformanceMetrics(
            net_cumulative_return=0.0,
            gross_cumulative_return=0.0,
            net_pnl_inr=0.0,
            total_realized_pnl_inr=0.0,
            final_unrealized_pnl_inr=0.0,
            final_wealth_inr=initial_capital,
            initial_capital=initial_capital,
            max_drawdown=0.0,
            drawdown_series=(0.0,),
            benchmark_return=benchmark_return,
            excess_return_over_benchmark=-benchmark_return if benchmark_return is not None else None,
        )

    final_step = steps[-1]
    final_wealth = final_step.portfolio_equity
    net_pnl = final_wealth - initial_capital
    net_cum_return = net_pnl / initial_capital

    gross_cumulative = math.prod(1.0 + s.step_gross_return for s in steps) - 1.0

    # Drawdown series and peak
    drawdown_series = tuple(s.current_drawdown for s in steps)
    max_dd = max(drawdown_series) if drawdown_series else 0.0

    # Realized and unrealized P&L
    total_realized_pnl = sum(s.realized_pnl for s in steps)
    final_unrealized = getattr(final_step, "unrealized_pnl", 0.0)

    # Turnover and friction
    total_turnover = sum(s.turnover for s in steps)
    total_friction_cost = sum(s.step_friction_cost for s in steps)

    # Reconstruct closed trades from step records
    closed_trades: list[ClosedTradeRecord] = []
    trade_pnls: list[float] = [s.realized_pnl for s in steps if abs(s.realized_pnl) > 1e-4]
    partial_exits_count = 0

    # Trade counts and win rate
    total_trades_count = 0
    long_trades: list[float] = []
    short_trades: list[float] = []

    for s in steps:
        if s.turnover > 1e-4 and s.action in (ActionType.BUY, ActionType.SELL, ActionType.SHORT, ActionType.COVER):
            total_trades_count += 1
        if abs(s.realized_pnl) > 1e-4:
            if s.action in (ActionType.SELL, ActionType.EXIT, ActionType.REDUCE):
                long_trades.append(s.realized_pnl)
            elif s.action == ActionType.COVER:
                short_trades.append(s.realized_pnl)

    winning_trade_pnls = [p for p in trade_pnls if p > 0.0]
    losing_trade_pnls = [p for p in trade_pnls if p < 0.0]

    winning_count = len(winning_trade_pnls)
    losing_count = len(losing_trade_pnls)
    evaluated_closed_count = winning_count + losing_count

    win_rate = winning_count / evaluated_closed_count if evaluated_closed_count > 0 else 0.0
    avg_win = sum(winning_trade_pnls) / winning_count if winning_count > 0 else None
    avg_loss = sum(losing_trade_pnls) / losing_count if losing_count > 0 else None

    # Profit factor: sum(gains) / abs(sum(losses))
    total_gains = sum(winning_trade_pnls)
    total_losses = abs(sum(losing_trade_pnls))
    if total_losses > 1e-4:
        profit_factor = round(total_gains / total_losses, 4)
    elif total_gains > 1e-4:
        profit_factor = None  # Infinite / undefined because no losses
    else:
        profit_factor = 0.0

    # Long vs Short
    long_count = len(long_trades)
    short_count = len(short_trades)
    long_realized = sum(long_trades)
    short_realized = sum(short_trades)
    long_win_count = sum(1 for p in long_trades if p > 0.0)
    short_win_count = sum(1 for p in short_trades if p > 0.0)
    long_win_rate = (long_win_count / long_count) if long_count > 0 else None
    short_win_rate = (short_win_count / short_count) if short_count > 0 else None

    # Holding durations, exposure, cash utilization
    active_steps = [s for s in steps if abs(s.target_exposure) > 1e-4]
    avg_exposure = sum(abs(s.target_exposure) for s in steps) / len(steps) if steps else 0.0
    avg_holding_duration = len(active_steps) / max(1, total_trades_count)

    # Cash utilization
    avg_cash_utilization = sum(
        min(1.0, max(0.0, 1.0 - abs(s.target_exposure))) for s in steps
    ) / len(steps) if steps else 1.0

    # Maximum margin usage
    max_margin = max(
        (abs(s.target_exposure) * s.portfolio_equity for s in steps if s.target_exposure < -1e-4),
        default=0.0,
    )

    # Benchmark comparison
    excess_ret = (net_cum_return - benchmark_return) if benchmark_return is not None else None

    return PerformanceMetrics(
        net_cumulative_return=net_cum_return,
        gross_cumulative_return=gross_cumulative,
        net_pnl_inr=net_pnl,
        total_realized_pnl_inr=total_realized_pnl,
        final_unrealized_pnl_inr=final_unrealized,
        final_wealth_inr=final_wealth,
        initial_capital=initial_capital,
        max_drawdown=max_dd,
        drawdown_series=drawdown_series,
        risk_violations_count=risk_violations,
        rejected_orders_count=rejected_orders,
        total_trades=total_trades_count,
        winning_trades=winning_count,
        losing_trades=losing_count,
        win_rate=win_rate,
        avg_winning_trade_inr=avg_win,
        avg_losing_trade_inr=avg_loss,
        profit_factor=profit_factor,
        total_turnover=total_turnover,
        total_friction_cost=total_friction_cost,
        total_fees_inr=0.0,  # Will be mapped from ReplaySummary
        partial_exits_count=getattr(final_step, "partial_exit_count", 0),
        avg_holding_duration_bars=avg_holding_duration,
        avg_cash_utilization=avg_cash_utilization,
        avg_exposure=avg_exposure,
        max_margin_usage_inr=max_margin,
        long_trades_count=long_count,
        short_trades_count=short_count,
        long_realized_pnl_inr=long_realized,
        short_realized_pnl_inr=short_realized,
        long_win_rate=long_win_rate,
        short_win_rate=short_win_rate,
        benchmark_return=benchmark_return,
        excess_return_over_benchmark=excess_ret,
    )

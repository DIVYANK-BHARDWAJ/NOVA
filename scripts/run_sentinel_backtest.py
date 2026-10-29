"""CLI Backtest Runner & Strategy Validator for Sentinel Trader (Person 3).

Executes point-in-time chronological backtesting, computes the complete Part 7
metrics suite, evaluates baselines (Buy & Hold, No Trade), and performs full
component ablation to quantify marginal utility contributions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
EXECUTION_SRC = REPO_ROOT / "execution" / "src"
KAFKA_SRC = REPO_ROOT / "kafka" / "src"

for d in (KAFKA_SRC, EXECUTION_SRC, SRC_DIR, REPO_ROOT):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from nova.evaluation.ablation import AblationRunner, AblationStudyResult
from nova.evaluation.metrics import PerformanceMetrics
from nova.evaluation.replay import ChronologicalReplayEngine, ReplaySummary
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import DataQualityStatus, MarketBar, TimestampSemantics
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.prediction import CausalRidgePredictor
from nova.decision.regime import MarketRegimeDetector


def generate_scenario_bars(pattern: str = "cyclical", n_bars: int = 80, ticker: str = "SENTINEL_EVAL") -> list[MarketBar]:
    base_time = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars: list[MarketBar] = []
    price = 100.0

    for i in range(n_bars):
        if pattern == "bullish":
            price *= 1.004
        elif pattern == "bearish":
            price *= 0.996
        elif pattern == "cyclical":
            # Oscillates in realistic waves
            phase = i % 12
            price += (1.5 if phase < 6 else -1.5)
        else:
            price += (0.1 if i % 2 == 0 else -0.1)

        t = base_time + timedelta(minutes=5 * i)
        bars.append(
            MarketBar(
                ticker=ticker,
                open=round(price * 0.999, 2),
                high=round(price * 1.004, 2),
                low=round(price * 0.996, 2),
                close=round(price, 2),
                volume=10000.0,
                source_timestamp=t,
                available_timestamp=t,
                timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                data_quality=DataQualityStatus.VALID,
                provenance="MANTHAN_EVALUATION_HARNESS",
            )
        )
    return bars


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel Trader Rigorous Backtesting & Strategy Validator")
    parser.add_argument("--pattern", choices=["bullish", "bearish", "cyclical"], default="cyclical", help="Market scenario pattern")
    parser.add_argument("--bars", type=int, default=80, help="Number of candlestick bars to simulate")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup period bars")
    parser.add_argument("--capital", type=float, default=1000000.0, help="Starting capital in INR (default 10,00,000)")
    parser.add_argument("--timing", choices=["CANDLE_CLOSE", "NEXT_OPEN"], default="CANDLE_CLOSE", help="Execution fill timing")
    parser.add_argument("--run-ablation", action="store_true", default=True, help="Execute full component ablation study")
    parser.add_argument("--output-report", type=str, default="reports/sentinel_backtest_report.json", help="Path for JSON audit report")
    args = parser.parse_args()

    bars = generate_scenario_bars(pattern=args.pattern, n_bars=args.bars)
    warmup = args.warmup

    print("=" * 88)
    print("SENTINEL TRADER -- PART 7 RIGOROUS STRATEGY BACKTESTING & VALIDATION")
    print("=" * 88)
    print(f"Scenario Pattern:          {args.pattern.upper()}")
    print(f"Total Candlestick Bars:    {len(bars)}")
    print(f"Warmup / Fitting Bars:     {warmup}")
    print(f"Execution Fill Timing:     {args.timing}")
    print(f"Starting Account Capital:  INR {args.capital:,.2f}")
    print("=" * 88)

    # 1. Setup Ablation Runner
    ablation_runner = AblationRunner()
    study_result: AblationStudyResult = ablation_runner.run_study(
        bars=bars,
        warmup_bars=warmup,
        initial_capital=args.capital,
        execution_timing=args.timing,
    )

    # Print Ablation Matrix
    print("\n" + study_result.format_table() + "\n")

    # 2. Detailed Full System Performance Metrics
    full_variant = study_result.variants[0]
    print("=" * 88)
    print("PRIMARY FULL SYSTEM PERFORMANCE METRICS BREAKDOWN")
    print("=" * 88)
    print(f"Net Cumulative Return:     {full_variant.net_return * 100:+.2f}%")
    print(f"Net Realized P&L (INR):    INR {full_variant.net_pnl_inr:+,.2f}")
    print(f"Maximum Drawdown:          {full_variant.max_drawdown * 100:.2f}%")
    print(f"Trade Win Rate:            {full_variant.win_rate * 100:.1f}%")
    print(f"Total Trades Executed:     {full_variant.total_trades}")
    print(f"Total Portfolio Turnover:  {full_variant.total_turnover:.2f}x")
    print(f"Total Trading Fees (INR):  INR {full_variant.total_fees_inr:,.2f}")
    print(f"Benchmark Buy & Hold:      {study_result.benchmark_return * 100:+.2f}%")
    print(f"Excess Alpha Return:       {(full_variant.net_return - study_result.benchmark_return) * 100:+.2f}%")
    print("=" * 88)

    # 3. Export Comprehensive JSON Audit Report
    report_data = {
        "report_type": "sentinel_trader_part7_backtest_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "market_scenario": {
            "pattern": args.pattern,
            "total_bars": len(bars),
            "warmup_bars": warmup,
            "ticker": bars[0].ticker,
            "initial_capital_inr": args.capital,
            "execution_timing": args.timing,
        },
        "benchmark": {
            "name": "Buy and Hold Asset Benchmark",
            "net_return": study_result.benchmark_return,
        },
        "ablation_variants": [
            {
                "id": v.variant_id,
                "name": v.variant_name,
                "description": v.description,
                "supported": v.is_supported,
                "unsupported_reason": v.unsupported_reason,
                "net_return": v.net_return,
                "net_pnl_inr": v.net_pnl_inr,
                "max_drawdown": v.max_drawdown,
                "win_rate": v.win_rate,
                "turnover": v.total_turnover,
                "fees_inr": v.total_fees_inr,
                "trade_count": v.total_trades,
                "marginal_contribution_bps": v.marginal_contribution_bps,
            }
            for v in study_result.variants
        ],
    }

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Complete Part 7 Backtest Audit Report written to: {out_path.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

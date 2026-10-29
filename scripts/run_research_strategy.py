#!/usr/bin/env python3
"""NOVA Research — Strategy Execution and Evaluation CLI Entry Point.

Executes quantitative strategies under strict point-in-time causality,
supporting deterministic offline synthetic demo mode and CSV file evaluation.

Usage:
    # Run synthetic demo mode
    python scripts/run_research_strategy.py --demo

    # Run on CSV market data file
    python scripts/run_research_strategy.py --file data/sample_bars.csv

    # Run with custom parameters and provisional friction
    python scripts/run_research_strategy.py --demo --fast-window 10 --slow-window 20 --brokerage-bps 2.0 --slippage-bps 5.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root / src is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.research.adapters import (
    generate_synthetic_ohlcv,
    load_market_bars_from_csv,
)
from nova.research.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)
from nova.research.evaluator import (
    FrictionModel,
    StrategyEvaluator,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NOVA Research — Fast/Slow EMA Crossover Strategy Evaluator"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Run deterministic synthetic demo (default)",
    )
    mode_group.add_argument(
        "--file",
        type=str,
        help="Path to CSV input file containing market bars",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SYNTH_TREND",
        help="Ticker symbol (used for synthetic data or when missing in CSV)",
    )
    parser.add_argument(
        "--fast-window",
        type=int,
        default=12,
        help="Fast EMA period (default: 12)",
    )
    parser.add_argument(
        "--slow-window",
        type=int,
        default=26,
        help="Slow EMA period (default: 26)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=120,
        help="Number of bars to generate in synthetic demo mode (default: 120)",
    )
    parser.add_argument(
        "--brokerage-bps",
        type=float,
        default=None,
        help="Provisional brokerage fee in basis points (e.g. 2.0 for 0.0002)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="Provisional slippage in basis points (e.g. 5.0 for 0.0005)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "reports" / "research_ema_crossover_report.json"),
        help="Path to output JSON evaluation report",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    print("=" * 78)
    print("NOVA Quantitative Research -- Strategy Evaluation Pipeline")
    print("Strategy: Fast/Slow EMA Crossover Trend-Following Baseline")
    print("=" * 78)

    # 1. Load or Generate Market Data
    if parsed.file:
        file_path = Path(parsed.file)
        print(f"[*] Ingesting market bars from file: {file_path}")
        try:
            bars = load_market_bars_from_csv(file_path, ticker=parsed.ticker)
        except Exception as exc:
            print(f"[!] Error loading market bars from CSV: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"[*] Running in DETERMINISTIC SYNTHETIC DEMO mode ({parsed.bars} bars, ticker='{parsed.ticker}')")
        try:
            bars = generate_synthetic_ohlcv(ticker=parsed.ticker, n_bars=parsed.bars)
        except Exception as exc:
            print(f"[!] Error generating synthetic data: {exc}", file=sys.stderr)
            return 1

    print(f"[*] Successfully validated {len(bars)} chronological bars ({bars[0].source_timestamp.isoformat()} to {bars[-1].source_timestamp.isoformat()})")

    # 2. Configure Strategy and Friction
    config = EMACrossoverConfig(
        fast_window=parsed.fast_window,
        slow_window=parsed.slow_window,
        allow_short=False,
    )
    strategy = EMACrossoverStrategy(config=config)

    if parsed.brokerage_bps is not None and parsed.slippage_bps is not None:
        friction = FrictionModel(
            brokerage_bps=parsed.brokerage_bps,
            slippage_bps=parsed.slippage_bps,
            status="VERIFIED_PROVISIONAL",
        )
        print(f"[*] Friction model configured: brokerage={parsed.brokerage_bps} bps, slippage={parsed.slippage_bps} bps")
    else:
        friction = FrictionModel(
            brokerage_bps=None,
            slippage_bps=None,
            status="UNVERIFIED_PENDING_PLATFORM_CONTRACT",
        )
        print("[*] Friction model: UNVERIFIED (Net performance marked UNAVAILABLE per contract invariants)")

    # 3. Execute Chronological Evaluation
    evaluator = StrategyEvaluator(strategy=strategy, friction=friction)
    try:
        report = evaluator.run_evaluation(bars)
    except Exception as exc:
        print(f"[!] Error during strategy evaluation: {exc}", file=sys.stderr)
        return 1

    # 4. Display Formatted Results Summary
    print("-" * 78)
    print("EVALUATION RESULTS SUMMARY")
    print("-" * 78)
    print(f"Ticker:                     {report.ticker}")
    print(f"Timing Convention:          {report.timing_convention}")
    print(f"Total Input Bars:           {report.total_bars_input}")
    print(f"Warm-up Bars:               {report.warmup_bars}")
    print(f"Evaluated Trade Steps:      {report.evaluated_steps}")
    print(f"Bullish Signals Emitted:    {report.bullish_signals_count}")
    print(f"Neutral Signals Emitted:    {report.neutral_signals_count}")
    print(f"Crossover Entries:          {report.crossover_entries_count}")
    print(f"Crossover Exits:            {report.crossover_exits_count}")
    print(f"Total Turnover:             {report.total_turnover:.2f}")
    print(f"Gross Cumulative Return:    {report.gross_cumulative_return * 100.0:+.2f}%")
    print(f"Benchmark (Buy & Hold):     {report.benchmark_cumulative_return * 100.0:+.2f}%")
    print(f"Maximum Drawdown:           {report.max_drawdown * 100.0:.2f}%")

    if report.net_cumulative_return is not None:
        print(f"Net Cumulative Return:      {report.net_cumulative_return * 100.0:+.2f}%")
    else:
        print(f"Net Cumulative Return:      UNAVAILABLE ({report.friction_status})")

    # 5. Write JSON Report
    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("-" * 78)
    print(f"[+] Full JSON evaluation report written to: {output_path.resolve()}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

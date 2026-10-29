"""CLI Runner for Sentinel Trader Replay Evaluation (Person 3).

Demonstrates the modular trend-following and mean-reversion strategies,
combined decision engine, all-in friction modeling, and deterministic risk gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import sys

# Ensure src is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.evaluation.replay import ChronologicalReplayEngine, ReplaySummary
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.quant.contracts import DataQualityStatus, MarketBar, TimestampSemantics
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy


def generate_market_scenario(pattern: str = "bullish", n_bars: int = 60) -> list[MarketBar]:
    base_time = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars: list[MarketBar] = []
    price = 100.0

    for i in range(n_bars):
        if pattern == "bullish":
            price *= 1.005
        elif pattern == "bearish":
            price *= 0.995
        elif pattern == "cyclical":
            price = 100.0 + 8.0 * (1.0 if (i // 5) % 2 == 0 else -1.0)
        else:
            price += (0.05 if i % 2 == 0 else -0.05)

        t = base_time + timedelta(minutes=5 * i)
        bars.append(
            MarketBar(
                ticker="SENTINEL_DEMO",
                open=round(price * 0.999, 2),
                high=round(price * 1.003, 2),
                low=round(price * 0.997, 2),
                close=round(price, 2),
                volume=5000.0,
                source_timestamp=t,
                available_timestamp=t,
                timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                data_quality=DataQualityStatus.VALID,
            )
        )
    return bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentinel Trader Decision Replay Evaluation")
    parser.add_argument("--pattern", choices=["bullish", "bearish", "cyclical"], default="bullish", help="Market trend pattern")
    parser.add_argument("--bars", type=int, default=60, help="Number of candlestick bars to simulate")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup period bars")
    parser.add_argument("--spread-bps", type=float, default=2.0, help="Bid-ask spread in bps")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Execution slippage in bps")
    parser.add_argument("--brokerage-bps", type=float, default=1.0, help="Brokerage commission in bps")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path for results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Configure Friction
    friction_params = FrictionParameters(
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        brokerage_bps=args.brokerage_bps,
        tax_bps=1.0,
    )
    friction_engine = MarketFrictionEngine(friction_params)

    # 2. Configure Risk Gate
    risk_config = RiskConfiguration(
        max_position_exposure=0.25,
        max_drawdown_limit=0.15,
        allow_short=False,
    )
    risk_gate = DeterministicRiskGate(risk_config)

    # 3. Configure Strategies & Decision Engine
    trend_strat = TrendFollowingStrategy(TrendFollowingConfig(fast_window=6, slow_window=14, momentum_window=5))
    mr_strat = MeanReversionStrategy(MeanReversionConfig(lookback_window=12, rsi_period=7))
    decision_engine = SentinelDecisionEngine(
        trend_strategy=trend_strat,
        mean_reversion_strategy=mr_strat,
        friction_engine=friction_engine,
        risk_gate=risk_gate,
    )

    # 4. Generate Causal Scenario Data
    bars = generate_market_scenario(pattern=args.pattern, n_bars=args.bars)

    # 5. Execute Chronological Replay
    replay_engine = ChronologicalReplayEngine(
        decision_engine=decision_engine,
        friction_engine=friction_engine,
    )
    summary: ReplaySummary = replay_engine.run(bars, warmup_bars=args.warmup)

    # 6. Output Results
    summary.print_summary()

    if args.output_json:
        out_dict = {
            "strategy": summary.strategy_label,
            "ticker": summary.ticker,
            "total_bars": summary.total_input_bars,
            "evaluated_steps": summary.evaluated_steps,
            "gross_return": summary.gross_cumulative_return,
            "net_return": summary.net_cumulative_return,
            "friction_cost": summary.total_friction_cost,
            "max_drawdown": summary.max_drawdown,
            "win_rate": summary.win_rate,
            "action_counts": summary.action_counts,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"[+] Replay report exported to: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

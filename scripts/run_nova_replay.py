"""CLI Runner for NOVA Trader Replay Evaluation (Person 3).

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
                ticker="NOVA_DEMO",
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
    parser = argparse.ArgumentParser(description="NOVA Trader Decision Replay Evaluation")
    parser.add_argument("--pattern", choices=["bullish", "bearish", "cyclical"], default="bullish", help="Market trend pattern")
    parser.add_argument("--engine", choices=["live", "baseline", "compare"], default="live", help="Which decision engine to run")
    parser.add_argument("--bars", type=int, default=60, help="Number of candlestick bars to simulate")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup period bars")
    parser.add_argument("--spread-bps", type=float, default=2.0, help="Bid-ask spread in bps")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Execution slippage in bps")
    parser.add_argument("--brokerage-bps", type=float, default=1.0, help="Brokerage commission in bps")
    parser.add_argument("--capital", type=float, default=1000000.0, help="Initial portfolio capital in INR (default INR 10,00,000)")
    parser.add_argument("--timing", choices=["NEXT_OPEN", "CANDLE_CLOSE"], default="NEXT_OPEN", help="Execution fill timing (NEXT_OPEN causal vs CANDLE_CLOSE simulator)")
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
        flat_brokerage_inr=20.0,
        flat_dp_charges_inr=15.34,
    )
    friction_engine = MarketFrictionEngine(friction_params)

    # 2. Configure Risk Gate
    risk_config = RiskConfiguration(
        max_position_exposure=0.25,
        max_drawdown_limit=0.15,
        allow_short=True,
        initial_capital=args.capital,
    )
    risk_gate = DeterministicRiskGate(risk_config)

    # 3. Setup Engines
    trend_strat = TrendFollowingStrategy(TrendFollowingConfig(fast_window=6, slow_window=14, momentum_window=5))
    mr_strat = MeanReversionStrategy(MeanReversionConfig(lookback_window=12, rsi_period=7))
    baseline_engine = SentinelDecisionEngine(
        trend_strategy=trend_strat,
        mean_reversion_strategy=mr_strat,
        friction_engine=friction_engine,
        risk_gate=risk_gate,
    )

    from nova.decision.live_engine import LiveDecisionConfig, LiveDecisionCoreEngine
    from nova.decision.prediction import CausalRidgePredictor

    live_config = LiveDecisionConfig(
        sma_fast_window=5,
        sma_slow_window=15,
        er_window=10,
        ema_long_filter_window=20,
        atr_period=10,
        theta_edge_ratio=0.10,
        min_trade_notional=5000.0,
        score_to_mu_scale=1.5,
    )
    live_engine = LiveDecisionCoreEngine(live_config)

    # 4. Generate Causal Scenario Data
    bars = generate_market_scenario(pattern=args.pattern, n_bars=args.bars)

    # Pre-fit predictor chronologically
    live_engine.predictor.fit_chronological(bars[:args.warmup], train_fraction=0.8)

    # 5. Execute Replay
    if args.engine == "baseline":
        replay_engine = ChronologicalReplayEngine(decision_engine=baseline_engine, friction_engine=friction_engine)
        summary = replay_engine.run(bars, warmup_bars=args.warmup, initial_capital=args.capital, execution_timing=args.timing)
        summary.print_summary()
    elif args.engine == "live":
        replay_engine = ChronologicalReplayEngine(decision_engine=live_engine, friction_engine=friction_engine)
        summary = replay_engine.run(bars, warmup_bars=args.warmup, initial_capital=args.capital, execution_timing=args.timing)
        summary.print_summary()
    else:  # compare
        rep_base = ChronologicalReplayEngine(decision_engine=baseline_engine, friction_engine=friction_engine)
        sum_base = rep_base.run(bars, warmup_bars=args.warmup, initial_capital=args.capital, execution_timing=args.timing)

        rep_live = ChronologicalReplayEngine(decision_engine=live_engine, friction_engine=friction_engine)
        sum_live = rep_live.run(bars, warmup_bars=args.warmup, initial_capital=args.capital, execution_timing=args.timing)

        print("=" * 86)
        print("NOVA TRADER -- HEAD-TO-HEAD REPLAY EVALUATION (BASELINE vs 14-STEP LIVE ENGINE)")
        print("=" * 86)
        print(f"{'Metric':<34} | {'Baseline Ensemble':<22} | {'14-Step Live Engine':<22}")
        print("-" * 86)
        print(f"{'Evaluated Steps':<34} | {sum_base.evaluated_steps:<22} | {sum_live.evaluated_steps:<22}")
        print(f"{'Executed Trades':<34} | {sum_base.total_trades:<22} | {sum_live.total_trades:<22}")
        print(f"{'Trading Fees Incurred (INR)':<34} | INR {sum_base.total_fees_inr:>16,.2f} | INR {sum_live.total_fees_inr:>16,.2f}")
        print(f"{'Total Portfolio Turnover':<34} | {sum_base.total_turnover:>21.2f}x | {sum_live.total_turnover:>21.2f}x")
        print(f"{'Gross Cumulative Return':<34} | {sum_base.gross_cumulative_return*100:>+20.2f}% | {sum_live.gross_cumulative_return*100:>+20.2f}%")
        print(f"{'Friction Ratio Drag':<34} | {sum_base.total_friction_cost*100:>21.3f}% | {sum_live.total_friction_cost*100:>21.3f}%")
        print(f"{'Net Cumulative Return':<34} | {sum_base.net_cumulative_return*100:>+20.2f}% | {sum_live.net_cumulative_return*100:>+20.2f}%")
        print(f"{'Net Realized P&L (INR)':<34} | INR {sum_base.net_pnl_inr:>+16,.2f} | INR {sum_live.net_pnl_inr:>+16,.2f}")
        print(f"{'Final Account Wealth (INR)':<34} | INR {sum_base.final_wealth_inr:>16,.2f} | INR {sum_live.final_wealth_inr:>16,.2f}")
        print(f"{'Maximum Drawdown':<34} | {sum_base.max_drawdown*100:>21.2f}% | {sum_live.max_drawdown*100:>21.2f}%")
        print(f"{'Trade Win Rate':<34} | {sum_base.win_rate*100:>21.1f}% | {sum_live.win_rate*100:>21.1f}%")
        print(f"{'Risk Gate Violations':<34} | {sum_base.risk_violations:<22} | {sum_live.risk_violations:<22}")
        print(f"{'Benchmark (Buy & Hold)':<34} | {sum_base.benchmark_buy_and_hold_return*100:>+20.2f}% | {sum_live.benchmark_buy_and_hold_return*100:>+20.2f}%")
        print("=" * 86)
        summary = sum_live

    if args.output_json:
        out_dict = {
            "strategy": summary.strategy_label,
            "ticker": summary.ticker,
            "total_bars": summary.total_input_bars,
            "evaluated_steps": summary.evaluated_steps,
            "total_trades": summary.total_trades,
            "trading_fees_inr": summary.total_fees_inr,
            "net_pnl_inr": summary.net_pnl_inr,
            "final_wealth_inr": summary.final_wealth_inr,
            "gross_return": summary.gross_cumulative_return,
            "net_return": summary.net_cumulative_return,
            "friction_cost": summary.total_friction_cost,
            "max_drawdown": summary.max_drawdown,
            "win_rate": summary.win_rate,
            "risk_violations": summary.risk_violations,
            "action_counts": summary.action_counts,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"[+] Replay report exported to: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""NOVA Autonomous Trading Bot — Simulation & Real-Time Strategy Execution Entrypoint.

Starts NOVA's market data ingestion pipeline connected to the quantitative EMACrossoverStrategy.

Execution Modes:
1. Live Competition Mode: Connects to competition Kafka broker (192.168.65.157:9094) on topic 'market.ticks'.
2. Simulated Sandbox Feed (--simulated-feed): Replays high-fidelity streaming candlestick ticks when offline
   or testing outside the competition local area network.

Strict Safety Invariants:
- ORDER EXECUTION IS DISABLED in research/simulation mode.
- Candidate signals are generated, logged, and audited; no live brokerage orders are submitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import socket
import sys
import time
from typing import Any

# Ensure repository root / src and kafka/src are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
KAFKA_SRC = REPO_ROOT / "kafka" / "src"
for d in (SRC_DIR, KAFKA_SRC):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from market_data.config import KafkaConfig
from market_data.consumer import MarketDataConsumer
from market_data.handlers import EventDispatcher
from market_data.models import KafkaMetadata, MarketTickEvent
from nova.quant.contracts import EvaluationStatus, SignalDirection
from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
from nova.research.kafka_connector import KafkaStrategyConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("nova.bot")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NOVA Autonomous Trading Bot — Strategy Simulation Runner"
    )
    parser.add_argument(
        "--simulated-feed",
        action="store_true",
        help="Run using internal streaming candlestick replay (for offline sandbox testing)",
    )
    parser.add_argument(
        "--fast-window",
        type=int,
        default=12,
        help="Fast EMA window (default: 12)",
    )
    parser.add_argument(
        "--slow-window",
        type=int,
        default=26,
        help="Slow EMA window (default: 26)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Limit execution to N ticks (default: continuous until Ctrl+C)",
    )
    parser.add_argument(
        "--tick-delay-seconds",
        type=float,
        default=0.2,
        help="Delay between ticks in simulated feed mode (default: 0.2s)",
    )
    return parser


def check_broker_reachability(host: str, port: int, timeout: float = 2.0) -> bool:
    """Perform quick TCP socket probe to verify broker reachability."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        s.close()
        return False


def run_simulated_stream(
    connector: KafkaStrategyConnector,
    max_ticks: int | None = None,
    delay: float = 0.2,
) -> None:
    """Run continuous streaming simulation feeding synthetic candlestick bars to the strategy."""
    logger.info("Initializing high-fidelity simulated market feed for symbol 'INFY'...")
    base_time = datetime.now(timezone.utc) - timedelta(minutes=60)
    current_price = 1500.0
    tick_count = 0

    try:
        while True:
            tick_count += 1
            if max_ticks is not None and tick_count > max_ticks:
                logger.info("Reached maximum tick limit (%d). Stopping simulation.", max_ticks)
                break

            # Deterministic regime drift: 26 bars warmup -> 20 bars surge -> 15 bars pullback
            if tick_count <= 26:
                pct = 0.0005 * ((tick_count % 3) - 1)
            elif tick_count <= 50:
                pct = 0.004 + 0.001 * (tick_count % 2)
            else:
                pct = -0.003 - 0.001 * (tick_count % 2)

            op = current_price
            cl = round(op * (1.0 + pct), 2)
            hi = round(max(op, cl) * 1.001, 2)
            lo = round(min(op, cl) * 0.999, 2)
            vol = round(1000.0 + (tick_count * 15.0), 0)
            current_price = cl

            ts = base_time + timedelta(minutes=tick_count * 5)
            tick_event = MarketTickEvent(
                symbol="INFY",
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=vol,
                timestamp=ts,
                simulated_date=ts.date(),
                metadata=KafkaMetadata(
                    topic="market.ticks",
                    partition=0,
                    offset=tick_count,
                    timestamp_ms=int(ts.timestamp() * 1000),
                ),
            )

            # Route to connector and strategy
            eval_res = connector.handle_market_tick(tick_event)

            # Visual signal logging
            if eval_res is not None:
                if eval_res.status == EvaluationStatus.CANDIDATE_PRODUCED and eval_res.candidate:
                    cand = eval_res.candidate
                    reason = cand.reason_codes[0] if cand.reason_codes else "SIGNAL"
                    if cand.direction == SignalDirection.BULLISH:
                        logger.info(
                            "[*] [TICK %03d] [%s] Price=%.2f | SIGNAL: BULLISH (Score: %+0.4f) [%s]",
                            tick_count,
                            cand.ticker,
                            cl,
                            cand.signal_score,
                            reason,
                        )
                    else:
                        logger.info(
                            "[*] [TICK %03d] [%s] Price=%.2f | SIGNAL: NEUTRAL (Flat/Exit) [%s]",
                            tick_count,
                            cand.ticker,
                            cl,
                            reason,
                        )
                elif eval_res.status == EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY:
                    history_len = len(connector.get_symbol_history("INFY"))
                    logger.info(
                        "[*] [TICK %03d] [INFY] Price=%.2f | WARM-UP (%d/%d bars)",
                        tick_count,
                        cl,
                        history_len,
                        connector.strategy.config.slow_window,
                    )

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal (Ctrl+C). Stopping simulation gracefully...")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print("=" * 78)
    print("NOVA AUTONOMOUS TRADING BOT -- HACKATHON SIMULATION RUNNER")
    print("Team: Nova Forge (Username: mn2651 | Group: group-mn2651)")
    print("Strategy: Fast/Slow EMA Crossover Trend-Following Baseline")
    print("Safety Status: SIMULATION MODE -- ORDER EXECUTION DISABLED")
    print("=" * 78)

    # 1. Load Kafka Configuration
    kafka_conf = KafkaConfig.load_from_env()
    print(f"[*] Kafka Broker Target:    {kafka_conf.bootstrap_servers}")
    print(f"[*] Consumer Group:         {kafka_conf.group_id}")
    print(f"[*] Subscribed Topic:       {kafka_conf.topic_ticks}")
    print(f"[*] Strategy Configuration: Fast EMA={args.fast_window}, Slow EMA={args.slow_window}")
    print("-" * 78)

    # 2. Initialize Strategy & Ingestion Connector
    strategy_conf = EMACrossoverConfig(
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        allow_short=False,
    )
    strategy = EMACrossoverStrategy(config=strategy_conf)
    connector = KafkaStrategyConnector(strategy=strategy)

    dispatcher = EventDispatcher()
    dispatcher.subscribe_tick(connector)

    # 3. Determine Execution Path (Live vs. Simulated)
    host, port_str = kafka_conf.bootstrap_servers.split(":", 1)
    is_live_reachable = check_broker_reachability(host, int(port_str), timeout=1.5)

    if args.simulated_feed or not is_live_reachable:
        if not is_live_reachable and not args.simulated_feed:
            print(f"[!] Competition broker ({kafka_conf.bootstrap_servers}) is not reachable on this network.")
            print("[*] Launching in AUTOMATIC SIMULATED SANDBOX FEED mode for verification.")
            print("[*] (When connected to the competition venue Wi-Fi/VPN, live broker stream will connect automatically).")
            print("-" * 78)
        run_simulated_stream(connector, max_ticks=args.max_ticks, delay=args.tick_delay_seconds)
    else:
        print(f"[+] Competition broker ({kafka_conf.bootstrap_servers}) reached successfully!")
        print("[*] Starting live Kafka consumer loop...")
        try:
            consumer = MarketDataConsumer(config=kafka_conf, dispatcher=dispatcher)
            consumer.run()
        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting cleanly...")

    # 4. Final Diagnostics Summary
    metrics = connector.metrics
    print("\n" + "=" * 78)
    print("BOT EXECUTION SESSION SUMMARY")
    print("=" * 78)
    print(f"Total Ticks Ingested:           {metrics.ticks_received}")
    print(f"Strategy Evaluations:           {metrics.evaluations_count}")
    print(f"Warm-up Abstentions:            {metrics.warmup_abstentions_count}")
    print(f"Bullish Candidate Signals:      {metrics.bullish_signals_count}")
    print(f"Neutral/Flat Candidate Signals: {metrics.neutral_signals_count}")
    print(f"Dropped Out-of-Order Ticks:     {metrics.ticks_dropped_out_of_order}")
    print(f"Order Placement:                ZERO (Disabled by Safety Invariant)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

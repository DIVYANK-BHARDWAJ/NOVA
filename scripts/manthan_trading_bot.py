#!/usr/bin/env python3
"""MANTHAN 2026 — Official NOVA Trading Bot & Pre-Flight Ingestion Runner.

Entry point for launching NOVA to connect to the MANTHAN 2026 Simulation Platform.
Coordinates:
1. REST API authentication & initial stage/clock/portfolio queries.
2. Kafka market data ingestion (market.ticks & market.news) under group 'group-mn2651'.
3. Asynchronous pre-flight heartbeat worker (POST /api/v1/bot/heartbeat).
4. Continuous simulation lifecycle monitor (GET /api/v1/stage and /api/v1/simulation/clock).
5. Quantitative alpha signal generation (Person 2 EMACrossoverStrategy).
6. Execution Adapter arming & fail-closed stage gating (Person 1).

Usage:
    python scripts/manthan_trading_bot.py
    python scripts/manthan_trading_bot.py --env-file kafka/.env
    python scripts/manthan_trading_bot.py --validate-only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import sys
import time

# 1. Ensure all repository source roots are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
KAFKA_SRC = REPO_ROOT / "kafka" / "src"
EXECUTION_SRC = REPO_ROOT / "execution" / "src"
NOVA_SRC = REPO_ROOT / "src"

for d in (KAFKA_SRC, EXECUTION_SRC, NOVA_SRC, REPO_ROOT):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from market_data.config import ConfigValidationError, KafkaConfig
from market_data.consumer import MarketDataConsumer, PlaceholderBrokerError
from market_data.handlers import (
    DiagnosticLoggingNewsHandler,
    DiagnosticLoggingTickHandler,
    EventDispatcher,
)
from execution.config import ExecutionConfig
from execution.client import TradingApiClient, TradingApiError
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from execution.adapter import ExecutionAdapter


def configure_logging(level_name: str = "INFO") -> None:
    """Configure unified logging format."""
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description="NOVA Autonomous Trading Bot — MANTHAN 2026 Simulation Runner"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env configuration file (default: searches kafka/.env and .env)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Perform read-only preflight verification and exit",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Consume up to N messages and exit (useful for test runs)",
    )
    return parser


def main() -> int:
    """Main execution entrypoint for NOVA bot."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Locate Environment Configuration
    env_path = Path(args.env_file) if args.env_file else None
    if env_path is None:
        candidates = [REPO_ROOT / "kafka" / ".env", REPO_ROOT / ".env"]
        for c in candidates:
            if c.is_file():
                env_path = c
                break

    # 2. Load and Validate Configuration
    k_config = KafkaConfig.load_from_env(env_path)
    ex_config = ExecutionConfig.load_from_env(env_path)

    configure_logging(k_config.log_level)
    logger = logging.getLogger("nova.bot")

    logger.info("=" * 78)
    logger.info("NOVA AUTONOMOUS TRADING BOT — MANTHAN 2026 RUNNER")
    logger.info("Team: Nova Forge (ID: team_m2pccp5) | Group: %s", k_config.group_id)
    logger.info("Trading API: %s | Kafka Broker: %s", ex_config.api_base_url, k_config.bootstrap_servers)
    logger.info("Topics: ticks='%s', news='%s'", k_config.topic_ticks, k_config.topic_news)
    logger.info("=" * 78)

    try:
        k_config.validate()
        ex_config.validate()
    except (ConfigValidationError, ValueError) as exc:
        logger.error("Configuration validation failure: %s", exc)
        return 1

    if k_config.is_placeholder_broker():
        logger.error("Cannot start: KAFKA_BOOTSTRAP_SERVERS is set to placeholder.")
        return 1

    # 3. Initialize REST Client & Query Platform State
    client = TradingApiClient(ex_config)
    adapter = ExecutionAdapter(config=ex_config, client=client)

    logger.info("[INIT] Querying simulation platform status...")
    try:
        stage = client.get_stage()
        clock = client.get_simulation_clock()
        cash = client.get_cash()
        logger.info(
            "[INIT] Stage: '%s' (Status: '%s') | Days: %d/%d | Simulated Date: %s",
            stage.stage,
            stage.status,
            stage.days_completed,
            stage.total_days,
            stage.current_date,
        )
        logger.info(
            "[INIT] Clock: '%s' | Day: %d/%d (Remaining: %d) | Available Cash: %.2f %s",
            clock.status,
            clock.current_day,
            clock.total_days,
            clock.remaining_days,
            cash.available_cash,
            cash.currency,
        )
    except TradingApiError as exc:
        logger.warning("[INIT] Unable to query initial platform state: %s", exc)

    if args.validate_only:
        logger.info("[INIT] --validate-only specified. Preflight checks passed. Exiting cleanly.")
        return 0

    # 4. Initialize Event Dispatcher & Handlers
    dispatcher = EventDispatcher()
    dispatcher.subscribe_tick(DiagnosticLoggingTickHandler())
    dispatcher.subscribe_news(DiagnosticLoggingNewsHandler())

    # 5. Start Pre-Flight Heartbeat Worker
    heartbeat_worker = HeartbeatWorker(client=client)
    heartbeat_worker.start()
    dispatcher.subscribe_preflight(heartbeat_worker.handle_event)
    dispatcher.subscribe_tick(heartbeat_worker.handle_event)
    dispatcher.subscribe_news(heartbeat_worker.handle_event)
    logger.info("[HEARTBEAT] HeartbeatWorker active and listening for pre-flight events.")

    # 6. Start Lifecycle Monitor
    def _on_stage_transition(new_stage, prev_stage):
        adapter.notify_stage_change(new_stage)
        logger.info(
            "[LIFECYCLE EVENT] Stage changed to '%s' (%s). ExecutionAdapter cache updated.",
            new_stage.stage,
            new_stage.status,
        )

    lifecycle_monitor = LifecycleMonitor(
        client=client,
        poll_interval_seconds=3.0,
        on_stage_change=_on_stage_transition,
        heartbeat_worker=heartbeat_worker,
    )
    lifecycle_monitor.start()
    logger.info("[LIFECYCLE] LifecycleMonitor active (polling every 3.0s).")

    # 7. Connect Person 2 Strategy Connector (if available)
    try:
        from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
        from nova.research.kafka_connector import KafkaStrategyConnector

        strat_cfg = EMACrossoverConfig(fast_window=12, slow_window=26, allow_short=False)
        strategy = EMACrossoverStrategy(config=strat_cfg)
        connector = KafkaStrategyConnector(strategy=strategy)
        dispatcher.subscribe_tick(connector.handle_market_tick)
        logger.info("[STRATEGY] Person 2 EMACrossoverStrategy registered to EventDispatcher.")
    except Exception as exc:
        logger.info("[STRATEGY] Strategy module not loaded: %s", exc)

    # 8. Start Kafka Consumer Loop
    consumer = MarketDataConsumer(config=k_config, dispatcher=dispatcher)
    lifecycle_monitor.set_consumer(consumer)

    # Register clean signal handlers
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received signal %s. Shutting down NOVA gracefully...", sig_name)
        consumer.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info("[KAFKA] Starting MarketDataConsumer on %s...", k_config.bootstrap_servers)
    try:
        consumer.run(max_messages=args.max_messages)
    except PlaceholderBrokerError as exc:
        logger.error("Placeholder broker error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting.")
    except Exception as exc:
        logger.critical("Fatal error in consumer: %s", exc, exc_info=True)
        return 2
    finally:
        logger.info("[SHUTDOWN] Stopping background services...")
        lifecycle_monitor.stop()
        heartbeat_worker.stop()
        logger.info("[SHUTDOWN] All services stopped. NOVA process completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

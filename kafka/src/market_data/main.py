"""CLI Entry point for the MANTHAN 2026 Kafka ingestion module.

Starts the Kafka market data consumer with registered diagnostic handlers.
Validates environment settings and prevents accidental connection to placeholder IPs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from market_data.config import ConfigValidationError, KafkaConfig
from market_data.consumer import MarketDataConsumer, PlaceholderBrokerError
from market_data.handlers import (
    DiagnosticLoggingNewsHandler,
    DiagnosticLoggingTickHandler,
    EventDispatcher,
)


def configure_logging(level_name: str = "INFO") -> None:
    """Configure standard Python logging format and level."""
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MANTHAN 2026 Kafka Market Data Ingestion Service"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env configuration file (default: searches kafka/.env and .env)",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit without connecting to Kafka",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Consume up to N messages and exit (useful for integration verification)",
    )
    return parser


def main() -> int:
    """Main execution function.

    Returns:
        Exit code: 0 for normal exit, 1 for configuration error, 2 for runtime error.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Load Configuration
    env_path = Path(args.env_file) if args.env_file else None
    config = KafkaConfig.load_from_env(env_path)

    configure_logging(config.log_level)
    logger = logging.getLogger("market_data.main")

    logger.info("Starting MANTHAN 2026 Kafka Ingestion Layer...")
    logger.info("Loaded Configuration: %s", config.safe_repr())

    # 2. Validate Configuration
    try:
        config.validate()
    except ConfigValidationError as exc:
        logger.error("Configuration validation error: %s", exc)
        return 1

    if args.validate_config:
        logger.info("Configuration is valid.")
        if config.is_placeholder_broker():
            logger.warning(
                "NOTE: KAFKA_BOOTSTRAP_SERVERS is currently set to placeholder '%s'. "
                "Update with the official competition broker IP before starting live ingestion.",
                config.bootstrap_servers,
            )
        return 0

    # 3. Guard Against Placeholder Broker IP
    if config.is_placeholder_broker():
        logger.error(
            "ABORTING: KAFKA_BOOTSTRAP_SERVERS is configured as placeholder '%s'. "
            "Live connection cannot be established until the competition broker address is provided. "
            "To test configuration without connecting, run with --validate-config.",
            config.bootstrap_servers,
        )
        return 1

    # 4. Initialize Dispatcher & Register Diagnostic Handlers
    dispatcher = EventDispatcher()
    dispatcher.subscribe_tick(DiagnosticLoggingTickHandler())
    dispatcher.subscribe_news(DiagnosticLoggingNewsHandler())

    heartbeat_worker = None
    lifecycle_monitor = None
    try:
        from execution.config import ExecutionConfig
        from execution.client import TradingApiClient
        from execution.heartbeat import HeartbeatWorker
        from execution.lifecycle import LifecycleMonitor

        ex_config = ExecutionConfig.load_from_env(env_path)
        client = TradingApiClient(ex_config)

        # 4a. Initialize & Start Preflight Heartbeat Worker
        heartbeat_worker = HeartbeatWorker(client=client)
        heartbeat_worker.start()
        dispatcher.subscribe_preflight(heartbeat_worker.handle_event)
        dispatcher.subscribe_tick(heartbeat_worker.handle_event)
        dispatcher.subscribe_news(heartbeat_worker.handle_event)
        logger.info("HeartbeatWorker successfully started and registered to event dispatcher.")

        # 4b. Initialize & Start Simulation Lifecycle Monitor
        lifecycle_monitor = LifecycleMonitor(
            client=client,
            poll_interval_seconds=3.0,
            heartbeat_worker=heartbeat_worker,
        )
        lifecycle_monitor.start()
        logger.info("LifecycleMonitor successfully started and polling /api/v1/stage and /api/v1/simulation/clock.")

        # 4c. Register Quantitative Alpha Strategy (Person 2)
        try:
            from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
            from nova.research.kafka_connector import KafkaStrategyConnector

            strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=12, slow_window=26))
            connector = KafkaStrategyConnector(strategy=strategy)
            dispatcher.subscribe_tick(connector.handle_market_tick)
            logger.info("Person 2 EMACrossoverStrategy registered to EventDispatcher via KafkaStrategyConnector.")
        except Exception as exc:
            logger.debug("Person 2 strategy connector not loaded: %s", exc)

    except Exception as exc:
        logger.warning("Execution/Lifecycle components not initialized: %s", exc)

    # 5. Run Consumer
    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)
    if lifecycle_monitor:
        lifecycle_monitor.set_consumer(consumer)

    try:
        consumer.run(max_messages=args.max_messages)
    except PlaceholderBrokerError as exc:
        logger.error("Placeholder broker detected: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting cleanly.")
    except Exception as exc:
        logger.critical("Fatal error during consumer execution: %s", exc, exc_info=True)
        return 2
    finally:
        if lifecycle_monitor:
            lifecycle_monitor.stop()
        if heartbeat_worker:
            heartbeat_worker.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

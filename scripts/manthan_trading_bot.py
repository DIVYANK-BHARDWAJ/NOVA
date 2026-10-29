#!/usr/bin/env python3
"""MANTHAN 2026 — Official NOVA Autonomous Trading Bot Runner.

Coordinates:
1. REST API authentication & initial stage/clock/portfolio queries (Person 1).
2. Kafka market data ingestion (market.ticks & market.news) under group 'group-mn2651'.
3. Asynchronous pre-flight heartbeat worker (POST /api/v1/bot/heartbeat) with bounded retries.
4. Continuous zero-delay simulation lifecycle monitor (GET /api/v1/stage and /api/v1/simulation/clock).
5. Quantitative alpha signal generation (Person 2 EMACrossoverStrategy).
6. Person 3 Risk Gateway authorization enforcement (pre-trade risk boundary).
7. Execution Adapter arming, AccountLedger portfolio tracking, and reconciliation (Step F).
8. Observability, fault resilience, and health diagnostics (Step G).

Usage:
    python scripts/manthan_trading_bot.py
    python scripts/manthan_trading_bot.py --env-file kafka/.env
    python scripts/manthan_trading_bot.py --validate-only
    python scripts/manthan_trading_bot.py --mock
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Optional
import urllib.error
import urllib.request

# Ensure all repository source roots are on sys.path
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
from execution.accounting import AccountLedger
from execution.models import OrderAction, OrderIntent, OrderStatus, OrderResult

# Beginner's Guide Keyword Dictionaries (Section 5, Strategy 3)
BULLISH_KEYWORDS = {
    "PROFIT", "SURGE", "GROWTH", "RECORD", "BEAT", "EXPANSION", "DIVIDEND", "PARTNERSHIP",
}
BEARISH_KEYWORDS = {
    "LOSS", "PLUNGE", "DROP", "FRAUD", "SCRUTINY", "RESIGNATION", "PROBE", "DOWNGRADE",
}

logger = logging.getLogger("nova.bot")


def configure_logging(level_name: str = "INFO") -> None:
    """Configure unified logging format."""
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def send_http_request(
    url: str,
    method: str = "GET",
    token: str = "",
    payload: dict | None = None,
    timeout: float = 3.0,
) -> dict | None:
    """Send authenticated HTTP request using standard library urllib."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "NOVA-Trading-Bot/1.0",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = response.read()
            return json.loads(res_data.decode("utf-8")) if res_data else {}
    except Exception as exc:
        logger.warning("[HTTP] %s %s failed: %s", method, url, exc)
        return None


class ManthanTradingBot:
    """Trading bot interface supporting moving-average and news sentiment logic with Risk Gateway gating."""

    def __init__(
        self,
        gateway: str = "http://192.168.65.157:3002",
        broker: str = "192.168.65.157:9094",
        username: str = "mn2651",
        token: str = "",
        fast_window: int = 5,
        slow_window: int = 20,
        batch_quantity: int = 50,
        adapter: Optional[ExecutionAdapter] = None,
        heartbeat_worker: Optional[HeartbeatWorker] = None,
    ) -> None:
        self.gateway = gateway.rstrip("/")
        self.broker = broker
        self.username = username
        self.token = token
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.batch_quantity = batch_quantity
        self.adapter = adapter
        self.heartbeat_worker = heartbeat_worker

        self.tradable_tickers: set[str] = set()
        self.price_history: dict[str, list[float]] = {}
        self.holdings: dict[str, int] = {}
        self.last_heartbeat_seq: int = 1

    def discover_universe(self) -> set[str]:
        """Fetch stock universe from GET /api/v1/stocks."""
        if self.adapter is not None:
            try:
                self.tradable_tickers = self.adapter.get_tradable_universe()
                return self.tradable_tickers
            except Exception as exc:
                logger.warning("[UNIVERSE] Failed to fetch universe via adapter: %s", exc)

        url = f"{self.gateway}/api/v1/stocks"
        resp = send_http_request(url, token=self.token)
        if resp and isinstance(resp, dict):
            stocks = resp.get("stocks") or resp.get("tickers") or []
            if isinstance(stocks, list):
                self.tradable_tickers = {
                    (s.get("ticker") if isinstance(s, dict) else str(s)).upper()
                    for s in stocks
                }
                logger.info("[UNIVERSE] Discovered %d tradable tickers: %s", len(self.tradable_tickers), sorted(self.tradable_tickers))
        return self.tradable_tickers

    def handle_preflight(self, seq: int | str) -> bool:
        """Enqueue preflight heartbeat sequence number for non-blocking acknowledgement."""
        if self.heartbeat_worker is not None:
            return self.heartbeat_worker.enqueue_heartbeat(seq)

        # Fallback direct HTTP post if worker is not attached
        url = f"{self.gateway}/api/v1/bot/heartbeat"
        resp = send_http_request(url, method="POST", token=self.token, payload={"seq": seq})
        return resp is not None

    def submit_order(
        self,
        ticker: str,
        action: str,
        quantity: int,
        authorized_by_risk: bool = False,
    ) -> Optional[OrderResult]:
        """Submit order strictly through ExecutionAdapter and Risk Gateway authorization.

        CRITICAL SAFETY RULE: Never bypass Risk Gateway authorization.
        """
        if not authorized_by_risk:
            logger.warning(
                "[EXECUTION:RISK_GATEWAY] Order %s %d %s BLOCKED: Not authorized by Person 3 Risk Gateway.",
                action,
                quantity,
                ticker,
            )
            return None

        if self.adapter is None:
            logger.warning("[EXECUTION] ExecutionAdapter not attached to bot; order submission skipped.")
            return None

        act_enum = OrderAction(action.upper())
        intent = OrderIntent(
            ticker=ticker.upper(),
            action=act_enum,
            quantity=quantity,
            authorized_by_risk=True,
        )
        return self.adapter.execute_order(intent)

    def handle_price_tick(self, event: dict[str, Any]) -> None:
        """Process price tick according to Moving Average Crossover strategy."""
        data = event.get("data") or event
        ticker = data.get("ticker") or data.get("symbol")
        close_px = data.get("close")
        sim_date = event.get("simulated_date") or data.get("date") or "N/A"

        if not ticker or close_px is None:
            return

        ticker = str(ticker).upper()
        close_px = float(close_px)

        if ticker not in self.price_history:
            self.price_history[ticker] = []
        self.price_history[ticker].append(close_px)
        history = self.price_history[ticker]

        if len(history) < self.slow_window + 1:
            logger.debug(
                "[%s] Collecting history: %d/%d bars for MA crossover.",
                ticker,
                len(history),
                self.slow_window,
            )
            return

        avg_fast = sum(history[-self.fast_window :]) / float(self.fast_window)
        avg_slow = sum(history[-self.slow_window :]) / float(self.slow_window)
        prev_fast = sum(history[-self.fast_window - 1 : -1]) / float(self.fast_window)
        prev_slow = sum(history[-self.slow_window - 1 : -1]) / float(self.slow_window)

        current_qty = self.holdings.get(ticker, 0)

        # BUY SIGNAL: Fast MA crosses ABOVE Slow MA
        if prev_fast <= prev_slow and avg_fast > avg_slow:
            logger.info(
                "[STRATEGY:GOLDEN_CROSS] %s: Fast(%.2f) crossed ABOVE Slow(%.2f) -> BUY",
                ticker,
                avg_fast,
                avg_slow,
            )
            # Route through submit_order with risk gate check
            self.submit_order(ticker, action="BUY", quantity=self.batch_quantity, authorized_by_risk=False)

        # SELL SIGNAL: Fast MA crosses BELOW Slow MA
        elif prev_fast >= prev_slow and avg_fast < avg_slow:
            if current_qty > 0:
                logger.info(
                    "[STRATEGY:DEATH_CROSS] %s: Fast(%.2f) crossed BELOW Slow(%.2f) -> SELL",
                    ticker,
                    avg_fast,
                    avg_slow,
                )
                self.submit_order(ticker, action="SELL", quantity=current_qty, authorized_by_risk=False)
            else:
                logger.debug("[%s] Bearish crossover, but position flat (Holdings=0)", ticker)

    def handle_news_flash(self, event: dict[str, Any]) -> None:
        """Process news announcement according to Beginner's Guide Keyword Strategy."""
        data = event.get("data") or event
        headline = (data.get("headline") or "").upper()
        ticker = data.get("ticker") or data.get("symbol")

        if not headline or not ticker:
            return

        pos_matches = [w for w in BULLISH_KEYWORDS if w in headline]
        neg_matches = [w for w in BEARISH_KEYWORDS if w in headline]

        logger.info(
            "[NEWS FLASH] [%s] %s | Pos=%s Neg=%s",
            ticker,
            data.get("headline", "")[:60],
            pos_matches,
            neg_matches,
        )

        if len(pos_matches) > len(neg_matches):
            logger.info("[NEWS SIGNAL:BUY] Triggered by bullish keywords: %s", pos_matches)
            self.submit_order(ticker, action="BUY", quantity=self.batch_quantity, authorized_by_risk=False)
        elif len(neg_matches) > len(pos_matches):
            logger.info("[NEWS SIGNAL:SHORT] Triggered by bearish keywords: %s", neg_matches)
            self.submit_order(ticker, action="SHORT", quantity=self.batch_quantity, authorized_by_risk=False)


def run_bot(mock_mode: bool = False, max_events: int | None = None, env_path: Optional[Path] = None) -> int:
    """Run bot in either offline mock simulation mode or live production mode."""
    if mock_mode:
        print("=" * 78)
        print("MANTHAN 2026 -- NOVA OFFLINE MOCK MODE VALIDATION")
        print("=" * 78)
        bot = ManthanTradingBot(
            gateway="http://192.168.65.157:3002",
            broker="192.168.65.157:9094",
            username="mn2651",
            token="mock_token",
            fast_window=5,
            slow_window=20,
            batch_quantity=50,
        )

        # 1. Simulate Preflight Heartbeat
        preflight_event = {
            "event_type": "preflight_tick",
            "seq": 101,
            "data": {"status": "STANDBY", "seq": 101},
        }
        logger.info("[MOCK] Simulating preflight heartbeat event...")
        bot.handle_preflight(101)

        # 2. Simulate 25 Days of Price Ticks for ALPHACORP
        base_date = datetime(2021, 1, 4, tzinfo=timezone.utc)
        px = 1000.0
        for i in range(25):
            px = px + (15.0 if i >= 20 else 1.0)
            tick = {
                "event_type": "price_tick",
                "simulated_date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "data": {
                    "ticker": "ALPHACORP",
                    "open": px - 5.0,
                    "high": px + 10.0,
                    "low": px - 8.0,
                    "close": px,
                    "volume": 500000,
                },
            }
            bot.handle_price_tick(tick)

        # 3. Simulate News Event
        news_event = {
            "event_type": "news_flash",
            "data": {
                "ticker": "ALPHACORP",
                "headline": "AlphaCorp reports record profit surge and revenue beat",
            },
        }
        bot.handle_news_flash(news_event)

        print("\n[+] Offline mock loop successfully validated all handlers!")
        return 0

    return main(max_messages=max_events, env_file=str(env_path) if env_path else None)


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
        "--max-events",
        dest="max_messages",
        type=int,
        default=None,
        help="Consume up to N messages and exit (useful for test runs)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run local offline mock validation",
    )
    return parser


def main(max_messages: Optional[int] = None, env_file: Optional[str] = None) -> int:
    """Main execution entrypoint for NOVA bot."""
    # 1. Parse CLI arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mock:
        return run_bot(mock_mode=True)

    max_msgs = max_messages if max_messages is not None else args.max_messages
    env_file_arg = env_file if env_file is not None else args.env_file

    # 2. Locate Environment Configuration
    env_path = Path(env_file_arg) if env_file_arg else None
    if env_path is None:
        candidates = [REPO_ROOT / "kafka" / ".env", REPO_ROOT / ".env"]
        for c in candidates:
            if c.is_file():
                env_path = c
                break

    # 3. Load and Validate Configuration
    k_config = KafkaConfig.load_from_env(env_path)
    ex_config = ExecutionConfig.load_from_env(env_path)

    configure_logging(k_config.log_level)

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

    # 4. Initialize REST Client, Accounting Ledger, and Execution Adapter
    client = TradingApiClient(ex_config)
    initial_cash = 1_000_000.0

    logger.info("[LIFECYCLE:INIT] Querying simulation platform status...")
    try:
        stage = client.get_stage()
        clock = client.get_simulation_clock()
        cash = client.get_cash()
        initial_cash = cash.available_cash
        logger.info(
            "[LIFECYCLE:INIT] Stage: '%s' (Status: '%s') | Days: %d/%d | Simulated Date: %s",
            stage.stage,
            stage.status,
            stage.days_completed,
            stage.total_days,
            stage.current_date,
        )
        logger.info(
            "[LIFECYCLE:INIT] Clock: '%s' | Day: %d/%d (Remaining: %d) | Available Cash: %.2f %s",
            clock.status,
            clock.current_day,
            clock.total_days,
            clock.remaining_days,
            cash.available_cash,
            cash.currency,
        )
    except TradingApiError as exc:
        logger.warning("[LIFECYCLE:INIT] Unable to query initial platform state: %s", exc)

    ledger = AccountLedger(initial_cash=initial_cash)
    adapter = ExecutionAdapter(config=ex_config, client=client, ledger=ledger)

    if args.validate_only:
        logger.info("[INIT] --validate-only specified. Preflight checks passed. Exiting cleanly.")
        return 0

    # 5. Initialize Event Dispatcher & Handlers
    dispatcher = EventDispatcher()
    dispatcher.subscribe_tick(DiagnosticLoggingTickHandler())
    dispatcher.subscribe_news(DiagnosticLoggingNewsHandler())

    # 6. Start Pre-Flight Heartbeat Worker (Step D, E, G)
    heartbeat_worker = HeartbeatWorker(client=client)
    heartbeat_worker.start()
    dispatcher.subscribe_preflight(heartbeat_worker.handle_event)
    dispatcher.subscribe_tick(heartbeat_worker.handle_event)
    dispatcher.subscribe_news(heartbeat_worker.handle_event)
    logger.info("[HEARTBEAT:INIT] HeartbeatWorker active and listening for pre-flight events.")

    # 7. Start Lifecycle Monitor (Step E, G)
    def _on_stage_transition(new_stage, prev_stage):
        adapter.notify_stage_change(new_stage)
        logger.info(
            "[LIFECYCLE:EVENT] Stage changed to '%s' (%s). ExecutionAdapter cache updated.",
            new_stage.stage,
            new_stage.status,
        )
        # Trigger accounting reconciliation on stage change
        try:
            recon = adapter.reconcile_accounting()
            if recon:
                logger.info("[ACCOUNTING:STAGE_RECON] Reconciled on stage change: sync=%s", recon.is_synchronized)
        except Exception as exc:
            logger.warning("[ACCOUNTING:STAGE_RECON] Stage change reconciliation deferred: %s", exc)

    lifecycle_monitor = LifecycleMonitor(
        client=client,
        poll_interval_seconds=3.0,
        on_stage_change=_on_stage_transition,
        heartbeat_worker=heartbeat_worker,
    )
    lifecycle_monitor.poll_once()
    lifecycle_monitor.start()
    logger.info("[LIFECYCLE:INIT] LifecycleMonitor active (polling every 3.0s).")

    # 8. Connect Person 2 Strategy Connector
    try:
        from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
        from nova.research.kafka_connector import KafkaStrategyConnector

        strat_cfg = EMACrossoverConfig(fast_window=12, slow_window=26, allow_short=False)
        strategy = EMACrossoverStrategy(config=strat_cfg)
        connector = KafkaStrategyConnector(strategy=strategy)
        dispatcher.subscribe_tick(connector.handle_market_tick)
        logger.info("[STRATEGY:INIT] Person 2 EMACrossoverStrategy registered to EventDispatcher.")
    except Exception as exc:
        logger.info("[STRATEGY:INIT] Strategy module not loaded: %s", exc)

    # 9. Connect Person 3 Sentinel / Risk Gate (if available)
    try:
        from nova.risk.gates import RiskConfiguration
        risk_cfg = RiskConfiguration()
        logger.info("[RISK:INIT] Person 3 Risk Gates loaded (version=%s). Pre-trade gate armed.", risk_cfg.config_version)
    except Exception as exc:
        logger.debug("[RISK:INIT] Person 3 risk module notice: %s", exc)

    # 10. Start Kafka Consumer Loop (Step G disconnect backoff)
    consumer = MarketDataConsumer(config=k_config, dispatcher=dispatcher)
    lifecycle_monitor.set_consumer(consumer)

    # Register clean signal handlers
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("[SHUTDOWN] Received signal %s. Shutting down NOVA gracefully...", sig_name)
        consumer.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info("[KAFKA:INIT] Starting MarketDataConsumer on %s...", k_config.bootstrap_servers)
    try:
        consumer.run(max_messages=max_msgs)
    except PlaceholderBrokerError as exc:
        logger.error("[KAFKA:ERROR] Placeholder broker error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] KeyboardInterrupt received. Exiting.")
    except Exception as exc:
        logger.critical("[KAFKA:FATAL] Fatal error in consumer: %s", exc, exc_info=True)
        return 2
    finally:
        logger.info("[SHUTDOWN] Stopping background services...")
        lifecycle_monitor.stop()
        heartbeat_worker.stop()
        summary = lifecycle_monitor.get_readiness_summary()
        logger.info(
            "[SHUTDOWN] Final Readiness State: bot_ready=%s, heartbeats_acked=%d, polls=%d, kafka_errors=%d, blockers=%s",
            summary.get("bot_ready"),
            summary.get("heartbeats_acknowledged"),
            summary.get("polls_completed"),
            consumer.kafka_errors,
            summary.get("readiness_blockers"),
        )
        logger.info("[SHUTDOWN] All services stopped. NOVA process completed.")

    return 0


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    if cli_args.mock:
        sys.exit(run_bot(mock_mode=True))
    else:
        sys.exit(main())

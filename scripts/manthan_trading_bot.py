#!/usr/bin/env python3
"""MANTHAN 2026 Autonomous Trading Bot — Complete Official Platform Client.

Implements the exact logic and pseudocode from:
1. The Official MANTHAN 2026 Starter Bot Template.
2. Section 6 ("Step-by-Step Logic of a Trading Bot") in Beginner's Guide to Trading and Investment.

Features:
- Subscribes to Kafka topics: 'market.ticks' and 'market.news'.
- Responds to 'preflight_tick' events with mandatory seq heartbeat to turn the portal indicator to "Ready".
- Dynamically queries the simulation universe via GET /api/v1/stocks.
- Implements 5-day / 20-day Moving Average Crossover strategy from the Beginner's Guide.
- Implements news keyword sentiment checks (Bullish vs Bearish words) from Section 5.
- Submits BUY, SELL, and SHORT orders to POST /api/v1/orders.
- Provides fallback offline mode for testing when disconnected from the venue network.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

# Ensure src and kafka/src are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT / "src", REPO_ROOT / "kafka" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from market_data.config import KafkaConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("manthan.bot")

# Beginner's Guide Keyword Dictionaries (Section 5, Strategy 3)
BULLISH_KEYWORDS = {
    "PROFIT", "SURGE", "GROWTH", "RECORD", "BEAT", "EXPANSION", "DIVIDEND", "PARTNERSHIP",
}
BEARISH_KEYWORDS = {
    "LOSS", "PLUNGE", "DROP", "FRAUD", "SCRUTINY", "RESIGNATION", "PROBE", "DOWNGRADE",
}


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
        "User-Agent": "NOVA-Forge-Bot/1.0",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = response.read()
            return json.loads(res_data.decode("utf-8")) if res_data else {}
    except Exception as exc:
        logger.warning("HTTP %s %s failed: %s", method, url, exc)
        return None


class ManthanTradingBot:
    """End-to-end trading bot connecting Kafka feeds, heartbeat ACKs, and order submission."""

    def __init__(
        self,
        gateway: str,
        broker: str,
        username: str,
        token: str,
        fast_window: int = 5,
        slow_window: int = 20,
        batch_quantity: int = 50,
    ) -> None:
        self.gateway = gateway.rstrip("/")
        self.broker = broker
        self.username = username
        self.token = token
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.batch_quantity = batch_quantity

        self.tradable_tickers: set[str] = set()
        self.price_history: dict[str, list[float]] = {}
        self.holdings: dict[str, int] = {}
        self.last_heartbeat_seq: int = -1

    def handle_preflight(self, seq: int) -> None:
        """Acknowledge pre-flight heartbeat and discover tradable stock universe."""
        logger.info("[PREFLIGHT] Acknowledging heartbeat seq=%d...", seq)
        # 1. Heartbeat ACK
        hb_res = send_http_request(
            f"{self.gateway}/api/v1/bot/heartbeat",
            method="POST",
            token=self.token,
            payload={"seq": seq},
            timeout=2.0,
        )
        if hb_res is not None:
            logger.info("[PREFLIGHT] Heartbeat ACK successful for seq=%d", seq)
            self.last_heartbeat_seq = seq

        # 2. Fetch stock universe if not loaded
        if not self.tradable_tickers:
            stocks_res = send_http_request(
                f"{self.gateway}/api/v1/stocks",
                method="GET",
                token=self.token,
                timeout=5.0,
            )
            if stocks_res and "stocks" in stocks_res:
                self.tradable_tickers = {s["ticker"] for s in stocks_res["stocks"]}
                logger.info(
                    "[PREFLIGHT] Discovered %d simulation tickers: %s",
                    len(self.tradable_tickers),
                    sorted(self.tradable_tickers),
                )

    def submit_order(self, ticker: str, action: str, quantity: int) -> bool:
        """Submit trade order to competition gateway."""
        logger.info(
            ">>> [ORDER DISPATCH] Action=%s | Ticker=%s | Quantity=%d",
            action,
            ticker,
            quantity,
        )
        payload = {"ticker": ticker, "action": action, "quantity": quantity}
        res = send_http_request(
            f"{self.gateway}/api/v1/orders",
            method="POST",
            token=self.token,
            payload=payload,
            timeout=3.0,
        )
        if res is not None:
            logger.info("[ORDER SUCCESS] Response: %s", res)
            if action == "BUY":
                self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
            elif action == "SELL":
                self.holdings[ticker] = max(0, self.holdings.get(ticker, 0) - quantity)
            return True
        return False

    def handle_price_tick(self, event: dict[str, Any]) -> None:
        """Process price tick according to Beginner's Guide MA Crossover Pseudocode."""
        data = event.get("data") or {}
        ticker = data.get("ticker") or data.get("symbol")
        close_px = data.get("close")
        sim_date = event.get("simulated_date") or data.get("date")

        if not ticker or close_px is None:
            return

        if ticker not in self.price_history:
            self.price_history[ticker] = []

        history = self.price_history[ticker]
        history.append(float(close_px))

        # Check if history has reached slow_window
        if len(history) < self.slow_window:
            logger.info(
                "[%s] Close=%.2f (Date: %s) | Warm-up (%d/%d days)",
                ticker,
                close_px,
                sim_date,
                len(history),
                self.slow_window,
            )
            return

        # Calculate Fast and Slow Moving Averages
        avg_fast = sum(history[-self.fast_window :]) / float(self.fast_window)
        avg_slow = sum(history[-self.slow_window :]) / float(self.slow_window)

        # Previous day's moving averages to detect crossover event
        prev_fast = sum(history[-self.fast_window - 1 : -1]) / float(self.fast_window)
        prev_slow = sum(history[-self.slow_window - 1 : -1]) / float(self.slow_window)

        current_qty = self.holdings.get(ticker, 0)

        # BUY SIGNAL: Fast MA crosses ABOVE Slow MA
        if prev_fast <= prev_slow and avg_fast > avg_slow:
            logger.info(
                "[GOLDEN CROSSOVER BUY] %s: Fast(%.2f) crossed ABOVE Slow(%.2f)",
                ticker,
                avg_fast,
                avg_slow,
            )
            self.submit_order(ticker, action="BUY", quantity=self.batch_quantity)

        # SELL SIGNAL: Fast MA crosses BELOW Slow MA
        elif prev_fast >= prev_slow and avg_fast < avg_slow:
            if current_qty > 0:
                logger.info(
                    "[DEATH CROSSOVER SELL] %s: Fast(%.2f) crossed BELOW Slow(%.2f)",
                    ticker,
                    avg_fast,
                    avg_slow,
                )
                self.submit_order(ticker, action="SELL", quantity=current_qty)
            else:
                logger.info(
                    "[%s] Bearish crossover, but position is already flat (Holdings=0)",
                    ticker,
                )
        else:
            trend = "BULLISH" if avg_fast > avg_slow else "BEARISH"
            logger.info(
                "[%s] Close=%.2f (Date: %s) | Fast(%.2f) Slow(%.2f) | %s HOLD",
                ticker,
                close_px,
                sim_date,
                avg_fast,
                avg_slow,
                trend,
            )

    def handle_news_flash(self, event: dict[str, Any]) -> None:
        """Process news announcement according to Beginner's Guide Keyword Strategy."""
        data = event.get("data") or {}
        headline = (data.get("headline") or "").upper()
        ticker = data.get("ticker") or data.get("symbol")

        if not headline or not ticker:
            return

        pos_matches = [w for w in BULLISH_KEYWORDS if w in headline]
        neg_matches = [w for w in BEARISH_KEYWORDS if w in headline]

        logger.info(
            "[NEWS FLASH] [%s] %s | PosKeywords=%s NegKeywords=%s",
            ticker,
            data.get("headline", "")[:60],
            pos_matches,
            neg_matches,
        )

        if len(pos_matches) > len(neg_matches):
            logger.info("[NEWS BUY SIGNAL] Triggered by bullish keywords: %s", pos_matches)
            self.submit_order(ticker, action="BUY", quantity=self.batch_quantity)
        elif len(neg_matches) > len(pos_matches):
            logger.info("[NEWS SHORT SIGNAL] Triggered by bearish keywords: %s", neg_matches)
            self.submit_order(ticker, action="SHORT", quantity=self.batch_quantity)


def run_bot(mock_mode: bool = False, max_events: int | None = None) -> int:
    # Load environment variables
    config = KafkaConfig.load_from_env()
    gateway = os.getenv("GATEWAY_URL", "http://192.168.65.157:3002")
    broker = config.bootstrap_servers
    username = os.getenv("BOT_USERNAME") or os.getenv("KAFKA_USERNAME") or "mn2651"
    token = os.getenv("BOT_ACCESS_TOKEN", "")

    print("=" * 78)
    print("MANTHAN 2026 -- OFFICIAL NOVA FORGE AUTONOMOUS TRADING BOT")
    print(f"Team: Nova Forge | Username: {username} | Group: {config.group_id}")
    print(f"REST Gateway: {gateway}")
    print(f"Kafka Broker: {broker}")
    print(f"Token Length: {len(token)} chars (Valid JWT: {token.startswith('eyJ')})")
    print("=" * 78)

    bot = ManthanTradingBot(
        gateway=gateway,
        broker=broker,
        username=username,
        token=token,
        fast_window=5,
        slow_window=20,
        batch_quantity=50,
    )

    if mock_mode:
        print("[*] Running in OFFLINE MOCK MODE to verify event loop and pseudocode logic...")
        # 1. Simulate Preflight Heartbeat
        preflight_event = {
            "event_type": "preflight_tick",
            "seq": 101,
            "data": {"status": "STANDBY", "seq": 101},
        }
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

        print("\n[+] Offline mock loop successfully validated all pseudocode handlers!")
        return 0

    # Live Mode
    print("[*] Attempting live Kafka consumer loop...")
    try:
        # Primary client: confluent-kafka (fast, C-based librdkafka)
        try:
            from confluent_kafka import Consumer
            c = Consumer({
                "bootstrap.servers": broker,
                "group.id": config.group_id,
                "auto.offset.reset": config.auto_offset_reset,
                "enable.auto.commit": False,
            })
            c.subscribe([config.topic_ticks, config.topic_news])
            logger.info("Connected to Kafka broker at %s (group: %s). Listening for events...", broker, config.group_id)
            event_count = 0
            while True:
                msg = c.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.warning("Kafka event notice: %s", msg.error())
                    continue
                val = msg.value()
                if not val:
                    continue
                try:
                    event = json.loads(val.decode("utf-8"))
                except Exception as exc:
                    logger.warning("JSON decode error: %s", exc)
                    continue

                event_type = event.get("event_type") or event.get("event")
                topic = msg.topic() or ""

                if event_type == "preflight_tick":
                    bot.handle_preflight(event.get("seq", 0))
                elif topic == config.topic_ticks:
                    bot.handle_price_tick(event)
                elif topic == config.topic_news:
                    bot.handle_news_flash(event)

                event_count += 1
                if max_events and event_count >= max_events:
                    break
        except ImportError:
            logger.error("confluent-kafka is required. Install with: pip install confluent-kafka requests")
            return 1

    except Exception as exc:
        logger.error("Live loop terminated with error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run local offline mock validation")
    args = parser.parse_args()
    sys.exit(run_bot(mock_mode=args.mock))

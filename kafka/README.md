# MANTHAN 2026: Kafka Ingestion Module

The `kafka/` module is the dedicated market data ingestion layer for the **MANTHAN 2026** autonomous trading bot.

---

## 1. Scope & Purpose

- **Role**: Continuously ingests streaming candlestick data and corporate news announcements from the competition Kafka broker.
- **Boundaries**: Strictly **strategy-agnostic**. It does **not** make trading decisions, calculate technical indicators (RSI, MACD, Moving Averages), score news sentiment, or place orders.
- **Integration**: Exposes a clean, event-driven interface (`EventDispatcher`, `MarketTickHandlerProtocol`, `MarketNewsHandlerProtocol`) allowing teammates developing strategy, risk, or execution modules to plug in without modifying the ingestion layer.

---

## 2. Directory Structure

```text
kafka/
├── src/
│   └── market_data/
│       ├── __init__.py         # Package exports & clean interface
│       ├── config.py           # Configuration loading & validation
│       ├── consumer.py         # Resilient Kafka consumer loop
│       ├── models.py           # Domain models (MarketTickEvent, MarketNewsEvent)
│       ├── normalizer.py       # JSON decoding, alias conflict checks & normalization
│       ├── validators.py       # OHLC consistency & news validation rules
│       ├── handlers.py         # EventDispatcher & downstream subscriber protocols
│       └── main.py             # CLI entrypoint with startup safeguards
├── tests/
│   ├── __init__.py
│   ├── test_config.py          # Config validation & group ID default tests
│   ├── test_consumer.py        # End-to-end mock Kafka pipeline & lifecycle tests
│   ├── test_handlers.py        # Subscriber dispatch, isolation & safety tests
│   ├── test_main.py            # CLI argument parsing & safety mode tests
│   ├── test_normalizer.py      # Normalization, alias conflicts & date tests
│   └── test_validators.py      # OHLC consistency & news validation tests
├── .env.example                # Documented configuration template
├── .gitignore                  # Git exclusions for env, venv, and cache
├── requirements.txt            # Minimal Python dependencies
├── pyproject.toml              # Build & pytest configuration
└── README.md                   # This documentation
```

---

## 3. macOS Environment Setup & Installation

### Prerequisites
- macOS (Apple Silicon / Intel)
- Python 3.11+ (recommended: Python 3.11 via Homebrew)

### 1. Create and Activate Virtual Environment
```bash
# From workspace root (/Users/harshitrajsingh/nova):
python3.11 -m venv kafka/.venv
source kafka/.venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r kafka/requirements.txt
```

---

## 4. Environment Configuration

Copy the example configuration to `.env`:
```bash
cp kafka/.env.example kafka/.env
```

### Confirmed Competition Settings

| Variable | Configured Value | Description |
| :--- | :--- | :--- |
| `KAFKA_BOOTSTRAP_SERVERS` | `192.168.65.157:9094` | Confirmed competition broker address |
| `KAFKA_SECURITY_PROTOCOL` | `PLAINTEXT` | Competition documented protocol |
| `KAFKA_GROUP_ID` | `group-mn2651` | **Mandatory assigned consumer group ID** |
| `KAFKA_TOPIC_TICKS` | `market.ticks` | Topic for streaming OHLCV candle records |
| `KAFKA_TOPIC_NEWS` | `market.news` | Topic for corporate news announcements |
| `KAFKA_AUTO_OFFSET_RESET` | `latest` | Offset strategy (`latest` for live, `earliest` for replay) |
| `LOG_LEVEL` | `INFO` | Console logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

> **Mandatory Consumer Group ID**:
> The consumer group ID is organizer-mandated in the format `group-${username}`. Our assigned ID is **`group-mn2651`**. It is configured as the default across `config.py` and `.env.example`.

---

## 5. Running Automated Tests

All tests run locally using mocks and synthetic data without needing access to the competition Kafka broker:

```bash
# Run the complete test suite
kafka/.venv/bin/pytest kafka/tests/ -v

# Run with test coverage analysis
kafka/.venv/bin/pytest kafka/tests/ --cov=kafka/src/market_data --cov-report=term-missing

# Run synthetic performance benchmark
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/tests/benchmark_ingestion.py
```

### Verified Test Coverage (86 Automated Tests — 91% Line Coverage)
- **Consumer Group ID**: Confirmed `group-mn2651` as default; confirmed effective client config uses it.
- **Delivery & Lifecycle**: Rebalance callbacks (`on_assign`, `on_revoke`), recoverable Kafka errors, graceful termination, signal handling, and clean close verified.
- **Edge-Case Validation**: Empty payloads (`b""`), whitespace, malformed JSON, invalid UTF-8, JSON arrays, JSON strings, missing OHLC fields, conflicting aliases (`open=100` vs `o=105`), NaN, infinity, extreme numbers ($> 10^9$ price, $> 10^{14}$ volume), invalid dates, and unknown extra fields.
- **Field Distinctness**: `timestamp` (UTC feed clock), `simulated_date` (simulation market date), and `news_date` (article date) are maintained as distinct fields.
- **Zero Fabrication**: Confirmed missing OHLC values are never fabricated or defaulted.
- **Dispatcher & Isolation**: Multi-subscriber dispatch, handler exception isolation (failing subscriber does not block other subscribers), and non-callable handler safety.
- **Security & Integrity**: Zero order submission interfaces, safe representation masking credentials, and .env exclusion.

---

## 5.1. Reproducible Synthetic Performance Benchmark

Run the synthetic benchmark to measure internal ingestion latency:
```bash
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/tests/benchmark_ingestion.py
```

### Benchmark Results (Sample Size: 10,000 Messages)
| Metric | Value |
| :--- | :--- |
| **Sample Size** | 10,000 messages (8,000 price ticks, 2,000 corporate news) |
| **Throughput** | **~57,500 messages/second** |
| **Total Ingestion Time** | 0.174 seconds |
| **Minimum Latency** | 10.79 μs (0.0108 ms) |
| **Median Latency (p50)** | **16.58 μs (0.0166 ms)** |
| **Mean Latency** | 17.32 μs (0.0173 ms) |
| **90th Percentile (p90)** | 18.00 μs (0.0180 ms) |
| **95th Percentile (p95)** | **18.96 μs (0.0190 ms)** |
| **99th Percentile (p99)** | **35.50 μs (0.0355 ms)** |

> **Important Distinction: Synthetic vs. Live Competition Performance**:
> - The synthetic benchmark measures pure local CPU processing latency (JSON decode + normalization + bounds validation + dispatch). Local overhead is sub-millisecond (< 0.02 ms median), consuming less than 0.003% of the competition's 3–4 second window.
> - **In real competition conditions**, latency will be dominated by external factors: network roundtrip time (RTT) to the broker (`192.168.65.157:9094`), broker queueing delay, downstream trading strategy calculations (indicators, order book updates), and order placement network latency to the exchange API. Ingestion speed does not guarantee that total strategy-to-order turnaround will meet the window if downstream handlers or networks are slow.

---

## 6. How to Start the Consumer

### Safe Configuration Check (Dry Run)
Validates your environment settings and verifies configuration:
```bash
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/src/market_data/main.py --validate-config
```

### Live Consumption
```bash
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/src/market_data/main.py
```

To limit consumption to $N$ messages (useful for integration testing):
```bash
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/src/market_data/main.py --max-messages 10
```

To stop the consumer, press `Ctrl+C`. The process traps `SIGINT`/`SIGTERM` and shuts down gracefully after closing Kafka consumer connections.

---

## 7. Event Models and Downstream Integration

Downstream teams (Strategy, Risk, Portfolio) should import from `market_data` and register event handlers with `EventDispatcher`.

### Event Models

#### `MarketTickEvent`
- `symbol: str`: Ticker symbol (e.g. `'ROUT'`).
- `open: float`: Candle open price.
- `high: float`: Candle high price.
- `low: float`: Candle low price.
- `close: float`: Candle close price.
- `volume: float`: Volume traded during the candle.
- `timestamp: Optional[datetime]`: UTC datetime of event emission / feed timestamp.
- `simulated_date: Optional[date]`: Historical market simulation date (e.g. `2024-10-25`).
- `metadata: Optional[KafkaMetadata]`: Transport metadata (`topic`, `partition`, `offset`, `timestamp_ms`).
- `is_bullish: bool`: Helper property (`close >= open`).
- `raw_payload: dict`: Full unparsed JSON dictionary.
- `extra: dict`: Top-level and unmapped metadata (`event_type`, `sig`, `sandbox`, etc.).

#### `MarketNewsEvent`
- `headline: str`: Announcement headline or text.
- `symbol: Optional[str]`: Ticker or company identifier (may be `None` for macro market news).
- `timestamp: Optional[datetime]`: UTC datetime of event emission / feed timestamp.
- `simulated_date: Optional[date]`: Historical market simulation date.
- `news_date: Optional[date]`: Article publication date reported in the news body.
- `metadata: Optional[KafkaMetadata]`: Transport metadata.
- `raw_payload: dict`: Full unparsed JSON dictionary.
- `extra: dict`: Description, source, urgency, etc.

### Critical Distinction: `timestamp` vs. `simulated_date`
- **`timestamp`**: The actual live clock timestamp (e.g. `2026-10-29T10:04:39Z`) when the message was generated or delivered by the competition broker.
- **`simulated_date`**: The historical calendar date (e.g. `2024-10-25`) being replayed inside the competition market simulation.
- Strategies must use `simulated_date` for calendar-based holding periods and chronological indicator windows, and `timestamp` for latency and feed health tracking.

### Downstream Integration Snippet

```python
import sys
from pathlib import Path

# Ensure kafka/src is in python path
sys.path.insert(0, str(Path("kafka/src").resolve()))

from market_data import (
    EventDispatcher,
    KafkaConfig,
    MarketDataConsumer,
    MarketNewsEvent,
    MarketTickEvent,
)

# 1. Define downstream strategy callbacks or classes
def on_market_tick(tick: MarketTickEvent) -> None:
    print(f"[{tick.symbol}] Close={tick.close:.2f} Vol={tick.volume:.0f} SimDate={tick.simulated_date}")

def on_market_news(news: MarketNewsEvent) -> None:
    print(f"[NEWS - {news.symbol or 'GLOBAL'}] {news.headline[:70]} (Date: {news.news_date})")

# 2. Setup Dispatcher and register handlers
dispatcher = EventDispatcher()
dispatcher.subscribe_tick(on_market_tick)
dispatcher.subscribe_news(on_market_news)

# 3. Initialize consumer with configuration
config = KafkaConfig.load_from_env()
consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

# 4. Run consumer loop (blocks until SIGINT / SIGTERM)
# consumer.run()
```

---

## 8. Offset and Delivery Semantics

- **Configuration**: `enable.auto.commit = True`, interval `5000` ms.
- **Delivery Semantics & Failure Modes**:
  1. **Abrupt Process Crash (`SIGKILL`, sudden power failure, OOM)**:
     - Messages processed within the active 5-second auto-commit interval may not have had their offsets committed to the Kafka broker.
     - On restart with `group-mn2651`, the consumer resumes from the last successfully committed offset.
     - **Result**: Those messages will be **redelivered** (duplicate processing). Downstream strategy handlers must implement idempotency (e.g., tracking `(symbol, timestamp)` or `(partition, offset)`).
  2. **Clean Shutdown (`SIGINT`, `SIGTERM`, graceful exit)**:
     - The consumer loop intercepts termination signals, completes the in-flight message dispatch, calls `close()`, and commits the current offset cleanly to the broker.
     - **Result**: No duplicate messages and no dropped messages on restart.
  3. **Unhandled Exception & Finalizer Commit**:
     - If an unhandled exception causes an exit via `finally: self.close()`, librdkafka commits the current polled offset during `close()`. If downstream processing failed midway on that specific message, it will not be re-polled on restart.
  4. **Initial Group Offsets (`auto_offset_reset`)**:
     - `latest` (Default): Joins at the log end; messages generated while the bot was offline before group registration are not consumed.
     - `earliest`: Replays from log start; recommended for replay backtests.
- **Genuine Guarantee**:
  - The module provides **at-least-once processing on abrupt restarts with potential duplicates**, and clean contiguous processing during graceful shutdowns. It does **not** provide exactly-once processing (which would require two-phase transactional commits).

---

## 9. Verified Broker Connectivity & Real Data Feed

| Check | Target | Result | Notes |
| :--- | :--- | :--- | :--- |
| **TCP Connectivity** | `192.168.65.157:9094` | **SUCCESS** | Port 9094 is open and responsive |
| **Kafka Handshake** | `192.168.65.157:9094` | **SUCCESS** | Single broker cluster (ID=1) |
| **Topics Discovered** | `market.ticks`, `market.news` | **SUCCESS** | 10 partitions each |
| **Group Assignment** | `group-mn2651` | **SUCCESS** | Partitions assigned cleanly |
| **Tick Ingestion** | `market.ticks` | **SUCCESS** | Verified live candlestick streaming |
| **News Ingestion** | `market.news` | **SUCCESS** | Verified live corporate news streaming |

---

## 10. Known Limitations

1. **Synchronous Handler Execution**: Downstream handlers execute synchronously within the consumer loop to achieve sub-millisecond dispatch and deterministic in-order processing. However, if any subscriber executes blocking operations (e.g., blocking HTTP requests, database transactions, heavy machine learning inference), the entire consumer loop is stalled.
2. **Duplicate Delivery on Hard Crash**: Because auto-commits occur on a 5000 ms periodic timer, an ungraceful crash (`kill -9`, hardware fault) will result in redelivery of up to 5 seconds of market messages upon restart. Downstream strategy modules must deduplicate records.
3. **Single Broker Resiliency**: The competition broker runs as a single Kafka node (`broker.id=1`). If this broker is taken offline or reboots, the consumer cannot fail over to an alternate broker; it will log recoverable errors and await broker recovery.
4. **Synthetic Benchmark vs. Real Environment**: While local synthetic ingestion latency is ~16 μs (median), real-world latency is subject to LAN/WAN round-trip time, network jitter, competition broker queueing, and the downstream execution pipeline.


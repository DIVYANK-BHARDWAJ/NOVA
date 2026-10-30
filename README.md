<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/Kafka-Confluent_2.3+-231F20?style=for-the-badge&logo=apachekafka&logoColor=white" alt="Kafka"/>
  <img src="https://img.shields.io/badge/XGBoost-3.0+-006600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Tests-500_Passed-brightgreen?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🌌 N O V A</h1>
<h3 align="center">Autonomous Quantitative Trading System</h3>
<p align="center"><em>Engineered for the MANTHAN 2026 Trading Challenge</em></p>

<p align="center">
  <strong>Team Nova Forge</strong> · Group <code>group-mn2651</code>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Team Ownership & Module Boundaries](#team-ownership--module-boundaries)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Running NOVA](#running-nova)
  - [Pre-Flight Validation](#1-pre-flight-validation-recommended-first-step)
  - [Kafka Ingestion](#2-kafka-ingestion-standalone)
  - [Heartbeat & Platform Probes](#3-heartbeat--platform-probes)
  - [Making the Bot Live](#4-making-the-bot-live)
  - [Simulation & Offline Modes](#5-simulation--offline-modes)
- [Testing](#testing)
- [System Components Deep Dive](#system-components-deep-dive)
- [Interface Contracts & JSON Schemas](#interface-contracts--json-schemas)
- [Safety Invariants](#safety-invariants)
- [Configuration Reference](#configuration-reference)
- [Model Artifacts Registry](#model-artifacts-registry)
- [Research & Strategy Scripts](#research--strategy-scripts)

---

## Overview

**NOVA** is a production-grade, fully autonomous quantitative trading system purpose-built for the **MANTHAN 2026** national-level algorithmic trading competition. It ingests real-time streaming market data via **Apache Kafka**, extracts causal quantitative features, applies regime-aware multi-strategy decision logic, enforces deterministic pre-trade risk controls, and executes orders through the competition's REST Trading API — all with **zero human intervention** during live simulation.

### Key Capabilities

| Capability | Description |
| :--- | :--- |
| **Real-Time Kafka Ingestion** | Sub-millisecond OHLCV tick and corporate news ingestion (~57,500 msgs/sec throughput) |
| **Causal Feature Engineering** | 40+ point-in-time indicators across trend, momentum, volatility, volume, and regime families |
| **Dual ML Backend** | Native XGBoost C++ engine with automatic fallback to pure-Python GBDT (zero-dependency) |
| **Market Regime Intelligence** | Transparent KER + EWMA regime detector classifying Trending/Ranging/High-Volatility states |
| **Multi-Strategy Fusion** | Regime-weighted blending of trend-following and mean-reversion strategies |
| **Realistic Friction Modeling** | Slippage, STT, brokerage, DP charges, market impact, and STCG/LTCG tax simulation |
| **Deterministic Risk Gate** | Position limits, drawdown kill-switch, cash buffer enforcement, stale-data rejection |
| **Profit Realization Engine** | Trailing stops, partial scale-outs, profit giveback protection, tax-aware exit deferral |
| **Execution Safety** | Stage gating, timeout reconciliation, duplicate prevention, concurrent submission protection |
| **Asynchronous Heartbeat** | Non-blocking pre-flight sequence acknowledgement on a background daemon thread |
| **Lifecycle Monitoring** | Continuous simulation stage and clock polling with automatic cache invalidation |
| **Portfolio Reconciliation** | Idempotent accounting ledger with remote API position and cash reconciliation |
| **Dynamic Stock Universe** | `GET /api/v1/stocks` with 60s TTL — zero hardcoded symbols anywhere in the codebase |
| **500 Automated Tests** | Full integration suite covering all subsystems in ~9 seconds |

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              N O V A    D A T A F L O W                            ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║  │  MANTHAN 2026 SIMULATION PLATFORM                                          │   ║
║  │  Kafka Broker: 192.168.65.157:9094  │  REST API: 192.168.65.157:3002       │   ║
║  └────────────────┬─────────────────────────────────────────┬──────────────────┘   ║
║                   │ market.ticks / market.news              │ REST Endpoints       ║
║                   ▼                                         ▼                      ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║  │  PERSON 1: INFRASTRUCTURE & INGESTION                         kafka/       │   ║
║  │  ● MarketDataConsumer — resilient Kafka polling with backoff               │   ║
║  │  ● JSON normalizer — alias conflict resolution, OHLCV validation          │   ║
║  │  ● EventDispatcher — fan-out to N subscribers with isolation              │   ║
║  │  ● HeartbeatWorker — async pre-flight sequence acknowledgement            │   ║
║  │  ● LifecycleMonitor — stage/clock polling (3s cadence)                    │   ║
║  └────────────────┬─────────────────────────────────────────────────────────────┘   ║
║                   │ Normalized MarketTickEvent / MarketNewsEvent                    ║
║                   ▼                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║  │  PERSON 2: QUANTITATIVE ALPHA & ML                   src/nova/features/    │   ║
║  │  ● CausalFeaturePipeline — 40+ point-in-time indicators                   │   ║
║  │  ● TransparentRegimeDetector — KER + EWMA regime classification           │   ║
║  │  ● XGBoost / Pure-Python GBDT — forward return prediction                 │   ║
║  │  ● EMACrossoverStrategy — secondary analytical signal generation          │   ║
║  │  ● KafkaStrategyConnector — Kafka-to-Strategy bridge adapter              │   ║
║  │  ▸ Emits: ModelPrediction, CandidateSignal (NEVER executes trades)        │   ║
║  └────────────────┬─────────────────────────────────────────────────────────────┘   ║
║                   │ ModelPrediction + CandidateSignal                               ║
║                   ▼                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║  │  PERSON 3: STRATEGY, RISK & DECISION ENGINE         src/nova/decision/     │   ║
║  │  ● SentinelDecisionEngine — multi-strategy fusion with net-edge scoring    │   ║
║  │  ● TrendFollowingStrategy — EMA crossover + momentum confirmation         │   ║
║  │  ● MeanReversionStrategy — z-score + RSI oscillator dislocation           │   ║
║  │  ● MarketFrictionEngine — all-in cost: spread, STT, impact, DP fees      │   ║
║  │  ● DeterministicRiskGate — position, leverage, drawdown, kill-switch      │   ║
║  │  ● ProfitRealizationManager — trailing stop, scale-out, tax deferral     │   ║
║  │  ▸ Emits: Authorized TradeDecision (OrderIntent)                          │   ║
║  └────────────────┬─────────────────────────────────────────────────────────────┘   ║
║                   │ Authorized TradeDecision                                       ║
║                   ▼                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║  │  PERSON 1: EXECUTION & ACCOUNTING                     execution/           │   ║
║  │  ● ExecutionAdapter — stage-gated order submission via REST API            │   ║
║  │  ● TradingApiClient — typed HTTP client (stdlib urllib, zero dependencies) │   ║
║  │  ● AccountLedger — idempotent fill processing, partial fill tracking      │   ║
║  │  ● Timeout reconciliation — snapshot-based disambiguation, no blind retry │   ║
║  │  ● IntegratedTradingPipeline — end-to-end Kafka→Decision→Execution glue  │   ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Team Ownership & Module Boundaries

NOVA enforces a strict **three-person modular separation of concerns**. No module may violate its ownership boundary.

### Person 1 — Infrastructure, Market Data, Execution & Accounting

| Directory | Responsibility |
| :--- | :--- |
| `kafka/` | Kafka consumer, normalizer, validators, event dispatcher, CLI entrypoint |
| `execution/` | REST API client, execution adapter, heartbeat worker, lifecycle monitor, accounting ledger |
| `schemas/market_event.schema.json` | Canonical market event JSON schema |
| `schemas/execution_event.schema.json` | Canonical execution event JSON schema |

### Person 2 — Quantitative Features, Machine Learning & Regime Intelligence

| Directory | Responsibility |
| :--- | :--- |
| `src/nova/features/` | Causal OHLCV feature engineering (EMA, RSI, ATR, ADX, VWAP, volatility, volume) |
| `src/nova/models/` | XGBoost / Pure-Python GBDT, model config, dataset, metrics, training |
| `src/nova/regime/` | Transparent KER + EWMA regime detector and configuration |
| `src/nova/research/` | Market adapters, EMA crossover strategy, Kafka connector, research evaluator |
| `src/nova/quant/` | Core data contracts (`MarketBar`, `CandidateSignal`, `ModelPrediction`) and configuration |
| `artifacts/` | Model registry, serialized weights, checksums, and evaluation records |

### Person 3 — Strategies, Decision Engine, Friction, Portfolio & Risk

| Directory | Responsibility |
| :--- | :--- |
| `src/nova/strategies/` | Trend-following and mean-reversion policy engines |
| `src/nova/decision/` | Sentinel decision engine, integrated pipeline, causal predictor, regime bridge |
| `src/nova/friction/` | Transaction cost model (slippage, STT, brokerage, market impact, DP charges) |
| `src/nova/portfolio/` | Capital allocation, profit realization manager, trailing stops, tax deferral |
| `src/nova/risk/` | Deterministic pre-trade risk gate (position limits, drawdown, kill-switch) |
| `src/nova/evaluation/` | Historical replay engine, ablation testing, end-to-end evaluation metrics |

---

## Repository Structure

```
nova/
├── kafka/                              # ─── PERSON 1: Kafka Ingestion Layer
│   ├── src/market_data/
│   │   ├── config.py                   #     Configuration loader & validation
│   │   ├── consumer.py                 #     Resilient Kafka consumer with backoff
│   │   ├── handlers.py                 #     EventDispatcher & subscriber protocols
│   │   ├── models.py                   #     MarketTickEvent, MarketNewsEvent domain models
│   │   ├── normalizer.py              #     JSON decode, alias resolution, normalization
│   │   ├── validators.py             #     OHLCV consistency & news validation rules
│   │   └── main.py                    #     CLI entrypoint (--validate-config, --max-messages)
│   ├── tests/                          #     9 test modules (109 tests)
│   ├── .env.example                    #     Documented configuration template
│   ├── requirements.txt                #     confluent-kafka, python-dotenv, pytest
│   └── README.md                       #     Kafka subsystem documentation
│
├── execution/                          # ─── PERSON 1: Execution & Accounting Layer
│   ├── src/execution/
│   │   ├── adapter.py                  #     Stage-gated ExecutionAdapter (~31KB)
│   │   ├── accounting.py              #     Idempotent AccountLedger & reconciliation (~37KB)
│   │   ├── client.py                   #     TradingApiClient (stdlib urllib, zero deps)
│   │   ├── config.py                   #     ExecutionConfig loader
│   │   ├── heartbeat.py               #     Async HeartbeatWorker (threaded, non-blocking)
│   │   ├── lifecycle.py               #     LifecycleMonitor (stage/clock polling)
│   │   └── models.py                   #     OrderResult, ExecutionEvent, SimulationStage models
│   ├── tests/                          #     11 test modules (122 tests)
│   ├── .env.example                    #     Execution config template
│   ├── requirements.txt                #     python-dotenv, pytest
│   └── README.md                       #     Execution subsystem documentation
│
├── src/nova/                           # ─── CORE QUANTITATIVE & STRATEGY ENGINE
│   ├── quant/                          #     Core contracts & interface definitions (~42KB)
│   │   ├── contracts.py                #       MarketBar, CandidateSignal, ModelPrediction, TradeDecision
│   │   └── config.py                   #       Centralized config loader
│   ├── features/                       #     Person 2: Causal feature pipeline (~57KB total)
│   │   ├── pipeline.py                 #       Unified multi-family feature orchestrator
│   │   ├── trend.py                    #       ADX, KER, momentum, trend slope
│   │   ├── moving_averages.py         #       EMA series, crossover features
│   │   ├── mean_reversion.py          #       RSI, z-score, rolling mean/std, VWAP
│   │   ├── volatility.py             #       ATR, realized vol, Parkinson, downside deviation
│   │   ├── volume.py                  #       OBV, volume divergence, relative volume
│   │   └── context.py                 #       Benchmark-relative returns, relative momentum
│   ├── models/                         #     Person 2: ML model engine (~56KB total)
│   │   ├── xgboost_model.py           #       Dual-backend XGBoost + Pure-Python GBDT
│   │   ├── training.py                #       Training loop, walk-forward validation
│   │   ├── dataset.py                 #       Point-in-time dataset construction
│   │   ├── metrics.py                 #       Evaluation metrics (Sharpe, hit rate, R²)
│   │   └── config.py                  #       Model hyperparameter configuration
│   ├── regime/                         #     Person 2: Regime intelligence
│   │   ├── transparent_detector.py    #       KER + EWMA transparent regime estimator
│   │   └── config.py                  #       Regime thresholds & lookback configuration
│   ├── research/                       #     Person 2: Research adapters & strategies
│   │   ├── adapters.py                #       MarketTickEvent → MarketBar conversion
│   │   ├── ema_crossover.py           #       EMACrossoverStrategy (research signal generator)
│   │   ├── kafka_connector.py         #       KafkaStrategyConnector (event→strategy bridge)
│   │   ├── evaluator.py               #       Walk-forward research evaluator
│   │   └── data_collection/           #       Data harvesting utilities
│   ├── strategies/                     #     Person 3: Trading strategy policies
│   │   ├── base.py                    #       BaseStrategy ABC & StrategySignal contract
│   │   ├── trend_following.py         #       EMA crossover + momentum trend strategy
│   │   └── mean_reversion.py          #       Z-score + RSI oscillator mean reversion
│   ├── decision/                       #     Person 3: Decision engine & pipeline
│   │   ├── engine.py                  #       SentinelDecisionEngine (multi-strategy fusion)
│   │   ├── live_engine.py             #       Live-mode decision engine variant
│   │   ├── pipeline.py                #       IntegratedTradingPipeline (end-to-end bridge)
│   │   ├── prediction.py             #       CausalRidgePredictor
│   │   ├── regime.py                  #       MarketRegimeDetector (Person 3 bridge)
│   │   └── contracts.py               #       TradeDecision, PortfolioState, PositionState
│   ├── friction/                       #     Person 3: Transaction cost models
│   │   └── models.py                  #       MarketFrictionEngine (spread, STT, impact, DP)
│   ├── portfolio/                      #     Person 3: Portfolio management
│   │   ├── allocator.py               #       Capital allocation engine
│   │   └── profit_realization.py      #       Trailing stop, scale-out, tax deferral
│   ├── risk/                           #     Person 3: Pre-trade risk controls
│   │   └── gates.py                   #       DeterministicRiskGate (limits, drawdown, kill-switch)
│   └── evaluation/                     #     Person 3: Backtesting & evaluation
│       ├── replay.py                  #       Chronological historical replay engine
│       ├── ablation.py                #       Feature ablation testing framework
│       └── metrics.py                 #       Portfolio metrics (Sharpe, max drawdown, CAGR)
│
├── tests/                              # ─── INTEGRATION & CROSS-SUBSYSTEM TESTS (27 files, 264 tests)
├── scripts/                            # ─── ENTRYPOINTS & RESEARCH SCRIPTS
│   ├── manthan_trading_bot.py         #     🚀 Primary live bot entrypoint (~28KB, 731 lines)
│   ├── start_nova_bot.py              #     Strategy simulation / replay runner
│   ├── probe_endpoints.py             #     Platform REST API discovery & data harvesting
│   ├── train_xgboost_model.py         #     XGBoost model training script
│   ├── run_sentinel_backtest.py       #     Sentinel decision engine backtester
│   ├── run_nova_replay.py             #     Full historical replay evaluator
│   └── ...                             #     Additional audit & research scripts
├── schemas/                            # ─── CANONICAL JSON SCHEMAS
│   ├── market_event.schema.json       #     Market event contract (17KB)
│   ├── execution_event.schema.json    #     Execution event contract (7KB)
│   ├── model_prediction.schema.json   #     ML prediction contract (3.4KB)
│   └── trade_decision.schema.json     #     Trade decision contract (3.2KB)
├── config/                             # ─── CENTRALIZED CONFIGURATION
│   ├── features_config.toml           #     Feature pipeline parameters
│   └── research_config.toml           #     Strategy & research parameters
├── artifacts/                          # ─── MODEL REGISTRY & WEIGHTS
│   ├── models/                        #     Serialized ML model artifacts
│   └── README.md                      #     Model governance & provenance table
├── data/                               # ─── MARKET DATA STORAGE
│   ├── raw/                           #     Raw OHLCV history (API-harvested)
│   └── replay/                        #     Formatted replay datasets
├── docs/                               # ─── ARCHITECTURE & DESIGN DOCUMENTS
│   ├── nova_architecture.md           #     System architecture specification
│   ├── contracts/                     #     Interface contract documentation
│   └── strategies/                    #     Strategy design documents
├── reports/                            # ─── BACKTEST & AUDIT REPORTS
├── pyproject.toml                      #     Build system, pytest, pyright configuration
└── .gitignore
```

**By the numbers:**

| Metric | Value |
| :--- | :--- |
| Source modules | 57 files |
| Source lines | ~17,100 |
| Test modules | 46 files |
| Test lines | ~14,300 |
| Total automated tests | **500 passed** |
| JSON Schema contracts | 4 schemas |
| Configuration files | 2 TOML + 2 `.env.example` |

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core runtime |
| **Streaming** | Apache Kafka (`confluent-kafka ≥2.3`) | Real-time market data ingestion |
| **ML** | XGBoost 3.0+ (native) / Pure-Python GBDT (fallback) | Forward return prediction |
| **HTTP** | `urllib` (stdlib) | Zero-dependency REST API client |
| **Config** | `python-dotenv`, `tomllib` | Environment and parameter management |
| **Testing** | `pytest ≥7.4` | Automated testing framework |
| **Data** | `pandas`, `numpy` | Feature engineering and data manipulation |
| **Build** | `setuptools ≥61.0` | Package build system |

---

## Getting Started

### Prerequisites

- **macOS** (Apple Silicon or Intel) — developed and tested on macOS
- **Python 3.11+** (recommended: install via Homebrew)
- **Network access** to competition LAN (`192.168.65.157`) when running live

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/DIVYANK-BHARDWAJ/NOVA.git
cd nova

# 2. Create and activate the virtual environment
python3.11 -m venv kafka/.venv
source kafka/.venv/bin/activate

# 3. Install all dependencies
pip install --upgrade pip
pip install -r kafka/requirements.txt
pip install -r execution/requirements.txt
pip install xgboost>=3.0.0 pandas numpy python-dotenv

# 4. Verify installation
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/pytest -q
# Expected: 500 passed in ~9s
```

### Environment Configuration

Copy and configure the environment files:

```bash
# Kafka ingestion configuration
cp kafka/.env.example kafka/.env

# Execution adapter configuration
cp execution/.env.example execution/.env
```

Edit `kafka/.env`:

```env
KAFKA_BOOTSTRAP_SERVERS=192.168.65.157:9094
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
KAFKA_GROUP_ID=group-mn2651
KAFKA_TOPIC_TICKS=market.ticks
KAFKA_TOPIC_NEWS=market.news
KAFKA_AUTO_OFFSET_RESET=latest
LOG_LEVEL=INFO
```

Edit `execution/.env` (or `kafka/.env` — the bot searches both):

```env
BOT_ACCESS_TOKEN=<your_competition_token>
TRADING_API_BASE_URL=http://192.168.65.157:3002
HTTP_TIMEOUT_SECONDS=5.0
STAGE_CACHE_TTL_SECONDS=2.0
STOCKS_CACHE_TTL_SECONDS=60.0
LOG_LEVEL=INFO
```

> ⚠️ **Security**: Never commit real `BOT_ACCESS_TOKEN` values. The token is masked in all logs, repr(), and exception tracebacks.

---

## Running NOVA

All commands are executed from the repository root (`/Users/harshitrajsingh/nova`).

### 1. Pre-Flight Validation (Recommended First Step)

Validates Kafka config, REST API connectivity, platform stage, clock, stock universe, and cash balance — **without submitting any orders or entering the live loop**:

```bash
# Kafka configuration dry-run (validates broker, topics, group ID)
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/src/market_data/main.py --validate-config

# Full bot preflight check (REST API, heartbeat, stage, universe, cash)
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python scripts/manthan_trading_bot.py --validate-only
```

### 2. Kafka Ingestion (Standalone)

```bash
# Live consumption from competition broker
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/src/market_data/main.py

# Smoke test: consume 10 messages and exit
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/src/market_data/main.py --max-messages 10

# Synthetic performance benchmark (~57,500 msgs/sec)
PYTHONPATH=kafka/src kafka/.venv/bin/python kafka/tests/benchmark_ingestion.py
```

### 3. Heartbeat & Platform Probes

```bash
# Quick heartbeat ping to competition gateway
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python -c "
from execution.client import TradingApiClient
from execution.config import ExecutionConfig
client = TradingApiClient(ExecutionConfig.load_from_env())
print('Heartbeat:', client.send_heartbeat(seq='1'))
"

# Full platform endpoint discovery (stocks, history, clock, stage)
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python scripts/probe_endpoints.py
```

### 4. Making the Bot Live

```bash
# 🚀 LAUNCH FULL AUTONOMOUS LIVE TRADING BOT
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python scripts/manthan_trading_bot.py

# Launch with explicit .env file
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python scripts/manthan_trading_bot.py --env-file kafka/.env

# Live smoke test: process first 50 ticks, then exit
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python scripts/manthan_trading_bot.py --max-messages 50
```

> Press `Ctrl+C` to gracefully shut down. The bot traps `SIGINT`/`SIGTERM`, flushes pending heartbeats, stops the lifecycle monitor, and reports final readiness state.

### 5. Simulation & Offline Modes

```bash
# Offline mock mode (synthetic ticks, mock execution — no network required)
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/python scripts/manthan_trading_bot.py --mock

# Strategy simulation with candlestick replay (order execution disabled)
PYTHONPATH=kafka/src:src kafka/.venv/bin/python scripts/start_nova_bot.py --simulated-feed
```

---

## Testing

NOVA ships with **500 automated tests** across three subsystems. All tests run locally using mocks and synthetic data — no live broker or API access required.

```bash
# ──── Run the full integrated test suite ────
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/pytest -q
# Expected: 500 passed in ~9s

# ──── Run individual subsystems ────
# Kafka subsystem (109 tests)
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/pytest kafka/tests/ -v

# Execution subsystem (122 tests)
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/pytest execution/tests/ -v

# Core quant, strategies, decision engine, risk, ML (264 tests)
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/pytest tests/ -v

# ──── Run with coverage analysis ────
PYTHONPATH=kafka/src:execution/src:src kafka/.venv/bin/pytest --cov=src --cov=kafka/src --cov=execution/src --cov-report=term-missing
```

### Test Coverage Highlights

| Subsystem | Tests | Coverage Areas |
| :--- | :---: | :--- |
| **Kafka** | 109 | Consumer lifecycle, rebalance, normalizer edge cases, OHLCV validation, dispatcher isolation, empty/malformed/conflicting payloads |
| **Execution** | 122 | Adapter stage gating, timeout reconciliation, duplicate prevention, accounting ledger, heartbeat worker, lifecycle monitor, health/readiness |
| **Integration** | 264 | Feature pipeline causality, XGBoost training/prediction, regime detection, decision engine, risk gate, friction model, replay engine, Kafka→Strategy→Execution pipeline, contract validation |

---

## System Components Deep Dive

### Kafka Ingestion (`kafka/`)

The ingestion layer is **strictly strategy-agnostic** — it normalizes and distributes market events but never makes trading decisions.

- **`MarketDataConsumer`** — Resilient Kafka polling loop with configurable backoff, `SIGINT`/`SIGTERM` handling, rebalance callbacks (`on_assign`, `on_revoke`), and placeholder broker detection
- **`NormalizedMarketData`** — Resolves JSON alias conflicts (e.g., `open` vs `o`), validates OHLCV relationships (`low ≤ open/close ≤ high`), rejects NaN/Infinity/extreme values, and preserves field distinctness (`timestamp` ≠ `simulated_date` ≠ `news_date`)
- **`EventDispatcher`** — Fan-out event distribution to N subscribers with exception isolation (one failing handler never blocks others)

### Execution Adapter (`execution/`)

- **Stage Gating** — Fail-closed policy: unknown stages, query failures, and non-running statuses immediately block order submission
- **Timeout Reconciliation** — Pre-submission snapshot + timestamp discrimination + `(ticker, action, qty)` matching. Never blind-retries. 0 or >1 candidates → `UNKNOWN` with `requires_reconciliation=True`
- **Concurrent Submission Protection** — Fine-grained mutex for microsecond dictionary checks, never held across network calls
- **`AccountLedger`** — Idempotent fill processing, partial fill tracking, remote position/cash reconciliation with variance detection

### Causal Feature Pipeline (`src/nova/features/`)

40+ quantitative indicators computed with strict **point-in-time causality** (bar $t$ uses strictly information $\leq t$):

| Family | Indicators |
| :--- | :--- |
| **Trend** | EMA crossover, momentum (ROC), trend slope, ADX, Kaufman Efficiency Ratio |
| **Mean Reversion** | RSI, z-score, rolling mean/std, VWAP |
| **Volatility** | ATR, realized vol, Parkinson vol, downside deviation, opening gap |
| **Volume** | OBV, price-volume divergence, relative volume ratio |
| **Context** | Benchmark-relative return, relative momentum |

### Regime Intelligence (`src/nova/regime/`)

The `TransparentRegimeDetector` classifies market states into five regimes:

| Regime | Trigger |
| :--- | :--- |
| `TRENDING_BULLISH` | High KER efficiency + upward EMA momentum |
| `TRENDING_BEARISH` | High KER efficiency + downward EMA momentum |
| `RANGING` | Low KER efficiency (oscillatory, mean-reverting) |
| `HIGH_VOLATILITY` | EWMA volatility exceeding threshold |
| `UNKNOWN` | Insufficient warmup history |

### Sentinel Decision Engine (`src/nova/decision/`)

The `SentinelDecisionEngine` fuses multiple strategy signals into a unified trade decision:

1. **Evaluates** trend-following and mean-reversion strategies independently
2. **Applies** regime-conditional weighting (trend weight increases in trending regimes)
3. **Calculates** net expected edge after all-in friction costs
4. **Enforces** minimum edge threshold (configurable, default 1 bps)
5. **Routes** through `DeterministicRiskGate` for position/drawdown/cash validation
6. **Emits** authorized `TradeDecision` with full provenance audit trail

### Risk Gate (`src/nova/risk/`)

Deterministic, configuration-driven pre-trade controls — **no ML model can override these**:

| Control | Default |
| :--- | :--- |
| Max position exposure | 25% per instrument |
| Max gross exposure | 100% |
| Drawdown kill-switch | 15% |
| Cash buffer (absolute minimum) | 5% |
| Cash reserve (MANTHAN cushion) | 20% |
| Stop-loss threshold | 4% |
| Short auto-settlement | 10 trading days |
| Max data staleness | 300 seconds |

---

## Interface Contracts & JSON Schemas

NOVA enforces strict typed contracts across all subsystem boundaries via four canonical JSON schemas in `schemas/`:

| Schema | Owner | Purpose |
| :--- | :--- | :--- |
| `market_event.schema.json` | Person 1 | Validates MarketTickEvent and MarketNewsEvent payloads |
| `execution_event.schema.json` | Person 1 | Validates order submission, fill, rejection, and reconciliation events |
| `model_prediction.schema.json` | Person 2 | Validates ModelPrediction outputs (returns, probabilities, regimes) |
| `trade_decision.schema.json` | Person 3 | Validates TradeDecision actions (buy, sell, hold, short, cover) |

---

## Safety Invariants

These invariants are **frozen** and enforced at every layer of the system:

1. **No Direct Execution by Strategy/Model Code** — Neither Person 2 nor Person 3 may bypass the risk gate or communicate with the exchange API directly. All orders pass through Person 3's risk check → Person 1's execution adapter.

2. **Strict Temporal Causality** — Information from timestamp $T+1$ must **never** influence calculations at timestamp $T$. Any feature, model, or decision violating this invariant is rejected.

3. **Disambiguation of Statistical Quantities** — Returns, calibrated probabilities, and sentiment scores are distinct mathematical objects and must not be treated interchangeably.

4. **Transparent Degradation** — If a complex ML model or external dependency fails, the system degrades gracefully to deterministic heuristic fallbacks without halting trading.

5. **Zero Fabrication** — Missing OHLC values are never fabricated, defaulted, or imputed. Missing data produces explicit validation errors.

6. **No Hardcoded Symbols** — The tradable universe is dynamically discovered via `GET /api/v1/stocks` with 60-second TTL. Zero hardcoded tickers exist anywhere in source code or configuration.

---

## Configuration Reference

### Feature Pipeline — `config/features_config.toml`

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `fast_ema_window` | 12 | Fast EMA lookback |
| `slow_ema_window` | 26 | Slow EMA lookback |
| `rsi_window` | 14 | RSI oscillator period |
| `atr_window` | 14 | Average True Range period |
| `adx_window` | 14 | ADX trend strength period |
| `realized_vol_window` | 20 | Realized volatility lookback |
| `vwap_window` | 20 | Volume-weighted average price period |

### Regime Detector — `config/features_config.toml [regime]`

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `ker_window` | 10 | Kaufman Efficiency Ratio lookback |
| `ker_trend_threshold` | 0.50 | KER above this → trending |
| `ker_ranging_threshold` | 0.25 | KER below this → ranging |
| `volatility_window` | 20 | EWMA volatility lookback |
| `high_volatility_threshold` | 0.025 | Volatility above this → HIGH_VOLATILITY |
| `ewma_lambda` | 0.94 | RiskMetrics EWMA decay factor |
| `min_warmup_bars` | 28 | Minimum bars before regime classification |

### Research Strategies — `config/research_config.toml`

| Strategy | Enabled | Description |
| :--- | :---: | :--- |
| `momentum` | ✅ | Adaptive momentum & trend acceleration |
| `ema_crossover` | ✅ | Fast/slow EMA crossover baseline |
| `relative_strength` | ✅ | Cross-sectional relative strength ranking |
| `regime` | ✅ | Market breadth & regime intelligence |
| `mean_reversion` | ❌ | Regime-conditional mean reversion |
| `breakout` | ❌ | Volatility-adjusted breakout |
| `news_reaction` | ❌ | News-confirmed price reaction |
| `ml_ranking` | ❌ | ML expected-return ranking |

---

## Model Artifacts Registry

| Model ID | Type | Status |
| :--- | :--- | :--- |
| `XGBOOST_NATIVE` | Native C++ XGBoost Regressor | ✅ Trained & Verified |
| `NOVA_XGBOOST_DIRECTIONAL_CLASSIFIER_V1` | Native C++ XGBoost Classifier | ✅ Trained & Verified |
| `NOVA_XGBOOST_DIRECTIONAL_CLASSIFIER_V2` | Native C++ XGBoost Classifier | ✅ Trained & Verified (Reconciled V2) |
| `GBDT_OHLCV_BASELINE` | Custom Pure-Python GBDT | ✅ Trained & Verified (Zero-Dependency) |
| `TIMESFM_ZERO_SHOT` | Foundation Time-Series | 🔄 Planned |
| `FINBERT_NEWS_NLP` | HuggingFace NLP Transformer | 🔄 Planned |

All model artifacts include SHA-256 checksums, training cutoff dates, and deterministic version identifiers. See [`artifacts/README.md`](artifacts/README.md) for the full provenance table.

---

## Research & Strategy Scripts

| Script | Purpose |
| :--- | :--- |
| `scripts/manthan_trading_bot.py` | 🚀 Primary live autonomous bot entrypoint |
| `scripts/start_nova_bot.py` | Strategy simulation with simulated or live feed |
| `scripts/train_xgboost_model.py` | XGBoost model training pipeline |
| `scripts/run_sentinel_backtest.py` | Sentinel decision engine backtester |
| `scripts/run_nova_replay.py` | Full historical replay evaluation |
| `scripts/probe_endpoints.py` | Platform REST API discovery & history download |
| `scripts/run_milestone_7_research.py` | Milestone 7 research pipeline |
| `scripts/validate_regime_empirical.py` | Empirical regime detector validation |
| `scripts/verify_kafka_strategy_integration.py` | Kafka → Strategy integration verification |
| `scripts/harvest_manthan_api_data.py` | Historical OHLCV data harvesting |
| `scripts/demo_causal_feature_pipeline.py` | Feature pipeline demonstration |

---

<p align="center">
  <br/>
  <strong>Nova Forge</strong> · MANTHAN 2026<br/>
  <em>Built with precision. Tested with rigor. Deployed with confidence.</em>
</p>

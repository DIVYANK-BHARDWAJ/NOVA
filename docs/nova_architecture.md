# NOVA — System Architecture Specification
## Three-Person Autonomous Quantitative Trading System
### Authoritative Blueprint for the MANTHAN 2026 Challenge

- **Document Version:** 1.0.0
- **Architectural Status:** ACTIVE & AUTHORITATIVE
- **Project Name:** NOVA
- **Repository:** `https://github.com/DIVYANK-BHARDWAJ/NOVA.git`
- **Development Branch:** `main`

---

## 1. System Mission & Core Operating Philosophy

NOVA is an autonomous, high-frequency-capable quantitative trading architecture engineered specifically for the MANTHAN 2026 trading challenge.

The architecture enforces a strict **three-person modular separation of concerns**:
1. Ingestion and execution infrastructure must remain independent of statistical models.
2. Quantitative feature engineering and predictive models produce standardized inferences (`ModelPrediction`), but **never execute trades directly**.
3. Trading strategies, transaction friction modeling, portfolio utility, and risk management govern all capital allocations and trade decisions (`TradeDecision`).

---

## 2. End-to-End Architectural Dataflow

```
+---------------------------------------------------------------------------------------------------+
|                                          NOVA DATAFLOW                                            |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ EXCHANGE / SIMULATION PLATFORM ]                                                               |
|        |                                                                                          |
|        | (Market Ticks / Candles / Orderbook / News Events via WebSocket, REST or Kafka)          |
|        v                                                                                          |
|  +---------------------------------------------------------------------------------------------+  |
|  |  PERSON 1: Ingestion & Infrastructure (`kafka/` & `src/nova/ingestion/`)                    |  |
|  |  - Ingests streaming events from Kafka / exchange feeds                                      |  |
|  |  - Validates and normalizes raw payloads into `MarketEvent` and text news streams            |  |
|  |  - Enforces clock synchronization, heartbeat tracking, and schema validation                |  |
|  +---------------------------------------------------------------------------------------------+  |
|        |                                                                                          |
|        | (Normalized MarketEvent Stream)                                                          |
|        v                                                                                          |
|  +---------------------------------------------------------------------------------------------+  |
|  |  PERSON 2: Features, Models & Regime Intelligence (`src/nova/{features,models,regime}`)    |  |
|  |  - Causally strict OHLCV feature engineering (moving averages, oscillators, volatility)      |  |
|  |  - Point-in-time Market Regime estimation (Trend / Volatility / Chop)                         |  |
|  |  - Custom ML Models (LightGBM/XGBoost) & Foundation Time-Series Models                       |  |
|  |  - Financial NLP sentiment extraction from news streams                                      |  |
|  |  - Emits: Standardized `ModelPrediction` objects (returns, probabilities, regimes)           |  |
|  |  - STRICT INVARIANT: Zero order execution, zero capital sizing, zero broker actions           |  |
|  +---------------------------------------------------------------------------------------------+  |
|        |                                                                                          |
|        | (ModelPrediction & Feature Indicators)                                                   |
|        v                                                                                          |
|  +---------------------------------------------------------------------------------------------+  |
|  |  PERSON 3: Strategies, Friction, Decision Engine & Risk (`src/nova/{strategies,risk}`)      |  |
|  |  - Strategy Policies: Trend-Following & Regime-Conditional Mean Reversion                   |  |
|  |  - Realistic Transaction Friction Model: Slippage, STT, Brokerage, and Market Impact        |  |
|  |  - Decision Engine: Action Selection & Portfolio Utility Optimization                        |  |
|  |  - Risk Gate: Position Limits, Leverage Limits, Drawdown Guard, Kill Switches                |  |
|  |  - Emits: Authoritative `TradeDecision` actions                                              |  |
|  +---------------------------------------------------------------------------------------------+  |
|        |                                                                                          |
|        | (Authorized TradeDecision Actions)                                                       |
|        v                                                                                          |
|  +---------------------------------------------------------------------------------------------+  |
|  |  PERSON 1: Execution & Accounting (`src/nova/execution/`, `src/nova/accounting/`)          |  |
|  |  - Routes orders to exchange / platform API                                                   |  |
|  |  - Reconciles fills, tracks open orders, and manages position ledger                         |  |
|  |  - Maintains immutable trade history and operational observability                           |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Modular Ownership & Directory Boundaries

### Person 1 — Infrastructure, Market Data, Execution & Accounting
- **Directories:**
  - `kafka/` (Kafka ingestion infrastructure, consumers, normalizers, handlers)
  - `src/nova/ingestion/`
  - `src/nova/execution/`
  - `src/nova/accounting/`
  - `src/nova/observability/`
- **Schemas:**
  - `schemas/market_event.schema.json`
  - `schemas/execution_event.schema.json`

### Person 2 — Quantitative Features, Machine Learning & Regime Intelligence
- **Directories:**
  - `src/nova/features/` (Causal OHLCV feature engineering library)
  - `src/nova/models/` (Custom GBDT, TS Foundation models, NLP sentiment)
  - `src/nova/regime/` (Transparent baseline & advanced regime classifiers)
  - `notebooks/` (Research and ablation notebooks)
  - `scripts/` (Model training and validation entrypoints)
  - `artifacts/README.md` (Model registry documentation)
- **Schemas:**
  - `schemas/model_prediction.schema.json`

### Person 3 — Strategies, Decision Engine, Friction, Portfolio & Risk
- **Directories:**
  - `src/nova/strategies/` (Trend-following and mean-reversion policy engines)
  - `src/nova/decision/` (Friction-aware decision engine)
  - `src/nova/friction/` (STT, exchange fees, slippage, impact models)
  - `src/nova/portfolio/` (Capital allocation and utility sizing)
  - `src/nova/risk/` (Pre-trade risk controls, VaR, drawdown kill-switch)
  - `src/nova/evaluation/` (Integrated end-to-end replay evaluation)
- **Schemas:**
  - `schemas/trade_decision.schema.json`

---

## 4. Frozen Core Invariants

1. **No Direct Execution by Strategy/Model Code:** Neither Person 2 nor Person 3 may bypass the risk gate or communicate with the exchange execution API directly. All orders must pass through Person 3's risk check and Person 1's execution adapter.
2. **Strict Temporal Causality:** Information from timestamp $T+1$ must never influence calculations at timestamp $T$. Any feature, model, or decision violating this invariant is rejected.
3. **Disambiguation of Statistical Quantities:** Returns, calibrated probabilities, and text sentiment scores are distinct mathematical objects and must not be treated interchangeably.
4. **Transparent Degradation:** If a complex ML model or external dependency fails or times out, the system must degrade gracefully to deterministic heuristic fallbacks without halting trading.


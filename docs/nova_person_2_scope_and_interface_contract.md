# NOVA — Person 2 Scope & Interface Contract
## Subsystem: Quantitative Features, ML Models, Time-Series Foundation Models, NLP Sentiment & Regime Estimation
### Status: PROPOSED — VERSION 0 (Pending Cross-Collaborator Contract Freeze)

---

## 1. Person 2 Exclusive Ownership & Boundaries

Under the **NOVA Master Specification**, system ownership is strictly divided among three dedicated engineers:

```
+-------------------------------------------------------------------------------------------------+
|                                         NOVA TOPOLOGY                                           |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [ PERSON 1: Infrastructure, Ingestion, Execution & Accounting ]                              |
|       - Exclusive ownership: `kafka/`, `src/nova/ingestion/`, `src/nova/execution/`,             |
|         `src/nova/accounting/`, `src/nova/observability/`                                       |
|       - Schemas: `market_event.schema.json`, `execution_event.schema.json`                      |
|       - Raw market data parsing, clock synchronization, broker connectivity, fill reconciliation |
|                                     |                                                           |
|                                     v (Normalized Market Events & News Stream)                  |
|   +-----------------------------------------------------------------------------------------+   |
|   |  PERSON 2: Features, Models & Regime Intelligence (OUR EXCLUSIVE DOMAIN)                |   |
|   |   - Exclusive ownership: `src/nova/features/`, `src/nova/models/`,                       |   |
|   |     `src/nova/regime/`, `notebooks/`, `scripts/`, `artifacts/README.md`                 |   |
|   |   - Schema: `schemas/model_prediction.schema.json`                                      |   |
|   |   - Output: Statistical forecasts, probabilities, feature vectors, regime state         |   |
|   |   - STRICT PROHIBITION: CANNOT place orders, size capital, or authorize actions          |   |
|   +-----------------------------------------------------------------------------------------+   |
|                                     |                                                           |
|                                     v (ModelPrediction & Feature Streams)                       |
|   [ PERSON 3: Strategies, Decision Engine, Friction, Portfolio & Risk ]                         |
|       - Exclusive ownership: `src/nova/strategies/`, `src/nova/decision/`,                      |
|         `src/nova/friction/`, `src/nova/portfolio/`, `src/nova/risk/`,                          |
|         `src/nova/evaluation/`                                                                  |
|       - Schema: `schemas/trade_decision.schema.json`                                            |
|       - Strategy trading policies (trend & mean reversion), transaction friction model,          |
|         action selection, portfolio utility, final risk gates, integrated evaluation            |
+-------------------------------------------------------------------------------------------------+
```

### Responsibility Matrix

| Artifact / Operation | Person 1 | Person 2 (Self) | Person 3 |
| :--- | :---: | :---: | :---: |
| Exchange WebSocket / REST / Kafka Connectivity | **OWNER** | Consumer | No Access |
| Normalized Market Data (`MarketEvent`) | **OWNER** | Consumer | Consumer |
| OHLCV Feature Engineering (`src/nova/features/`) | No | **OWNER** | Consumer |
| Machine Learning Models (GBDT, TS, NLP) | No | **OWNER** | Consumer |
| Market Regime Estimation (`src/nova/regime/`) | No | **OWNER** | Consumer |
| Model Prediction Schema (`schemas/model_prediction.schema.json`) | No | **OWNER** | Consumer |
| Trend-Following & Mean-Reversion Strategy Policies | No | Feature Provider | **OWNER** |
| Friction Modeling (Slippage, STT, Brokerage) | Data Input | No | **OWNER** |
| Final Action Selection & Portfolio Sizing | No | **FORBIDDEN** | **OWNER** |
| Pre-Trade Risk Gates & Kill Switches | No | **FORBIDDEN** | **OWNER** |
| Order Routing, Fills & Ledger Accounting | **OWNER** | No Access | Directs |

---

## 2. Model Prediction Contract Proposal (`ModelPrediction`)

The legacy `CandidateSignal` is superseded by `ModelPrediction`. Person 2 produces **pure quantitative hypotheses and statistical inferences**, leaving all transaction decisions to Person 3.

### 2.1 Proposed Schema Specification (`schemas/model_prediction.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://nova-trading.org/schemas/model_prediction.schema.json",
  "title": "ModelPrediction",
  "description": "Standardized prediction output produced by Person 2 quantitative models in NOVA",
  "type": "object",
  "required": [
    "model_id",
    "model_version",
    "prediction_timestamp",
    "input_cutoff_timestamp",
    "instrument_id",
    "prediction_type",
    "horizon_bars",
    "horizon_unit",
    "prediction_value",
    "availability_status",
    "contract_version"
  ],
  "properties": {
    "model_id": {
      "type": "string",
      "minLength": 1,
      "description": "Unique model identifier (e.g. 'GBDT_OHLCV_V1', 'TIMESFM_DENSE_1M', 'FINBERT_SENTIMENT')"
    },
    "model_version": {
      "type": "string",
      "minLength": 1,
      "description": "Semantic version or artifact hash of the trained model"
    },
    "prediction_timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 UTC timestamp when prediction was computed"
    },
    "input_cutoff_timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Point-in-time timestamp: no market information beyond this moment was used (strictly <= prediction_timestamp)"
    },
    "instrument_id": {
      "type": "string",
      "minLength": 1,
      "description": "Target ticker or cross-asset identifier"
    },
    "prediction_type": {
      "type": "string",
      "enum": [
        "EXPECTED_RETURN_DECIMAL",
        "DIRECTIONAL_PROBABILITY",
        "REGIME_PROBABILITY",
        "VOLATILITY_ESTIMATE",
        "SENTIMENT_SCORE"
      ],
      "description": "Exact statistical semantic of prediction_value"
    },
    "horizon_bars": {
      "type": "integer",
      "minimum": 1,
      "description": "Prediction horizon duration in candle bars"
    },
    "horizon_unit": {
      "type": "string",
      "default": "BARS",
      "description": "Time unit for prediction horizon"
    },
    "prediction_value": {
      "type": "number",
      "description": "Primary numerical forecast. Strictly finite (no NaN/inf)"
    },
    "prediction_distribution": {
      "type": ["object", "null"],
      "description": "Optional quantile breakdown (e.g. q10, q50, q90) or distribution parameters",
      "default": null
    },
    "confidence_or_uncertainty": {
      "type": ["number", "null"],
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Model heuristic confidence or uncertainty penalty (distinct from calibrated probability)",
      "default": null
    },
    "availability_status": {
      "type": "string",
      "enum": ["READY", "STALE", "FALLBACK_HEURISTIC", "UNAVAILABLE"],
      "description": "Model inference readiness and data-quality status"
    },
    "metadata": {
      "type": "object",
      "description": "Reproducibility metadata: configuration fingerprint, training dataset ID, random seed",
      "default": {}
    },
    "contract_version": {
      "type": "string",
      "default": "PROPOSED_V0"
    }
  },
  "additionalProperties": false
}
```

---

## 3. Strict Semantic Distinctions

To prevent dangerous misuse of statistical outputs:

1. **Return vs. Probability vs. Sentiment Disambiguation:**
   - `EXPECTED_RETURN_DECIMAL`: Dimensionless price return fraction (e.g. `+0.0150` for `+1.50%`). Never bounded in `[0, 1]`.
   - `DIRECTIONAL_PROBABILITY`: Calibrated probability in `[0.0, 1.0]` that price return is positive.
   - `REGIME_PROBABILITY`: Probability distribution over discrete market regimes (summing to 1.0).
   - `SENTIMENT_SCORE`: Bounded text sentiment in `[-1.0, 1.0]`. **MUST NOT be treated as a direct return forecast.**
   - These quantities are mathematically incompatible and must never be combined without explicit, documented transformation functions.
2. **Zero-Execution Invariant:**
   - `ModelPrediction` contains no order sides (`BUY`/`SELL`), no execution quantities, no broker order types, and no limit prices.
   - It possesses zero trade authorization capability.
3. **Point-in-Time Availability Guard:**
   - All predictions enforce `input_cutoff_timestamp <= prediction_timestamp`.
   - If feature data after `input_cutoff_timestamp` leaked into the feature matrix, the prediction is invalid.

---

## 4. Cross-Collaborator Integration Interfaces

### 4.1 Interface with Person 1 (Infrastructure & Data)
- **Inputs Consumed:**
  - `MarketEvent` stream/cache (`timestamp`, `ticker`, `open`, `high`, `low`, `close`, `volume`, `arrival_timestamp`) from Kafka / REST.
  - News/Text stream (`news_id`, `publication_timestamp`, `headline`, `content`, `affected_tickers`).
- **Required SLA / Assurances:**
  - Clock synchronization within `max_clock_skew_seconds` (default 5s).
  - Explicit flag if market event arrived out of order.

### 4.2 Interface with Person 3 (Strategies, Friction, Portfolio & Risk)
- **Outputs Provided:**
  - Standardized `ModelPrediction` stream.
  - Moving-average crossover features (fast EMA, slow EMA, slope, crossover recency).
  - Mean-reversion features (rolling z-scores, ATR, Bollinger bandwidth).
  - Market regime classifications (`RegimeState`).
- **Downstream Invariants for Person 3:**
  - Person 3 applies friction models (slippage, STT, exchange fees). Person 2 does not deduct fees from statistical return predictions.
  - Person 3 handles abstention (`availability_status == 'UNAVAILABLE'`) by maintaining neutral/hold positions.

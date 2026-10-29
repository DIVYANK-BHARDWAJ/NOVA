# NOVA MAX — Candidate Signal & Strategy Contract Specification
## Subsystem: Quantitative Research & Strategy Engine (Engineer B)
### Contract Document Status: PROPOSED — VERSION 0 (Pending Cross-Engineer Review)

---

## 1. Executive Summary & Ownership Boundaries

This specification defines the formal data contracts, validation invariants, and interface boundaries between **Engineer B (Quantitative Research & Strategy Engine)** and the rest of the NOVA MAX trading system (**Engineer A**, **Engineer C**, and **Engineer D**).

### Immutable Ownership Boundary
```
+-------------------------------------------------------------------------------+
|                             ENGINEER A (Platform & Ingestion)                 |
|  - Ingests raw exchange ticks/bars via WebSocket / REST                       |
|  - Standardizes raw payload into normalized MarketBar                         |
|  - Tags raw timestamp semantics and provenance                                |
+-------------------------------------------------------------------------------+
                                       |
                                       v (MarketBar stream / cache)
+-------------------------------------------------------------------------------+
|                             ENGINEER B (Quantitative Research)                |
|  - Invariant: Zero look-ahead bias, strict point-in-time causality            |
|  - Computes features, regime intelligence, and multi-factor alphas            |
|  - Emits: StrategyEvaluation & CandidateSignal                                |
|  - STRICT PROHIBITION: CANNOT place orders, size capital, or execute trades    |
+-------------------------------------------------------------------------------+
                                       |
                                       v (CandidateSignal stream)
+-------------------------------------------------------------------------------+
|                             ENGINEER C (Portfolio & Risk Gateway)             |
|  - Evaluates CandidateSignal against portfolio constraints & VaR              |
|  - Authorizes / rejects candidate signals                                     |
|  - Determines executable order sizes and routes orders to exchange            |
+-------------------------------------------------------------------------------+
                                       |
                                       v (Audit logs / Invariants)
+-------------------------------------------------------------------------------+
|                             ENGINEER D (QA & Observability)                   |
|  - Invariant testing: temporal integrity, unit compatibility, fuzzing         |
|  - Monitoring: signal freshness, latency, drift, and strategy promotion       |
+-------------------------------------------------------------------------------+
```

---

## 2. Core Data Models

### 2.1 `MarketBar`
Normalized OHLCV representation of an asset candle.

| Field Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `ticker` | `str` | Non-empty string | Ticker symbol (dynamic, not hardcoded) |
| `open` | `float` | Strictly positive (> 0.0) | Opening price in decimal currency (e.g. INR) |
| `high` | `float` | Strictly positive (> 0.0) | High price in decimal currency |
| `low` | `float` | Strictly positive (> 0.0) | Low price in decimal currency |
| `close` | `float` | Strictly positive (> 0.0) | Closing price in decimal currency |
| `volume` | `float` | Non-negative (>= 0.0) | Executed volume in share / contract units |
| `source_timestamp` | `datetime` | Timezone-aware UTC | Source timestamp from exchange feed |
| `available_timestamp` | `datetime \| None` | Timezone-aware UTC | Earliest timestamp this bar was available locally |
| `timestamp_semantics` | `TimestampSemantics` | Enum | Semantic meaning (`OPEN_TIME`, `CLOSE_TIME`, `UNKNOWN`) |
| `data_quality` | `DataQualityStatus` | Enum | `VALID`, `SUSPECT`, `STALE`, `INTERPOLATED`, `INVALID` |
| `provenance` | `str \| None` | Optional identifier | Ingestion feed / batch reference |
| `contract_version` | `str` | Fixed string | `"PROPOSED_V0"` |

#### Validation Invariants:
1. `low <= open <= high` and `low <= close <= high`.
2. Prices strictly positive (`> 0.0`); volume non-negative (`>= 0.0`); booleans strictly rejected.
3. `is_causally_available(as_of: datetime) -> bool`:
   - If `timestamp_semantics == TimestampSemantics.UNKNOWN`, returns `False`.
   - If `available_timestamp is None`, returns `False`.
   - Returns `available_timestamp <= as_of`.

---

### 2.2 `CandidateSignal`
Quantitative research hypothesis produced by an individual strategy.

| Field Name | Type | Unit / Format | Description |
| :--- | :--- | :--- | :--- |
| `strategy_id` | `str` | Non-empty string | Identifier of originating strategy (e.g. `"MOM_ADAPTIVE"`) |
| `strategy_version` | `str` | SemVer string | Strategy code version (e.g. `"1.0.0"`) |
| `ticker` | `str` | Non-empty string | Target asset symbol |
| `direction` | `SignalDirection` | `BULLISH`, `BEARISH`, `NEUTRAL` | Analytical direction (NOT broker order side) |
| `signal_score` | `float` | Finite numeric | Bounded or standardized per `score_semantics` |
| `score_semantics` | `ScoreSemantics` | Enum | Meaning & range (`RAW_INTENSITY` in [-1.0, 1.0], `RANK_DECILE` in [1.0, 10.0], `RANK_PERCENTILE` in [0.0, 1.0], `Z_SCORE`, `EMPIRICAL_EDGE_BPS`, `UNSPECIFIED`) |
| `evaluation_timestamp`| `datetime` | Timezone-aware UTC | Moment this signal was computed |
| `prediction_horizon_bars`| `int \| None` | Positive integer (non-boolean) | Expected holding duration in bar counts |
| `prediction_horizon_unit`| `str` | Default `"BARS"` | Explicit non-empty time unit for prediction horizon |
| `expected_gross_return` | `float \| None` | Dimensionless decimal fraction | Expected price change (e.g. `+0.0150` = `+1.50%`) |
| `trading_cost_estimate` | `float \| None` | Dimensionless decimal fraction | Brokerage + exchange fees (e.g. `0.0020` = `20 bps`) |
| `execution_friction_estimate` | `float \| None` | Dimensionless decimal fraction | Expected slippage / impact (e.g. `0.0010` = `10 bps`) |
| `uncertainty_penalty` | `float \| None` | Dimensionless decimal fraction | Volatility / parameter uncertainty deduction (>= 0.0) |
| `expected_net_edge` | `float \| None` | Dimensionless decimal fraction | Net edge: gross - (cost + friction + penalty) |
| `data_quality` | `DataQualityStatus` | Enum | Data health classification |
| `source_event_refs` | `tuple[str, ...]` | Tuple of strings | Input bar / tick IDs used in calculation |
| `supporting_evidence` | `tuple[tuple[str, float], ...]` | Key-value pairs | Feature snapshot (e.g. `(("rsi_14", 62.5),)`) |
| `invalidation_conditions` | `tuple[str, ...]` | Tuple of strings | Conditions that void signal (e.g. `"SL_BREACH_105.2"`) |
| `reason_codes` | `tuple[str, ...]` | Tuple of strings | Human-readable rationale codes |
| `config_version` | `str \| None` | Optional string | Configuration hash or version reference |
| `model_version` | `str \| None` | Optional string | ML model weight hash if applicable |
| `contract_version` | `str` | Fixed string | `"PROPOSED_V0"` |

#### Critical Invariants:
1. **No Execution Capability:** `CandidateSignal` contains zero order IDs, broker transaction types, quantities, account numbers, or execution methods (`submit_order()` does not exist).
2. **Units Consistency:** All returns and costs are dimensionless decimal return fractions. Monetary fees (e.g., ₹15 flat fee) cannot be directly subtracted without asset price normalization.
3. **No Fabricated Zeros for Net Edge:** If ANY of gross return, trading cost, execution friction, or uncertainty penalty is missing, `compute_expected_net_edge()` returns `None`. Supplying an explicit `expected_net_edge` when any component is missing raises `ContractValidationError`.
4. **Strict Score Semantics Enforced:** `RAW_INTENSITY` is validated in `[-1.0, 1.0]`. `RANK_DECILE` is validated in `[1.0, 10.0]`. `RANK_PERCENTILE` is validated in `[0.0, 1.0]`. Uncalibrated scores must not be represented as probabilities.

---

## 3. Decisions Requiring Cross-Engineer Review

### Review Items for Engineer A (Platform & Ingestion):
1. **Timestamp Semantics:** Confirm whether exchange candles provide start time, end time, or arrival time. Default must remain `UNKNOWN` until verified.
2. **Dynamic Universe Discovery:** Confirm how new symbol additions or intraday halts are communicated (e.g. heartbeat event vs metadata endpoint).
3. **News Feed Specification:** If Strategy 6 (News-Confirmed Price Reaction) is pursued, provide exact JSON/event schema, publication timestamp semantics, and sentiment metadata.

### Review Items for Engineer C (Portfolio & Risk Gateway):
1. **Candidate Ingestion Interface:** Confirm whether Engineer C will consume candidate signals via in-process queue, Python generator, or inter-process message bus.
2. **Net Edge vs Raw Intensity:** Confirm whether Engineer C's portfolio optimizer prefers decimal net edge (`expected_net_edge`) or ranking deciles (`signal_score`).
3. **Abstention Handling:** Confirm that Engineer C interprets missing or abstained candidates as holding current positions (or neutral), never as a forced sell/close.

### Verification Items for Engineer D (QA & Observability):
1. **Temporal Causality Fuzzing:** Automated tests that perturb future bars ($T+1$) and assert bitwise identical output at $T$.
2. **Unit Consistency Checks:** Invariant tests ensuring no dimensional mismatches between currency costs and decimal returns.
3. **Abstention Invariant:** Asserting that `StrategyEvaluation` with `status != CANDIDATE_PRODUCED` never passes a non-None candidate to downstream consumers.

---

## 4. Unresolved Platform Assumptions (MANTHAN 2026)

The following items remain UNVERIFIED and require confirmation from official competition materials:
- Official exchange transaction tax (STT) and fee schedule.
- Intraday short-selling constraints (permitted or prohibited).
- Margin and leverage restrictions.
- Maximum request rate limits and candle history depth.

# NOVA Execution Adapter (MANTHAN 2026)

The **Execution Adapter** connects authorized trading decisions from Person 3's risk gateway to the official MANTHAN 2026 Trading Simulation Platform REST API (`http://192.168.65.157:3002`).

---

## 1. Architectural Boundaries & Risk Gateway Integration

- **Separation of Concerns**:
  - **Person 2 (Quantitative Alpha)** produces `CandidateSignal` hypotheses based on market events.
  - **Person 3 (Risk & Portfolio Gateway)** validates risk limits, portfolio exposure, capital allocation, and outputs authorized `OrderIntent` objects.
  - **Person 1 (Execution Adapter)** enforces stage gating, validates universe membership, prevents concurrent duplicate execution, submits market orders via the official REST API, protects against duplicate execution on timeouts, handles pre-flight heartbeats, and reconciles portfolio positions.

### Risk-Gateway Boundary & Provable Authorization Limitations
- **Current Limitation**:
  In a shared Python runtime, the flag `OrderIntent.authorized_by_risk: bool = True` is an in-memory convention. It does **not** provide cryptographic proof of provenance; any module or thread could instantiate an `OrderIntent(..., authorized_by_risk=True)`.
- **Concrete Recommendations for Person 3**:
  1. **HMAC Risk Tokens**: Person 3's Risk Gateway should sign authorized intents using a session key (`risk_authorization_id = hmac.new(key, payload).hexdigest()`), verified by the adapter before transmission.
  2. **Private Gateway Factory**: Prevent strategy modules from instantiating `OrderIntent` directly. Only `RiskGateway.authorize()` should construct intents, and the `ExecutionAdapter` should be invoked exclusively through the gateway pipeline.
  3. **Timestamp / TTL Guard**: Include a monotonic `created_at` timestamp check to reject intents older than 5 seconds to prevent stale replay.

---

## 2. Execution Safety Mechanisms (Step D.2 Fixes)

### Priority 1: Stage Gating & Transition Correctness
- **Fail-Closed Policy**: Any unknown simulation stage (e.g. `AUCTION`, `MAINTENANCE`, `SETTLEMENT`) or non-running status immediately fails closed (`ORDER_REJECTED`).
- **Query Failures Fail Closed**: If `GET /api/v1/stage` encounters a network error or timeout, order submission is blocked (`STAGE_QUERY_FAILED`).
- **Safe Stage Caching**: `STAGE_CACHE_TTL_SECONDS` defaults to a conservative `2.0s` (configurable, supports `0.0s` for zero-cache mode).
- **Explicit Invalidation**: Calling `invalidate_stage_cache()` or `notify_stage_change()` forces an immediate fresh query on the next order, preventing stale authorizations across stage transitions.
- **Development vs. Scored Competition**: `allow_development_stage: bool = True` (configurable via `ALLOW_DEVELOPMENT_STAGE`). When set to `False`, orders in `development` stage or `is_sandbox_stream=True` are strictly blocked (`STAGE_DEVELOPMENT_BLOCKED`).
- **Official Stage Ambiguity**: The participant portal explicitly specifies `"development"` for sandbox/loop, but exact live tournament stage identifiers (e.g. `"competition"`, `"round_1"`) are not exhaustively documented. Permitted stages are safely configurable via `PERMITTED_STAGES` (default: `"development,running,live,competition"`).

### Priority 2: Timeout Reconciliation Safety & Duplicate Prevention
- **No Blind Retries**: The adapter **never** re-submits an order payload following a timeout or connection drop.
- **Pre-Existing Order Snapshot**: Before sending an order request, the adapter snapshots all previously known order IDs (`_known_order_ids`). Historical orders seen before submission are strictly excluded during reconciliation.
- **Timestamp Discrimination**: Historical orders older than the submission window are ignored.
- **Disambiguation Rule**:
  - Exactly 1 new candidate matching `(ticker, action, quantity)` $\rightarrow$ reconciled to that order.
  - 0 candidates OR $> 1$ candidates $\rightarrow$ remains `OrderStatus.UNKNOWN` with `requires_reconciliation=True`. No guessing!
- **Server Idempotency Limitation**: The official REST API lacks a client-generated idempotency key mechanism (no `client_order_id` in request body or headers). Duplicate prevention across network disconnects is therefore best-effort and heuristic.

### Priority 3: Concurrent Submission Protection
- **Fine-Grained Concurrency Lock**: A lightweight mutex (`self._intent_lock`) synchronizes access to duplicate-sensitive state (`_active_intents`, `_completed_intents`).
- **Zero Network Blocking**: Locks are held strictly for microsecond dictionary/set checks, **never** held across slow network HTTP calls.
- **In-Flight Deduplication**: If multiple threads attempt to execute the same `intent_id` simultaneously, exactly one proceeds; all others are immediately rejected with `CONCURRENT_DUPLICATE_INTENT`.
- **Sequential Duplicate Rejection**: Resubmissions of already completed intents are rejected with `DUPLICATE_INTENT`.

---

## 3. Documented REST Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/v1/orders` | `POST` | Submit market order (`BUY`, `SELL`, `SHORT`, `COVER`) |
| `/api/v1/orders` | `GET` | Retrieve order history (supports `page`, `limit`, `status`, `action`, `ticker`) |
| `/api/v1/portfolio` | `GET` | Retrieve overall account equity, P&L, and open positions |
| `/api/v1/portfolio/cash` | `GET` | Retrieve available cash, cash balance, and margin |
| `/api/v1/stocks` | `GET` | Discover active tradable stock universe |
| `/api/v1/stocks/history` | `GET` | Retrieve historical candles for a stock |
| `/api/v1/simulation/clock` | `GET` | Retrieve simulation day, time remaining, and simulated date |
| `/api/v1/stage` | `GET` | Retrieve platform simulation stage and status |
| `/api/v1/bot/heartbeat` | `POST` | Send pre-flight heartbeat sequence acknowledgement |

---

## 4. Configuration & Security

Configure via environment variables or local `.env`:

```env
BOT_ACCESS_TOKEN=<your_competition_token>
TRADING_API_BASE_URL=http://192.168.65.157:3002
HTTP_TIMEOUT_SECONDS=5.0
STAGE_CACHE_TTL_SECONDS=2.0
STOCKS_CACHE_TTL_SECONDS=60.0
ALLOW_DEVELOPMENT_STAGE=true
PERMITTED_STAGES=development,running,live,competition
LOG_LEVEL=INFO
```

> **Security Guarantee**:
> `BOT_ACCESS_TOKEN` is strictly masked in `repr()`, string representations, logs, and exception tracebacks. Headers are sanitized before logging.

---

## 5. Event Schema & Inter-Module Contracts (`schemas/execution_event.schema.json`)

To enable strict contract validation across Person 1 (Execution), Person 2 (Quant Alpha), and Person 3 (Risk Gateway), all execution lifecycle events conform to the canonical JSON Schema defined at:

**Canonical Schema Path**: `schemas/execution_event.schema.json`

### Event Envelope Structure
* `event_type`: (Required string enum) One of:
  - `ORDER_SUBMISSION_ATTEMPT`: Adapter has started submission pipeline.
  - `ORDER_ACCEPTED`: Order acknowledged and placed on exchange books (`PENDING`/`SUBMITTED`).
  - `ORDER_FILLED`: Order filled completely by exchange.
  - `ORDER_PARTIALLY_FILLED`: Order partially executed.
  - `ORDER_REJECTED`: Order rejected by risk gateway, validation, stage gating, or exchange API.
  - `ORDER_UNCERTAIN`: Order timed out; state unknown, no blind retry, requires reconciliation.
  - `ORDER_CANCELLED`: Order cancelled.
* `timestamp`: (Required string) UTC ISO-8601 formatted event emission timestamp.
* `order_result`: (Required object) Full `OrderResult` outcome payload.
  - `client_intent_id`: Correlates with `OrderIntent.intent_id`.
  - `ticker`: Uppercase symbol (e.g. `ROUT`).
  - `action`: `BUY`, `SELL`, `SHORT`, or `COVER`.
  - `quantity`: Positive integer (`>= 1`).
  - `status`: `PENDING`, `SUBMITTED`, `FILLED`, `PARTIALLY_FILLED`, `REJECTED`, `CANCELLED`, `UNKNOWN`.
  - `order_id`: Server order ID string (or `null`).
  - `execution_price`: Fill price float (or `null`).
  - `filled_quantity`: Integer filled (`>= 0`).
  - `charges`: Float fees/commissions in INR (`>= 0.0`).
  - `rejection_reason`: Error description string (or `null`).
  - `requires_reconciliation`: Boolean flag (`true` on `UNKNOWN` / ambiguous states).
  - `timestamp`: UTC ISO-8601 timestamp.
* `extra`: (Optional object)
  - `strategy_id`: Alpha strategy identifier from Person 2.
  - `risk_authorization_id`: Risk token or HMAC signature from Person 3.

### Safety Semantics Enforced by Schema
1. **UNKNOWN Disambiguation**: When `status == "UNKNOWN"`, `requires_reconciliation` MUST be `true`.
2. **Rejection Provenance**: When `status == "REJECTED"`, `rejection_reason` MUST be a non-empty string.
3. **Confirmed Fill Invariant**: When `status == "FILLED"`, `filled_quantity` MUST be an integer `>= 1`.
4. **Timeout Disambiguation**: When `event_type == "ORDER_UNCERTAIN"`, `status` MUST be `UNKNOWN`.

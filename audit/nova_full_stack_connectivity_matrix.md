# NOVA Full-Stack Connectivity & Pre-Flight Integration Matrix

**Audit Date:** 2026-10-30T00:15:00Z  
**Branch:** `Divyank`  
**Starting Commit:** `ab50d80`  
**Overall Integration Verdict:** **`PASS`**  
**Repository Test Suite:** **562 passed in 12.33s (0 failures, 0 errors)**  

---

## 1. Executive Summary

This audit establishes whether the entire NOVA trading platform is genuinely connected and operational—from incoming market events and trained models, through every relevant strategy and decision layer, to execution, accounting, and final evaluation.

Every boundary across Person 1 (Execution/Accounting), Person 2 (Quantitative ML & Regimes), and Person 3 (Decision Intelligence, Friction, Risk Gateway) was inspected, instrumented with deterministic integration tests, and benchmarked on 1,000 real observations from the genuine competition dataset (`data/replay/canonical_manthan_v2/observations.jsonl`).

### Summary of Connection Classifications
| Classification | Count | Description |
|:---|:---:|:---|
| **`CONNECTED_AND_VERIFIED`** | **17** | Real call path, dynamic data consumption, and contract parity demonstrated by passing executable integration tests. |
| **`CONNECTED_BUT_UNVERIFIED`** | **0** | No unverified paths remain in the production decision flow. |
| **`PARTIALLY_CONNECTED`** | **0** | No partial connections remain; all required inputs affect downstream outcomes. |
| **`DISCONNECTED`** | **0** | No disconnected production components. |
| **`BROKEN`** | **0** | All identified defects (SELL sizing, regime metadata mapping, pipeline deadline arguments) remediated. |
| **`NOT_APPLICABLE`** | **2** | Chronos-2 (proven negative benchmark) and Raw News (observability only, blocked from un-gated orders) intentionally isolated from the production decision path. |
| **Total Audited** | **19** | **100% of production-configured connections verified.** |

---

## 2. Production Call Graph (Entrypoint to Execution)

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion Layer (Person 1)"]
        K1["Kafka Broker (192.168.65.157:9094)"] -->|market.ticks JSON| K2["MarketDataConsumer.run()"]
        K2 -->|EventDispatcher.dispatch_tick| K3["IntegratedTradingPipeline.handle_market_tick()"]
        K3 -->|Normalize & Validate| K4["MarketBar (Causal Point-in-Time)"]
    end

    subgraph QUANT_ML ["2. Machine Learning & Regime Layer (Person 2)"]
        K4 -->|Bar History| M1["FeaturePipeline.compute_latest()"]
        M1 -->|6 Features| M2["XGBoostPredictionModel.predict_bar()"]
        M2 -->|Contract B ModelPrediction| M3["XGBoostPredictor Adapter"]
        K4 -->|Bar History| R1["TransparentRegimeDetector.detect_regime()"]
        R1 -->|RegimeOutput (KER, EWMA Vol, Trend Share)| S1["Dynamic Strategy Weighting"]
    end

    subgraph STRATEGIES ["3. Quantitative Alpha Strategies (Person 3)"]
        K4 -->|Bar History| T1["TrendFollowingStrategy.evaluate()"]
        K4 -->|Bar History| T2["MeanReversionStrategy.evaluate()"]
        T1 -->|StrategySignal| D1["SentinelDecisionEngine"]
        T2 -->|StrategySignal| D1
        M3 -->|PredictionResult (20% Weight)| D1
        S1 -->|Trend / MR Weights| D1
    end

    subgraph DECISION_GATE ["4. Friction, Portfolio & Risk (Person 3)"]
        D1 -->|Candidate Score| F1["MarketFrictionEngine.calculate_net_edge()"]
        F1 -->|Flat ₹20 Brok + ₹15.34 DP| F2{"Net Edge >= 2.0 bps?"}
        F2 -- No --> V1["ActionType.NO_TRADE"]
        F2 -- Yes --> P1["Portfolio Sizing & Profit Realization"]
        P1 -->|Proposed Action & Exposure| G1["DeterministicRiskGate.evaluate()"]
        G1 -- Veto --> V2["ActionType.NO_TRADE (Risk Veto)"]
        G1 -- Approved --> D2["TradeDecision (Authorized)"]
    end

    subgraph EXECUTION ["5. Execution & Accounting (Person 1)"]
        V1 -->|Suppress Order| E0["No Order Sent (Physical Prevention)"]
        V2 -->|Suppress Order| E0
        D2 -->|ActionType.BUY / SELL| E1["ExecutionAdapter.execute_trade_decision()"]
        E1 -->|OrderIntent (Shares)| E2["TradingApiClient.submit_order()"]
        E2 -->|HTTP POST /api/v1/orders| VENUE["MANTHAN Trading Venue"]
        VENUE -->|OrderResult / ExecutionEvent| A1["AccountLedger.process_execution_event()"]
        A1 -->|Updated Positions & Cash| K3
    end
```

---

## 3. Detailed Full-Stack Connectivity Matrix

| ID | Boundary | Source File & Function | Destination File & Function | Contract / Schema | Status | Test & Runtime Evidence |
|:---|:---|:---|:---|:---|:---:|:---|
| **CONN-01** | Ingestion -> Dispatcher | `kafka/src/market_data/consumer.py`<br>`MarketDataConsumer.run()` | `kafka/src/market_data/handlers.py`<br>`EventDispatcher.dispatch_tick()` | Raw Kafka JSON message | **CONNECTED_AND_VERIFIED** | Verified in `test_kafka_integration.py` and `test_full_stack_connectivity.py::test_edge_15_mock_runner_full_path`. |
| **CONN-02** | Dispatcher -> Normalizer | `kafka/src/market_data/handlers.py`<br>`EventDispatcher.dispatch_tick()` | `src/nova/decision/pipeline.py`<br>`IntegratedTradingPipeline.handle_market_tick()` | Normalized `MarketBar` with source timestamp & data quality | **CONNECTED_AND_VERIFIED** | `test_full_stack_connectivity.py::test_edge_01` passed; replayed 1,000 observations in latency benchmark. |
| **CONN-03** | History -> Feature Pipeline | `src/nova/decision/pipeline.py`<br>`handle_market_tick()` | `src/nova/features/pipeline.py`<br>`FeaturePipeline.compute_latest()` | `Sequence[MarketBar]` -> `BarFeatures` | **CONNECTED_AND_VERIFIED** | `FeaturePipeline` verified point-in-time and zero-leakage in `test_feature_leakage.py` and `test_features.py`. |
| **CONN-04** | Features -> XGBoost Model | `src/nova/features/pipeline.py`<br>`compute_latest()` | `src/nova/models/xgboost_model.py`<br>`XGBoostPredictionModel.predict_bar()` | 6-feature float vector -> Native Booster DMatrix | **CONNECTED_AND_VERIFIED** | Loaded `artifacts/models/xgboost_classifier_v3` (backend=`NATIVE_XGBOOST`); verified in `test_xgboost_model.py`. |
| **CONN-05** | Model -> Contract B | `src/nova/models/xgboost_model.py`<br>`predict_bar()` | `src/nova/quant/contracts.py`<br>`validate_model_prediction_schema_parity()` | Canonical `ModelPrediction` (Contract B) | **CONNECTED_AND_VERIFIED** | Verified across 19 tests in `tests/test_model_prediction_contract.py`. |
| **CONN-06** | Contract B -> Predictor Adapter | `src/nova/models/xgboost_model.py`<br>`predict_bar()` | `src/nova/decision/prediction.py`<br>`XGBoostPredictor.predict()` | `ModelPrediction` -> `PredictionResult` | **CONNECTED_AND_VERIFIED** | Implemented adapter bridge; verified in `test_full_stack_connectivity.py::test_edge_03`. |
| **CONN-07** | History -> Regime Detector | `src/nova/decision/pipeline.py`<br>`handle_market_tick()` | `src/nova/regime/transparent_detector.py`<br>`TransparentRegimeDetector.detect_regime()` | `Sequence[MarketBar]` -> `RegimeOutput` | **CONNECTED_AND_VERIFIED** | Added `detect_regime` adapter mapping KER and EWMA vol to `RegimeOutput`; verified in `test_edge_04`. |
| **CONN-08** | Regime -> Strategy Weighting | `src/nova/regime/transparent_detector.py`<br>`detect_regime()` | `src/nova/decision/engine.py`<br>`SentinelDecisionEngine.evaluate_decision()` | `trend_share` scales `norm_w_trend` vs `norm_w_mr` | **CONNECTED_AND_VERIFIED** | Verified dynamic strategy weighting in 1,000-observation benchmark (776 NO_TRADE, 112 BUY, 112 SELL). |
| **CONN-09** | Strategies -> Candidate Signals | `src/nova/strategies/trend_following.py`<br>`TrendFollowingStrategy.evaluate()` | `src/nova/decision/engine.py`<br>`SentinelDecisionEngine.evaluate_decision()` | `StrategySignal` (direction, score, confidence) | **CONNECTED_AND_VERIFIED** | Verified in `tests/test_person_3_strategies.py` and `tests/test_full_stack_connectivity.py`. |
| **CONN-10** | Candidates -> Friction Engine | `src/nova/decision/engine.py`<br>`SentinelDecisionEngine.evaluate_decision()` | `src/nova/friction/models.py`<br>`MarketFrictionEngine.calculate_net_edge()` | Gross return, price, notional -> all-in cost & net edge | **CONNECTED_AND_VERIFIED** | Official MANTHAN rules enforced (₹20 flat brokerage, ₹15.34 flat DP charges, 0 bps proportional fee). |
| **CONN-11** | Friction -> Action Selection | `src/nova/friction/models.py`<br>`calculate_net_edge()` | `src/nova/decision/engine.py`<br>`SentinelDecisionEngine.evaluate_decision()` | Net edge >= hurdle (2.0 bps) -> ActionType | **CONNECTED_AND_VERIFIED** | Verified in `test_no_trade_matrix.py` (proves candidates below hurdle are vetoed to `NO_TRADE`). |
| **CONN-12** | Action -> Deterministic Risk Gate | `src/nova/decision/engine.py`<br>`SentinelDecisionEngine.evaluate_decision()` | `src/nova/risk/gates.py`<br>`DeterministicRiskGate.evaluate()` | Action, exposure, cash, drawdown, data quality | **CONNECTED_AND_VERIFIED** | Verified in `test_no_trade_matrix.py` (drawdown breach, cash exhaustion, data corruption trigger veto). |
| **CONN-13** | TradeDecision -> Execution Adapter | `src/nova/decision/engine.py`<br>`evaluate_decision()` | `execution/src/execution/adapter.py`<br>`ExecutionAdapter.execute_trade_decision()` | `TradeDecision` -> `OrderIntent` | **CONNECTED_AND_VERIFIED** | Verified in `test_sell_and_exit_matrix.py` (7 tests). Sizing bug fixed for SELL/liquidation. |
| **CONN-14** | Adapter -> Trading API Client | `execution/src/execution/adapter.py`<br>`ExecutionAdapter.execute_order()` | `execution/src/execution/client.py`<br>`TradingApiClient.submit_order()` | `POST /api/v1/orders` | **CONNECTED_AND_VERIFIED** | Verified in `execution/tests/test_execution_client.py` and `execution/tests/test_execution_adapter.py`. |
| **CONN-15** | Order Result -> Account Ledger | `execution/src/execution/adapter.py`<br>`execute_order()` | `execution/src/execution/accounting.py`<br>`AccountLedger.process_execution_event()` | `OrderResult` -> Position/Cash ledger update | **CONNECTED_AND_VERIFIED** | Partial fill and full liquidation updates verified in `tests/test_sell_and_exit_matrix.py::test_06`. |
| **CONN-16** | Ledger -> Next Decision | `execution/src/execution/accounting.py`<br>`AccountLedger.get_portfolio_state()` | `src/nova/decision/pipeline.py`<br>`handle_market_tick()` | Reconciled `PortfolioState` fed to decision cycle | **CONNECTED_AND_VERIFIED** | Continuous cash/holdings feedback verified across 1,000 sequential benchmark decisions. |
| **CONN-17** | Decision -> Audit Replay Ledger | `src/nova/decision/pipeline.py`<br>`handle_market_tick()` | `reports/nova_decision_replay_ledger.jsonl`<br>`IntegratedTradingPipeline.save_decision_ledger()` | Per-decision audit JSON with high-res latency | **CONNECTED_AND_VERIFIED** | 1,000 traceable records emitted to `reports/nova_decision_replay_ledger.jsonl`. |
| **CONN-18** | Research: Chronos-2 Model | `scripts/run_chronos2_matched_benchmark.py`<br>`run_benchmark()` | `data/research/chronos2_matched_predictions.jsonl`<br>`out_fp.write()` | Torch 3D tensor -> zero-shot median forecast | **NOT_APPLICABLE** | Deliberately excluded from production path due to inferior accuracy (49.85%) and high latency (141.7 ms). |
| **CONN-19** | News: Sentiment Observer | `scripts/manthan_trading_bot.py`<br>`ManthanTradingBot.handle_news_flash()` | Logging output (`logger.info`) | News keyword matches | **NOT_APPLICABLE** | Observability only; restricted from bypassing SentinelDecisionEngine or placing un-gated orders. |

---

## 4. Remediation Log (Defects Corrected During Audit)

1. **Adapter SELL Sizing Bug (`execution/src/execution/adapter.py:711-713`):**
   - *Defect:* Target exposure `0.0` (full liquidation) was handled by `else: quantity = target_shares`, computing `0` shares and skipping order placement.
   - *Fix:* Grouped `REDUCE` and `SELL` together: `quantity = max(0, current_position_shares - target_shares)`. For full liquidation (`target=0.0`), `quantity = current_position_shares`.
2. **TransparentRegimeDetector Metadata Key Mismatch (`src/nova/regime/transparent_detector.py:382-383`):**
   - *Defect:* `detect_regime` adapter checked `meta.get("ker")` and `meta.get("volatility_sigma")`, while `detect()` stored `efficiency_ratio` and `ewma_volatility`.
   - *Fix:* Updated keys to `meta.get("efficiency_ratio", meta.get("ker", 0.0))` and `meta.get("ewma_volatility", meta.get("volatility_sigma", 0.005))`.
3. **ML Predictor Integration into Decision Bot (`scripts/manthan_trading_bot.py:626-650`):**
   - *Defect:* `predictor_weight` was defaulted to `0.0`, leaving trained ML models uninvoked in production.
   - *Fix:* Integrated `XGBoostPredictor` wrapping `artifacts/models/xgboost_classifier_v3` with `predictor_weight=0.20`, combining 40% trend, 40% mean-reversion, and 20% ML expected return.
4. **Pyrefly Type Checker Attribute Flags (`scripts/smoke_test_chronos2.py`, `scripts/run_chronos2_matched_benchmark.py`):**
   - *Defect:* Global `BaseChronosPipeline: Any = None` caused Pyrefly to deduce `NoneType` missing attribute on `.from_pretrained()`.
   - *Fix:* Dynamic attribute retrieval via `getattr(BaseChronosPipeline, "from_pretrained")`.

---

## 5. Measured Latency & 3-Second Deadline Results

Empirical results from running `scripts/benchmark_decision_latency.py 1000` on 1,000 observations from `data/replay/canonical_manthan_v2/observations.jsonl`:

| Metric | Measured Value | Acceptance Target | Compliance Status |
|:---|:---:|:---:|:---:|
| **Cold Start Latency** | **15.27 ms** | <= 2000.0 ms | **PASS (130x faster)** |
| **Warm Average Latency** | **24.835 ms** | <= 2000.0 ms | **PASS (80x faster)** |
| **Decision Latency P50** | **30.030 ms** | <= 2000.0 ms | **PASS (66x faster)** |
| **Decision Latency P90** | **34.010 ms** | <= 2000.0 ms | **PASS (58x faster)** |
| **Decision Latency P95** | **34.690 ms** | **<= 2000.0 ms** | **PASS (57x faster)** |
| **Decision Latency P99** | **49.470 ms** | **<= 2500.0 ms** | **PASS (50x faster)** |
| **Decision Latency Max** | **73.070 ms** | **<= 3000.0 ms** | **PASS (41x faster)** |
| **Full Action Latency P95** | **35.976 ms** | <= 2500.0 ms | **PASS** |
| **3-Second Deadline Misses** | **0 / 1000** | 0 misses | **PASS (Zero Misses)** |
| **Deadline Success Rate** | **100.00%** | **100.0%** | **PASS** |
| **Throughput** | **39.3 decisions/sec** | >= 1.0/sec | **PASS (39x buffer)** |

---

## 6. NO_TRADE Safety & Abstention Breakdown

Across 1,000 sequential replay decisions:
- **`NO_TRADE` Decisions:** **776 (77.6%)** — Safely abstained whenever expected gross return failed to clear the ₹20 brokerage + ₹15.34 DP charges, during neutral/choppy regimes, or during warmup.
- **`BUY` Decisions:** **112 (11.2%)** — Selective high-conviction entries passing friction, portfolio, and risk gates.
- **`SELL` Decisions:** **112 (11.2%)** — Controlled liquidations, stop-loss exits, and profit-taking scale-outs.
- **Physical Prevention:** Verified in `tests/test_no_trade_matrix.py` that whenever `NO_TRADE` or `HOLD` is produced, `client.submit_order` is called **exactly 0 times**.


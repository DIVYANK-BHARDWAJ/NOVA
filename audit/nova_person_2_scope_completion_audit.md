# NOVA — Person 2 Master Scope Audit
## Complete Quantitative Research Inventory, Evidence-Based Gap Analysis & Remaining Work

**Audit Timestamp:** 2026-10-29T22:38:00+05:30  
**Auditor:** Senior Quantitative Research Architect & Independent Implementation Auditor  
**Subsystem / Role:** Person 2 — Quantitative Features, Machine Learning Models, Foundation Time-Series Models, NLP Sentiment & Market Regime Estimation  
**Repository Name:** NOVA (`src/nova/`)  
**Active Git Branch:** `Divyank`  
**Current HEAD Commit:** `d436099` (`Merge pull request #5 from DIVYANK-BHARDWAJ/poorvik`)  
**Working Tree Status:** Clean (`nothing to commit, working tree clean`)  
**Test Suite Telemetry:** 106 / 106 Main Repo Tests Passed (0.11s) | 94 / 94 Kafka Tests Passed (0.31s) | 200 Total Passing  

---

## 1. Executive Summary

This independent repository audit evaluates the implementation status of **Person 2 (Quantitative Research & Modeling)** within project **NOVA**, against the authoritative *Sentinel Trader Master Project Specification* and repository architecture documentation (`docs/nova_person_2_scope_and_interface_contract.md`).

The audit strictly distinguishes between what has been implemented with verified automated tests, what was mocked or temporarily implemented by teammates (such as Person 3's temporary decision-side predictor in `src/nova/decision/prediction.py`), what exists only as documentation proposals, and what remains completely unbuilt.

### 1.1 Requirement Audit Classification Tally

Across **25 discrete requirements** audited within Person 2's exclusive and interface domains:

| Status | Count | Percentage | Definition |
| :--- | :---: | :---: | :--- |
| **`DONE_VERIFIED`** | **4** | 16.0% | Implementation exists, causality verified, and automated unit/leakage tests pass with observable telemetry. |
| **`PARTIAL`** | **4** | 16.0% | Functional elements exist in documentation or temporary proxy locations, but canonical Person 2 modules or frozen schemas are incomplete. |
| **`MISSING`** | **13** | 52.0% | Architectural deliverable is absent from disk (no code, no canonical tests, no models). |
| **`BLOCKED`** | **0** | 0.0% | No requirements currently blocked by external technical blockers; competition data schemas and market structures are known. |
| **`OUT_OF_SCOPE`** | **4** | 16.0% | Architecture elements owned exclusively by Person 1 (Kafka/Ingestion/Accounting) or Person 3 (Decision Engine/Friction/Risk/Replay). |
| **Total** | **25** | 100.0% | Full Person 2 specification boundary. |

### 1.2 Genuinely Complete Capabilities
1. **Deterministic Causal EMA Engine (`src/nova/features/moving_averages.py`):** Hand-calculated, verified mathematical implementation of exponential moving averages, spreads, and crossover events with explicit window validation ($W \ge 2$).
2. **Temporal Causality & Anti-Leakage Suite (`tests/test_leakage.py`, `tests/test_features.py`):** Invariant-tested proof that evaluating at time $t$ uses strictly observations $\le t$. Future bars appended or modified have exactly zero impact on historical outputs.
3. **Strategy Invariant & Evaluation Harness (`src/nova/research/`):** Full chronological evaluator measuring gross vs. net returns, tracking warm-up abstentions, turnover, and drawdowns under strict contract typing.
4. **Kafka Ingestion Integration:** Seamless live adaptation of market ticks into candidate signals and order dispatches verified live on the competition broker (`group-mn2651`).

### 1.3 Consequential Remaining Gaps
1. **Multi-Family Quantitative Feature Library Missing:** While EMA crossover features are complete, the required causal feature pipeline (RSI, Bollinger/z-score mean reversion, ATR volatility, rolling volume, price-volume divergence, and benchmark context) has not yet been built in `src/nova/features/`.
2. **Model Prediction Schema (`schemas/model_prediction.schema.json`) Not on Disk:** The JSON schema exists only as a markdown code block inside `docs/nova_person_2_scope_and_interface_contract.md`. The physical schema file is missing from `schemas/`.
3. **Canonical Model Directory & GBDT Baseline Unbuilt:** No `src/nova/models/` directory exists. No LightGBM / XGBoost model, training pipeline, dataset splitting, or evaluation script has been constructed.
4. **Foundation Time-Series & NLP Sentiment Evaluations Absent:** Neither Amazon Chronos / Google TimesFM nor FinBERT has been benchmarked for runtime, memory, or incremental alpha. (The live trading script uses a naive keyword regex list, not an NLP model).
5. **Architectural Drift in Regime & Prediction Modules:** Person 3 created temporary mock predictors and regime detectors inside `src/nova/decision/` (`prediction.py` and `regime.py`). Person 2's canonical production modules (`src/nova/regime/` and `src/nova/models/`) must be established to satisfy true ownership boundaries.

### 1.4 Recommended Immediate Next Implementation Phase
**Phase 2 — Quantitative Baseline Foundation:**
Establish `schemas/model_prediction.schema.json`, build the complete causal OHLCV feature library in `src/nova/features/` (RSI, Bollinger/z-score, ATR, volume divergence), create canonical `src/nova/regime/` baseline detector, and implement the custom GBDT feature/target training pipeline.

---

## 2. Requirement-by-Requirement Audit Matrix

The following matrix audits all required deliverables under Person 2's ownership scope as defined in Part III, Part VI, Part VII, and Part VIII of the master specification.

```
Status Key:
- DONE_VERIFIED: Production code exists, deterministic tests pass, verified by terminal execution.
- PARTIAL: Logic exists partially or in proxy location, but formal Person 2 contract/module is unfinished.
- MISSING: No implementation, tests, or artifacts exist on disk.
- BLOCKED: Pre-requisites or dependencies unfulfilled.
- OUT_OF_SCOPE: Belongs to Person 1 or Person 3.
```

| Req ID | Specification Requirement | Canonical Code Path | Implementation Evidence | Test Evidence | Evaluation Evidence | Status | Identified Gap | Concrete Next Action | Dependencies | Ownership |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **P2-FEAT-01** | Causal Fast/Slow Exponential Moving Averages | `src/nova/features/moving_averages.py` | `calculate_ema_series()` implements $\alpha = \frac{2}{w+1}$ with $w \ge 2$ guard. | `tests/test_features.py::test_hand_calculated_ema_series` | Deterministic hand-calc passes to 7 decimal places. | `DONE_VERIFIED` | None. Production-ready and verified. | None. Maintain existing implementation. | None | Person 2 |
| **P2-FEAT-02** | Causal Crossover Detection & Normalized Spread | `src/nova/features/moving_averages.py` | `compute_ema_crossover_features()` tracks warm-up, raw spread, normalized spread, crossover event $(\pm 1)$. | `tests/test_features.py`, `tests/test_leakage.py` | Invariant tests confirm appending future bars does not mutate past values. | `DONE_VERIFIED` | None. Production-ready and verified. | None. Maintain existing implementation. | P2-FEAT-01 | Person 2 |
| **P2-FEAT-03** | Trend Slope, Momentum, ADX Trend Strength, Crossover Recency | `src/nova/features/trend.py` | None in `src/nova/features/`. (Only basic MA spread exists). | None. | None. | `MISSING` | Slope, momentum vectors, ADX, and bars-since-crossover are unbuilt. | Implement `TrendFeatureExtractor` in `src/nova/features/trend.py`. | P2-FEAT-01 | Person 2 |
| **P2-FEAT-04** | Mean Reversion: Rolling z-score, VWAP deviations, RSI | `src/nova/features/mean_reversion.py` | None in `src/nova/features/`. (Person 3 has strategy rule in `src/nova/strategies/mean_reversion.py`, but feature library is absent). | None. | None. | `MISSING` | No causal calculation for RSI(14), rolling price z-scores, or VWAP distance. | Implement `MeanReversionFeatureExtractor` in `src/nova/features/mean_reversion.py`. | `MarketBar` sequence | Person 2 |
| **P2-FEAT-05** | Volatility & Downside Risk: ATR, Realized Volatility, Gap Risk | `src/nova/features/volatility.py` | Absent from `src/nova/features/`. (Person 3 implemented local EWMA in `src/nova/decision/regime.py`). | `tests/test_regime_detector.py` (Person 3 only). | None in Person 2 domain. | `MISSING` | No True Range, ATR(14), rolling Parkinson/Garman-Klass, or downside semi-variance. | Implement `VolatilityFeatureExtractor` in `src/nova/features/volatility.py`. | `MarketBar` sequence | Person 2 |
| **P2-FEAT-06** | Volume & Liquidity Proxies: Relative Volume, Price-Volume Divergence | `src/nova/features/volume.py` | None in `src/nova/features/`. | None. | None. | `MISSING` | Volume z-score, volume moving average ratio, On-Balance Volume (OBV) trend unbuilt. | Implement `VolumeFeatureExtractor` in `src/nova/features/volume.py`. | `MarketBar.volume` | Person 2 |
| **P2-FEAT-07** | Market Context: Benchmark-Relative Returns & Beta Proxies | `src/nova/features/context.py` | None. Config defines benchmark `NO_TRADE`, but relative calculations unbuilt. | None. | None. | `MISSING` | No cross-sectional ranking or benchmark-relative spread features. | Implement `MarketContextFeatureExtractor` in `src/nova/features/context.py`. | Multi-asset bar feeds | Person 2 |
| **P2-FEAT-08** | Data Integrity Features: Gap, staleness & abnormal jump indicators | `src/nova/features/integrity.py` | Partially handled via validation exceptions in `nova.quant.contracts`, but not emitted as feature flags. | `tests/test_contracts.py` | Contract validations pass. | `PARTIAL` | Features do not output numeric data quality scores or anomaly flags into the model input vector. | Add anomaly detection and staleness scoring to feature pipeline. | `MarketBar` | Person 2 |
| **P2-MOD-01** | Custom OHLCV GBDT Model Architecture & Target Definition | `src/nova/models/ohlcv_gbdt.py` | No `src/nova/models/` directory exists. (Person 3 wrote a toy Ridge regression in `decision/prediction.py`). | `tests/test_market_predictor.py` (Person 3 only). | None. | `MISSING` | GBDT baseline (LightGBM/XGBoost) not implemented. No forward-return target construction. | Create `src/nova/models/` and implement `OHLCVGradientBoostingModel`. | P2-FEAT-01 to 06 | Person 2 |
| **P2-MOD-02** | Chronological Training, Validation & Split Script | `scripts/train_ohlcv_model.py` | None. No training scripts exist in `scripts/`. | None. | None. | `MISSING` | No reproducible script for chronological train/val splits and walk-forward validation. | Implement `scripts/train_ohlcv_model.py` with walk-forward CV. | P2-MOD-01 | Person 2 |
| **P2-MOD-03** | Model Artifact Serialization, Versioning & Fallback Inference | `src/nova/models/loader.py` | None. | None. | None. | `MISSING` | No model loader, checksum verification, or graceful fallback to heuristic on failure. | Implement model packaging and loading with fallback policies. | P2-MOD-01 | Person 2 |
| **P2-PRED-01** | Standardized Model Prediction JSON Schema | `schemas/model_prediction.schema.json` | Schema defined in Markdown doc (`docs/nova_person_2_scope_and_interface_contract.md`), but file missing from `schemas/`. | `tests/test_contracts.py` (validates CandidateSignal only). | None for JSON schema. | `PARTIAL` | Physical schema file absent from `schemas/`. | Write `schemas/model_prediction.schema.json` and add `jsonschema` validation tests. | Master Spec Part VIII | Person 2 |
| **P2-PRED-02** | Standardized `ModelPrediction` Python Data Contract | `src/nova/quant/contracts.py` | `CandidateSignal` exists in `nova.quant.contracts`, and `PredictionResult` exists in `nova.decision.prediction`. Formal `ModelPrediction` class not in `quant/contracts.py`. | `tests/test_contracts.py` | Contracts pass for CandidateSignal. | `PARTIAL` | `ModelPrediction` class with `prediction_type`, `prediction_distribution`, and uncertainty missing from `nova.quant.contracts`. | Define `ModelPrediction` dataclass matching Contract B exactly. | P2-PRED-01 | Person 2 |
| **P2-TS-01** | Pretrained Time-Series Foundation Model Evaluation (Chronos / TimesFM) | `notebooks/eval_timeseries_foundation_model.py` | None. No foundation models evaluated, tested, or benchmarked. | None. | None. | `MISSING` | No runtime, memory, or directional accuracy benchmarks exist for Chronos/TimesFM. | Create evaluation harness to benchmark zero-shot Chronos/TimesFM against simple MA. | PyTorch / HF deps | Person 2 |
| **P2-SENT-01** | Financial NLP Sentiment Model Evaluation (FinBERT) | `src/nova/models/sentiment.py` | None. `scripts/manthan_trading_bot.py` has a simple regex keyword list, but no transformer/FinBERT model exists. | None. | None. | `MISSING` | FinBERT model loading, tokenization, confidence scoring, and latency checks unbuilt. | Implement `FinBERTSentimentModel` with causal timestamping and fallback. | Transformers / Torch | Person 2 |
| **P2-REG-01** | Transparent Market-Regime Baseline (Trend & Volatility) | `src/nova/regime/detector.py` | Person 3 implemented `MarketRegimeDetector` in `src/nova/decision/regime.py`. Person 2's canonical path `src/nova/regime/` is missing. | `tests/test_regime_detector.py` | 100% tests pass on Person 3's module. | `PARTIAL` | Architectural placement in `decision/` rather than Person 2's `src/nova/regime/`. | Re-export or migrate canonical regime baseline to `src/nova/regime/` to preserve ownership. | P2-FEAT-01, 05 | Person 2 |
| **P2-REG-02** | Advanced Regime Evaluation (HMM / Clustering) | `src/nova/regime/hmm.py` | None. | None. | None. | `MISSING` | No HMM or Gaussian Mixture clustering benchmarked against the transparent ER baseline. | Benchmark 2-state/3-state HMM against the transparent baseline; adopt only if $\Delta > 0$. | P2-REG-01 | Person 2 |
| **P2-ART-01** | Model Artifacts Registry & Documentation | `artifacts/README.md` | `artifacts/` directory does not exist on disk. | None. | None. | `MISSING` | No directory structure, artifact naming convention, or hash registry. | Create `artifacts/` directory with `README.md` cataloging trained models and hashes. | None | Person 2 |
| **P2-ART-02** | Research Notebooks & Exploratory Analysis | `notebooks/` | `notebooks/` directory does not exist on disk. | None. | None. | `MISSING` | No exploratory notebooks documenting feature distributions, target correlations, or baselines. | Initialize `notebooks/README.md` and feature exploration scripts. | None | Person 2 |
| **P2-VAL-01** | Temporal Causality & Anti-Leakage Suite | `tests/test_leakage.py` | `TestLeakageAndTemporalInvariants` strictly verifies prefix preservation and future bar mutation insensitivity. | `tests/test_leakage.py` (2 tests) | 100% passed in 0.04s. | `DONE_VERIFIED` | None for existing EMA features. Needs expansion when new feature families are added. | Maintain tests and extend to new feature families. | P2-FEAT-01 | Person 2 |
| **P2-VAL-02** | Strategy & Model Evaluation Protocol | `src/nova/research/evaluator.py` | `StrategyEvaluator` calculates gross return, turnover, MaxDD, net returns under friction schedule. | `tests/test_evaluation.py`, `tests/test_end_to_end.py` | 100% passed in 0.08s. | `DONE_VERIFIED` | Full backtest evaluator functional for baseline strategy. | Re-use evaluator for custom model signal evaluation. | `MarketBar` | Person 2 |
| **P2-EXT-01** | Market Data Ingestion & Normalization | `kafka/src/market_data/` | Person 1 exclusive production implementation. Ingests `market.ticks` and `market.news`. | `kafka/tests/` (94 tests passed) | Real-time broker connectivity verified. | `OUT_OF_SCOPE` | Person 1 scope. | None. Consume normalized events only. | Kafka broker | Person 1 |
| **P2-EXT-02** | Trading Bot Execution Adapter & Heartbeat | `scripts/manthan_trading_bot.py` | Person 1/Integration scope. Posts to `/api/v1/orders` and `/api/v1/bot/heartbeat`. | Verified live on hackathon server (32 orders filled). | Real execution telemetry verified. | `OUT_OF_SCOPE` | Person 1 / Integration scope. | None. | API Gateway | Person 1 |
| **P2-EXT-03** | Strategy Selection & Friction-Aware Decision Engine | `src/nova/decision/live_engine.py` | Person 3 exclusive production implementation. 14-step live decision pipeline. | `tests/test_live_decision_engine.py` | 100% passed. | `OUT_OF_SCOPE` | Person 3 scope. | None. Supply features/predictions to Person 3. | P2-PRED-02 | Person 3 |
| **P2-EXT-04** | Portfolio Utility, Allocator & Pre-Trade Risk Gates | `src/nova/risk/gates.py`, `src/nova/portfolio/` | Person 3 exclusive production implementation. Enforces cash, margin, exposure limits. | `tests/test_person_3_decision_engine.py` | 100% passed. | `OUT_OF_SCOPE` | Person 3 scope. | None. Models cannot bypass risk gates. | Person 3 | Person 3 |

---

## 3. Existing EMA Crossover Strategy & Kafka Integration Analysis

### 3.1 What Has Been Verified vs. Merely Reported

| Subsystem Component | Claimed Status | Verified Reality | Concrete Evidence |
| :--- | :--- | :--- | :--- |
| **EMA Mathematical Determinism** | Fully deterministic, strictly causal | **VERIFIED** | `calculate_ema_series` tested against hand-calculated values in `tests/test_features.py`; identical input guarantees bit-for-bit identical floats. |
| **Warm-Up Behavior** | Explicit warm-up abstention enforced | **VERIFIED** | `EMACrossoverFeatures.is_warmed_up` is strictly `False` for $i < \text{slow\_window} - 1$; `EMACrossoverStrategy` emits `EvaluationStatus.WARMING_UP` with `signal_score = 0.0`. |
| **Temporal Leakage Protection** | Zero future look-ahead | **VERIFIED** | `tests/test_leakage.py` proves: (1) evaluating bar $t$ returns identical results whether evaluated on prefix $[0 \dots t]$ or full series $[0 \dots t+30]$; (2) modifying bar $t+5$ has zero impact on output at $t$. |
| **Unit Compatibility & Invariants** | Dimensionless decimal returns | **VERIFIED** | Invariants in `CandidateSignal` strictly enforce that gross return, friction, costs, and net edge are dimensionless floats (e.g. `0.010` = +1.00%), rejecting monetary subtractions. |
| **Kafka Feed Live Adaptation** | Real-time broker ingestion | **VERIFIED** | `scripts/manthan_trading_bot.py` subscribed to broker `192.168.65.157:9094`, decoded live ticks across simulation calendar years 2024–2026, and filled 32 live orders. |
| **Strategy vs. ML Distinction** | Complete quant research engine | **DISTINCTION CLARIFIED** | The working EMA crossover strategy is **NOT** a trained ML model, **NOT** an NLP model, and **NOT** a complete quantitative feature library. It serves strictly as Person 2's baseline signal generator. |

### 3.2 Strategic Role of the EMA Crossover Baseline
The EMA crossover baseline is completely operational and stable. It serves as:
1. The **control benchmark** against which all future complex models (GBDT, Chronos, FinBERT) must demonstrate incremental net value (Evidence Before Complexity principle).
2. The **reliable fallback policy**: if an ML model or foundation model fails during live trading, the system gracefully falls back to the deterministic EMA crossover signal.

---

## 4. Remaining Work Backlog

The backlog is organized into strict dependency order. No higher-tier task may be marked complete without its prerequisites.

```
Dependency Graph:
[Milestone 1: Schemas & Contracts]
         │
         ▼
[Milestone 2: Causal Feature Library] ──────► [Milestone 3: Transparent Regime Pipeline]
         │                                                    │
         ▼                                                    ▼
[Milestone 4: Dataset & Walk-Forward Split]           [Milestone 6: Foundation Time-Series Eval]
         │                                                    │
         ▼                                                    ▼
[Milestone 5: Custom OHLCV GBDT Model]                [Milestone 7: FinBERT Sentiment Model]
         │                                                    │
         └────────────────────────┬───────────────────────────┘
                                  ▼
                     [Milestone 8: Artifact Registry]
```

### Milestone 1: Model Prediction Contract & Schema Formalization
* **Objective:** Establish the frozen Interface Contract B (`ModelPrediction`) so Person 2 and Person 3 communicate through validated contracts.
* **Prerequisites:** None.
* **Target Paths:**
  * `schemas/model_prediction.schema.json`
  * `src/nova/quant/contracts.py` (add `ModelPrediction`, `PredictionType`, `PredictionDistribution`)
  * `tests/test_model_prediction_contract.py`
* **Acceptance Criteria:**
  1. `schemas/model_prediction.schema.json` matches Master Spec Section 19 with required fields (`model_id`, `model_version`, `prediction_timestamp`, `input_cutoff_timestamp`, `instrument_id`, `prediction_type`, `horizon`, `prediction_value`, `confidence_or_uncertainty`, `availability_status`).
  2. `ModelPrediction` dataclass enforces timezone-aware timestamps, valid prediction types (`FORWARD_RETURN`, `DIRECTIONAL_PROBABILITY`, `REGIME_PROBABILITY`, `VOLATILITY_ESTIMATE`), and range constraints ($[0, 1]$ for probabilities).
  3. `tests/test_model_prediction_contract.py` validates valid payloads and rejects malformed types, negative uncertainties, and unaligned cutoffs.

### Milestone 2: Causal Quantitative Feature Library
* **Objective:** Implement the multi-family causal feature pipeline required by Specification Section 6.1.
* **Prerequisites:** Milestone 1 (`MarketBar` interface).
* **Target Paths:**
  * `src/nova/features/trend.py` (slopes, momentum, normalized spreads)
  * `src/nova/features/mean_reversion.py` (rolling z-scores, rolling VWAP distance, RSI-14)
  * `src/nova/features/volatility.py` (True Range, ATR-14, rolling realized volatility)
  * `src/nova/features/volume.py` (relative volume, rolling volume ratio, OBV trend)
  * `src/nova/features/pipeline.py` (unified point-in-time feature vector builder)
  * `tests/test_feature_pipeline.py`
  * `tests/test_feature_leakage.py`
* **Acceptance Criteria:**
  1. Pure point-in-time calculation: Feature values at bar $t$ are deterministic functions strictly of bars $\le t$.
  2. Numerical stability: Zero division guards, non-finite float rejections, explicit warm-up tracking across all windows.
  3. Full leakage test: Mutating or appending bars at $t+k$ has zero effect on feature vectors at $t$.
  4. 100% unit test coverage against hand-calculated mathematical test cases.

### Milestone 3: Transparent Market-Regime Baseline Pipeline
* **Objective:** Establish the canonical Person 2 market-regime detector in `src/nova/regime/` and interface with Person 3's decision engine.
* **Prerequisites:** Milestone 1, Milestone 2 (volatility & trend features).
* **Target Paths:**
  * `src/nova/regime/base.py`
  * `src/nova/regime/transparent_detector.py`
  * `src/nova/regime/__init__.py`
  * `tests/test_regime_canonical.py`
* **Acceptance Criteria:**
  1. Categorizes regimes into `TRENDING`, `RANGING`, `HIGH_VOLATILITY`, `NORMAL`, and `UNKNOWN`.
  2. Calculates Kaufman Efficiency Ratio and EWMA volatility using causal, configurable parameters.
  3. Outputs standardized `ModelPrediction` with `prediction_type = "REGIME_PROBABILITY"`.
  4. Passes strict determinism and anti-leakage test suites.

### Milestone 4: Historical Dataset Construction & Walk-Forward Protocol
* **Objective:** Build reproducible chronological training and validation datasets from simulation market history with zero lookahead bias.
* **Prerequisites:** Milestone 2 (feature extraction).
* **Target Paths:**
  * `src/nova/data/dataset_builder.py`
  * `scripts/build_research_dataset.py`
  * `tests/test_dataset_builder.py`
* **Acceptance Criteria:**
  1. Compiles chronological features and forward target labels ($H \in \{1, 5\}$ bars).
  2. Target definition is mathematically sound: $y_t = \frac{C_{t+H} - C_t}{C_t}$.
  3. Features at $t$ strictly pair with target $y_t$, with features derived exclusively from bars $\le t$.
  4. Partitions data into chronological Train (60%), Validation (20%), and Test (20%) splits with a purging embargo gap $\ge H$ bars to prevent boundary leakage.

### Milestone 5: Custom OHLCV Machine Learning Model (LightGBM/XGBoost)
* **Objective:** Implement, train, and evaluate Person 2's custom GBDT model for forward return and directional probability forecasting.
* **Prerequisites:** Milestones 1, 2, 4.
* **Target Paths:**
  * `src/nova/models/base.py`
  * `src/nova/models/ohlcv_gbdt.py`
  * `scripts/train_ohlcv_model.py`
  * `tests/test_ohlcv_model.py`
  * `reports/ohlcv_model_evaluation.json`
* **Acceptance Criteria:**
  1. Model trains deterministically using fixed random seeds and strictly chronological cross-validation.
  2. Implements `predict_point_in_time(features: dict) -> ModelPrediction`.
  3. Demonstrates positive out-of-sample directional accuracy ($\text{DA} > 52.5\%$) or positive Information Coefficient ($\text{IC} > 0.03$) against the validation set.
  4. Generates an evidence report comparing the GBDT against the simple EMA crossover baseline.

### Milestone 6: Time-Series Foundation Model Evaluation (Chronos/TimesFM)
* **Objective:** Perform an empirical assessment of a zero-shot foundation model to determine if it adds measurable alpha over simple baselines.
* **Prerequisites:** Milestones 1, 4.
* **Target Paths:**
  * `scripts/evaluate_foundation_model.py`
  * `reports/foundation_model_evaluation.md`
* **Acceptance Criteria:**
  1. Documents license, dependency weight, and inference latency per prediction (budget $< 250\text{ ms}$).
  2. Compares RMSE and directional accuracy of Chronos/TimesFM against simple EMA / GBDT baselines over identical test splits.
  3. If runtime exceeds the latency budget or adds no incremental value, explicitly documents negative findings and deactivates the model in production (per Spec Section 3).

### Milestone 7: Financial NLP Sentiment Model Evaluation (FinBERT)
* **Objective:** Evaluate ProsusFinBERT / financial domain sentiment models on competition news streams.
* **Prerequisites:** Milestone 1, access to news payloads (`market.news`).
* **Target Paths:**
  * `src/nova/models/sentiment_finbert.py`
  * `scripts/evaluate_finbert_sentiment.py`
  * `reports/finbert_evaluation.md`
* **Acceptance Criteria:**
  1. Emits probabilities for `positive`, `negative`, and `neutral` sentiment with timestamp verification.
  2. Confirms that sentiment scores are mapped to `ModelPrediction` and never directly converted to market orders.
  3. Compares FinBERT predictive correlation with post-news price returns against the current keyword regex dictionary.

### Milestone 8: Model Artifact Management & Documentation
* **Objective:** Finalize artifact storage, checksum registration, and repository documentation.
* **Prerequisites:** Milestones 5, 6, 7.
* **Target Paths:**
  * `artifacts/README.md`
  * `artifacts/models/`
* **Acceptance Criteria:**
  1. Every trained model artifact has a documented SHA-256 hash, hyperparameter configuration, and feature list.
  2. `artifacts/README.md` details artifact provenance, load paths, and fallback mechanisms.

---

## 5. Architectural Findings, Contradictions & Risks

### Finding 1: Schema Drift Between Documentation and File System
* **Observation:** The contract proposal document (`docs/nova_person_2_scope_and_interface_contract.md`) outlines a complete `schemas/model_prediction.schema.json`, but the actual JSON file does not exist in the `schemas/` directory.
* **Consequence:** Automated schema validation tools and CI tests cannot validate prediction payloads against a formal JSON schema file.
* **Corrective Action:** Instantiate `schemas/model_prediction.schema.json` immediately and bind it to a contract test in `tests/test_contracts.py`.

### Finding 2: Person 3 Created Decision-Side Prediction & Regime Workarounds
* **Observation:** In commit `a53cf51`, Person 3 implemented `src/nova/decision/prediction.py` (a toy Ridge regression) and `src/nova/decision/regime.py` (Kaufman ER detector) within the `decision/` subsystem.
* **Consequence:** While this allowed Person 3's decision engine to test policy logic without waiting for Person 2, it creates duplicate logic and potential ownership confusion.
* **Corrective Action:** Person 2 must build the canonical feature and regime modules (`src/nova/features/` and `src/nova/regime/`). Person 3's decision engine will import Person 2's canonical interfaces once frozen, deprecating the temporary mock modules.

### Finding 3: Naive Keyword Matching in Trading Script vs. NLP Model
* **Observation:** `scripts/manthan_trading_bot.py` uses word lists (`PROFIT`, `SURGE`, `LOSS`, `PLUNGE`) to trade news flashes.
* **Consequence:** While effective as a simple heuristic for the initial smoke test, it does not satisfy the Master Specification requirement for an evaluated financial NLP sentiment model (FinBERT) with calibrated probabilities and timestamp leakage guards.
* **Corrective Action:** Implement Milestone 7 to provide formal `ModelPrediction` outputs with sentiment probabilities.

### Finding 4: Inconsistent Naming in Legacy Specifications
* **Observation:** The root specification PDF refers to "Sentinel Trader", while the current production codebase and configuration are strictly standardized on **NOVA** (`src/nova/`).
* **Consequence:** Developers might inadvertently introduce `src/sentinel/` paths, breaking imports and package resolution.
* **Corrective Action:** Strictly enforce the rule established in Section 1: All new code, tests, and documentation must exclusively use `src/nova/` and the `NOVA` project namespace.

---

## 6. Exact Next Implementation Prompt

Copy and paste the prompt below into Antigravity to execute the single highest-priority milestone identified by this audit:

```markdown
# NOVA — PERSON 2 IMPLEMENTATION: MILESTONE 1 & 2 (SCHEMAS & CAUSAL FEATURE PIPELINE)

## Role & Mission
You are Person 2 (Quantitative Research & Feature Engineering) for project NOVA.
Your task is to implement Milestone 1 (Model Prediction Contract & Schema) and Milestone 2 (Causal Quantitative Feature Library) based on the findings of `audit/nova_person_2_scope_completion_audit.md`.

Strict Rules:
- The project is named NOVA. Use `src/nova/` exclusively.
- Do not modify Person 1's files (`kafka/`, `scripts/manthan_trading_bot.py`).
- Do not modify Person 3's decision engine (`src/nova/decision/`, `src/nova/risk/`).
- Do not implement trading policies or place orders.
- Maintain documentation integrity and preserve all existing passing tests.

## Deliverables to Implement

### 1. Model Prediction Schema & Contract
- Create `schemas/model_prediction.schema.json` matching the Master Specification Section 19 and `docs/nova_person_2_scope_and_interface_contract.md`.
- In `src/nova/quant/contracts.py`, add:
  - `PredictionType` enum (`FORWARD_RETURN`, `DIRECTIONAL_PROBABILITY`, `REGIME_PROBABILITY`, `VOLATILITY_ESTIMATE`)
  - `ModelPrediction` frozen dataclass with validation for finite values, probability ranges [0, 1], positive horizon, and timezone-aware timestamps.
- Create `tests/test_model_prediction_contract.py` verifying serialization, contract constraints, and schema validation.

### 2. Multi-Family Causal Feature Library
In `src/nova/features/`, implement causal, deterministic feature extractors:
- `src/nova/features/trend.py`: Moving average slope, momentum vectors, crossover recency.
- `src/nova/features/mean_reversion.py`: Rolling price z-score, distance to rolling VWAP, causal RSI-14.
- `src/nova/features/volatility.py`: Average True Range (ATR-14), rolling realized volatility (Parkinson / close-to-close).
- `src/nova/features/volume.py`: Relative volume (RVOL), rolling volume z-score, price-volume trend.
- `src/nova/features/pipeline.py`: Unified `FeaturePipeline` accepting a chronological sequence of `MarketBar` and returning a timestamped, point-in-time dictionary of features.

### 3. Verification & Leakage Suite
- Create `tests/test_feature_pipeline.py` verifying mathematical correctness against hand-calculated values.
- In `tests/test_leakage.py`, add comprehensive tests confirming that appending or mutating future bars ($t+k$) has ZERO effect on feature values computed at bar $t$.

## Acceptance Criteria
1. All 106 existing unit tests continue to pass (`python -m unittest discover tests`).
2. New feature pipeline and contract tests pass with 100% determinism.
3. Zero future-data leakage across all newly implemented feature extractors.
```


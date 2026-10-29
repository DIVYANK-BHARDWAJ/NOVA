# NOVA — Person 2 Execution Plan
## Quantitative Features, ML Models, Time-Series Foundation Models, NLP Sentiment & Regime Estimation
### Multi-Phase Engineering Roadmap & Stop/Go Gates

- **Project:** NOVA
- **Author:** Person 2
- **Subsystem:** Feature Engineering & Model Prediction
- **Target Repository:** `https://github.com/DIVYANK-BHARDWAJ/NOVA.git`
- **Target Branch:** `main`

---

## 1. Roadmap Architecture & Dependency Chain

```
+----------------------------------------------------------------------------------------------------+
|  Phase 0: Architecture & Repository Rebaseline (COMPLETED IN THIS TURN)                           |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Model Prediction Contract (`schemas/model_prediction.schema.json`) & Foundation Setup   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 2: Causal OHLCV Feature Engine (`src/nova/features/`)                                       |
|           - Trend, Moving Average Crossovers, Mean Reversion, Volatility, Gaps, Volume            |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 3: Research Datasets & Temporal Validation Harness                                          |
|           - Point-in-time construction, chronological splits, horizon leakage prevention           |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 4: Transparent Market-Regime Baseline (`src/nova/regime/baseline.py`)                       |
|           - Deterministic Trend & Volatility Classifier                                            |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 5: Custom OHLCV Machine Learning Model (`src/nova/models/custom_ohlcv/`)                    |
|           - LightGBM / XGBoost reproducible training, out-of-sample prediction                     |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 6: Pretrained Time-Series Foundation Model Assessment (Chronos / TimesFM)                  |
|           - Feasibility, latency, memory, licensing, incremental predictive edge                   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 7: Financial-News Sentiment Model Assessment (FinBERT / FinNLP)                             |
|           - Publication timing, missing news handling, sentiment scores distinct from returns      |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 8: Advanced Market-Regime Research (HMM / Clustering)                                       |
|           - Empirical ablation vs. Phase 4 baseline (promoted only if justified)                   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 9: Model Serving, Configured Artifacts & Graceful Fallback (`artifacts/README.md`)          |
|           - In-memory inference, artifact validation, degradation fallbacks                        |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|  Phase 10: Final Model Validation, Ablations & Person 3 Integration Handoff                        |
|           - Stable prediction stream, ablation reports, collaboration sign-off                     |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Phase-by-Phase Execution Specifications

### Phase 0: Repository & Specification Rebaseline
- **Objective:** Establish the authoritative NOVA three-person collaboration model, preserve legacy work, and design the transition workflow.
- **Deliverables:**
  - `docs/nova_person_2_migration_audit.md` (legacy inventory and disposition).
  - `docs/nova_person_2_scope_and_interface_contract.md` (ownership and boundaries).
  - `docs/nova_person_2_execution_plan.md` (this phased roadmap).
  - `docs/nova_architecture.md` (system blueprint).
- **Exit Gate / Acceptance Criteria:** Legacy work preserved, zero deleted files, clean Git preflight on `main`, stop for human review.

---

### Phase 1: Model Prediction Contract & Foundation Setup
- **Objective:** Formally define the `ModelPrediction` JSON schema and establish the `src/nova/` package structure for Person 2.
- **Prerequisites:** Approval of Phase 0 rebaseline.
- **Affected Modules:**
  - `schemas/model_prediction.schema.json`
  - `src/nova/__init__.py`
  - `src/nova/models/__init__.py`
  - `src/nova/models/contracts.py` (Typed Python dataclasses matching JSON schema)
  - `tests/test_model_prediction_schema.py`
- **Expected Deliverables:** Typed `ModelPrediction` class with strict non-boolean numeric validation, timezone-awareness checks, and zero-order execution invariants.
- **Acceptance Criteria / Stop-Go Gate:** Schema validated via unit tests; passes schema compliance tests; zero execution attributes exist on prediction objects.
- **Cross-Person Dependencies:** Coordinate schema review with Person 1 and Person 3.

---

### Phase 2: Causal OHLCV Feature Engine
- **Objective:** Implement deterministic, point-in-time mathematical feature extractors from normalized OHLCV candle streams in `src/nova/features/`.
- **Prerequisites:** Phase 1 data models established.
- **Feature Families to Implement:**
  1. *Returns & Structure:* Log returns, simple returns, candle body/shadow ratios.
  2. *Moving Averages & Crossovers:* Fast/slow EMA, crossover recency, slope, price-to-MA ratio.
  3. *Mean-Reversion Oscillators:* Rolling z-score of price, RSI, Bollinger Bandwidth.
  4. *Volatility & Risk:* ATR, realized standard deviation, Parkinson volatility, overnight gap metrics.
  5. *Price-Volume Interactions:* Volume moving averages, VWAP deviation, volume momentum.
  6. *Causality Metadata:* Feature availability timestamp, warmup indicator, data-quality flags.
- **Affected Modules:** `src/nova/features/engine.py`, `src/nova/features/indicators.py`, `tests/test_features.py`.
- **Acceptance Criteria / Stop-Go Gate:**
  - *Future Perturbation Test:* Perturbing $T+1$ candle has zero effect on features computed at $T$.
  - *Warmup Enforcement:* Features raise or flag `INSUFFICIENT_HISTORY` during warmup window.
  - Zero divide-by-zero or non-finite outputs.

---

### Phase 3: Research Datasets & Temporal Validation
- **Objective:** Construct leak-free training and validation datasets with strict prediction-horizon separation.
- **Prerequisites:** Phase 2 causal feature engine.
- **Affected Modules:** `src/nova/features/dataset.py`, `tests/test_dataset_leakage.py`.
- **Expected Deliverables:** Chronological dataset builder enforcing non-overlapping holdout windows, training-only transformation fitting (scalers/imputers), and label generation respecting prediction horizon.
- **Acceptance Criteria / Stop-Go Gate:** Automated leakage tests verify no future label contamination in feature matrices; holdout dataset remains strictly quarantined.

---

### Phase 4: Transparent Market-Regime Baseline
- **Objective:** Build an interpretable, deterministic regime classifier based on trend persistence and realized volatility in `src/nova/regime/`.
- **Prerequisites:** Phase 2 feature engine.
- **Affected Modules:** `src/nova/regime/baseline.py`, `tests/test_regime_baseline.py`.
- **Expected Deliverables:** Deterministic classifier emitting `RegimeState` (`TRENDING_BULLISH`, `TRENDING_BEARISH`, `HIGH_VOLATILITY_CHOP`, `LOW_VOLATILITY_COMPRESSION`, `UNKNOWN`).
- **Acceptance Criteria / Stop-Go Gate:** Validated on synthetic historical regimes; correctly flags chop/low-volatility compression with structured reason codes.

---

### Phase 5: Custom OHLCV Machine Learning Model
- **Objective:** Train, validate, and serialize a reproducible gradient-boosted decision tree (LightGBM/XGBoost) baseline in `src/nova/models/custom_ohlcv/`.
- **Prerequisites:** Phases 2, 3, and 4 completed.
- **Affected Modules:** `src/nova/models/custom_ohlcv/`, `scripts/train_ohlcv_model.py`, `tests/test_custom_model.py`.
- **Expected Deliverables:** Model artifact, configuration file with deterministic SHA-256 fingerprint, evaluation report documenting out-of-sample directional accuracy, Sharpe ratio contribution, and calibration.
- **Acceptance Criteria / Stop-Go Gate:** Out-of-sample predictive edge verified against simple momentum baseline; model artifact loads and infers within latency budget (< 5ms per candle).

---

### Phase 6: Pretrained Time-Series Foundation Model Assessment
- **Objective:** Evaluate feasibility of pretrained time-series models (e.g., Chronos, TimesFM).
- **Prerequisites:** Phase 5 evaluation baseline.
- **Affected Modules:** `notebooks/foundation_ts_evaluation.ipynb`, `docs/foundation_ts_feasibility_report.md`.
- **Evaluation Criteria:** License compatibility, GPU/RAM footprint, inference latency, dependency footprint, and whether out-of-the-box forecasts exceed the Phase 5 GBDT baseline.
- **Acceptance Criteria / Stop-Go Gate:** Stop/Go Gate: If foundation model latency > 500ms or adds no statistical alpha over Phase 5, reject for live inference and retain for offline benchmark only.

---

### Phase 7: Financial-News Sentiment Model Assessment
- **Objective:** Evaluate NLP sentiment models (e.g., FinBERT) respecting publication timestamps and handling missing/late news.
- **Prerequisites:** Person 1 news feed interface specification.
- **Affected Modules:** `src/nova/models/sentiment/`, `tests/test_sentiment_model.py`.
- **Expected Deliverables:** Point-in-time sentiment extractor; sentiment scores strictly bounded in `[-1.0, 1.0]`; explicit graceful fallback (`SENTIMENT_UNAVAILABLE`) on empty news windows.
- **Acceptance Criteria / Stop-Go Gate:** Causality tests prove no news published at $T+1$ influences sentiment at $T$; sentiment scores are strictly separated from return expectations.

---

### Phase 8: Advanced Market-Regime Research
- **Objective:** Empirically evaluate Hidden Markov Models (HMM) or unsupervised clustering against the Phase 4 deterministic baseline.
- **Prerequisites:** Phase 4 baseline and Phase 5 custom model.
- **Affected Modules:** `src/nova/regime/advanced_hmm.py`, `docs/regime_ablation_report.md`.
- **Acceptance Criteria / Stop-Go Gate:** Strict Ablation Gate: Advanced regime models are promoted to production ONLY if they achieve a statistically significant reduction in false breakout signals over the Phase 4 baseline without instability.

---

### Phase 9: Model Serving, Configured Artifacts & Graceful Fallback
- **Objective:** Implement production inference wrappers, artifact loaders, configuration verification, and graceful degradation paths.
- **Prerequisites:** Trained model artifacts from Phases 4, 5, (and 7 if adopted).
- **Affected Modules:** `src/nova/models/serving.py`, `artifacts/README.md`, `tests/test_serving.py`.
- **Expected Deliverables:** High-performance inference engine verifying model SHA-256 hashes against configuration fingerprints. If model artifact is missing or corrupted, engine degrades gracefully to heuristic moving-average baseline with status `FALLBACK_HEURISTIC`.
- **Acceptance Criteria / Stop-Go Gate:** Zero unhandled runtime exceptions on missing or corrupted artifact files; latency verified < 10ms end-to-end.

---

### Phase 10: Final Model Validation, Ablations & Person 3 Integration Handoff
- **Objective:** Execute comprehensive model validation, ablation reports, and provide frozen prediction interfaces to Person 3.
- **Prerequisites:** All prior phases completed.
- **Affected Modules:** `docs/model_validation_report.md`, `docs/ablation_study.md`, integration tests.
- **Expected Deliverables:** Complete empirical report covering feature importance, walk-forward out-of-sample metrics, regime-conditioned performance, and latency benchmarks. Stable in-process interface delivered to Person 3.
- **Acceptance Criteria / Stop-Go Gate:** Person 3 confirms end-to-end integration without importing training dependencies or duplicating inference logic.

# NOVA — Person 2 Migration Audit & Legacy Inventory
## Transition from NOVA MAX (4-Engineer Model) to NOVA (3-Person Architecture)

- **Project:** NOVA
- **Author:** Person 2 (Quantitative Features, ML Models, Foundation Time-Series Models, NLP Sentiment & Regime Estimation)
- **Status:** Phase 0 Rebaseline Complete — Audit Grounded in Actual Repository State
- **Canonical Repository:** `https://github.com/DIVYANK-BHARDWAJ/NOVA.git`
- **Active Branch:** `main`

---

## 1. Executive Summary

This document presents a comprehensive audit of all existing artifacts in the repository during the architecture transition to **NOVA's 3-Person collaboration model**.

### Primary Audit Findings:
1. **Preservation Invariant:** In strict accordance with the rebaseline guidelines, **zero legacy files were deleted or renamed**. All previously committed and remediated work remains intact.
2. **Superseded Strategy Ownership:** The legacy NOVA MAX Engineer B plan assigned ownership of 11 discrete trading strategies to Engineer B. Under the authoritative NOVA specification, trading strategy policies, action selection, friction-aware utility, portfolio sizing, and risk gates are exclusively owned by **Person 3**.
3. **Person 2 Exclusive Mandate:** Person 2 owns OHLCV feature engineering (`src/nova/features/`), custom machine learning models (`src/nova/models/`), time-series foundation model evaluation, financial-news sentiment models, market-regime estimation (`src/nova/regime/`), model prediction schemas (`schemas/model_prediction.schema.json`), and model artifact verification.
4. **Reusable Engineering Assets:** The core numerical safeguards, point-in-time timestamp semantics, causal availability gating, immutable configuration structures (`MappingProxyType`), and deterministic SHA-256 fingerprinting developed in Phase 1 and 1.1 provide high-value, thoroughly tested components that map directly into `src/nova/`.
5. **Upstream Ingestion Layer:** The `kafka/` module introduced by Person 1 on `origin/main` (commit `82d2646`) establishes the live market data and news ingestion foundation, providing the raw inputs for Person 2's feature engine.

---

## 2. File-by-File Disposition Matrix

Each file currently present in the repository has been inspected and assigned one of the five mandatory dispositions:
- **REUSE:** Retain with no material redesign.
- **ADAPT:** Reuse part of the implementation with documented changes for the new NOVA architecture.
- **REPLACE AFTER REVIEW:** A replacement is justified, but preserved until formal migration.
- **RETAIN AS HISTORICAL:** Preserve as legacy project history / audit record.
- **DEFER:** Awaiting upstream specifications or external interfaces.

| File Path | Original Purpose | Test Coverage / Status | NOVA Compatibility | Disposition | Justification & Migration Plan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `.gitignore` | Standard git exclusions (pycache, env, logs) | N/A (Standard tool) | Fully compatible | **REUSE** | Retain as repository root exclusion list. Integrate upstream pattern updates cleanly. |
| `master-prompt-engineer.md` | Persona prompt guide from initial commit | Untested | Documentation artifact | **RETAIN AS HISTORICAL** | Preserved from commit `2f7d676` as initial repository history. |
| `audit/engineer_b_phase_00_repository_audit.md` | Phase 0 repository audit for NOVA MAX | Verified commit `e831bce` | Historical context | **RETAIN AS HISTORICAL** | Immutable historical record of repository state prior to implementation. |
| `audit/engineer_b_phase_1_1_remediation.md` | Phase 1.1 defect remediation report | Verified by 36 unit tests | Historical context | **RETAIN AS HISTORICAL** | Immutable audit trail documenting resolution of zero-cost fabrication and immutability defects. |
| `docs/contracts/ENGINEER_B_CANDIDATE_SIGNAL_CONTRACT_PROPOSAL.md` | CandidateSignal contract proposal for NOVA MAX | Documents CandidateSignal | Superseded by NOVA 3-Person model | **RETAIN AS HISTORICAL** | Preserved for lineage. Superseded by `docs/nova_person_2_scope_and_interface_contract.md` and `schemas/model_prediction.schema.json`. |
| `config/research_config.toml` | Centralized research config for 11 strategies | Verified by `test_config.py` | Partially compatible; contains strategy parameters | **ADAPT** | Separate into Person 2 feature/model configuration (`config/features.toml`, `config/models.toml`). Strategy, friction, and portfolio parameters will transition to Person 3. |
| `src/nova/__init__.py` | Package init for `nova` | N/A | Canonical package root | **REUSE** | Canonical package root for NOVA. Preserved as base namespace. |
| `src/nova/quant/__init__.py` | Package init for `nova.quant` | N/A | Legacy subpackage | **RETAIN AS HISTORICAL** | Preserved alongside existing `nova.quant` modules. |
| `src/nova/quant/contracts.py` | Data contracts: MarketBar, CandidateSignal, RegimeState | Covered by 27 unit tests | High reuse potential for MarketBar & RegimeState; CandidateSignal superseded | **ADAPT** | Extract `MarketBar`, `TimestampSemantics`, `DataQualityStatus`, `RegimeState`, and validation utilities (`validate_finite`, `validate_timezone_aware`) into `src/nova/features/` and `src/nova/regime/`. CandidateSignal is superseded by `ModelPrediction`. |
| `src/nova/quant/config.py` | Immutable config parser & SHA-256 fingerprinting | Covered by 9 unit tests | Architecture pattern directly reusable | **ADAPT** | Adapt recursive `MappingProxyType` freezing, non-boolean numeric validation, and canonical SHA-256 fingerprinting into Person 2's `src/nova/models/config.py`. |
| `tests/__init__.py` | Test package root | N/A | Fully compatible | **REUSE** | Standard test runner root for Python `unittest`. |
| `tests/test_contracts.py` | 27 unit tests for contracts and causal integrity | 27 passing tests (0.011s) | Causality & numeric tests reusable | **ADAPT** | Migrate causal availability tests (`is_causally_available`), OHLC validation tests, and regime tests to `tests/test_features.py` and `tests/test_regime.py`. |
| `tests/test_config.py` | 9 unit tests for config immutability & fingerprinting | 9 passing tests (0.003s) | Immutability & fingerprint tests reusable | **ADAPT** | Adapt immutability, boolean-rejection, and fingerprint-sensitivity assertions for the NOVA feature and model configs. |

---

## 3. Detailed Component Mapping: Legacy NOVA MAX to NOVA 3-Person Model

```
+----------------------------------------------------------------------------------------------------+
|                                    LEGACY TO NOVA MAPPING                                          |
+----------------------------------------------------------------------------------------------------+
|  LEGACY COMPONENT (NOVA MAX)           DISPOSITION        NOVA TARGET (PERSON 2)                   |
|  ---------------------------------    -------------      ----------------------------------------  |
|  MarketBar & Causal Invariants  -----> [ ADAPT ]  -----> src/nova/features/base.py                 |
|  TimestampSemantics & Quality   -----> [ ADAPT ]  -----> src/nova/features/base.py                 |
|  RegimeState & Categories       -----> [ ADAPT ]  -----> src/nova/regime/base.py                   |
|  CandidateSignal (Old Alpha)    -----> [REPLACE]  -----> schemas/model_prediction.schema.json      |
|  StrategyEvaluation (Old Alloc) -----> [REPLACE]  -----> Owned by Person 3 (Decision Engine)       |
|  ResearchConfig & Fingerprint   -----> [ ADAPT ]  -----> src/nova/models/config.py                 |
|  11 Strategy Implementations    -----> [REASSIGN] -----> Owned by Person 3 (Strategy Policies)     |
|  Audit Reports (Phase 0, 1.1)   -----> [RETAIN]   -----> audit/ (Historical Evidence)              |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 `MarketBar` & Temporal Integrity
- **Legacy State:** `MarketBar` in `contracts.py` enforces positive prices, non-negative volume, valid OHLC bounding, timezone awareness, and causal availability gating (`is_causally_available()`).
- **NOVA Target:** Person 1 owns `schemas/market_event.schema.json` and the raw data parser in `kafka/` and `src/nova/ingestion/`. Person 2 consumes these events. Person 2 will adapt the causal validation principles of `MarketBar` into `src/nova/features/base.py` to ensure that feature calculation pipelines reject bars with unknown timestamp semantics or missing arrival timestamps.

### 3.2 `CandidateSignal` vs. `ModelPrediction`
- **Legacy State:** `CandidateSignal` was a hybrid between a directional signal, return forecast, and edge estimator.
- **NOVA Target:** Fully replaced by `ModelPrediction`. Person 2 produces **standardized statistical predictions** (return expectations, directional probabilities, regime state, volatility, and sentiment scores). Person 2 does **NOT** emit trading decisions, execution intent, or position sizing. Those responsibilities transition exclusively to Person 3's `trade_decision.schema.json`.

### 3.3 `RegimeState`
- **Legacy State:** `RegimeState` in `contracts.py` provided regime labels (`TRENDING_BULLISH`, `TRENDING_BEARISH`, `HIGH_VOLATILITY_CHOP`, `LOW_VOLATILITY_COMPRESSION`, `UNKNOWN`), heuristic confidence scores, supporting indicators, and structured reason codes.
- **NOVA Target:** Highly reusable. Person 2 will adapt `RegimeState` into `src/nova/regime/base.py` as the standard container for both transparent trend/volatility baselines and advanced HMM/clustering models.

### 3.4 Research Configuration & Deterministic Fingerprints
- **Legacy State:** `config.py` enforces recursive immutability (`MappingProxyType`), non-boolean numeric validation, and canonical SHA-256 fingerprinting representing all behavioral parameters.
- **NOVA Target:** Adapted directly into `src/nova/models/config.py` and `src/nova/features/config.py`. Model training and inference will strictly record this deterministic fingerprint to guarantee 100% experiment reproducibility.

---

## 4. Preservation & Non-Destructive Guarantee

- No files have been deleted, moved, or overwritten in this phase.
- All 36 legacy unit tests continue to pass with zero failures (`python -m unittest discover -s tests -p "test*.py"`).
- The transition from legacy modules to new feature and model modules will occur progressively in dedicated subsequent phases with explicit human approval at each gate.

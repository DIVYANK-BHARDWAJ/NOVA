# NOVA Model Artifacts & Research Registry
## Quantitative Models, Feature Pipelines & Evaluation Records

This directory serves as the canonical registry for Person 2 machine learning model artifacts, foundation model evaluation records, feature pipelines, and serialized weights within project **NOVA**.

---

### 1. Directory Structure

```
artifacts/
├── README.md                          # This document (registry specification & checksums)
├── models/                            # Serialized ML model artifacts (GBDT, Ridge, Neural)
│   └── .gitkeep
├── scalers/                           # Point-in-time feature transformation scalers
│   └── .gitkeep
└── evaluations/                       # Out-of-sample benchmark reports & telemetry logs
    └── .gitkeep
```

---

### 2. Artifact Governance & Provenance Rules

Under Person 2's quantitative research charter:
1. **Model Identifiers:** Every model artifact must have a deterministic `model_id` (e.g. `GBDT_OHLCV_V1`, `TIMESFM_ZERO_SHOT`, `FINBERT_NEWS_V1`).
2. **Deterministic Versioning:** All weights, trees, and checkpoints must record an SHA-256 checksum and a training dataset identifier.
3. **No Secret Ingestion:** Model artifacts and metadata must never record credentials, API keys, or private environment variables.
4. **Fallback Requirement:** If an artifact fails checksum verification or encounters inference errors, the runtime must fall back to the deterministic heuristic baseline (`FALLBACK_HEURISTIC`).

---

### 3. Registry Table

| Model ID | Version | Type | Checksum (SHA-256) | Training Cutoff | Primary Horizon | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `XGBOOST_NATIVE` | `1.0.0` | Native C++ XGBoost | `d930a05ffe39875f7e6c162406d95e403a62645a52937e0503dc109fc5d67d33` | 2016-12-02 to 2022-04-14 (Test 2024-02 to 2025-11) | 1 Bar | Trained & Verified (Native C++) |
| `NOVA_XGBOOST_DIRECTIONAL_CLASSIFIER_V1` | `1.0.0` | Native C++ XGBoost Classifier | `1ef343ad65aa1fa4c2e29fa98b00762ad1ccef7bb0e46be33c7620ba2115f575` | 2016-12-02 to 2022-04-14 (Test 2024-02 to 2025-11) | 1 Bar | Trained & Verified (Binary Logistic) |
| `NOVA_XGBOOST_DIRECTIONAL_CLASSIFIER_V2` | `2.0.0` | Native C++ XGBoost Classifier | `311652bf1a1fd0029b4e64fbf0e6fdf24fae0413009943486a4c2cb0f9d9859f` | 2016-12-02 to 2022-08-02 (Test 2024-06 to 2026-05) | 1 Bar | Trained & Verified (Reconciled V2) |
| `NOVA_XGBOOST_DIRECTIONAL_CLASSIFIER_V3` | `3.0.0` | Native C++ Compact Parsimonious | `d51a91e2ac950521106d45600745b83664de15f62e14c68058e5f0dd87070897` | 2016-12-02 to 2022-08-02 (Test 2024-06 to 2026-05) | 1 Bar | Trained & Verified (Compact 6-Feature Parsimonious) |
| `GBDT_OHLCV_BASELINE` | `1.0.0` | Custom Pure-Python GBDT | `a053acc8189a20288a2b8e0d30fee234dfd193a4c9015ab4733e01eb607ddf5e` | 2016-12-02 to 2022-04-14 (Test 2024-02 to 2025-11) | 1 Bar | Trained & Verified (Custom Baseline) |
| `TIMESFM_ZERO_SHOT` | `1.0.0` | Foundation TS | *Pending Milestone 6* | Zero-Shot Pretrained | 1-5 Bars | Planned |
| `FINBERT_NEWS_NLP` | `1.0.0` | HuggingFace NLP | *Pending Milestone 7* | Pretrained Transformer | Immediate | Planned |


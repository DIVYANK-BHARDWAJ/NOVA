# NOVA PERSON 2 — MILESTONE 6B AUDIT REPORT
## Paired Out-of-Sample Statistical Acceptance Audit

**Project:** NOVA  
**Role:** Senior Quantitative Machine Learning Engineer (Person 2)  
**Milestone:** Milestone 6B — Paired Out-of-Sample Statistical Acceptance Audit  
**Audit Timestamp:** 2026-10-30T01:56:00+05:30  
**Repository Branch:** `Divyank`  
**Execution Environment:** Windows AMD64, Python 3.11.9, Native `xgboost==3.2.0`, `numpy==2.4.6`, `scipy==1.17.1`  
**Audit Verdict:** **`PASS_ENGINEERING_ONLY`**  
**Automated Verification:** **309 / 309 automated tests PASSING** (215 in `tests/`, 94 in `kafka/tests/`)  

---

## 1. Executive Summary & Audit Verdict

| Audit Dimension | Status / Classification | Evidence & Evaluation |
| :--- | :---: | :--- |
| **Final Audit Status** | **`PASS_ENGINEERING_ONLY`** | Model engineering, C++ native XGBoost execution, leak-free chronological embargo, and unit tests pass 100%. However, paired statistical testing proves that incremental predictive value over the unconditional drift baseline is **not established**. |
| **Out-of-Sample Test Observations** | **IDENTICAL & UNTOUCHED** | Exactly 4,493 forward test observations across 10 instruments evaluated identically across all predictors. |
| **Directional Hit Rate vs. Drift** | **NO INCREMENTAL EDGE** | Native XGBoost hit rate is **52.15%** (2,343/4,493), compared to **52.41%** (2,355/4,493) for the training-mean drift forecast ($\Delta = -0.27\%$, McNemar paired test $p = 0.6674$). |
| **Paired Loss Reductions** | **NEGLIGIBLE EDGE** | Paired MAE reduction vs. Zero Return is $+0.18$ bps; aggregate RMSE reduction vs. Zero Return is $+0.08\%$ (+0.19 bps), with 95% bootstrap confidence intervals spanning zero across all block lengths. |
| **Trading-Date Block Sensitivity** | **CONVERGENT UNCERTAINTY** | Joint trading-date block bootstrap across 5, 10, 20, and 40 trading days demonstrates that the confidence interval for directional superiority strictly contains zero. |
| **Delivery Count Reconciliation** | **RECONCILED (37,188 = 2x RUNS)** | Exactly 18,594 deliveries logged per capture session. Cumulative 37,188 deliveries deduplicate perfectly to the identical 18,494 unique observations with zero payload conflicts. |
| **Purged Boundary Embargo** | **LEAK-FREE & AUDITED** | Exactly 20 observations purged across train/val and val/test boundaries ($H=1$). No test information was used in model fitting or threshold tuning. |

---

## 2. Independent Reproduction of Out-of-Sample Results

All models evaluated against the untouched forward out-of-sample test set (**4,493 daily observations spanning February 2024 to November 2025**):

| Predictor / Baseline | Forecasting Rule / Target | Out-of-Sample MAE | Out-of-Sample RMSE | Directional Hit Rate | Out-of-Sample $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Native XGBoost (`1.0.0`)** | $\hat{y} = f_{XGB}(x_t)$ | **0.015408** | **0.024355** | **52.15%** | **-0.000624** |
| **Custom Pure-Python GBDT (`1.0.0`)** | $\hat{y} = f_{GBDT}(x_t)$ | 0.015423 | 0.024398 | 51.75% | -0.004203 |
| **Zero-Return Baseline** | $\hat{y} = 0.0$ | 0.015426 | 0.024374 | 0.22% (tie-fail) | -0.002169 |
| **Historical Mean (Strict Out-of-Sample)** | $\hat{y} = \bar{y}_{train} = +0.000083$ | 0.015423 | 0.024370 | **52.41%** | -0.001863 |

---

## 3. Observation-Level Paired Loss Differences

To eliminate sample-composition variance, paired observation-level losses were computed for every test observation $i \in \{1..4493\}$:
- **Sign Convention:** $\Delta L = L_{baseline} - L_{model}$. Positive values indicate that Native XGBoost achieved a lower error (reduction in loss) than the baseline.
- **Sample Count:** $N = 4,493$ paired test observations.

### Paired Differences vs. Zero-Return Baseline ($\hat{y} = 0.0$)
- **Paired Mean Absolute Error (MAE) Reduction:** $+0.000018$ ($+0.18$ bps error reduction)
- **Paired Mean Squared Error (MSE) Reduction:** $+0.00000092$
- **Aggregate RMSE Reduction:** $+0.000019$ ($+0.08\%$ relative reduction)

### Paired Differences vs. Training-Mean Baseline ($\hat{y} = \bar{y}_{train}$)
- **Paired Mean Absolute Error (MAE) Reduction:** $+0.000014$ ($+0.14$ bps error reduction)
- **Paired Mean Squared Error (MSE) Reduction:** $+0.00000073$
- **Aggregate RMSE Reduction:** $+0.000015$ ($+0.06\%$ relative reduction)

---

## 4. Corrected Directional Benchmark & Contingency Analysis

### Why Does the Training-Mean Baseline Report 52.41% Hit Rate?
1. In the training split (`2016-12-02` to `2022-04-14`), the average 1-day log return was positive: $\bar{y}_{train} = +0.000083$ ($+0.8$ bps).
2. Consequently, the constant historical-mean forecast predicts $\hat{y} > 0$ on **100% of days**, betting on persistent positive market drift.
3. During the test period (`2024-02-02` to `2025-11-19`), the realized daily return across all 10 stocks was positive on **52.41% of trading days** (2,355 positive days vs. 2,138 negative or zero days).
4. Therefore, simply forecasting constant positive drift yields a **52.41% directional hit rate** with zero modeling complexity.

### Paired Contingency Table (Native XGBoost vs. Training-Mean Drift)

| Outcome Category | Observation Count | Percentage of Test Set |
| :--- | :---: | :---: |
| **Both Predictors Correct** | 1,959 | 43.6% |
| **Both Predictors Incorrect** | 1,754 | 39.0% |
| **XGBoost Correct, Mean Drift Incorrect** (XGB exclusive wins) | **384** | **8.5%** |
| **Mean Drift Correct, XGBoost Incorrect** (Drift exclusive wins) | **396** | **8.8%** |
| **Total Test Observations** | **4,493** | **100.0%** |

### Statistical Test for Incremental Directional Value
- **Hit Rate Delta ($\Delta Hit = Hit_{XGB} - Hit_{Mean}$):** **`-0.27%`** ($52.15\% - 52.41\%$)
- **McNemar's Paired Chi-Square Test:**
  $$\chi^2 = \frac{(n_{10} - n_{01})^2}{n_{10} + n_{01}} = \frac{(384 - 396)^2}{384 + 396} = \frac{(-12)^2}{780} = 0.1846$$
- **$p$-value:** **`0.6674`** (Two-sided $\chi^2$ distribution with 1 degree of freedom).
- **Inference:** Because $p = 0.6674 \gg 0.05$, there is **no statistically significant difference** in directional forecasting accuracy between Native XGBoost and the simple unconditional positive drift baseline. Native XGBoost does NOT establish an incremental directional edge over the drift benchmark.

---

## 5. Joint Trading-Date Block Bootstrap & Sensitivity Analysis

To rigorously respect serial correlation and multi-asset cross-sectional dependence, the 4,493 test observations were grouped strictly by **calendar trading date** ($T = 452$ unique dates, averaging $\approx 9.9$ instruments per date).

Blocks of contiguous trading dates were jointly resampled across all 10 instruments simultaneously over $B = 2,000$ bootstrap iterations.

### Block-Length Sensitivity Analysis ($B \in \{5, 10, 20, 40\}$ Trading Days)

| Block Length | Directional Delta vs. Drift (95% CI) | RMSE Reduction vs. Zero (95% CI) | $p$-value (RMSE $> 0$) |
| :---: | :---: | :---: | :---: |
| **5 trading days** ($\approx 1$ week) | -0.26% [-1.78%, +1.22%] | +0.000020 [-0.000012, +0.000064] | $p = 0.1600$ |
| **10 trading days** ($\approx 2$ weeks) | -0.24% [-1.86%, +1.37%] | +0.000020 [-0.000013, +0.000065] | $p = 0.1690$ |
| **20 trading days** ($\approx 1$ month) | -0.28% [-1.75%, +1.17%] | +0.000022 [-0.000011, +0.000065] | $p = 0.1365$ |
| **40 trading days** ($\approx 2$ months) | -0.22% [-1.42%, +0.95%] | +0.000022 [-0.000013, +0.000075] | $p = 0.1920$ |

### Interpretation
- Across all four block lengths (from 1 week to 2 months), the 95% confidence intervals for both loss reduction and directional delta **span zero**.
- The probability that Native XGBoost achieves lower RMSE than Zero Return is between $81\%$ and $87\%$ ($p$-value for superiority is $0.1365$ to $0.1920$), which is insufficient to reject the null hypothesis of equal forecasting performance.

---

## 6. Provenance & Delivery Count Reconciliation

- **Canonical Unique Observations:** Exactly **18,494** unique observations across all 10 instruments in `data/replay/canonical_manthan_v1/observations.jsonl`.
- **Cumulative Deliveries:** Exactly **37,188** delivery entries logged in `deliveries.jsonl`.
- **Reconciliation Audit:**
  - Session 1 (Initial Recording): 18,594 deliveries logged (18,494 unique observations + 100 repeated pollings).
  - Session 2 (Audit Re-run): Exactly 18,594 additional deliveries received from re-running the recording script during Milestone 5 audit.
  - Cumulative Total: $18,594 + 18,594 = 37,188$ deliveries.
  - Deduplication Parity: All 37,188 deliveries deduplicate perfectly into the identical 18,494 unique observations with **ZERO (0) conflicting price, volume, or timestamp values**.

---

## 7. Chronology, Anti-Leakage & Split-Boundary Embargo

- **Timeline Partitions:**
  - Train: `2016-12-02` to `2022-04-14` (9,201 rows)
  - Validation: `2022-04-19` to `2024-01-31` (4,490 rows)
  - Test: `2024-02-02` to `2025-11-19` (4,493 rows)
  - Date Separation: `train_end < val_start` and `val_end < test_start` are strictly satisfied.
- **Split-Boundary Embargo:** Exactly **20 observations purged** (10 at train/val boundary, 10 at val/test boundary). No return target spans across split boundaries.
- **Zero Test Information Used:** Hyperparameters ($depth=3, \eta=0.05, n=35$) and early stopping were determined exclusively on training and validation splits. The test set was touched strictly for final evaluation.

---

## 8. Final Research Verdict & Role Handoff

In strict adherence to the acceptance rules established in Milestone 6B:
- **Verdict:** **`PASS_ENGINEERING_ONLY`**
- **Justification:**
  1. The software engineering implementation, dual-backend XGBoost/GBDT execution, data pipeline, leak-free embargo, serialization fidelity, and test suite (309/309 tests passing) are production-grade.
  2. However, paired statistical testing proves that the model's 52.15% directional accuracy does not demonstrate statistically significant incremental predictive skill over the simple unconditional positive drift baseline (52.41%, McNemar $p = 0.6674$).
  3. The model is therefore certified as a verified engineering pipeline and research baseline, but **cannot be certified as a profitable standalone alpha strategy** without additional predictive signals (e.g. TimesFM foundation time-series features or FinBERT financial news sentiment in subsequent milestones).

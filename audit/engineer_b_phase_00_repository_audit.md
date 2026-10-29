# NOVA MAX — Engineer B (Quantitative Research & Strategy Engine)
# Phase 00 Audit Report: Repository Audit, Contract Verification, and Implementation Plan

- **Date:** 2026-10-29
- **Author:** Engineer B (Quantitative Research & Strategy Engine)
- **Role:** Quantitative Research, Signal Generation, Point-in-Time Backtesting, Regime Intelligence
- **Status:** Phase 0 Completed — Greenfield Repository Established — Awaiting Contract Confirmation

---

## 1. Executive Summary

This Phase 0 Audit was performed under strict non-destructive constraints in accordance with the NOVA MAX engineering guidelines for the MANTHAN 2026 challenge.

### Key Audit Findings:
1. **Greenfield Repository:** The repository root (`C:\Users\DIVYANK BHARDWAJ\Desktop\Projects\jnncehackathon\jnncehackathon`) contains no pre-existing production code, configuration files, test suites, or package setup. Git is initialized on branch `Divyank` with a single initial commit (`2f7d676`) introducing only `master-prompt-engineer.md`.
2. **Missing External Specifications:** No formal MANTHAN 2026 problem statement, official exchange API documentation, Kafka event schema, or inter-engineer integration contract documents currently exist inside the repository.
3. **Scope & Contract Baseline Established:** Engineer B's functional scope is explicitly defined based on the briefing: quantitative research, point-in-time signal generation, research features, leakage-free backtesting, and empirical strategy promotion. Under no circumstances will Engineer B's strategies submit orders directly or bypass Engineer C's risk gateway.
4. **Implementation Status:** 0% implemented (Greenfield). No tests exist (`Ran 0 tests in 0.000s` using standard `unittest`). No dependencies are installed. Python 3.11.9 is available in the environment.
5. **Phase 1 Recommendation:** Define domain data models, shared candidate-signal data contracts, abstract strategy interfaces, and chronological backtesting foundations without hardcoding unverified exchange assumptions.

---

## 2. Repository Inspection & Environmental State

### 2.1 Git & Directory Inventory

| Property | Value / Finding | Verification Method |
| :--- | :--- | :--- |
| **Absolute Repository Root** | `C:\Users\DIVYANK BHARDWAJ\Desktop\Projects\jnncehackathon\jnncehackathon` | `Get-Location` |
| **Git Initialized** | Yes | `git status` |
| **Current Branch** | `Divyank` | `git branch --show-current` |
| **HEAD Commit** | `2f7d676 added master prompt engineer file` | `git log -n 1 --oneline` |
| **Working Tree Status** | Clean (prior to generating this audit report) | `git status` |
| **Remote Origin** | `https://github.com/DIVYANK-BHARDWAJ/jnncehackathon.git` | `git remote -v` |
| **Existing Tracked Files** | `master-prompt-engineer.md` | `git ls-tree -r HEAD` |
| **Untracked Files** | None (prior to this audit report) | `git status` |
| **Directory Structure** | Greenfield. No `src/`, `tests/`, `config/`, `docs/`, or `data/` directories exist. | `Get-ChildItem -Force` |

### 2.2 Runtime & Tooling Inspection

| Component | Status | Details |
| :--- | :--- | :--- |
| **Python Runtime** | `Python 3.11.9` | Verified via `python --version` |
| **Dependency Manager** | Unspecified / None configured | No `pyproject.toml`, `requirements.txt`, `Pipfile`, or `poetry.lock` exists. |
| **Test Runner** | Standard Library `unittest` available; `pytest` not found | `pytest --version` exited code 1 (CommandNotFound). `python -m unittest` functional. |
| **Secrets / Credentials** | None found | No `.env` or credential files present in the repo. |
| **Colleague Codebases** | None found | No work from Engineer A (Platform), Engineer C (Execution/Risk), or Engineer D (QA) is present. |

---

## 3. Documentation and Contract Audit

### 3.1 Document Availability Matrix

| Document Required by Phase 0 Scope | Repository Path / State | Status | Impact on Engineer B |
| :--- | :--- | :--- | :--- |
| **Official MANTHAN 2026 Problem Statement** | Not found in repo | **UNAVAILABLE** | Market rules, competition calendar, universe parameters, and evaluation metrics cannot be verified from source. |
| **Official Exchange / Platform API Docs & Starter Code** | Not found in repo | **UNAVAILABLE** | REST/WebSocket endpoints, request throttling, order book depth, and historical data limits are unverified. |
| **AGENTS.md / Team Operating Instructions** | Not found in repo | **UNAVAILABLE** | Multi-agent coordination protocols must be established from the user briefing. |
| **NOVA MAX Master System Architecture** | Not found in repo | **UNAVAILABLE** | Subsystem boundaries, messaging topologies (e.g., Kafka vs direct Python calls), and serialization schemas are unverified. |
| **NOVA MAX Strategy & Discipline Blueprint** | Derived from briefing | **PARTIAL (BRIEFING)** | 11 locked strategies defined; parameters and execution thresholds require specification. |
| **Four Engineer Assignment Briefs (A, B, C, D)** | Derived from briefing | **PARTIAL (BRIEFING)** | Clean division of responsibilities: A (Data), B (Alpha), C (Risk/Execution), D (QA). |
| **Master Integration Contract (Candidate Signal Contract)** | Not found in repo | **UNAVAILABLE** | Must be formalized in Phase 1 before inter-module coupling occurs. |

### 3.2 Categorized Requirement Distinctions

#### A. Verified Requirements (from Engineer B Briefing)
1. **Scope Exclusivity:** Engineer B is responsible for quantitative research, feature engineering, signal generation, market regime classification, backtesting, and strategy validation evidence.
2. **Order Execution Prohibition:** Engineer B strategies **MUST NOT** directly place orders or communicate with the exchange execution API. All outputs must terminate as candidate signals or target weights routed to Engineer C's risk gateway.
3. **Strict Temporal Integrity:** All calculations at timestamp $T$ must strictly utilize information available at or before $T$. Zero look-ahead bias is tolerated.
4. **Honest Benchmarks:** Every strategy must be evaluated against a no-trade benchmark and conservative transaction cost model (brokerage, exchange fees, slippage).
5. **Strategy Portfolio:** Baseline suite contains 11 strategies, prioritized starting with Adaptive Momentum and Cross-Sectional Relative Strength backed by Market Regime Intelligence.
6. **Abstention as Valid State:** A strategy must output an explicit abstention (`HOLD` / `NEUTRAL` with reason) when regime conditions or data quality are inadequate.

#### B. Existing Implementation
- **None:** The repository is currently greenfield.

#### C. Missing Implementation
- Standardized data schemas: `Bar`, `Quote`, `MarketEvent`, `SignalCandidate`, `RegimeState`.
- Point-in-time calculation engine avoiding look-ahead bias.
- Discrete strategy engines (Momentum, Relative Strength, Breakout, Mean Reversion, etc.).
- Event-driven historical simulation / backtesting engine with realistic fill and cost modeling.
- Strategy promotion harness with out-of-sample statistical tests (Sharpe, Calmar, Deflated Sharpe Ratio).

#### D. Unresolved Questions & Critical Assumptions (Requiring Upstream Confirmation)
1. **Market Data Event Timestamps:** Does a candle with timestamp `2026-03-01 10:00:00` represent the open time (meaning available only at `10:01:00` or `10:05:00`), or the close time?
2. **Dynamic Ticker Universe:** How and when does the platform announce the tradable universe? Are new IPOs or delisted tickers introduced dynamically during competition sessions?
3. **Execution Fee Schedule:** What are the exact exchange transaction charges, securities transaction taxes (STT), clearing fees, and slippage assumptions enforced by the contest engine?
4. **Short Selling & Margin:** Are intraday short positions permitted? Are overnight shorts allowed? What are the margin and leverage restrictions?
5. **News Feed Granularity:** If news feeds are provided, what are the exact timestamp formats, sentiment annotations, latency characteristics, and publishing delay?
6. **Inter-Engine Communication:** Will Engineer B communicate candidate signals to Engineer C via in-process typed Python dataclasses, asyncio queues, shared memory, or a distributed message broker (e.g., Kafka / Redis)?

---

## 4. Engineer B Ownership & Boundary Interface

```
+-------------------------------------------------------------------------------------------------+
|                                        NOVA MAX ARCHITECTURE                                    |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [ ENGINEER A: Ingestion & Platform ]                                                          |
|       - Platform Connectivity (WebSocket / REST)                                                |
|       - Raw Market Data Ingestion & Normalization                                               |
|       - Clock Synchronization & Replay Stream                                                   |
|                                     |                                                           |
|                                     v (Historical & Real-Time Market Events)                    |
|   +-----------------------------------------------------------------------------------------+   |
|   |  ENGINEER B: Quantitative Research & Strategy Engine (OUR DOMAIN)                       |   |
|   |   - Point-in-Time Feature Store & Historical Bar Cache                                  |   |
|   |   - Market Regime & Breadth Engine                                                      |   |
|   |   - Core Alphas (Adaptive Momentum, Cross-Sectional RS, Breakouts, Mean Reversion)      |   |
|   |   - Historical Chronological Backtester & Leakage Validator                             |   |
|   |   - Candidate Signal Output Schema (No direct orders!)                                  |   |
|   +-----------------------------------------------------------------------------------------+   |
|                                     |                                                           |
|                                     v (CandidateSignal / TargetAllocation Contracts)            |
|   [ ENGINEER C: Portfolio, Risk Gate & Execution ]                                              |
|       - Risk Authorization (VaR, Leverage Limits, Drawdown Guard)                               |
|       - Portfolio Sizing & Optimization                                                         |
|       - Order Slicing, Route Submission & Fill Reconciliation                                   |
|                                                                                                 |
|   [ ENGINEER D: QA, Observability & Verification ]                                              |
|       - Test Harnesses, Continuous Integration & Stress Testing                                 |
|       - Invariant Monitoring (Data freshness, execution latency, anomaly detection)             |
|       - Production Release Sign-Off                                                             |
+-------------------------------------------------------------------------------------------------+
```

### Responsibility Matrix

| Feature / Action | Engineer A | Engineer B (Self) | Engineer C | Engineer D |
| :--- | :---: | :---: | :---: | :---: |
| Platform WebSocket/REST Connection | **OWNER** | No Access | Consumer | Monitor |
| Ingestion of Raw Market Ticks/Bars | **OWNER** | Consumer | Consumer | Monitor |
| Point-in-Time Feature Engineering | No | **OWNER** | No | Auditor |
| Alpha Signals & Strategy Models | No | **OWNER** | No | Test Partner |
| Backtesting Engine & Leakage Auditing | No | **OWNER** | Input on Fills | Test Partner |
| Portfolio Optimization & Risk Gate | No | Proposes Edge | **OWNER** | Monitor |
| Order Generation & Submission | No | **FORBIDDEN** | **OWNER** | Monitor |
| Integration Testing & QA Gate | Fixtures | Unit/PIT Tests | Stress Tests | **OWNER** |

---

## 5. Current Implementation Inventory

Assessment Codes:
- `VERIFIED_EXISTING`: Fully implemented, inspected, and verified in codebase.
- `PARTIALLY_IMPLEMENTED`: Present in codebase but incomplete or lacking tests.
- `MISSING`: Required for functionality but completely absent from repository.
- `BLOCKED_BY_CONTRACT`: Cannot be safely designed or implemented without upstream specification.
- `NOT_VERIFIABLE`: Stated in secondary notes but cannot be confirmed from existing repository files.

| Component | Status | Supporting Evidence & Details |
| :--- | :--- | :--- |
| **1. Strategy Interface & Candidate Signal Schema** | `MISSING` | No base classes, typed models, or signal dataclasses exist in the workspace. |
| **2. Feature Computation & Timestamp Handling** | `MISSING` | No rolling window, exponential filter, or point-in-time feature library exists. |
| **3. Dynamic Universe Data Representation** | `MISSING` | No dynamic symbol tracking or universe mapping modules exist. |
| **4. Strategy 1: Adaptive Momentum & Trend Acceleration** | `MISSING` | No implementation found. |
| **5. Strategy 2: Cross-Sectional Relative Strength** | `MISSING` | No ranking or cross-sectional quantile implementation found. |
| **6. Strategy 3: Market Breadth & Regime Intelligence** | `MISSING` | No advance-decline, ADX, or regime state classifiers exist. |
| **7. Strategies 4-8: Breakout, Vol Shock, News, Mean Rev, Pairs** | `MISSING` | No implementation found. |
| **8. Strategy 9: ML Expected-Return Ranking** | `MISSING` | No model pipelines, training schedules, or feature matrix transformers exist. |
| **9. Strategy 10: Portfolio-Aware Capital Allocation Interface** | `MISSING` | No expected-edge scoring or signal confidence interfaces exist. |
| **10. Strategy 11: Adaptive Strategy Selection** | `MISSING` | No meta-allocation or strategy weighting engine exists. |
| **11. Cost & Expected Edge Estimation** | `BLOCKED_BY_CONTRACT` | Official exchange fee schedule and spread assumptions are not yet documented in repo. |
| **12. Historical Replay & Chronological Backtesting** | `MISSING` | No simulation loop, event scheduler, or trade ledger exists. |
| **13. Data Splits & Look-Ahead Protections** | `MISSING` | No temporal cross-validation or causality verification harnesses exist. |
| **14. Experiment Tracking & Benchmark Reports** | `MISSING` | No logging framework or metric calculation reports exist. |
| **15. Unit, Integration, & Invariant Tests** | `MISSING` | `python -m unittest discover` discovers 0 tests. No test directory exists. |

---

## 6. Gap Analysis & Dependency Chain

The quantitative strategy engine requires a strict bottom-up dependency ordering to ensure mathematical validity and avoid architectural rework:

```
+-----------------------------------------------------------------------------------+
| Step 1: Phase 0 Audit & Contract Baseline (COMPLETED IN THIS PHASE)               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 2: Shared Data Contracts & Signal Interface (`nova/core/contracts/`)          |
|         - CandidateSignal, MarketEvent, Bar, Quote, PositionState                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 3: Point-in-Time Data & Feature Engine (`nova/research/features/`)           |
|         - Causally strict rolling windows, no future leakage                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 4: No-Trade Benchmark & Chronological Backtester (`nova/research/backtest/`) |
|         - Conservative friction, realistic fill simulation, walk-forward harness  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 5: Core Alpha 1 — Adaptive Momentum (`nova/strategies/momentum/`)            |
| Step 6: Core Alpha 2 — Cross-Sectional RS (`nova/strategies/relative_strength/`)  |
| Step 7: Regime Intelligence & Breadth (`nova/strategies/regime/`)                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 8: Expected Edge & Signal Ranking Interface (`nova/research/ranking/`)        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 9-11: Secondary Alphas & Feasibility (Breakout, Mean Rev, Pairs, ML)        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 12: Independent Validation, Ablation, & Leakage Audits                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 13: Final Integration Handoff to Engineers A, C, and D                       |
+-----------------------------------------------------------------------------------+
```

---

## 7. Risk Register

| Risk ID | Category | Description | Severity | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | **Look-Ahead Leakage** | Feature calculations or rolling aggregations inadvertently include $T_{future}$ observations. | **CRITICAL** | Medium | Implement automated temporal-integrity tests (shuffled time test, future perturbation test). Strict indexing enforcement. |
| **RSK-02** | **Timestamp Ambiguity** | Bar timestamp ambiguity (open vs close vs arrival time) causing execution at unobserved prices. | **HIGH** | High | Standardize `available_at` timestamp semantic distinct from `candle_start` and `candle_end`. All signals trigger on $T_{available}$. |
| **RSK-03** | **Cost Underestimation** | Strategy alpha is wiped out in live trading due to unrealistically low slippage, fees, or STT assumptions. | **HIGH** | Medium | Build conservative tiered fee and slippage modules into the backtester. Reject any strategy with Sharpe < 1.0 after 2x worst-case friction. |
| **RSK-04** | **Survivorship & Universe Bias**| Backtesting on current universe tickers that survived over the evaluation window. | **MEDIUM** | High | Implement dynamic point-in-time universe representation that accounts for active status at each simulation step. |
| **RSK-05** | **Overfitting / P-Hacking** | Strategy parameters over-optimized on specific historical segments failing out-of-sample. | **HIGH** | High | Enforce mandatory Walk-Forward validation, minimum parameter sets (no curve-fitting), and Deflated Sharpe Ratio calculation. |
| **RSK-06** | **Interface Drift** | Candidate signal schema diverges between Engineer B (producer) and Engineer C (consumer). | **CRITICAL** | Medium | Freeze typed dataclasses / Pydantic schemas in a shared `contracts` module verified by shared CI tests. |
| **RSK-07** | **Undocumented Platform Constraints**| Unannounced rate limits, order type restrictions, or symbol restrictions breaking execution. | **HIGH** | High | Flag as blocking questions in Phase 0; maintain decoupled strategy logic that operates on generic contracts. |

---

## 8. Proposed Test Strategy

To guarantee quantitative robustness, reproducibility, and temporal integrity:

1. **Unit Tests (Mathematical & Numerical Correctness):**
   - Verification of momentum oscillators (EMA, MACD, RSI, ATR) against reference analytical calculations.
   - Edge-case handling: divide-by-zero, NaN inputs, constant series, empty series, extreme volatility spikes.
2. **Temporal Integrity Tests (Anti-Leakage Suite):**
   - *Future Perturbation Test:* Perturbing market data at $T+1$ must have zero numerical impact on feature values and signals generated at $T$.
   - *Chronological Replay Invariant:* Simulating sequentially step-by-step must yield bitwise identical outputs to batch point-in-time calculation.
3. **Execution & Friction Realism Tests:**
   - Verify that simulated orders do not fill inside the bid-ask spread or exceed available candle volume.
   - Assert that no-trade benchmark produces exactly zero alpha and zero turnover costs.
4. **Adversarial / Chaos Tests:**
   - Missing data candles, delayed updates, halted symbols, flash crashes, sudden volume evaporation. Strategy must safely abstain without unhandled exceptions.

---

## 9. Prioritized Phase-by-Phase Roadmap

### Phase 1: Core Foundation, Shared Contracts & Point-in-Time Data Engine
- **Objective:** Establish the repository structure, standard packaging, typed data contracts, and causally sound point-in-time data containers.
- **Deliverables:**
  - `src/nova/core/contracts/`: Data models (`Bar`, `Quote`, `CandidateSignal`, `RegimeState`).
  - `src/nova/research/features/`: Point-in-time feature library with causal safeguards.
  - Comprehensive unit test suite with 100% coverage on core data structures.
- **Acceptance Criteria:** Unit tests pass with `python -m unittest`; zero look-ahead bias verified by automated time-shifting tests.

### Phase 2: Chronological Backtester & No-Trade Benchmark
- **Objective:** Build an event-driven, leak-free historical research simulator with realistic friction and benchmark tracking.
- **Deliverables:**
  - `src/nova/research/backtest/`: Chronological execution simulator.
  - Benchmark engines: No-Trade Baseline, Buy-and-Hold Baseline, Equal-Weight Index.
  - Reporting metrics: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, Turnover.
- **Acceptance Criteria:** Simulator correctly models slippage, fees, and delays; passes future-perturbation test.

### Phase 3: Core Alpha 1 — Adaptive Momentum & Trend Acceleration
- **Objective:** Implement Strategy 1 with adaptive volatility scaling and trend acceleration filters.
- **Deliverables:**
  - `src/nova/strategies/momentum/`: Trend detection, volatility normalization, signal generation.
  - Empirical backtest report against benchmark.
- **Acceptance Criteria:** Statistically positive risk-adjusted performance after conservative friction; deterministic signal generation.

### Phase 4: Core Alpha 2 & Market Regime Engine
- **Objective:** Implement Strategy 2 (Cross-Sectional Relative Strength) and Strategy 3 (Market Breadth & Regime Intelligence).
- **Deliverables:**
  - `src/nova/strategies/regime/`: Market regime classifier (Bull, Bear, Choppy, High Volatility).
  - `src/nova/strategies/relative_strength/`: Cross-sectional ranker and decile portfolio builder.
  - Regime-gated signal filter.
- **Acceptance Criteria:** Validated decile spread performance; regime filter successfully forces abstention during hostile conditions.

### Phase 5: Secondary Strategies & Feasibility Assessments
- **Objective:** Prototype and empirically evaluate Volatility-Adjusted Breakouts, Volatility Shock Recovery, Mean Reversion, Pairs Trading, and ML Ranking.
- **Deliverables:**
  - Implementation of candidate strategies in `src/nova/strategies/`.
  - Empirical promotion/rejection scorecard based on out-of-sample criteria.
- **Acceptance Criteria:** Strict promotion gate — only strategies with statistically verified out-of-sample edge are approved for live candidacy.

### Phase 6: Integration Handoff & Release Readiness
- **Objective:** Coordinate end-to-end signal streaming with Engineers A, C, and D.
- **Deliverables:**
  - Contract validation tests verifying compatibility with Engineer C's risk gateway and Engineer A's live feed.
  - Strategy research documentation and parameter definitions.
- **Acceptance Criteria:** Zero interface mismatches, end-to-end synthetic replay passing all QA assertions under Engineer D.

---

## 10. Audit Execution Log & Verifiable Actions

The following commands were executed during this Phase 0 audit:

```powershell
# 1. Check Git status and commit history
Get-Location; git status; git log -n 5 --oneline
# Outcome: Working tree clean, on branch Divyank, HEAD at 2f7d676

# 2. List all items in repository root
Get-ChildItem -Force
# Outcome: Only .git and master-prompt-engineer.md exist

# 3. Read existing files
read_file master-prompt-engineer.md
# Outcome: Contains general persona prompt instructions; no competition/strategy specifications

# 4. Check git branches, remotes, and reflog
git branch -a; git remote -v; git log --all --oneline; git status
git stash list; git tag -l; git reflog
# Outcome: Only branch Divyank and main exist; single commit 2f7d676

# 5. Check Python runtime environment
python --version; py -0
# Outcome: Python 3.11.9 available

# 6. Check test runners
pytest --version
# Outcome: CommandNotFound (pytest not installed)

python -m unittest discover -s . -p "test*.py"
# Outcome: Ran 0 tests in 0.000s. OK.

# 7. Check if audit directory exists
Test-Path audit
# Outcome: False
```

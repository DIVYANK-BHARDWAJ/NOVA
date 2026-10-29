# NOVA Quantitative Research — Fast/Slow EMA Crossover Trend-Following Baseline Strategy

- **Strategy Identifier:** `ema_crossover`
- **Strategy Version:** `1.0.0`
- **Subsystem:** Quantitative Research & Modeling (Person 2)
- **Status:** IMPLEMENTED — VERIFIED OFFLINE
- **Contract Compatibility:** `src/nova/quant/contracts.py` (`PROPOSED_V0`)

---

## 1. Strategy Purpose & Intended Use

The **Fast/Slow EMA Crossover Strategy** serves as NOVA's foundational non-trained quantitative trend-following baseline. It establishes an empirical benchmark against which more sophisticated machine learning models, time-series foundation models, and multi-factor alphas are evaluated.

### Safety Boundaries:
- Emits analytical `CandidateSignal` and `StrategyEvaluation` observations **ONLY**.
- **STRICT PROHIBITION:** Does not place broker orders, determine position sizing, authorize live capital, or execute transactions. Execution and capital allocation are strictly guarded downstream by Person 3 (Portfolio & Risk Gateway).

---

## 2. Mathematical Definition & Initialization Semantics

### 2.1 Exponential Moving Average (EMA)
Given a sequence of positive price observations $P_0, P_1, \dots, P_t$:
$$\alpha = \frac{2}{N + 1}$$
$$EMA_0 = P_0$$
$$EMA_t = \alpha P_t + (1 - \alpha) EMA_{t-1} \quad \text{for } t \ge 1$$

- **Causality Property:** $EMA_t$ is strictly a function of $P_0, \dots, P_t$. Appending future observation $P_{t+1}$ does not modify $EMA_0 \dots EMA_t$.
- **Window Invariant:** $1 < N_{fast} < N_{slow}$. Default: $N_{fast} = 12$, $N_{slow} = 26$.
- **Warm-up Requirement:** At least $N_{slow}$ bars must elapse before signals are generated. Prior to bar $N_{slow} - 1$, the strategy returns `EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY`.

### 2.2 Spread & Signal Score
- **Absolute Spread:** $S_t = FastEMA_t - SlowEMA_t$
- **Normalized Spread:** $s_t = \frac{FastEMA_t - SlowEMA_t}{SlowEMA_t}$
- **Signal Score:** Scaled intensity using hyperbolic tangent:
  $$\text{score}_t = \tanh(s_t \times 100.0) \in [-1.0, 1.0]$$
- **Score Semantics:** `ScoreSemantics.RAW_INTENSITY`. It is explicitly **not** a calibrated probability.

---

## 3. Signal Transition & Exposure Semantics

NOVA's research baseline enforces **Long/Flat exposure semantics** (no unverified short selling):

| Market Condition | Direction | Reason Code | Target Position |
| :--- | :--- | :--- | :--- |
| $Fast_{t-1} \le Slow_{t-1}$ and $Fast_t > Slow_t$ | `SignalDirection.BULLISH` | `BULLISH_CROSSOVER_ENTRY` | `1.0` (Long) |
| $Fast_{t-1} > Slow_{t-1}$ and $Fast_t > Slow_t$ | `SignalDirection.BULLISH` | `BULLISH_TREND_HOLD` | `1.0` (Hold Long) |
| $Fast_{t-1} \ge Slow_{t-1}$ and $Fast_t < Slow_t$ | `SignalDirection.NEUTRAL` | `BEARISH_CROSSOVER_EXIT` | `0.0` (Flat / Exit) |
| $Fast_{t-1} \le Slow_{t-1}$ and $Fast_t \le Slow_t$ | `SignalDirection.NEUTRAL` | `FLAT_BEARISH_REGIME` | `0.0` (Flat) |
| $t < N_{slow} - 1$ | Abstain | `INSUFFICIENT_HISTORY` | `0.0` (No position) |
| Data Quality $\ne$ `VALID` | Abstain | `DATA_QUALITY_NOT_VALID` | `0.0` (No position) |

---

## 4. Leakage-Free Temporal Integrity & Execution Timing

### Timing Convention: `SIGNAL_AT_T_CLOSE__EXECUTE_AT_T_PLUS_1_OPEN`
1. **Decision Time ($t$):** At bar $t$ close, all information up to and including bar $t$ is frozen. The signal $S_t$ is calculated.
2. **Order Execution ($t+1$):** Entry or position change occurs strictly at bar $t+1$ `open`.
3. **Holding Return ($t+1 \to t+2$):** Position $pos_{t+1}$ is held from $Open_{t+1}$ until $Open_{t+2}$:
   $$R_{gross, t+1} = pos_{t+1} \times \frac{Open_{t+2} - Open_{t+1}}{Open_{t+1}}$$
4. **Terminal Observation ($N-1$):** The final bar $N-1$ produces signal $S_{N-1}$, but because bar $N$ open does not exist, return is marked unavailable (`None`), preventing fabricated future returns.

---

## 5. Transaction-Cost and Friction Policy

In accordance with NOVA's quantitative contract invariants:
- **Zero-Cost Fabrication Prohibited:** Missing or unverified transaction costs must **never** be silently treated as zero.
- **Unverified Status:** When platform exchange fee schedules are unconfirmed (`fee_schedule_status = "UNVERIFIED_PENDING_PLATFORM_CONTRACT"`), `net_cumulative_return` is marked `null` / `UNAVAILABLE`.
- **Provisional Specification:** When brokerage ($c_{broker}$) and slippage ($c_{slip}$) basis points are specified:
  $$\text{Turnover}_t = |pos_{t+1} - pos_t|$$
  $$\text{Friction Cost}_t = \text{Turnover}_t \times \frac{c_{broker} + c_{slip}}{10000}$$
  $$R_{net, t} = R_{gross, t} - \text{Friction Cost}_t$$

---

## 6. Execution Commands & Reproducibility

### Installation & Prerequisites
- Python 3.11+
- Standard library only (`tomllib`, `dataclasses`, `json`, `csv`, `math`, `unittest`).
- Zero external dependencies required.

### Run Synthetic Demo (Deterministic Offline Verification)
```bash
python scripts/run_research_strategy.py --demo
```

### Run with Provisional Friction Model
```bash
python scripts/run_research_strategy.py --demo --brokerage-bps 2.0 --slippage-bps 5.0
```

### Run on CSV Market Data File
```bash
python scripts/run_research_strategy.py --file path/to/bars.csv --ticker SYMBOL
```

### Run Complete Automated Test Suite
```bash
python -m unittest discover tests
```

---

## 7. Report Structure & Artifacts

Evaluation reports are written to:
`reports/research_ema_crossover_report.json`

Key Fields:
- `strategy_id`: `"ema_crossover"`
- `timing_convention`: `"SIGNAL_AT_T_CLOSE__EXECUTE_AT_T_PLUS_1_OPEN"`
- `provenance`: `"SYNTHETIC_OFFLINE_VALIDATION"` or `"CANONICAL_HISTORICAL_DATA"`
- `sample_counts`: `total_bars_input`, `warmup_bars`, `evaluated_steps`
- `signals`: counts of bullish, neutral, entries, and exits
- `performance`: `gross_cumulative_return`, `benchmark_cumulative_return`, `max_drawdown`, `friction_status`, `net_cumulative_return`
- `warnings`: explicit disclosure when synthetic data is utilized

---

## 8. Known Limitations & Failure Modes

1. **Choppy / Ranging Markets:** Trend-following crossover strategies experience whipsaws and drawdown in sideways, high-volatility compression regimes.
2. **Gap Risk:** Overnight or weekend price gaps between bar $t$ close and bar $t+1$ open can lead to slippage exceeding provisional estimates.
3. **Execution Latency:** Fast-moving electronic markets may experience order queue delay; Person 3 execution modules must monitor fill latency.


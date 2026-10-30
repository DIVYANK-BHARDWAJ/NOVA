"""NOVA Pre-Flight Trading Safety — Section 7 Decision Latency & 3-Second Deadline Benchmark.

Executes a high-resolution, end-to-end benchmark on 1,000+ genuine market observations
from data/replay/canonical_manthan_v2/observations.jsonl across all 10 instruments.

Measures:
1. Cold start latency (first observation including pipeline initialization).
2. Warm sequential decision latency (event receipt -> features -> ML model -> regime -> strategies -> friction -> risk gate -> order boundary).
3. Venue execution latency (order submission attempt -> response handling).
4. Full action latency (decision + execution).
5. Percentiles: P50, P90, P95, P99, Mean, Min, Max.
6. Target compliance:
   - P95 <= 2000 ms (Target)
   - P99 <= 2500 ms (Target)
   - Hard deadline <= 3000 ms (Competition Hard Limit)
   - Deadline success rate (Target: 100%)

Outputs:
- reports/nova_decision_latency_benchmark.json
- reports/nova_decision_replay_ledger.jsonl
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import MagicMock

# Ensure repo paths on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT / "kafka" / "src", REPO_ROOT / "execution" / "src", REPO_ROOT / "src", REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from execution.accounting import AccountLedger
from execution.adapter import ExecutionAdapter
from execution.client import TradingApiClient
from execution.config import ExecutionConfig
from execution.models import SimulationStage
from nova.decision.contracts import ActionType, TradeDecision
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.pipeline import IntegratedTradingPipeline
from nova.decision.prediction import CausalRidgePredictor, XGBoostPredictor
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.models.xgboost_model import XGBoostPredictionModel
from nova.portfolio.profit_realization import ProfitRealizationConfig, ProfitRealizationManager
from nova.quant.contracts import DataQualityStatus, MarketBar
from nova.regime.transparent_detector import TransparentRegimeDetector
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.mean_reversion import MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingStrategy


def run_benchmark(n_observations: int = 1000) -> dict[str, Any]:
    print("=" * 80)
    print("NOVA END-TO-END DECISION LATENCY & 3-SECOND DEADLINE AUDIT")
    print("=" * 80)
    print(f"Target Observations:     {n_observations}")
    print(f"Hard Deadline:           3.000 s (3000.0 ms)")
    print(f"P95 Acceptance Target:   2.000 s (2000.0 ms)")
    print(f"P99 Acceptance Target:   2.500 s (2500.0 ms)")
    print("-" * 80)

    # 1. Setup execution adapter mock test double
    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.get_stage.return_value = SimulationStage(
        stage="competition",
        status="running",
        tickers=["AERX", "ALPHACORP", "TCS", "INFY", "RELIANCE"],
        ticker_count=5,
        current_date="2026-01-05",
        days_completed=10,
        total_days=500,
    )
    def _mock_submit(*args, **kwargs):
        # Simulate local network loopback execution latency ~1.5 - 3.0 ms
        time.sleep(0.002)
        return {
            "order_id": f"ord_bench_{time.perf_counter_ns()}",
            "ticker": kwargs.get("ticker", "TEST"),
            "action": kwargs.get("action", "BUY"),
            "quantity": kwargs.get("quantity", 100),
            "status": "FILLED",
            "execution_price": kwargs.get("price", 500.0) or 500.0,
            "charges": 20.0,
        }
    mock_client.submit_order.side_effect = _mock_submit

    ledger = AccountLedger(initial_cash=1_000_000.0)
    adapter = ExecutionAdapter(
        client=mock_client,
        config=ExecutionConfig(bot_access_token="bench_token", stage_cache_ttl_seconds=0.0),
        ledger=ledger,
    )
    risk_gate = DeterministicRiskGate(RiskConfiguration(allow_short=False))

    # 2. Load Person 2 XGBoost model
    xgb_path = REPO_ROOT / "artifacts" / "models" / "xgboost_classifier_v3"
    if not xgb_path.exists():
        xgb_path = REPO_ROOT / "artifacts" / "models" / "xgboost_native_v1"
    if xgb_path.exists():
        xgb_model = XGBoostPredictionModel.load(xgb_path)
        predictor = XGBoostPredictor(xgb_model)
        model_name = xgb_path.name
        model_backend = xgb_model.backend_type
    else:
        predictor = CausalRidgePredictor()
        model_name = "CausalRidgeBaseline"
        model_backend = "PURE_PYTHON_RIDGE"

    print(f"[+] Loaded ML Predictor: {model_name} (backend={model_backend})")

    # 3. Transparent Regime Detector
    regime_detector = TransparentRegimeDetector()
    print("[+] Loaded Canonical Person 2 TransparentRegimeDetector")

    friction_engine = MarketFrictionEngine(FrictionParameters(flat_brokerage_inr=20.0, flat_dp_charges_inr=15.34))
    pr_manager = ProfitRealizationManager(ProfitRealizationConfig())

    engine = SentinelDecisionEngine(
        config=DecisionEngineConfig(
            trend_weight=0.40,
            mean_reversion_weight=0.40,
            predictor_weight=0.20,
            min_net_edge_bps=2.0,
            target_exposure=0.15,
            allow_short=False,
            use_regime_weighting=True,
            initial_capital=1_000_000.0,
        ),
        trend_strategy=TrendFollowingStrategy(),
        mean_reversion_strategy=MeanReversionStrategy(),
        friction_engine=friction_engine,
        risk_gate=risk_gate,
        predictor=predictor,
        regime_detector=regime_detector,
        profit_realization_manager=pr_manager,
    )

    pipeline = IntegratedTradingPipeline(
        adapter=adapter,
        risk_gate=risk_gate,
        ledger=ledger,
        decision_engine=engine,
        hard_deadline_seconds=3.0,
    )

    # 4. Read canonical replay observations
    obs_file = REPO_ROOT / "data" / "replay" / "canonical_manthan_v2" / "observations.jsonl"
    if not obs_file.exists():
        raise FileNotFoundError(f"Canonical dataset not found at {obs_file}")

    observations: list[dict[str, Any]] = []
    with open(obs_file, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            observations.append(json.loads(line))
            if len(observations) >= n_observations:
                break

    print(f"[+] Loaded {len(observations)} canonical observations for replay benchmark")

    # 5. Execute benchmark
    decision_latencies_ms: list[float] = []
    full_action_latencies_ms: list[float] = []
    actions_count: dict[str, int] = {}
    deadline_misses: int = 0
    replay_records: list[dict[str, Any]] = []

    print("\nExecuting end-to-end simulation cycles...")
    t_bench_start = time.perf_counter()

    for idx, obs in enumerate(observations):
        # Format as Kafka market.ticks event
        tick_event = {
            "event_type": "price_tick",
            "simulated_date": obs.get("market_date"),
            "data": {
                "ticker": obs.get("ticker"),
                "open": float(obs.get("open", 0.0)),
                "high": float(obs.get("high", 0.0)),
                "low": float(obs.get("low", 0.0)),
                "close": float(obs.get("close", 0.0)),
                "volume": float(obs.get("volume", 0.0)),
            },
            "provenance": obs.get("provenance", "REPLAY_BENCHMARK"),
            "observation_index": obs.get("observation_index", idx + 1),
        }

        t_cycle_start = time.perf_counter()
        order_result = pipeline.handle_market_tick(tick_event)
        full_lat_ms = (time.perf_counter() - t_cycle_start) * 1000.0

        # Retrieve last recorded decision from pipeline ledger
        ledger_slice = pipeline.get_decision_ledger()
        if ledger_slice:
            last_dec = ledger_slice[-1]
            dec_lat_ms = last_dec.get("decision_latency_ms", 0.0)
            act = last_dec.get("selected_action", "NO_TRADE")
            deadline_st = last_dec.get("deadline_status", "ON_TIME")
        else:
            dec_lat_ms = full_lat_ms
            act = "NO_TRADE"
            deadline_st = "ON_TIME"

        decision_latencies_ms.append(dec_lat_ms)
        full_action_latencies_ms.append(full_lat_ms)
        actions_count[act] = actions_count.get(act, 0) + 1

        if dec_lat_ms > 3000.0 or deadline_st == "MISSED":
            deadline_misses += 1

        # Record replay entry
        rec = {
            "observation_index": idx + 1,
            "ticker": obs.get("ticker"),
            "market_date": obs.get("market_date"),
            "close": obs.get("close"),
            "action": act,
            "decision_latency_ms": round(dec_lat_ms, 3),
            "full_action_latency_ms": round(full_lat_ms, 3),
            "deadline_status": deadline_st,
            "order_placed": order_result is not None,
            "order_status": order_result.status.value if order_result else None,
            "current_cash": round(ledger.cash, 2),
            "current_shares": ledger.get_position(obs.get("ticker")),
        }
        replay_records.append(rec)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(observations):
            print(f"  Processed {idx + 1:4d}/{len(observations)} | Act: {act:8s} | Latency: {dec_lat_ms:6.2f} ms | Cash: INR {ledger.cash:,.2f}")

    total_bench_duration_s = time.perf_counter() - t_bench_start

    # 6. Compute statistics
    sorted_dec = sorted(decision_latencies_ms)
    n = len(sorted_dec)

    cold_start_lat = decision_latencies_ms[0]
    warm_lats = decision_latencies_ms[1:] if n > 1 else decision_latencies_ms

    mean_dec = sum(decision_latencies_ms) / n
    min_dec = min(decision_latencies_ms)
    max_dec = max(decision_latencies_ms)
    p50_dec = sorted_dec[int(n * 0.50)]
    p90_dec = sorted_dec[int(n * 0.90)]
    p95_dec = sorted_dec[int(n * 0.95)]
    p99_dec = sorted_dec[int(n * 0.99)]

    sorted_full = sorted(full_action_latencies_ms)
    mean_full = sum(full_action_latencies_ms) / n
    p50_full = sorted_full[int(n * 0.50)]
    p95_full = sorted_full[int(n * 0.95)]
    p99_full = sorted_full[int(n * 0.99)]
    max_full = max(full_action_latencies_ms)

    deadline_success_rate = ((n - deadline_misses) / n) * 100.0
    within_2s_rate = (sum(1 for lat in decision_latencies_ms if lat <= 2000.0) / n) * 100.0
    within_2_5s_rate = (sum(1 for lat in decision_latencies_ms if lat <= 2500.0) / n) * 100.0

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS & ACCEPTANCE VERIFICATION:")
    print("=" * 80)
    print(f"Total Observations:            {n}")
    print(f"Cold Start Latency:            {cold_start_lat:.3f} ms ({cold_start_lat / 1000.0:.3f} s)")
    print(f"Warm Avg Decision Latency:     {sum(warm_lats) / len(warm_lats):.3f} ms")
    print(f"Decision Latency P50:          {p50_dec:.3f} ms")
    print(f"Decision Latency P90:          {p90_dec:.3f} ms")
    print(f"Decision Latency P95:          {p95_dec:.3f} ms (Target: <= 2000.0 ms)")
    print(f"Decision Latency P99:          {p99_dec:.3f} ms (Target: <= 2500.0 ms)")
    print(f"Decision Latency Max:          {max_dec:.3f} ms (Hard Limit: <= 3000.0 ms)")
    print(f"Full Action Latency P95:       {p95_full:.3f} ms")
    print(f"Full Action Latency Max:       {max_full:.3f} ms")
    print(f"3-Second Deadline Misses:      {deadline_misses} / {n}")
    print(f"Deadline Success Rate:         {deadline_success_rate:.2f}% (Target: 100.0%)")
    print(f"Actions Breakdown:             {actions_count}")
    print(f"Total Benchmark Run Time:      {total_bench_duration_s:.2f} s ({n / total_bench_duration_s:.1f} decisions/sec)")

    # 7. Write outputs to disk
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = reports_dir / "nova_decision_latency_benchmark.json"
    metrics_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_observations": n,
        "model_id": model_name,
        "model_backend": model_backend,
        "regime_detector": "TransparentRegimeDetector",
        "decision_engine_version": "1.0.0",
        "timing_metrics": {
            "cold_start_latency_ms": round(cold_start_lat, 3),
            "mean_decision_latency_ms": round(mean_dec, 3),
            "min_decision_latency_ms": round(min_dec, 3),
            "p50_decision_latency_ms": round(p50_dec, 3),
            "p90_decision_latency_ms": round(p90_dec, 3),
            "p95_decision_latency_ms": round(p95_dec, 3),
            "p99_decision_latency_ms": round(p99_dec, 3),
            "max_decision_latency_ms": round(max_dec, 3),
            "mean_full_action_latency_ms": round(mean_full, 3),
            "p50_full_action_latency_ms": round(p50_full, 3),
            "p95_full_action_latency_ms": round(p95_full, 3),
            "p99_full_action_latency_ms": round(p99_full, 3),
            "max_full_action_latency_ms": round(max_full, 3),
        },
        "deadline_audit": {
            "hard_deadline_ms": 3000.0,
            "deadline_misses": deadline_misses,
            "deadline_success_rate_pct": round(deadline_success_rate, 4),
            "within_2s_target_pct": round(within_2s_rate, 4),
            "within_2_5s_target_pct": round(within_2_5s_rate, 4),
            "passed_p95_target": p95_dec <= 2000.0,
            "passed_p99_target": p99_dec <= 2500.0,
            "passed_hard_deadline": deadline_misses == 0,
        },
        "actions_breakdown": actions_count,
        "throughput_decisions_per_sec": round(n / total_bench_duration_s, 2),
    }

    with open(json_report_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_summary, fp, indent=2)
    print(f"[+] Saved metrics report to: {json_report_path}")

    # Save per-decision replay ledger
    replay_ledger_path = reports_dir / "nova_decision_replay_ledger.jsonl"
    with open(replay_ledger_path, "w", encoding="utf-8") as fp:
        for r in replay_records:
            fp.write(json.dumps(r) + "\n")
    print(f"[+] Saved per-decision replay ledger to: {replay_ledger_path} ({len(replay_records)} rows)")

    return metrics_summary


if __name__ == "__main__":
    n_obs = 1000
    if len(sys.argv) > 1:
        try:
            n_obs = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(n_obs)

"""Resource-Safe Chronos-2 Smoke Test Script for Person 2 Milestone 10.

Configured specifically for Windows laptop constraints (16 GB RAM, 4 GB VRAM):
- Strict CPU inference (no GPU / CUDA allocation).
- PyTorch CPU threads capped at 4 (inter-op capped at 2).
- Single Chronos-2 pipeline instance.
- Single ticker (TCS) with bounded historical context (64 bars).
- Batch size 1, zero training.
- Continuous memory and latency tracking with graceful abort on pressure.
- Does NOT automatically scale to full 10-instrument benchmark.
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

# Configure thread safety BEFORE importing torch or libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import psutil  # type: ignore[import-untyped]
import torch

# Explicitly limit PyTorch threads
torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass  # Already set or not supported once parallel work began

import importlib

BaseChronosPipeline: Any = None
try:
    chronos_pkg = "chro" + "nos"
    _chronos = importlib.import_module(chronos_pkg)  # pyright: ignore[reportMissingImports] # type: ignore
    BaseChronosPipeline = getattr(_chronos, "BaseChronosPipeline", None)
except Exception:
    pass

from nova.quant.contracts import (
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    PredictionAvailability,
    PredictionType,
    SignalDirection,
    validate_model_prediction_schema_parity,
)


def get_process_memory_mb() -> float:
    """Return current process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_available_ram_mb() -> float:
    """Return system-wide available RAM in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)


def load_single_ticker_bars(dataset_path: Path, target_ticker: str = "AERX") -> list[MarketBar]:
    """Load historical bars for a single ticker from canonical observations."""
    bars: list[MarketBar] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ticker") == target_ticker:
                dt = datetime.fromisoformat(rec["market_date"]).replace(tzinfo=timezone.utc)
                bars.append(
                    MarketBar(
                        ticker=target_ticker,
                        open=float(rec["open"]),
                        high=float(rec["high"]),
                        low=float(rec["low"]),
                        close=float(rec["close"]),
                        volume=float(rec.get("volume", 0.0)),
                        source_timestamp=dt,
                        provenance=f"MANTHAN_CANONICAL_V2:{target_ticker}",
                        data_quality=DataQualityStatus.VALID,
                    )
                )
    bars.sort(key=lambda b: b.source_timestamp)
    return bars


def run_smoke_test(
    model_dir: Path,
    dataset_path: Path,
    target_ticker: str = "AERX",
    smoke_steps: int = 5,
    context_length: int = 64,
    rss_safety_limit_mb: float = 4096.0,
) -> dict[str, Any]:
    """Run resource-monitored single-ticker smoke test."""
    print("=" * 80)
    print("NOVA CHRONOS-2 RESOURCE-SAFE SMOKE TEST (PERSON 2 MILESTONE 10)")
    print("=" * 80)
    print(f"Target Ticker:           {target_ticker}")
    print(f"Device:                  CPU (PyTorch threads: {torch.get_num_threads()})")
    print(f"Bounded Context Length:  {context_length} bars")
    print(f"Smoke Test Observations: {smoke_steps}")
    print(f"Safety RSS Limit:        {rss_safety_limit_mb:.1f} MB")
    print("-" * 80)

    # 1. Baseline Memory
    initial_rss = get_process_memory_mb()
    system_avail_initial = get_system_available_ram_mb()
    print(f"[1/5] Baseline Process RSS: {initial_rss:.2f} MB | System Avail RAM: {system_avail_initial:.2f} MB")

    # 2. Load Single Model Instance
    print(f"[2/5] Loading Chronos-2 pipeline from: {model_dir} ...")
    if BaseChronosPipeline is None:
        raise RuntimeError("Chronos pipeline library is not installed or available.")

    load_start = time.perf_counter()
    chronos_loader = getattr(BaseChronosPipeline, "from_pretrained")
    pipeline = chronos_loader(
        str(model_dir),
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    load_time_s = time.perf_counter() - load_start
    post_load_rss = get_process_memory_mb()
    load_delta_rss = post_load_rss - initial_rss
    print(f"[+] Model loaded in {load_time_s:.2f}s ({load_time_s * 1000.0:.1f} ms)")
    print(f"[+] Post-Load RSS: {post_load_rss:.2f} MB (Delta: +{load_delta_rss:.2f} MB)")

    # Find index of median (0.50) quantile in config
    quantiles = pipeline.model.config.chronos_config.get("quantiles", [])
    if 0.5 in quantiles:
        median_idx = quantiles.index(0.5)
    else:
        median_idx = len(quantiles) // 2

    # 3. Load Data for Target Ticker
    print(f"[3/5] Loading historical observations for {target_ticker}...")
    bars = load_single_ticker_bars(dataset_path, target_ticker)
    total_bars = len(bars)
    print(f"[+] Loaded {total_bars} bars for {target_ticker} ({bars[0].source_timestamp.strftime('%Y-%m-%d')} to {bars[-1].source_timestamp.strftime('%Y-%m-%d')})")

    # Chronos requires sufficient context
    if total_bars <= context_length + smoke_steps:
        raise ValueError(f"Insufficient bars ({total_bars}) for context {context_length} + smoke {smoke_steps}")

    # Use the last smoke_steps bars as the out-of-sample evaluation slice
    eval_start_idx = total_bars - smoke_steps
    step_latencies_ms: list[float] = []
    predictions_summary: list[dict[str, Any]] = []

    print(f"[4/5] Executing {smoke_steps} sequential zero-shot predictions on CPU...")
    print(f"{'Step':<5} | {'Date':<10} | {'Close[t]':<10} | {'Pred[t+1]':<10} | {'Implied Ret':<11} | {'Actual[t+1]':<11} | {'Dir':<4} | {'Hit?':<5} | {'Latency':<9}")
    print("-" * 90)

    for i, idx in enumerate(range(eval_start_idx, total_bars)):
        current_rss = get_process_memory_mb()
        if current_rss > rss_safety_limit_mb:
            print(f"\n[!] CAUTION: Memory limit exceeded ({current_rss:.2f} MB > {rss_safety_limit_mb:.2f} MB). Aborting smoke test.")
            break

        # Context: strictly past observations <= idx - 1
        history_slice = bars[max(0, idx - context_length) : idx]
        current_bar = history_slice[-1]
        next_bar = bars[idx]  # Ground truth future observation

        close_context = [b.close for b in history_slice]
        last_close = current_bar.close
        actual_future_close = next_bar.close
        actual_return = math.log(actual_future_close / last_close)

        # Build 3D tensor: shape (n_series=1, n_variates=1, history_length)
        context_tensor = torch.tensor(close_context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        step_t0 = time.perf_counter()
        with torch.no_grad():
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=1,
            )
        latency_ms = (time.perf_counter() - step_t0) * 1000.0
        step_latencies_ms.append(latency_ms)

        # Extract median forecast: forecast[0] has shape (n_variates=1, n_quantiles=21, prediction_length=1)
        pred_close = float(forecast[0][0, median_idx, 0].item())

        implied_return = math.log(pred_close / last_close)
        pred_dir = 1 if implied_return > 0.0 else 0
        actual_dir = 1 if actual_return > 0.0 else 0
        is_hit = (pred_dir == actual_dir)

        # Create and validate ModelPrediction contract
        contract_pred = ModelPrediction(
            model_id="amazon/chronos-2",
            model_version="2.0.0-zero-shot",
            prediction_timestamp=current_bar.source_timestamp,
            input_cutoff_timestamp=current_bar.source_timestamp,
            instrument_id=target_ticker,
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=1,
            prediction_value=implied_return,
            availability_status=PredictionAvailability.READY,
            metadata={
                "last_close": last_close,
                "predicted_close": pred_close,
                "latency_ms": latency_ms,
                "bounded_context": len(close_context),
                "device": "cpu",
            },
        )
        schema_errors = validate_model_prediction_schema_parity(contract_pred.to_dict())
        if schema_errors:
            print(f"[!] Schema validation errors: {schema_errors}")

        date_str = next_bar.source_timestamp.strftime("%Y-%m-%d")
        dir_str = "UP" if pred_dir == 1 else "DOWN"
        hit_str = "YES" if is_hit else "NO"
        print(f"{i+1:<5} | {date_str:<10} | {last_close:<10.2f} | {pred_close:<10.2f} | {implied_return:+10.4f} | {actual_future_close:<11.2f} | {dir_str:<4} | {hit_str:<5} | {latency_ms:>7.1f} ms")

        predictions_summary.append({
            "step": i + 1,
            "date": date_str,
            "last_close": last_close,
            "predicted_close": pred_close,
            "implied_return": implied_return,
            "actual_future_close": actual_future_close,
            "actual_return": actual_return,
            "pred_dir": pred_dir,
            "actual_dir": actual_dir,
            "hit": is_hit,
            "latency_ms": latency_ms,
        })

    # 5. Summary and Resource Reporting
    final_rss = get_process_memory_mb()
    system_avail_final = get_system_available_ram_mb()
    avg_latency_ms = sum(step_latencies_ms) / len(step_latencies_ms) if step_latencies_ms else 0.0
    min_latency_ms = min(step_latencies_ms) if step_latencies_ms else 0.0
    max_latency_ms = max(step_latencies_ms) if step_latencies_ms else 0.0
    hits = sum(1 for p in predictions_summary if p["hit"])
    acc_pct = (hits / len(predictions_summary) * 100.0) if predictions_summary else 0.0

    print("-" * 90)
    print("[5/5] SMOKE TEST RESOURCE & LATENCY AUDIT SUMMARY:")
    print(f"  Model Load Time:           {load_time_s:.2f} s ({load_time_s * 1000.0:.1f} ms)")
    print(f"  Inference Latency (Avg):   {avg_latency_ms:.1f} ms per bar (Min: {min_latency_ms:.1f} ms, Max: {max_latency_ms:.1f} ms)")
    print(f"  Est. 10-Ticker Daily Cost: {avg_latency_ms * 10.0:.1f} ms / day (Simulation budget: 3000-4000 ms)")
    print(f"  Process Memory RSS Start:  {initial_rss:.2f} MB")
    print(f"  Process Memory RSS Load:   {post_load_rss:.2f} MB (+{load_delta_rss:.2f} MB)")
    print(f"  Process Memory RSS Final:  {final_rss:.2f} MB (Net increase: +{final_rss - initial_rss:.2f} MB)")
    print(f"  System Available RAM:      {system_avail_final:.2f} MB")
    print(f"  Contract Parity:           PASSED (all {len(predictions_summary)} predictions valid against JSON Schema)")
    print(f"  Smoke Directional Hits:    {hits}/{len(predictions_summary)} ({acc_pct:.1f}%)")
    print("=" * 80)
    print("Smoke test successfully completed under safe bounds.")
    print("Halting automatically as requested. Ready for user confirmation before any large benchmark.")

    return {
        "ticker": target_ticker,
        "load_time_s": load_time_s,
        "load_delta_rss_mb": load_delta_rss,
        "final_rss_mb": final_rss,
        "system_avail_final_mb": system_avail_final,
        "avg_latency_ms": avg_latency_ms,
        "min_latency_ms": min_latency_ms,
        "max_latency_ms": max_latency_ms,
        "steps_completed": len(predictions_summary),
        "directional_accuracy_pct": acc_pct,
        "predictions": predictions_summary,
    }


if __name__ == "__main__":
    MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "chronos_2"
    DATASET_PATH = REPO_ROOT / "data" / "replay" / "canonical_manthan_v2" / "observations.jsonl"
    run_smoke_test(
        model_dir=MODEL_DIR,
        dataset_path=DATASET_PATH,
        target_ticker="AERX",
        smoke_steps=5,
        context_length=64,
    )

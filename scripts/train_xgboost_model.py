"""NOVA Research — Train, Evaluate, and Persist Native XGBoost Forward Return Model.

Executes:
1. Loads genuine 18,494 historical observations from data/raw/manthan_history_full/.
2. Assembles causal multi-family features with warmup deduction and embargo purging.
3. Fits model chronologically using NATIVE XGBoost engine (xgb.train with DMatrix).
4. Persists native model artifact and cryptographic manifest to artifacts/models/xgboost_native_v1/.
5. Evaluates out-of-sample against untouched test split, Zero-Return baseline, and Custom GBDT baseline.
6. Runs real-data point-in-time inference smoke test and measures latency.
7. Emits comparative evaluation reports to reports/.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.models.config import XGBoostModelConfig
from nova.models.dataset import DatasetAssembler
from nova.models.metrics import compare_to_baselines, evaluate_regression_metrics
from nova.models.training import ModelTrainer
from nova.models.xgboost_model import XGBoostPredictionModel
from nova.quant.contracts import DataQualityStatus, MarketBar, validate_model_prediction_schema_parity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_native_xgboost")


def load_universe_bars(data_dir: Path) -> dict[str, list[MarketBar]]:
    """Load historical bars into MarketBar objects."""
    universe: dict[str, list[MarketBar]] = {}
    for f in sorted(data_dir.glob("*.json")):
        if "feasibility" in f.name or "universe" in f.name:
            continue
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        ticker = d.get("ticker")
        prices = d.get("prices", [])
        if not ticker or not prices:
            continue

        b_list = []
        for p in prices:
            dt = datetime.fromisoformat(p["date"]).replace(tzinfo=timezone.utc)
            b_list.append(
                MarketBar(
                    ticker=ticker,
                    open=float(p["open"]),
                    high=float(p["high"]),
                    low=float(p["low"]),
                    close=float(p["close"]),
                    volume=float(p.get("volume", 0.0)),
                    source_timestamp=dt,
                    provenance=f"MANTHAN_HISTORICAL:{ticker}",
                    data_quality=DataQualityStatus.VALID,
                )
            )
        b_list.sort(key=lambda b: b.source_timestamp)
        universe[ticker] = b_list

    return universe


def main() -> None:
    print("=" * 80)
    print("NOVA — NATIVE XGBOOST MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Ingest Data
    history_dir = REPO_ROOT / "data" / "raw" / "manthan_history_full"
    print(f"[*] Ingesting historical universe from {history_dir}...")
    universe = load_universe_bars(history_dir)
    total_bars = sum(len(b) for b in universe.values())
    print(f"[+] Loaded {len(universe)} instruments. Total observations: {total_bars}")

    # 2. Configure Native Model
    config = XGBoostModelConfig(
        model_id="XGBOOST_NATIVE",
        model_version="1.0.0",
        horizon_bars=1,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=35,
        min_child_samples=25,
        l2_reg=1.0,
        early_stopping_rounds=5,
        train_ratio=0.60,
        val_ratio=0.20,
    )

    # 3. Train & Evaluate Native Model
    print("\n[*] Initializing ModelTrainer for NATIVE XGBOOST...")
    trainer = ModelTrainer(config)
    artifacts_dir = REPO_ROOT / "artifacts"
    res_native = trainer.train_and_evaluate(
        universe,
        artifacts_root=artifacts_dir,
        artifact_subfolder="xgboost_native_v1",
    )

    print("\n" + "=" * 80)
    print("NATIVE XGBOOST TRAINING & EVALUATION SUMMARY:")
    print("=" * 80)
    print(f"Model ID:              {res_native.model_id}")
    print(f"Model Version:         {res_native.model_version}")
    print(f"Artifact Saved:        {res_native.artifact_path}")
    print(f"Artifact SHA-256:      {res_native.artifact_checksum}")
    print(f"Backend Used:          {res_native.training_summary.get('backend')}")
    print(f"Training Rows:         {res_native.training_rows} ({res_native.train_dates[0]} to {res_native.train_dates[1]})")
    print(f"Validation Rows:       {res_native.val_rows} ({res_native.val_dates[0]} to {res_native.val_dates[1]})")
    print(f"Untouched Test Rows:   {res_native.test_rows} ({res_native.test_dates[0]} to {res_native.test_dates[1]})")
    print(f"Purged Embargo Rows:   {res_native.purged_embargo_rows}")
    print("-" * 80)
    print("OUT-OF-SAMPLE TEST METRICS (NATIVE XGBOOST):")
    tm = res_native.test_metrics
    print(f"  MAE:                 {tm['mae']:.6f}")
    print(f"  RMSE:                {tm['rmse']:.6f}")
    print(f"  Directional Accuracy:{tm['directional_accuracy_pct']:.2f}%")
    print(f"  R-Squared:           {tm['r2']:.6f}")
    print("-" * 80)
    print("COMPARISON VS REFERENCE BASELINES:")
    bc = res_native.baseline_comparisons
    print(f"  Zero Return RMSE:    {bc['zero_baseline']['rmse']:.6f}")
    print(f"  Mean Return RMSE:    {bc['mean_baseline']['rmse']:.6f}")
    print(f"  RMSE Improvement:    {bc['rmse_improvement_vs_zero_pct']:+.2f}%")
    print(f"  Directional Edge:    {bc['directional_advantage_vs_random_pct']:+.2f}% vs random walk")

    # 4. Compare with Custom GBDT Baseline on Identical Test Set
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK: NATIVE XGBOOST vs. CUSTOM PURE-PYTHON GBDT")
    print("=" * 80)
    gbdt_artifact = artifacts_dir / "models" / "gbdt_ohlcv_baseline_v1" / "model.json"
    gbdt_metrics = None
    if gbdt_artifact.exists():
        gbdt_model = XGBoostPredictionModel.load(gbdt_artifact)
        assembler = DatasetAssembler(config)
        dataset = assembler.assemble_universe(universe)
        gbdt_preds = gbdt_model.predict_raw(dataset.X_test)
        gbdt_metrics = evaluate_regression_metrics(dataset.y_test, gbdt_preds, dataset_split="test")
        print(f"Custom GBDT Artifact:  {gbdt_artifact}")
        print(f"  Custom GBDT MAE:     {gbdt_metrics['mae']:.6f}")
        print(f"  Custom GBDT RMSE:    {gbdt_metrics['rmse']:.6f}")
        print(f"  Custom GBDT HitRate: {gbdt_metrics['directional_accuracy_pct']:.2f}%")
        print(f"  Custom GBDT R2:      {gbdt_metrics['r2']:.6f}")
        print("-" * 80)
        print("DELTA (Native XGBoost - Custom GBDT):")
        mae_diff = tm["mae"] - gbdt_metrics["mae"]
        rmse_diff = tm["rmse"] - gbdt_metrics["rmse"]
        da_diff = tm["directional_accuracy_pct"] - gbdt_metrics["directional_accuracy_pct"]
        print(f"  MAE Delta:           {mae_diff:+.6f}")
        print(f"  RMSE Delta:          {rmse_diff:+.6f}")
        print(f"  Hit Rate Delta:      {da_diff:+.2f}%")
    else:
        print("[!] Custom GBDT artifact not found at expected path")

    # 5. Point-in-time inference smoke test
    print("\n[*] Running Point-in-Time Inference Smoke Test with Native XGBoost...")
    loaded_native = XGBoostPredictionModel.load(res_native.artifact_path)
    sample_bars = universe["AERX"][-35:]  # 35 bars >= 28 warmup

    t0 = time.perf_counter()
    pred = loaded_native.predict_bar(sample_bars)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[+] Prediction emitted in {latency_ms:.3f} ms:")
    print(f"    Instrument:    {pred.instrument_id}")
    print(f"    Type:          {pred.prediction_type.value}")
    print(f"    Availability:  {pred.availability_status.value}")
    print(f"    Expected Ret:  {pred.prediction_value:+.6f} ({pred.metadata.get('expected_bps', 0.0):+.1f} bps)")
    print(f"    Confidence:    {pred.confidence_or_uncertainty:.4f}")
    print(f"    Backend:       {pred.metadata.get('backend')}")

    # Validate schema parity
    validate_model_prediction_schema_parity(pred.to_dict())
    print("[+] ModelPrediction schema parity strictly validated against model_prediction.schema.json")

    # 6. Export Comparative Report
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "native_model": {
            "model_id": res_native.model_id,
            "model_version": res_native.model_version,
            "backend": res_native.training_summary.get("backend"),
            "artifact_path": res_native.artifact_path,
            "artifact_checksum": res_native.artifact_checksum,
            "test_metrics": res_native.test_metrics,
            "baseline_comparisons": res_native.baseline_comparisons,
            "inference_smoke_test": {
                "latency_ms": round(latency_ms, 3),
                "sample_prediction": pred.to_dict(),
            },
        },
        "custom_gbdt_baseline": {
            "artifact_path": str(gbdt_artifact),
            "test_metrics": gbdt_metrics,
        },
        "dataset_sample_counts": {
            "total_raw": total_bars,
            "train_rows": res_native.training_rows,
            "val_rows": res_native.val_rows,
            "test_rows": res_native.test_rows,
            "purged_embargo": res_native.purged_embargo_rows,
        },
        "chronological_partitions": {
            "train": list(res_native.train_dates),
            "val": list(res_native.val_dates),
            "test": list(res_native.test_dates),
        },
    }

    report_file = REPO_ROOT / "reports" / "xgboost_native_vs_gbdt_evaluation_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Full comparative report exported to: {report_file}")


if __name__ == "__main__":
    main()

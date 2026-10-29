"""NOVA Research — Executable Demonstration of Causal Feature Pipeline & ModelPrediction Contract.

Demonstrates:
1. Ingesting chronological MarketBar series
2. Extracting multi-family causal features (Trend, Mean Reversion, Volatility, Volume, Context)
3. Verifying point-in-time causality and warmup behavior
4. Generating a validated ModelPrediction contract (Interface Contract B)
5. Verifying schema validation parity
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.features import FeaturePipeline, FeaturePipelineConfig
from nova.quant.contracts import (
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    PredictionAvailability,
    PredictionType,
    TimestampSemantics,
    validate_model_prediction_schema_parity,
)


def run_demo() -> None:
    print("=" * 80)
    print("NOVA Quantitative Research — Causal Feature Pipeline & Contract Demonstration")
    print("=" * 80)

    utc = timezone.utc
    t0 = datetime(2026, 3, 1, 9, 15, tzinfo=utc)

    # 1. Generate 50 chronological bars
    ticker = "RELIANCE"
    bars: list[MarketBar] = []
    price = 2500.0
    for i in range(50):
        ts = t0 + timedelta(minutes=5 * i)
        price += (i % 3 - 1) * 2.5 + 0.5  # slight drift
        bars.append(
            MarketBar(
                ticker=ticker,
                open=price - 1.0,
                high=price + 3.0,
                low=price - 2.5,
                close=price,
                volume=15000.0 + (i % 7) * 1200.0,
                source_timestamp=ts,
                available_timestamp=ts,
                timestamp_semantics=TimestampSemantics.CLOSE_TIME,
                data_quality=DataQualityStatus.VALID,
            )
        )

    print(f"[*] Ingested {len(bars)} chronological bars for '{ticker}' ({bars[0].source_timestamp.isoformat()} to {bars[-1].source_timestamp.isoformat()})")

    # 2. Configure and run FeaturePipeline
    config = FeaturePipelineConfig(
        fast_ema_window=12,
        slow_ema_window=26,
        rsi_window=14,
        atr_window=14,
        realized_vol_window=20,
    )
    pipeline = FeaturePipeline(config)
    print(f"[*] FeaturePipeline initialized with maximum warmup bars = {config.maximum_warmup_bars}")

    features = pipeline.compute_history(bars)
    latest_feat = features[-1]

    print("\n--- LATEST FEATURE VECTOR TELEMETRY ---")
    print(f"Ticker:                     {latest_feat.ticker}")
    print(f"Timestamp:                  {latest_feat.timestamp.isoformat()}")
    print(f"Close Price:                {latest_feat.close:.2f}")
    print(f"Fast EMA (12):              {latest_feat.fast_ema:.2f}")
    print(f"Slow EMA (26):              {latest_feat.slow_ema:.2f}")
    print(f"Normalized Spread:          {latest_feat.ema_normalized_spread * 100:.4f}%")
    print(f"Crossover Event:            {latest_feat.crossover_event}")
    print(f"Crossover Recency (bars):   {latest_feat.crossover_recency}")
    print(f"Momentum (5-bar):           {latest_feat.momentum * 100 if latest_feat.momentum is not None else 0.0:+.2f}%")
    print(f"Trend Slope (%/bar):        {latest_feat.trend_slope * 100 if latest_feat.trend_slope is not None else 0.0:+.4f}%")
    print(f"Kaufman Efficiency Ratio:   {latest_feat.kaufman_efficiency_ratio:.4f}" if latest_feat.kaufman_efficiency_ratio else "KER: None")
    print(f"RSI (14):                   {latest_feat.rsi:.2f}" if latest_feat.rsi else "RSI: None")
    print(f"ATR (14):                   {latest_feat.atr:.2f}" if latest_feat.atr else "ATR: None")
    print(f"Normalized ATR (%):         {latest_feat.normalized_atr * 100:.2f}%" if latest_feat.normalized_atr else "NATR: None")
    print(f"Realized Volatility:        {latest_feat.realized_volatility:.4f}" if latest_feat.realized_volatility else "Vol: None")
    print(f"Relative Volume (RVOL):     {latest_feat.relative_volume:.2f}x" if latest_feat.relative_volume else "RVOL: None")
    print(f"Warmup Complete:            {latest_feat.is_warmed_up}")

    # 3. Formulate canonical ModelPrediction contract
    prediction = ModelPrediction(
        model_id="GBDT_OHLCV_BASELINE",
        model_version="1.0.0",
        prediction_timestamp=latest_feat.timestamp,
        input_cutoff_timestamp=latest_feat.input_cutoff_timestamp,
        instrument_id=ticker,
        prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
        horizon_bars=5,
        prediction_value=0.0125,  # +1.25% expected forward return
        confidence_or_uncertainty=0.85,
        availability_status=PredictionAvailability.READY,
        metadata={"num_features": len(latest_feat.to_dict()), "pipeline_version": config.config_version},
    )

    print("\n--- MODEL PREDICTION CONTRACT (INTERFACE CONTRACT B) ---")
    print(prediction.to_json(indent=2))

    # 4. Verify Schema Parity
    schema_errors = validate_model_prediction_schema_parity(prediction.to_dict())
    if not schema_errors:
        print("\n[+] SCHEMA PARITY CHECK: PASSED (100% compliant with schemas/model_prediction.schema.json)")
    else:
        print(f"\n[-] SCHEMA PARITY CHECK FAILED: {schema_errors}")

    print("=" * 80)


if __name__ == "__main__":
    run_demo()


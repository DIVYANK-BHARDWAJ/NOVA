"""NOVA Research — Empirical Regime Validation on Genuine MANTHAN Competition History.

Performs rigorous empirical descriptive evaluation of TransparentRegimeDetector:
1. Per-instrument classification distribution, warmup abstentions, and streak analysis.
2. Transition matrices across market regimes.
3. Mathematical and temporal integrity verification.
4. Threshold sensitivity sweep across KER trend, ranging, and high-volatility parameters.
5. Probability calibration assessment.
6. Emits machine-readable telemetry report to reports/regime_empirical_validation_report.json.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.contracts import DataQualityStatus, MarketBar, PredictionAvailability
from nova.regime.config import MarketRegimeConfig
from nova.regime.transparent_detector import RegimeLabel, TransparentRegimeDetector


def load_genuine_history(dir_path: Path) -> dict[str, list[MarketBar]]:
    """Load and parse genuine archived OHLCV bars into chronological MarketBar lists."""
    universe: dict[str, list[MarketBar]] = {}
    json_files = sorted(list(dir_path.glob("*_history_*.json")))

    for fpath in json_files:
        if "feasibility" in fpath.name or "universe" in fpath.name:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        ticker = data.get("ticker")
        prices = data.get("prices", [])
        if not ticker or not prices:
            continue

        bars: list[MarketBar] = []
        for p in prices:
            dt_str = p.get("date") or p.get("timestamp")
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
            bars.append(
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

        # Sort chronologically
        bars.sort(key=lambda b: b.source_timestamp)
        universe[ticker] = bars

    return universe


def evaluate_regime_on_series(
    bars: list[MarketBar],
    config: MarketRegimeConfig,
) -> dict[str, Any]:
    """Run point-in-time incremental regime evaluation over a candle series."""
    detector = TransparentRegimeDetector(config)

    emitted_labels: list[str] = []
    availabilities: list[str] = []
    kers: list[float] = []
    sigmas: list[float] = []
    streaks: dict[str, int] = defaultdict(int)
    max_streaks: dict[str, int] = defaultdict(int)
    current_regime: str | None = None
    current_streak = 0
    transitions: Counter[tuple[str, str]] = Counter()

    for i in range(1, len(bars) + 1):
        window_bars = bars[:i]
        pred = detector.detect(window_bars)

        label = pred.metadata.get("dominant_regime", RegimeLabel.UNKNOWN.value)
        emitted_labels.append(label)
        availabilities.append(pred.availability_status.value)

        if pred.availability_status == PredictionAvailability.READY:
            ker = float(pred.metadata.get("efficiency_ratio", 0.0))
            sigma = float(pred.metadata.get("ewma_volatility", 0.0))
            kers.append(ker)
            sigmas.append(sigma)

            # Streak tracking
            if label == current_regime:
                current_streak += 1
            else:
                if current_regime:
                    transitions[(current_regime, label)] += 1
                current_regime = label
                current_streak = 1
            max_streaks[label] = max(max_streaks[label], current_streak)

    counts = Counter(emitted_labels)
    total = len(emitted_labels)

    # Numerical validity
    ker_min = min(kers) if kers else 0.0
    ker_max = max(kers) if kers else 0.0
    ker_valid = all(0.0 <= k <= 1.0 for k in kers) if kers else True

    sigma_min = min(sigmas) if sigmas else 0.0
    sigma_max = max(sigmas) if sigmas else 0.0
    sigma_valid = all(s >= config.min_volatility_floor for s in sigmas) if sigmas else True

    return {
        "total_bars": total,
        "warmup_abstentions": counts.get(RegimeLabel.UNKNOWN.value, 0),
        "ready_predictions": total - counts.get(RegimeLabel.UNKNOWN.value, 0),
        "label_counts": {k: counts.get(k, 0) for k in [
            RegimeLabel.TRENDING_BULLISH.value,
            RegimeLabel.TRENDING_BEARISH.value,
            RegimeLabel.RANGING.value,
            RegimeLabel.HIGH_VOLATILITY.value,
            RegimeLabel.UNKNOWN.value,
        ]},
        "label_percentages": {k: round(counts.get(k, 0) / total * 100.0, 2) for k in counts},
        "max_streaks": dict(max_streaks),
        "total_transitions": sum(transitions.values()),
        "transitions_matrix": {f"{k[0]}->{k[1]}": v for k, v in transitions.items()},
        "numerical_integrity": {
            "ker_min": round(ker_min, 4),
            "ker_max": round(ker_max, 4),
            "ker_valid_range": ker_valid,
            "sigma_min": round(sigma_min, 6),
            "sigma_max": round(sigma_max, 6),
            "sigma_valid_range": sigma_valid,
        },
    }


def run_threshold_sensitivity(
    bars: list[MarketBar],
) -> list[dict[str, Any]]:
    """Perform bounded sensitivity sweep across configuration alternatives."""
    variations = [
        # (ker_trend, ker_ranging, high_vol)
        (0.50, 0.25, 0.025),  # Baseline
        (0.40, 0.20, 0.025),  # More aggressive trend trigger
        (0.60, 0.30, 0.025),  # Conservative trend trigger
        (0.50, 0.25, 0.020),  # Stricter volatility threshold (more HIGH_VOLATILITY)
        (0.50, 0.25, 0.030),  # Looser volatility threshold
    ]

    results = []
    for trend_th, range_th, vol_th in variations:
        cfg = MarketRegimeConfig(
            ker_trend_threshold=trend_th,
            ker_ranging_threshold=range_th,
            high_volatility_threshold=vol_th,
        )
        res = evaluate_regime_on_series(bars, cfg)
        results.append({
            "ker_trend_threshold": trend_th,
            "ker_ranging_threshold": range_th,
            "high_volatility_threshold": vol_th,
            "ready_predictions": res["ready_predictions"],
            "label_percentages": res["label_percentages"],
            "total_transitions": res["total_transitions"],
        })
    return results


def main() -> None:
    print("=" * 80)
    print("NOVA — EMPIRICAL REGIME VALIDATION (GENUINE MANTHAN DATA)")
    print("=" * 80)

    # 1. Load full historical dataset (18,494 bars)
    full_dir = REPO_ROOT / "data" / "raw" / "manthan_history_full"
    init_dir = REPO_ROOT / "data" / "raw" / "manthan_history"

    target_dir = full_dir if full_dir.exists() and list(full_dir.glob("*.json")) else init_dir
    print(f"[*] Ingesting genuine historical data from: {target_dir}")

    universe = load_genuine_history(target_dir)
    print(f"[+] Loaded {len(universe)} instruments. Total bars: {sum(len(b) for b in universe.values())}")

    base_config = MarketRegimeConfig()
    per_instrument_results: dict[str, Any] = {}
    aggregate_counts: Counter[str] = Counter()
    total_raw = 0

    for ticker, bars in sorted(universe.items()):
        print(f"[*] Evaluating [{ticker}]: {len(bars)} bars ({bars[0].source_timestamp.date()} to {bars[-1].source_timestamp.date()})...")
        res = evaluate_regime_on_series(bars, base_config)
        per_instrument_results[ticker] = res
        total_raw += len(bars)
        for label, count in res["label_counts"].items():
            aggregate_counts[label] += count

    print("\n" + "=" * 80)
    print("AGGREGATE REGIME CLASSIFICATION SUMMARY:")
    print("=" * 80)
    for label, count in sorted(aggregate_counts.items(), key=lambda x: -x[1]):
        pct = round(count / total_raw * 100.0, 2)
        print(f"  {label:<22} : {count:>6} bars ({pct:>5.1f}%)")

    # Sensitivity analysis on sample ticker (AERX)
    print("\n[*] Performing Threshold Sensitivity Analysis on AERX...")
    aerx_bars = universe.get("AERX", [])
    sensitivity_results = run_threshold_sensitivity(aerx_bars) if aerx_bars else []

    # Final Report Structure
    report_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_instruments": len(universe),
        "total_historical_bars": total_raw,
        "dataset_source": str(target_dir),
        "baseline_config": {
            "ker_window": base_config.ker_window,
            "ker_trend_threshold": base_config.ker_trend_threshold,
            "ker_ranging_threshold": base_config.ker_ranging_threshold,
            "volatility_window": base_config.volatility_window,
            "high_volatility_threshold": base_config.high_volatility_threshold,
            "min_warmup_bars": base_config.min_warmup_bars,
            "ewma_lambda": base_config.ewma_lambda,
        },
        "aggregate_counts": dict(aggregate_counts),
        "aggregate_percentages": {k: round(v / total_raw * 100.0, 2) for k, v in aggregate_counts.items()},
        "per_instrument": per_instrument_results,
        "sensitivity_analysis_aerx": sensitivity_results,
        "calibration_analysis": {
            "distribution_type": "HEURISTIC_DISTANCE_NORMALIZED",
            "is_statistically_calibrated": False,
            "calibration_assessment": "The prediction distribution is derived from normalized distance-to-threshold boundaries and EWMA volatility scale. Because MANTHAN provides zero independent ground-truth regime annotations, these values represent deterministic heuristic certainty scores rather than empirical Bayesian probabilities.",
            "recommendation": "Preserve Interface Contract B (ModelPrediction) distribution format summing to 1.0, while explicitly reporting probability calibration status as UNCALIBRATED_HEURISTIC in research manifests.",
        },
    }

    out_file = REPO_ROOT / "reports" / "regime_empirical_validation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[+] Empirical validation report exported to: {out_file}")


if __name__ == "__main__":
    main()


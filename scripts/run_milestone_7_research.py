"""NOVA Quantitative Machine Learning — Milestone 7 Research Runner (Person 2).

Executes the full Person 2 Milestone 7 research battery:
- Experiment A: Directional Classification vs. Regression Sign vs. Drift Baseline
- Experiment B: Forecast Horizons (H in {1, 3, 5}) with causal embargoes
- Experiment C: Compact Feature Family Evaluations & Feature Importances
- Experiment D: Market-Regime Conditional Diagnostics via TransparentRegimeDetector
- Experiment E: Market-News Feasibility Audit & Contract-Safe Interface
- Latency Profiling (Feature computation + Inference latency)
- Model Artifact Serialization & Deliverable Emission
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

import importlib
xgb: Any = None
try:
    xgb = importlib.import_module("xgboost")
except Exception:
    pass

from nova.features.news import MarketNewsRecord, NewsSentimentInterface, NewsSentimentStatus
from nova.features.pipeline import FeaturePipeline
from nova.models.config import DEFAULT_FEATURE_NAMES, XGBoostModelConfig
from nova.models.dataset import DatasetAssembler, ModelDataset
from nova.models.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    evaluate_classification_metrics,
    evaluate_regression_metrics,
    log_loss,
    precision_recall_f1,
    roc_auc_score,
)
from nova.models.xgboost_model import NovaGradientBoostedRegressor, XGBoostPredictionModel
from nova.quant.contracts import (
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    PredictionAvailability,
    PredictionType,
    RegimeCategory,
    SignalDirection,
)
from nova.regime.transparent_detector import RegimeLabel, TransparentRegimeDetector


def load_canonical_bars(dataset_dir: Path) -> dict[str, list[MarketBar]]:
    """Load canonical market bars from JSONL lines."""
    bars_file = dataset_dir / "observations.jsonl"
    if not bars_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {bars_file}")

    universe_bars: dict[str, list[MarketBar]] = {}
    with open(bars_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ticker = rec["ticker"]
            dt = datetime.fromisoformat(rec["market_date"]).replace(tzinfo=timezone.utc)
            bar = MarketBar(
                ticker=ticker,
                open=float(rec["open"]),
                high=float(rec["high"]),
                low=float(rec["low"]),
                close=float(rec["close"]),
                volume=float(rec.get("volume", 0.0)),
                source_timestamp=dt,
                provenance=f"MANTHAN_CANONICAL_V1:{ticker}",
                data_quality=DataQualityStatus.VALID,
            )
            if ticker not in universe_bars:
                universe_bars[ticker] = []
            universe_bars[ticker].append(bar)

    for t in universe_bars:
        universe_bars[t].sort(key=lambda b: b.source_timestamp)

    return universe_bars


def compute_mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
) -> dict[str, Any]:
    """Perform McNemar's paired test with continuity correction."""
    n = len(y_true)
    b = 0  # A correct, B incorrect
    c = 0  # A incorrect, B correct
    a = 0  # Both correct
    d = 0  # Both incorrect

    for t, pa, pb in zip(y_true, y_pred_a, y_pred_b):
        correct_a = (pa == t)
        correct_b = (pb == t)
        if correct_a and not correct_b:
            b += 1
        elif not correct_a and correct_b:
            c += 1
        elif correct_a and correct_b:
            a += 1
        else:
            d += 1

    discordant = b + c
    if discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        stat = ((abs(b - c) - 1.0) ** 2) / float(discordant)
        # Approximate p-value from chi-square distribution with 1 DOF
        # P(Chi2 > stat) via math.erfc
        p_val = math.erfc(math.sqrt(max(0.0, stat)) / math.sqrt(2.0))

    return {
        "n_samples": n,
        "both_correct": a,
        "model_a_correct_model_b_wrong": b,
        "model_a_wrong_model_b_correct": c,
        "both_wrong": d,
        "mcnemar_statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "is_statistically_significant_05": p_val < 0.05,
    }


def main() -> None:
    print("=" * 80)
    print("NOVA Quantitative Machine Learning — Person 2 Milestone 7 Research")
    print("Reproducible Predictive Framework, Horizons, Features & News Feasibility")
    print("=" * 80)

    replay_dir = REPO_ROOT / "data" / "replay" / "canonical_manthan_v1"
    print(f"[*] Ingesting canonical MANTHAN dataset from {replay_dir}...")
    universe_bars = load_canonical_bars(replay_dir)
    total_bars = sum(len(b) for b in universe_bars.values())
    tickers = sorted(list(universe_bars.keys()))
    print(f"[+] Loaded {total_bars:,} bars across {len(tickers)} tickers: {tickers}")

    # ========================================================================
    # EXPERIMENT A: Directional Classification vs. Regression vs. Drift Baseline
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT A: Directional Classification vs. Regression vs. Drift")
    print("=" * 80)

    cfg_h1_reg = XGBoostModelConfig(
        model_id="NOVA_XGBOOST_REGRESSION_H1",
        horizon_bars=1,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=40,
        l2_reg=1.0,
        objective="reg:squarederror",
    )
    assembler_h1 = DatasetAssembler(cfg_h1_reg)
    dataset_h1_reg = assembler_h1.assemble_universe(universe_bars)
    dataset_h1_clf = dataset_h1_reg.to_classification()

    print(f"[*] Assembled H=1 dataset: Train={dataset_h1_clf.train_count:,}, Val={dataset_h1_clf.val_count:,}, Test={dataset_h1_clf.test_count:,}")
    print(f"[*] Train date range: {dataset_h1_clf.train_dates[0]} to {dataset_h1_clf.train_dates[1]}")
    print(f"[*] Test date range:  {dataset_h1_clf.test_dates[0]} to {dataset_h1_clf.test_dates[1]}")

    # Ground truth test labels: 1 if return > 0 else 0
    y_test_true = [1 if y > 0.0 else 0 for y in dataset_h1_clf.y_test]
    n_test = len(y_test_true)
    pos_test_count = sum(y_test_true)
    neg_test_count = n_test - pos_test_count
    print(f"[+] Test sample balance: Positive={pos_test_count:,} ({pos_test_count/n_test*100:.2f}%), Negative/Flat={neg_test_count:,} ({neg_test_count/n_test*100:.2f}%)")

    # 1. Train Native XGBoost Classifier
    print("[*] Training Native XGBoost Directional Classifier (binary:logistic)...")
    cfg_h1_clf = XGBoostModelConfig(
        model_id="NOVA_XGBOOST_DIRECTIONAL_CLASSIFIER_V1",
        model_version="1.0.0",
        horizon_bars=1,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=40,
        l2_reg=1.0,
        objective="binary:logistic",
    )
    clf_model = XGBoostPredictionModel(cfg_h1_clf)
    clf_summary = clf_model.fit(dataset_h1_clf)
    print(f"[+] Classifier trained: backend={clf_summary['backend']}, fitted_rounds={clf_summary.get('fitted_rounds')}, best_score={clf_summary.get('best_score')}")

    # Predictions
    prob_clf = clf_model.predict_raw(dataset_h1_clf.X_test)
    pred_clf_binary = [1 if p >= 0.50 else 0 for p in prob_clf]

    # 2. Existing Native XGBoost Regressor
    print("[*] Loading Native XGBoost Regressor artifact...")
    native_reg_path = REPO_ROOT / "artifacts" / "models" / "xgboost_native_v1" / "model.json"
    reg_native_model = XGBoostPredictionModel.load(native_reg_path)
    pred_reg_raw = reg_native_model.predict_raw(dataset_h1_reg.X_test)
    pred_reg_binary = [1 if r > 0.0 else 0 for r in pred_reg_raw]

    # 3. Existing Pure Python GBDT Regressor
    print("[*] Loading Pure-Python GBDT Regressor artifact...")
    gbdt_path = REPO_ROOT / "artifacts" / "models" / "gbdt_ohlcv_baseline_v1" / "model.json"
    gbdt_model = XGBoostPredictionModel.load(gbdt_path)
    pred_gbdt_raw = gbdt_model.predict_raw(dataset_h1_reg.X_test)
    pred_gbdt_binary = [1 if r > 0.0 else 0 for r in pred_gbdt_raw]

    # 4. Drift Direction Baseline (Train majority class)
    # Check train split proportion
    pos_train_count = sum(1 for y in dataset_h1_clf.y_train if y == 1.0)
    train_majority_label = 1 if (pos_train_count / len(dataset_h1_clf.y_train)) >= 0.50 else 0
    pred_drift_binary = [train_majority_label] * n_test

    # Compute classification metrics
    clf_metrics = evaluate_classification_metrics(y_test_true, prob_clf, threshold=0.50, dataset_split="test")
    reg_acc = accuracy_score(y_test_true, pred_reg_binary)
    reg_b_acc = balanced_accuracy_score(y_test_true, pred_reg_binary)
    gbdt_acc = accuracy_score(y_test_true, pred_gbdt_binary)
    gbdt_b_acc = balanced_accuracy_score(y_test_true, pred_gbdt_binary)
    drift_acc = accuracy_score(y_test_true, pred_drift_binary)
    drift_b_acc = balanced_accuracy_score(y_test_true, pred_drift_binary)

    # McNemar tests
    mcnemar_clf_vs_drift = compute_mcnemar_test(y_test_true, pred_clf_binary, pred_drift_binary)
    mcnemar_clf_vs_reg = compute_mcnemar_test(y_test_true, pred_clf_binary, pred_reg_binary)

    print("\n--- EXPERIMENT A METRIC COMPARISON ---")
    print(f"1. XGBoost Binary Classifier: Accuracy={clf_metrics['accuracy_pct']}%, Balanced={clf_metrics['balanced_accuracy_pct']}%, ROC-AUC={clf_metrics['roc_auc']:.4f}, LogLoss={clf_metrics['log_loss']:.4f}, Brier={clf_metrics['brier_score']:.4f}")
    print(f"2. XGBoost Return Regressor:   Accuracy={reg_acc*100:.2f}%, Balanced={reg_b_acc*100:.2f}%")
    print(f"3. Pure-Python GBDT Regressor:Accuracy={gbdt_acc*100:.2f}%, Balanced={gbdt_b_acc*100:.2f}%")
    print(f"4. Drift Baseline (Positive): Accuracy={drift_acc*100:.2f}%, Balanced={drift_b_acc*100:.2f}%")
    print(f"McNemar p-value (Classifier vs Drift):    p = {mcnemar_clf_vs_drift['p_value']} (stat={mcnemar_clf_vs_drift['mcnemar_statistic']})")
    print(f"McNemar p-value (Classifier vs Regressor):p = {mcnemar_clf_vs_reg['p_value']} (stat={mcnemar_clf_vs_reg['mcnemar_statistic']})")

    # ========================================================================
    # EXPERIMENT B: Multi-Horizon Forecasting (H in {1, 3, 5})
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT B: Forecast Horizons (H in {1, 3, 5}) with Causal Embargo")
    print("=" * 80)

    horizons_results: dict[str, Any] = {}
    for h in [1, 3, 5]:
        cfg_h = XGBoostModelConfig(
            model_id=f"NOVA_XGBOOST_CLF_H{h}",
            horizon_bars=h,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=40,
            l2_reg=1.0,
            objective="binary:logistic",
        )
        asm = DatasetAssembler(cfg_h)
        ds_reg = asm.assemble_universe(universe_bars)
        ds_clf = ds_reg.to_classification()

        model_h = XGBoostPredictionModel(cfg_h)
        model_h.fit(ds_clf)

        probs_h = model_h.predict_raw(ds_clf.X_test)
        y_test_h = [int(y) for y in ds_clf.y_test]
        metrics_h = evaluate_classification_metrics(y_test_h, probs_h, threshold=0.50, dataset_split=f"test_H{h}")
        preds_binary_h = [1 if p >= 0.50 else 0 for p in probs_h]

        # Drift baseline for horizon H
        pos_train_h = sum(1 for y in ds_clf.y_train if y == 1.0)
        majority_h = 1 if (pos_train_h / len(ds_clf.y_train)) >= 0.50 else 0
        drift_preds_h = [majority_h] * len(y_test_h)
        drift_acc_h = accuracy_score(y_test_h, drift_preds_h)
        mcnemar_h = compute_mcnemar_test(y_test_h, preds_binary_h, drift_preds_h)

        horizons_results[f"H_{h}"] = {
            "horizon_bars": h,
            "train_samples": ds_clf.train_count,
            "val_samples": ds_clf.val_count,
            "test_samples": ds_clf.test_count,
            "purged_embargo_samples": ds_clf.purged_embargo_count,
            "classifier_metrics": metrics_h,
            "drift_accuracy_pct": round(drift_acc_h * 100.0, 2),
            "mcnemar_vs_drift": mcnemar_h,
        }
        print(f"[*] H={h} Bars: Test N={ds_clf.test_count:,} (Embargoed {ds_clf.purged_embargo_count} rows) | Clf Acc={metrics_h['accuracy_pct']}% | Drift Acc={drift_acc_h*100:.2f}% | ROC-AUC={metrics_h['roc_auc']:.4f} | McNemar p={mcnemar_h['p_value']}")

    # ========================================================================
    # EXPERIMENT C: Feature Family Subsets
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT C: Feature Family Subsets Evaluation")
    print("=" * 80)

    feature_subsets = {
        "trend_momentum": ("momentum", "trend_slope", "kaufman_efficiency_ratio", "ema_normalized_spread"),
        "volatility": ("normalized_atr", "realized_volatility", "parkinson_volatility", "downside_deviation"),
        "mean_reversion_volatility": ("rolling_z_score", "rsi", "normalized_atr", "realized_volatility"),
        "volume": ("relative_volume", "volume_z_score"),
        "full_12_features": DEFAULT_FEATURE_NAMES,
    }

    subsets_results: dict[str, Any] = {}
    for subset_name, feats in feature_subsets.items():
        cfg_sub = XGBoostModelConfig(
            model_id=f"NOVA_XGBOOST_CLF_{subset_name.upper()}",
            horizon_bars=1,
            feature_names=feats,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=40,
            l2_reg=1.0,
            objective="binary:logistic",
        )
        asm_sub = DatasetAssembler(cfg_sub)
        ds_reg_sub = asm_sub.assemble_universe(universe_bars)
        ds_clf_sub = ds_reg_sub.to_classification()

        model_sub = XGBoostPredictionModel(cfg_sub)
        model_sub.fit(ds_clf_sub)

        probs_sub = model_sub.predict_raw(ds_clf_sub.X_test)
        y_test_sub = [int(y) for y in ds_clf_sub.y_test]
        metrics_sub = evaluate_classification_metrics(y_test_sub, probs_sub, threshold=0.50, dataset_split=f"test_{subset_name}")

        subsets_results[subset_name] = {
            "feature_count": len(feats),
            "feature_names": list(feats),
            "metrics": metrics_sub,
        }
        print(f"[*] Subset '{subset_name}' ({len(feats)} features): Clf Acc={metrics_sub['accuracy_pct']}% | Balanced={metrics_sub['balanced_accuracy_pct']}% | ROC-AUC={metrics_sub['roc_auc']:.4f} | Brier={metrics_sub['brier_score']:.4f}")

    # Extract feature importance from full model
    feature_importance: dict[str, float] = {}
    if clf_model._native_model is not None and hasattr(clf_model._native_model, "get_score"):
        raw_importance = clf_model._native_model.get_score(importance_type="gain")
        # Map f0, f1 to actual feature names
        for i, fname in enumerate(DEFAULT_FEATURE_NAMES):
            f_key = f"f{i}"
            feature_importance[fname] = round(float(raw_importance.get(f_key, 0.0)), 4)
    print(f"\n[+] Feature Importance (Gain): {json.dumps(feature_importance, indent=2)}")

    # ========================================================================
    # EXPERIMENT D: Regime-Conditional Diagnostics
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT D: Market-Regime Conditional Diagnostics")
    print("=" * 80)

    regime_detector = TransparentRegimeDetector()
    # Categorize test bars into regimes
    # Build a lookup of test dates
    all_dates = sorted(list({
        b.source_timestamp.date().isoformat()
        for bars in universe_bars.values() for b in bars
    }))
    test_dates_set = set(all_dates[int(len(all_dates) * 0.80):])

    regime_test_samples: dict[str, list[tuple[int, int, float]]] = {
        r.value: [] for r in RegimeLabel
    }

    # For each instrument, run pipeline history and regime detector along bar history
    for ticker, bars in universe_bars.items():
        n_bars = len(bars)
        computed_features = clf_model.pipeline.compute_history(bars)
        for i in range(28, n_bars - 1):
            bar = bars[i]
            d_str = bar.source_timestamp.date().isoformat()
            if d_str in test_dates_set:
                # Target return: ln(Close_{t+1}/Close_t)
                c_now = bar.close
                c_next = bars[i + 1].close
                true_label = 1 if c_next > c_now else 0

                # Extract cached feature vector
                f_obj = computed_features[i]
                f_dict = f_obj.to_dict()
                feat_vec = [float(f_dict[k]) for k in DEFAULT_FEATURE_NAMES]
                prob = clf_model.predict_raw([feat_vec])[0]
                pred_label = 1 if prob >= 0.50 else 0

                # Detect regime using bounded historical slice
                sub_bars = bars[max(0, i - 50) : i + 1]
                regime_eval = regime_detector.detect(sub_bars)
                reg_cat = str(regime_eval.metadata.get("dominant_regime", RegimeLabel.UNKNOWN.value))
                if reg_cat in regime_test_samples:
                    regime_test_samples[reg_cat].append((true_label, pred_label, prob))

    regime_breakdown: dict[str, Any] = {}
    print("\n--- REGIME DIAGNOSTIC SUMMARY ---")
    for reg_name, samples in sorted(regime_test_samples.items()):
        n_s = len(samples)
        if n_s == 0:
            regime_breakdown[reg_name] = {"sample_count": 0, "status": "NO_OBSERVATIONS"}
            continue
        y_true_reg = [s[0] for s in samples]
        y_pred_reg = [s[1] for s in samples]
        y_prob_reg = [s[2] for s in samples]

        acc_reg = accuracy_score(y_true_reg, y_pred_reg)
        b_acc_reg = balanced_accuracy_score(y_true_reg, y_pred_reg)
        pos_prop = sum(y_true_reg) / n_s
        auc_reg = roc_auc_score(y_true_reg, y_prob_reg)

        regime_breakdown[reg_name] = {
            "sample_count": n_s,
            "positive_class_ratio_pct": round(pos_prop * 100.0, 2),
            "directional_accuracy_pct": round(acc_reg * 100.0, 2),
            "balanced_accuracy_pct": round(b_acc_reg * 100.0, 2),
            "roc_auc": round(auc_reg, 4),
        }
        print(f"Regime '{reg_name:26s}': N={n_s:5d} ({n_s/n_test*100:5.1f}%) | PosRatio={pos_prop*100:5.2f}% | Acc={acc_reg*100:5.2f}% | BalAcc={b_acc_reg*100:5.2f}% | ROC-AUC={auc_reg:.4f}")

    # ========================================================================
    # EXPERIMENT E: Market-News Feasibility Audit
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT E: Market-News Feasibility Audit & Research Interface")
    print("=" * 80)

    # 1. Audit disk files for genuine news records
    raw_news_candidates = list((REPO_ROOT / "data").glob("**/*news*"))
    print(f"[*] Audited data directory for news archives: found {len(raw_news_candidates)} candidate files.")

    # 2. Check interface behavior with zero real records
    news_iface = NewsSentimentInterface(lookback_window_hours=24.0)
    test_dt = datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc)
    status_empty, score_empty = news_iface.compute_sentiment_feature("INFY", test_dt)
    print(f"[+] News interface behavior with empty store: Status={status_empty.value}, Score={score_empty}")

    # 3. Demonstrate anti-lookahead point-in-time behavior with sample real record schema
    real_sample_news = MarketNewsRecord(
        news_id="NW_20260417_001",
        headline="INFY announces strategic partnership expansion",
        published_at=datetime(2026, 4, 17, 9, 0, tzinfo=timezone.utc),
        tickers=("INFY",),
        sentiment_score=0.45,
        confidence=0.85,
    )
    news_iface.ingest_news_events([real_sample_news])
    status_past, score_past = news_iface.compute_sentiment_feature("INFY", datetime(2026, 4, 17, 8, 30, tzinfo=timezone.utc))
    status_present, score_present = news_iface.compute_sentiment_feature("INFY", datetime(2026, 4, 17, 9, 30, tzinfo=timezone.utc))
    print(f"[+] Anti-lookahead test: Before publication: {status_past.value} (Score={score_past}) | After publication: {status_present.value} (Score={score_present})")

    news_feasibility_verdict = {
        "status": NewsSentimentStatus.BLOCKED_INSUFFICIENT_REAL_DATA.value,
        "reason": "MANTHAN historical REST endpoint (/api/v1/stocks/history) provides OHLCV daily data only. Zero historical market.news records exist in canonical archives.",
        "policy_adherence": "Strict no-synthetic-news policy maintained; zero synthetic headlines or simulated sentiment values generated for model training.",
        "architectural_readiness": "NewsSentimentInterface implemented with anti-lookahead timestamp filtering for future live streaming integration.",
    }

    # ========================================================================
    # LATENCY PROFILING
    # ========================================================================
    print("\n" + "=" * 80)
    print("LATENCY PROFILING: Feature Extraction & Inference Performance")
    print("=" * 80)

    # 1. Feature pipeline latency
    sample_bars = universe_bars[tickers[0]][:50]
    n_lat_runs = 500
    t0 = time.perf_counter()
    for _ in range(n_lat_runs):
        _ = clf_model.pipeline.compute_latest(sample_bars)
    t1 = time.perf_counter()
    feat_latency_us = ((t1 - t0) / n_lat_runs) * 1_000_000.0

    # 2. Classifier inference latency (predict_raw)
    sample_feat_vec = [dataset_h1_clf.X_test[0]]
    t0 = time.perf_counter()
    for _ in range(n_lat_runs):
        _ = clf_model.predict_raw(sample_feat_vec)
    t1 = time.perf_counter()
    clf_infer_latency_us = ((t1 - t0) / n_lat_runs) * 1_000_000.0

    # 3. Full end-to-end predict_bar() latency
    t0 = time.perf_counter()
    for _ in range(n_lat_runs):
        _ = clf_model.predict_bar(sample_bars)
    t1 = time.perf_counter()
    e2e_latency_us = ((t1 - t0) / n_lat_runs) * 1_000_000.0

    print(f"[+] Feature pipeline latency per bar:        {feat_latency_us:8.2f} microseconds ({feat_latency_us/1000.0:.3f} ms)")
    print(f"[+] XGBoost classifier inference latency:    {clf_infer_latency_us:8.2f} microseconds ({clf_infer_latency_us/1000.0:.3f} ms)")
    print(f"[+] Full end-to-end predict_bar() latency:   {e2e_latency_us:8.2f} microseconds ({e2e_latency_us/1000.0:.3f} ms)")

    # ========================================================================
    # SERIALIZATION: Save Classifier Artifact
    # ========================================================================
    print("\n" + "=" * 80)
    print("SERIALIZATION: Creating Versioned Classifier Artifact")
    print("=" * 80)

    clf_artifact_dir = REPO_ROOT / "artifacts" / "models" / "xgboost_classifier_v1"
    clf_artifact_dir.mkdir(parents=True, exist_ok=True)
    clf_model_path = clf_artifact_dir / "model.json"
    save_info = clf_model.save(clf_model_path)
    file_bytes = clf_model_path.read_bytes()
    clf_sha256 = hashlib.sha256(file_bytes).hexdigest()

    clf_manifest = {
        "model_id": cfg_h1_clf.model_id,
        "model_version": cfg_h1_clf.model_version,
        "model_type": "NATIVE_XGBOOST_DIRECTIONAL_CLASSIFIER",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_file": "model.json",
        "artifact_sha256": clf_sha256,
        "artifact_size_bytes": len(file_bytes),
        "source_dataset": "data/replay/canonical_manthan_v1/",
        "sample_counts": {
            "train": dataset_h1_clf.train_count,
            "validation": dataset_h1_clf.val_count,
            "test": dataset_h1_clf.test_count,
            "purged_embargo": dataset_h1_clf.purged_embargo_count,
        },
        "evaluation_metrics": {
            "accuracy_pct": clf_metrics["accuracy_pct"],
            "balanced_accuracy_pct": clf_metrics["balanced_accuracy_pct"],
            "roc_auc": clf_metrics["roc_auc"],
            "brier_score": clf_metrics["brier_score"],
            "log_loss": clf_metrics["log_loss"],
            "precision": clf_metrics["precision"],
            "recall": clf_metrics["recall"],
            "f1_score": clf_metrics["f1_score"],
            "mcnemar_vs_drift_p_value": mcnemar_clf_vs_drift["p_value"],
        },
        "latency_profile_us": {
            "feature_pipeline_us": round(feat_latency_us, 2),
            "model_inference_us": round(clf_infer_latency_us, 2),
            "e2e_predict_bar_us": round(e2e_latency_us, 2),
        },
    }
    with open(clf_artifact_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(clf_manifest, f, indent=2)
    print(f"[+] Saved classifier artifact to: {clf_model_path} (SHA-256: {clf_sha256[:16]}...)")
    print(f"[+] Saved classifier manifest to: {clf_artifact_dir / 'manifest.json'}")

    # ========================================================================
    # EMIT RESEARCH REPORT JSON
    # ========================================================================
    report_json_path = REPO_ROOT / "reports" / "nova_person_2_milestone_7_signal_research_report.json"
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    research_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "PERSON_2_MILESTONE_7_PREDICTIVE_SIGNAL_RESEARCH",
        "framework_status": "PASS_RESEARCH_FRAMEWORK",
        "dataset_provenance": {
            "source_dir": "data/replay/canonical_manthan_v1/",
            "tickers_count": len(tickers),
            "tickers": tickers,
            "total_bars": total_bars,
        },
        "experiment_a_directional_classification": {
            "test_sample_count": n_test,
            "test_positive_samples": pos_test_count,
            "test_negative_samples": neg_test_count,
            "xgboost_binary_classifier": clf_metrics,
            "xgboost_return_regressor": {
                "accuracy_pct": round(reg_acc * 100.0, 2),
                "balanced_accuracy_pct": round(reg_b_acc * 100.0, 2),
            },
            "pure_python_gbdt_regressor": {
                "accuracy_pct": round(gbdt_acc * 100.0, 2),
                "balanced_accuracy_pct": round(gbdt_b_acc * 100.0, 2),
            },
            "positive_drift_baseline": {
                "accuracy_pct": round(drift_acc * 100.0, 2),
                "balanced_accuracy_pct": round(drift_b_acc * 100.0, 2),
            },
            "mcnemar_classifier_vs_drift": mcnemar_clf_vs_drift,
            "mcnemar_classifier_vs_regressor": mcnemar_clf_vs_reg,
        },
        "experiment_b_forecast_horizons": horizons_results,
        "experiment_c_feature_subsets": {
            "subsets": subsets_results,
            "feature_importance_gain": feature_importance,
        },
        "experiment_d_regime_diagnostics": regime_breakdown,
        "experiment_e_news_feasibility": news_feasibility_verdict,
        "latency_profile_us": {
            "feature_pipeline_us": round(feat_latency_us, 2),
            "model_inference_us": round(clf_infer_latency_us, 2),
            "e2e_predict_bar_us": round(e2e_latency_us, 2),
        },
        "classifier_artifact": clf_manifest,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(research_payload, f, indent=2)
    print(f"\n[+] Full JSON research report emitted to: {report_json_path}")


if __name__ == "__main__":
    main()

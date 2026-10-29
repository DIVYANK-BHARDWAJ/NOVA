"""Unit tests for Person 2 Milestone 7 Predictive Signal Research & Classification Framework.

Validates:
- Classification metrics accuracy, confusion matrix, and ROC-AUC calculation
- ModelDataset binary direction conversion and split preservation
- XGBoost classifier contract-valid ModelPrediction emission
- NewsSentimentInterface point-in-time anti-lookahead behavior and empty data handling
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.features.news import MarketNewsRecord, NewsSentimentInterface, NewsSentimentStatus
from nova.models.config import XGBoostModelConfig
from nova.models.dataset import ModelDataset
from nova.models.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score,
    confusion_matrix,
    evaluate_classification_metrics,
    log_loss,
    precision_recall_f1,
    roc_auc_score,
)
from nova.models.xgboost_model import XGBoostPredictionModel
from nova.quant.contracts import (
    DataQualityStatus,
    MarketBar,
    PredictionAvailability,
    PredictionType,
)


class TestClassificationMetrics(unittest.TestCase):
    """Test suite for pure-Python classification metrics."""

    def test_accuracy_and_confusion_matrix(self) -> None:
        y_true = [1, 0, 1, 1, 0, 0]
        y_pred = [1, 0, 0, 1, 1, 0]

        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(acc, 4 / 6)

        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm["tp"], 2)  # indices 0, 3
        self.assertEqual(cm["tn"], 2)  # indices 1, 5
        self.assertEqual(cm["fp"], 1)  # index 4
        self.assertEqual(cm["fn"], 1)  # index 2

        b_acc = balanced_accuracy_score(y_true, y_pred)
        # Sensitivity = 2/3, Specificity = 2/3 -> Balanced = 2/3
        self.assertAlmostEqual(b_acc, 2 / 3)

    def test_roc_auc_score(self) -> None:
        # Perfect discrimination
        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.3, 0.1]
        self.assertAlmostEqual(roc_auc_score(y_true, y_prob), 1.0)

        # Inverted discrimination
        y_prob_inv = [0.1, 0.2, 0.7, 0.8]
        self.assertAlmostEqual(roc_auc_score(y_true, y_prob_inv), 0.0)

        # Single-class returns uninformative 0.50
        self.assertEqual(roc_auc_score([1, 1], [0.8, 0.9]), 0.50)

    def test_brier_and_log_loss(self) -> None:
        y_true = [1, 0]
        y_prob = [0.9, 0.1]
        # Brier: ((0.9-1)^2 + (0.1-0)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
        self.assertAlmostEqual(brier_score(y_true, y_prob), 0.01)

        # Log loss: (-ln(0.9) - ln(0.9)) / 2 = -ln(0.9) approx 0.10536
        self.assertAlmostEqual(log_loss(y_true, y_prob), -math.log(0.9), places=4)


class TestDatasetClassificationConversion(unittest.TestCase):
    """Test binary target conversion on ModelDataset."""

    def test_to_classification(self) -> None:
        ds = ModelDataset(
            feature_names=("feat1",),
            X_train=[[1.0], [2.0]],
            y_train=[0.05, -0.02],
            X_val=[[3.0]],
            y_val=[0.0],  # flat return -> 0.0
            X_test=[[4.0]],
            y_test=[0.01],
            train_dates=("2026-01-01", "2026-01-02"),
            val_dates=("2026-01-03", "2026-01-03"),
            test_dates=("2026-01-04", "2026-01-04"),
            total_raw_observations=4,
            total_labeled_observations=4,
            purged_embargo_count=0,
        )

        clf_ds = ds.to_classification()
        self.assertEqual(clf_ds.y_train, [1.0, 0.0])
        self.assertEqual(clf_ds.y_val, [0.0])
        self.assertEqual(clf_ds.y_test, [1.0])
        self.assertEqual(clf_ds.metadata["target_type"], "binary_direction")


class TestNewsSentimentInterface(unittest.TestCase):
    """Test anti-lookahead behavior of NewsSentimentInterface."""

    def test_empty_store_returns_blocked_status(self) -> None:
        iface = NewsSentimentInterface(lookback_window_hours=24.0)
        now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
        status, score = iface.compute_sentiment_feature("INFY", now)
        self.assertEqual(status, NewsSentimentStatus.BLOCKED_INSUFFICIENT_REAL_DATA)
        self.assertEqual(score, 0.0)

    def test_anti_lookahead_filtering(self) -> None:
        iface = NewsSentimentInterface(lookback_window_hours=24.0)
        pub_time = datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc)
        record = MarketNewsRecord(
            news_id="NW_001",
            headline="Quarterly Earnings Beat",
            published_at=pub_time,
            tickers=("INFY",),
            sentiment_score=0.75,
            confidence=0.90,
        )
        iface.ingest_news_events([record])

        # Prior to publication: must not see the event
        before_pub = datetime(2026, 4, 17, 9, 59, tzinfo=timezone.utc)
        status_before, score_before = iface.compute_sentiment_feature("INFY", before_pub)
        self.assertEqual(status_before, NewsSentimentStatus.NO_EVENTS_FOR_WINDOW)
        self.assertEqual(score_before, 0.0)

        # After publication: sees the event
        after_pub = datetime(2026, 4, 17, 10, 1, tzinfo=timezone.utc)
        status_after, score_after = iface.compute_sentiment_feature("INFY", after_pub)
        self.assertEqual(status_after, NewsSentimentStatus.READY)
        self.assertEqual(score_after, 0.75)


class TestXGBoostClassifierModel(unittest.TestCase):
    """Test native XGBoost directional classification predictions."""

    def test_classification_prediction_contract(self) -> None:
        cfg = XGBoostModelConfig(
            model_id="TEST_XGBOOST_CLASSIFIER",
            horizon_bars=1,
            max_depth=2,
            n_estimators=5,
            objective="binary:logistic",
        )
        model = XGBoostPredictionModel(cfg)

        # Minimal synthetic dataset with clear linear separation
        X_tr = [[-1.0] * 12, [-0.5] * 12, [0.5] * 12, [1.0] * 12]
        y_tr = [0.0, 0.0, 1.0, 1.0]
        X_v = [[-0.8] * 12, [0.8] * 12]
        y_v = [0.0, 1.0]

        ds = ModelDataset(
            feature_names=cfg.feature_names,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_v,
            y_val=y_v,
            X_test=X_v,
            y_test=y_v,
            train_dates=("2026-01-01", "2026-01-02"),
            val_dates=("2026-01-03", "2026-01-03"),
            test_dates=("2026-01-04", "2026-01-04"),
            total_raw_observations=6,
            total_labeled_observations=6,
            purged_embargo_count=0,
        )

        model.fit(ds)
        self.assertTrue(model.is_trained)

        # Test predict_bar on synthetic bars
        base_dt = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            MarketBar(
                ticker="TEST",
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000.0,
                source_timestamp=datetime.fromtimestamp(base_dt.timestamp() + i * 60, tz=timezone.utc),
                provenance="TEST",
                data_quality=DataQualityStatus.VALID,
            )
            for i in range(35)
        ]

        pred = model.predict_bar(bars)
        self.assertEqual(pred.prediction_type, PredictionType.DIRECTIONAL_PROBABILITY)
        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertGreaterEqual(pred.prediction_value, 0.0)
        self.assertLessEqual(pred.prediction_value, 1.0)
        self.assertIn("predicted_probability", pred.metadata)


if __name__ == "__main__":
    unittest.main()

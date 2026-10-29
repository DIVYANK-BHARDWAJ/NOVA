"""Unit tests for CausalRidgePredictor (chronological splits, zero leakage, fallbacks)."""

from datetime import datetime, timezone
import unittest

from nova.models.predictor import CausalRidgePredictor, PredictionResult, PredictorConfig
from nova.quant.contracts import DataQualityStatus
from tests.test_person_3_strategies import create_synthetic_bars


class TestCausalRidgePredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PredictorConfig(
            horizon_bars=1,
            l2_regularization=0.5,
            min_train_bars=25,
            lookback_features=(1, 3, 5),
        )
        self.predictor = CausalRidgePredictor(self.config)

    def test_unfitted_model_graceful_fallback(self) -> None:
        # Before fitting, prediction falls back gracefully to causal momentum heuristic
        bars = create_synthetic_bars([100.0 + i for i in range(15)])
        pred: PredictionResult = self.predictor.predict(bars)
        self.assertTrue(pred.is_fallback)
        self.assertGreater(pred.confidence, 0.0)
        self.assertTrue(any("FALLBACK_UNFITTED_MOMENTUM" in r for r in pred.reason_codes))

    def test_chronological_fit_and_prediction(self) -> None:
        # Train on first 70% of chronological bars
        prices = [100.0 * (1.002 ** i) for i in range(50)]
        bars = create_synthetic_bars(prices)
        self.predictor.fit_chronological(bars, train_fraction=0.70)
        self.assertTrue(self.predictor.is_fitted)

        # Predict as of latest bar
        pred = self.predictor.predict(bars)
        self.assertFalse(pred.is_fallback)
        self.assertGreater(pred.expected_return, 0.0)
        self.assertGreater(pred.directional_probability, 0.50)
        self.assertGreaterEqual(pred.confidence, 0.20)

    def test_invalid_data_fallback(self) -> None:
        prices = [100.0 + i for i in range(30)]
        bars = create_synthetic_bars(prices, quality=DataQualityStatus.INVALID)
        pred = self.predictor.predict(bars)
        self.assertTrue(pred.is_fallback)
        self.assertEqual(pred.confidence, 0.0)
        self.assertTrue(any("FALLBACK_DATA_QUALITY_INVALID" in r for r in pred.reason_codes))


if __name__ == "__main__":
    unittest.main()

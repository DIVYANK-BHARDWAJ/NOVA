"""Unit tests for NOVA Canonical ModelPrediction Contract & Schema Parity (Milestone 1).

Tests all invariants:
- Timezone awareness and cutoff causality
- Prediction types and numerical semantic boundaries
- Serialization / deserialization round-trip determinism
- Schema validation parity with schemas/model_prediction.schema.json
- Robust negative tests for invalid, uncalibrated, or out-of-bounds data
"""

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.contracts import (
    ContractValidationError,
    ModelPrediction,
    NumericalValidationError,
    PredictionAvailability,
    PredictionType,
    TemporalValidationError,
    validate_model_prediction_schema_parity,
)


class TestModelPredictionContract(unittest.TestCase):
    """Verifies strict contract invariants and serialization for ModelPrediction."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.pred_time = datetime(2026, 3, 1, 10, 0, tzinfo=self.utc)
        self.cutoff_time = datetime(2026, 3, 1, 9, 59, 59, tzinfo=self.utc)

    def test_schema_file_exists_and_is_valid_json(self) -> None:
        """Verify schemas/model_prediction.schema.json exists and is valid JSON."""
        schema_path = Path(__file__).resolve().parent.parent / "schemas" / "model_prediction.schema.json"
        self.assertTrue(schema_path.exists(), f"Schema file missing at {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertEqual(schema.get("title"), "ModelPrediction")
        self.assertIn("required", schema)
        self.assertIn("properties", schema)
        self.assertFalse(schema.get("additionalProperties"))

    def test_valid_expected_return_prediction(self) -> None:
        """Verify valid EXPECTED_RETURN_DECIMAL prediction."""
        pred = ModelPrediction(
            model_id="GBDT_RETURN_FORECASTER",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="RELIANCE",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=5,
            prediction_value=0.0150,  # +1.50% forward return
            confidence_or_uncertainty=0.82,
            availability_status=PredictionAvailability.READY,
            metadata={"seed": 42, "num_features": 24},
        )
        self.assertEqual(pred.model_id, "GBDT_RETURN_FORECASTER")
        self.assertEqual(pred.prediction_value, 0.0150)
        self.assertEqual(pred.prediction_type, PredictionType.EXPECTED_RETURN_DECIMAL)
        self.assertEqual(pred.horizon_unit, "BARS")
        self.assertEqual(pred.contract_version, "PROPOSED_V0")

        # Verify parity with JSON schema
        errors = validate_model_prediction_schema_parity(pred.to_dict())
        self.assertEqual(errors, [])

    def test_valid_directional_probability_prediction(self) -> None:
        """Verify valid DIRECTIONAL_PROBABILITY prediction bounded in [0.0, 1.0]."""
        pred = ModelPrediction(
            model_id="TIMESFM_DIRECTIONAL",
            model_version="2.1.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="TCS",
            prediction_type=PredictionType.DIRECTIONAL_PROBABILITY,
            horizon_bars=1,
            prediction_value=0.685,  # 68.5% probability of positive return
            confidence_or_uncertainty=0.75,
            availability_status=PredictionAvailability.READY,
        )
        self.assertAlmostEqual(pred.prediction_value, 0.685)
        self.assertEqual(validate_model_prediction_schema_parity(pred.to_dict()), [])

    def test_valid_regime_probability_with_distribution(self) -> None:
        """Verify REGIME_PROBABILITY with discrete regime distribution summing to 1.0."""
        dist = {
            "TRENDING": 0.65,
            "RANGING": 0.25,
            "HIGH_VOLATILITY": 0.10,
        }
        pred = ModelPrediction(
            model_id="REGIME_DETECTOR_V1",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="INFY",
            prediction_type=PredictionType.REGIME_PROBABILITY,
            horizon_bars=1,
            prediction_value=0.65,  # Target regime probability (TRENDING)
            prediction_distribution=dist,
            confidence_or_uncertainty=0.90,
            availability_status=PredictionAvailability.READY,
        )
        self.assertAlmostEqual(pred.prediction_value, 0.65)
        self.assertEqual(validate_model_prediction_schema_parity(pred.to_dict()), [])

    def test_valid_volatility_estimate(self) -> None:
        """Verify VOLATILITY_ESTIMATE must be non-negative."""
        pred = ModelPrediction(
            model_id="GARCH_VOLATILITY",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="HDFCBANK",
            prediction_type=PredictionType.VOLATILITY_ESTIMATE,
            horizon_bars=10,
            prediction_value=0.0185,  # 1.85% estimated daily volatility
            availability_status=PredictionAvailability.READY,
        )
        self.assertEqual(pred.prediction_value, 0.0185)
        self.assertEqual(validate_model_prediction_schema_parity(pred.to_dict()), [])

    def test_valid_sentiment_score(self) -> None:
        """Verify SENTIMENT_SCORE must be in [-1.0, 1.0]."""
        pred = ModelPrediction(
            model_id="FINBERT_NEWS_NLP",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="SBIN",
            prediction_type=PredictionType.SENTIMENT_SCORE,
            horizon_bars=1,
            prediction_value=-0.75,  # Strong negative sentiment
            confidence_or_uncertainty=0.88,
            availability_status=PredictionAvailability.READY,
        )
        self.assertEqual(pred.prediction_value, -0.75)
        self.assertEqual(validate_model_prediction_schema_parity(pred.to_dict()), [])

    def test_explicit_unavailable_and_insufficient_history_predictions(self) -> None:
        """Verify UNAVAILABLE and INSUFFICIENT_HISTORY states are explicitly supported."""
        unavail = ModelPrediction(
            model_id="GBDT_OHLCV_V1",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="RELIANCE",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=5,
            prediction_value=0.0,
            availability_status=PredictionAvailability.UNAVAILABLE,
            metadata={"reason": "INFERENCE_TIMEOUT"},
        )
        self.assertEqual(unavail.availability_status, PredictionAvailability.UNAVAILABLE)
        self.assertEqual(validate_model_prediction_schema_parity(unavail.to_dict()), [])

        warmup = ModelPrediction(
            model_id="GBDT_OHLCV_V1",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="RELIANCE",
            prediction_type=PredictionType.DIRECTIONAL_PROBABILITY,
            horizon_bars=5,
            prediction_value=0.5,
            availability_status=PredictionAvailability.INSUFFICIENT_HISTORY,
            metadata={"warmup_bars": 12, "required_bars": 30},
        )
        self.assertEqual(warmup.availability_status, PredictionAvailability.INSUFFICIENT_HISTORY)
        self.assertEqual(validate_model_prediction_schema_parity(warmup.to_dict()), [])

    def test_serialization_round_trip_deterministic(self) -> None:
        """Verify to_dict / from_dict and to_json / from_json round-trips match exactly."""
        pred = ModelPrediction(
            model_id="GBDT_MODEL",
            model_version="1.0.0",
            prediction_timestamp=self.pred_time,
            input_cutoff_timestamp=self.cutoff_time,
            instrument_id="RELIANCE",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=3,
            prediction_value=0.0210,
            confidence_or_uncertainty=0.79,
            metadata={"metric": "ic_ir", "fold": 2},
        )
        json_str = pred.to_json()
        reconstructed = ModelPrediction.from_json(json_str)

        self.assertEqual(pred.model_id, reconstructed.model_id)
        self.assertEqual(pred.model_version, reconstructed.model_version)
        self.assertEqual(pred.prediction_timestamp, reconstructed.prediction_timestamp)
        self.assertEqual(pred.input_cutoff_timestamp, reconstructed.input_cutoff_timestamp)
        self.assertEqual(pred.instrument_id, reconstructed.instrument_id)
        self.assertEqual(pred.prediction_type, reconstructed.prediction_type)
        self.assertEqual(pred.horizon_bars, reconstructed.horizon_bars)
        self.assertEqual(pred.prediction_value, reconstructed.prediction_value)
        self.assertEqual(pred.confidence_or_uncertainty, reconstructed.confidence_or_uncertainty)
        self.assertEqual(pred.availability_status, reconstructed.availability_status)
        self.assertEqual(pred.metadata, reconstructed.metadata)

    def test_cutoff_cannot_be_after_prediction_timestamp(self) -> None:
        """Temporal Invariant: input_cutoff_timestamp cannot be in the future of prediction_timestamp."""
        future_cutoff = self.pred_time + timedelta(seconds=1)
        with self.assertRaises(TemporalValidationError):
            ModelPrediction(
                model_id="M1",
                model_version="1.0",
                prediction_timestamp=self.pred_time,
                input_cutoff_timestamp=future_cutoff,
                instrument_id="TCS",
                prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                horizon_bars=1,
                prediction_value=0.01,
            )

    def test_naive_timestamps_rejected(self) -> None:
        """Temporal Invariant: Timestamps without timezone must be rejected."""
        naive_dt = datetime(2026, 3, 1, 10, 0)  # No tzinfo
        with self.assertRaises(TemporalValidationError):
            ModelPrediction(
                model_id="M1",
                model_version="1.0",
                prediction_timestamp=naive_dt,
                input_cutoff_timestamp=self.cutoff_time,
                instrument_id="TCS",
                prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                horizon_bars=1,
                prediction_value=0.01,
            )
        with self.assertRaises(TemporalValidationError):
            ModelPrediction(
                model_id="M1",
                model_version="1.0",
                prediction_timestamp=self.pred_time,
                input_cutoff_timestamp=naive_dt,
                instrument_id="TCS",
                prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                horizon_bars=1,
                prediction_value=0.01,
            )

    def test_blank_identifiers_rejected(self) -> None:
        """Empty or blank model_id, model_version, or instrument_id must be rejected."""
        for kw in [
            {"model_id": ""},
            {"model_id": "   "},
            {"model_version": ""},
            {"instrument_id": ""},
            {"horizon_unit": ""},
        ]:
            args = {
                "model_id": "M1",
                "model_version": "1.0",
                "prediction_timestamp": self.pred_time,
                "input_cutoff_timestamp": self.cutoff_time,
                "instrument_id": "TCS",
                "prediction_type": PredictionType.EXPECTED_RETURN_DECIMAL,
                "horizon_bars": 1,
                "prediction_value": 0.01,
            }
            args.update(kw)
            with self.assertRaises(ContractValidationError):
                ModelPrediction(**args)

    def test_invalid_horizon_rejected(self) -> None:
        """horizon_bars must be an integer >= 1 (non-boolean)."""
        for bad_h in [0, -1, True, False, 1.5, "1"]:
            with self.assertRaises(ContractValidationError):
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.pred_time,
                    input_cutoff_timestamp=self.cutoff_time,
                    instrument_id="TCS",
                    prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                    horizon_bars=bad_h,  # type: ignore
                    prediction_value=0.01,
                )

    def test_non_finite_prediction_value_rejected(self) -> None:
        """NaN, +inf, -inf, and booleans in prediction_value must be rejected."""
        for bad_val in [float("nan"), float("inf"), float("-inf"), True, False, "0.05"]:
            with self.assertRaises(NumericalValidationError):
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.pred_time,
                    input_cutoff_timestamp=self.cutoff_time,
                    instrument_id="TCS",
                    prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                    horizon_bars=1,
                    prediction_value=bad_val,  # type: ignore
                )

    def test_probability_range_violations_rejected(self) -> None:
        """Probabilities outside [0.0, 1.0] must raise NumericalValidationError."""
        for bad_prob in [-0.01, 1.01, 1.5, -1.0]:
            with self.assertRaises(NumericalValidationError):
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.pred_time,
                    input_cutoff_timestamp=self.cutoff_time,
                    instrument_id="TCS",
                    prediction_type=PredictionType.DIRECTIONAL_PROBABILITY,
                    horizon_bars=1,
                    prediction_value=bad_prob,
                )
            with self.assertRaises(NumericalValidationError):
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.pred_time,
                    input_cutoff_timestamp=self.cutoff_time,
                    instrument_id="TCS",
                    prediction_type=PredictionType.REGIME_PROBABILITY,
                    horizon_bars=1,
                    prediction_value=bad_prob,
                )

    def test_volatility_negative_rejected(self) -> None:
        """VOLATILITY_ESTIMATE cannot be negative."""
        with self.assertRaises(NumericalValidationError):
            ModelPrediction(
                model_id="M1",
                model_version="1.0",
                prediction_timestamp=self.pred_time,
                input_cutoff_timestamp=self.cutoff_time,
                instrument_id="TCS",
                prediction_type=PredictionType.VOLATILITY_ESTIMATE,
                horizon_bars=1,
                prediction_value=-0.05,
            )

    def test_sentiment_out_of_bounds_rejected(self) -> None:
        """SENTIMENT_SCORE outside [-1.0, 1.0] must be rejected."""
        for bad_sent in [-1.05, 1.05, 2.0]:
            with self.assertRaises(NumericalValidationError):
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.pred_time,
                    input_cutoff_timestamp=self.cutoff_time,
                    instrument_id="TCS",
                    prediction_type=PredictionType.SENTIMENT_SCORE,
                    horizon_bars=1,
                    prediction_value=bad_sent,
                )

    def test_confidence_out_of_bounds_rejected(self) -> None:
        """confidence_or_uncertainty outside [0.0, 1.0] must be rejected."""
        for bad_c in [-0.1, 1.1]:
            with self.assertRaises(NumericalValidationError):
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.pred_time,
                    input_cutoff_timestamp=self.cutoff_time,
                    instrument_id="TCS",
                    prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                    horizon_bars=1,
                    prediction_value=0.01,
                    confidence_or_uncertainty=bad_c,
                )

    def test_regime_distribution_sum_violation_rejected(self) -> None:
        """Regime distribution probabilities must sum to ~1.0."""
        invalid_dist = {"TRENDING": 0.5, "RANGING": 0.2}  # Sums to 0.7 != 1.0
        with self.assertRaises(NumericalValidationError):
            ModelPrediction(
                model_id="M1",
                model_version="1.0",
                prediction_timestamp=self.pred_time,
                input_cutoff_timestamp=self.cutoff_time,
                instrument_id="TCS",
                prediction_type=PredictionType.REGIME_PROBABILITY,
                horizon_bars=1,
                prediction_value=0.5,
                prediction_distribution=invalid_dist,
            )

    def test_schema_parity_catches_invalid_dict(self) -> None:
        """validate_model_prediction_schema_parity catches missing properties and bad types."""
        # Missing required property
        bad_payload = {
            "model_id": "M1",
            "model_version": "1.0",
            # missing prediction_timestamp
            "input_cutoff_timestamp": "2026-03-01T10:00:00+00:00",
            "instrument_id": "TCS",
            "prediction_type": "EXPECTED_RETURN_DECIMAL",
            "horizon_bars": 1,
            "horizon_unit": "BARS",
            "prediction_value": 0.01,
            "availability_status": "READY",
            "contract_version": "PROPOSED_V0",
        }
        errors = validate_model_prediction_schema_parity(bad_payload)
        self.assertTrue(any("Missing required property: 'prediction_timestamp'" in e for e in errors))

        # Unknown additional property
        bad_payload["prediction_timestamp"] = "2026-03-01T10:00:00+00:00"
        bad_payload["unauthorized_field"] = "malicious"
        errors = validate_model_prediction_schema_parity(bad_payload)
        self.assertTrue(any("Additional property not permitted" in e for e in errors))


if __name__ == "__main__":
    unittest.main()


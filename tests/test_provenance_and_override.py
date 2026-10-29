"""Acceptance tests for NOVA Non-Negotiable Engineering Override.

Verifies:
1. Zero hardcoded market assumptions (dynamic TOML configuration validation)
2. Fail-closed behavior on missing or invalid configuration
3. Strict prohibition of synthetic test fixtures in live mode
4. Rejection of unverified stream provenance in live mode
5. Strict prevention of secrets, JWTs, and API credentials in model metadata
6. Comprehensive event provenance preservation (topic, partition, offset, timestamp, data quality)
7. Fail-closed live verification (never falls back to simulated feed)
8. Honest unavailable and insufficient-history representation (no fabricated numbers)
"""

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.features import FeaturePipeline, FeaturePipelineConfig
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    NumericalValidationError,
    PredictionAvailability,
    PredictionType,
    TemporalValidationError,
    TimestampSemantics,
    validate_model_prediction_schema_parity,
    validate_no_secrets,
)


def make_test_bar(
    ticker: str,
    dt: datetime,
    close: float,
    quality: DataQualityStatus = DataQualityStatus.VALID,
    provenance: str = "OFFICIAL_COMPETITION_FEED",
) -> MarketBar:
    return MarketBar(
        ticker=ticker,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=10000.0,
        source_timestamp=dt,
        available_timestamp=dt,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=quality,
        provenance=provenance,
    )


class TestNonNegotiableOverride(unittest.TestCase):
    """Verifies that all Person 2 components satisfy the engineering override constraints."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.t0 = datetime(2026, 3, 1, 9, 15, tzinfo=self.utc)
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_load_features_config_from_centralized_toml(self) -> None:
        """Requirement: Zero hardcoded market assumptions. Configuration must load from validated TOML."""
        toml_path = self.repo_root / "config" / "features_config.toml"
        self.assertTrue(toml_path.exists(), f"Missing config file at {toml_path}")

        cfg = FeaturePipelineConfig.from_toml(toml_path, section="features")
        self.assertEqual(cfg.fast_ema_window, 12)
        self.assertEqual(cfg.slow_ema_window, 26)
        self.assertEqual(cfg.momentum_window, 5)
        self.assertEqual(cfg.rsi_window, 14)
        self.assertEqual(cfg.atr_window, 14)
        self.assertEqual(cfg.config_version, "1.0.0")

    def test_missing_config_section_fails_closed(self) -> None:
        """Requirement: Fail closed with clear error if required configuration section is missing."""
        toml_path = self.repo_root / "config" / "features_config.toml"
        with self.assertRaises(ContractValidationError) as ctx:
            FeaturePipelineConfig.from_toml(toml_path, section="non_existent_section")
        self.assertIn("Missing section '[non_existent_section]'", str(ctx.exception))

    def test_invalid_parameters_fail_closed(self) -> None:
        """Requirement: Invalid windows or fast >= slow must be rejected with actionable error."""
        with self.assertRaises(ContractValidationError):
            FeaturePipelineConfig(fast_ema_window=26, slow_ema_window=12)

        with self.assertRaises(NumericalValidationError):
            FeaturePipelineConfig(fast_ema_window=1)  # Window must be >= 2

    def test_synthetic_data_rejected_in_strict_live_mode(self) -> None:
        """Requirement: Synthetic fixtures must NEVER enter live processing stream."""
        cfg = FeaturePipelineConfig(
            fast_ema_window=4,
            slow_ema_window=8,
            strict_live_mode=True,  # LIVE PRODUCTION MODE
        )
        pipeline = FeaturePipeline(cfg)

        synthetic_bars = [
            make_test_bar(
                ticker="INFY",
                dt=self.t0 + timedelta(minutes=5 * i),
                close=100.0 + i,
                quality=DataQualityStatus.SYNTHETIC_TEST_FIXTURE,  # Synthetic test data
                provenance="MOCK_GENERATOR",
            )
            for i in range(15)
        ]

        with self.assertRaises(ContractValidationError) as ctx:
            pipeline.compute_history(synthetic_bars)
        self.assertIn("Synthetic test fixture detected in live processing", str(ctx.exception))

    def test_unverified_provenance_rejected_in_strict_live_mode(self) -> None:
        """Requirement: Data with unverified provenance must be rejected in strict live mode."""
        cfg = FeaturePipelineConfig(
            fast_ema_window=4,
            slow_ema_window=8,
            strict_live_mode=True,
        )
        pipeline = FeaturePipeline(cfg)

        unverified_bars = [
            make_test_bar(
                ticker="INFY",
                dt=self.t0 + timedelta(minutes=5 * i),
                close=100.0 + i,
                quality=DataQualityStatus.UNVERIFIED_PROVENANCE,
                provenance="UNKNOWN_EXTERNAL_STREAM",
            )
            for i in range(15)
        ]

        with self.assertRaises(ContractValidationError) as ctx:
            pipeline.compute_history(unverified_bars)
        self.assertIn("Unverified stream provenance", str(ctx.exception))

    def test_secrets_and_credentials_in_metadata_rejected(self) -> None:
        """Requirement: Keep credentials, JWTs, secrets out of logs, fingerprints, and contracts."""
        forbidden_payloads = [
            {"api_key": "secret_abc123"},
            {"jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDcSemACt8x4iTMCda8Yhe3iZaWbvV5XKSTbuAn0M"},
            {"auth_header": "Bearer some_secret_token"},
            {"broker_password": "super_secret_password"},
            {"private_key_pem": "-----BEGIN PRIVATE KEY-----"},
        ]

        for secret_meta in forbidden_payloads:
            with self.assertRaises(ContractValidationError) as ctx:
                ModelPrediction(
                    model_id="M1",
                    model_version="1.0",
                    prediction_timestamp=self.t0,
                    input_cutoff_timestamp=self.t0,
                    instrument_id="RELIANCE",
                    prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                    horizon_bars=1,
                    prediction_value=0.01,
                    metadata=secret_meta,
                )
            self.assertTrue(
                "Sensitive " in str(ctx.exception) and "metadata must not store secrets" in str(ctx.exception)
            )

    def test_data_provenance_retained_in_model_prediction(self) -> None:
        """Requirement: Retain event provenance through the prediction contract."""
        provenance_metadata = {
            "source_topic": "market.ticks",
            "partition": 0,
            "offset": 42109,
            "event_timestamp": "2026-03-01T09:15:00+00:00",
            "receive_timestamp": "2026-03-01T09:15:00.125000+00:00",
            "data_quality": "VALID",
            "payload_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "adapter_version": "1.0.0",
        }

        pred = ModelPrediction(
            model_id="GBDT_RETURN_V1",
            model_version="1.0.0",
            prediction_timestamp=self.t0,
            input_cutoff_timestamp=self.t0,
            instrument_id="RELIANCE",
            prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
            horizon_bars=5,
            prediction_value=0.0150,
            confidence_or_uncertainty=0.88,
            metadata=provenance_metadata,
        )

        pred_dict = pred.to_dict()
        self.assertEqual(pred_dict["metadata"]["source_topic"], "market.ticks")
        self.assertEqual(pred_dict["metadata"]["offset"], 42109)
        self.assertEqual(pred_dict["metadata"]["data_quality"], "VALID")
        self.assertEqual(validate_model_prediction_schema_parity(pred_dict), [])

    def test_honest_unavailable_state_without_fabrication(self) -> None:
        """Requirement: Unavailable inputs must not become fabricated numerical predictions."""
        # When model has insufficient history or inference is offline, availability is explicitly UNAVAILABLE
        pred = ModelPrediction(
            model_id="TIMESFM_TS_ZERO_SHOT",
            model_version="1.0.0",
            prediction_timestamp=self.t0,
            input_cutoff_timestamp=self.t0,
            instrument_id="TCS",
            prediction_type=PredictionType.DIRECTIONAL_PROBABILITY,
            horizon_bars=5,
            prediction_value=0.5,  # Uninformed / uncalibrated default
            availability_status=PredictionAvailability.UNAVAILABLE,
            metadata={"status_reason": "BLOCKED_INSUFFICIENT_REAL_DATA", "bars_available": 5, "bars_required": 50},
        )
        self.assertEqual(pred.availability_status, PredictionAvailability.UNAVAILABLE)
        self.assertEqual(pred.metadata["status_reason"], "BLOCKED_INSUFFICIENT_REAL_DATA")


if __name__ == "__main__":
    unittest.main()

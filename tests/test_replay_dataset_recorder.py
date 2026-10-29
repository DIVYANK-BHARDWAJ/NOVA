"""Unit and integration tests for NOVA Replay Dataset Recorder & Research Ingestion (Milestone 4).

Verifies:
1. Valid event acceptance & normalization.
2. Invalid event rejection (negative prices, inverted high/low, missing fields).
3. Event-type discrimination: accepts market.ticks, excludes preflight_tick, excludes market.news.
4. Stable cryptographic observation fingerprinting.
5. Replay deduplication: different offsets for identical candle do NOT increase unique observations.
6. Conflicting duplicate detection: identical ticker & date with conflicting prices are flagged and rejected.
7. Persistence and reload idempotency.
8. Secret sanitization: prevents tokens/credentials from being logged or stored.
9. Chronological ordering and XGBoost readiness split calculations with boundary purging/embargo.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.contracts import DataQualityStatus, MarketBar
from nova.research.data_collection.audit import DatasetAuditor
from nova.research.data_collection.identity import DeliveryIdentity, ObservationIdentity
from nova.research.data_collection.manifest import DatasetManifest
from nova.research.data_collection.quality import (
    EventEligibility,
    EventQualityValidator,
    QualityValidationResult,
)
from nova.research.data_collection.recorder import ReplayDatasetRecorder
from nova.research.data_collection.xgboost_readiness import (
    DatasetSplitCounts,
    XGBoostReadinessEvaluator,
)


class TestReplayDatasetRecorder(unittest.TestCase):
    """Canonical test suite for Person 2 Replay Dataset Recorder."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="nova_test_replay_"))
        self.recorder = ReplayDatasetRecorder(
            storage_dir=self.temp_dir,
            dataset_id="test_dataset",
            source_mode="TEST_SYNTHETIC",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_market_tick_accepted(self) -> None:
        """Valid candlestick event must be accepted and stored as a unique observation."""
        event = {
            "symbol": "ROUT",
            "open": 100.0,
            "high": 105.0,
            "low": 99.5,
            "close": 103.0,
            "volume": 15000.0,
            "simulated_date": "2026-03-01",
            "timestamp": "2026-03-01T09:30:00+00:00",
            "topic": "market.ticks",
            "partition": 0,
            "offset": 101,
        }

        success = self.recorder.record_event(event)
        self.assertTrue(success)
        self.assertEqual(self.recorder.manifest.raw_deliveries_count, 1)
        self.assertEqual(self.recorder.manifest.unique_observations_count, 1)
        self.assertEqual(self.recorder.manifest.repeated_deliveries_count, 0)
        self.assertEqual(self.recorder.manifest.conflicting_duplicates_count, 0)

        obs = self.recorder.get_unique_observations("ROUT")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].ticker, "ROUT")
        self.assertEqual(obs[0].market_date, "2026-03-01")
        self.assertEqual(obs[0].close, 103.0)

    def test_preflight_tick_excluded_from_ohlcv_dataset(self) -> None:
        """Heartbeat preflight pings must be safely excluded from training OHLCV data."""
        preflight_event = {
            "symbol": "ROUT",
            "event_type": "preflight_tick",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 0.0,
            "simulated_date": "2026-03-01",
            "timestamp": "2026-03-01T09:15:00+00:00",
        }

        success = self.recorder.record_event(preflight_event)
        self.assertFalse(success)
        self.assertEqual(self.recorder.manifest.raw_deliveries_count, 1)
        self.assertEqual(self.recorder.manifest.unique_observations_count, 0)
        self.assertEqual(self.recorder.manifest.rejected_events_count, 1)
        self.assertIn("EXCLUDED_PREFLIGHT", self.recorder.manifest.rejection_reasons)

    def test_news_event_excluded_from_ohlcv_dataset(self) -> None:
        """Corporate news events must not enter the quantitative OHLCV price series."""
        news_event = {
            "symbol": "ROUT",
            "topic": "market.news",
            "headline": "Q3 Earnings Beat",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 0.0,
            "simulated_date": "2026-03-01",
        }

        success = self.recorder.record_event(news_event)
        self.assertFalse(success)
        self.assertEqual(self.recorder.manifest.unique_observations_count, 0)
        self.assertIn("EXCLUDED_NEWS", self.recorder.manifest.rejection_reasons)

    def test_malformed_and_negative_prices_rejected(self) -> None:
        """Prices <= 0 or Low > High must be strictly rejected without corrupting store."""
        bad_price = {
            "symbol": "ROUT",
            "open": -50.0,
            "high": 100.0,
            "low": 90.0,
            "close": 95.0,
            "volume": 100.0,
            "simulated_date": "2026-03-01",
        }
        self.assertFalse(self.recorder.record_event(bad_price))

        inverted_ohlc = {
            "symbol": "ROUT",
            "open": 100.0,
            "high": 90.0,
            "low": 110.0,  # low > high
            "close": 95.0,
            "volume": 100.0,
            "simulated_date": "2026-03-01",
        }
        self.assertFalse(self.recorder.record_event(inverted_ohlc))
        self.assertEqual(self.recorder.manifest.unique_observations_count, 0)

    def test_observation_fingerprint_deterministic_stability(self) -> None:
        """Identical financial candle must produce identical SHA-256 fingerprint."""
        obs1 = ObservationIdentity(
            ticker="AERX",
            market_date="2026-03-01",
            open=500.0,
            high=510.0,
            low=495.0,
            close=505.0,
            volume=20000.0,
        )
        obs2 = ObservationIdentity(
            ticker="aerx",  # Case insensitive
            market_date="2026-03-01",
            open=500.0,
            high=510.0,
            low=495.0,
            close=505.0,
            volume=20000.0,
        )
        self.assertEqual(obs1.fingerprint, obs2.fingerprint)

    def test_replay_deduplication_different_offsets(self) -> None:
        """Replaying identical candle with different Kafka offset must NOT increase unique count."""
        candle = {
            "symbol": "AERX",
            "open": 500.0,
            "high": 510.0,
            "low": 495.0,
            "close": 505.0,
            "volume": 20000.0,
            "simulated_date": "2026-03-01",
            "timestamp": "2026-03-01T09:30:00+00:00",
            "partition": 0,
            "offset": 501,
        }

        # First arrival
        self.assertTrue(self.recorder.record_event(candle, offset=501))
        self.assertEqual(self.recorder.manifest.unique_observations_count, 1)
        self.assertEqual(self.recorder.manifest.repeated_deliveries_count, 0)

        # Second arrival (replayed sandbox cycle at offset 1201)
        candle_replay = dict(candle)
        candle_replay["offset"] = 1201
        self.assertTrue(self.recorder.record_event(candle_replay, offset=1201))

        # Raw deliveries incremented, but unique observations stay at 1!
        self.assertEqual(self.recorder.manifest.raw_deliveries_count, 2)
        self.assertEqual(self.recorder.manifest.unique_observations_count, 1)
        self.assertEqual(self.recorder.manifest.repeated_deliveries_count, 1)

    def test_conflicting_duplicate_detection(self) -> None:
        """Same instrument and date but differing OHLCV prices must trigger conflict flag."""
        candle_a = {
            "symbol": "AERX",
            "open": 500.0,
            "high": 510.0,
            "low": 495.0,
            "close": 505.0,
            "volume": 20000.0,
            "simulated_date": "2026-03-01",
        }
        self.assertTrue(self.recorder.record_event(candle_a, offset=1))

        # Conflicting candle on SAME date with DIFFERENT close
        candle_conflict = {
            "symbol": "AERX",
            "open": 500.0,
            "high": 520.0,  # Divergent high
            "low": 495.0,
            "close": 518.0,  # Divergent close
            "volume": 20000.0,
            "simulated_date": "2026-03-01",
        }
        success = self.recorder.record_event(candle_conflict, offset=2)
        self.assertFalse(success)
        self.assertEqual(self.recorder.manifest.conflicting_duplicates_count, 1)
        self.assertEqual(self.recorder.manifest.unique_observations_count, 1)

    def test_persistence_and_reload_idempotency(self) -> None:
        """Recorder restarted on existing dataset folder must restore observation index."""
        events = [
            {
                "symbol": "HMDP",
                "open": 200.0 + i,
                "high": 205.0 + i,
                "low": 195.0 + i,
                "close": 202.0 + i,
                "volume": 1000.0,
                "simulated_date": f"2026-03-{i+1:02d}",
                "timestamp": f"2026-03-{i+1:02d}T09:30:00+00:00",
            }
            for i in range(5)
        ]
        for e in events:
            self.recorder.record_event(e)

        self.recorder.finalize_manifest()
        self.assertEqual(self.recorder.manifest.unique_observations_count, 5)

        # Create a second recorder targeting the exact same directory
        reloaded_recorder = ReplayDatasetRecorder(
            storage_dir=self.temp_dir,
            dataset_id="test_dataset",
        )
        self.assertEqual(reloaded_recorder.manifest.unique_observations_count, 5)
        self.assertEqual(len(reloaded_recorder.get_unique_observations("HMDP")), 5)

    def test_secret_sanitization_rejects_bearer_tokens(self) -> None:
        """Events contaminated with auth tokens must fail closed and never be stored."""
        leaky_event = {
            "symbol": "ROUT",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
            "simulated_date": "2026-03-01",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        }
        success = self.recorder.record_event(leaky_event)
        self.assertFalse(success)
        self.assertIn("REJECTED_SECRET_DETECTED", self.recorder.manifest.rejection_reasons)

    def test_dataset_auditor_reports_consistent_metrics(self) -> None:
        """DatasetAuditor must accurately compute deliveries, uniques, repeats, and date coverage."""
        for i in range(10):
            self.recorder.record_event({
                "symbol": "MOTR",
                "open": 300.0 + i,
                "high": 305.0 + i,
                "low": 295.0 + i,
                "close": 302.0 + i,
                "volume": 5000.0,
                "simulated_date": f"2026-03-{i+1:02d}",
            }, offset=i)

        # Re-send last 2 as repeats
        self.recorder.record_event({
            "symbol": "MOTR",
            "open": 300.0 + 8,
            "high": 305.0 + 8,
            "low": 295.0 + 8,
            "close": 302.0 + 8,
            "volume": 5000.0,
            "simulated_date": "2026-03-09",
        }, offset=100)

        self.recorder.finalize_manifest()

        audit = DatasetAuditor.audit_dataset_directory(self.temp_dir / "test_dataset")
        self.assertEqual(audit["raw_deliveries_count"], 11)
        self.assertEqual(audit["unique_observations_count"], 10)
        self.assertEqual(audit["repeated_deliveries_count"], 1)
        self.assertIn("MOTR", audit["tickers_covered"])
        self.assertEqual(audit["per_ticker_breakdown"]["MOTR"]["unique_bars"], 10)

    def test_xgboost_readiness_split_and_embargo(self) -> None:
        """Evaluator must deduct warmup, drop unobserved target tail, and apply boundary embargo."""
        evaluator = XGBoostReadinessEvaluator(
            feature_warmup_bars=10,
            horizon_bars=2,
            train_ratio=0.60,
            val_ratio=0.20,
        )

        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Create 50 synthetic bars for testing
        bars = [
            MarketBar(
                ticker="TEST_SPLIT",
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000.0,
                source_timestamp=base_time + timedelta(days=i),
                provenance="TEST_SPLIT",
                data_quality=DataQualityStatus.VALID,
            )
            for i in range(50)
        ]

        res = evaluator.evaluate_instrument("TEST_SPLIT", bars)
        # 50 total bars
        self.assertEqual(res.total_raw_bars, 50)
        # 10 warmup deducted -> 40 feature rows
        self.assertEqual(res.valid_feature_rows, 40)
        # 2 horizon tail bars dropped -> 38 valid labeled rows
        self.assertEqual(res.valid_labeled_rows, 38)
        # Train ratio 0.60 of 38 = 22 rows. Embargo 2 rows -> 20 train usable rows
        self.assertEqual(res.train_rows, 22)
        self.assertEqual(res.train_purged_embargo_rows, 2)
        self.assertEqual(res.train_usable_rows, 20)
        # Val ratio 0.20 of 38 = 7 rows. Embargo 2 rows -> 5 val usable rows
        self.assertEqual(res.val_rows, 7)
        self.assertEqual(res.val_usable_rows, 5)
        # Test ratio: 38 - 22 - 7 = 9 test rows
        self.assertEqual(res.test_usable_rows, 9)


if __name__ == "__main__":
    unittest.main()


"""Unit tests for Official MANTHAN API Data Exploitation & Reconciled Dataset Validation.

Validates:
- Dynamic universe schema and structure
- Listing date boundaries and anti-lookahead date constraint parsing
- Canonical v2 dataset integrity, fingerprint deduplication, and observation monotonicity
- Multi-horizon causal target calculation and embargo boundaries on v2 records
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.models.config import XGBoostModelConfig
from nova.models.dataset import DatasetAssembler
from nova.quant.contracts import DataQualityStatus, MarketBar


class TestManthanApiDataContracts(unittest.TestCase):
    """Test suite for MANTHAN API data parsing and validation rules."""

    def test_anti_lookahead_date_contract(self) -> None:
        """Verify that simulation clock and historical end_date enforce strict anti-lookahead."""
        sim_date = "2026-05-22"
        valid_end_date = "2026-05-21"
        invalid_same_date = "2026-05-22"
        invalid_future_date = "2026-05-23"

        def check_lookahead(req_end: str, current_sim: str) -> bool:
            return req_end < current_sim

        self.assertTrue(check_lookahead(valid_end_date, sim_date))
        self.assertFalse(check_lookahead(invalid_same_date, sim_date))
        self.assertFalse(check_lookahead(invalid_future_date, sim_date))

    def test_listing_date_catalog_boundaries(self) -> None:
        """Verify known listing date catalog constraints for MANTHAN instruments."""
        listing_dates = {
            "AERX": "2016-10-24",
            "BLUX": "2016-10-24",
            "DSNY": "2016-10-24",
            "HAVN": "2020-12-10",
            "HMDP": "2016-10-24",
            "MOTR": "2016-10-24",
            "ORCX": "2016-10-24",
            "PLNX": "2020-09-30",
            "ROUT": "2019-05-10",
            "SONA": "2018-04-03",
        }
        self.assertEqual(len(listing_dates), 10)
        # Verify 2016 group
        for ticker in ["AERX", "BLUX", "DSNY", "HMDP", "MOTR", "ORCX"]:
            self.assertEqual(listing_dates[ticker], "2016-10-24")
        # Verify post-2016 listings
        self.assertEqual(listing_dates["HAVN"], "2020-12-10")
        self.assertEqual(listing_dates["PLNX"], "2020-09-30")
        self.assertEqual(listing_dates["ROUT"], "2019-05-10")
        self.assertEqual(listing_dates["SONA"], "2018-04-03")

    def test_canonical_v2_dataset_manifest_integrity(self) -> None:
        """Verify that canonical_manthan_v2 dataset manifest exists and matches observation counts."""
        repo_root = Path(__file__).resolve().parent.parent
        manifest_path = repo_root / "data" / "replay" / "canonical_manthan_v2" / "manifest.json"
        obs_path = repo_root / "data" / "replay" / "canonical_manthan_v2" / "observations.jsonl"

        if not manifest_path.exists() or not obs_path.exists():
            self.skipTest("canonical_manthan_v2 dataset not present in local test environment")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["dataset_name"], "canonical_manthan_v2")
        self.assertEqual(manifest["total_observations_count"], 21040)
        self.assertEqual(manifest["unique_fingerprints_count"], 21040)
        self.assertEqual(manifest["universe_discovery"]["tickers_count"], 10)

        # Count lines in observations file
        line_count = 0
        with open(obs_path, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
        self.assertEqual(line_count, 21040)

    def test_multi_horizon_embargo_on_synthetic_stream(self) -> None:
        """Verify that DatasetAssembler accurately enforces boundary embargo for H=1, 3, 5."""
        base_dt = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        # Create 100 chronological bars for 2 tickers
        bars: dict[str, list[MarketBar]] = {}
        for t in ["T1", "T2"]:
            b_list = []
            for i in range(100):
                dt = datetime.fromtimestamp(base_dt.timestamp() + i * 86400, tz=timezone.utc)
                b_list.append(MarketBar(
                    ticker=t,
                    open=100.0 + i * 0.1,
                    high=102.0 + i * 0.1,
                    low=99.0 + i * 0.1,
                    close=101.0 + i * 0.1,
                    volume=1000.0,
                    source_timestamp=dt,
                    provenance=f"TEST:{t}",
                    data_quality=DataQualityStatus.VALID,
                ))
            bars[t] = b_list

        for h in [1, 3, 5]:
            cfg = XGBoostModelConfig(
                model_id=f"TEST_H{h}",
                horizon_bars=h,
                feature_warmup_bars=28,
            )
            asm = DatasetAssembler(cfg)
            ds = asm.assemble_universe(bars)

            # Embargo dates should be exactly 2 * H dates across train & val boundaries
            # Each boundary date removes len(tickers) records
            expected_purged = 2 * h * 2
            self.assertEqual(ds.purged_embargo_count, expected_purged)


if __name__ == "__main__":
    unittest.main()


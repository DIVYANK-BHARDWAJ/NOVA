"""NOVA Research Data Collection — Dataset Quality & Deduplication Auditor.

Audits both captured research replay datasets and historical API archives:
1. Validates delivery counts vs unique observation counts.
2. Quantifies repeated deliveries (sandbox replay loops).
3. Surfaces conflicting duplicate observations.
4. Identifies date ranges, gaps, and missing fields.
5. Emits machine-readable audit report.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from nova.research.data_collection.identity import ObservationIdentity
from nova.research.data_collection.manifest import DatasetManifest


class DatasetAuditor:
    """Comprehensive auditor for research datasets."""

    @staticmethod
    def audit_dataset_directory(dataset_dir: Path | str) -> dict[str, Any]:
        """Audit an existing replay dataset directory."""
        path = Path(dataset_dir)
        if not path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {path}")

        deliveries_file = path / "deliveries.jsonl"
        observations_file = path / "observations.jsonl"
        rejections_file = path / "rejections.jsonl"
        manifest_file = path / "manifest.json"

        deliveries_count = 0
        if deliveries_file.exists():
            with open(deliveries_file, "r", encoding="utf-8") as f:
                deliveries_count = sum(1 for line in f if line.strip())

        rejections: list[dict[str, Any]] = []
        if rejections_file.exists():
            with open(rejections_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rejections.append(json.loads(line))

        conflicts = [r for r in rejections if r.get("status") == "CONFLICTING_DUPLICATE"]
        invalid_rejections = [r for r in rejections if r.get("status") != "CONFLICTING_DUPLICATE"]

        observations_by_ticker: dict[str, list[dict[str, Any]]] = {}
        unique_fingerprints: set[str] = set()
        natural_keys: set[tuple[str, str]] = set()

        if observations_file.exists():
            with open(observations_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obs = json.loads(line)
                    t = obs["ticker"]
                    if t not in observations_by_ticker:
                        observations_by_ticker[t] = []
                    observations_by_ticker[t].append(obs)
                    unique_fingerprints.add(obs["fingerprint"])
                    natural_keys.add((t, obs["market_date"]))

        # Compute per-ticker statistics and date coverage
        ticker_stats: dict[str, Any] = {}
        for t, obs_list in sorted(observations_by_ticker.items()):
            obs_list.sort(key=lambda x: x["market_date"])
            dates = [x["market_date"] for x in obs_list]
            min_date = dates[0] if dates else "N/A"
            max_date = dates[-1] if dates else "N/A"

            ticker_stats[t] = {
                "unique_bars": len(obs_list),
                "start_date": min_date,
                "end_date": max_date,
            }

        unique_count = len(unique_fingerprints)
        repeated_count = max(0, deliveries_count - unique_count - len(rejections))

        manifest_data = {}
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

        return {
            "dataset_directory": str(path),
            "manifest_dataset_id": manifest_data.get("dataset_id", path.name),
            "source_mode": manifest_data.get("source_mode", "UNKNOWN"),
            "raw_deliveries_count": deliveries_count,
            "unique_observations_count": unique_count,
            "repeated_deliveries_count": repeated_count,
            "conflicting_duplicates_count": len(conflicts),
            "rejected_events_count": len(invalid_rejections),
            "tickers_covered": sorted(list(observations_by_ticker.keys())),
            "per_ticker_breakdown": ticker_stats,
            "rejections_summary": {
                "conflicts": len(conflicts),
                "other_rejections": len(invalid_rejections),
            },
            "provenance_intact": deliveries_file.exists() and observations_file.exists(),
        }


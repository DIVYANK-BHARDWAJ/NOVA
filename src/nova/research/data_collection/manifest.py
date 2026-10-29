"""NOVA Research Data Collection — Dataset Manifest & Lineage Tracking.

Maintains immutable manifests documenting:
- Capture run identifiers and execution windows
- Provenance source modes (HISTORICAL_API, KAFKA_STREAM, TEST_SYNTHETIC)
- Raw deliveries vs unique observations counts
- Deduplication and conflict statistics
- File hashes for cryptographic auditability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from nova.quant.contracts import ContractValidationError, validate_no_secrets


@dataclass
class DatasetManifest:
    """Cryptographic manifest documenting a captured research dataset."""
    dataset_id: str
    created_at_utc: str
    source_mode: str  # HISTORICAL_IMPORT, KAFKA_STREAM, TEST_SYNTHETIC
    raw_deliveries_count: int = 0
    unique_observations_count: int = 0
    repeated_deliveries_count: int = 0
    conflicting_duplicates_count: int = 0
    rejected_events_count: int = 0
    tickers_covered: list[str] = field(default_factory=list)
    date_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    file_checksums: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        validate_no_secrets(self.metadata, "DatasetManifest.metadata")
        return {
            "dataset_id": self.dataset_id,
            "created_at_utc": self.created_at_utc,
            "source_mode": self.source_mode,
            "raw_deliveries_count": self.raw_deliveries_count,
            "unique_observations_count": self.unique_observations_count,
            "repeated_deliveries_count": self.repeated_deliveries_count,
            "conflicting_duplicates_count": self.conflicting_duplicates_count,
            "rejected_events_count": self.rejected_events_count,
            "tickers_covered": sorted(self.tickers_covered),
            "date_ranges": {k: list(v) for k, v in self.date_ranges.items()},
            "rejection_reasons": dict(self.rejection_reasons),
            "file_checksums": dict(self.file_checksums),
            "metadata": dict(self.metadata),
        }

    def save(self, output_path: Path) -> str:
        """Serialize manifest to JSON file and return SHA-256 hash."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        output_path.write_text(content, encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, file_path: Path) -> DatasetManifest:
        """Load manifest from JSON file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            dataset_id=data["dataset_id"],
            created_at_utc=data["created_at_utc"],
            source_mode=data["source_mode"],
            raw_deliveries_count=data.get("raw_deliveries_count", 0),
            unique_observations_count=data.get("unique_observations_count", 0),
            repeated_deliveries_count=data.get("repeated_deliveries_count", 0),
            conflicting_duplicates_count=data.get("conflicting_duplicates_count", 0),
            rejected_events_count=data.get("rejected_events_count", 0),
            tickers_covered=data.get("tickers_covered", []),
            date_ranges={k: (v[0], v[1]) for k, v in data.get("date_ranges", {}).items()},
            rejection_reasons=data.get("rejection_reasons", {}),
            file_checksums=data.get("file_checksums", {}),
            metadata=data.get("metadata", {}),
        )


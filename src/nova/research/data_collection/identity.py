"""NOVA Research Data Collection — Stable Observation Identity & Deduplication.

Differentiates between:
1. Delivery Identity: The transport-level event arrival (topic, partition, offset, capture run, local receive timestamp).
2. Underlying Observation Identity: The canonical financial reality (ticker, simulated market date, normalized OHLCV prices, volume).

Replay Invariants:
- Replayed deliveries of the identical market candle share the same Observation Identity.
- Delivery offset must NEVER be used as the observation identity.
- Local wall-clock receive timestamp must NEVER be used as the observation identity.
- Conflicting records (same ticker and market date, but different OHLCV) are flagged as CONFLICTING_DUPLICATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
from typing import Any, Mapping

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    NumericalValidationError,
    validate_finite,
    validate_no_secrets,
)


@dataclass(frozen=True, slots=True)
class DeliveryIdentity:
    """Transport-layer identification of a specific event delivery."""
    source_system: str
    topic: str
    partition: int
    offset: int
    receive_timestamp_utc: datetime
    capture_run_id: str

    def __post_init__(self) -> None:
        if not self.source_system.strip():
            raise ContractValidationError("source_system cannot be blank")
        if not self.topic.strip():
            raise ContractValidationError("topic cannot be blank")
        if self.partition < 0:
            raise ContractValidationError(f"partition must be >= 0, got {self.partition}")
        if self.offset < 0:
            raise ContractValidationError(f"offset must be >= 0, got {self.offset}")
        if self.receive_timestamp_utc.tzinfo is None:
            raise ContractValidationError("receive_timestamp_utc must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "receive_timestamp_utc": self.receive_timestamp_utc.isoformat(),
            "capture_run_id": self.capture_run_id,
        }


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """Canonical point-in-time identity of an underlying market observation.

    Generates a deterministic cryptographic fingerprint from the asset, market date,
    and rounded OHLCV tuple.
    """
    ticker: str
    market_date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", self.compute_fingerprint())

    def compute_fingerprint(self) -> str:
        """Compute stable SHA-256 digest of underlying financial observation."""
        canonical_str = (
            f"{self.ticker.upper().strip()}|"
            f"{self.market_date.strip()}|"
            f"{self.open:.4f}|"
            f"{self.high:.4f}|"
            f"{self.low:.4f}|"
            f"{self.close:.4f}|"
            f"{self.volume:.2f}"
        )
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @property
    def natural_key(self) -> tuple[str, str]:
        """The logical market key: (ticker, market_date)."""
        return (self.ticker.upper().strip(), self.market_date.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "market_date": self.market_date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "fingerprint": self.fingerprint,
        }


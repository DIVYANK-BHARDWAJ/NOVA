"""NOVA Research Data Collection — Event Filtering & Data Quality Verification.

Enforces strict research data quality:
1. Event Type Discrimination:
   - Accept: Valid price candlestick ticks from 'market.ticks'.
   - Exclude: 'preflight_tick' events (heartbeat pings, not real market bars).
   - Exclude: Corporate news events ('market.news').
2. Financial Numerical Sanity:
   - Positive prices: Open, High, Low, Close > 0.
   - Non-negative volume: Volume >= 0.
   - Price consistency: Low <= Open, Close <= High.
3. Secret Sanitization:
   - Metadata and payloads must never contain tokens, JWTs, or passwords.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math
from typing import Any, Mapping

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    NumericalValidationError,
    validate_finite,
    validate_no_secrets,
)


class EventEligibility(str, Enum):
    """Quality and eligibility status of an incoming market event."""
    ELIGIBLE = "ELIGIBLE"
    EXCLUDED_PREFLIGHT = "EXCLUDED_PREFLIGHT"
    EXCLUDED_NEWS = "EXCLUDED_NEWS"
    EXCLUDED_UNKNOWN_TYPE = "EXCLUDED_UNKNOWN_TYPE"
    REJECTED_MALFORMED = "REJECTED_MALFORMED"
    REJECTED_NEGATIVE_PRICE = "REJECTED_NEGATIVE_PRICE"
    REJECTED_OHLC_MISMATCH = "REJECTED_OHLC_MISMATCH"
    REJECTED_SECRET_DETECTED = "REJECTED_SECRET_DETECTED"


@dataclass(frozen=True, slots=True)
class QualityValidationResult:
    """Detailed diagnosis of an event validation cycle."""
    is_eligible: bool
    status: EventEligibility
    reason: str
    cleaned_fields: dict[str, Any] | None = None


class EventQualityValidator:
    """Validates raw and normalized market events for research ingestion."""

    @staticmethod
    def validate_event(event: Any) -> QualityValidationResult:
        """Evaluate raw dictionary, MarketTickEvent, or MarketBar for eligibility."""
        # 1. Check for sensitive credential leakage
        if isinstance(event, Mapping):
            try:
                validate_no_secrets(dict(event), "market_event")
            except ContractValidationError as e:
                return QualityValidationResult(
                    is_eligible=False,
                    status=EventEligibility.REJECTED_SECRET_DETECTED,
                    reason=str(e),
                )

        # 2. Extract fields based on object type
        event_type = None
        topic = None
        symbol = None
        open_p = None
        high_p = None
        low_p = None
        close_p = None
        volume = None
        market_date_str = None
        source_ts = None

        if isinstance(event, Mapping):
            event_type = event.get("event_type") or event.get("type")
            topic = event.get("topic") or "market.ticks"
            symbol = event.get("symbol") or event.get("ticker")
            open_p = event.get("open")
            high_p = event.get("high")
            low_p = event.get("low")
            close_p = event.get("close")
            volume = event.get("volume", 0.0)
            market_date_str = event.get("simulated_date") or event.get("date")
            source_ts = event.get("timestamp") or event.get("source_timestamp")
        elif hasattr(event, "symbol"):  # MarketTickEvent from kafka package
            event_type = getattr(event, "extra", {}).get("event_type")
            meta = getattr(event, "metadata", None)
            topic = getattr(meta, "topic", "market.ticks") if meta is not None else "market.ticks"
            symbol = getattr(event, "symbol")
            open_p = getattr(event, "open")
            high_p = getattr(event, "high")
            low_p = getattr(event, "low")
            close_p = getattr(event, "close")
            volume = getattr(event, "volume", 0.0)
            s_date = getattr(event, "simulated_date", None)
            market_date_str = s_date.isoformat() if hasattr(s_date, "isoformat") else str(s_date) if s_date else None
            source_ts = getattr(event, "timestamp", None)
        elif hasattr(event, "ticker"):  # MarketBar from contracts
            topic = "market.ticks"
            symbol = getattr(event, "ticker")
            open_p = getattr(event, "open")
            high_p = getattr(event, "high")
            low_p = getattr(event, "low")
            close_p = getattr(event, "close")
            volume = getattr(event, "volume", 0.0)
            source_ts = getattr(event, "source_timestamp", None)
            if source_ts:
                market_date_str = source_ts.date().isoformat()
        else:
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_MALFORMED,
                reason=f"Unsupported event payload type: {type(event)}",
            )

        # 3. Check event type discrimination
        if event_type == "preflight_tick":
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.EXCLUDED_PREFLIGHT,
                reason="Event is a preflight standby ping, not an OHLCV price bar",
            )
        if topic == "market.news" or event_type in ("news", "corporate_news"):
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.EXCLUDED_NEWS,
                reason="News events are excluded from the quantitative OHLCV dataset",
            )

        # 4. Mandatory symbol check
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_MALFORMED,
                reason="Missing or blank ticker symbol",
            )

        # 5. Price numeric validation
        if open_p is None or high_p is None or low_p is None or close_p is None:
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_MALFORMED,
                reason="One or more OHLC price fields are missing (None)",
            )

        assert open_p is not None and high_p is not None and low_p is not None and close_p is not None
        try:
            o = float(open_p)
            h = float(high_p)
            l = float(low_p)
            c = float(close_p)
            v = float(volume) if volume is not None else 0.0
        except (ValueError, TypeError) as e:
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_MALFORMED,
                reason=f"Non-numeric price or volume encountered: {e}",
            )

        for name, val in [("open", o), ("high", h), ("low", l), ("close", c)]:
            if math.isnan(val) or math.isinf(val) or val <= 0.0:
                return QualityValidationResult(
                    is_eligible=False,
                    status=EventEligibility.REJECTED_NEGATIVE_PRICE,
                    reason=f"Price '{name}' must be strictly positive and finite, got {val}",
                )

        if math.isnan(v) or math.isinf(v) or v < 0.0:
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_MALFORMED,
                reason=f"Volume must be non-negative and finite, got {v}",
            )

        # 6. OHLC candle logic
        if l > h:
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_OHLC_MISMATCH,
                reason=f"Low ({l}) exceeds High ({h})",
            )
        if o < l or o > h or c < l or c > h:
            return QualityValidationResult(
                is_eligible=False,
                status=EventEligibility.REJECTED_OHLC_MISMATCH,
                reason=f"Open ({o}) or Close ({c}) outside [Low, High] = [{l}, {h}]",
            )

        # 7. Date resolution
        if not market_date_str:
            if isinstance(source_ts, datetime):
                market_date_str = source_ts.date().isoformat()
            elif isinstance(source_ts, str):
                try:
                    market_date_str = datetime.fromisoformat(source_ts).date().isoformat()
                except Exception:
                    market_date_str = source_ts[:10]
            else:
                return QualityValidationResult(
                    is_eligible=False,
                    status=EventEligibility.REJECTED_MALFORMED,
                    reason="Missing market date or timestamp",
                )

        cleaned = {
            "symbol": symbol.upper().strip(),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "market_date": market_date_str[:10],
            "source_timestamp": source_ts.isoformat() if hasattr(source_ts, "isoformat") else str(source_ts) if source_ts else None,
            "topic": topic,
        }

        return QualityValidationResult(
            is_eligible=True,
            status=EventEligibility.ELIGIBLE,
            reason="Validated OHLCV observation",
            cleaned_fields=cleaned,
        )


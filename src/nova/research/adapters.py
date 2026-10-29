"""NOVA Research Adapters — MarketBar Validation, CSV Loading, Event Translation & Synthetic Generation.

Ensures strict point-in-time input validation, standardizes ingestion payloads into
the canonical MarketBar contract, and provides deterministic offline synthetic data.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone, timedelta
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    NumericalValidationError,
    TemporalValidationError,
    TimestampSemantics,
    validate_finite,
    validate_timezone_aware,
)


def validate_market_bar_sequence(bars: Sequence[MarketBar]) -> None:
    """Validate chronological monotonicity, uniqueness, and ticker consistency.

    Invariants:
    1. Sequence must be non-empty.
    2. All bars must have the same ticker.
    3. Timestamps must be strictly monotonically increasing (no duplicates, no backwards steps).
    4. Each bar must satisfy its own internal MarketBar invariants.
    """
    if not bars:
        raise ContractValidationError("MarketBar sequence cannot be empty")

    base_ticker = bars[0].ticker
    prev_time: datetime | None = None

    for i, bar in enumerate(bars):
        if not isinstance(bar, MarketBar):
            raise ContractValidationError(f"Item at index {i} must be a MarketBar, got {type(bar)}")

        if bar.ticker != base_ticker:
            raise ContractValidationError(
                f"Cross-ticker mixing detected at index {i}: expected '{base_ticker}', got '{bar.ticker}'"
            )

        if prev_time is not None:
            if bar.source_timestamp == prev_time:
                raise TemporalValidationError(
                    f"Duplicate timestamp detected at index {i}: {bar.source_timestamp.isoformat()}"
                )
            if bar.source_timestamp < prev_time:
                raise TemporalValidationError(
                    f"Chronological ordering violation at index {i}: {bar.source_timestamp.isoformat()} < {prev_time.isoformat()}"
                )

        prev_time = bar.source_timestamp


def parse_timestamp_iso(ts_str: str) -> datetime:
    """Parse an ISO format timestamp and ensure timezone awareness (defaults to UTC)."""
    if not isinstance(ts_str, str) or not ts_str.strip():
        raise TemporalValidationError("Timestamp string cannot be empty")

    clean_str = ts_str.strip()
    # Replace trailing 'Z' with '+00:00' for compatibility across Python versions
    if clean_str.endswith("Z"):
        clean_str = clean_str[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(clean_str)
    except ValueError as exc:
        raise TemporalValidationError(f"Invalid timestamp format: '{ts_str}'") from exc

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        # Default to UTC if not specified, preserving timezone-awareness invariant
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def load_market_bars_from_csv(
    file_path: str | Path,
    ticker: str | None = None,
    timestamp_semantics: TimestampSemantics = TimestampSemantics.CLOSE_TIME,
) -> list[MarketBar]:
    """Load and validate MarketBar objects from a CSV file.

    Expected CSV header columns (case-insensitive):
    - timestamp (or time, datetime, date)
    - open
    - high
    - low
    - close
    - volume
    - ticker (optional if passed as argument)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Market data file not found: {path}")

    bars: list[MarketBar] = []
    with path.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ContractValidationError(f"CSV file '{path}' is empty or malformed")

        # Normalize field names to lowercase
        field_map = {fn.strip().lower(): fn for fn in reader.fieldnames if fn}

        # Validate required columns
        ts_col = next((field_map[k] for k in ["timestamp", "time", "datetime", "date"] if k in field_map), None)
        if not ts_col:
            raise ContractValidationError("Missing timestamp column in CSV (expected 'timestamp', 'time', or 'date')")

        for req in ["open", "high", "low", "close", "volume"]:
            if req not in field_map:
                raise ContractValidationError(f"Missing required column '{req}' in CSV")

        ticker_col = field_map.get("ticker") or field_map.get("symbol")

        for line_num, row in enumerate(reader, start=2):
            raw_ts = row.get(ts_col)
            if not raw_ts:
                raise ContractValidationError(f"Line {line_num}: Empty timestamp")

            bar_ts = parse_timestamp_iso(raw_ts)

            row_ticker = row.get(ticker_col) if ticker_col else None
            bar_ticker = (row_ticker.strip() if row_ticker else None) or ticker
            if not bar_ticker:
                raise ContractValidationError(f"Line {line_num}: Ticker not specified in row or parameter")

            try:
                op = float(row[field_map["open"]])
                hi = float(row[field_map["high"]])
                lo = float(row[field_map["low"]])
                cl = float(row[field_map["close"]])
                vol = float(row[field_map["volume"]])
            except (ValueError, TypeError) as exc:
                raise NumericalValidationError(f"Line {line_num}: Invalid numeric price or volume") from exc

            bar = MarketBar(
                ticker=bar_ticker,
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=vol,
                source_timestamp=bar_ts,
                available_timestamp=bar_ts,
                timestamp_semantics=timestamp_semantics,
                data_quality=DataQualityStatus.VALID,
                provenance=str(path.name),
            )
            bars.append(bar)

    validate_market_bar_sequence(bars)
    return bars


def parse_simulated_date_to_timestamp(value: Any) -> datetime:
    """Parse supported simulated_date formats and derive a UTC timestamp.

    Supported formats:
    - datetime.date (converted to UTC midnight)
    - datetime.datetime (converted or set to UTC)
    - ISO-8601 string ('YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS', etc.)

    Raises:
        TemporalValidationError: If value is empty, invalid, unparseable, or unsupported.
    """
    if value is None:
        raise TemporalValidationError("simulated_date cannot be None")

    if isinstance(value, datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)

    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            raise TemporalValidationError("simulated_date string cannot be empty")
        clean_str = val_str
        if clean_str.endswith("Z"):
            clean_str = clean_str[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(clean_str)
        except (ValueError, TypeError) as exc:
            raise TemporalValidationError(f"Invalid simulated_date format: '{value}'") from exc

        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    raise TemporalValidationError(
        f"Unsupported simulated_date type: {type(value).__name__} (expected date, datetime, or ISO string)"
    )


def market_tick_to_market_bar(
    tick: Any,
    available_timestamp: datetime | None = None,
    provenance: str = "kafka.market.ticks",
) -> MarketBar:
    """Adapter converting Person 1's MarketTickEvent to canonical MarketBar.

    Maintains loose coupling with duck-typing support.
    """
    symbol = getattr(tick, "symbol", None)
    if not symbol and isinstance(tick, dict):
        symbol = tick.get("symbol") or tick.get("ticker")
    if not symbol:
        raise ContractValidationError("MarketTick has no symbol")

    ts = getattr(tick, "timestamp", None)
    if ts is None and isinstance(tick, dict):
        ts = tick.get("timestamp")

    if ts is not None:
        if isinstance(ts, str):
            ts = parse_timestamp_iso(ts)
        elif isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
        elif isinstance(ts, (int, float)):
            ts_num = float(ts)
            try:
                if ts_num > 1e14:
                    ts = datetime.fromtimestamp(ts_num / 1e6, tz=timezone.utc)
                elif ts_num > 1e11:
                    ts = datetime.fromtimestamp(ts_num / 1e3, tz=timezone.utc)
                else:
                    ts = datetime.fromtimestamp(ts_num, tz=timezone.utc)
            except (ValueError, OverflowError, OSError) as exc:
                raise TemporalValidationError(f"Invalid numeric timestamp: {ts}") from exc
        else:
            raise TemporalValidationError(f"Unsupported timestamp type: {type(ts).__name__}")
    else:
        # Fallback to simulated_date
        sim_date = getattr(tick, "simulated_date", None)
        if sim_date is None and isinstance(tick, dict):
            sim_date = tick.get("simulated_date")
        if sim_date is None:
            raise TemporalValidationError("MarketTick has no timestamp or simulated_date")
        ts = parse_simulated_date_to_timestamp(sim_date)

    avail_ts = available_timestamp or ts
    if avail_ts.tzinfo is None:
        avail_ts = avail_ts.replace(tzinfo=timezone.utc)
    else:
        avail_ts = avail_ts.astimezone(timezone.utc)

    open_val = tick["open"] if isinstance(tick, dict) else tick.open
    high_val = tick["high"] if isinstance(tick, dict) else tick.high
    low_val = tick["low"] if isinstance(tick, dict) else tick.low
    close_val = tick["close"] if isinstance(tick, dict) else tick.close
    vol_val = tick["volume"] if isinstance(tick, dict) else tick.volume

    return MarketBar(
        ticker=symbol,
        open=float(open_val),
        high=float(high_val),
        low=float(low_val),
        close=float(close_val),
        volume=float(vol_val),
        source_timestamp=ts,
        available_timestamp=avail_ts,
        timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        data_quality=DataQualityStatus.VALID,
        provenance=provenance,
    )


def generate_synthetic_ohlcv(
    ticker: str = "SYNTH_TREND",
    n_bars: int = 120,
    start_price: float = 100.0,
    base_timestamp: datetime | None = None,
    bar_interval_minutes: int = 5,
) -> list[MarketBar]:
    """Generate a deterministic synthetic OHLCV sequence for testing and demo execution.

    Simulates a distinct multi-phase price path (consolidation -> uptrend -> peak -> downtrend)
    to deterministically exercise both bullish and bearish crossover mechanics.
    """
    if n_bars < 30:
        raise ContractValidationError(f"n_bars ({n_bars}) must be at least 30 to test EMA crossovers")
    if start_price <= 0.0:
        raise NumericalValidationError(f"start_price must be > 0.0, got {start_price}")

    start_time = base_timestamp or datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    validate_timezone_aware(start_time, "base_timestamp")

    bars: list[MarketBar] = []
    current_close = start_price

    for i in range(n_bars):
        ts = start_time + timedelta(minutes=i * bar_interval_minutes)

        # Deterministic cyclical wave + regime drift
        # Phase 1: 0 to 20: gentle oscillation around start_price
        # Phase 2: 21 to 60: steady upward trend (+0.8% per bar)
        # Phase 3: 61 to 75: peak plateau oscillation
        # Phase 4: 76 to 120: downward trend (-0.6% per bar)
        if i < 20:
            pct_change = 0.002 * math.sin(i * 0.5)
        elif i < 60:
            pct_change = 0.008 + 0.003 * math.sin(i * 0.3)
        elif i < 75:
            pct_change = 0.001 * math.cos(i * 0.4)
        else:
            pct_change = -0.007 + 0.002 * math.sin(i * 0.4)

        open_price = current_close
        close_price = round(open_price * (1.0 + pct_change), 4)
        # Ensure OHLC logic
        high_price = round(max(open_price, close_price) * 1.002, 4)
        low_price = round(min(open_price, close_price) * 0.998, 4)
        volume = round(1000.0 + 500.0 * math.sin(i * 0.2) ** 2, 2)

        bar = MarketBar(
            ticker=ticker,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            source_timestamp=ts,
            available_timestamp=ts,
            timestamp_semantics=TimestampSemantics.CLOSE_TIME,
            data_quality=DataQualityStatus.VALID,
            provenance="SYNTHETIC_GENERATOR_DETERMINISTIC",
        )
        bars.append(bar)
        current_close = close_price

    validate_market_bar_sequence(bars)
    return bars


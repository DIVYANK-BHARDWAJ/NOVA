"""Tests for payload decoding, timestamp parsing, and message normalization."""

import json
from datetime import date, datetime, timezone
import pytest

from market_data.models import KafkaMetadata
from market_data.normalizer import (
    NormalizationError,
    decode_json_payload,
    normalize_market_news,
    normalize_market_tick,
    parse_date,
    parse_timestamp,
)


def test_decode_valid_json_bytes():
    """Verify UTF-8 bytes containing valid JSON object decode properly."""
    raw = json.dumps({"symbol": "TCS", "price": 3500.0}).encode("utf-8")
    result = decode_json_payload(raw)
    assert result == {"symbol": "TCS", "price": 3500.0}


def test_decode_malformed_json():
    """Verify invalid JSON raises NormalizationError without crashing."""
    raw = b'{"symbol": "TCS", "price": 3500.0'  # missing closing brace
    with pytest.raises(NormalizationError) as exc_info:
        decode_json_payload(raw)
    assert "Malformed JSON payload" in str(exc_info.value)


def test_decode_non_dict_json():
    """Verify JSON arrays or primitives raise NormalizationError."""
    raw = json.dumps([1, 2, 3]).encode("utf-8")
    with pytest.raises(NormalizationError) as exc_info:
        decode_json_payload(raw)
    assert "Expected JSON object (dict)" in str(exc_info.value)


def test_decode_invalid_utf8_bytes():
    """Verify non-UTF8 bytes raise NormalizationError."""
    raw = b"\xff\xfe\x00\x00"
    with pytest.raises(NormalizationError) as exc_info:
        decode_json_payload(raw)
    assert "Failed to decode message bytes as UTF-8" in str(exc_info.value)


def test_decode_empty_payloads():
    """Verify empty bytes and empty strings raise NormalizationError."""
    for empty_input in [b"", b"   ", "", "   "]:
        with pytest.raises(NormalizationError) as exc_info:
            decode_json_payload(empty_input)
        assert "Empty payload" in str(exc_info.value)


@pytest.mark.parametrize(
    "primitive_bytes,expected_type",
    [
        (b'"just a string"', "str"),
        (b"12345", "int"),
        (b"true", "bool"),
        (b"null", "NoneType"),
        (b"[1, 2, 3]", "list"),
    ],
)
def test_decode_non_dict_primitives(primitive_bytes, expected_type):
    """Verify non-object JSON payloads (string, int, bool, null, list) raise NormalizationError."""
    with pytest.raises(NormalizationError) as exc_info:
        decode_json_payload(primitive_bytes)
    assert f"Expected JSON object (dict), received {expected_type}" in str(exc_info.value)


def test_parse_timestamp_formats():
    """Verify timestamps in various candidate formats parse correctly to UTC datetime."""
    # 1. Seconds epoch
    dt1 = parse_timestamp(1700000000)
    assert dt1 == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    # 2. Milliseconds epoch
    dt2 = parse_timestamp(1700000000000)
    assert dt2 == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    # 3. Microseconds epoch
    dt3 = parse_timestamp(1700000000000000)
    assert dt3 == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    # 4. ISO-8601 string
    dt4 = parse_timestamp("2026-10-29T10:30:00Z")
    assert dt4 == datetime(2026, 10, 29, 10, 30, 0, tzinfo=timezone.utc)

    # 5. Missing / None
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None

    # 6. Unparseable string does not crash and does NOT fabricate a timestamp
    assert parse_timestamp("invalid_date_format") is None


def test_parse_date_formats_and_invalids():
    """Verify calendar date parsing handles dates, ISO strings, and invalid values safely."""
    # 1. Valid date string
    assert parse_date("2024-10-25") == date(2024, 10, 25)

    # 2. String with time part
    assert parse_date("2024-10-25T14:30:00Z") == date(2024, 10, 25)

    # 3. Already a date object
    d = date(2025, 11, 14)
    assert parse_date(d) == d

    # 4. Already a datetime object
    dt = datetime(2025, 11, 14, 10, 0, 0, tzinfo=timezone.utc)
    assert parse_date(dt) == date(2025, 11, 14)

    # 5. Missing / None / Empty
    assert parse_date(None) is None
    assert parse_date("") is None

    # 6. Invalid calendar dates
    assert parse_date("not-a-date") is None
    assert parse_date("2024-02-31") is None  # Invalid day in Feb
    assert parse_date(12345) is None  # Numeric not accepted for date


def test_normalize_market_tick_standard():
    """Verify standard candlestick payload normalizes to MarketTickEvent."""
    payload = {
        "symbol": "RELIANCE",
        "open": 2500.0,
        "high": 2550.0,
        "low": 2490.0,
        "close": 2530.0,
        "volume": 100000.0,
        "timestamp": 1700000000,
        "vwap": 2525.0,  # candidate extra field
        "exchange": "NSE",  # candidate extra field
    }
    meta = KafkaMetadata(topic="market.ticks", partition=0, offset=42)

    event = normalize_market_tick(payload, metadata=meta)

    assert event.symbol == "RELIANCE"
    assert event.open == 2500.0
    assert event.high == 2550.0
    assert event.low == 2490.0
    assert event.close == 2530.0
    assert event.volume == 100000.0
    assert event.timestamp is not None
    assert event.is_bullish is True
    assert event.metadata == meta
    assert event.raw_payload == payload
    # Verify extra candidate fields are preserved
    assert event.extra == {"vwap": 2525.0, "exchange": "NSE"}


def test_normalize_market_tick_candidate_aliases():
    """Verify key aliases (ticker, o, h, l, c, v) normalize correctly."""
    payload = {
        "ticker": "INFY",
        "o": 1400.0,
        "h": 1420.0,
        "l": 1395.0,
        "c": 1410.0,
        "v": 50000.0,
        "t": 1700000000000,
    }
    event = normalize_market_tick(payload)
    assert event.symbol == "INFY"
    assert event.open == 1400.0
    assert event.high == 1420.0
    assert event.low == 1395.0
    assert event.close == 1410.0
    assert event.volume == 50000.0
    assert event.timestamp is not None


def test_normalize_market_tick_missing_required_fields():
    """Verify missing OHLCV field raises NormalizationError."""
    payload = {
        "symbol": "TCS",
        "open": 3500.0,
        # missing high, low, close, volume
    }
    with pytest.raises(NormalizationError) as exc_info:
        normalize_market_tick(payload)
    assert "Missing required tick field 'high'" in str(exc_info.value)


def test_normalize_market_tick_non_numeric_price():
    """Verify non-numeric field value raises NormalizationError."""
    payload = {
        "symbol": "TCS",
        "open": "not_a_number",
        "high": 3550.0,
        "low": 3480.0,
        "close": 3520.0,
        "volume": 1000.0,
    }
    with pytest.raises(NormalizationError) as exc_info:
        normalize_market_tick(payload)
    assert "Invalid numeric value for 'open'" in str(exc_info.value)


def test_normalize_market_news_standard():
    """Verify corporate news payload normalizes to MarketNewsEvent."""
    payload = {
        "headline": "Q3 earnings exceed analyst forecasts by 15%",
        "symbol": "HDFCBANK",
        "timestamp": "2026-10-29T09:15:00Z",
        "source": "Bloomberg",
        "urgency": "HIGH",
    }
    meta = KafkaMetadata(topic="market.news", partition=1, offset=105)

    event = normalize_market_news(payload, metadata=meta)

    assert event.headline == "Q3 earnings exceed analyst forecasts by 15%"
    assert event.symbol == "HDFCBANK"
    assert event.timestamp is not None
    assert event.metadata == meta
    assert event.raw_payload == payload
    # Verify unmapped source metadata is preserved
    assert event.extra == {"source": "Bloomberg", "urgency": "HIGH"}


def test_normalize_market_news_missing_optional_symbol():
    """Verify news event without symbol preserves headline and unconfirmed structure."""
    payload = {
        "title": "RBI announces monetary policy committee schedule",
        "published_at": 1700000000,
    }
    event = normalize_market_news(payload)
    assert event.headline == "RBI announces monetary policy committee schedule"
    assert event.symbol is None
    assert event.timestamp is not None


def test_normalize_competition_nested_tick():
    """Verify MANTHAN 2026 nested 'data' payload structure normalizes properly."""
    payload = {
        "event_type": "price_tick",
        "timestamp": "2026-10-29T07:55:07.258Z",
        "simulated_date": "2024-10-25",
        "data": {
            "ticker": "ROUT",
            "open": 78.26,
            "high": 78.84,
            "low": 76.96,
            "close": 77.62,
            "volume": 11727400,
        },
        "sig": "285eefed9dedd90bcf31637160e47da148cb95aa8e7890fca31f4317df9a0c58",
        "sandbox": {"is_sandbox": True, "stage": "development"},
    }
    event = normalize_market_tick(payload)
    assert event.symbol == "ROUT"
    assert event.open == 78.26
    assert event.high == 78.84
    assert event.low == 76.96
    assert event.close == 77.62
    assert event.volume == 11727400.0
    assert event.timestamp == datetime(2026, 10, 29, 7, 55, 7, 258000, tzinfo=timezone.utc)
    assert event.simulated_date is not None
    assert str(event.simulated_date) == "2024-10-25"
    assert event.extra["event_type"] == "price_tick"


def test_normalize_competition_nested_news():
    """Verify MANTHAN 2026 nested 'data' news payload structure normalizes properly."""
    payload = {
        "event_type": "news_flash",
        "timestamp": "2026-10-29T07:55:36.757Z",
        "simulated_date": "2024-10-30",
        "data": {
            "ticker": "ROUT",
            "headline": "RouteWave drivers expand EV fleet across 10 cities",
            "description": "Company announces major expansion",
            "date": "2024-10-30",
        },
        "sandbox": {"is_sandbox": True},
    }
    event = normalize_market_news(payload)
    assert event.symbol == "ROUT"
    assert event.headline == "RouteWave drivers expand EV fleet across 10 cities"
    assert event.timestamp == datetime(2026, 10, 29, 7, 55, 36, 757000, tzinfo=timezone.utc)
    assert str(event.simulated_date) == "2024-10-30"
    assert str(event.news_date) == "2024-10-30"
    assert event.extra["event_type"] == "news_flash"
    assert event.extra["description"] == "Company announces major expansion"


def test_distinct_timestamp_and_simulated_date():
    """Verify event timestamp and simulation market date remain strictly separate."""
    payload = {
        "ticker": "ROUT",
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 50000.0,
        "timestamp": "2026-10-29T15:30:00Z",  # Feed timestamp
        "simulated_date": "2024-10-25",         # Historical simulation date
    }
    event = normalize_market_tick(payload)
    assert event.timestamp.year == 2026
    assert event.simulated_date.year == 2024
    assert event.timestamp.isoformat() != str(event.simulated_date)


def test_conflicting_numeric_aliases_rejected():
    """Verify conflicting numeric values between aliases (e.g. open vs o) raise NormalizationError."""
    payload = {
        "symbol": "ROUT",
        "open": 100.0,
        "o": 105.0,  # Conflicting with open
        "high": 110.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    }
    with pytest.raises(NormalizationError) as exc_info:
        normalize_market_tick(payload)
    assert "Conflicting numeric values for 'open'" in str(exc_info.value)


def test_conflicting_symbol_aliases_rejected():
    """Verify conflicting ticker values between aliases (symbol vs ticker) raise NormalizationError."""
    payload = {
        "symbol": "ROUT",
        "ticker": "DIFFERENT",  # Conflicting with symbol
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    }
    with pytest.raises(NormalizationError) as exc_info:
        normalize_market_tick(payload)
    assert "Conflicting values detected for 'symbol'" in str(exc_info.value)


def test_consistent_aliases_accepted():
    """Verify identical/consistent alias values are accepted without error."""
    payload = {
        "symbol": "ROUT",
        "ticker": "ROUT",  # Consistent with symbol
        "open": 100.0,
        "o": 100.0,       # Consistent with open
        "high": 110.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    }
    event = normalize_market_tick(payload)
    assert event.symbol == "ROUT"
    assert event.open == 100.0

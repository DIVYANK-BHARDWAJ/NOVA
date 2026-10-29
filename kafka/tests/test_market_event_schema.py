"""Focused contract validation tests for schemas/market_event.schema.json.

Ensures that the JSON Schema Draft-07 file is syntactically and semantically valid,
accurately models actual competition event formats (nested/flat price ticks,
nested/flat news flashes, and preflight/heartbeat pings without OHLCV data),
enforces required fields and type constraints, rejects malformed payloads,
and guarantees compatibility with Person 2's quantitative strategy contracts.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pytest

from market_data.models import KafkaMetadata, MarketTickEvent, MarketNewsEvent
from market_data.normalizer import normalize_market_tick, normalize_market_news
from nova.research.adapters import market_tick_to_market_bar
from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
from nova.quant.contracts import MarketBar, TemporalValidationError

SCHEMA_PATH = Path("schemas/market_event.schema.json")


def load_schema() -> dict[str, Any]:
    """Load and parse the JSON Schema file."""
    assert SCHEMA_PATH.is_file(), f"Schema file not found at {SCHEMA_PATH}"
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class SchemaValidator:
    """Lightweight Draft-07 JSON Schema validator for market events."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.definitions = schema.get("definitions", {})

    def resolve_ref(self, ref: str) -> dict[str, Any]:
        """Resolve a local #/definitions/Name JSON reference."""
        assert ref.startswith("#/definitions/"), f"Unsupported reference: {ref}"
        name = ref.split("/")[-1]
        assert name in self.definitions, f"Definition '{name}' not found in schema"
        return self.definitions[name]

    def validate_type(self, value: Any, expected_type: str | list[str]) -> bool:
        """Check JSON type compatibility."""
        types = [expected_type] if isinstance(expected_type, str) else expected_type
        for t in types:
            if t == "object" and isinstance(value, dict):
                return True
            if t == "array" and isinstance(value, list):
                return True
            if t == "string" and isinstance(value, str):
                return True
            if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
                return True
            if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if t == "boolean" and isinstance(value, bool):
                return True
            if t == "null" and value is None:
                return True
        return False

    def validate_subschema(self, payload: Any, subschema: dict[str, Any]) -> list[str]:
        """Validate payload against a subschema definition."""
        errors: list[str] = []

        if "$ref" in subschema:
            resolved = self.resolve_ref(subschema["$ref"])
            return self.validate_subschema(payload, resolved)

        # Check type
        if "type" in subschema:
            if not self.validate_type(payload, subschema["type"]):
                errors.append(f"Expected type {subschema['type']}, got {type(payload).__name__}")
                return errors

        # String validations
        if isinstance(payload, str):
            if "minLength" in subschema and len(payload) < subschema["minLength"]:
                errors.append(f"String '{payload}' is shorter than minLength {subschema['minLength']}")
            if "enum" in subschema and payload not in subschema["enum"]:
                errors.append(f"String '{payload}' not in enum {subschema['enum']}")

        # Numeric validations
        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            if "minimum" in subschema and payload < subschema["minimum"]:
                errors.append(f"Numeric value {payload} < minimum {subschema['minimum']}")
            if "exclusiveMinimum" in subschema and payload <= subschema["exclusiveMinimum"]:
                errors.append(f"Numeric value {payload} <= exclusiveMinimum {subschema['exclusiveMinimum']}")

        # Object validations
        if isinstance(payload, dict):
            # Check required
            for req in subschema.get("required", []):
                if req not in payload:
                    errors.append(f"Missing required property '{req}'")

            # Check properties
            props = subschema.get("properties", {})
            for k, val in payload.items():
                if k in props:
                    prop_errs = self.validate_subschema(val, props[k])
                    for pe in prop_errs:
                        errors.append(f"Property '{k}': {pe}")

            # Check anyOf
            if "anyOf" in subschema:
                any_ok = False
                for branch in subschema["anyOf"]:
                    if not self.validate_subschema(payload, branch):
                        any_ok = True
                        break
                if not any_ok:
                    errors.append("Failed anyOf constraint")

            # Check not
            if "not" in subschema:
                if not self.validate_subschema(payload, subschema["not"]):
                    errors.append("Matched prohibited 'not' schema condition")

        return errors

    def validate(self, payload: Any) -> list[str]:
        """Validate payload against root oneOf schema."""
        if not isinstance(payload, dict):
            return ["Payload must be a JSON object (dict)"]

        one_of = self.schema.get("oneOf", [])
        assert one_of, "Schema has no 'oneOf' variants"

        matching_variants: list[int] = []
        variant_errors: list[list[str]] = []

        for idx, variant in enumerate(one_of):
            errs = self.validate_subschema(payload, variant)
            if not errs:
                matching_variants.append(idx)
            variant_errors.append(errs)

        if len(matching_variants) == 1:
            return []
        elif len(matching_variants) == 0:
            return [f"Payload does not match any recognized market event variant. Errors: {variant_errors}"]
        else:
            return [f"Payload matches multiple conflicting variants ({matching_variants}). Ambiguous event."]


# ---------------------------------------------------------------------------
# Tests: 1. Schema Syntax and Draft Validation (Requirement 7.9)
# ---------------------------------------------------------------------------

def test_schema_syntax_and_draft():
    """Verify that the JSON schema file is syntactically valid JSON and Draft-07."""
    schema = load_schema()
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$id"] == "https://nova.manthan2026.internal/schemas/market_event.schema.json"
    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 5
    assert "definitions" in schema

    # Verify all expected definitions exist
    expected_defs = {
        "PriceTickNested",
        "NestedTickData",
        "PriceTickFlat",
        "NewsFlashNested",
        "NestedNewsData",
        "NewsFlashFlat",
        "PreflightHeartbeat",
        "NormalizedMarketTick",
        "NormalizedMarketNews",
        "KafkaMetadata",
    }
    for edef in expected_defs:
        assert edef in schema["definitions"], f"Missing expected definition: {edef}"


# ---------------------------------------------------------------------------
# Tests: 2. Valid Market Tick Examples Accepted (Requirement 7.1)
# ---------------------------------------------------------------------------

def test_valid_competition_nested_tick_accepted():
    """Verify MANTHAN 2026 competition nested price_tick payload is valid."""
    schema = load_schema()
    validator = SchemaValidator(schema)

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
    errors = validator.validate(payload)
    assert not errors, f"Expected valid tick payload, got: {errors}"


def test_valid_flat_tick_accepted():
    """Verify flat candlestick payload is accepted by canonical schema."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    payload = {
        "symbol": "RELIANCE",
        "open": 2500.0,
        "high": 2550.0,
        "low": 2490.0,
        "close": 2530.0,
        "volume": 100000.0,
        "timestamp": 1700000000,
    }
    errors = validator.validate(payload)
    assert not errors, f"Expected valid flat tick payload, got: {errors}"


# ---------------------------------------------------------------------------
# Tests: 3. Valid Market News Examples Accepted (Requirement 7.2)
# ---------------------------------------------------------------------------

def test_valid_competition_nested_news_accepted():
    """Verify MANTHAN 2026 competition nested news_flash payload is valid."""
    schema = load_schema()
    validator = SchemaValidator(schema)

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
    errors = validator.validate(payload)
    assert not errors, f"Expected valid nested news payload, got: {errors}"


def test_valid_flat_news_with_and_without_symbol_accepted():
    """Verify corporate news and macroeconomic news (symbol null) are valid."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    # Corporate news
    corp_news = {
        "headline": "Q3 earnings exceed analyst forecasts by 15%",
        "symbol": "HDFCBANK",
        "timestamp": "2026-10-29T09:15:00Z",
        "source": "Bloomberg",
        "urgency": "HIGH",
    }
    assert not validator.validate(corp_news)

    # Macroeconomic news (symbol is null)
    macro_news = {
        "headline": "RBI announces monetary policy committee schedule",
        "symbol": None,
        "timestamp": 1700000000,
    }
    assert not validator.validate(macro_news)


# ---------------------------------------------------------------------------
# Tests: 4. Valid Pre-flight / Heartbeat Examples Accepted (Requirement 7.3)
# ---------------------------------------------------------------------------

def test_valid_preflight_and_heartbeat_accepted():
    """Verify preflight events with sequence numbers are accepted."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    # 1. Direct sequence tick
    preflight1 = {
        "event_type": "preflight_tick",
        "seq": 54,
        "timestamp": "2026-03-25T10:00:00Z",
    }
    assert not validator.validate(preflight1)

    # 2. Standby preflight with nested status
    preflight2 = {
        "event_type": "preflight_tick",
        "seq": 1041,
        "timestamp": "2026-10-29T07:55:00.000Z",
        "data": {"status": "STANDBY"},
    }
    assert not validator.validate(preflight2)

    # 3. Heartbeat event
    heartbeat = {
        "event_type": "heartbeat",
        "seq": 99,
        "timestamp": 1700000000000,
    }
    assert not validator.validate(heartbeat)


# ---------------------------------------------------------------------------
# Tests: 5. Required Fields Enforced for Each Event Family (Requirement 7.4)
# ---------------------------------------------------------------------------

def test_required_fields_enforced_for_ticks():
    """Verify tick payloads missing required OHLCV or ticker are rejected."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    # Missing close price in nested data
    bad_nested_tick = {
        "event_type": "price_tick",
        "data": {
            "ticker": "ROUT",
            "open": 78.26,
            "high": 78.84,
            "low": 76.96,
            # missing close
        },
    }
    assert len(validator.validate(bad_nested_tick)) > 0

    # Missing open price in flat tick
    bad_flat_tick = {
        "symbol": "TCS",
        "high": 3550.0,
        "low": 3480.0,
        "close": 3520.0,
        "volume": 1000.0,
    }
    assert len(validator.validate(bad_flat_tick)) > 0


def test_required_fields_enforced_for_news():
    """Verify news payload missing headline is rejected."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    bad_news = {
        "symbol": "INFY",
        "timestamp": "2026-10-29T10:00:00Z",
        # missing headline
    }
    assert len(validator.validate(bad_news)) > 0


# ---------------------------------------------------------------------------
# Tests: 6. Invalid Field Types are Rejected (Requirement 7.5)
# ---------------------------------------------------------------------------

def test_invalid_field_types_rejected():
    """Verify string price, negative price, and negative volume are rejected."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    # Non-numeric price string
    bad_type = {
        "event_type": "price_tick",
        "data": {
            "ticker": "ROUT",
            "open": "seventy-eight",
            "high": 80.0,
            "low": 75.0,
            "close": 78.0,
            "volume": 100,
        },
    }
    assert len(validator.validate(bad_type)) > 0

    # Negative price (exclusiveMinimum: 0)
    negative_price = {
        "symbol": "ROUT",
        "open": -10.0,
        "high": 80.0,
        "low": 75.0,
        "close": 78.0,
        "volume": 100,
    }
    assert len(validator.validate(negative_price)) > 0

    # Negative volume (minimum: 0)
    negative_vol = {
        "symbol": "ROUT",
        "open": 78.0,
        "high": 80.0,
        "low": 75.0,
        "close": 78.0,
        "volume": -50.0,
    }
    assert len(validator.validate(negative_vol)) > 0


# ---------------------------------------------------------------------------
# Tests: 7. Malformed Events are Rejected (Requirement 7.6)
# ---------------------------------------------------------------------------

def test_malformed_events_rejected():
    """Verify empty objects, primitives, and unrecognized structures are rejected."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    assert len(validator.validate({})) > 0
    assert len(validator.validate("just a string")) > 0
    assert len(validator.validate([1, 2, 3])) > 0
    assert len(validator.validate({"unknown_key": "some_value"})) > 0


# ---------------------------------------------------------------------------
# Tests: 8. Pre-flight Events Without OHLCV Not Rejected (Requirement 7.7)
# ---------------------------------------------------------------------------

def test_preflight_without_ohlcv_accepted():
    """Verify preflight events are NOT rejected merely because they lack OHLCV data."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    # Preflight event has no open, high, low, close, volume
    preflight = {
        "event_type": "preflight_tick",
        "seq": 42,
        "timestamp": "2026-10-29T10:00:00Z",
    }
    errors = validator.validate(preflight)
    assert not errors, f"Preflight event without OHLCV should be valid, got: {errors}"


# ---------------------------------------------------------------------------
# Tests: 9. Person 2 Strategy Inputs Remain Compatible (Requirement 7.8)
# ---------------------------------------------------------------------------

def test_person2_strategy_compatibility():
    """Verify valid schema market events normalize and adapt to Person 2 contracts."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    raw_payload = {
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
    }

    # 1. Raw event validates against canonical schema
    assert not validator.validate(raw_payload)

    # 2. Ingestion normalizer produces MarketTickEvent
    event = normalize_market_tick(raw_payload)
    assert isinstance(event, MarketTickEvent)
    assert event.symbol == "ROUT"
    assert event.close == 77.62

    # 3. Person 2 research adapter converts MarketTickEvent to MarketBar
    market_bar = market_tick_to_market_bar(event)
    assert isinstance(market_bar, MarketBar)
    assert market_bar.ticker == "ROUT"
    assert market_bar.close == 77.62
    assert market_bar.source_timestamp.tzinfo is not None

    # 4. Strategy evaluation executes without error on sequential bars
    bars = []
    for i in range(10):
        tick = MarketTickEvent(
            symbol="ROUT",
            open=78.0 + i * 0.1,
            high=78.5 + i * 0.1,
            low=77.5 + i * 0.1,
            close=78.2 + i * 0.1,
            volume=1000.0,
            timestamp=market_bar.source_timestamp + timedelta(minutes=i * 5),
        )
        bars.append(market_tick_to_market_bar(tick))

    strategy = EMACrossoverStrategy(config=EMACrossoverConfig(fast_window=2, slow_window=4))
    eval_res = strategy.evaluate(bars)
    assert eval_res.status.value in {"CANDIDATE_PRODUCED", "ABSTAIN_INSUFFICIENT_HISTORY", "ABSTAIN_MARKET_CONDITIONS"}


# ---------------------------------------------------------------------------
# Tests: 10. Rejection of Incompatible/Conflicting Formats (Requirement 7.10)
# ---------------------------------------------------------------------------

def test_schema_rejects_conflicting_or_incompatible_formats():
    """Verify that malformed or ambiguous payloads are rejected."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    # Tick event containing empty string ticker
    empty_ticker = {
        "symbol": "   ",
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    }
    # Note: minLength is 1; empty string stripped is invalid
    bad_empty_str = {
        "symbol": "",
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    }
    assert len(validator.validate(bad_empty_str)) > 0

    # News event with empty headline
    empty_headline = {
        "headline": "",
        "symbol": "ROUT",
    }
    assert len(validator.validate(empty_headline)) > 0

    # Unrecognized event type with random payload
    unrecognized = {
        "event_type": "mystery_event",
        "arbitrary_field": 12345,
    }
    assert len(validator.validate(unrecognized)) > 0


# ---------------------------------------------------------------------------
# Tests: 11. Normalized Market Event Definitions Match Domain Models
# ---------------------------------------------------------------------------

def test_normalized_definitions_match_models():
    """Verify schema definitions for NormalizedMarketTick and NormalizedMarketNews match domain models."""
    schema = load_schema()
    validator = SchemaValidator(schema)

    tick_def = schema["definitions"]["NormalizedMarketTick"]
    news_def = schema["definitions"]["NormalizedMarketNews"]

    # Verify NormalizedMarketTick required fields match MarketTickEvent
    assert set(tick_def["required"]) == {"symbol", "open", "high", "low", "close", "volume"}

    # Test normalized tick dictionary validates against definition
    meta = KafkaMetadata(topic="market.ticks", partition=0, offset=123)
    norm_tick = MarketTickEvent(
        symbol="ROUT",
        open=78.26,
        high=78.84,
        low=76.96,
        close=77.62,
        volume=11727400.0,
        timestamp=datetime(2026, 10, 29, 7, 55, 7, tzinfo=timezone.utc),
        metadata=meta,
    )
    norm_dict = norm_tick.to_dict()
    errors = validator.validate_subschema(norm_dict, tick_def)
    assert not errors, f"NormalizedMarketTick validation failed: {errors}"

    # Verify NormalizedMarketNews required fields match MarketNewsEvent
    assert set(news_def["required"]) == {"headline"}
    norm_news = MarketNewsEvent(
        headline="RouteWave expansion",
        symbol="ROUT",
        timestamp=datetime(2026, 10, 29, 7, 55, 36, tzinfo=timezone.utc),
        metadata=meta,
    )
    news_dict = norm_news.to_dict()
    errors = validator.validate_subschema(news_dict, news_def)
    assert not errors, f"NormalizedMarketNews validation failed: {errors}"


# ---------------------------------------------------------------------------
# Tests: 12. Simulated Date Fallback in Research Adapter (Regression Tests)
# ---------------------------------------------------------------------------

def test_market_tick_to_market_bar_with_simulated_date_fallback():
    """Verify market_tick_to_market_bar parses simulated_date when timestamp=None."""
    # 1. date object fallback
    tick_date = MarketTickEvent(
        symbol="ROUT",
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=50000.0,
        timestamp=None,
        simulated_date=date(2026, 3, 1),
    )
    bar_date = market_tick_to_market_bar(tick_date)
    assert bar_date.ticker == "ROUT"
    assert bar_date.source_timestamp == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert bar_date.available_timestamp == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    # 2. ISO string YYYY-MM-DD
    tick_str = MarketTickEvent(
        symbol="INFY",
        open=1500.0,
        high=1520.0,
        low=1490.0,
        close=1510.0,
        volume=20000.0,
        timestamp=None,
        simulated_date="2026-03-05",
    )
    bar_str = market_tick_to_market_bar(tick_str)
    assert bar_str.ticker == "INFY"
    assert bar_str.source_timestamp == datetime(2026, 3, 5, 0, 0, 0, tzinfo=timezone.utc)

    # 3. ISO string with time component
    tick_iso = MarketTickEvent(
        symbol="TCS",
        open=3500.0,
        high=3550.0,
        low=3480.0,
        close=3520.0,
        volume=15000.0,
        timestamp=None,
        simulated_date="2026-03-10T09:15:00Z",
    )
    bar_iso = market_tick_to_market_bar(tick_iso)
    assert bar_iso.source_timestamp == datetime(2026, 3, 10, 9, 15, 0, tzinfo=timezone.utc)

    # 4. Explicit timestamp takes precedence over simulated_date
    explicit_ts = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
    tick_explicit = MarketTickEvent(
        symbol="ROUT",
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=50000.0,
        timestamp=explicit_ts,
        simulated_date=date(2026, 3, 1),
    )
    bar_explicit = market_tick_to_market_bar(tick_explicit)
    assert bar_explicit.source_timestamp == explicit_ts

    # 5. Duck-typed dictionary with simulated_date
    dict_tick = {
        "symbol": "WIPRO",
        "open": 450.0,
        "high": 460.0,
        "low": 445.0,
        "close": 455.0,
        "volume": 12000.0,
        "timestamp": None,
        "simulated_date": "2026-03-20",
    }
    bar_dict = market_tick_to_market_bar(dict_tick)
    assert bar_dict.ticker == "WIPRO"
    assert bar_dict.source_timestamp == datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)


def test_market_tick_to_market_bar_rejects_missing_and_invalid_simulated_date():
    """Verify rejection of absent or malformed simulated_date when timestamp is missing."""
    # 1. Missing both timestamp and simulated_date
    tick_missing = MarketTickEvent(
        symbol="ROUT",
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=50000.0,
        timestamp=None,
        simulated_date=None,
    )
    with pytest.raises(TemporalValidationError) as exc1:
        market_tick_to_market_bar(tick_missing)
    assert "no timestamp or simulated_date" in str(exc1.value)

    # 2. Empty string simulated_date
    tick_empty = MarketTickEvent(
        symbol="ROUT",
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=50000.0,
        timestamp=None,
        simulated_date="   ",
    )
    with pytest.raises(TemporalValidationError) as exc2:
        market_tick_to_market_bar(tick_empty)
    assert "cannot be empty" in str(exc2.value)

    # 3. Malformed string simulated_date
    tick_invalid = MarketTickEvent(
        symbol="ROUT",
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=50000.0,
        timestamp=None,
        simulated_date="not-a-valid-date",
    )
    with pytest.raises(TemporalValidationError) as exc3:
        market_tick_to_market_bar(tick_invalid)
    assert "Invalid simulated_date format" in str(exc3.value)

    # 4. Unsupported data type
    tick_bad_type = MarketTickEvent(
        symbol="ROUT",
        open=100.0,
        high=105.0,
        low=98.0,
        close=102.0,
        volume=50000.0,
        timestamp=None,
        simulated_date={"year": 2026},
    )
    with pytest.raises(TemporalValidationError) as exc4:
        market_tick_to_market_bar(tick_bad_type)
    assert "Unsupported simulated_date type" in str(exc4.value)

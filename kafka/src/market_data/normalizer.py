"""Deterministic parsing and normalization of Kafka payloads for MANTHAN 2026.

Converts raw Kafka message bytes or dictionaries into strategy-independent
MarketTickEvent and MarketNewsEvent objects. Accommodates candidate schema
variations while preserving unmapped fields, distinct timestamps, and Kafka metadata.
Detects conflicting aliases rather than silently picking arbitrary values.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Optional

from market_data.models import KafkaMetadata, MarketNewsEvent, MarketTickEvent


class NormalizationError(ValueError):
    """Raised when a message cannot be parsed, decoded, or normalized."""
    pass


# Candidate key aliases (checked in order of preference)
TICK_SYMBOL_KEYS = ("symbol", "ticker", "stock", "instrument", "sec_id", "scrip")
TICK_OPEN_KEYS = ("open", "o", "open_price")
TICK_HIGH_KEYS = ("high", "h", "high_price")
TICK_LOW_KEYS = ("low", "l", "low_price")
TICK_CLOSE_KEYS = ("close", "c", "close_price", "last_price")
TICK_VOLUME_KEYS = ("volume", "v", "vol", "qty", "volume_traded")
TIMESTAMP_KEYS = ("timestamp", "time", "ts", "datetime", "date_time", "published_at", "published", "created_at", "t")
SIMULATED_DATE_KEYS = ("simulated_date", "sim_date", "market_date")
NEWS_DATE_KEYS = ("date", "news_date", "published_date", "pub_date")

NEWS_HEADLINE_KEYS = ("headline", "title", "news", "text", "content", "summary", "announcement")
NEWS_SYMBOL_KEYS = ("symbol", "ticker", "company", "company_id", "stock")


def _find_field_with_conflict_check(
    data: dict[str, Any],
    keys: tuple[str, ...],
    field_name: str,
    numeric: bool = False,
) -> tuple[Optional[str], Any, set[str]]:
    """Find matching field alias with conflict detection.

    If multiple aliases from `keys` are present in `data`, verifies that their
    values are consistent. Raises NormalizationError on conflicting values.

    Returns:
        tuple of (canonical key name, value, set of all matched keys consumed).
    """
    matched: list[tuple[str, Any]] = []
    lower_map = {k.lower(): k for k in data.keys()}

    for k in keys:
        if k in data:
            matched.append((k, data[k]))
        elif k.lower() in lower_map:
            actual_key = lower_map[k.lower()]
            if (actual_key, data[actual_key]) not in matched:
                matched.append((actual_key, data[actual_key]))

    if not matched:
        return None, None, set()

    all_matched_keys = {m[0] for m in matched}
    primary_key, primary_val = matched[0]

    # Check for conflicts if more than one alias is present
    if len(matched) > 1:
        for other_key, other_val in matched[1:]:
            if other_val is None and primary_val is None:
                continue
            if other_val is None or primary_val is None:
                raise NormalizationError(
                    f"Conflicting aliases detected for '{field_name}': "
                    f"'{primary_key}'={primary_val!r} vs '{other_key}'={other_val!r}"
                )

            if numeric:
                try:
                    num1 = float(primary_val)
                    num2 = float(other_val)
                    if abs(num1 - num2) > 1e-4:
                        raise NormalizationError(
                            f"Conflicting numeric values for '{field_name}': "
                            f"'{primary_key}'={primary_val} vs '{other_key}'={other_val}"
                        )
                except (ValueError, TypeError) as exc:
                    raise NormalizationError(
                        f"Non-numeric value encountered while comparing aliases for '{field_name}': {exc}"
                    ) from exc
            else:
                str1 = str(primary_val).strip()
                str2 = str(other_val).strip()
                if str1.upper() != str2.upper():
                    raise NormalizationError(
                        f"Conflicting values detected for '{field_name}': "
                        f"'{primary_key}'={primary_val!r} vs '{other_key}'={other_val!r}"
                    )

    return primary_key, primary_val, all_matched_keys


def parse_date(value: Any) -> Optional[date]:
    """Safely parse a calendar date string (e.g. '2024-10-25') without fabricating values.

    Returns:
        datetime.date if successfully parsed, or None.
    """
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return None
        # Handle 'YYYY-MM-DD' prefix
        date_part = val_str.split("T")[0].split(" ")[0]
        try:
            return date.fromisoformat(date_part)
        except ValueError:
            pass

    return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Safely parse timestamps without fabricating values.

    Supports:
    - POSIX seconds, milliseconds, or microseconds (int / float / numeric string)
    - ISO-8601 formatted strings (e.g. '2026-10-29T10:00:00Z', '2026-10-29 10:00:00')
    - Already parsed datetime objects

    Returns:
        datetime in UTC if successfully parsed, or None if missing or unparseable.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    # Numeric timestamp (int, float, or string representing number)
    if isinstance(value, (int, float)):
        ts_num = float(value)
        try:
            if ts_num > 1e14:
                return datetime.fromtimestamp(ts_num / 1e6, tz=timezone.utc)
            elif ts_num > 1e11:
                return datetime.fromtimestamp(ts_num / 1e3, tz=timezone.utc)
            else:
                return datetime.fromtimestamp(ts_num, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None

    if isinstance(value, str):
        value_str = value.strip()
        if not value_str:
            return None

        # Check if numeric string
        try:
            ts_num = float(value_str)
            if ts_num > 1e14:
                return datetime.fromtimestamp(ts_num / 1e6, tz=timezone.utc)
            elif ts_num > 1e11:
                return datetime.fromtimestamp(ts_num / 1e3, tz=timezone.utc)
            else:
                return datetime.fromtimestamp(ts_num, tz=timezone.utc)
        except (ValueError, OverflowError):
            pass

        # Try ISO format
        try:
            clean_str = value_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    return None


def decode_json_payload(raw_bytes: bytes | str) -> dict[str, Any]:
    """Safely decode and parse JSON payload.

    Args:
        raw_bytes: Raw bytes or string from Kafka message.

    Returns:
        Parsed dictionary.

    Raises:
        NormalizationError: If message cannot be decoded or is not a JSON object.
    """
    if isinstance(raw_bytes, bytes):
        if not raw_bytes or not raw_bytes.strip():
            raise NormalizationError("Empty payload bytes received")
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NormalizationError(f"Failed to decode message bytes as UTF-8: {exc}") from exc
    elif isinstance(raw_bytes, str):
        if not raw_bytes or not raw_bytes.strip():
            raise NormalizationError("Empty payload string received")
        text = raw_bytes
    else:
        raise NormalizationError(f"Unsupported payload type: {type(raw_bytes).__name__}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NormalizationError(f"Malformed JSON payload: {exc.msg} at line {exc.lineno} col {exc.colno}") from exc

    if not isinstance(data, dict):
        raise NormalizationError(f"Expected JSON object (dict), received {type(data).__name__}")

    return data


def _extract_search_context(payload: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Unpack nested 'data', 'payload', or 'record' objects while preserving top-level attributes.

    Returns:
        tuple of (combined search dictionary, set of nested container key names).
    """
    combined: dict[str, Any] = dict(payload)
    nested_containers: set[str] = set()

    for container_key in ("data", "payload", "record"):
        nested_val = payload.get(container_key)
        if isinstance(nested_val, dict):
            nested_containers.add(container_key)
            for k, v in nested_val.items():
                combined[k] = v

    return combined, nested_containers


def normalize_market_tick(
    payload: dict[str, Any],
    metadata: Optional[KafkaMetadata] = None,
) -> MarketTickEvent:
    """Normalize a raw dictionary payload into a MarketTickEvent.

    Extracts OHLCV candidate fields, parses timestamp, extracts simulated_date separately,
    detects conflicting aliases, preserves unmapped fields in 'extra', and retains the
    original payload and transport metadata. Supports both flat and nested 'data' schemas.

    Args:
        payload: Decoded JSON dictionary.
        metadata: Optional Kafka message metadata.

    Returns:
        MarketTickEvent instance.

    Raises:
        NormalizationError: If required fields are missing, invalid, or aliases conflict.
    """
    if not isinstance(payload, dict):
        raise NormalizationError(f"Payload must be a dictionary, got {type(payload).__name__}")

    search_data, nested_containers = _extract_search_context(payload)
    consumed_keys: set[str] = set()

    # 1. Extract Symbol with conflict detection
    _, raw_symbol, sym_keys = _find_field_with_conflict_check(
        search_data, TICK_SYMBOL_KEYS, "symbol", numeric=False
    )
    consumed_keys.update(sym_keys)
    symbol_str = str(raw_symbol).strip() if raw_symbol is not None else ""

    # 2. Extract OHLCV with conflict detection
    def _extract_float(keys: tuple[str, ...], name: str) -> float:
        _, val, matched_keys = _find_field_with_conflict_check(
            search_data, keys, name, numeric=True
        )
        consumed_keys.update(matched_keys)
        if val is None:
            raise NormalizationError(f"Missing required tick field '{name}' (searched: {keys})")
        try:
            return float(val)
        except (ValueError, TypeError) as exc:
            raise NormalizationError(f"Invalid numeric value for '{name}': {val!r}") from exc

    open_val = _extract_float(TICK_OPEN_KEYS, "open")
    high_val = _extract_float(TICK_HIGH_KEYS, "high")
    low_val = _extract_float(TICK_LOW_KEYS, "low")
    close_val = _extract_float(TICK_CLOSE_KEYS, "close")
    volume_val = _extract_float(TICK_VOLUME_KEYS, "volume")

    # 3. Extract Timestamp (UTC datetime of feed/emission)
    _, raw_ts, ts_keys = _find_field_with_conflict_check(
        search_data, TIMESTAMP_KEYS, "timestamp", numeric=False
    )
    consumed_keys.update(ts_keys)
    parsed_dt = parse_timestamp(raw_ts)

    # 4. Extract Simulated Date (Historical simulation market date, e.g. '2024-10-25')
    _, raw_sim_date, sim_keys = _find_field_with_conflict_check(
        search_data, SIMULATED_DATE_KEYS, "simulated_date", numeric=False
    )
    consumed_keys.update(sim_keys)
    parsed_sim_date = parse_date(raw_sim_date)

    # 5. Collect extra unmapped fields
    extra = {k: v for k, v in search_data.items() if k not in consumed_keys}

    return MarketTickEvent(
        symbol=symbol_str,
        open=open_val,
        high=high_val,
        low=low_val,
        close=close_val,
        volume=volume_val,
        timestamp=parsed_dt,
        simulated_date=parsed_sim_date,
        metadata=metadata,
        raw_payload=payload,
        extra=extra,
    )


def normalize_market_news(
    payload: dict[str, Any],
    metadata: Optional[KafkaMetadata] = None,
) -> MarketNewsEvent:
    """Normalize a raw dictionary payload into a MarketNewsEvent.

    Extracts headline/text and candidate symbol, parses timestamp and simulated_date
    separately from internal news_date, detects conflicting aliases, preserves
    unmapped fields in 'extra', and retains the original payload and metadata.
    Supports both flat and nested 'data' schemas.

    Args:
        payload: Decoded JSON dictionary.
        metadata: Optional Kafka message metadata.

    Returns:
        MarketNewsEvent instance.

    Raises:
        NormalizationError: If required fields cannot be extracted or aliases conflict.
    """
    if not isinstance(payload, dict):
        raise NormalizationError(f"Payload must be a dictionary, got {type(payload).__name__}")

    search_data, nested_containers = _extract_search_context(payload)
    consumed_keys: set[str] = set()

    # 1. Extract Headline / Text with conflict detection
    _, raw_headline, hl_keys = _find_field_with_conflict_check(
        search_data, NEWS_HEADLINE_KEYS, "headline", numeric=False
    )
    consumed_keys.update(hl_keys)
    headline_str = str(raw_headline).strip() if raw_headline is not None else ""

    # 2. Extract Symbol (optional in news) with conflict detection
    _, raw_symbol, sym_keys = _find_field_with_conflict_check(
        search_data, NEWS_SYMBOL_KEYS, "symbol", numeric=False
    )
    consumed_keys.update(sym_keys)
    symbol_str = str(raw_symbol).strip() if raw_symbol is not None else None

    # 3. Extract Timestamp (UTC datetime of feed/emission)
    _, raw_ts, ts_keys = _find_field_with_conflict_check(
        search_data, TIMESTAMP_KEYS, "timestamp", numeric=False
    )
    consumed_keys.update(ts_keys)
    parsed_dt = parse_timestamp(raw_ts)

    # 4. Extract Simulated Date (Historical simulation market date)
    _, raw_sim_date, sim_keys = _find_field_with_conflict_check(
        search_data, SIMULATED_DATE_KEYS, "simulated_date", numeric=False
    )
    consumed_keys.update(sim_keys)
    parsed_sim_date = parse_date(raw_sim_date)

    # 5. Extract News Publication Date (date inside the news content)
    _, raw_news_date, news_date_keys = _find_field_with_conflict_check(
        search_data, NEWS_DATE_KEYS, "news_date", numeric=False
    )
    consumed_keys.update(news_date_keys)
    parsed_news_date = parse_date(raw_news_date)

    # 6. Collect extra unmapped fields
    extra = {k: v for k, v in search_data.items() if k not in consumed_keys}

    return MarketNewsEvent(
        headline=headline_str,
        symbol=symbol_str,
        timestamp=parsed_dt,
        simulated_date=parsed_sim_date,
        news_date=parsed_news_date,
        metadata=metadata,
        raw_payload=payload,
        extra=extra,
    )

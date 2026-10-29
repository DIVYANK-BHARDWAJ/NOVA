"""Tests for market tick and corporate news validation logic."""

from datetime import datetime, timezone
import pytest

from market_data.models import MarketNewsEvent, MarketTickEvent
from market_data.validators import (
    ValidationError,
    assert_valid_news,
    assert_valid_tick,
    validate_market_news,
    validate_market_tick,
)


def _create_valid_tick(**overrides) -> MarketTickEvent:
    """Helper creating a baseline valid MarketTickEvent."""
    defaults = {
        "symbol": "RELIANCE",
        "open": 2500.0,
        "high": 2550.0,
        "low": 2490.0,
        "close": 2530.0,
        "volume": 1000.0,
        "timestamp": datetime.now(timezone.utc),
        "raw_payload": {},
        "extra": {"custom_indicator": 12.5},
    }
    defaults.update(overrides)
    return MarketTickEvent(**defaults)


def test_valid_tick():
    """Verify standard valid tick passes validation."""
    event = _create_valid_tick()
    result = validate_market_tick(event)
    assert result.is_valid is True
    assert len(result.errors) == 0
    # Should not raise
    assert_valid_tick(event)


def test_tick_empty_symbol():
    """Verify empty or whitespace symbol fails validation."""
    event = _create_valid_tick(symbol="")
    result = validate_market_tick(event)
    assert result.is_valid is False
    assert any("symbol is required" in e for e in result.errors)

    with pytest.raises(ValidationError):
        assert_valid_tick(event)


@pytest.mark.parametrize("price_field", ["open", "high", "low", "close"])
def test_tick_non_positive_prices(price_field):
    """Verify zero or negative prices fail validation."""
    # Zero price
    event_zero = _create_valid_tick(**{price_field: 0.0})
    result_zero = validate_market_tick(event_zero)
    assert result_zero.is_valid is False
    assert any("must be positive" in e for e in result_zero.errors)

    # Negative price
    event_neg = _create_valid_tick(**{price_field: -10.0})
    result_neg = validate_market_tick(event_neg)
    assert result_neg.is_valid is False
    assert any("must be positive" in e for e in result_neg.errors)


def test_tick_high_less_than_low():
    """Verify High < Low fails validation."""
    event = _create_valid_tick(open=2500.0, high=2450.0, low=2550.0, close=2500.0)
    result = validate_market_tick(event)
    assert result.is_valid is False
    assert any("cannot be less than Low" in e for e in result.errors)


def test_tick_open_outside_high_low():
    """Verify Open price outside [Low, High] fails validation."""
    # Open > High
    event_above = _create_valid_tick(open=2600.0, high=2550.0, low=2490.0, close=2500.0)
    result_above = validate_market_tick(event_above)
    assert result_above.is_valid is False
    assert any("Open price" in e and "outside valid range" in e for e in result_above.errors)

    # Open < Low
    event_below = _create_valid_tick(open=2400.0, high=2550.0, low=2490.0, close=2500.0)
    result_below = validate_market_tick(event_below)
    assert result_below.is_valid is False
    assert any("Open price" in e and "outside valid range" in e for e in result_below.errors)


def test_tick_close_outside_high_low():
    """Verify Close price outside [Low, High] fails validation."""
    # Close > High
    event_above = _create_valid_tick(open=2500.0, high=2550.0, low=2490.0, close=2600.0)
    result_above = validate_market_tick(event_above)
    assert result_above.is_valid is False
    assert any("Close price" in e and "outside valid range" in e for e in result_above.errors)

    # Close < Low
    event_below = _create_valid_tick(open=2500.0, high=2550.0, low=2490.0, close=2400.0)
    result_below = validate_market_tick(event_below)
    assert result_below.is_valid is False
    assert any("Close price" in e and "outside valid range" in e for e in result_below.errors)


def test_tick_negative_volume():
    """Verify negative volume fails validation."""
    event = _create_valid_tick(volume=-1.0)
    result = validate_market_tick(event)
    assert result.is_valid is False
    assert any("Volume cannot be negative" in e for e in result.errors)


def test_tick_zero_volume_is_valid():
    """Verify zero volume is valid (e.g. no trades during candle)."""
    event = _create_valid_tick(volume=0.0)
    result = validate_market_tick(event)
    assert result.is_valid is True


def test_tick_missing_timestamp_is_accepted():
    """Verify tick without timestamp still passes validation if OHLCV is valid."""
    event = _create_valid_tick(timestamp=None)
    result = validate_market_tick(event)
    assert result.is_valid is True


def test_news_validation():
    """Verify corporate news validation rules."""
    # Valid news
    news_valid = MarketNewsEvent(
        headline="Board announces dividend of Rs 15 per share",
        symbol="TCS",
    )
    res_valid = validate_market_news(news_valid)
    assert res_valid.is_valid is True
    assert_valid_news(news_valid)

    # Empty headline fails
    news_empty = MarketNewsEvent(headline="   ", symbol="TCS")
    res_empty = validate_market_news(news_empty)
    assert res_empty.is_valid is False
    assert any("headline" in e for e in res_empty.errors)

    with pytest.raises(ValidationError):
        assert_valid_news(news_empty)

    # Empty string symbol fails (symbol should be None or non-empty string)
    news_bad_sym = MarketNewsEvent(headline="Breaking news", symbol="  ")
    res_bad_sym = validate_market_news(news_bad_sym)
    assert res_bad_sym.is_valid is False
    assert any("News symbol" in e for e in res_bad_sym.errors)

    # News without symbol is valid (e.g. macro market announcement)
    news_no_sym = MarketNewsEvent(headline="Market opens higher following global cues", symbol=None)
    res_no_sym = validate_market_news(news_no_sym)
    assert res_no_sym.is_valid is True


def test_tick_nan_prices_rejected():
    """Verify that NaN prices fail validation and cannot bypass bounds checks."""
    nan_val = float("nan")
    for price_field in ["open", "high", "low", "close"]:
        event = _create_valid_tick(**{price_field: nan_val})
        result = validate_market_tick(event)
        assert result.is_valid is False
        assert any("finite number" in e for e in result.errors)


def test_tick_infinity_prices_rejected():
    """Verify that positive and negative infinity prices fail validation."""
    for inf_val in [float("inf"), float("-inf")]:
        for price_field in ["open", "high", "low", "close"]:
            event = _create_valid_tick(**{price_field: inf_val})
            result = validate_market_tick(event)
            assert result.is_valid is False
            assert any("finite number" in e for e in result.errors)


def test_tick_extreme_prices_rejected():
    """Verify that unrealistic extreme prices (> 1e9) fail validation."""
    event = _create_valid_tick(open=1e10, high=1.1e10, low=9e9, close=1e10)
    result = validate_market_tick(event)
    assert result.is_valid is False
    assert any("exceeds reasonable maximum" in e for e in result.errors)


def test_tick_nan_and_infinity_volume_rejected():
    """Verify that NaN and Infinity volume values fail validation."""
    for bad_vol in [float("nan"), float("inf"), float("-inf")]:
        event = _create_valid_tick(volume=bad_vol)
        result = validate_market_tick(event)
        assert result.is_valid is False
        assert any("finite number" in e for e in result.errors)


def test_tick_extreme_volume_rejected():
    """Verify that extreme volume (> 1e14) fails validation."""
    event = _create_valid_tick(volume=1e15)
    result = validate_market_tick(event)
    assert result.is_valid is False
    assert any("exceeds reasonable maximum" in e for e in result.errors)


def test_tick_custom_configurable_bounds():
    """Verify that callers can supply custom price and volume upper bounds."""
    # A price of 500 fails if max_price is 400
    event = _create_valid_tick(open=500.0, high=520.0, low=490.0, close=510.0, volume=1000.0)

    # Valid with default (1e9)
    res_default = validate_market_tick(event)
    assert res_default.is_valid is True

    # Invalid with custom lower max_price (400.0)
    res_custom = validate_market_tick(event, max_price=400.0)
    assert res_custom.is_valid is False
    assert any("exceeds reasonable maximum (400.0)" in e for e in res_custom.errors)

    # Invalid with custom lower max_volume (500.0)
    res_vol = validate_market_tick(event, max_volume=500.0)
    assert res_vol.is_valid is False
    assert any("Volume exceeds reasonable maximum (500.0)" in e for e in res_vol.errors)


"""Independent validation rules for market tick and news events.

Guarantees that corrupt, malformed, non-finite (NaN/Inf), or out-of-bounds data never reaches
downstream consumers. Never silently transforms invalid data into valid-looking data.
"""

from __future__ import annotations

import math
from market_data.models import MarketTickEvent, MarketNewsEvent, ValidationResult

# Guard rails against extreme corrupted values (configurable with sensible defaults)
DEFAULT_MAX_PRICE = 1e9    # 1 Billion price cap
DEFAULT_MAX_VOLUME = 1e14  # 100 Trillion volume cap

# Retain module constants for backward compatibility
MAX_REASONABLE_PRICE = DEFAULT_MAX_PRICE
MAX_REASONABLE_VOLUME = DEFAULT_MAX_VOLUME


class ValidationError(ValueError):
    """Raised when an event fails domain validation."""
    pass


def _is_valid_finite_price(
    val: float,
    name: str,
    errors: list[str],
    max_price: float = DEFAULT_MAX_PRICE,
) -> bool:
    """Validate that price is finite, positive, and within reasonable market bounds."""
    if not isinstance(val, (int, float)) or not math.isfinite(val):
        errors.append(f"{name} price must be a finite number (got {val!r})")
        return False
    if val <= 0:
        errors.append(f"{name} price must be positive (> 0), got {val}")
        return False
    if val > max_price:
        errors.append(f"{name} price exceeds reasonable maximum ({max_price}), got {val}")
        return False
    return True


def validate_market_tick(
    event: MarketTickEvent,
    max_price: float = DEFAULT_MAX_PRICE,
    max_volume: float = DEFAULT_MAX_VOLUME,
) -> ValidationResult:
    """Validate candidate OHLCV candlestick records against financial consistency rules.

    Checks:
    1. Ticker / symbol is non-empty.
    2. Open, High, Low, Close prices are finite, positive (> 0), and within bounds.
    3. High is greater than or equal to Low (High >= Low).
    4. Open lies within the [Low, High] range.
    5. Close lies within the [Low, High] range.
    6. Volume is finite and non-negative (Volume >= 0).

    Args:
        event: Normalized MarketTickEvent instance.
        max_price: Configurable upper price threshold (default: 1e9).
        max_volume: Configurable upper volume threshold (default: 1e14).

    Returns:
        ValidationResult indicating validity and any violation descriptions.
    """
    errors: list[str] = []

    # 1. Symbol non-empty
    if not event.symbol or not event.symbol.strip():
        errors.append("Market tick symbol is required and cannot be empty")

    # 2. Finite positive prices
    open_ok = _is_valid_finite_price(event.open, "Open", errors, max_price=max_price)
    high_ok = _is_valid_finite_price(event.high, "High", errors, max_price=max_price)
    low_ok = _is_valid_finite_price(event.low, "Low", errors, max_price=max_price)
    close_ok = _is_valid_finite_price(event.close, "Close", errors, max_price=max_price)

    # 3. OHLC bounds consistency (only check if all prices were valid and finite)
    if open_ok and high_ok and low_ok and close_ok:
        if event.high < event.low:
            errors.append(f"High price ({event.high}) cannot be less than Low price ({event.low})")

        # Open in [Low, High] with floating-point tolerance
        if event.open < (event.low - 1e-9) or event.open > (event.high + 1e-9):
            errors.append(
                f"Open price ({event.open}) outside valid range [Low={event.low}, High={event.high}]"
            )

        # Close in [Low, High] with floating-point tolerance
        if event.close < (event.low - 1e-9) or event.close > (event.high + 1e-9):
            errors.append(
                f"Close price ({event.close}) outside valid range [Low={event.low}, High={event.high}]"
            )

    # 4. Volume finite and non-negative
    if not isinstance(event.volume, (int, float)) or not math.isfinite(event.volume):
        errors.append(f"Volume must be a finite number (got {event.volume!r})")
    elif event.volume < 0:
        errors.append(f"Volume cannot be negative, got {event.volume}")
    elif event.volume > max_volume:
        errors.append(f"Volume exceeds reasonable maximum ({max_volume}), got {event.volume}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)



def validate_market_news(event: MarketNewsEvent) -> ValidationResult:
    """Validate corporate news records.

    Cautious validation aligned with verified competition schema:
    1. Headline / news content must be non-empty.
    2. If a symbol is present, it must be a non-empty string.
    3. Does not enforce unconfirmed fields (e.g. symbol can be None for macro news).

    Args:
        event: Normalized MarketNewsEvent instance.

    Returns:
        ValidationResult indicating validity and any violation descriptions.
    """
    errors: list[str] = []

    if not event.headline or not event.headline.strip():
        errors.append("News headline or text content is required and cannot be empty")

    if event.symbol is not None and not event.symbol.strip():
        errors.append("News symbol, if provided, cannot be an empty string")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def assert_valid_tick(
    event: MarketTickEvent,
    max_price: float = DEFAULT_MAX_PRICE,
    max_volume: float = DEFAULT_MAX_VOLUME,
) -> None:
    """Validate market tick and raise ValidationError if invalid."""
    result = validate_market_tick(event, max_price=max_price, max_volume=max_volume)
    if not result.is_valid:
        raise ValidationError(f"Invalid market tick [{event.symbol}]: {result.error_message}")



def assert_valid_news(event: MarketNewsEvent) -> None:
    """Validate market news and raise ValidationError if invalid."""
    result = validate_market_news(event)
    if not result.is_valid:
        raise ValidationError(f"Invalid market news: {result.error_message}")

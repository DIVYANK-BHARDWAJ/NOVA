"""Unit tests for models post_init validation invariants and properties."""

import pytest

from execution.models import (
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
)


def test_order_intent_empty_intent_id():
    """Empty or non-string intent_id raises ValueError."""
    with pytest.raises(ValueError, match="intent_id must be a non-empty string"):
        OrderIntent(intent_id="", ticker="ROUT", action=OrderAction.BUY, quantity=10)

    with pytest.raises(ValueError, match="intent_id must be a non-empty string"):
        OrderIntent(intent_id=None, ticker="ROUT", action=OrderAction.BUY, quantity=10)  # type: ignore


def test_order_intent_empty_ticker():
    """Empty or whitespace ticker raises ValueError."""
    with pytest.raises(ValueError, match="ticker must be a non-empty string"):
        OrderIntent(intent_id="id1", ticker="", action=OrderAction.BUY, quantity=10)

    with pytest.raises(ValueError, match="ticker must be a non-empty string"):
        OrderIntent(intent_id="id2", ticker="   ", action=OrderAction.BUY, quantity=10)


def test_order_intent_invalid_action_type():
    """Action of non-string and non-OrderAction type raises ValueError."""
    with pytest.raises(ValueError, match="action must be an OrderAction enum"):
        OrderIntent(intent_id="id1", ticker="ROUT", action=12345, quantity=10)  # type: ignore


def test_order_result_properties():
    """Test is_filled, is_rejected, is_uncertain properties."""
    res_filled = OrderResult(
        client_intent_id="i1",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        status=OrderStatus.FILLED,
    )
    assert res_filled.is_filled is True
    assert res_filled.is_rejected is False
    assert res_filled.is_uncertain is False

    res_rej = OrderResult(
        client_intent_id="i2",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        status=OrderStatus.REJECTED,
    )
    assert res_rej.is_filled is False
    assert res_rej.is_rejected is True

    res_unc = OrderResult(
        client_intent_id="i3",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        status=OrderStatus.UNKNOWN,
        requires_reconciliation=True,
    )
    assert res_unc.is_uncertain is True

"""Unit tests for ExecutionAdapter order submission, validation, and reconciliation."""

import threading
import time
from unittest.mock import MagicMock
import pytest

from execution.adapter import ExecutionAdapter
from execution.client import (
    AuthenticationError,
    RequestTimeoutError,
    ServerError,
    TradingApiClient,
)
from execution.config import ExecutionConfig
from execution.models import (
    ExecutionEvent,
    OrderAction,
    OrderIntent,
    OrderStatus,
    SimulationStage,
    StockInfo,
)


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=TradingApiClient)
    # Default valid stage
    client.get_stage.return_value = SimulationStage(
        stage="development",
        status="running",
        tickers=["ROUT", "BLUX", "AERX"],
        ticker_count=3,
    )
    # Default valid universe
    client.get_stocks.return_value = [
        StockInfo(symbol="ROUT", ticker="ROUT", name="RouteWave"),
        StockInfo(symbol="BLUX", ticker="BLUX", name="BlueMatrix"),
        StockInfo(symbol="AERX", ticker="AERX", name="AeroVanguard"),
    ]
    return client


@pytest.fixture
def adapter(mock_client: MagicMock) -> ExecutionAdapter:
    cfg = ExecutionConfig(bot_access_token="test-token")
    return ExecutionAdapter(client=mock_client, config=cfg)


def test_successful_order_submission(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Authorized intent is submitted and parsed as FILLED."""
    mock_client.submit_order.return_value = {
        "status": "FILLED",
        "order": {
            "order_id": "ord_1001",
            "ticker": "ROUT",
            "action": "BUY",
            "quantity": 50,
            "price": 78.50,
            "status": "FILLED",
            "charges": 15.0,
        },
    }

    intent = OrderIntent(
        intent_id="intent_abc",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=50,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.is_filled is True
    assert result.status == OrderStatus.FILLED
    assert result.order_id == "ord_1001"
    assert result.execution_price == 78.50
    assert result.filled_quantity == 50
    assert result.charges == 15.0
    assert result.requires_reconciliation is False

    mock_client.submit_order.assert_called_once_with(
        ticker="ROUT",
        action="BUY",
        quantity=50,
        date=None,
    )


def test_risk_gateway_authorization_enforcement(adapter: ExecutionAdapter, mock_client: MagicMock):
    """OrderIntent without risk gateway authorization must be REJECTED without sending network request."""
    intent = OrderIntent(
        intent_id="intent_unauth",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=False,  # NOT authorized by Person 3
    )

    result = adapter.execute_order(intent)

    assert result.is_rejected is True
    assert result.status == OrderStatus.REJECTED
    assert "UNAUTHORIZED_BY_RISK_GATEWAY" in result.rejection_reason

    # No network call to submit_order
    mock_client.submit_order.assert_not_called()


def test_invalid_ticker_not_in_universe(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Ticker outside active stock universe is rejected before submission."""
    intent = OrderIntent(
        intent_id="intent_bad_ticker",
        ticker="UNKNOWN_TICKER",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.is_rejected is True
    assert result.status == OrderStatus.REJECTED
    assert "INVALID_TICKER" in result.rejection_reason
    mock_client.submit_order.assert_not_called()


def test_quantity_validations_zero_negative_fractional_non_integer():
    """Quantity must be strictly integer > 0; bool, float, 0, negative raise ValueError."""
    # Zero
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        OrderIntent(intent_id="1", ticker="ROUT", action=OrderAction.BUY, quantity=0, authorized_by_risk=True)

    # Negative
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        OrderIntent(intent_id="2", ticker="ROUT", action=OrderAction.BUY, quantity=-10, authorized_by_risk=True)

    # Float / Fractional
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        OrderIntent(intent_id="3", ticker="ROUT", action=OrderAction.BUY, quantity=10.5, authorized_by_risk=True)  # type: ignore

    # Boolean (True in Python is int subclass, must be rejected!)
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        OrderIntent(intent_id="4", ticker="ROUT", action=OrderAction.BUY, quantity=True, authorized_by_risk=True)  # type: ignore

    # String
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        OrderIntent(intent_id="5", ticker="ROUT", action=OrderAction.BUY, quantity="50", authorized_by_risk=True)  # type: ignore


def test_invalid_action():
    """Invalid action string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid action"):
        OrderIntent(intent_id="1", ticker="ROUT", action="HOLD", quantity=10, authorized_by_risk=True)  # type: ignore


def test_simulation_stage_not_permitted(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Trading is blocked when platform is in non-permitted stage or stopped status."""
    mock_client.get_stage.return_value = SimulationStage(
        stage="completed",
        status="stopped",
        tickers=["ROUT"],
    )

    intent = OrderIntent(
        intent_id="intent_blocked_stage",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.is_rejected is True
    assert "STAGE_NOT_PERMITTED" in result.rejection_reason
    mock_client.submit_order.assert_not_called()


def test_stage_query_failure(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Failure to query stage safely blocks order submission without treating unknown as permission."""
    mock_client.get_stage.side_effect = ServerError("500 Internal Server Error")

    intent = OrderIntent(
        intent_id="intent_stage_fail",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.is_rejected is True
    assert "STAGE_QUERY_FAILED" in result.rejection_reason
    mock_client.submit_order.assert_not_called()


def test_request_timeout_unknown_outcome_no_blind_retry(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Request timeout marks outcome as UNKNOWN, requires reconciliation, and does NOT blindly resubmit."""
    mock_client.submit_order.side_effect = RequestTimeoutError("Request timed out after 5.0s")
    # Order history does not show the order
    mock_client.get_orders.return_value = {"orders": []}

    intent = OrderIntent(
        intent_id="intent_timeout_1",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=25,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.is_uncertain is True
    assert result.status == OrderStatus.UNKNOWN
    assert result.requires_reconciliation is True
    assert "TIMEOUT_UNCERTAIN" in result.rejection_reason

    # Verify submit_order was only called ONCE (no blind retry!)
    assert mock_client.submit_order.call_count == 1
    # Verify order history was queried for reconciliation
    mock_client.get_orders.assert_called_once_with(limit=10, ticker="ROUT", action="BUY")


def test_request_timeout_reconciled_with_history(adapter: ExecutionAdapter, mock_client: MagicMock):
    """If timeout occurs but order is found in order history, resolves status cleanly."""
    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout")
    # Order history returns the recorded order
    mock_client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "ord_recovered_99",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 25,
                "price": 78.20,
                "status": "FILLED",
            }
        ]
    }

    intent = OrderIntent(
        intent_id="intent_timeout_reconciled",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=25,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.status == OrderStatus.FILLED
    assert result.order_id == "ord_recovered_99"
    assert result.execution_price == 78.20
    assert result.requires_reconciliation is False


def test_api_rejection_payload_handling(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Server returning JSON error message produces REJECTED OrderResult."""
    mock_client.submit_order.return_value = {
        "status": "error",
        "error": "INSUFFICIENT_FUNDS",
        "message": "Available cash INR 1000 is less than required margin INR 5000",
    }

    intent = OrderIntent(
        intent_id="intent_insufficient_funds",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=100,
        authorized_by_risk=True,
    )

    result = adapter.execute_order(intent)

    assert result.status == OrderStatus.REJECTED
    assert "Available cash INR 1000" in result.rejection_reason


def test_execution_event_listener_notification(mock_client: MagicMock):
    """on_execution_event callback receives ExecutionEvent notifications."""
    events: list[ExecutionEvent] = []

    def _capture(evt: ExecutionEvent):
        events.append(evt)

    cfg = ExecutionConfig(bot_access_token="test-token")
    adapter = ExecutionAdapter(client=mock_client, config=cfg, on_execution_event=_capture)

    mock_client.submit_order.return_value = {
        "status": "FILLED",
        "order": {"order_id": "ord_e1", "ticker": "ROUT", "action": "BUY", "quantity": 10, "status": "FILLED"},
    }

    intent = OrderIntent(
        intent_id="intent_event_test",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )

    adapter.execute_order(intent)

    # Expected events: ORDER_SUBMISSION_ATTEMPT and ORDER_FILLED
    assert len(events) == 2
    assert events[0].event_type == "ORDER_SUBMISSION_ATTEMPT"
    assert events[1].event_type == "ORDER_FILLED"
    assert events[1].order_result.order_id == "ord_e1"


def test_ttl_caching_for_stage_and_universe(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Stage and universe queries are cached within TTL."""
    # First call queries client
    s1 = adapter.get_simulation_stage()
    u1 = adapter.get_tradable_universe()
    assert mock_client.get_stage.call_count == 1
    assert mock_client.get_stocks.call_count == 1

    # Second call uses cache
    s2 = adapter.get_simulation_stage()
    u2 = adapter.get_tradable_universe()
    assert mock_client.get_stage.call_count == 1
    assert mock_client.get_stocks.call_count == 1
    assert s1 == s2
    assert u1 == u2

    # Force refresh bypasses cache
    adapter.get_simulation_stage(force_refresh=True)
    adapter.get_tradable_universe(force_refresh=True)
    assert mock_client.get_stage.call_count == 2
    assert mock_client.get_stocks.call_count == 2


def test_event_callback_exception_does_not_crash_adapter(mock_client: MagicMock):
    """Exceptions raised in on_execution_event callback are handled without crashing."""
    def _exploding_callback(evt):
        raise RuntimeError("Callback crashed")

    cfg = ExecutionConfig(bot_access_token="test-token")
    adapter = ExecutionAdapter(client=mock_client, config=cfg, on_execution_event=_exploding_callback)

    mock_client.submit_order.return_value = {
        "status": "FILLED",
        "order": {"order_id": "ord_safe", "ticker": "ROUT", "action": "BUY", "quantity": 10, "status": "FILLED"},
    }

    intent = OrderIntent(intent_id="i_crash", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    result = adapter.execute_order(intent)
    assert result.status == OrderStatus.FILLED


def test_order_statuses_parsing(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Test parsing of PENDING, PARTIAL, REJECTED, and unknown statuses."""
    intent = OrderIntent(intent_id="i_stat", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)

    # 1. PENDING
    mock_client.submit_order.return_value = {"order": {"status": "ACCEPTED", "id": "1"}}
    r1 = adapter.execute_order(OrderIntent(intent_id="i_stat_1", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True))
    assert r1.status == OrderStatus.PENDING

    # 2. PARTIAL
    mock_client.submit_order.return_value = {"order": {"status": "PARTIAL", "id": "2", "filled_quantity": 5}}
    r2 = adapter.execute_order(OrderIntent(intent_id="i_stat_2", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True))
    assert r2.status == OrderStatus.PARTIALLY_FILLED
    assert r2.filled_quantity == 5

    # 3. REJECTED
    mock_client.submit_order.return_value = {"order": {"status": "FAILED", "id": "3"}}
    r3 = adapter.execute_order(OrderIntent(intent_id="i_stat_3", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True))
    assert r3.status == OrderStatus.REJECTED

    # 4. Unknown status string defaults safely to PENDING
    mock_client.submit_order.return_value = {"order": {"status": "SOME_NEW_STATUS", "id": "4"}}
    r4 = adapter.execute_order(OrderIntent(intent_id="i_stat_4", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True))
    assert r4.status == OrderStatus.PENDING


def test_trading_api_error_during_submission(adapter: ExecutionAdapter, mock_client: MagicMock):
    """TradingApiError (e.g. 403 or 429) produces REJECTED OrderResult with error message."""
    mock_client.submit_order.side_effect = AuthenticationError("401 Unauthorized")
    intent = OrderIntent(intent_id="i_auth_err", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)

    result = adapter.execute_order(intent)
    assert result.status == OrderStatus.REJECTED
    assert "401 Unauthorized" in result.rejection_reason


def test_unexpected_exception_during_submission(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Unexpected exception produces REJECTED OrderResult with UNEXPECTED_ERROR."""
    mock_client.submit_order.side_effect = RuntimeError("Catastrophic client failure")
    intent = OrderIntent(intent_id="i_unexp", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)

    result = adapter.execute_order(intent)
    assert result.status == OrderStatus.REJECTED
    assert "UNEXPECTED_ERROR" in result.rejection_reason


def test_non_float_price_in_order_response(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Non-float price in response is handled safely without crashing."""
    mock_client.submit_order.return_value = {
        "status": "FILLED",
        "order": {"order_id": "ord_bad_price", "ticker": "ROUT", "action": "BUY", "quantity": 10, "price": "invalid-price", "status": "FILLED"},
    }
    intent = OrderIntent(intent_id="i_bad_px", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.FILLED
    assert res.execution_price is None


def test_timeout_reconciliation_get_orders_failure(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Exception when querying get_orders during timeout reconciliation falls back to UNKNOWN result."""
    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout")
    mock_client.get_orders.side_effect = ServerError("500 Internal Server Error")

    intent = OrderIntent(intent_id="i_timeout_fail", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.UNKNOWN
    assert res.requires_reconciliation is True


def test_universe_refresh_failure_falls_back_to_cache(adapter: ExecutionAdapter, mock_client: MagicMock):
    """When get_stocks fails during order execution, adapter falls back to cached universe."""
    # Seed cache
    adapter.get_tradable_universe()
    assert "ROUT" in adapter._cached_universe

    # Now make get_stocks fail
    mock_client.get_stocks.side_effect = ServerError("500 Internal Server Error")
    mock_client.submit_order.return_value = {
        "status": "FILLED",
        "order": {"order_id": "ord_cached_univ", "ticker": "ROUT", "action": "BUY", "quantity": 10, "status": "FILLED"},
    }

    intent = OrderIntent(intent_id="i_univ_fail", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.FILLED


def test_stage_cache_invalidation_and_notify_stage_change(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Stage cache invalidation and notification forces fresh query and prevents stale authorization."""
    # First query fetches from client
    s1 = adapter.get_simulation_stage()
    assert s1.stage == "development"
    assert mock_client.get_stage.call_count == 1

    # Second query uses cache
    s2 = adapter.get_simulation_stage()
    assert s2.stage == "development"
    assert mock_client.get_stage.call_count == 1

    # Invalidate cache
    adapter.invalidate_stage_cache()
    s3 = adapter.get_simulation_stage()
    assert mock_client.get_stage.call_count == 2

    # Notify stage transition to stopped
    stopped_stage = SimulationStage(stage="stopped", status="stopped", tickers=[])
    adapter.notify_stage_change(stopped_stage)
    assert adapter.get_simulation_stage().status == "stopped"


def test_unknown_simulation_stage_fails_closed(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Unknown simulation stage strings fail closed and reject order submission."""
    mock_client.get_stage.return_value = SimulationStage(
        stage="AUCTION_CALL",
        status="running",
        tickers=["ROUT"],
        ticker_count=1,
    )
    adapter.invalidate_stage_cache()

    intent = OrderIntent(
        intent_id="intent_unknown_stage",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.REJECTED
    assert "STAGE_NOT_PERMITTED" in res.rejection_reason
    assert mock_client.submit_order.call_count == 0


def test_development_stage_blocked_when_allow_dev_false(mock_client: MagicMock):
    """When allow_development_stage is False, orders in sandbox or development stages are blocked."""
    cfg = ExecutionConfig(bot_access_token="test-token", allow_development_stage=False)
    strict_adapter = ExecutionAdapter(client=mock_client, config=cfg)

    # 1. Development stage name
    mock_client.get_stage.return_value = SimulationStage(
        stage="development",
        status="running",
        tickers=["ROUT"],
        ticker_count=1,
    )
    intent1 = OrderIntent(intent_id="i_dev_blocked_1", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res1 = strict_adapter.execute_order(intent1)
    assert res1.status == OrderStatus.REJECTED
    assert "STAGE_DEVELOPMENT_BLOCKED" in res1.rejection_reason

    # 2. Sandbox stream flag
    mock_client.get_stage.return_value = SimulationStage(
        stage="competition",
        status="running",
        tickers=["ROUT"],
        ticker_count=1,
        is_sandbox_stream=True,
    )
    strict_adapter.invalidate_stage_cache()
    intent2 = OrderIntent(intent_id="i_dev_blocked_2", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res2 = strict_adapter.execute_order(intent2)
    assert res2.status == OrderStatus.REJECTED
    assert "STAGE_DEVELOPMENT_BLOCKED" in res2.rejection_reason


def test_timeout_reconciliation_ignores_pre_existing_order(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Timeout reconciliation ignores orders that were already known before the submission."""
    # Seed a known pre-existing order
    adapter.record_known_order_id("older_order_id_123")

    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout after 5s")
    # History returns only the older order
    mock_client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "older_order_id_123",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 10,
                "status": "FILLED",
                "timestamp": time.time(),
            }
        ]
    }

    intent = OrderIntent(
        intent_id="i_timeout_pre_exist",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.UNKNOWN
    assert res.requires_reconciliation is True
    assert "0 matching candidate orders found" in res.rejection_reason


def test_timeout_reconciliation_ignores_older_timestamp_order(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Timeout reconciliation ignores historical orders with timestamps older than the submission window."""
    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout")
    # History returns order from 2 hours ago
    mock_client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "stale_order_456",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 10,
                "status": "FILLED",
                "timestamp": time.time() - 7200,
            }
        ]
    }

    intent = OrderIntent(
        intent_id="i_timeout_old_ts",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.UNKNOWN
    assert res.requires_reconciliation is True


def test_timeout_reconciliation_multiple_matching_candidates_remains_unknown(adapter: ExecutionAdapter, mock_client: MagicMock):
    """When multiple identical matching orders exist in history, outcome remains UNKNOWN rather than guessed."""
    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout")
    now_ts = time.time()
    mock_client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "candidate_1",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 10,
                "status": "FILLED",
                "timestamp": now_ts,
            },
            {
                "order_id": "candidate_2",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 10,
                "status": "FILLED",
                "timestamp": now_ts,
            },
        ]
    }

    intent = OrderIntent(
        intent_id="i_timeout_ambig",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.UNKNOWN
    assert res.requires_reconciliation is True
    assert "2 matching candidate orders found" in res.rejection_reason


def test_timeout_reconciliation_single_matching_candidate_reconciles(adapter: ExecutionAdapter, mock_client: MagicMock):
    """When exactly one unambiguous matching order exists in history, reconciliation succeeds."""
    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout")
    now_ts = time.time()
    mock_client.get_orders.return_value = {
        "orders": [
            {
                "order_id": "candidate_unique_999",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 10,
                "status": "FILLED",
                "price": 105.5,
                "timestamp": now_ts,
            }
        ]
    }

    intent = OrderIntent(
        intent_id="i_timeout_unambig",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.FILLED
    assert res.order_id == "candidate_unique_999"
    assert res.execution_price == 105.5
    assert adapter.is_intent_completed("i_timeout_unambig") is True


def test_concurrent_duplicate_intent_execution(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Concurrent threads attempting to submit the same intent_id are rejected."""
    results = []

    def slow_submit_order(**kwargs):
        time.sleep(0.05)
        return {"order": {"id": "ord_concurrent", "status": "FILLED", "price": 100.0}}

    mock_client.submit_order.side_effect = slow_submit_order
    shared_intent = OrderIntent(
        intent_id="intent_shared_concurrent",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=True,
    )

    threads = []
    for _ in range(5):
        t = threading.Thread(target=lambda: results.append(adapter.execute_order(shared_intent)))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 5
    # Exactly one submission should succeed or proceed to network
    filled_count = sum(1 for r in results if r.status == OrderStatus.FILLED)
    rejected_count = sum(1 for r in results if r.status == OrderStatus.REJECTED)

    assert filled_count == 1
    assert rejected_count == 4
    # All rejections must be due to concurrent duplicate or duplicate intent
    for r in results:
        if r.status == OrderStatus.REJECTED:
            assert "DUPLICATE_INTENT" in r.rejection_reason

    # The actual server endpoint must only have been invoked ONCE
    assert mock_client.submit_order.call_count == 1


def test_sequential_duplicate_intent_rejected(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Submitting an intent with an already processed intent_id is rejected without network call."""
    mock_client.submit_order.return_value = {"order": {"id": "ord_seq_1", "status": "FILLED", "price": 50.0}}
    intent = OrderIntent(
        intent_id="intent_seq_dup",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=5,
        authorized_by_risk=True,
    )

    r1 = adapter.execute_order(intent)
    assert r1.status == OrderStatus.FILLED
    assert mock_client.submit_order.call_count == 1

    # Second submission
    r2 = adapter.execute_order(intent)
    assert r2.status == OrderStatus.REJECTED
    assert "DUPLICATE_INTENT" in r2.rejection_reason
    # Verify submit_order was NOT called a second time
    assert mock_client.submit_order.call_count == 1


def test_unauthorized_intent_never_touches_stage_or_network(adapter: ExecutionAdapter, mock_client: MagicMock):
    """OrderIntent with authorized_by_risk=False fails immediately without querying stage or exchange."""
    unauthorized_intent = OrderIntent(
        intent_id="unauth_intent_test",
        ticker="ROUT",
        action=OrderAction.BUY,
        quantity=10,
        authorized_by_risk=False,
    )

    res = adapter.execute_order(unauthorized_intent)
    assert res.status == OrderStatus.REJECTED
    assert "UNAUTHORIZED_BY_RISK_GATEWAY" in res.rejection_reason

    mock_client.get_stage.assert_not_called()
    mock_client.get_stocks.assert_not_called()
    mock_client.submit_order.assert_not_called()


def test_adapter_helper_methods_and_universe_invalidation(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Test helper queries (is_intent_active, get_completed_order, invalidate_universe_cache)."""
    assert adapter.is_intent_active("non_existent") is False
    assert adapter.get_completed_order("non_existent") is None

    # Universe invalidation
    adapter.get_tradable_universe()
    assert adapter._cached_universe is not None
    adapter.invalidate_universe_cache()
    assert adapter._cached_universe is None

    # Execute an order to populate completed intents
    mock_client.submit_order.return_value = {"order": {"id": "ord_hlp", "status": "FILLED"}}
    intent = OrderIntent(intent_id="i_hlp", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.FILLED

    assert adapter.is_intent_completed("i_hlp") is True
    completed = adapter.get_completed_order("i_hlp")
    assert completed is not None
    assert completed.client_intent_id == "i_hlp"


def test_timeout_reconciliation_iso_timestamp_and_non_dict_items(adapter: ExecutionAdapter, mock_client: MagicMock):
    """Test timeout reconciliation handling of ISO timestamp, non-dict entries, and mismatched tickers."""
    from datetime import datetime, timezone
    mock_client.submit_order.side_effect = RequestTimeoutError("Timeout")
    iso_now = datetime.now(timezone.utc).isoformat()

    mock_client.get_orders.return_value = {
        "orders": [
            "not-a-dict",  # non-dict
            {"order_id": "bad_ticker", "ticker": "OTHER", "action": "BUY", "quantity": 10},
            {"order_id": "bad_qty", "ticker": "ROUT", "action": "BUY", "quantity": "invalid-int"},
            {
                "order_id": "iso_reconciled",
                "ticker": "ROUT",
                "action": "BUY",
                "quantity": 10,
                "status": "FILLED",
                "timestamp": iso_now,
            },
        ]
    }

    intent = OrderIntent(intent_id="i_iso_recon", ticker="ROUT", action=OrderAction.BUY, quantity=10, authorized_by_risk=True)
    res = adapter.execute_order(intent)
    assert res.status == OrderStatus.FILLED
    assert res.order_id == "iso_reconciled"

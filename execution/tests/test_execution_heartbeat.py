"""Unit tests for asynchronous Pre-Flight Heartbeat Worker."""

import time
from unittest.mock import MagicMock
import pytest

from execution.client import ServerError, TradingApiClient
from execution.heartbeat import HeartbeatWorker


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=TradingApiClient)
    client.send_heartbeat.return_value = {"status": "ok"}
    return client


def test_heartbeat_worker_async_and_deduplication(mock_client: MagicMock):
    """HeartbeatWorker processes sequence numbers asynchronously and deduplicates."""
    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        # Enqueue sequence 5001
        res1 = worker.enqueue_heartbeat("5001")
        assert res1 is True

        # Enqueue duplicate 5001 -> must be rejected
        res2 = worker.enqueue_heartbeat("5001")
        assert res2 is False

        # Enqueue sequence 5002
        res3 = worker.enqueue_heartbeat(5002)
        assert res3 is True

        # Wait briefly for daemon thread to process
        time.sleep(0.3)

        assert worker.acknowledged_count == 2
        assert worker.duplicate_count == 1
        assert worker.failure_count == 0

        # Verify client calls
        assert mock_client.send_heartbeat.call_count == 2
        mock_client.send_heartbeat.assert_any_call("5001")
        mock_client.send_heartbeat.assert_any_call("5002")

    finally:
        worker.stop()


def test_heartbeat_worker_http_error_resilience(mock_client: MagicMock):
    """Worker handles HTTP failures gracefully without terminating."""
    mock_client.send_heartbeat.side_effect = ServerError("500 Internal Server Error")

    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        worker.enqueue_heartbeat("9001")
        time.sleep(0.3)

        assert worker.failure_count == 1
        assert worker.is_running is True

    finally:
        worker.stop()


def test_heartbeat_worker_handle_event(mock_client: MagicMock):
    """handle_event extracts sequence number from preflight_tick event payloads."""
    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        # 1. Matching preflight tick dictionary
        dict_event = {
            "event_type": "preflight_tick",
            "seq": "7788",
            "timestamp": "2026-03-25T10:00:00Z",
        }
        handled = worker.handle_event(dict_event)
        assert handled is True

        # 2. Non-preflight tick dictionary
        normal_tick = {"event_type": "price_tick", "seq": "9999"}
        handled_normal = worker.handle_event(normal_tick)
        assert handled_normal is False

        # 3. Object with extra dict (like MarketTickEvent)
        class MockEvent:
            def __init__(self, extra):
                self.extra = extra

        obj_event = MockEvent({"event_type": "preflight_tick", "seq": "7789"})
        handled_obj = worker.handle_event(obj_event)
        assert handled_obj is True

        time.sleep(0.3)
        assert mock_client.send_heartbeat.call_count == 2
        mock_client.send_heartbeat.assert_any_call("7788")
        mock_client.send_heartbeat.assert_any_call("7789")

    finally:
        worker.stop()


def test_heartbeat_worker_edge_cases(mock_client: MagicMock):
    """Test start() idempotency, empty seq, queue full, and generic exception."""
    worker = HeartbeatWorker(client=mock_client, max_queue_size=1)
    worker.start()
    worker.start()  # Idempotent second start

    try:
        # 1. Blank seq
        assert worker.enqueue_heartbeat("") is False
        assert worker.enqueue_heartbeat("   ") is False

        # 2. Queue full test (max_queue_size=1)
        # Block worker thread from popping immediately
        worker._running = False  # pause processing
        assert worker.enqueue_heartbeat("101") is True
        assert worker.enqueue_heartbeat("102") is False  # queue full

    finally:
        worker.stop()


def test_heartbeat_worker_generic_exception(mock_client: MagicMock):
    """Worker catches generic unexpected exceptions without terminating."""
    mock_client.send_heartbeat.side_effect = RuntimeError("Catastrophic error")
    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        worker.enqueue_heartbeat("9999")
        time.sleep(0.3)
        assert worker.failure_count == 1
    finally:
        worker.stop()


def test_heartbeat_worker_retry_after_transient_failure(mock_client: MagicMock):
    """If send_heartbeat fails, sequence is released from in-flight to permit redelivery/retry."""
    # First call fails, second succeeds
    mock_client.send_heartbeat.side_effect = [
        ServerError("503 Gateway Unavailable"),
        {"status": "ok", "seq": 2048},
    ]

    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        assert worker.enqueue_heartbeat("2048") is True
        time.sleep(0.3)

        assert worker.failure_count == 1
        assert worker.acknowledged_count == 0

        # Now redeliver / retry seq 2048 -> must be accepted, not dropped as duplicate
        assert worker.enqueue_heartbeat("2048") is True
        time.sleep(0.3)

        assert worker.acknowledged_count == 1
        assert worker.failure_count == 1

        # Now that it succeeded, a third attempt MUST be rejected as duplicate
        assert worker.enqueue_heartbeat("2048") is False
        assert worker.duplicate_count == 1
    finally:
        worker.stop()


def test_heartbeat_worker_queue_overflow_allows_later_enqueue(mock_client: MagicMock):
    """Sequence rejected due to full queue is NOT marked acknowledged and can be enqueued later."""
    worker = HeartbeatWorker(client=mock_client, max_queue_size=1)
    worker.start()
    worker._running = False  # Pause worker loop so queue fills

    try:
        assert worker.enqueue_heartbeat("3001") is True
        assert worker.enqueue_heartbeat("3002") is False  # Queue full!

        # Resume worker loop to drain queue
        worker._running = True
        worker._thread = None
        worker.start()
        time.sleep(0.3)

        assert worker.acknowledged_count == 1

        # Now enqueue 3002 again -> must succeed now that queue has room
        assert worker.enqueue_heartbeat("3002") is True
        time.sleep(0.3)
        assert worker.acknowledged_count == 2
    finally:
        worker.stop()


def test_heartbeat_worker_preflight_news_and_event_types(mock_client: MagicMock):
    """handle_event extracts sequences from preflight_tick, preflight_news, and nested payloads."""
    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        # Preflight news dictionary
        news_event = {
            "event": "preflight_news",
            "event_type": "preflight_news",
            "timestamp": "2026-10-29T13:38:23.094Z",
            "seq": 1040,
            "message": "Pre-flight standby ping",
        }
        assert worker.handle_event(news_event) is True

        # Preflight tick with nested data dict
        nested_event = {
            "event": "preflight_tick",
            "data": {"status": "STANDBY", "seq": 1041},
        }
        assert worker.handle_event(nested_event) is True

        # Heartbeat event
        hb_event = {
            "event_type": "heartbeat",
            "seq": 1042,
        }
        assert worker.handle_event(hb_event) is True

        time.sleep(0.4)
        assert worker.acknowledged_count == 3
        mock_client.send_heartbeat.assert_any_call("1040")
        mock_client.send_heartbeat.assert_any_call("1041")
        mock_client.send_heartbeat.assert_any_call("1042")
    finally:
        worker.stop()


def test_heartbeat_worker_processes_subsequent_sequences(mock_client: MagicMock):
    """Processes multiple subsequent sequence numbers continuously."""
    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        sequences = [1, 2, 54, 55, 100, 1041]
        for s in sequences:
            assert worker.enqueue_heartbeat(s) is True

        time.sleep(0.5)
        assert worker.acknowledged_count == len(sequences)
        assert worker.failure_count == 0
        assert mock_client.send_heartbeat.call_count == len(sequences)
    finally:
        worker.stop()

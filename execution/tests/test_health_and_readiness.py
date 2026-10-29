"""Targeted tests for Step G: Health/Readiness diagnostics, fault resilience, and shutdown behavior."""

import json
import time
from unittest.mock import MagicMock
import pytest

from execution.client import ServerError, TradingApiClient, TradingApiError
from execution.config import ExecutionConfig
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from execution.models import SimulationClock, SimulationStage
from market_data.config import KafkaConfig
from market_data.consumer import MarketDataConsumer


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock TradingApiClient with standard responses."""
    client = MagicMock(spec=TradingApiClient)
    client.get_stage.return_value = SimulationStage(
        stage="preflight",
        status="running",
        tickers=["ROUT", "SONA"],
        ticker_count=2,
        current_date="2026-03-25",
        days_completed=0,
        total_days=500,
        interval_seconds=4,
    )
    client.get_simulation_clock.return_value = SimulationClock(
        status="running",
        simulated_date="2026-03-25",
        current_day=0,
        days_completed=0,
        total_days=500,
        remaining_days=500,
        interval_seconds=4,
    )
    client.send_heartbeat.return_value = {"status": "ok", "message": "Heartbeat acknowledged"}
    return client


# ============================================================================
# 1. READINESS INVARIANTS & BLOCKERS
# ============================================================================

def test_readiness_false_when_heartbeats_missing_in_preflight(mock_client: MagicMock):
    """In preflight stage, bot_ready MUST NOT report True if heartbeats acknowledged == 0."""
    worker = HeartbeatWorker(client=mock_client)
    mock_consumer = MagicMock()
    mock_consumer.is_running = True
    mock_consumer.assigned_partitions = ["part-0", "part-1"]

    monitor = LifecycleMonitor(
        client=mock_client,
        heartbeat_worker=worker,
        consumer=mock_consumer,
    )
    monitor.poll_once()

    # Start worker, but do NOT enqueue any heartbeat
    worker.start()
    try:
        assert worker.is_running is True
        assert worker.acknowledged_count == 0

        summary = monitor.get_readiness_summary()
        # In preflight with 0 acks, readiness MUST be False!
        assert summary["bot_ready"] is False
        assert summary["is_preflight"] is True
        assert any("Heartbeat acknowledgements missing in preflight" in b for b in summary["readiness_blockers"])
    finally:
        worker.stop()


def test_readiness_true_when_preflight_heartbeat_acknowledged(mock_client: MagicMock):
    """bot_ready reports True only when all components are active and preflight heartbeat is acked."""
    worker = HeartbeatWorker(client=mock_client)
    mock_consumer = MagicMock()
    mock_consumer.is_running = True
    mock_consumer.assigned_partitions = ["part-0", "part-1"]

    monitor = LifecycleMonitor(
        client=mock_client,
        heartbeat_worker=worker,
        consumer=mock_consumer,
    )
    monitor.poll_once()

    worker.start()
    try:
        worker.enqueue_heartbeat("1001")
        time.sleep(0.2)

        assert worker.acknowledged_count == 1
        summary = monitor.get_readiness_summary()

        assert summary["bot_ready"] is True
        assert len(summary["readiness_blockers"]) == 0
        assert summary["kafka_connected"] is True
        assert summary["kafka_partitions_assigned"] == 2
        assert summary["heartbeat_worker_active"] is True
        assert summary["heartbeats_acknowledged"] == 1
        assert summary["api_reachable"] is True
    finally:
        worker.stop()


def test_readiness_false_when_kafka_stopped_or_zero_partitions(mock_client: MagicMock):
    """bot_ready reports False if Kafka consumer is stopped or has 0 assigned partitions."""
    worker = HeartbeatWorker(client=mock_client)
    worker.start()
    try:
        worker.enqueue_heartbeat("1002")
        time.sleep(0.2)
        assert worker.acknowledged_count == 1

        # Case A: Consumer stopped
        mock_consumer = MagicMock()
        mock_consumer.is_running = False
        mock_consumer.assigned_partitions = ["part-0"]

        monitor = LifecycleMonitor(
            client=mock_client,
            heartbeat_worker=worker,
            consumer=mock_consumer,
        )
        monitor.poll_once()

        summary_a = monitor.get_readiness_summary()
        assert summary_a["bot_ready"] is False
        assert "Kafka consumer is not running" in summary_a["readiness_blockers"]

        # Case B: Consumer running but 0 partitions assigned
        mock_consumer.is_running = True
        mock_consumer.assigned_partitions = []

        summary_b = monitor.get_readiness_summary()
        assert summary_b["bot_ready"] is False
        assert "Kafka consumer has 0 assigned partitions" in summary_b["readiness_blockers"]
    finally:
        worker.stop()


def test_readiness_false_when_api_unreachable(mock_client: MagicMock):
    """bot_ready reports False and records API blocker when API queries fail."""
    mock_client.get_stage.side_effect = ServerError("503 Service Unavailable")
    mock_client.get_simulation_clock.side_effect = ServerError("503 Service Unavailable")

    worker = HeartbeatWorker(client=mock_client)
    mock_consumer = MagicMock()
    mock_consumer.is_running = True
    mock_consumer.assigned_partitions = ["part-0"]

    monitor = LifecycleMonitor(
        client=mock_client,
        heartbeat_worker=worker,
        consumer=mock_consumer,
    )
    monitor.poll_once()

    summary = monitor.get_readiness_summary()
    assert summary["bot_ready"] is False
    assert summary["api_reachable"] is False
    assert any("Trading API query failed" in b for b in summary["readiness_blockers"])


def test_readiness_false_when_heartbeat_worker_inactive(mock_client: MagicMock):
    """bot_ready reports False when heartbeat worker is None or inactive."""
    mock_consumer = MagicMock()
    mock_consumer.is_running = True
    mock_consumer.assigned_partitions = ["part-0"]

    # No heartbeat worker attached
    monitor = LifecycleMonitor(
        client=mock_client,
        heartbeat_worker=None,
        consumer=mock_consumer,
    )
    monitor.poll_once()

    summary = monitor.get_readiness_summary()
    assert summary["bot_ready"] is False
    assert "HeartbeatWorker is not active" in summary["readiness_blockers"]


# ============================================================================
# 2. DUPLICATE WORKER & POLLING PREVENTION
# ============================================================================

def test_duplicate_worker_start_idempotency(mock_client: MagicMock):
    """Multiple calls to start() on HeartbeatWorker and LifecycleMonitor are idempotent."""
    worker = HeartbeatWorker(client=mock_client)
    monitor = LifecycleMonitor(client=mock_client)

    # 1. HeartbeatWorker duplicate start
    worker.start()
    thread_1 = worker._thread
    worker.start()  # Duplicate
    thread_2 = worker._thread
    assert thread_1 is thread_2
    assert worker.is_running is True
    worker.stop()

    # 2. LifecycleMonitor duplicate start
    monitor.start()
    mon_thread_1 = monitor._thread
    monitor.start()  # Duplicate
    mon_thread_2 = monitor._thread
    assert mon_thread_1 is mon_thread_2
    assert monitor.is_running is True
    monitor.stop()


def test_lifecycle_monitor_min_poll_interval_floor(mock_client: MagicMock):
    """Enforces non-zero minimum poll interval to prevent excessive polling."""
    monitor = LifecycleMonitor(client=mock_client, poll_interval_seconds=0.001)
    assert monitor.poll_interval_seconds >= 0.05


# ============================================================================
# 3. BOUNDED RETRIES & DEDUPLICATION
# ============================================================================

def test_heartbeat_worker_bounded_retries_and_dropping(mock_client: MagicMock):
    """Worker retries recoverable failures up to max_retries and drops on exhaustion."""
    mock_client.send_heartbeat.side_effect = ServerError("500 Internal Error")

    worker = HeartbeatWorker(client=mock_client, max_retries=2)
    worker.start()

    try:
        # Enqueue seq 3001
        assert worker.enqueue_heartbeat("3001") is True
        time.sleep(0.3)
        assert worker.failure_count >= 1

        # Attempt retry of 3001 (attempt 2)
        assert worker.enqueue_heartbeat("3001") is True
        time.sleep(0.3)

        # Now attempt count is 2 (>= max_retries=2) -> Next enqueue must be dropped!
        res = worker.enqueue_heartbeat("3001")
        assert res is False
        assert worker.acknowledged_count == 0
    finally:
        worker.stop()


def test_heartbeat_worker_duplicate_prevention(mock_client: MagicMock):
    """Worker rejects duplicate sequence IDs while in-flight or acknowledged."""
    worker = HeartbeatWorker(client=mock_client)
    worker.start()

    try:
        # First enqueue succeeds
        assert worker.enqueue_heartbeat("4001") is True
        # Immediate duplicate rejection
        assert worker.enqueue_heartbeat("4001") is False
        assert worker.duplicate_count == 1

        time.sleep(0.2)
        assert worker.acknowledged_count == 1

        # Post-acknowledgement duplicate rejection
        assert worker.enqueue_heartbeat("4001") is False
        assert worker.duplicate_count == 2
    finally:
        worker.stop()


# ============================================================================
# 4. KAFKA DISCONNECT & CONSUMER FAULT RESILIENCE
# ============================================================================

class MockMessage:
    def __init__(self, topic="market.ticks", value=None, error=None, partition=0, offset=1):
        self._topic = topic
        self._value = value
        self._error = error
        self._partition = partition
        self._offset = offset

    def topic(self):
        return self._topic

    def value(self):
        return self._value

    def error(self):
        return self._error

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def timestamp(self):
        return (1, 1700000000000)


def test_kafka_consumer_disconnect_handling_and_health_status():
    """Consumer handles broker transport disconnect gracefully and reports health metrics."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    consumer = MarketDataConsumer(config=config, disconnect_backoff_seconds=0.01)

    mock_err = MagicMock()
    mock_err.code.return_value = -187
    mock_err.str.return_value = "Broker transport failure"

    msg_err = MockMessage(error=mock_err)

    # Process disconnect message
    handled = consumer.process_raw_message(msg_err)
    assert handled is False
    assert consumer.kafka_errors == 1

    health = consumer.get_health_status()
    assert health["kafka_errors"] == 1
    assert health["bootstrap_servers"] == "10.0.0.1:9094"
    assert health["is_running"] is False


def test_kafka_consumer_poll_transient_error_recovery():
    """Consumer survives transient poll exceptions and recovers to process subsequent messages."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    mock_client = MagicMock()

    valid_payload = json.dumps({
        "symbol": "TCS",
        "open": 3500.0,
        "high": 3550.0,
        "low": 3480.0,
        "close": 3520.0,
        "volume": 1000.0,
    }).encode("utf-8")
    msg_valid = MockMessage(topic="market.ticks", value=valid_payload)

    # Poll 1: Broker disconnect exception -> Poll 2: Valid message -> Poll 3: None
    mock_client.poll.side_effect = [
        RuntimeError("Broker transport disconnect occurred"),
        msg_valid,
        None,
    ]

    consumer = MarketDataConsumer(
        config=config,
        consumer_factory=lambda _: mock_client,
        disconnect_backoff_seconds=0.01,
    )

    consumer.run(max_messages=1)

    assert consumer.kafka_errors == 1
    assert consumer.messages_consumed["market.ticks"] == 1
    assert consumer.messages_dispatched["market.ticks"] == 1


# ============================================================================
# 5. CREDENTIAL MASKING & LOG SAFETY
# ============================================================================

def test_sensitive_credentials_never_exposed_in_logs_or_repr():
    """Confirms access tokens are strictly masked in __repr__, __str__, and request headers."""
    secret_token = "SUPER_SECRET_COMPETITION_JWT_TOKEN_XYZ987"
    config = ExecutionConfig(
        bot_access_token=secret_token,
        api_base_url="http://192.168.65.157:3002",
    )

    # 1. Config repr and str masking
    assert secret_token not in repr(config)
    assert secret_token not in str(config)
    assert "***REDACTED***" in repr(config)
    assert "***REDACTED***" in str(config)

    # 2. Client header sanitization
    client = TradingApiClient(config)
    sanitized = client._sanitize_headers_for_logging({
        "Authorization": f"Bearer {secret_token}",
        "Content-Type": "application/json",
    })
    assert secret_token not in sanitized["Authorization"]
    assert sanitized["Authorization"] == "Bearer ***REDACTED***"


# ============================================================================
# 6. GRACEFUL SHUTDOWN BEHAVIOR
# ============================================================================

def test_clean_graceful_shutdown_behavior(mock_client: MagicMock):
    """LifecycleMonitor and HeartbeatWorker cleanly stop background threads within timeout."""
    worker = HeartbeatWorker(client=mock_client)
    monitor = LifecycleMonitor(client=mock_client, poll_interval_seconds=0.1)

    worker.start()
    monitor.start()

    assert worker.is_running is True
    assert monitor.is_running is True

    # Stop services
    worker.stop(timeout=1.0)
    monitor.stop(timeout=1.0)

    assert worker.is_running is False
    assert monitor.is_running is False
    assert not worker._thread.is_alive()
    assert not monitor._thread.is_alive()

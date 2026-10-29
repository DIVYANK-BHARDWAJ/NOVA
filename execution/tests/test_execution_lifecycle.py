"""Unit tests for asynchronous Simulation Lifecycle Monitor."""

import time
from unittest.mock import MagicMock
import pytest

from execution.client import ServerError, TradingApiClient
from execution.lifecycle import LifecycleMonitor
from execution.models import SimulationClock, SimulationStage


@pytest.fixture
def mock_client() -> MagicMock:
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
    return client


def test_lifecycle_monitor_poll_and_properties(mock_client: MagicMock):
    """LifecycleMonitor polls stage and clock, and accurately reflects properties."""
    monitor = LifecycleMonitor(client=mock_client)

    stage, clock = monitor.poll_once()
    assert stage is not None
    assert stage.stage == "preflight"
    assert clock is not None
    assert clock.current_day == 0

    assert monitor.is_preflight is True
    assert monitor.is_simulation_active is False
    assert monitor.is_completed is False
    assert monitor.poll_count == 1
    assert monitor.error_count == 0


def test_lifecycle_monitor_stage_transition_callback(mock_client: MagicMock):
    """Transitions between stages trigger on_stage_change callback."""
    stage_changes = []

    def _callback(new_s, old_s):
        stage_changes.append((new_s.stage, old_s.stage if old_s else None))

    monitor = LifecycleMonitor(client=mock_client, on_stage_change=_callback)

    # Poll 1: initial preflight stage
    monitor.poll_once()
    assert len(stage_changes) == 1
    assert stage_changes[0] == ("preflight", None)

    # Poll 2: identical stage -> no transition callback
    monitor.poll_once()
    assert len(stage_changes) == 1

    # Poll 3: platform moves to simulation running
    mock_client.get_stage.return_value = SimulationStage(
        stage="simulation",
        status="running",
        tickers=["ROUT", "SONA"],
        ticker_count=2,
        current_date="2026-03-26",
        days_completed=1,
        total_days=500,
        interval_seconds=4,
    )
    monitor.poll_once()
    assert len(stage_changes) == 2
    assert stage_changes[1] == ("simulation", "preflight")
    assert monitor.is_preflight is False
    assert monitor.is_simulation_active is True
    assert monitor.stage_transitions_count == 2


def test_lifecycle_monitor_completed_detection(mock_client: MagicMock):
    """Detects when simulation has reached completed status."""
    monitor = LifecycleMonitor(client=mock_client)
    mock_client.get_stage.return_value = SimulationStage(
        stage="simulation",
        status="completed",
        tickers=["ROUT"],
        ticker_count=1,
        current_date="2026-10-23",
        days_completed=500,
        total_days=500,
        interval_seconds=4,
    )
    monitor.poll_once()
    assert monitor.is_completed is True
    assert monitor.is_simulation_active is False


def test_lifecycle_monitor_error_resilience(mock_client: MagicMock):
    """Monitors catch network / server errors gracefully without throwing."""
    mock_client.get_stage.side_effect = ServerError("502 Bad Gateway")
    mock_client.get_simulation_clock.side_effect = ServerError("502 Bad Gateway")

    monitor = LifecycleMonitor(client=mock_client)
    stage, clock = monitor.poll_once()

    assert stage is None
    assert clock is None
    assert monitor.error_count == 2
    assert "502 Bad Gateway" in str(monitor.last_error)


def test_lifecycle_monitor_async_thread_lifecycle(mock_client: MagicMock):
    """LifecycleMonitor starts background daemon thread and shuts down cleanly."""
    monitor = LifecycleMonitor(client=mock_client, poll_interval_seconds=0.1)
    monitor.start()
    assert monitor.is_running is True

    time.sleep(0.35)
    assert monitor.poll_count >= 2

    summary = monitor.get_readiness_summary()
    assert summary["polls_completed"] >= 2
    assert summary["stage"] == "preflight"

    monitor.stop()
    assert monitor.is_running is False


def test_lifecycle_monitor_readiness_summary_with_components(mock_client: MagicMock):
    """Readiness summary accurately reflects Kafka connection and heartbeat acknowledgment status."""
    from execution.heartbeat import HeartbeatWorker

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

    # Before worker starts: worker inactive, heartbeats acked = 0, bot not ready
    summary = monitor.get_readiness_summary()
    assert summary["kafka_connected"] is True
    assert summary["kafka_partitions_assigned"] == 2
    assert summary["heartbeat_worker_active"] is False
    assert summary["heartbeats_acknowledged"] == 0
    assert summary["bot_ready"] is False

    # Start worker and acknowledge a pre-flight tick
    worker.start()
    try:
        worker.enqueue_heartbeat("1041")
        time.sleep(0.2)

        summary_ready = monitor.get_readiness_summary()
        assert summary_ready["heartbeat_worker_active"] is True
        assert summary_ready["heartbeats_acknowledged"] == 1
        assert summary_ready["bot_ready"] is True
    finally:
        worker.stop()

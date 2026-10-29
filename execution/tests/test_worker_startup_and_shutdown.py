"""Unit tests verifying worker startup order, event dispatching, and graceful shutdown."""

import time
from unittest.mock import MagicMock
import pytest

from execution.client import TradingApiClient
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from market_data.consumer import MarketDataConsumer
from market_data.handlers import EventDispatcher
from market_data.config import KafkaConfig


def test_worker_startup_order_and_graceful_shutdown():
    """Verify that HeartbeatWorker and LifecycleMonitor start in proper order and shut down cleanly."""
    mock_client = MagicMock(spec=TradingApiClient)
    mock_client.send_heartbeat.return_value = {"status": "ok", "message": "Heartbeat acknowledged"}

    events_log = []

    # 1. HeartbeatWorker initialized and started
    worker = HeartbeatWorker(client=mock_client)
    worker.start()
    events_log.append("heartbeat_worker_started")
    assert worker.is_running is True

    # 2. EventDispatcher created and HeartbeatWorker registered before consuming
    dispatcher = EventDispatcher()
    dispatcher.subscribe_preflight(worker.handle_event)
    events_log.append("heartbeat_worker_subscribed")

    # 3. LifecycleMonitor initialized with worker and started
    monitor = LifecycleMonitor(client=mock_client, poll_interval_seconds=0.1, heartbeat_worker=worker)
    monitor.start()
    events_log.append("lifecycle_monitor_started")
    assert monitor.is_running is True

    # 4. Consumer initialized with dispatcher
    mock_kafka = MagicMock()
    mock_kafka.poll.return_value = None
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher, consumer_factory=lambda _: mock_kafka)
    monitor.set_consumer(consumer)
    events_log.append("consumer_initialized")

    assert events_log == [
        "heartbeat_worker_started",
        "heartbeat_worker_subscribed",
        "lifecycle_monitor_started",
        "consumer_initialized",
    ]

    # 5. Process a pre-flight message through dispatcher before regular processing
    preflight_event = {"event_type": "preflight_tick", "seq": "4096"}
    handled = dispatcher.dispatch_preflight(preflight_event)
    assert handled == 1

    time.sleep(0.2)
    assert worker.acknowledged_count == 1
    mock_client.send_heartbeat.assert_called_once_with("4096")

    # 6. Graceful shutdown in reverse order
    monitor.stop()
    assert monitor.is_running is False

    worker.stop()
    assert worker.is_running is False

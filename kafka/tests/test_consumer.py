"""Tests for MarketDataConsumer integration with mock Kafka broker."""

import json
from typing import Any
from unittest.mock import MagicMock
import pytest

from market_data.config import KafkaConfig
from market_data.consumer import MarketDataConsumer, PlaceholderBrokerError
from market_data.handlers import EventDispatcher


class MockKafkaMessage:
    """Mock representing a confluent-kafka Message object."""

    def __init__(
        self,
        topic: str = "market.ticks",
        value: bytes | None = None,
        partition: int = 0,
        offset: int = 1,
        timestamp_ms: int = 1700000000000,
        error: Any = None,
    ):
        self._topic = topic
        self._value = value
        self._partition = partition
        self._offset = offset
        self._timestamp_ms = timestamp_ms
        self._error = error

    def topic(self) -> str:
        return self._topic

    def value(self) -> bytes | None:
        return self._value

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def timestamp(self) -> tuple[int, int]:
        return (1, self._timestamp_ms)

    def error(self) -> Any:
        return self._error


def test_placeholder_broker_connection_refusal():
    """Verify consumer refuses to connect to default placeholder broker address."""
    config = KafkaConfig(bootstrap_servers="<SERVER_IP>:9094")
    consumer = MarketDataConsumer(config=config)

    with pytest.raises(PlaceholderBrokerError) as exc_info:
        consumer.run(max_messages=1)
    assert "set to placeholder" in str(exc_info.value)


def test_consumer_end_to_end_synthetic_tick():
    """Verify consumer processes valid tick message through the full pipeline."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()
    received_ticks = []
    dispatcher.subscribe_tick(lambda e: received_ticks.append(e))

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    raw_payload = json.dumps({
        "symbol": "RELIANCE",
        "open": 2500.0,
        "high": 2550.0,
        "low": 2490.0,
        "close": 2540.0,
        "volume": 75000.0,
        "timestamp": 1700000000,
    }).encode("utf-8")

    msg = MockKafkaMessage(topic="market.ticks", value=raw_payload, partition=0, offset=10)

    success = consumer.process_raw_message(msg)
    assert success is True
    assert len(received_ticks) == 1
    event = received_ticks[0]
    assert event.symbol == "RELIANCE"
    assert event.close == 2540.0
    assert event.metadata.topic == "market.ticks"
    assert event.metadata.offset == 10
    assert consumer.messages_consumed["market.ticks"] == 1
    assert consumer.messages_dispatched["market.ticks"] == 1
    assert consumer.validation_errors["market.ticks"] == 0


def test_consumer_end_to_end_synthetic_news():
    """Verify consumer processes valid news message through the full pipeline."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()
    received_news = []
    dispatcher.subscribe_news(lambda e: received_news.append(e))

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    raw_payload = json.dumps({
        "headline": "Q3 net profit surges 25% year-on-year",
        "symbol": "TCS",
        "timestamp": "2026-10-29T11:00:00Z",
    }).encode("utf-8")

    msg = MockKafkaMessage(topic="market.news", value=raw_payload, partition=1, offset=22)

    success = consumer.process_raw_message(msg)
    assert success is True
    assert len(received_news) == 1
    event = received_news[0]
    assert event.headline == "Q3 net profit surges 25% year-on-year"
    assert event.symbol == "TCS"
    assert consumer.messages_consumed["market.news"] == 1
    assert consumer.messages_dispatched["market.news"] == 1


def test_consumer_invalid_messages_never_reach_handlers():
    """Verify invalid OHLC and malformed JSON never reach registered handlers."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()
    received_ticks = []
    dispatcher.subscribe_tick(lambda e: received_ticks.append(e))

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    # 1. Invalid OHLC (High < Low)
    bad_ohlc = json.dumps({
        "symbol": "INFY",
        "open": 1500.0,
        "high": 1400.0,  # Invalid: high < low
        "low": 1600.0,
        "close": 1550.0,
        "volume": 100.0,
    }).encode("utf-8")
    msg1 = MockKafkaMessage(topic="market.ticks", value=bad_ohlc)
    assert consumer.process_raw_message(msg1) is False
    assert consumer.validation_errors["market.ticks"] == 1

    # 2. Malformed JSON
    bad_json = b"{'symbol': INFY, invalid"
    msg2 = MockKafkaMessage(topic="market.ticks", value=bad_json)
    assert consumer.process_raw_message(msg2) is False
    assert consumer.decode_errors["market.ticks"] == 1

    # 3. Negative volume
    neg_vol = json.dumps({
        "symbol": "INFY",
        "open": 1500.0,
        "high": 1550.0,
        "low": 1480.0,
        "close": 1520.0,
        "volume": -50.0,
    }).encode("utf-8")
    msg3 = MockKafkaMessage(topic="market.ticks", value=neg_vol)
    assert consumer.process_raw_message(msg3) is False
    assert consumer.validation_errors["market.ticks"] == 2

    # None of these invalid messages must reach the handler!
    assert len(received_ticks) == 0


def test_consumer_kafka_error_handling():
    """Verify Kafka errors and partition EOF are handled gracefully."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    consumer = MarketDataConsumer(config=config)

    # 1. Partition EOF (code -191 or str _PARTITION_EOF)
    mock_eof_err = MagicMock()
    mock_eof_err.code.return_value = -191
    mock_eof_err.str.return_value = "_PARTITION_EOF"
    msg_eof = MockKafkaMessage(error=mock_eof_err)

    assert consumer.process_raw_message(msg_eof) is False
    assert consumer.kafka_errors == 0  # EOF is expected, not an error

    # 2. General Kafka error
    mock_general_err = MagicMock()
    mock_general_err.code.return_value = -187
    mock_general_err.str.return_value = "Broker transport failure"
    msg_err = MockKafkaMessage(error=mock_general_err)

    assert consumer.process_raw_message(msg_err) is False
    assert consumer.kafka_errors == 1


def test_consumer_loop_lifecycle():
    """Verify consumer poll loop runs and terminates on stop or max_messages."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")

    mock_client = MagicMock()
    # Return 2 messages then None
    tick_payload = json.dumps({
        "symbol": "TCS",
        "open": 3500.0,
        "high": 3550.0,
        "low": 3480.0,
        "close": 3520.0,
        "volume": 1000.0,
    }).encode("utf-8")
    mock_client.poll.side_effect = [
        MockKafkaMessage(topic="market.ticks", value=tick_payload),
        MockKafkaMessage(topic="market.ticks", value=tick_payload),
        None,
    ]

    consumer = MarketDataConsumer(
        config=config,
        consumer_factory=lambda _: mock_client,
    )

    consumer.run(max_messages=2)

    assert mock_client.subscribe.called
    assert mock_client.close.called
    assert consumer.messages_consumed["market.ticks"] == 2
    assert consumer.is_running is False


def test_exact_group_id_and_topics_passed_to_client():
    """Verify that exact configured group_id and topics are passed to confluent-kafka."""
    captured_config = {}
    mock_client = MagicMock()

    def mock_factory(conf_dict):
        nonlocal captured_config
        captured_config = conf_dict
        mock_client.poll.return_value = None
        return mock_client

    config = KafkaConfig(
        bootstrap_servers="10.0.0.1:9094",
        group_id="group-mn2651",
        topic_ticks="market.ticks",
        topic_news="market.news",
    )
    consumer = MarketDataConsumer(config=config, consumer_factory=mock_factory)
    consumer.run(max_messages=0)

    # Verify group.id passed to client
    assert captured_config["group.id"] == "group-mn2651"
    assert captured_config["bootstrap.servers"] == "10.0.0.1:9094"
    assert captured_config["security.protocol"] == "PLAINTEXT"
    # Verify consumer subscribed to both topics with rebalance callbacks
    call_args = mock_client.subscribe.call_args
    assert call_args is not None
    assert call_args[0][0] == ["market.ticks", "market.news"]
    assert "on_assign" in call_args[1]
    assert "on_revoke" in call_args[1]


def test_comprehensive_edge_case_data_validation_never_reaches_handler():
    """Verify empty payloads, malformed JSON, invalid UTF-8, arrays, strings,

    conflicting aliases, NaN, infinity, and extreme numbers never reach handlers.
    """
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()
    received_ticks: list[MarketTickEvent] = []
    dispatcher.subscribe_tick(lambda e: received_ticks.append(e))
    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    invalid_payloads = [
        # 1. Empty payload bytes
        b"",
        # 2. Whitespace payload bytes
        b"   ",
        # 3. Malformed JSON
        b'{"symbol": "TCS", "open": 3500.0, invalid_syntax',
        # 4. Invalid UTF-8
        b"\xff\xfe\x00\x00\xaa\xbb",
        # 5. JSON array instead of object
        b'[{"symbol": "TCS", "open": 3500.0}]',
        # 6. JSON string primitive
        b'"just a plain string"',
        # 7. Missing required OHLC fields (only open provided)
        b'{"symbol": "TCS", "open": 3500.0}',
        # 8. Conflicting aliases (open=100 vs o=105)
        b'{"symbol": "TCS", "open": 100.0, "o": 105.0, "high": 110.0, "low": 95.0, "close": 102.0, "volume": 100.0}',
        # 9. NaN price
        b'{"symbol": "TCS", "open": NaN, "high": 110.0, "low": 95.0, "close": 102.0, "volume": 100.0}',
        # 10. Infinity price
        b'{"symbol": "TCS", "open": Infinity, "high": 110.0, "low": 95.0, "close": 102.0, "volume": 100.0}',
        # 11. Extreme number price (> 1e9)
        b'{"symbol": "TCS", "open": 1e11, "high": 1.1e11, "low": 9e10, "close": 1e11, "volume": 100.0}',
        # 12. Extreme number volume (> 1e14)
        b'{"symbol": "TCS", "open": 100.0, "high": 110.0, "low": 95.0, "close": 102.0, "volume": 1e15}',
    ]

    for raw in invalid_payloads:
        msg = MockKafkaMessage(topic="market.ticks", value=raw)
        success = consumer.process_raw_message(msg)
        assert success is False, f"Payload unexpectedly succeeded: {raw!r}"

    # Verify that NONE of the invalid payloads ever reached the handler!
    assert len(received_ticks) == 0
    assert sum(consumer.decode_errors.values()) + sum(consumer.validation_errors.values()) == len(invalid_payloads)


def test_consumer_accepts_unknown_fields_without_dropping():
    """Verify valid messages with unknown/extra fields reach handlers with extra preserved."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()
    received_ticks: list[MarketTickEvent] = []
    dispatcher.subscribe_tick(lambda e: received_ticks.append(e))
    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    raw_payload = json.dumps({
        "ticker": "ROUT",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 5000.0,
        "quantum_indicator": 42.5,
        "cluster_node": "alpha-3",
    }).encode("utf-8")

    msg = MockKafkaMessage(topic="market.ticks", value=raw_payload)
    success = consumer.process_raw_message(msg)

    assert success is True
    assert len(received_ticks) == 1
    tick = received_ticks[0]
    assert tick.symbol == "ROUT"
    assert tick.extra.get("quantum_indicator") == 42.5
    assert tick.extra.get("cluster_node") == "alpha-3"


def test_distinct_timestamp_simulated_date_and_news_date_delivered_to_handlers():
    """Verify timestamp, simulated_date, and news_date arrive as distinct fields."""
    from datetime import date
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()
    received_news: list[MarketNewsEvent] = []
    dispatcher.subscribe_news(lambda e: received_news.append(e))
    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    raw_payload = json.dumps({
        "event_type": "news_flash",
        "timestamp": "2026-10-29T10:04:49.438Z",  # Feed emission clock
        "simulated_date": "2024-10-30",             # Simulation market date
        "data": {
            "ticker": "ROUT",
            "headline": "RouteWave expands operations",
            "date": "2024-10-28",                   # News article publication date
        },
    }).encode("utf-8")

    msg = MockKafkaMessage(topic="market.news", value=raw_payload)
    success = consumer.process_raw_message(msg)

    assert success is True
    assert len(received_news) == 1
    news = received_news[0]
    assert news.timestamp.year == 2026
    assert news.simulated_date == date(2024, 10, 30)
    assert news.news_date == date(2024, 10, 28)
    assert news.timestamp.date() != news.simulated_date
    assert news.simulated_date != news.news_date


def test_consumer_rebalance_callbacks_execution():
    """Verify partition rebalance callbacks execute cleanly."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    mock_client = MagicMock()

    consumer = MarketDataConsumer(config=config, consumer_factory=lambda _: mock_client)
    consumer.run(max_messages=0)

    # Extract the registered callbacks
    subscribe_call = mock_client.subscribe.call_args
    on_assign = subscribe_call[1]["on_assign"]
    on_revoke = subscribe_call[1]["on_revoke"]

    # Call callbacks with mock partitions (simulate Kafka cluster rebalance)
    mock_partitions = [MagicMock(topic="market.ticks", partition=0), MagicMock(topic="market.news", partition=1)]
    on_assign(mock_client, mock_partitions)
    on_revoke(mock_client, mock_partitions)


def test_consumer_recoverable_errors_lifecycle_does_not_busy_loop():
    """Verify recoverable Kafka errors do not cause busy loops and recover cleanly."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    mock_client = MagicMock()

    # 1. Transport error (recoverable)
    mock_err = MagicMock()
    mock_err.code.return_value = -187
    mock_err.str.return_value = "Broker transport failure"
    msg_err = MockKafkaMessage(error=mock_err)

    # 2. Poll timeout (None)
    # 3. Valid message when connection recovers
    valid_payload = json.dumps({
        "symbol": "TCS",
        "open": 3500.0,
        "high": 3550.0,
        "low": 3480.0,
        "close": 3520.0,
        "volume": 1000.0,
    }).encode("utf-8")
    msg_valid = MockKafkaMessage(topic="market.ticks", value=valid_payload)

    mock_client.poll.side_effect = [msg_err, None, msg_valid, None]

    consumer = MarketDataConsumer(config=config, consumer_factory=lambda _: mock_client)
    consumer.run(max_messages=2)

    assert consumer.kafka_errors == 1
    assert consumer.messages_consumed["market.ticks"] == 1
    assert consumer.messages_dispatched["market.ticks"] == 1
    assert mock_client.close.called


def test_consumer_null_payload_and_unrecognized_topic():
    """Verify null message payloads (tombstones) and unrecognized topics are safely handled."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    consumer = MarketDataConsumer(config=config)

    # Null value
    msg_null = MockKafkaMessage(topic="market.ticks", value=None)
    assert consumer.process_raw_message(msg_null) is False
    assert consumer.decode_errors["market.ticks"] == 1

    # Unrecognized topic
    msg_unrecognized = MockKafkaMessage(topic="market.unknown_topic", value=b'{"test": 1}')
    assert consumer.process_raw_message(msg_unrecognized) is False


def test_consumer_manual_commit_invoked_when_auto_commit_disabled():
    """Verify synchronous manual commit is called after processing when enable_auto_commit is False."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094", enable_auto_commit=False)
    mock_client = MagicMock()

    tick_payload = json.dumps({
        "symbol": "TCS",
        "open": 3500.0,
        "high": 3550.0,
        "low": 3480.0,
        "close": 3520.0,
        "volume": 1000.0,
    }).encode("utf-8")
    msg = MockKafkaMessage(topic="market.ticks", value=tick_payload)
    mock_client.poll.side_effect = [msg, None]

    consumer = MarketDataConsumer(config=config, consumer_factory=lambda _: mock_client)
    consumer.run(max_messages=1)

    # Verify manual commit was called synchronously
    assert mock_client.commit.called
    commit_call = mock_client.commit.call_args
    assert commit_call[1]["message"] == msg
    assert commit_call[1]["asynchronous"] is False


def test_consumer_reports_failure_when_all_handlers_fail():
    """Verify process_raw_message returns False when all registered handlers fail."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()

    def crashing_handler(e):
        raise RuntimeError("Complete strategy failure")

    dispatcher.subscribe_tick(crashing_handler)
    dispatcher.subscribe_news(crashing_handler)

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    tick_payload = json.dumps({
        "symbol": "ROUT",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 1000.0,
    }).encode("utf-8")
    tick_msg = MockKafkaMessage(topic="market.ticks", value=tick_payload)

    # Dispatch should report False because the only handler crashed
    success = consumer.process_raw_message(tick_msg)
    assert success is False
    assert consumer.messages_dispatched["market.ticks"] == 0

    news_payload = json.dumps({
        "headline": "Company announces merger",
        "symbol": "ROUT",
    }).encode("utf-8")
    news_msg = MockKafkaMessage(topic="market.news", value=news_payload)

    news_success = consumer.process_raw_message(news_msg)
    assert news_success is False
    assert consumer.messages_dispatched["market.news"] == 0


def test_partition_failure_blocks_later_commits_on_same_partition():
    """Verify that a later successful message cannot commit past an earlier failed message on the same partition."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094", enable_auto_commit=False)
    mock_client = MagicMock()
    dispatcher = EventDispatcher()

    # Strategy that fails on ROUT but succeeds on INFY
    def conditional_handler(e):
        if e.symbol == "ROUT":
            raise RuntimeError("Strategy crash on ROUT")

    dispatcher.subscribe_tick(conditional_handler)

    # Message 1: ROUT on partition 0 (FAILS)
    msg1_payload = json.dumps({"symbol": "ROUT", "open": 100.0, "high": 110.0, "low": 95.0, "close": 102.0, "volume": 1000.0}).encode("utf-8")
    msg1 = MockKafkaMessage(topic="market.ticks", partition=0, offset=10, value=msg1_payload)

    # Message 2: INFY on partition 0 (SUCCEEDS, but partition 0 is in failed state!)
    msg2_payload = json.dumps({"symbol": "INFY", "open": 1500.0, "high": 1520.0, "low": 1490.0, "close": 1510.0, "volume": 2000.0}).encode("utf-8")
    msg2 = MockKafkaMessage(topic="market.ticks", partition=0, offset=11, value=msg2_payload)

    # Message 3: INFY on partition 1 (SUCCEEDS on DIFFERENT partition!)
    msg3 = MockKafkaMessage(topic="market.ticks", partition=1, offset=5, value=msg2_payload)

    mock_client.poll.side_effect = [msg1, msg2, msg3, None]

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher, consumer_factory=lambda _: mock_client)
    consumer.run(max_messages=3)

    # Check commit calls
    committed_messages = [call[1]["message"] for call in mock_client.commit.call_args_list if "message" in call[1]]

    # msg1 (failed on partition 0) must NOT be committed
    assert msg1 not in committed_messages

    # msg2 (succeeded on partition 0) must NOT be committed because earlier msg1 failed on partition 0!
    assert msg2 not in committed_messages

    # msg3 (succeeded on partition 1) MUST be committed because partition 1 is clean!
    assert msg3 in committed_messages


def test_malformed_poison_pill_commits_in_manual_mode_to_prevent_infinite_loop():
    """Verify malformed JSON poison pill offset is committed to prevent infinite restart loop."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094", enable_auto_commit=False)
    mock_client = MagicMock()

    bad_json_msg = MockKafkaMessage(topic="market.ticks", partition=0, offset=20, value=b"invalid json")
    mock_client.poll.side_effect = [bad_json_msg, None]

    consumer = MarketDataConsumer(config=config, consumer_factory=lambda _: mock_client)
    consumer.run(max_messages=1)

    # Poison pill offset MUST be committed so consumer does not re-read and crash forever on restart
    assert mock_client.commit.called
    assert mock_client.commit.call_args[1]["message"] == bad_json_msg


def test_consumer_preflight_event_interception_before_normalization():
    """Verify preflight events with sequence numbers are intercepted before OHLCV normalization."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094")
    dispatcher = EventDispatcher()

    preflight_received = []
    ticks_received = []

    dispatcher.subscribe_preflight(lambda payload: preflight_received.append(payload))
    dispatcher.subscribe_tick(lambda tick: ticks_received.append(tick))

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher)

    # 1. Preflight tick with seq=54 (missing 'open' - would fail tick normalization if not intercepted)
    msg1 = MockKafkaMessage(
        topic="market.ticks",
        value=json.dumps({"event_type": "preflight_tick", "seq": 54, "timestamp": "2026-03-25T10:00:00Z"}).encode(),
    )
    res1 = consumer.process_raw_message(msg1)
    assert res1 is True
    assert len(preflight_received) == 1
    assert preflight_received[0]["seq"] == 54
    assert len(ticks_received) == 0

    # 2. Standby event with seq=1041 (missing 'open' - must be intercepted)
    msg2 = MockKafkaMessage(
        topic="market.ticks",
        value=json.dumps({"seq": 1041, "data": {"status": "STANDBY"}}).encode(),
    )
    res2 = consumer.process_raw_message(msg2)
    assert res2 is True
    assert len(preflight_received) == 2
    assert preflight_received[1]["seq"] == 1041
    assert len(ticks_received) == 0

    # 3. Standard market tick (must NOT be intercepted by preflight, must reach ticks_received)
    msg3 = MockKafkaMessage(
        topic="market.ticks",
        value=json.dumps({
            "symbol": "ROUT",
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 102.0,
            "volume": 50000.0,
        }).encode(),
    )
    res3 = consumer.process_raw_message(msg3)
    assert res3 is True
    assert len(preflight_received) == 2
    assert len(ticks_received) == 1
    assert ticks_received[0].symbol == "ROUT"


def test_kafka_partition_failure_isolation_allows_healthy_partitions_to_progress():
    """Verify that handler failure on one partition isolates only that partition, while healthy partitions commit offsets and progress."""
    config = KafkaConfig(bootstrap_servers="10.0.0.1:9094", enable_auto_commit=False)
    mock_client = MagicMock()
    dispatcher = EventDispatcher()

    # Fail handler on symbol 'BADTICK', succeed on all others
    def selective_handler(event):
        if event.symbol == "BADTICK":
            raise RuntimeError("Deliberate handler failure on BADTICK")

    dispatcher.subscribe_tick(selective_handler)

    # Partition 0: BADTICK (fails)
    p0_fail_payload = json.dumps({"symbol": "BADTICK", "open": 50.0, "high": 55.0, "low": 48.0, "close": 52.0, "volume": 1000.0}).encode("utf-8")
    msg_p0_fail = MockKafkaMessage(topic="market.ticks", partition=0, offset=100, value=p0_fail_payload)

    # Partition 0: GOODTICK later on same partition (must NOT commit because partition 0 failed)
    p0_good_payload = json.dumps({"symbol": "GOODTICK", "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 2000.0}).encode("utf-8")
    msg_p0_later = MockKafkaMessage(topic="market.ticks", partition=0, offset=101, value=p0_good_payload)

    # Partition 1: HEALTHY1 (succeeds, MUST commit)
    p1_payload = json.dumps({"symbol": "HEALTHY1", "open": 200.0, "high": 205.0, "low": 198.0, "close": 202.0, "volume": 3000.0}).encode("utf-8")
    msg_p1 = MockKafkaMessage(topic="market.ticks", partition=1, offset=200, value=p1_payload)

    # Partition 2: HEALTHY2 (succeeds, MUST commit)
    p2_payload = json.dumps({"symbol": "HEALTHY2", "open": 300.0, "high": 305.0, "low": 298.0, "close": 302.0, "volume": 4000.0}).encode("utf-8")
    msg_p2 = MockKafkaMessage(topic="market.ticks", partition=2, offset=300, value=p2_payload)

    mock_client.poll.side_effect = [msg_p0_fail, msg_p0_later, msg_p1, msg_p2, None]

    consumer = MarketDataConsumer(config=config, dispatcher=dispatcher, consumer_factory=lambda _: mock_client)
    consumer.run(max_messages=4)

    # Inspect all messages passed to client.commit(message=...)
    committed_messages = [call[1]["message"] for call in mock_client.commit.call_args_list if "message" in call[1]]

    # 1. Failed partition 0 messages must never be committed
    assert msg_p0_fail not in committed_messages
    assert msg_p0_later not in committed_messages

    # 2. Healthy partitions 1 and 2 must be committed
    assert msg_p1 in committed_messages
    assert msg_p2 in committed_messages

    # 3. Verify internal partition failure tracking
    assert ("market.ticks", 0) in consumer._failed_partitions
    assert ("market.ticks", 1) not in consumer._failed_partitions
    assert ("market.ticks", 2) not in consumer._failed_partitions

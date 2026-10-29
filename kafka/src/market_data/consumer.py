"""Kafka Ingestion Consumer for MANTHAN 2026.

Consumes streaming market data and corporate news, validates payloads, and
dispatches events to registered downstream listeners. Provides graceful
shutdown, metrics tracking, error resilience, and placeholder broker protection.
"""

from __future__ import annotations

import collections
import logging
import signal
from typing import Any, Callable, Optional

from market_data.config import KafkaConfig
from market_data.handlers import EventDispatcher
from market_data.models import KafkaMetadata
from market_data.normalizer import (
    NormalizationError,
    decode_json_payload,
    normalize_market_news,
    normalize_market_tick,
)
from market_data.validators import validate_market_news, validate_market_tick

logger = logging.getLogger(__name__)


class PlaceholderBrokerError(RuntimeError):
    """Raised when an attempt is made to connect to a placeholder broker IP."""
    pass


class MarketDataConsumer:
    """Resilient Kafka consumer for market ticks and corporate news."""

    def __init__(
        self,
        config: KafkaConfig,
        dispatcher: Optional[EventDispatcher] = None,
        consumer_factory: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        """Initialize consumer with configuration and optional dispatcher/factory.

        Args:
            config: KafkaConfig instance.
            dispatcher: EventDispatcher instance. Defaults to creating a new one.
            consumer_factory: Factory function returning a Kafka Consumer (used for testing/mocking).
        """
        self.config = config
        self.dispatcher = dispatcher or EventDispatcher()
        self._consumer_factory = consumer_factory
        self._consumer: Any = None
        self._running = False

        # Ingestion Metrics
        self.messages_consumed: dict[str, int] = collections.defaultdict(int)
        self.messages_dispatched: dict[str, int] = collections.defaultdict(int)
        self.decode_errors: dict[str, int] = collections.defaultdict(int)
        self.validation_errors: dict[str, int] = collections.defaultdict(int)
        self.kafka_errors: int = 0
        self._failed_partitions: set[tuple[str, int]] = set()
        self.assigned_partitions: list[Any] = []

    @property
    def is_running(self) -> bool:
        """Return running status of consumer loop."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Return True if consumer is active and assigned partitions."""
        return self._running and (len(self.assigned_partitions) > 0 or self._consumer is not None)

    def _create_kafka_consumer(self) -> Any:
        """Instantiate Kafka consumer client."""
        if self.config.is_placeholder_broker():
            raise PlaceholderBrokerError(
                f"Cannot connect to broker: KAFKA_BOOTSTRAP_SERVERS is set to placeholder "
                f"'{self.config.bootstrap_servers}'. Please configure the actual competition "
                f"broker address in kafka/.env before starting live consumption."
            )

        if self._consumer_factory is not None:
            return self._consumer_factory(self.config.to_confluent_kafka_dict())

        # Lazy import confluent_kafka only when creating real consumer
        try:
            from confluent_kafka import Consumer
            return Consumer(self.config.to_confluent_kafka_dict())
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "confluent-kafka is required to run the consumer. "
                "Install it using 'pip install confluent-kafka'."
            ) from exc

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown (SIGINT, SIGTERM)."""
        def _handle_signal(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received termination signal %s. Initiating graceful shutdown...", sig_name)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError):
            # Happens in non-main threads or specific test environments
            logger.debug("Signal handlers could not be installed (running outside main thread).")

    def process_raw_message(self, msg: Any) -> bool:
        """Process a single Kafka message through decode -> normalize -> validate -> dispatch.

        Args:
            msg: Kafka Message object (from confluent_kafka or mock).

        Returns:
            True if message was successfully dispatched, False otherwise.
        """
        if msg is None:
            return False

        # Handle Kafka-level errors
        err = msg.error()
        if err is not None:
            # Check partition EOF or recoverable error
            err_code = getattr(err, "code", lambda: None)()
            is_eof = getattr(err, "str", lambda: "")() == "_PARTITION_EOF" or str(err_code) == "-191"
            if is_eof:
                logger.debug("Reached end of partition for %s [%s]", msg.topic(), msg.partition())
                return False

            self.kafka_errors += 1
            logger.warning("Kafka error received: %s (code=%s)", err, err_code)
            return False

        topic = msg.topic() or "unknown"
        self.messages_consumed[topic] += 1

        # Extract Kafka Metadata
        ts_type, ts_val = None, None
        try:
            ts_result = msg.timestamp()
            if isinstance(ts_result, tuple) and len(ts_result) == 2:
                ts_type, ts_val = ts_result
        except Exception:
            pass

        metadata = KafkaMetadata(
            topic=topic,
            partition=msg.partition() if callable(getattr(msg, "partition", None)) else -1,
            offset=msg.offset() if callable(getattr(msg, "offset", None)) else -1,
            timestamp_ms=ts_val,
            timestamp_type=ts_type,
        )

        # 1. Message Decoding & JSON Parsing
        raw_val = msg.value()
        if raw_val is None:
            logger.warning("Received message with null payload on topic %s", topic)
            self.decode_errors[topic] += 1
            return False

        try:
            payload = decode_json_payload(raw_val)
        except NormalizationError as exc:
            self.decode_errors[topic] += 1
            logger.warning("Failed to decode message on topic '%s': %s", topic, exc)
            return False

        # 2. Preflight / Heartbeat Event Detection
        ev_type = str(payload.get("event_type") or payload.get("event") or payload.get("type") or "").strip().lower()
        is_preflight = (
            ev_type.startswith("preflight")
            or "heartbeat" in ev_type
            or ("seq" in payload and "open" not in payload)
            or ("seq" in payload and "data" in payload and isinstance(payload.get("data"), dict) and payload.get("data", {}).get("status") == "STANDBY")
        )
        if is_preflight:
            return self._handle_preflight_record(payload, metadata, topic)

        # 3. Topic Routing, Normalization & Validation
        if topic == self.config.topic_ticks:
            return self._handle_tick_record(payload, metadata, topic)
        elif topic == self.config.topic_news:
            return self._handle_news_record(payload, metadata, topic)
        else:
            logger.warning("Received message on unrecognized topic '%s'. Skipping.", topic)
            return False

    def _handle_preflight_record(self, payload: dict[str, Any], metadata: KafkaMetadata, topic: str) -> bool:
        """Process and dispatch preflight tick/news events for heartbeat acknowledgement."""
        seq = (
            payload.get("seq")
            or payload.get("sequence")
            or (payload.get("data", {}) or {}).get("seq")
            or (payload.get("data", {}) or {}).get("sequence")
        )
        logger.info("Preflight event received on %s [%s:%s]: seq=%s", topic, metadata.partition, metadata.offset, seq)
        self.dispatcher.dispatch_preflight(payload)
        self.messages_dispatched[topic] += 1
        return True

    def _handle_tick_record(self, payload: dict[str, Any], metadata: KafkaMetadata, topic: str) -> bool:
        """Normalize, validate, and dispatch a market tick message."""
        try:
            event = normalize_market_tick(payload, metadata)
        except NormalizationError as exc:
            self.decode_errors[topic] += 1
            logger.warning("Normalization error for market tick: %s", exc)
            return False

        val_res = validate_market_tick(event)
        if not val_res.is_valid:
            self.validation_errors[topic] += 1
            logger.warning("Validation failed for tick [%s]: %s", event.symbol, val_res.error_message)
            return False

        success_count = self.dispatcher.dispatch_tick(event)
        if len(self.dispatcher.tick_handlers) > 0 and success_count == 0:
            logger.error("All registered tick handlers failed for symbol %s on %s [%s]", event.symbol, metadata.topic, metadata.partition)
            self._failed_partitions.add((metadata.topic, metadata.partition))
            return False

        self.messages_dispatched[topic] += 1
        return True

    def _handle_news_record(self, payload: dict[str, Any], metadata: KafkaMetadata, topic: str) -> bool:
        """Normalize, validate, and dispatch a corporate news message."""
        try:
            event = normalize_market_news(payload, metadata)
        except NormalizationError as exc:
            self.decode_errors[topic] += 1
            logger.warning("Normalization error for market news: %s", exc)
            return False

        val_res = validate_market_news(event)
        if not val_res.is_valid:
            self.validation_errors[topic] += 1
            logger.warning("Validation failed for news event: %s", val_res.error_message)
            return False

        success_count = self.dispatcher.dispatch_news(event)
        if len(self.dispatcher.news_handlers) > 0 and success_count == 0:
            logger.error("All registered news handlers failed for news headline: %s on %s [%s]", event.headline[:60], metadata.topic, metadata.partition)
            self._failed_partitions.add((metadata.topic, metadata.partition))
            return False

        self.messages_dispatched[topic] += 1
        return True

    def run(self, poll_timeout_seconds: float = 1.0, max_messages: Optional[int] = None) -> None:
        """Run continuous consumption loop until stopped or max_messages reached.

        Args:
            poll_timeout_seconds: Kafka poll timeout in seconds.
            max_messages: Optional upper limit on processed messages (useful for tests/batches).
        """
        self.config.validate()
        self._consumer = self._create_kafka_consumer()

        topics = [self.config.topic_ticks, self.config.topic_news]
        logger.info(
            "Subscribing to topics %s with group ID '%s' on %s",
            topics,
            self.config.group_id,
            self.config.bootstrap_servers,
        )

        def _on_partitions_assigned(c: Any, partitions: list[Any]) -> None:
            self.assigned_partitions = list(partitions)
            logger.info("Partition rebalance: Assigned %d partition(s): %s", len(partitions), partitions)

        def _on_partitions_revoked(c: Any, partitions: list[Any]) -> None:
            logger.info("Partition rebalance: Revoked %d partition(s): %s", len(partitions), partitions)
            for p in partitions:
                t = getattr(p, "topic", "")
                pid = getattr(p, "partition", -1)
                self._failed_partitions.discard((t, pid))
            self.assigned_partitions = [
                p for p in self.assigned_partitions
                if not (getattr(p, "topic", "") == getattr(p, "topic", None) and getattr(p, "partition", -1) in [getattr(rev, "partition", -1) for rev in partitions])
            ]

        try:
            self._consumer.subscribe(
                topics,
                on_assign=_on_partitions_assigned,
                on_revoke=_on_partitions_revoked,
            )
        except TypeError:
            # Fallback for simple mock objects in testing
            self._consumer.subscribe(topics)

        self._setup_signal_handlers()
        self._running = True
        total_processed = 0

        try:
            while self._running:
                if max_messages is not None and total_processed >= max_messages:
                    logger.info("Reached max_messages limit (%d). Exiting loop.", max_messages)
                    break

                msg = self._consumer.poll(timeout=poll_timeout_seconds)
                if msg is None:
                    continue

                self.process_raw_message(msg)
                total_processed += 1

                # Safe manual commit path when auto-commit is disabled
                if not self.config.enable_auto_commit and self._consumer is not None:
                    # Do not commit Kafka error objects or partition EOFs
                    if msg.error() is None and hasattr(self._consumer, "commit") and callable(getattr(self._consumer, "commit")):
                        topic = msg.topic() or ""
                        part_id = msg.partition() if callable(getattr(msg, "partition", None)) else -1
                        part_key = (topic, part_id)

                        if part_key in self._failed_partitions:
                            logger.warning(
                                "Partition %s [%s] has an unhandled subscriber failure; "
                                "skipping offset commit to prevent skipping failed records on restart.",
                                topic,
                                part_id,
                            )
                        else:
                            try:
                                self._consumer.commit(message=msg, asynchronous=False)
                            except Exception as exc:
                                logger.error("Manual synchronous offset commit failed for %s [%s]: %s", topic, part_id, exc)

        except Exception as exc:
            logger.critical("Unexpected error in consumer loop: %s", exc, exc_info=True)
            raise
        finally:
            self.close()

    def stop(self) -> None:
        """Request graceful shutdown of the consumer loop."""
        self._running = False

    def close(self) -> None:
        """Cleanly close the underlying Kafka consumer and log summary metrics."""
        self._running = False
        if self._consumer is not None:
            try:
                logger.info("Closing Kafka consumer cleanly...")
                self._consumer.close()
            except Exception as exc:
                logger.warning("Exception during Kafka consumer close: %s", exc)
            finally:
                self._consumer = None

        logger.info(
            "Consumer shutdown complete. Summary Metrics: consumed=%s dispatched=%s "
            "decode_errors=%s validation_errors=%s kafka_errors=%d",
            dict(self.messages_consumed),
            dict(self.messages_dispatched),
            dict(self.decode_errors),
            dict(self.validation_errors),
            self.kafka_errors,
        )

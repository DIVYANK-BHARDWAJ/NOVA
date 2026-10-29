"""Asynchronous Pre-Flight Heartbeat Worker for MANTHAN 2026.

Listens for preflight tick events, extracts sequence numbers, deduplicates them,
and acknowledges them via POST /api/v1/bot/heartbeat without blocking the Kafka
polling loop or execution pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Optional

from execution.client import TradingApiClient, TradingApiError

logger = logging.getLogger(__name__)


class HeartbeatWorker:
    """Asynchronous worker that acknowledges pre-flight tick heartbeats."""

    def __init__(self, client: TradingApiClient, max_queue_size: int = 1000) -> None:
        """Initialize HeartbeatWorker.

        Args:
            client: TradingApiClient instance.
            max_queue_size: Maximum backlog of pending heartbeats.
        """
        self.client = client
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue_size)
        self._in_flight_seqs: set[str] = set()
        self._acknowledged_seqs: set[str] = set()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Metrics
        self.enqueued_count: int = 0
        self.acknowledged_count: int = 0
        self.failure_count: int = 0
        self.duplicate_count: int = 0

    @property
    def is_running(self) -> bool:
        """Return running status of heartbeat worker."""
        return self._running

    def start(self) -> None:
        """Start background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="HeartbeatWorker", daemon=True)
        self._thread.start()
        logger.info("HeartbeatWorker started.")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background worker thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(
            "HeartbeatWorker stopped. Metrics: enqueued=%d acked=%d failed=%d dups=%d",
            self.enqueued_count,
            self.acknowledged_count,
            self.failure_count,
            self.duplicate_count,
        )

    def enqueue_heartbeat(self, seq: int | str) -> bool:
        """Enqueue a sequence number for acknowledgement.

        Thread-safe, non-blocking. Prevents duplicate acknowledgements.

        Args:
            seq: Sequence number from preflight tick.

        Returns:
            True if enqueued, False if duplicate, invalid, or queue is full.
        """
        seq_str = str(seq).strip()
        if not seq_str:
            return False

        with self._lock:
            if seq_str in self._acknowledged_seqs or seq_str in self._in_flight_seqs:
                self.duplicate_count += 1
                logger.debug("Heartbeat seq %s already queued or acknowledged. Skipping.", seq_str)
                return False

            try:
                self._queue.put_nowait(seq_str)
                self._in_flight_seqs.add(seq_str)
                self.enqueued_count += 1
                return True
            except queue.Full:
                logger.error("Heartbeat queue is full! Dropping heartbeat seq %s.", seq_str)
                return False

    def handle_event(self, event: Any) -> bool:
        """Generic event handler that extracts seq if event is a preflight or heartbeat event.

        Args:
            event: Event object or dict.

        Returns:
            True if heartbeat was enqueued, False otherwise.
        """
        seq: Optional[str] = None

        if isinstance(event, dict):
            event_type = event.get("event_type") or event.get("type") or event.get("event")
            if event_type and (str(event_type).lower().startswith("preflight") or "heartbeat" in str(event_type).lower()):
                seq = event.get("seq") or event.get("sequence") or (event.get("data", {}) or {}).get("seq")
            elif "seq" in event and "preflight" in str(event).lower():
                seq = event.get("seq") or event.get("sequence") or (event.get("data", {}) or {}).get("seq")
        else:
            # Check attribute-based models (e.g. MarketTickEvent)
            extra = getattr(event, "extra", {})
            if isinstance(extra, dict):
                ev_type = extra.get("event_type") or extra.get("type") or extra.get("event")
                if ev_type and (str(ev_type).lower().startswith("preflight") or "heartbeat" in str(ev_type).lower()):
                    seq = extra.get("seq") or extra.get("sequence") or (extra.get("data", {}) or {}).get("seq")
            raw = getattr(event, "raw_payload", {})
            if seq is None and isinstance(raw, dict):
                ev_type = raw.get("event_type") or raw.get("type") or raw.get("event")
                if ev_type and (str(ev_type).lower().startswith("preflight") or "heartbeat" in str(ev_type).lower()):
                    seq = raw.get("seq") or raw.get("sequence") or (raw.get("data", {}) or {}).get("seq")

        if seq is not None:
            return self.enqueue_heartbeat(seq)
        return False

    def _worker_loop(self) -> None:
        """Continuous background loop processing heartbeat acknowledgements."""
        while self._running:
            try:
                seq = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.client.send_heartbeat(seq)
                with self._lock:
                    self._in_flight_seqs.discard(seq)
                    self._acknowledged_seqs.add(seq)
                self.acknowledged_count += 1
                logger.debug("Successfully sent pre-flight heartbeat for seq=%s", seq)
            except TradingApiError as exc:
                self.failure_count += 1
                with self._lock:
                    self._in_flight_seqs.discard(seq)
                logger.warning("Failed to send heartbeat for seq=%s: %s", seq, exc)
            except Exception as exc:
                self.failure_count += 1
                with self._lock:
                    self._in_flight_seqs.discard(seq)
                logger.error("Unexpected error sending heartbeat for seq=%s: %s", seq, exc)
            finally:
                self._queue.task_done()

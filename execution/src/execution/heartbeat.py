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

    def __init__(
        self,
        client: TradingApiClient,
        max_queue_size: int = 1000,
        max_retries: int = 3,
    ) -> None:
        """Initialize HeartbeatWorker.

        Args:
            client: TradingApiClient instance.
            max_queue_size: Maximum backlog of pending heartbeats.
            max_retries: Maximum attempts permitted for a single heartbeat sequence.
        """
        self.client = client
        self.max_retries = max(1, int(max_retries))
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue_size)
        self._in_flight_seqs: set[str] = set()
        self._acknowledged_seqs: set[str] = set()
        self._attempt_counts: dict[str, int] = {}
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
        """Start background daemon thread. Thread-safe and idempotent."""
        with self._lock:
            if self._running or (self._thread is not None and self._thread.is_alive()):
                logger.warning("[HEARTBEAT:WORKER] HeartbeatWorker already running; ignoring duplicate start()")
                return
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, name="HeartbeatWorker", daemon=True)
            self._thread.start()
            logger.info("[HEARTBEAT:START] HeartbeatWorker started.")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background worker thread."""
        with self._lock:
            self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(
            "[HEARTBEAT:STOP] HeartbeatWorker stopped. Metrics: enqueued=%d acked=%d failed=%d dups=%d",
            self.enqueued_count,
            self.acknowledged_count,
            self.failure_count,
            self.duplicate_count,
        )

    def enqueue_heartbeat(self, seq: int | str) -> bool:
        """Enqueue a sequence number for acknowledgement.

        Thread-safe, non-blocking. Prevents duplicate acknowledgements.
        Enforces max_retries limit to prevent unbounded retries.

        Args:
            seq: Sequence number from preflight tick.

        Returns:
            True if enqueued, False if duplicate, invalid, retry-exhausted, or queue full.
        """
        seq_str = str(seq).strip()
        if not seq_str:
            return False

        with self._lock:
            if seq_str in self._acknowledged_seqs or seq_str in self._in_flight_seqs:
                self.duplicate_count += 1
                logger.debug("[HEARTBEAT:DUPLICATE] Heartbeat seq %s already queued or acknowledged. Skipping.", seq_str)
                return False

            if self._attempt_counts.get(seq_str, 0) >= self.max_retries:
                logger.warning(
                    "[HEARTBEAT:DROP] Heartbeat seq %s has exceeded max retries (%d). Dropping without re-queue.",
                    seq_str,
                    self.max_retries,
                )
                self.failure_count += 1
                return False

            try:
                self._queue.put_nowait(seq_str)
                self._in_flight_seqs.add(seq_str)
                self.enqueued_count += 1
                logger.debug("[HEARTBEAT:ENQUEUE] Heartbeat enqueued for seq=%s (backlog: %d)", seq_str, self._queue.qsize())
                return True
            except queue.Full:
                logger.error("[HEARTBEAT:OVERFLOW] Heartbeat queue is full! Dropping heartbeat seq %s.", seq_str)
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

    def get_health_status(self) -> dict[str, Any]:
        """Return health diagnostics for HeartbeatWorker."""
        with self._lock:
            return {
                "is_running": self._running,
                "enqueued_count": self.enqueued_count,
                "acknowledged_count": self.acknowledged_count,
                "failure_count": self.failure_count,
                "duplicate_count": self.duplicate_count,
                "queue_backlog": self._queue.qsize(),
                "in_flight_count": len(self._in_flight_seqs),
                "is_healthy": self._running and (self.failure_count == 0 or self.acknowledged_count > 0),
            }

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
                    self._attempt_counts.pop(seq, None)
                self.acknowledged_count += 1
                logger.info("[HEARTBEAT:ACK] Successfully acknowledged heartbeat seq=%s (total acked: %d)", seq, self.acknowledged_count)
            except TradingApiError as exc:
                self.failure_count += 1
                with self._lock:
                    self._in_flight_seqs.discard(seq)
                    current_attempts = self._attempt_counts.get(seq, 0) + 1
                    self._attempt_counts[seq] = current_attempts
                if current_attempts < self.max_retries:
                    logger.warning(
                        "[HEARTBEAT:RETRY] Failed to send heartbeat for seq=%s (attempt %d/%d): %s",
                        seq,
                        current_attempts,
                        self.max_retries,
                        exc,
                    )
                else:
                    logger.error(
                        "[HEARTBEAT:DROP] Exhausted max retries (%d/%d) for heartbeat seq=%s: %s",
                        current_attempts,
                        self.max_retries,
                        seq,
                        exc,
                    )
            except Exception as exc:
                self.failure_count += 1
                with self._lock:
                    self._in_flight_seqs.discard(seq)
                    current_attempts = self._attempt_counts.get(seq, 0) + 1
                    self._attempt_counts[seq] = current_attempts
                logger.error(
                    "[HEARTBEAT:ERROR] Unexpected error sending heartbeat seq=%s (attempt %d/%d): %s",
                    seq,
                    current_attempts,
                    self.max_retries,
                    exc,
                )
            finally:
                self._queue.task_done()

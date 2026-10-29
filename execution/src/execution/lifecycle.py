"""Asynchronous Simulation Lifecycle Monitor for MANTHAN 2026.

Periodically queries zero-delay endpoints (GET /api/v1/stage and GET /api/v1/simulation/clock)
to detect platform lifecycle transitions (preflight -> ready -> simulation -> completed).
Notifies registered callbacks and execution adapters without stalling the Kafka ingestion loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from execution.client import TradingApiClient, TradingApiError
from execution.models import SimulationClock, SimulationStage

logger = logging.getLogger(__name__)

StageChangeCallback = Callable[[SimulationStage, Optional[SimulationStage]], None]
ClockTickCallback = Callable[[SimulationClock], None]


class LifecycleMonitor:
    """Monitors simulation lifecycle stage and clock progression asynchronously."""

    def __init__(
        self,
        client: TradingApiClient,
        poll_interval_seconds: float = 3.0,
        on_stage_change: Optional[StageChangeCallback] = None,
        on_clock_tick: Optional[ClockTickCallback] = None,
        heartbeat_worker: Optional[Any] = None,
        consumer: Optional[Any] = None,
    ) -> None:
        """Initialize LifecycleMonitor.

        Args:
            client: TradingApiClient instance.
            poll_interval_seconds: Polling interval in seconds (default: 3.0s).
            on_stage_change: Callback invoked on stage or status transition.
            on_clock_tick: Callback invoked on clock progression.
            heartbeat_worker: Optional HeartbeatWorker instance for readiness reporting.
            consumer: Optional MarketDataConsumer instance for readiness reporting.
        """
        self.client = client
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.on_stage_change = on_stage_change
        self.on_clock_tick = on_clock_tick
        self.heartbeat_worker = heartbeat_worker
        self.consumer = consumer

        self._current_stage: Optional[SimulationStage] = None
        self._current_clock: Optional[SimulationClock] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Observability Metrics
        self.poll_count: int = 0
        self.error_count: int = 0
        self.stage_transitions_count: int = 0
        self.last_poll_timestamp: Optional[float] = None
        self.last_error: Optional[str] = None

    def set_heartbeat_worker(self, worker: Any) -> None:
        """Set or update HeartbeatWorker reference."""
        self.heartbeat_worker = worker

    def set_consumer(self, consumer: Any) -> None:
        """Set or update MarketDataConsumer reference."""
        self.consumer = consumer

    @property
    def is_running(self) -> bool:
        """Return True if background monitor loop is active."""
        return self._running

    @property
    def current_stage(self) -> Optional[SimulationStage]:
        """Return the most recently queried simulation stage."""
        with self._lock:
            return self._current_stage

    @property
    def current_clock(self) -> Optional[SimulationClock]:
        """Return the most recently queried simulation clock."""
        with self._lock:
            return self._current_clock

    @property
    def is_preflight(self) -> bool:
        """Return True if platform is in preflight, standby, or ready stage."""
        with self._lock:
            if self._current_stage is None:
                return False
            stage_str = str(self._current_stage.stage).lower()
            status_str = str(self._current_stage.status).lower()
            return stage_str in ("preflight", "standby", "ready") or status_str in ("preflight", "standby", "ready")

    @property
    def is_simulation_active(self) -> bool:
        """Return True if platform is actively running the simulation."""
        with self._lock:
            if self._current_stage is None:
                return False
            return (
                str(self._current_stage.stage).lower() == "simulation"
                and str(self._current_stage.status).lower() == "running"
            )

    @property
    def is_completed(self) -> bool:
        """Return True if simulation has completed."""
        with self._lock:
            if self._current_stage and str(self._current_stage.status).lower() == "completed":
                return True
            if self._current_clock and str(self._current_clock.status).lower() == "completed":
                return True
            return False

    def start(self) -> None:
        """Start background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="LifecycleMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "LifecycleMonitor started (polling interval: %.1fs).",
            self.poll_interval_seconds,
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background monitoring thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(
            "LifecycleMonitor stopped. Metrics: polls=%d transitions=%d errors=%d",
            self.poll_count,
            self.stage_transitions_count,
            self.error_count,
        )

    def poll_once(self) -> tuple[Optional[SimulationStage], Optional[SimulationClock]]:
        """Perform a single synchronous query of stage and clock endpoints.

        Returns:
            Tuple of (SimulationStage, SimulationClock), or None values on error.
        """
        stage: Optional[SimulationStage] = None
        clock: Optional[SimulationClock] = None

        # 1. Query Stage
        try:
            stage = self.client.get_stage()
        except TradingApiError as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.warning("[LIFECYCLE] Failed to query /api/v1/stage: %s", exc)
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("[LIFECYCLE] Unexpected error querying stage: %s", exc)

        # 2. Query Clock
        try:
            clock = self.client.get_simulation_clock()
        except TradingApiError as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.warning("[LIFECYCLE] Failed to query /api/v1/simulation/clock: %s", exc)
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("[LIFECYCLE] Unexpected error querying clock: %s", exc)

        self.poll_count += 1
        self.last_poll_timestamp = time.time()

        # 3. Detect Stage Transition
        if stage is not None:
            prev_stage: Optional[SimulationStage] = None
            stage_changed = False

            with self._lock:
                prev_stage = self._current_stage
                if prev_stage is None or (
                    stage.stage != prev_stage.stage or stage.status != prev_stage.status
                ):
                    stage_changed = True
                    self.stage_transitions_count += 1
                    self._current_stage = stage

            if stage_changed:
                prev_str = f"'{prev_stage.stage}' ({prev_stage.status})" if prev_stage else "None"
                curr_str = f"'{stage.stage}' ({stage.status})"
                logger.info(
                    "[LIFECYCLE] Stage transition detected: %s -> %s | Days: %d/%d (Date: %s)",
                    prev_str,
                    curr_str,
                    stage.days_completed,
                    stage.total_days,
                    stage.current_date,
                )
                if self.on_stage_change is not None:
                    try:
                        self.on_stage_change(stage, prev_stage)
                    except Exception as exc:
                        logger.error("[LIFECYCLE] Error in on_stage_change callback: %s", exc, exc_info=True)

        # 4. Detect Clock Progression
        if clock is not None:
            with self._lock:
                self._current_clock = clock

            if self.on_clock_tick is not None:
                try:
                    self.on_clock_tick(clock)
                except Exception as exc:
                    logger.error("[LIFECYCLE] Error in on_clock_tick callback: %s", exc, exc_info=True)

        return stage, clock

    def get_readiness_summary(self) -> dict[str, Any]:
        """Return structured summary of lifecycle and readiness state."""
        with self._lock:
            stage_val = self._current_stage.stage if self._current_stage else None
            status_val = self._current_stage.status if self._current_stage else None
            curr_day = self._current_clock.current_day if self._current_clock else None
            total_days = self._current_clock.total_days if self._current_clock else None
            sim_date = self._current_clock.simulated_date if self._current_clock else None

        hb_active = bool(self.heartbeat_worker and getattr(self.heartbeat_worker, "is_running", False))
        hb_acks = getattr(self.heartbeat_worker, "acknowledged_count", 0) if self.heartbeat_worker else 0
        hb_enqueued = getattr(self.heartbeat_worker, "enqueued_count", 0) if self.heartbeat_worker else 0
        hb_failures = getattr(self.heartbeat_worker, "failure_count", 0) if self.heartbeat_worker else 0

        kafka_running = bool(self.consumer and getattr(self.consumer, "is_running", False))
        assigned = getattr(self.consumer, "assigned_partitions", []) if self.consumer else []
        kafka_partitions = len(assigned)

        bot_ready = kafka_running and hb_active

        return {
            "stage": stage_val,
            "status": status_val,
            "is_preflight": self.is_preflight,
            "is_simulation_active": self.is_simulation_active,
            "is_completed": self.is_completed,
            "current_day": curr_day,
            "total_days": total_days,
            "simulated_date": sim_date,
            "polls_completed": self.poll_count,
            "transitions_detected": self.stage_transitions_count,
            "poll_errors": self.error_count,
            "last_error": self.last_error,
            # Component readiness
            "kafka_connected": kafka_running,
            "kafka_partitions_assigned": kafka_partitions,
            "heartbeat_worker_active": hb_active,
            "heartbeats_acknowledged": hb_acks,
            "heartbeats_enqueued": hb_enqueued,
            "heartbeats_failed": hb_failures,
            "bot_ready": bot_ready,
        }

    def _monitor_loop(self) -> None:
        """Continuous background polling loop."""
        while self._running:
            t0 = time.time()
            self.poll_once()

            if self.poll_count % 10 == 0:
                summary = self.get_readiness_summary()
                logger.info(
                    "[LIFECYCLE STATUS] Stage: '%s' (%s) | Day: %s/%s | Kafka: %s (%d parts) | Heartbeats Acked: %d | Bot Ready: %s",
                    summary["stage"],
                    summary["status"],
                    summary["current_day"],
                    summary["total_days"],
                    "CONNECTED" if summary["kafka_connected"] else "DISCONNECTED",
                    summary["kafka_partitions_assigned"],
                    summary["heartbeats_acknowledged"],
                    summary["bot_ready"],
                )

            elapsed = time.time() - t0
            sleep_time = max(0.1, self.poll_interval_seconds - elapsed)

            # Sleep in small increments for responsive stop()
            end_sleep = time.time() + sleep_time
            while self._running and time.time() < end_sleep:
                time.sleep(0.1)

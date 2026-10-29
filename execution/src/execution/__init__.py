"""Execution and Trading Adapter for MANTHAN 2026.

Connects authorized trading decisions from Person 3's risk gateway to the official
competition REST API, manages order lifecycle, protects against duplicate execution,
and supports portfolio accounting and preflight heartbeat acknowledgement.
"""

from __future__ import annotations

from execution.config import ExecutionConfig
from execution.client import TradingApiClient, TradingApiError
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from execution.models import (
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    ExecutionEvent,
    StockInfo,
    SimulationStage,
    SimulationClock,
    Portfolio,
    CashBalance,
)

__all__ = [
    "ExecutionConfig",
    "TradingApiClient",
    "TradingApiError",
    "ExecutionAdapter",
    "HeartbeatWorker",
    "LifecycleMonitor",
    "OrderAction",
    "OrderIntent",
    "OrderResult",
    "OrderStatus",
    "ExecutionEvent",
    "StockInfo",
    "SimulationStage",
    "SimulationClock",
    "Portfolio",
    "CashBalance",
]

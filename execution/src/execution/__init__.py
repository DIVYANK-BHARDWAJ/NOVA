"""Execution and Trading Adapter for MANTHAN 2026.

Connects authorized trading decisions from Person 3's risk gateway to the official
competition REST API, manages order lifecycle, protects against duplicate execution,
and supports portfolio accounting and preflight heartbeat acknowledgement.
"""

from __future__ import annotations

from execution.accounting import (
    AccountLedger,
    CashReconciliationReport,
    FullReconciliationReport,
    PositionLedgerEntry,
    PositionReconciliationReport,
    UncertainOrderRecord,
    reconcile_cash,
    reconcile_positions,
)
from execution.adapter import ExecutionAdapter
from execution.client import TradingApiClient, TradingApiError
from execution.config import ExecutionConfig
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from execution.models import (
    CashBalance,
    ExecutionEvent,
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Portfolio,
    SimulationClock,
    SimulationStage,
    StockInfo,
)

__all__ = [
    "AccountLedger",
    "CashBalance",
    "CashReconciliationReport",
    "ExecutionAdapter",
    "ExecutionConfig",
    "ExecutionEvent",
    "FullReconciliationReport",
    "HeartbeatWorker",
    "LifecycleMonitor",
    "OrderAction",
    "OrderIntent",
    "OrderResult",
    "OrderStatus",
    "Portfolio",
    "PositionLedgerEntry",
    "PositionReconciliationReport",
    "SimulationClock",
    "SimulationStage",
    "StockInfo",
    "TradingApiClient",
    "TradingApiError",
    "UncertainOrderRecord",
    "reconcile_cash",
    "reconcile_positions",
]

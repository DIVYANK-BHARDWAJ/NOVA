"""Portfolio accounting and reconciliation helpers.

Compares local internal ledger states against official exchange portfolio responses,
detecting position variances, cash discrepancies, and unaccounted fills. Kept strictly
isolated from REST client transport logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from execution.models import CashBalance, Portfolio


@dataclass(frozen=True)
class PositionReconciliationReport:
    """Report summarizing comparison between local and remote positions."""
    matched: dict[str, int]
    discrepancies: dict[str, dict[str, int]]  # ticker -> {'local': X, 'remote': Y, 'diff': Z}
    missing_in_local: dict[str, int]
    missing_in_remote: dict[str, int]
    is_synchronized: bool


@dataclass(frozen=True)
class CashReconciliationReport:
    """Report summarizing comparison between local and remote cash."""
    local_cash: float
    remote_cash: float
    variance: float
    is_synchronized: bool


def reconcile_positions(
    local_positions: dict[str, int],
    remote_portfolio: Portfolio,
) -> PositionReconciliationReport:
    """Reconcile local position tracker against remote exchange portfolio.

    Args:
        local_positions: Mapping of ticker to quantity from internal ledger.
        remote_portfolio: Official Portfolio instance from competition API.

    Returns:
        PositionReconciliationReport instance.
    """
    remote_dict: dict[str, int] = {}
    for h in remote_portfolio.holdings:
        remote_dict[h.ticker.upper()] = h.quantity

    local_dict: dict[str, int] = {k.upper(): v for k, v in local_positions.items()}

    matched: dict[str, int] = {}
    discrepancies: dict[str, dict[str, int]] = {}
    missing_in_local: dict[str, int] = {}
    missing_in_remote: dict[str, int] = {}

    all_tickers = set(local_dict.keys()) | set(remote_dict.keys())

    for ticker in all_tickers:
        loc_qty = local_dict.get(ticker)
        rem_qty = remote_dict.get(ticker)

        if loc_qty is not None and rem_qty is not None:
            if loc_qty == rem_qty:
                matched[ticker] = loc_qty
            else:
                discrepancies[ticker] = {
                    "local": loc_qty,
                    "remote": rem_qty,
                    "diff": rem_qty - loc_qty,
                }
        elif loc_qty is None and rem_qty is not None:
            if rem_qty != 0:
                missing_in_local[ticker] = rem_qty
        elif loc_qty is not None and rem_qty is None:
            if loc_qty != 0:
                missing_in_remote[ticker] = loc_qty

    is_sync = len(discrepancies) == 0 and len(missing_in_local) == 0 and len(missing_in_remote) == 0

    return PositionReconciliationReport(
        matched=matched,
        discrepancies=discrepancies,
        missing_in_local=missing_in_local,
        missing_in_remote=missing_in_remote,
        is_synchronized=is_sync,
    )


def reconcile_cash(
    local_cash: float,
    remote_cash: CashBalance,
    tolerance: float = 0.01,
) -> CashReconciliationReport:
    """Reconcile local cash tracker against remote available cash balance.

    Args:
        local_cash: Internal tracked cash amount.
        remote_cash: CashBalance from GET /api/v1/portfolio/cash.
        tolerance: Allowed decimal variance for rounding.

    Returns:
        CashReconciliationReport instance.
    """
    variance = remote_cash.available_cash - local_cash
    is_sync = abs(variance) <= tolerance

    return CashReconciliationReport(
        local_cash=local_cash,
        remote_cash=remote_cash.available_cash,
        variance=variance,
        is_synchronized=is_sync,
    )

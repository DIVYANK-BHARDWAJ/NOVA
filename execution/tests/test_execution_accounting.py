"""Unit tests for portfolio and cash reconciliation accounting."""

import pytest

from execution.accounting import reconcile_cash, reconcile_positions
from execution.models import CashBalance, Portfolio, PortfolioHolding


def test_reconcile_positions_perfect_match():
    """Identical local and remote holdings produce synchronized report."""
    local = {"ROUT": 100, "BLUX": 50}
    remote = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=100000.0,
        cash_balance=100000.0,
        portfolio_market_value=15000.0,
        total_wealth=115000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[
            PortfolioHolding(ticker="ROUT", quantity=100),
            PortfolioHolding(ticker="BLUX", quantity=50),
        ],
    )

    report = reconcile_positions(local, remote)
    assert report.is_synchronized is True
    assert report.matched == {"ROUT": 100, "BLUX": 50}
    assert len(report.discrepancies) == 0


def test_reconcile_positions_discrepancies_and_missing():
    """Detects quantity mismatches, missing local positions, and missing remote positions."""
    local = {"ROUT": 100, "MOTR": 20}  # MOTR not in remote
    remote = Portfolio(
        team_id="team_1",
        team_name="Nova",
        available_cash=100000.0,
        cash_balance=100000.0,
        portfolio_market_value=15000.0,
        total_wealth=115000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_pnl=0.0,
        total_charges=0.0,
        total_taxes=0.0,
        holdings=[
            PortfolioHolding(ticker="ROUT", quantity=80),  # Discrepancy: local 100, remote 80
            PortfolioHolding(ticker="BLUX", quantity=30),  # Missing in local
        ],
    )

    report = reconcile_positions(local, remote)
    assert report.is_synchronized is False
    assert "ROUT" in report.discrepancies
    assert report.discrepancies["ROUT"]["local"] == 100
    assert report.discrepancies["ROUT"]["remote"] == 80
    assert report.discrepancies["ROUT"]["diff"] == -20

    assert report.missing_in_local == {"BLUX": 30}
    assert report.missing_in_remote == {"MOTR": 20}


def test_reconcile_cash():
    """Reconciles local cash against remote balance within tolerance."""
    remote_cash = CashBalance(
        team_id="team_1",
        team_name="Nova",
        available_cash=500000.0,
        cash_balance=500000.0,
        capital_remaining=500000.0,
        margin_locked=0.0,
        initial_capital=1000000.0,
        realized_pnl=0.0,
    )

    # 1. Matching cash
    report_match = reconcile_cash(500000.0, remote_cash)
    assert report_match.is_synchronized is True
    assert report_match.variance == 0.0

    # 2. Minor rounding within tolerance
    report_minor = reconcile_cash(499999.995, remote_cash, tolerance=0.01)
    assert report_minor.is_synchronized is True

    # 3. Discrepancy exceeding tolerance
    report_diff = reconcile_cash(480000.0, remote_cash)
    assert report_diff.is_synchronized is False
    assert report_diff.variance == 20000.0

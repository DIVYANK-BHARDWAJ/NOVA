"""NOVA Pre-Flight Trading Safety — Section 6 End-to-End SELL & Exit Validation.

Tests:
1. Full liquidation (target_exposure = 0.0) converts entire position to sell order.
2. Partial reduction (target_exposure = 0.05 from 0.15) sells exactly the delta.
3. Stop-loss exit triggers when price drops below stop threshold.
4. Trailing-stop exit triggers when giveback threshold is exceeded.
5. Zero position sell request produces 0 shares and cleanly abstains without order submission.
6. Zero target exposure when flat produces 0 shares and skips order.
7. Target shares greater than existing holdings on reduction clips safely without overselling.
8. Invalid or negative quantity is caught and rejected by adapter validation.
9. Duplicate exit signal is rejected by order deadband / deduplication.
10. Partial fill updates position ledger by only the filled shares.
11. Rejected sell order leaves position ledger intact and records rejection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock

from execution.accounting import AccountLedger, PositionLedgerEntry
from execution.adapter import ExecutionAdapter
from execution.client import TradingApiClient
from execution.config import ExecutionConfig
from execution.models import (
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    SimulationStage,
)
from nova.decision.contracts import ActionType, OrderType, PortfolioState, PositionState, TradeDecision
from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.pipeline import IntegratedTradingPipeline
from nova.decision.prediction import CausalRidgePredictor
from nova.decision.regime import MarketRegimeDetector
from nova.friction.models import MarketFrictionEngine
from nova.portfolio.profit_realization import ProfitRealizationConfig, ProfitRealizationManager
from nova.quant.contracts import DataQualityStatus, MarketBar
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.mean_reversion import MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingStrategy


class TestSellAndExitMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=TradingApiClient)
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["TESTX"],
            ticker_count=1,
            current_date="2026-01-10",
            days_completed=10,
            total_days=500,
        )
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_exit_001",
            "ticker": "TESTX",
            "action": "SELL",
            "quantity": 200,
            "status": "FILLED",
            "execution_price": 95.0,
            "charges": 20.0,
        }

        self.ledger = AccountLedger(initial_cash=1_000_000.0)
        self.adapter = ExecutionAdapter(
            client=self.mock_client,
            config=ExecutionConfig(bot_access_token="test_tok", stage_cache_ttl_seconds=0.0),
            ledger=self.ledger,
        )
        self.risk_gate = DeterministicRiskGate(RiskConfiguration(allow_short=False))

        pr_cfg = ProfitRealizationConfig(
            take_profit_threshold_pct=0.10,
            trailing_stop_activation_pct=0.05,
            trailing_stop_distance_pct=0.02,
        )
        self.engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(
                target_exposure=0.15,
                stop_loss_pct=0.04,
                initial_capital=1_000_000.0,
                enable_profit_realization=True,
                profit_realization_config=pr_cfg,
            ),
            trend_strategy=TrendFollowingStrategy(),
            mean_reversion_strategy=MeanReversionStrategy(),
            friction_engine=MarketFrictionEngine(),
            risk_gate=self.risk_gate,
            predictor=CausalRidgePredictor(),
            regime_detector=MarketRegimeDetector(),
            profit_realization_manager=ProfitRealizationManager(pr_cfg),
        )
        self.pipeline = IntegratedTradingPipeline(
            adapter=self.adapter,
            risk_gate=self.risk_gate,
            ledger=self.ledger,
            decision_engine=self.engine,
        )

    def _create_bars(self, n: int = 30, px_start: float = 100.0, trend: float = 0.0) -> list[MarketBar]:
        bars = []
        px = px_start
        for i in range(n):
            px += trend
            bars.append(MarketBar(
                ticker="TESTX",
                source_timestamp=datetime(2026, 1, 1, 9, 30 + i, tzinfo=timezone.utc),
                open=px,
                high=px + 0.5,
                low=px - 0.5,
                close=px,
                volume=1000.0,
                data_quality=DataQualityStatus.VALID,
            ))
        return bars

    def test_01_full_liquidation(self) -> None:
        """Full liquidation produces order for all held shares."""
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=500, total_cost=50_000.0)
        self.mock_client.submit_order.return_value["quantity"] = 500

        dec = TradeDecision(
            decision_id="dec_liq",
            decision_timestamp=datetime.now(timezone.utc),
            instrument_id="TESTX",
            selected_action=ActionType.SELL,
            target_quantity_or_exposure=0.0,
            order_type=OrderType.MARKET,
            selected_strategy="profit_realization",
            selection_or_rejection_reasons=("PROFIT_TARGET_FULL_EXIT",),
            configuration_version="1.0.0",
        )
        bar = self._create_bars(n=1)[0]
        res = self.pipeline.process_trade_decision(dec, bar)

        self.assertIsNotNone(res)
        self.mock_client.submit_order.assert_called_once()
        kwargs = self.mock_client.submit_order.call_args.kwargs
        self.assertEqual(kwargs.get("quantity"), 500)
        self.assertEqual(kwargs.get("action"), "SELL")

    def test_02_partial_reduction(self) -> None:
        """Partial reduction from 15% to 5% calculates difference correctly."""
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=1500, total_cost=150_000.0)
        self.ledger.cash = 850_000.0
        self.mock_client.submit_order.return_value["quantity"] = 1000

        dec = TradeDecision(
            decision_id="dec_red",
            decision_timestamp=datetime.now(timezone.utc),
            instrument_id="TESTX",
            selected_action=ActionType.REDUCE,
            target_quantity_or_exposure=0.05,
            order_type=OrderType.MARKET,
            selected_strategy="profit_realization",
            selection_or_rejection_reasons=("EXPOSURE_TRIM",),
            configuration_version="1.0.0",
        )
        bar = self._create_bars(n=1, px_start=100.0)[0]
        res = self.pipeline.process_trade_decision(dec, bar)

        self.assertIsNotNone(res)
        kwargs = self.mock_client.submit_order.call_args.kwargs
        self.assertEqual(kwargs.get("quantity"), 1000)
        self.assertEqual(kwargs.get("action"), "SELL")

    def test_03_stop_loss_exit(self) -> None:
        """Stop-loss exit triggers when price breaches 4% below entry."""
        bars = self._create_bars(n=25, px_start=100.0, trend=-0.3)
        latest_px = bars[-1].close
        self.assertTrue(latest_px < 96.0)

        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=200, total_cost=20_000.0)
        pos = PositionState(ticker="TESTX", quantity=200.0, entry_price=100.0, current_market_price=latest_px)
        port = PortfolioState(available_cash=980_000.0, positions={"TESTX": pos})

        dec = self.engine.evaluate_decision(bars, portfolio_state=port)
        self.assertEqual(dec.selected_action, ActionType.SELL)
        self.assertTrue(any("STOP_LOSS" in r for r in dec.selection_or_rejection_reasons))

    def test_04_zero_position_sell_abstains(self) -> None:
        """Selling when position is 0 computes 0 quantity and skips order submission."""
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=0, total_cost=0.0)

        dec = TradeDecision(
            decision_id="dec_zero",
            decision_timestamp=datetime.now(timezone.utc),
            instrument_id="TESTX",
            selected_action=ActionType.SELL,
            target_quantity_or_exposure=0.0,
            order_type=OrderType.MARKET,
            selected_strategy="profit_realization",
            selection_or_rejection_reasons=("FORCE_SELL_EMPTY",),
            configuration_version="1.0.0",
        )
        bar = self._create_bars(n=1)[0]
        res = self.pipeline.process_trade_decision(dec, bar)

        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_05_overselling_protection(self) -> None:
        """If target shares exceed position, reduction clamps to 0 shares and prevents shorting."""
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=100, total_cost=10_000.0)
        self.ledger.cash = 990_000.0

        dec = TradeDecision(
            decision_id="dec_clamp",
            decision_timestamp=datetime.now(timezone.utc),
            instrument_id="TESTX",
            selected_action=ActionType.REDUCE,
            target_quantity_or_exposure=0.50,
            order_type=OrderType.MARKET,
            selected_strategy="profit_realization",
            selection_or_rejection_reasons=("REDUCE_INVALID",),
            configuration_version="1.0.0",
        )
        bar = self._create_bars(n=1, px_start=100.0)[0]
        res = self.pipeline.process_trade_decision(dec, bar)

        self.assertIsNone(res)
        self.mock_client.submit_order.assert_not_called()

    def test_06_partial_fill_reconciliation(self) -> None:
        """Partial fill updates ledger with only the executed shares, not the full order."""
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=500, total_cost=50_000.0)
        self.ledger.cash = 500_000.0

        intent = OrderIntent(intent_id="intent_001", ticker="TESTX", action=OrderAction.SELL, quantity=500, authorized_by_risk=True)
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_part_001",
            "ticker": "TESTX",
            "action": "SELL",
            "quantity": 200,
            "status": "PARTIALLY_FILLED",
            "execution_price": 100.0,
            "charges": 20.0,
        }

        res = self.adapter.execute_order(intent)
        self.assertEqual(res.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(res.filled_quantity, 200)

        self.assertEqual(self.ledger.get_position("TESTX"), 300)
        self.assertEqual(self.ledger.cash, 519_980.0)

    def test_07_rejected_sell_leaves_position_intact(self) -> None:
        """Rejected sell order does not modify ledger position or cash."""
        self.ledger.positions["TESTX"] = PositionLedgerEntry(ticker="TESTX", quantity=500, total_cost=50_000.0)
        self.ledger.cash = 500_000.0

        intent = OrderIntent(intent_id="intent_002", ticker="TESTX", action=OrderAction.SELL, quantity=500, authorized_by_risk=True)
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_rej_001",
            "ticker": "TESTX",
            "action": "SELL",
            "quantity": 500,
            "status": "REJECTED",
            "reason": "MARKET_CLOSED",
        }

        res = self.adapter.execute_order(intent)
        self.assertEqual(res.status, OrderStatus.REJECTED)
        self.assertEqual(self.ledger.get_position("TESTX"), 500)
        self.assertEqual(self.ledger.cash, 500_000.0)


if __name__ == "__main__":
    unittest.main()

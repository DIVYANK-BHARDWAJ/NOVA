"""NOVA Step H — End-to-End Pipeline Integration & Risk Enforcement Tests.

Tests the complete integrated pipeline:
Kafka market data -> EventDispatcher -> Person 2 Alpha Strategy -> Person 3 Decision/Risk Gateway -> ExecutionAdapter -> Trading API -> AccountLedger -> Reconciliation.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import unittest
from unittest.mock import MagicMock, patch

from execution.adapter import ExecutionAdapter
from execution.accounting import AccountLedger
from execution.client import TradingApiClient, TradingApiError
from execution.config import ExecutionConfig
from execution.models import (
    OrderAction,
    OrderIntent,
    OrderResult,
    OrderStatus,
    SimulationStage,
    StockInfo,
)
from market_data.handlers import EventDispatcher
from market_data.models import KafkaMetadata, MarketTickEvent
from nova.decision.contracts import ActionType, OrderType, TradeDecision
from nova.decision.pipeline import IntegratedTradingPipeline
from nova.quant.contracts import (
    CandidateSignal,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    ScoreSemantics,
    SignalDirection,
    StrategyEvaluation,
)
from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
from nova.research.kafka_connector import KafkaStrategyConnector
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration, RiskGateResult


class TestEndToEndPipelineIntegration(unittest.TestCase):
    """End-to-end integration verifying real module connections and strict safety boundaries."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=TradingApiClient)
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["INFY", "ALPHACORP"],
            days_completed=10,
            total_days=500,
            current_date="2026-01-15",
            is_sandbox_stream=False,
        )
        self.mock_client.get_stocks.return_value = [
            StockInfo(symbol="INFY", ticker="INFY", name="Infosys Ltd"),
            StockInfo(symbol="ALPHACORP", ticker="ALPHACORP", name="Alpha Corp"),
        ]
        self.mock_client.submit_order.return_value = {
            "order_id": "ord_mock_123",
            "ticker": "INFY",
            "action": "BUY",
            "quantity": 100,
            "status": "FILLED",
            "execution_price": 1500.0,
            "charges": 20.0,
        }

        self.config = ExecutionConfig(
            api_base_url="http://mock-api:3002",
            bot_access_token="mock_token",
            permitted_stages=("competition", "development"),
            allow_development_stage=True,
            stage_cache_ttl_seconds=60.0,
            stocks_cache_ttl_seconds=60.0,
        )
        self.ledger = AccountLedger(initial_cash=1_000_000.0)
        self.adapter = ExecutionAdapter(
            client=self.mock_client,
            config=self.config,
            ledger=self.ledger,
        )
        self.risk_gate = DeterministicRiskGate(RiskConfiguration(
            max_position_exposure=0.25,
            max_drawdown_limit=0.15,
            allow_short=False,
            min_cash_buffer_fraction=0.05,
            max_staleness_seconds=300.0,
        ))
        self.pipeline = IntegratedTradingPipeline(
            adapter=self.adapter,
            risk_gate=self.risk_gate,
            ledger=self.ledger,
            default_target_exposure=0.15,
        )

    def test_full_pipeline_kafka_tick_to_ledger_fill(self) -> None:
        """Trace: MarketTickEvent -> EventDispatcher -> KafkaStrategyConnector -> Risk Gate -> ExecutionAdapter -> Ledger."""
        dispatcher = EventDispatcher()
        strategy = EMACrossoverStrategy(EMACrossoverConfig(fast_window=3, slow_window=5))
        connector = KafkaStrategyConnector(
            strategy=strategy,
            on_candidate_signal=self.pipeline.process_candidate_signal,
        )
        dispatcher.subscribe_tick(connector.handle_market_tick)

        base_time = datetime.now(timezone.utc) - timedelta(seconds=70)
        # Send 4 flat warmup ticks (3 fast, 5 slow needs 5 bars)
        prices = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1050.0, 1100.0]
        for i, px in enumerate(prices):
            ts = base_time + timedelta(seconds=i * 10)
            event = MarketTickEvent(
                symbol="INFY",
                open=px,
                high=px + 2.0,
                low=px - 2.0,
                close=px,
                volume=10000.0,
                timestamp=ts,
                simulated_date=ts.date(),
                metadata=KafkaMetadata(topic="market.ticks", partition=0, offset=i),
            )
            dispatcher.dispatch_tick(event)

        # Bullish crossover must trigger order submission and update ledger
        self.assertGreater(self.pipeline.signals_processed, 0)
        self.assertGreater(self.pipeline.orders_authorized, 0)
        self.mock_client.submit_order.assert_called()

        # Check that ledger recorded the fill
        self.assertEqual(self.ledger.get_position("INFY"), 270)
        self.assertLess(self.ledger.cash, 1_000_000.0)

    def test_risk_gate_drawdown_circuit_breaker_vetoes_order(self) -> None:
        """When drawdown exceeds limit (15%), Risk Gate vetoes BUY candidate and 0 orders reach API."""
        # Simulate drawdown by setting peak portfolio value much higher
        self.ledger.set_peak_portfolio_value(2_000_000.0)
        # Current portfolio value is 1,000,000 -> 50% drawdown (> 15% limit)

        now = datetime.now(timezone.utc)
        candidate = CandidateSignal(
            strategy_id="test_strat",
            strategy_version="1.0.0",
            ticker="INFY",
            direction=SignalDirection.BULLISH,
            signal_score=1.0,
            score_semantics=ScoreSemantics.UNSPECIFIED,
            evaluation_timestamp=now,
        )
        bar = MarketBar(
            ticker="INFY",
            source_timestamp=now,
            open=1500.0,
            high=1510.0,
            low=1490.0,
            close=1505.0,
            volume=50000.0,
            data_quality=DataQualityStatus.VALID,
        )

        result = self.pipeline.process_candidate_signal(candidate, bar)
        self.assertIsNone(result)
        self.assertEqual(self.pipeline.orders_vetoed, 1)
        # Verify Trading API was NEVER called
        self.mock_client.submit_order.assert_not_called()

    def test_risk_gate_stale_data_vetoes_order(self) -> None:
        """When market bar is older than max_staleness_seconds (300s), order is vetoed."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=600)
        candidate = CandidateSignal(
            strategy_id="test_strat",
            strategy_version="1.0.0",
            ticker="INFY",
            direction=SignalDirection.BULLISH,
            signal_score=1.0,
            score_semantics=ScoreSemantics.UNSPECIFIED,
            evaluation_timestamp=old_time,
        )
        bar = MarketBar(
            ticker="INFY",
            source_timestamp=old_time,
            open=1500.0,
            high=1510.0,
            low=1490.0,
            close=1505.0,
            volume=50000.0,
            data_quality=DataQualityStatus.VALID,
        )

        result = self.pipeline.process_candidate_signal(candidate, bar)
        self.assertIsNone(result)
        self.assertEqual(self.pipeline.orders_vetoed, 1)
        self.mock_client.submit_order.assert_not_called()

    def test_risk_gate_cash_buffer_violation_vetoes_order(self) -> None:
        """When cash drops below min_cash_buffer_fraction (5%), BUY is vetoed."""
        # Drain ledger cash to 2% of portfolio
        self.ledger.cash = 20_000.0
        self.ledger.set_peak_portfolio_value(1_000_000.0)

        now = datetime.now(timezone.utc)
        candidate = CandidateSignal(
            strategy_id="test_strat",
            strategy_version="1.0.0",
            ticker="INFY",
            direction=SignalDirection.BULLISH,
            signal_score=1.0,
            score_semantics=ScoreSemantics.UNSPECIFIED,
            evaluation_timestamp=now,
        )
        bar = MarketBar(
            ticker="INFY",
            source_timestamp=now,
            open=1500.0,
            high=1510.0,
            low=1490.0,
            close=1505.0,
            volume=50000.0,
            data_quality=DataQualityStatus.VALID,
        )

        result = self.pipeline.process_candidate_signal(candidate, bar)
        self.assertIsNone(result)
        self.assertEqual(self.pipeline.orders_vetoed, 1)
        self.mock_client.submit_order.assert_not_called()

    def test_unauthorized_order_intent_rejected_at_adapter_boundary(self) -> None:
        """Direct OrderIntent submission with authorized_by_risk=False must be rejected immediately."""
        intent = OrderIntent(
            intent_id="intent_unauth_test",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            authorized_by_risk=False,
        )
        result = self.adapter.execute_order(intent)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("UNAUTHORIZED_BY_RISK_GATEWAY", result.rejection_reason or "")
        self.mock_client.submit_order.assert_not_called()

    def test_invalid_universe_ticker_rejected_before_api(self) -> None:
        """Symbols not in active tradable universe are rejected and never reach API."""
        intent = OrderIntent(
            intent_id="intent_bad_ticker",
            ticker="NON_EXISTENT_CO",
            action=OrderAction.BUY,
            quantity=50,
            authorized_by_risk=True,
        )
        result = self.adapter.execute_order(intent)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("INVALID_TICKER", result.rejection_reason or "")
        self.mock_client.submit_order.assert_not_called()

    def test_inactive_stage_blocks_order(self) -> None:
        """Closed or paused simulation stage blocks order submission."""
        self.mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="paused",
            tickers=["INFY", "ALPHACORP"],
            days_completed=10,
            total_days=500,
            current_date="2026-01-15",
            is_sandbox_stream=False,
        )
        self.adapter.invalidate_stage_cache()

        intent = OrderIntent(
            intent_id="intent_paused_stage",
            ticker="INFY",
            action=OrderAction.BUY,
            quantity=50,
            authorized_by_risk=True,
        )
        result = self.adapter.execute_order(intent)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("STAGE_NOT_PERMITTED", result.rejection_reason or "")
        self.mock_client.submit_order.assert_not_called()

    def test_execute_trade_decision_contract_c_abstentions_and_entries(self) -> None:
        """Verify TradeDecision (Interface Contract C) execution."""
        now = datetime.now(timezone.utc)
        # 1. NO_TRADE decision -> returns None, 0 API calls
        abstain_dec = TradeDecision(
            decision_id="dec_abstain",
            decision_timestamp=now,
            instrument_id="INFY",
            selected_action=ActionType.NO_TRADE,
            target_quantity_or_exposure=0.0,
            order_type=OrderType.NONE,
            selected_strategy="ensemble_test",
            selection_or_rejection_reasons=("NO_EDGE",),
            configuration_version="1.0.0",
        )
        res_abstain = self.adapter.execute_trade_decision(
            decision=abstain_dec,
            current_price=1500.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertIsNone(res_abstain)
        self.mock_client.submit_order.assert_not_called()

        # 2. BUY decision -> sized and submitted
        buy_dec = TradeDecision(
            decision_id="dec_buy",
            decision_timestamp=now,
            instrument_id="INFY",
            selected_action=ActionType.BUY,
            target_quantity_or_exposure=0.15,  # 15% of 1,000,000 = 150,000 / 1500 = 100 shares
            order_type=OrderType.MARKET,
            selected_strategy="ensemble_test",
            selection_or_rejection_reasons=("POSITIVE_EDGE",),
            configuration_version="1.0.0",
        )
        res_buy = self.adapter.execute_trade_decision(
            decision=buy_dec,
            current_price=1500.0,
            current_portfolio_value=1_000_000.0,
        )
        self.assertIsNotNone(res_buy)
        self.assertEqual(res_buy.status, OrderStatus.FILLED)
        self.mock_client.submit_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()

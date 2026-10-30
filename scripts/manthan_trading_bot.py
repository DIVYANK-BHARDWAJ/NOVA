#!/usr/bin/env python3
"""MANTHAN 2026 — Official NOVA Autonomous Trading Bot Runner.

Coordinates:
1. REST API authentication & initial stage/clock/portfolio queries (Person 1).
2. Kafka market data ingestion (market.ticks & market.news) under group 'group-mn2651'.
3. Asynchronous pre-flight heartbeat worker (POST /api/v1/bot/heartbeat) with bounded retries.
4. Continuous zero-delay simulation lifecycle monitor (GET /api/v1/stage and /api/v1/simulation/clock).
5. Quantitative alpha signal generation (Person 2 EMACrossoverStrategy).
6. Person 3 Risk Gateway authorization enforcement (pre-trade risk boundary).
7. Execution Adapter arming, AccountLedger portfolio tracking, and reconciliation (Step F).
8. Observability, fault resilience, and health diagnostics (Step G).

Usage:
    python scripts/manthan_trading_bot.py
    python scripts/manthan_trading_bot.py --env-file kafka/.env
    python scripts/manthan_trading_bot.py --validate-only
    python scripts/manthan_trading_bot.py --mock
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Optional
import urllib.error
import urllib.request

# Ensure all repository source roots are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
KAFKA_SRC = REPO_ROOT / "kafka" / "src"
EXECUTION_SRC = REPO_ROOT / "execution" / "src"
NOVA_SRC = REPO_ROOT / "src"

for d in (KAFKA_SRC, EXECUTION_SRC, NOVA_SRC, REPO_ROOT):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from market_data.config import ConfigValidationError, KafkaConfig
from market_data.consumer import MarketDataConsumer, PlaceholderBrokerError
from market_data.handlers import (
    DiagnosticLoggingNewsHandler,
    DiagnosticLoggingTickHandler,
    EventDispatcher,
)
from execution.config import ExecutionConfig
from execution.client import TradingApiClient, TradingApiError
from execution.heartbeat import HeartbeatWorker
from execution.lifecycle import LifecycleMonitor
from execution.adapter import ExecutionAdapter
from execution.accounting import AccountLedger
from execution.models import OrderAction, OrderIntent, OrderStatus, OrderResult

# Beginner's Guide Keyword Dictionaries (Section 5, Strategy 3)
BULLISH_KEYWORDS = {
    "PROFIT", "SURGE", "GROWTH", "RECORD", "BEAT", "EXPANSION", "DIVIDEND", "PARTNERSHIP",
}
BEARISH_KEYWORDS = {
    "LOSS", "PLUNGE", "DROP", "FRAUD", "SCRUTINY", "RESIGNATION", "PROBE", "DOWNGRADE",
}

logger = logging.getLogger("nova.bot")


def configure_logging(level_name: str = "INFO") -> None:
    """Configure unified logging format."""
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def send_http_request(
    url: str,
    method: str = "GET",
    token: str = "",
    payload: dict | None = None,
    timeout: float = 3.0,
) -> dict | None:
    """Send authenticated HTTP request using standard library urllib."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "NOVA-Trading-Bot/1.0",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = response.read()
            return json.loads(res_data.decode("utf-8")) if res_data else {}
    except Exception as exc:
        logger.warning("[HTTP] %s %s failed: %s", method, url, exc)
        return None


class ManthanTradingBot:
    """Trading bot interface supporting moving-average and news sentiment logic with Risk Gateway gating."""

    def __init__(
        self,
        gateway: str = "",
        broker: str = "",
        username: str = "",
        token: str = "",
        fast_window: int = 5,
        slow_window: int = 20,
        batch_quantity: int = 50,
        adapter: Optional[ExecutionAdapter] = None,
        heartbeat_worker: Optional[HeartbeatWorker] = None,
        risk_gate: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        mock_mode: bool = False,
    ) -> None:
        effective_gw = gateway or (adapter.config.api_base_url if adapter else "")
        self.gateway = effective_gw.rstrip("/") if effective_gw else ""
        self.broker = broker
        self.username = username
        self.token = token
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.batch_quantity = batch_quantity
        self.adapter = adapter
        self.heartbeat_worker = heartbeat_worker
        self.risk_gate = risk_gate
        self.decision_engine = decision_engine
        self.pipeline = pipeline
        self.mock_mode = mock_mode

        self.tradable_tickers: set[str] = set()
        self.price_history: dict[str, list[float]] = {}
        self.holdings: dict[str, int] = {}
        self.last_heartbeat_seq: int = 1

    def discover_universe(self) -> set[str]:
        """Fetch stock universe from GET /api/v1/stocks."""
        if self.adapter is not None:
            try:
                self.tradable_tickers = self.adapter.get_tradable_universe()
                return self.tradable_tickers
            except Exception as exc:
                logger.warning("[UNIVERSE] Failed to fetch universe via adapter: %s", exc)

        if self.mock_mode:
            self.tradable_tickers = {"ALPHACORP"}
            return self.tradable_tickers

        url = f"{self.gateway}/api/v1/stocks"
        resp = send_http_request(url, token=self.token)
        if resp and isinstance(resp, dict):
            stocks = resp.get("stocks") or resp.get("tickers") or []
            if isinstance(stocks, list):
                self.tradable_tickers = {
                    (s.get("ticker") if isinstance(s, dict) else str(s)).upper()
                    for s in stocks
                }
                logger.info("[UNIVERSE] Discovered %d tradable tickers: %s", len(self.tradable_tickers), sorted(self.tradable_tickers))
        return self.tradable_tickers

    def handle_preflight(self, seq: int | str) -> bool:
        """Enqueue preflight heartbeat sequence number for non-blocking acknowledgement."""
        if self.mock_mode:
            logger.info("[MOCK] Heartbeat seq=%s acknowledged in mock mode (remote HTTP call skipped).", seq)
            try:
                self.last_heartbeat_seq = int(seq)
            except (ValueError, TypeError):
                self.last_heartbeat_seq = 1
            return True

        if self.heartbeat_worker is not None:
            return self.heartbeat_worker.enqueue_heartbeat(seq)

        # Fallback direct HTTP post if worker is not attached
        url = f"{self.gateway}/api/v1/bot/heartbeat"
        resp = send_http_request(url, method="POST", token=self.token, payload={"seq": seq})
        return resp is not None

    def submit_order(
        self,
        ticker: str,
        action: str,
        quantity: int,
        authorized_by_risk: bool = False,
        market_price: float = 0.0,
    ) -> Optional[OrderResult]:
        """Submit order strictly through ExecutionAdapter and Risk Gateway authorization.

        CRITICAL SAFETY RULE: Never bypass Risk Gateway authorization.
        """
        # If risk_gate is attached and order hasn't been explicitly pre-authorized, evaluate it
        if not authorized_by_risk and self.risk_gate is not None:
            now_utc = datetime.now(timezone.utc)
            initial_c = self.adapter.ledger.initial_cash if (self.adapter and self.adapter.ledger) else 1_000_000.0
            avail_cash = initial_c
            port_val = initial_c
            dd = 0.0
            current_pos = 0.0
            if self.adapter and self.adapter.ledger:
                summary = self.adapter.ledger.get_portfolio_summary()
                avail_cash = summary.available_cash
                port_val = summary.total_portfolio_value
                dd = summary.drawdown_fraction
                current_pos = float(self.adapter.ledger.get_position(ticker))

            effective_market_price = (
                float(market_price)
                if market_price > 0
                else (self.price_history.get(ticker.upper(), [0.0])[-1] if self.price_history.get(ticker.upper()) else 0.0)
            )

            try:
                from nova.decision.contracts import ActionType
                from nova.quant.contracts import DataQualityStatus
                act_type = ActionType(action.upper())
                risk_eval = self.risk_gate.evaluate(
                    proposed_action=act_type,
                    proposed_exposure=0.10,
                    data_quality=DataQualityStatus.VALID,
                    bar_timestamp=now_utc,
                    as_of_time=now_utc,
                    current_drawdown=dd,
                    available_cash=avail_cash,
                    current_portfolio_value=port_val,
                    current_position_quantity=current_pos,
                    current_market_price=effective_market_price,
                    ticker=ticker,
                )
                if risk_eval.is_approved:
                    authorized_by_risk = True
                else:
                    logger.warning(
                        "[EXECUTION:RISK_GATEWAY] Order %s %d %s VETOED by Risk Gate: %s",
                        action,
                        quantity,
                        ticker,
                        risk_eval.risk_reason_codes,
                    )
            except Exception as exc:
                logger.warning("[EXECUTION:RISK_GATEWAY] Risk gate evaluation error: %s", exc)

        if not authorized_by_risk:
            logger.warning(
                "[EXECUTION:RISK_GATEWAY] Order %s %d %s BLOCKED: Not authorized by Person 3 Risk Gateway.",
                action,
                quantity,
                ticker,
            )
            return None

        if self.adapter is None:
            logger.warning("[EXECUTION] ExecutionAdapter not attached to bot; order submission skipped.")
            return None

        act_enum = OrderAction(action.upper())
        intent = OrderIntent(
            ticker=ticker.upper(),
            action=act_enum,
            quantity=quantity,
            authorized_by_risk=True,
        )
        return self.adapter.execute_order(intent)

    def handle_price_tick(self, event: dict[str, Any]) -> None:
        """Process price tick through the integrated SentinelDecisionEngine pipeline."""
        # 1. If integrated pipeline is attached, delegate authoritative decision and execution
        if self.pipeline is not None:
            self.pipeline.handle_market_tick(event)
            return

        # 2. Fallback diagnostic history tracking if running standalone without pipeline
        data = event.get("data") or event
        ticker = data.get("ticker") or data.get("symbol")
        close_px = data.get("close")

        if not ticker or close_px is None:
            return

        ticker = str(ticker).upper()
        close_px = float(close_px)

        if ticker not in self.price_history:
            self.price_history[ticker] = []
        self.price_history[ticker].append(close_px)

    def handle_news_flash(self, event: dict[str, Any]) -> None:
        """Process news announcement for sentiment observability.

        Adheres to Person 3 specification: Raw news headlines do not bypass
        the integrated SentinelDecisionEngine or trigger un-gated live orders.
        """
        data = event.get("data") or event
        headline = (data.get("headline") or "").upper()
        ticker = data.get("ticker") or data.get("symbol")

        if not headline or not ticker:
            return

        pos_matches = [w for w in BULLISH_KEYWORDS if w in headline]
        neg_matches = [w for w in BEARISH_KEYWORDS if w in headline]

        logger.info(
            "[NEWS FLASH] [%s] %s | Pos=%s Neg=%s",
            ticker,
            data.get("headline", "")[:60],
            pos_matches,
            neg_matches,
        )


def run_bot(mock_mode: bool = False, max_events: int | None = None, env_path: Optional[Path] = None) -> int:
    """Run bot in either offline mock simulation mode or live production mode."""
    if mock_mode:
        print("=" * 78)
        print("MANTHAN 2026 -- NOVA OFFLINE MOCK MODE VALIDATION")
        print("=" * 78)

        from unittest.mock import MagicMock
        from execution.models import SimulationStage, StockInfo, OrderStatus, OrderResult, OrderAction
        from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
        from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
        from nova.decision.prediction import CausalRidgePredictor
        from nova.decision.regime import MarketRegimeDetector
        from nova.friction.models import MarketFrictionEngine
        from nova.portfolio.profit_realization import ProfitRealizationManager
        from nova.strategies.trend_following import TrendFollowingStrategy
        from nova.strategies.mean_reversion import MeanReversionStrategy
        from nova.decision.pipeline import IntegratedTradingPipeline

        mock_client = MagicMock(spec=TradingApiClient)
        mock_client.get_stage.return_value = SimulationStage(
            stage="competition",
            status="running",
            tickers=["ALPHACORP"],
            ticker_count=1,
            current_date="2021-01-04",
            days_completed=1,
            total_days=500,
        )
        mock_client.get_stocks.return_value = [
            StockInfo(symbol="ALPHACORP", ticker="ALPHACORP", name="AlphaCorp Inc")
        ]
        mock_client.submit_order.return_value = {
            "order_id": "mock_ord_001",
            "ticker": "ALPHACORP",
            "action": "BUY",
            "quantity": 50,
            "status": "FILLED",
            "execution_price": 1000.0,
            "charges": 20.0,
        }

        mock_ledger = AccountLedger(initial_cash=1_000_000.0)
        mock_adapter = ExecutionAdapter(
            client=mock_client,
            config=ExecutionConfig(
                bot_access_token="mock_token",
                stage_cache_ttl_seconds=0.0,
                stocks_cache_ttl_seconds=0.0,
            ),
            ledger=mock_ledger,
        )
        mock_risk = DeterministicRiskGate(RiskConfiguration(allow_short=False))
        mock_ml_predictor = None
        try:
            from nova.models.xgboost_model import XGBoostPredictionModel
            from nova.decision.prediction import XGBoostPredictor
            p_xgb = REPO_ROOT / "artifacts" / "models" / "xgboost_classifier_v3"
            if p_xgb.exists():
                m_xgb = XGBoostPredictionModel.load(p_xgb)
                mock_ml_predictor = XGBoostPredictor(m_xgb)
        except Exception:
            pass
        if mock_ml_predictor is None:
            mock_ml_predictor = CausalRidgePredictor()

        mock_engine = SentinelDecisionEngine(
            config=DecisionEngineConfig(
                trend_weight=0.40,
                mean_reversion_weight=0.40,
                predictor_weight=0.20,
                min_net_edge_bps=1.0,
                target_exposure=0.15,
                allow_short=False,
                use_regime_weighting=True,
                initial_capital=1_000_000.0,
            ),
            trend_strategy=TrendFollowingStrategy(),
            mean_reversion_strategy=MeanReversionStrategy(),
            friction_engine=MarketFrictionEngine(),
            risk_gate=mock_risk,
            predictor=mock_ml_predictor,
            regime_detector=MarketRegimeDetector(),
            profit_realization_manager=ProfitRealizationManager(),
        )
        mock_pipeline = IntegratedTradingPipeline(
            adapter=mock_adapter,
            risk_gate=mock_risk,
            ledger=mock_ledger,
            decision_engine=mock_engine,
        )

        bot = ManthanTradingBot(
            gateway="http://127.0.0.1:3002",
            broker="127.0.0.1:9094",
            username="mock_user",
            token="mock_token",
            fast_window=5,
            slow_window=20,
            batch_quantity=50,
            adapter=mock_adapter,
            risk_gate=mock_risk,
            decision_engine=mock_engine,
            pipeline=mock_pipeline,
            mock_mode=True,
        )

        # 1. Simulate Preflight Heartbeat
        preflight_event = {
            "event_type": "preflight_tick",
            "seq": 101,
            "data": {"status": "STANDBY", "seq": 101},
        }
        logger.info("[MOCK] Simulating preflight heartbeat event...")
        bot.handle_preflight(101)

        # 2. Simulate 25 Days of Price Ticks for ALPHACORP
        base_date = datetime(2021, 1, 4, tzinfo=timezone.utc)
        px = 1000.0
        for i in range(25):
            px = px + (15.0 if i >= 20 else 1.0)
            tick = {
                "event_type": "price_tick",
                "simulated_date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "data": {
                    "ticker": "ALPHACORP",
                    "open": px - 5.0,
                    "high": px + 10.0,
                    "low": px - 8.0,
                    "close": px,
                    "volume": 500000,
                },
            }
            bot.handle_price_tick(tick)

        # 3. Simulate News Event
        news_event = {
            "event_type": "news_flash",
            "data": {
                "ticker": "ALPHACORP",
                "headline": "AlphaCorp reports record profit surge and revenue beat",
            },
        }
        bot.handle_news_flash(news_event)

        print("\n[+] Offline mock loop successfully validated all handlers!")
        return 0

    return main(max_messages=max_events, env_file=str(env_path) if env_path else None)


def build_arg_parser() -> argparse.ArgumentParser:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description="NOVA Autonomous Trading Bot — MANTHAN 2026 Simulation Runner"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env configuration file (default: searches kafka/.env and .env)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Perform read-only preflight verification and exit",
    )
    parser.add_argument(
        "--max-messages",
        "--max-events",
        dest="max_messages",
        type=int,
        default=None,
        help="Consume up to N messages and exit (useful for test runs)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run local offline mock validation",
    )
    return parser


def main(max_messages: Optional[int] = None, env_file: Optional[str] = None) -> int:
    """Main execution entrypoint for NOVA bot."""
    # 1. Parse CLI arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mock:
        return run_bot(mock_mode=True)

    max_msgs = max_messages if max_messages is not None else args.max_messages
    env_file_arg = env_file if env_file is not None else args.env_file

    # 2. Locate Environment Configuration
    env_path = Path(env_file_arg) if env_file_arg else None
    if env_path is None:
        candidates = [REPO_ROOT / "kafka" / ".env", REPO_ROOT / ".env"]
        for c in candidates:
            if c.is_file():
                env_path = c
                break

    # 3. Load and Validate Configuration
    k_config = KafkaConfig.load_from_env(env_path)
    ex_config = ExecutionConfig.load_from_env(env_path)

    configure_logging(k_config.log_level)

    logger.info("=" * 78)
    logger.info("NOVA AUTONOMOUS TRADING BOT — MANTHAN 2026 RUNNER")
    logger.info("Team: Nova Forge (ID: team_m2pccp5) | Group: %s", k_config.group_id)
    logger.info("Trading API: %s | Kafka Broker: %s", ex_config.api_base_url, k_config.bootstrap_servers)
    logger.info("Topics: ticks='%s', news='%s'", k_config.topic_ticks, k_config.topic_news)
    logger.info("=" * 78)

    try:
        k_config.validate()
        ex_config.validate()
    except (ConfigValidationError, ValueError) as exc:
        logger.error("Configuration validation failure: %s", exc)
        return 1

    if k_config.is_placeholder_broker():
        logger.error("Cannot start: KAFKA_BOOTSTRAP_SERVERS is set to placeholder.")
        return 1

    # 4. Initialize REST Client, Accounting Ledger, and Execution Adapter
    client = TradingApiClient(ex_config)
    initial_cash = 1_000_000.0

    logger.info("[LIFECYCLE:INIT] Querying simulation platform status...")
    try:
        stage = client.get_stage()
        clock = client.get_simulation_clock()
        cash = client.get_cash()
        initial_cash = cash.available_cash
        logger.info(
            "[LIFECYCLE:INIT] Stage: '%s' (Status: '%s') | Days: %d/%d | Simulated Date: %s",
            stage.stage,
            stage.status,
            stage.days_completed,
            stage.total_days,
            stage.current_date,
        )
        logger.info(
            "[LIFECYCLE:INIT] Clock: '%s' | Day: %d/%d (Remaining: %d) | Available Cash: %.2f %s",
            clock.status,
            clock.current_day,
            clock.total_days,
            clock.remaining_days,
            cash.available_cash,
            cash.currency,
        )
    except TradingApiError as exc:
        logger.warning("[LIFECYCLE:INIT] Unable to query initial platform state: %s", exc)

    ledger = AccountLedger(initial_cash=initial_cash)
    adapter = ExecutionAdapter(config=ex_config, client=client, ledger=ledger)

    if args.validate_only:
        logger.info("[INIT] --validate-only specified. Preflight checks passed. Exiting cleanly.")
        return 0

    # 5. Initialize Event Dispatcher & Handlers
    dispatcher = EventDispatcher()
    dispatcher.subscribe_tick(DiagnosticLoggingTickHandler())
    dispatcher.subscribe_news(DiagnosticLoggingNewsHandler())

    # 6. Start Pre-Flight Heartbeat Worker (Step D, E, G)
    heartbeat_worker = HeartbeatWorker(client=client)
    heartbeat_worker.start()
    dispatcher.subscribe_preflight(heartbeat_worker.handle_event)
    dispatcher.subscribe_tick(heartbeat_worker.handle_event)
    dispatcher.subscribe_news(heartbeat_worker.handle_event)
    logger.info("[HEARTBEAT:INIT] HeartbeatWorker active and listening for pre-flight events.")

    # 7. Start Lifecycle Monitor (Step E, G)
    def _on_stage_transition(new_stage, prev_stage):
        adapter.notify_stage_change(new_stage)
        logger.info(
            "[LIFECYCLE:EVENT] Stage changed to '%s' (%s). ExecutionAdapter cache updated.",
            new_stage.stage,
            new_stage.status,
        )
        # Trigger accounting reconciliation on stage change
        try:
            recon = adapter.reconcile_accounting()
            if recon:
                logger.info("[ACCOUNTING:STAGE_RECON] Reconciled on stage change: sync=%s", recon.is_synchronized)
        except Exception as exc:
            logger.warning("[ACCOUNTING:STAGE_RECON] Stage change reconciliation deferred: %s", exc)

    lifecycle_monitor = LifecycleMonitor(
        client=client,
        poll_interval_seconds=3.0,
        on_stage_change=_on_stage_transition,
        heartbeat_worker=heartbeat_worker,
    )
    lifecycle_monitor.poll_once()
    lifecycle_monitor.start()
    logger.info("[LIFECYCLE:INIT] LifecycleMonitor active (polling every 3.0s).")

    # 8. Connect Person 3 Sentinel Decision Engine and Risk Gate
    risk_gate = None
    decision_engine = None
    try:
        from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
        from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
        from nova.decision.prediction import CausalRidgePredictor
        from nova.decision.regime import MarketRegimeDetector
        from nova.friction.models import MarketFrictionEngine
        from nova.portfolio.profit_realization import ProfitRealizationManager
        from nova.strategies.trend_following import TrendFollowingStrategy
        from nova.strategies.mean_reversion import MeanReversionStrategy

        risk_cfg = RiskConfiguration()
        risk_gate = DeterministicRiskGate(risk_cfg)

        ml_pred = None
        try:
            from nova.models.xgboost_model import XGBoostPredictionModel
            from nova.decision.prediction import XGBoostPredictor
            p_xgb = REPO_ROOT / "artifacts" / "models" / "xgboost_classifier_v3"
            if not p_xgb.exists():
                p_xgb = REPO_ROOT / "artifacts" / "models" / "xgboost_native_v1"
            if p_xgb.exists():
                m_xgb = XGBoostPredictionModel.load(p_xgb)
                ml_pred = XGBoostPredictor(m_xgb)
                logger.info("[ML:INIT] Loaded Person 2 XGBoost predictor: %s (backend=%s)", p_xgb.name, m_xgb.backend_type)
        except Exception as exc:
            logger.warning("[ML:INIT] Fallback to CausalRidgePredictor: %s", exc)
        if ml_pred is None:
            ml_pred = CausalRidgePredictor()

        # Select canonical regime detector
        reg_detector = None
        try:
            from nova.regime.transparent_detector import TransparentRegimeDetector
            reg_detector = TransparentRegimeDetector()
            logger.info("[REGIME:INIT] Canonical Person 2 TransparentRegimeDetector loaded.")
        except Exception:
            reg_detector = MarketRegimeDetector()

        engine_cfg = DecisionEngineConfig(
            trend_weight=0.40,
            mean_reversion_weight=0.40,
            predictor_weight=0.20,
            min_net_edge_bps=2.0,
            target_exposure=0.15,
            allow_short=False,
            use_regime_weighting=True,
            initial_capital=initial_cash,
        )
        decision_engine = SentinelDecisionEngine(
            config=engine_cfg,
            trend_strategy=TrendFollowingStrategy(),
            mean_reversion_strategy=MeanReversionStrategy(),
            friction_engine=MarketFrictionEngine(),
            risk_gate=risk_gate,
            predictor=ml_pred,
            regime_detector=reg_detector,
            profit_realization_manager=ProfitRealizationManager(),
        )
        logger.info(
            "[DECISION:INIT] SentinelDecisionEngine armed (version=%s, risk_gate=%s).",
            engine_cfg.engine_version,
            risk_cfg.config_version,
        )
    except Exception as exc:
        logger.warning("[DECISION:INIT] SentinelDecisionEngine initialization notice: %s", exc)

    # 9. Connect Integrated Trading Pipeline & Strategy Connectors
    try:
        from nova.decision.pipeline import IntegratedTradingPipeline
        from nova.research.ema_crossover import EMACrossoverConfig, EMACrossoverStrategy
        from nova.research.kafka_connector import KafkaStrategyConnector

        pipeline = IntegratedTradingPipeline(
            adapter=adapter,
            risk_gate=risk_gate,
            ledger=ledger,
            decision_engine=decision_engine,
        )

        # Primary decision path: subscribe pipeline directly to market ticks
        dispatcher.subscribe_tick(pipeline.handle_market_tick)
        logger.info("[PIPELINE:INIT] IntegratedTradingPipeline connected to SentinelDecisionEngine and EventDispatcher.")

        # Secondary research path: purely analytical observation (zero order placement)
        strat_cfg = EMACrossoverConfig(fast_window=12, slow_window=26, allow_short=False)
        strategy = EMACrossoverStrategy(config=strat_cfg)

        def _on_research_signal(cand: Any, b: Any) -> None:
            logger.debug("[RESEARCH:EMA_SIGNAL] Analytical signal for %s: %s (non-trading observer)", b.ticker, cand.direction.value)

        connector = KafkaStrategyConnector(
            strategy=strategy,
            on_candidate_signal=_on_research_signal,
        )
        dispatcher.subscribe_tick(connector.handle_market_tick)
        logger.info("[STRATEGY:INIT] Person 2 EMACrossoverStrategy connected as secondary analytical listener (non-executing).")
    except Exception as exc:
        logger.warning("[PIPELINE:INIT] Pipeline initialization notice: %s", exc)

    # 10. Start Kafka Consumer Loop (Step G disconnect backoff)
    consumer = MarketDataConsumer(config=k_config, dispatcher=dispatcher)
    lifecycle_monitor.set_consumer(consumer)

    # Register clean signal handlers
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("[SHUTDOWN] Received signal %s. Shutting down NOVA gracefully...", sig_name)
        consumer.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info("[KAFKA:INIT] Starting MarketDataConsumer on %s...", k_config.bootstrap_servers)
    try:
        consumer.run(max_messages=max_msgs)
    except PlaceholderBrokerError as exc:
        logger.error("[KAFKA:ERROR] Placeholder broker error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] KeyboardInterrupt received. Exiting.")
    except Exception as exc:
        logger.critical("[KAFKA:FATAL] Fatal error in consumer: %s", exc, exc_info=True)
        return 2
    finally:
        logger.info("[SHUTDOWN] Stopping background services...")
        lifecycle_monitor.stop()
        heartbeat_worker.stop()
        summary = lifecycle_monitor.get_readiness_summary()
        logger.info(
            "[SHUTDOWN] Final Readiness State: bot_ready=%s, heartbeats_acked=%d, polls=%d, kafka_errors=%d, blockers=%s",
            summary.get("bot_ready"),
            summary.get("heartbeats_acknowledged"),
            summary.get("polls_completed"),
            consumer.kafka_errors,
            summary.get("readiness_blockers"),
        )
        logger.info("[SHUTDOWN] All services stopped. NOVA process completed.")

    return 0


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    if cli_args.mock:
        sys.exit(run_bot(mock_mode=True))
    else:
        sys.exit(main())

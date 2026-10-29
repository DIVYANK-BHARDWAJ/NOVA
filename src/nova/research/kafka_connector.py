"""NOVA Research — Kafka Market Data Ingestion to Strategy Connector.

Provides a research-only integration seam connecting Person 1's MarketDataConsumer /
EventDispatcher directly to Person 2's EMACrossoverStrategy.

Strict Safety Boundaries:
1. Analytical observation ONLY — Never submits orders, modifies positions, or routes trades.
2. Preserves strict temporal causality and rejects duplicate / out-of-order ticks.
3. Completely isolates downstream strategy evaluation exceptions from terminating Kafka consumer.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Sequence

from nova.quant.contracts import (
    CandidateSignal,
    ContractValidationError,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    SignalDirection,
    StrategyEvaluation,
)
from nova.research.adapters import market_tick_to_market_bar
from nova.research.ema_crossover import EMACrossoverStrategy

logger = logging.getLogger(__name__)


@dataclass
class ConnectorMetrics:
    """Metrics tracking event ingestion, conversion, and strategy evaluation."""
    ticks_received: int = 0
    ticks_converted: int = 0
    ticks_dropped_out_of_order: int = 0
    evaluations_count: int = 0
    bullish_signals_count: int = 0
    neutral_signals_count: int = 0
    warmup_abstentions_count: int = 0
    unreliable_data_abstentions_count: int = 0
    evaluation_errors_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "ticks_received": self.ticks_received,
            "ticks_converted": self.ticks_converted,
            "ticks_dropped_out_of_order": self.ticks_dropped_out_of_order,
            "evaluations_count": self.evaluations_count,
            "bullish_signals_count": self.bullish_signals_count,
            "neutral_signals_count": self.neutral_signals_count,
            "warmup_abstentions_count": self.warmup_abstentions_count,
            "unreliable_data_abstentions_count": self.unreliable_data_abstentions_count,
            "evaluation_errors_count": self.evaluation_errors_count,
        }


class KafkaStrategyConnector:
    """Bridges Kafka MarketTickEvents to the quantitative EMACrossoverStrategy."""

    def __init__(
        self,
        strategy: EMACrossoverStrategy,
        max_history_per_symbol: int = 500,
    ) -> None:
        self.strategy = strategy
        self.max_history_per_symbol = max_history_per_symbol
        self._history: dict[str, list[MarketBar]] = collections.defaultdict(list)
        self._metrics = ConnectorMetrics()
        self.last_evaluation_by_symbol: dict[str, StrategyEvaluation] = {}
        self.last_candidate_by_symbol: dict[str, CandidateSignal] = {}

    @property
    def metrics(self) -> ConnectorMetrics:
        """Return diagnostic metrics."""
        return self._metrics

    def get_symbol_history(self, symbol: str) -> list[MarketBar]:
        """Return current in-memory bar history for a given symbol."""
        return list(self._history.get(symbol, []))

    def reset(self) -> None:
        """Clear all in-memory histories and reset metrics."""
        self._history.clear()
        self.last_evaluation_by_symbol.clear()
        self.last_candidate_by_symbol.clear()
        self._metrics = ConnectorMetrics()

    def handle_market_tick(self, event: Any) -> StrategyEvaluation | None:
        """Process incoming MarketTickEvent, maintain history, and invoke strategy.

        Args:
            event: MarketTickEvent from kafka.src.market_data.models.

        Returns:
            StrategyEvaluation if evaluation was successfully executed, None otherwise.
        """
        self._metrics.ticks_received += 1

        # 1. Adapt MarketTickEvent to canonical MarketBar
        try:
            bar = market_tick_to_market_bar(event)
            self._metrics.ticks_converted += 1
        except Exception as exc:
            self._metrics.evaluation_errors_count += 1
            logger.warning("Failed to convert MarketTickEvent to MarketBar: %s", exc)
            return None

        symbol = bar.ticker
        symbol_history = self._history[symbol]

        # 2. Strict Chronological Monotonicity Enforcement
        if symbol_history:
            last_ts = symbol_history[-1].source_timestamp
            if bar.source_timestamp <= last_ts:
                self._metrics.ticks_dropped_out_of_order += 1
                logger.warning(
                    "Dropping out-of-order/duplicate tick for symbol %s: %s <= %s",
                    symbol,
                    bar.source_timestamp.isoformat(),
                    last_ts.isoformat(),
                )
                return None

        # 3. Buffer Bar in Chronological History
        symbol_history.append(bar)
        if len(symbol_history) > self.max_history_per_symbol:
            self._history[symbol] = symbol_history[-self.max_history_per_symbol :]

        # 4. Invoke Strategy (Isolated from Kafka loop failure)
        try:
            eval_result = self.strategy.evaluate(self._history[symbol])
            self._metrics.evaluations_count += 1
        except Exception as exc:
            self._metrics.evaluation_errors_count += 1
            logger.error("Exception during strategy evaluation for %s: %s", symbol, exc, exc_info=True)
            return None

        # 5. Record Diagnostics & Signals
        if eval_result.status == EvaluationStatus.CANDIDATE_PRODUCED:
            assert eval_result.candidate is not None
            if eval_result.candidate.direction == SignalDirection.BULLISH:
                self._metrics.bullish_signals_count += 1
            else:
                self._metrics.neutral_signals_count += 1
            self.last_candidate_by_symbol[symbol] = eval_result.candidate
        elif eval_result.status == EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY:
            self._metrics.warmup_abstentions_count += 1
        elif eval_result.status == EvaluationStatus.ABSTAIN_UNRELIABLE_DATA:
            self._metrics.unreliable_data_abstentions_count += 1

        self.last_evaluation_by_symbol[symbol] = eval_result
        return eval_result

    def __call__(self, event: Any) -> StrategyEvaluation | None:
        """Callable interface allowing direct subscription to EventDispatcher."""
        return self.handle_market_tick(event)

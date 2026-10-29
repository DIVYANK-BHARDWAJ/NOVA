"""NOVA Research — Exponential Moving Average Crossover Baseline Strategy.

Implements the canonical Fast/Slow EMA Trend-Following baseline adhering strictly to
NOVA's quantitative contracts, zero look-ahead bias, and long/flat exposure semantics.

Safety Boundaries:
- Produces analytical CandidateSignal observations ONLY.
- CANNOT execute broker orders, size capital, or bypass risk gateways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Sequence

from nova.features.moving_averages import compute_ema_crossover_features, validate_window
from nova.quant.contracts import (
    CandidateSignal,
    ContractValidationError,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    ScoreSemantics,
    SignalDirection,
    StrategyEvaluation,
)
from nova.research.adapters import validate_market_bar_sequence


@dataclass(frozen=True, slots=True)
class EMACrossoverConfig:
    """Validated configuration for EMA Crossover baseline."""
    fast_window: int = 12
    slow_window: int = 26
    allow_short: bool = False
    prediction_horizon_bars: int = 5
    strategy_id: str = "ema_crossover"
    strategy_version: str = "1.0.0"

    def __post_init__(self) -> None:
        validate_window(self.fast_window, "fast_window")
        validate_window(self.slow_window, "slow_window")
        if self.fast_window >= self.slow_window:
            raise ContractValidationError(
                f"fast_window ({self.fast_window}) must be strictly less than slow_window ({self.slow_window})"
            )
        if isinstance(self.allow_short, bool) is False:
            raise ContractValidationError("allow_short must be a boolean")
        if self.prediction_horizon_bars <= 0:
            raise ContractValidationError("prediction_horizon_bars must be positive")


class EMACrossoverStrategy:
    """Causal, deterministic trend-following baseline using Fast/Slow EMA crossover.

    Signal Logic:
    1. Warm-up: First slow_window bars yield ABSTAIN_INSUFFICIENT_HISTORY.
    2. Bullish Crossover: Fast EMA crosses strictly above Slow EMA -> BULLISH entry.
    3. Bullish Continuation: Fast EMA remains above Slow EMA -> BULLISH hold.
    4. Bearish Crossover / Regime: Fast EMA crosses below or remains below Slow EMA -> NEUTRAL (flat).
       (Under long/flat policy, bearish regimes signal NEUTRAL/exit, not short exposure).
    """

    def __init__(self, config: EMACrossoverConfig | None = None) -> None:
        self.config = config or EMACrossoverConfig()

    def evaluate(self, bars: Sequence[MarketBar]) -> StrategyEvaluation:
        """Evaluate strategy causally as of the latest bar in the sequence."""
        if not bars:
            raise ContractValidationError("Cannot evaluate strategy on empty bar sequence")

        validate_market_bar_sequence(bars)
        current_bar = bars[-1]
        as_of = current_bar.source_timestamp

        # Quality Check
        if current_bar.data_quality != DataQualityStatus.VALID:
            return StrategyEvaluation(
                status=EvaluationStatus.ABSTAIN_UNRELIABLE_DATA,
                strategy_id=self.config.strategy_id,
                strategy_version=self.config.strategy_version,
                evaluation_timestamp=as_of,
                candidate=None,
                reason_codes=("DATA_QUALITY_NOT_VALID", f"STATUS_{current_bar.data_quality.value}"),
            )

        # Warm-up Check
        if len(bars) < self.config.slow_window:
            return StrategyEvaluation(
                status=EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY,
                strategy_id=self.config.strategy_id,
                strategy_version=self.config.strategy_version,
                evaluation_timestamp=as_of,
                candidate=None,
                reason_codes=(
                    "INSUFFICIENT_HISTORY",
                    f"BAR_COUNT_{len(bars)}_OF_REQUIRED_{self.config.slow_window}",
                ),
            )

        # Extract causal series up to latest bar
        closes = [b.close for b in bars]
        timestamps = [b.source_timestamp for b in bars]

        features = compute_ema_crossover_features(
            closes=closes,
            timestamps=timestamps,
            fast_window=self.config.fast_window,
            slow_window=self.config.slow_window,
        )

        curr_feat = features[-1]
        crossover_event = curr_feat.crossover_event
        is_bullish = curr_feat.fast_ema > curr_feat.slow_ema

        # Long/Flat Semantics
        if is_bullish:
            direction = SignalDirection.BULLISH
            if crossover_event == 1:
                reason = "BULLISH_CROSSOVER_ENTRY"
            else:
                reason = "BULLISH_TREND_HOLD"
        else:
            direction = SignalDirection.NEUTRAL
            if crossover_event == -1:
                reason = "BEARISH_CROSSOVER_EXIT"
            else:
                reason = "FLAT_BEARISH_REGIME"

        # Signal score in RAW_INTENSITY bounded [-1.0, 1.0]
        # Normalized spread scaled via tanh for smooth intensity
        raw_score = math.tanh(curr_feat.normalized_spread * 100.0)
        bounded_score = max(-1.0, min(1.0, round(raw_score, 6)))

        candidate = CandidateSignal(
            strategy_id=self.config.strategy_id,
            strategy_version=self.config.strategy_version,
            ticker=current_bar.ticker,
            direction=direction,
            signal_score=bounded_score,
            score_semantics=ScoreSemantics.RAW_INTENSITY,
            evaluation_timestamp=as_of,
            prediction_horizon_bars=self.config.prediction_horizon_bars,
            prediction_horizon_unit="BARS",
            expected_gross_return=None,
            trading_cost_estimate=None,
            execution_friction_estimate=None,
            uncertainty_penalty=None,
            expected_net_edge=None,
            data_quality=current_bar.data_quality,
            supporting_evidence=(
                ("fast_ema", round(curr_feat.fast_ema, 4)),
                ("slow_ema", round(curr_feat.slow_ema, 4)),
                ("spread", round(curr_feat.spread, 4)),
                ("normalized_spread", round(curr_feat.normalized_spread, 6)),
                ("crossover_event", float(crossover_event)),
            ),
            reason_codes=(reason,),
        )

        return StrategyEvaluation(
            status=EvaluationStatus.CANDIDATE_PRODUCED,
            strategy_id=self.config.strategy_id,
            strategy_version=self.config.strategy_version,
            evaluation_timestamp=as_of,
            candidate=candidate,
            reason_codes=(reason,),
        )

    def evaluate_all(self, bars: Sequence[MarketBar]) -> list[StrategyEvaluation]:
        """Causally evaluate the strategy for every bar in the sequence.

        At each bar index t, evaluation uses ONLY bars[0...t].
        """
        validate_market_bar_sequence(bars)
        evaluations: list[StrategyEvaluation] = []
        for i in range(len(bars)):
            evaluations.append(self.evaluate(bars[: i + 1]))
        return evaluations


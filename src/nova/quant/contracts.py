"""NOVA MAX Quantitative Subsystem — Data Contracts & Interface Definitions.

STATUS: PROPOSED — VERSION 0 (Pending Cross-Engineer Review).

Ownership Boundaries:
- Engineer B owns: Quantitative research, point-in-time features, candidate signals,
  regime representations, backtesting invariants, and strategy evaluation.
- Engineer A owns: Exchange connectivity, real-time ingestion, and normalized market data.
- Engineer C owns: Portfolio allocation, risk authorization, order management, and execution.
- Engineer D owns: System-wide QA, observability, integration invariants, and release evidence.

CRITICAL INVARIANTS:
1. CandidateSignal is a research observation. It CANNOT execute trades, authorize positions,
   or bypass Engineer C's risk gateway.
2. Temporal Integrity: Timestamps MUST be timezone-aware. Unknown availability semantics
   must NEVER be assumed causally available at evaluation time.
3. Unit Compatibility: Return and cost metrics must be expressed as dimensionless decimal
   returns (e.g. 0.015 for +1.50%). Monetary fees must not be directly subtracted from returns.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Mapping, Sequence


# ============================================================================
# Exceptions
# ============================================================================

class ContractValidationError(ValueError):
    """Base error for any quantitative contract validation failure."""
    pass


class TemporalValidationError(ContractValidationError):
    """Raised when temporal semantics, timezones, or causality are violated."""
    pass


class NumericalValidationError(ContractValidationError):
    """Raised when values are non-finite, negative where prohibited, or out of range."""
    pass


class UnitIncompatibilityError(ContractValidationError):
    """Raised when attempting to combine estimates with incompatible units."""
    pass


# ============================================================================
# Enums
# ============================================================================

class TimestampSemantics(str, Enum):
    """Semantic meaning of a raw source timestamp."""
    OPEN_TIME = "OPEN_TIME"                 # Time interval opened (e.g. 09:15 for 09:15-09:20 bar)
    CLOSE_TIME = "CLOSE_TIME"               # Time interval closed (e.g. 09:20 for 09:15-09:20 bar)
    SNAPSHOT_TIME = "SNAPSHOT_TIME"         # Point-in-time state capture
    EVENT_ARRIVAL_TIME = "EVENT_ARRIVAL_TIME" # Time the event was ingested locally
    UNKNOWN = "UNKNOWN"                     # Unverified platform convention (CANNOT assume available)


class DataQualityStatus(str, Enum):
    """Data quality classification for market inputs and derived research signals."""
    VALID = "VALID"                         # Fully inspected and validated
    SUSPECT = "SUSPECT"                     # Anomalous movement, high spread, or unusual volume
    STALE = "STALE"                         # Clock skew or delayed arrival
    INTERPOLATED = "INTERPOLATED"           # Synthetic fill or imputed value
    MISSING_METRICS = "MISSING_METRICS"     # Incomplete fields in payload
    INVALID = "INVALID"                     # Corrupt, contradictory, or unparsable


class SignalDirection(str, Enum):
    """Analytical direction of a quantitative hypothesis.

    DISTINCTION: This represents analytical direction (predictive expectation),
    NOT a broker order side (BUY/SELL) and NOT a target portfolio weight.
    """
    BULLISH = "BULLISH"                     # Expected positive relative outperformance
    BEARISH = "BEARISH"                     # Expected negative relative underperformance
    NEUTRAL = "NEUTRAL"                     # No discernible directional edge / abstention view


class ScoreSemantics(str, Enum):
    """Documented semantics of the numerical signal_score.

    NOTE: A score must NOT be called a probability unless calibration is explicitly demonstrated.
    """
    RAW_INTENSITY = "RAW_INTENSITY"         # Bounded [-1.0, 1.0], 0.0 is neutral
    RANK_DECILE = "RANK_DECILE"             # Cross-sectional decile [1.0, 10.0] or quantile [0.0, 1.0]
    Z_SCORE = "Z_SCORE"                     # Standardized score in standard deviations
    EMPIRICAL_EDGE_BPS = "EMPIRICAL_EDGE_BPS" # Estimated expected edge in basis points
    UNSPECIFIED = "UNSPECIFIED"             # Uncalibrated score (requires review)


class EvaluationStatus(str, Enum):
    """Outcome status of a strategy evaluation cycle."""
    CANDIDATE_PRODUCED = "CANDIDATE_PRODUCED"             # Valid candidate signal generated
    ABSTAIN_MARKET_CONDITIONS = "ABSTAIN_MARKET_CONDITIONS" # Intentional abstention due to regime/volatility
    ABSTAIN_INSUFFICIENT_HISTORY = "ABSTAIN_INSUFFICIENT_HISTORY" # Warmup period not reached
    ABSTAIN_UNRELIABLE_DATA = "ABSTAIN_UNRELIABLE_DATA"   # Data quality rejected
    EVALUATION_FAILED = "EVALUATION_FAILED"               # Unhandled runtime evaluation exception


class RegimeCategory(str, Enum):
    """Provisional regime classifications for market breadth and regime intelligence."""
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    HIGH_VOLATILITY_CHOP = "HIGH_VOLATILITY_CHOP"
    LOW_VOLATILITY_COMPRESSION = "LOW_VOLATILITY_COMPRESSION"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_timezone_aware(dt: datetime, field_name: str) -> None:
    """Ensure datetime is timezone-aware to prevent ambiguous comparisons."""
    if not isinstance(dt, datetime):
        raise TemporalValidationError(f"Field '{field_name}' must be a datetime instance, got {type(dt)}")
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise TemporalValidationError(
            f"Field '{field_name}' must be timezone-aware (tzinfo cannot be None). Value: {dt!r}"
        )


def validate_finite(value: float, field_name: str) -> None:
    """Ensure float is finite (not NaN, +inf, -inf)."""
    if not isinstance(value, (int, float)):
        raise NumericalValidationError(f"Field '{field_name}' must be numeric, got {type(value)}")
    if not math.isfinite(value):
        raise NumericalValidationError(f"Field '{field_name}' must be finite, got {value}")


# ============================================================================
# Core Quantitative Data Models
# ============================================================================

@dataclass(frozen=True, slots=True)
class MarketBar:
    """Normalized OHLCV representation of an asset candle.

    All prices are strictly positive currency units.
    Volume is in non-negative share/contract units.
    """
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_timestamp: datetime
    available_timestamp: datetime | None = None
    timestamp_semantics: TimestampSemantics = TimestampSemantics.UNKNOWN
    data_quality: DataQualityStatus = DataQualityStatus.VALID
    provenance: str | None = None
    contract_version: str = "PROPOSED_V0"

    def __post_init__(self) -> None:
        # Ticker validation
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ContractValidationError("MarketBar 'ticker' cannot be blank or non-string")

        # Numeric finiteness and positivity
        for name, val in [("open", self.open), ("high", self.high), ("low", self.low), ("close", self.close)]:
            validate_finite(val, name)
            if val <= 0.0:
                raise NumericalValidationError(f"MarketBar price '{name}' must be strictly positive, got {val}")

        validate_finite(self.volume, "volume")
        if self.volume < 0.0:
            raise NumericalValidationError(f"MarketBar 'volume' cannot be negative, got {self.volume}")

        # OHLC logical relationships
        if self.low > self.high:
            raise ContractValidationError(f"MarketBar low ({self.low}) cannot exceed high ({self.high})")
        if self.open < self.low or self.open > self.high:
            raise ContractValidationError(
                f"MarketBar open ({self.open}) must be within [low, high] = [{self.low}, {self.high}]"
            )
        if self.close < self.low or self.close > self.high:
            raise ContractValidationError(
                f"MarketBar close ({self.close}) must be within [low, high] = [{self.low}, {self.high}]"
            )

        # Timestamps
        validate_timezone_aware(self.source_timestamp, "source_timestamp")
        if self.available_timestamp is not None:
            validate_timezone_aware(self.available_timestamp, "available_timestamp")

        if not isinstance(self.timestamp_semantics, TimestampSemantics):
            raise ContractValidationError(f"Invalid timestamp_semantics: {self.timestamp_semantics}")
        if not isinstance(self.data_quality, DataQualityStatus):
            raise ContractValidationError(f"Invalid data_quality: {self.data_quality}")

    def is_causally_available(self, as_of: datetime) -> bool:
        """Determines if this bar was causally available to a strategy at decision time as_of.

        CAUSAL RULE: If timestamp_semantics is UNKNOWN, or available_timestamp is None,
        this bar CANNOT be assumed available and returns False to protect against leakage.
        """
        validate_timezone_aware(as_of, "as_of")
        if self.timestamp_semantics == TimestampSemantics.UNKNOWN:
            return False
        if self.available_timestamp is None:
            return False
        return self.available_timestamp <= as_of


@dataclass(frozen=True, slots=True)
class CandidateSignal:
    """Quantitative research hypothesis produced by a strategy engine.

    IMPORTANT:
    This is an analytical observation, NOT an order, NOT an authorization,
    and NOT an executable portfolio allocation. Engineer B's code cannot place orders.
    """
    strategy_id: str
    strategy_version: str
    ticker: str
    direction: SignalDirection
    signal_score: float
    score_semantics: ScoreSemantics
    evaluation_timestamp: datetime
    prediction_horizon_bars: int | None = None
    prediction_horizon_unit: str = "BARS"
    expected_gross_return: float | None = None  # Dimensionless decimal return fraction (e.g. +0.015 = +1.50%)
    trading_cost_estimate: float | None = None  # Dimensionless decimal return fraction (e.g. 0.002 = 20 bps)
    execution_friction_estimate: float | None = None  # Dimensionless decimal return fraction (e.g. 0.001 = 10 bps)
    uncertainty_penalty: float | None = None    # Dimensionless decimal return fraction (>= 0.0)
    expected_net_edge: float | None = None      # Dimensionless decimal return fraction
    data_quality: DataQualityStatus = DataQualityStatus.VALID
    source_event_refs: tuple[str, ...] = field(default_factory=tuple)
    supporting_evidence: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    invalidation_conditions: tuple[str, ...] = field(default_factory=tuple)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    config_version: str | None = None
    model_version: str | None = None
    contract_version: str = "PROPOSED_V0"

    def __post_init__(self) -> None:
        # Identity validation
        if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
            raise ContractValidationError("CandidateSignal 'strategy_id' cannot be blank")
        if not isinstance(self.strategy_version, str) or not self.strategy_version.strip():
            raise ContractValidationError("CandidateSignal 'strategy_version' cannot be blank")
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ContractValidationError("CandidateSignal 'ticker' cannot be blank")

        if not isinstance(self.direction, SignalDirection):
            raise ContractValidationError(f"Invalid SignalDirection: {self.direction}")

        validate_finite(self.signal_score, "signal_score")
        if not isinstance(self.score_semantics, ScoreSemantics):
            raise ContractValidationError(f"Invalid ScoreSemantics: {self.score_semantics}")

        validate_timezone_aware(self.evaluation_timestamp, "evaluation_timestamp")

        # Horizon validation
        if self.prediction_horizon_bars is not None:
            if not isinstance(self.prediction_horizon_bars, int) or self.prediction_horizon_bars <= 0:
                raise ContractValidationError("prediction_horizon_bars must be a positive integer")

        # Decimal return and cost estimates validation (dimensionless decimal fraction)
        for name, val in [
            ("expected_gross_return", self.expected_gross_return),
            ("trading_cost_estimate", self.trading_cost_estimate),
            ("execution_friction_estimate", self.execution_friction_estimate),
            ("uncertainty_penalty", self.uncertainty_penalty),
            ("expected_net_edge", self.expected_net_edge),
        ]:
            if val is not None:
                validate_finite(val, name)

        # Cost and penalty positivity
        if self.trading_cost_estimate is not None and self.trading_cost_estimate < 0.0:
            raise NumericalValidationError(f"trading_cost_estimate cannot be negative: {self.trading_cost_estimate}")
        if self.execution_friction_estimate is not None and self.execution_friction_estimate < 0.0:
            raise NumericalValidationError(f"execution_friction_estimate cannot be negative: {self.execution_friction_estimate}")
        if self.uncertainty_penalty is not None and self.uncertainty_penalty < 0.0:
            raise NumericalValidationError(f"uncertainty_penalty cannot be negative: {self.uncertainty_penalty}")

        # Consistency of expected_net_edge if explicitly provided
        if self.expected_net_edge is not None and self.expected_gross_return is not None and self.trading_cost_estimate is not None:
            computed = self.expected_gross_return - (
                self.trading_cost_estimate + (self.execution_friction_estimate or 0.0) + (self.uncertainty_penalty or 0.0)
            )
            if not math.isclose(self.expected_net_edge, computed, rel_tol=1e-6, abs_tol=1e-7):
                raise ContractValidationError(
                    f"Provided expected_net_edge ({self.expected_net_edge}) does not match computed net edge ({computed})"
                )

        if not isinstance(self.data_quality, DataQualityStatus):
            raise ContractValidationError(f"Invalid data_quality: {self.data_quality}")

    def compute_expected_net_edge(self) -> float | None:
        """Calculates expected net edge if required return and cost components are present.

        Returns None if gross return or cost estimate is unavailable (NO fabricated zeros).
        """
        if self.expected_gross_return is None or self.trading_cost_estimate is None:
            return None
        return self.expected_gross_return - (
            self.trading_cost_estimate + (self.execution_friction_estimate or 0.0) + (self.uncertainty_penalty or 0.0)
        )


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    """Represents the complete result of a strategy evaluation cycle."""
    status: EvaluationStatus
    strategy_id: str
    strategy_version: str
    evaluation_timestamp: datetime
    candidate: CandidateSignal | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    contract_version: str = "PROPOSED_V0"

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
            raise ContractValidationError("StrategyEvaluation 'strategy_id' cannot be blank")
        if not isinstance(self.strategy_version, str) or not self.strategy_version.strip():
            raise ContractValidationError("StrategyEvaluation 'strategy_version' cannot be blank")

        validate_timezone_aware(self.evaluation_timestamp, "evaluation_timestamp")

        if not isinstance(self.status, EvaluationStatus):
            raise ContractValidationError(f"Invalid EvaluationStatus: {self.status}")

        # Invariant: CANDIDATE_PRODUCED requires candidate; all other statuses FORBID candidate
        if self.status == EvaluationStatus.CANDIDATE_PRODUCED:
            if self.candidate is None:
                raise ContractValidationError(
                    "StrategyEvaluation status is CANDIDATE_PRODUCED but candidate is None"
                )
            if self.candidate.strategy_id != self.strategy_id:
                raise ContractValidationError(
                    f"Strategy ID mismatch between evaluation ({self.strategy_id}) and candidate ({self.candidate.strategy_id})"
                )
        else:
            if self.candidate is not None:
                raise ContractValidationError(
                    f"Contradictory evaluation: status is {self.status.value} but candidate payload is present"
                )

        for code in self.reason_codes:
            if not isinstance(code, str) or not code.strip():
                raise ContractValidationError("reason_codes elements must be non-empty strings")


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Provisional market regime intelligence representation."""
    regime_label: RegimeCategory
    evaluation_timestamp: datetime
    data_quality: DataQualityStatus = DataQualityStatus.VALID
    confidence_score: float | None = None  # Heuristic confidence in [0.0, 1.0], NOT calibrated probability
    supporting_indicators: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    uncertainty_metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    contract_version: str = "PROPOSED_V0"

    def __post_init__(self) -> None:
        if not isinstance(self.regime_label, RegimeCategory):
            raise ContractValidationError(f"Invalid RegimeCategory: {self.regime_label}")

        validate_timezone_aware(self.evaluation_timestamp, "evaluation_timestamp")

        if self.confidence_score is not None:
            validate_finite(self.confidence_score, "confidence_score")
            if not (0.0 <= self.confidence_score <= 1.0):
                raise NumericalValidationError(
                    f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
                )

        if not isinstance(self.data_quality, DataQualityStatus):
            raise ContractValidationError(f"Invalid data_quality: {self.data_quality}")


@dataclass(frozen=True, slots=True)
class ResearchMetadata:
    """Metadata tracking quantitative experiments and configuration lineage."""
    strategy_id: str
    strategy_version: str
    config_version: str
    config_fingerprint: str
    run_id: str | None = None
    dataset_id: str | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        for name, val in [
            ("strategy_id", self.strategy_id),
            ("strategy_version", self.strategy_version),
            ("config_version", self.config_version),
            ("config_fingerprint", self.config_fingerprint),
        ]:
            if not isinstance(val, str) or not val.strip():
                raise ContractValidationError(f"ResearchMetadata '{name}' cannot be blank")

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
4. Expected Net Edge: Requires all 4 components (gross return, trading cost, execution friction,
   uncertainty penalty). Missing components produce None (NO fabricated zeros).
5. Score Semantics: Enforces strictly documented valid ranges per semantic definition.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence


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
    VALID = "VALID"                                     # Fully inspected and validated from genuine competition feed
    SUSPECT = "SUSPECT"                                 # Anomalous movement, high spread, or unusual volume
    STALE = "STALE"                                     # Clock skew or delayed arrival
    INTERPOLATED = "INTERPOLATED"                       # Synthetic fill or imputed value
    MISSING_METRICS = "MISSING_METRICS"                 # Incomplete fields in payload
    INVALID = "INVALID"                                 # Corrupt, contradictory, or unparsable
    UNVERIFIED_PROVENANCE = "UNVERIFIED_PROVENANCE"     # Stream provenance not authenticated as official competition feed
    SYNTHETIC_TEST_FIXTURE = "SYNTHETIC_TEST_FIXTURE"   # Isolated unit-test fixture (STRICTLY FORBIDDEN in live mode)


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
    RAW_INTENSITY = "RAW_INTENSITY"               # Bounded [-1.0, 1.0], 0.0 is neutral
    RANK_DECILE = "RANK_DECILE"                   # Cross-sectional decile in [1.0, 10.0]
    RANK_PERCENTILE = "RANK_PERCENTILE"           # Cross-sectional percentile in [0.0, 1.0]
    Z_SCORE = "Z_SCORE"                           # Standardized score in standard-deviation units
    EMPIRICAL_EDGE_BPS = "EMPIRICAL_EDGE_BPS"     # Estimated expected edge in basis points
    UNSPECIFIED = "UNSPECIFIED"                   # Explicitly uncalibrated/unspecified score


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


class PredictionType(str, Enum):
    """Exact statistical semantic of a ModelPrediction output.

    CRITICAL: A return forecast must never be confused with a probability or sentiment.
    """
    EXPECTED_RETURN_DECIMAL = "EXPECTED_RETURN_DECIMAL"  # Forward price return decimal (e.g. +0.0150 = +1.50%)
    DIRECTIONAL_PROBABILITY = "DIRECTIONAL_PROBABILITY"  # Calibrated P(return > 0) in [0.0, 1.0]
    REGIME_PROBABILITY = "REGIME_PROBABILITY"            # Probability of target regime in [0.0, 1.0]
    VOLATILITY_ESTIMATE = "VOLATILITY_ESTIMATE"          # Realized or expected volatility (>= 0.0)
    SENTIMENT_SCORE = "SENTIMENT_SCORE"                  # Bounded NLP text sentiment in [-1.0, 1.0]


class PredictionAvailability(str, Enum):
    """Inference readiness and operational status of a model prediction."""
    READY = "READY"                                      # Normal, reliable model inference
    STALE = "STALE"                                      # Input data clock skew or delayed bar
    FALLBACK_HEURISTIC = "FALLBACK_HEURISTIC"            # Heuristic / fallback rule invoked
    UNAVAILABLE = "UNAVAILABLE"                          # Inference failed or model offline
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"        # Insufficient warmup bars to compute features


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
    """Ensure float is finite (not NaN, +inf, -inf) and not a boolean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericalValidationError(f"Field '{field_name}' must be numeric (non-boolean), got {type(value)}")
    if not math.isfinite(value):
        raise NumericalValidationError(f"Field '{field_name}' must be finite, got {value}")


SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "jwt",
    "token",
    "bearer",
    "credential",
    "private_key",
)


def validate_no_secrets(mapping: Mapping[str, Any], field_name: str = "metadata") -> None:
    """Ensure no credentials, JWTs, API keys, or private secrets are stored in metadata."""
    if not isinstance(mapping, Mapping):
        return
    for k, v in mapping.items():
        k_lower = k.lower()
        for sens in SENSITIVE_KEY_SUBSTRINGS:
            if sens in k_lower:
                raise ContractValidationError(
                    f"Sensitive credential detected in '{field_name}' key '{k}': metadata must not store secrets"
                )
        if isinstance(v, str):
            v_lower = v.lower()
            if "bearer " in v_lower or v_lower.startswith("eyj"):  # Common JWT prefix eyJ
                raise ContractValidationError(
                    f"Sensitive token value detected in '{field_name}' key '{k}': metadata must not store secrets"
                )


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

        # Signal score finiteness and semantic boundary validation
        validate_finite(self.signal_score, "signal_score")
        if not isinstance(self.score_semantics, ScoreSemantics):
            raise ContractValidationError(f"Invalid ScoreSemantics: {self.score_semantics}")

        if self.score_semantics == ScoreSemantics.RAW_INTENSITY:
            if not (-1.0 <= self.signal_score <= 1.0):
                raise NumericalValidationError(
                    f"RAW_INTENSITY signal_score must be in [-1.0, 1.0], got {self.signal_score}"
                )
        elif self.score_semantics == ScoreSemantics.RANK_DECILE:
            if not (1.0 <= self.signal_score <= 10.0):
                raise NumericalValidationError(
                    f"RANK_DECILE signal_score must be in [1.0, 10.0], got {self.signal_score}"
                )
        elif self.score_semantics == ScoreSemantics.RANK_PERCENTILE:
            if not (0.0 <= self.signal_score <= 1.0):
                raise NumericalValidationError(
                    f"RANK_PERCENTILE signal_score must be in [0.0, 1.0], got {self.signal_score}"
                )

        validate_timezone_aware(self.evaluation_timestamp, "evaluation_timestamp")

        # Horizon validation
        if self.prediction_horizon_bars is not None:
            if isinstance(self.prediction_horizon_bars, bool) or not isinstance(self.prediction_horizon_bars, int) or self.prediction_horizon_bars <= 0:
                raise ContractValidationError("prediction_horizon_bars must be a positive integer (non-boolean)")

        if not isinstance(self.prediction_horizon_unit, str) or not self.prediction_horizon_unit.strip():
            raise ContractValidationError("prediction_horizon_unit must be a non-empty string")

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

        # Strict Net Edge Validation:
        # If expected_net_edge is explicitly supplied, all 4 components MUST be present.
        # It is forbidden to supply an explicit net edge with incomplete components (no fabricated zeros).
        if self.expected_net_edge is not None:
            if (
                self.expected_gross_return is None
                or self.trading_cost_estimate is None
                or self.execution_friction_estimate is None
                or self.uncertainty_penalty is None
            ):
                raise ContractValidationError(
                    "Cannot supply explicit expected_net_edge when return, cost, friction, or penalty estimates are incomplete"
                )
            computed = self.expected_gross_return - (
                self.trading_cost_estimate + self.execution_friction_estimate + self.uncertainty_penalty
            )
            if not math.isclose(self.expected_net_edge, computed, rel_tol=1e-6, abs_tol=1e-7):
                raise ContractValidationError(
                    f"Provided expected_net_edge ({self.expected_net_edge}) does not match computed net edge ({computed})"
                )

        if not isinstance(self.data_quality, DataQualityStatus):
            raise ContractValidationError(f"Invalid data_quality: {self.data_quality}")

    def compute_expected_net_edge(self) -> float | None:
        """Calculates expected net edge if all return and cost components are present.

        Returns None if ANY component is missing (NO fabricated zeros).
        Formula: expected_gross_return - (trading_cost + execution_friction + uncertainty_penalty)
        """
        if (
            self.expected_gross_return is None
            or self.trading_cost_estimate is None
            or self.execution_friction_estimate is None
            or self.uncertainty_penalty is None
        ):
            return None
        return self.expected_gross_return - (
            self.trading_cost_estimate + self.execution_friction_estimate + self.uncertainty_penalty
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


# ============================================================================
# Canonical Model Prediction Contract (Interface Contract B)
# ============================================================================

@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Standardized prediction output contract produced by Person 2 quantitative models (Interface Contract B).

    CRITICAL INVARIANTS:
    1. Temporal Integrity: Both prediction_timestamp and input_cutoff_timestamp MUST be timezone-aware.
       input_cutoff_timestamp CANNOT be later than prediction_timestamp (input_cutoff <= prediction_timestamp).
    2. Statistical Disambiguation:
       - EXPECTED_RETURN_DECIMAL: dimensionless price return decimal fraction (e.g. +0.0150 = +1.50%).
       - DIRECTIONAL_PROBABILITY: calibrated probability in [0.0, 1.0].
       - REGIME_PROBABILITY: calibrated probability in [0.0, 1.0].
       - VOLATILITY_ESTIMATE: non-negative finite float (>= 0.0).
       - SENTIMENT_SCORE: bounded NLP intensity in [-1.0, 1.0].
    3. Finiteness: prediction_value and confidence_or_uncertainty reject NaN, +inf, -inf, and booleans.
    4. Availability: UNAVAILABLE or INSUFFICIENT_HISTORY explicitly signifies that normal inference was not possible.
    5. No Trade Execution: ModelPrediction is an analytical forecast, not an order or authorization.
    """
    model_id: str
    model_version: str
    prediction_timestamp: datetime
    input_cutoff_timestamp: datetime
    instrument_id: str
    prediction_type: PredictionType
    horizon_bars: int
    prediction_value: float
    horizon_unit: str = "BARS"
    prediction_distribution: Mapping[str, float] | None = None
    confidence_or_uncertainty: float | None = None
    availability_status: PredictionAvailability = PredictionAvailability.READY
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = "PROPOSED_V0"

    def __post_init__(self) -> None:
        # String validations
        for name, val in [
            ("model_id", self.model_id),
            ("model_version", self.model_version),
            ("instrument_id", self.instrument_id),
            ("horizon_unit", self.horizon_unit),
            ("contract_version", self.contract_version),
        ]:
            if not isinstance(val, str) or not val.strip():
                raise ContractValidationError(f"ModelPrediction '{name}' cannot be blank or non-string")

        # Timestamps
        validate_timezone_aware(self.prediction_timestamp, "prediction_timestamp")
        validate_timezone_aware(self.input_cutoff_timestamp, "input_cutoff_timestamp")
        if self.input_cutoff_timestamp > self.prediction_timestamp:
            raise TemporalValidationError(
                f"input_cutoff_timestamp ({self.input_cutoff_timestamp}) cannot be after "
                f"prediction_timestamp ({self.prediction_timestamp})"
            )

        # Enums normalization / validation
        if isinstance(self.prediction_type, str):
            try:
                object.__setattr__(self, "prediction_type", PredictionType(self.prediction_type))
            except ValueError:
                raise ContractValidationError(f"Invalid PredictionType: {self.prediction_type}")
        elif not isinstance(self.prediction_type, PredictionType):
            raise ContractValidationError(f"Invalid PredictionType: {self.prediction_type}")

        if isinstance(self.availability_status, str):
            try:
                object.__setattr__(self, "availability_status", PredictionAvailability(self.availability_status))
            except ValueError:
                raise ContractValidationError(f"Invalid PredictionAvailability: {self.availability_status}")
        elif not isinstance(self.availability_status, PredictionAvailability):
            raise ContractValidationError(f"Invalid PredictionAvailability: {self.availability_status}")

        # Horizon bars validation
        if isinstance(self.horizon_bars, bool) or not isinstance(self.horizon_bars, int) or self.horizon_bars <= 0:
            raise ContractValidationError("horizon_bars must be a positive integer (non-boolean)")

        # Prediction value validation
        validate_finite(self.prediction_value, "prediction_value")

        # Statistical type range constraints
        if self.prediction_type == PredictionType.DIRECTIONAL_PROBABILITY:
            if not (0.0 <= self.prediction_value <= 1.0):
                raise NumericalValidationError(
                    f"DIRECTIONAL_PROBABILITY must be in [0.0, 1.0], got {self.prediction_value}"
                )
        elif self.prediction_type == PredictionType.REGIME_PROBABILITY:
            if not (0.0 <= self.prediction_value <= 1.0):
                raise NumericalValidationError(
                    f"REGIME_PROBABILITY must be in [0.0, 1.0], got {self.prediction_value}"
                )
        elif self.prediction_type == PredictionType.VOLATILITY_ESTIMATE:
            if self.prediction_value < 0.0:
                raise NumericalValidationError(
                    f"VOLATILITY_ESTIMATE must be non-negative (>= 0.0), got {self.prediction_value}"
                )
        elif self.prediction_type == PredictionType.SENTIMENT_SCORE:
            if not (-1.0 <= self.prediction_value <= 1.0):
                raise NumericalValidationError(
                    f"SENTIMENT_SCORE must be in [-1.0, 1.0], got {self.prediction_value}"
                )
        elif self.prediction_type == PredictionType.EXPECTED_RETURN_DECIMAL:
            # Must be finite float, already checked
            pass

        # Confidence / uncertainty validation
        if self.confidence_or_uncertainty is not None:
            validate_finite(self.confidence_or_uncertainty, "confidence_or_uncertainty")
            if not (0.0 <= self.confidence_or_uncertainty <= 1.0):
                raise NumericalValidationError(
                    f"confidence_or_uncertainty must be in [0.0, 1.0], got {self.confidence_or_uncertainty}"
                )

        # Prediction distribution validation
        if self.prediction_distribution is not None:
            if not isinstance(self.prediction_distribution, Mapping):
                raise ContractValidationError("prediction_distribution must be a mapping of string keys to numeric values")
            for k, v in self.prediction_distribution.items():
                if not isinstance(k, str) or not k.strip():
                    raise ContractValidationError("prediction_distribution keys must be non-empty strings")
                validate_finite(v, f"prediction_distribution[{k}]")
            if self.prediction_type == PredictionType.REGIME_PROBABILITY:
                for k, v in self.prediction_distribution.items():
                    if not (0.0 <= v <= 1.0):
                        raise NumericalValidationError(
                            f"REGIME_PROBABILITY distribution value for '{k}' must be in [0.0, 1.0], got {v}"
                        )
                dist_sum = sum(self.prediction_distribution.values())
                if not math.isclose(dist_sum, 1.0, abs_tol=1e-2):
                    raise NumericalValidationError(
                        f"REGIME_PROBABILITY distribution values must sum to approximately 1.0, got {dist_sum}"
                    )

        # Metadata validation
        if not isinstance(self.metadata, Mapping):
            raise ContractValidationError("metadata must be a mapping (dict-like)")
        validate_no_secrets(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        """Convert ModelPrediction instance to a JSON-serializable dictionary matching the schema."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "input_cutoff_timestamp": self.input_cutoff_timestamp.isoformat(),
            "instrument_id": self.instrument_id,
            "prediction_type": self.prediction_type.value,
            "horizon_bars": self.horizon_bars,
            "horizon_unit": self.horizon_unit,
            "prediction_value": self.prediction_value,
            "prediction_distribution": dict(self.prediction_distribution) if self.prediction_distribution is not None else None,
            "confidence_or_uncertainty": self.confidence_or_uncertainty,
            "availability_status": self.availability_status.value,
            "metadata": dict(self.metadata),
            "contract_version": self.contract_version,
        }

    def to_json(self, indent: int | None = None) -> str:
        """Serialize ModelPrediction to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelPrediction:
        """Construct ModelPrediction from dictionary with strict parsing."""
        if not isinstance(data, Mapping):
            raise ContractValidationError(f"Expected dictionary payload, got {type(data)}")

        required = [
            "model_id",
            "model_version",
            "prediction_timestamp",
            "input_cutoff_timestamp",
            "instrument_id",
            "prediction_type",
            "horizon_bars",
            "horizon_unit",
            "prediction_value",
            "availability_status",
            "contract_version",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            raise ContractValidationError(f"Missing required fields for ModelPrediction: {missing}")

        pred_ts = data["prediction_timestamp"]
        if isinstance(pred_ts, str):
            try:
                pred_ts = datetime.fromisoformat(pred_ts)
            except Exception as e:
                raise TemporalValidationError(f"Invalid ISO format for prediction_timestamp: {e}")

        cutoff_ts = data["input_cutoff_timestamp"]
        if isinstance(cutoff_ts, str):
            try:
                cutoff_ts = datetime.fromisoformat(cutoff_ts)
            except Exception as e:
                raise TemporalValidationError(f"Invalid ISO format for input_cutoff_timestamp: {e}")

        pred_type_raw = data["prediction_type"]
        if isinstance(pred_type_raw, str):
            try:
                pred_type = PredictionType(pred_type_raw)
            except ValueError:
                raise ContractValidationError(f"Invalid PredictionType value: {pred_type_raw}")
        elif isinstance(pred_type_raw, PredictionType):
            pred_type = pred_type_raw
        else:
            raise ContractValidationError(f"prediction_type must be str or PredictionType, got {type(pred_type_raw)}")

        avail_raw = data["availability_status"]
        if isinstance(avail_raw, str):
            try:
                avail = PredictionAvailability(avail_raw)
            except ValueError:
                raise ContractValidationError(f"Invalid PredictionAvailability value: {avail_raw}")
        elif isinstance(avail_raw, PredictionAvailability):
            avail = avail_raw
        else:
            raise ContractValidationError(f"availability_status must be str or PredictionAvailability, got {type(avail_raw)}")

        return cls(
            model_id=data["model_id"],
            model_version=data["model_version"],
            prediction_timestamp=pred_ts,
            input_cutoff_timestamp=cutoff_ts,
            instrument_id=data["instrument_id"],
            prediction_type=pred_type,
            horizon_bars=data["horizon_bars"],
            prediction_value=data["prediction_value"],
            horizon_unit=data.get("horizon_unit", "BARS"),
            prediction_distribution=data.get("prediction_distribution"),
            confidence_or_uncertainty=data.get("confidence_or_uncertainty"),
            availability_status=avail,
            metadata=data.get("metadata", {}),
            contract_version=data.get("contract_version", "PROPOSED_V0"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ModelPrediction:
        """Parse ModelPrediction from a JSON string."""
        if not isinstance(json_str, str):
            raise ContractValidationError(f"Expected json_str to be string, got {type(json_str)}")
        try:
            payload = json.loads(json_str)
        except Exception as e:
            raise ContractValidationError(f"Failed to parse JSON string: {e}")
        return cls.from_dict(payload)


def validate_model_prediction_schema_parity(data: Mapping[str, Any] | ModelPrediction) -> list[str]:
    """Validates a dictionary or ModelPrediction instance against canonical schemas/model_prediction.schema.json rules.

    Pure standard-library implementation ensuring 100% parity without third-party dependencies.
    Returns a list of error strings (empty list indicates full compliance).
    """
    if isinstance(data, ModelPrediction):
        data = data.to_dict()
    elif hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
        data = data.to_dict()

    errors: list[str] = []
    if not isinstance(data, Mapping):
        return [f"Payload must be an object (Mapping), got {type(data)}"]

    allowed_properties = {
        "model_id",
        "model_version",
        "prediction_timestamp",
        "input_cutoff_timestamp",
        "instrument_id",
        "prediction_type",
        "horizon_bars",
        "horizon_unit",
        "prediction_value",
        "prediction_distribution",
        "confidence_or_uncertainty",
        "availability_status",
        "metadata",
        "contract_version",
    }
    # additionalProperties: false
    for key in data:
        if key not in allowed_properties:
            errors.append(f"Additional property not permitted: '{key}'")

    required = [
        "model_id",
        "model_version",
        "prediction_timestamp",
        "input_cutoff_timestamp",
        "instrument_id",
        "prediction_type",
        "horizon_bars",
        "horizon_unit",
        "prediction_value",
        "availability_status",
        "contract_version",
    ]
    for req in required:
        if req not in data:
            errors.append(f"Missing required property: '{req}'")

    if errors:
        return errors

    # Check types and constraints
    for str_prop in ["model_id", "model_version", "instrument_id", "horizon_unit", "contract_version"]:
        val = data.get(str_prop)
        if not isinstance(val, str) or len(val) < 1:
            errors.append(f"Property '{str_prop}' must be a non-empty string")

    # Timestamps format check
    for ts_prop in ["prediction_timestamp", "input_cutoff_timestamp"]:
        ts_val = data.get(ts_prop)
        if not isinstance(ts_val, str):
            errors.append(f"Property '{ts_prop}' must be an ISO 8601 date-time string")
        else:
            try:
                parsed_dt = datetime.fromisoformat(ts_val)
                if parsed_dt.tzinfo is None or parsed_dt.tzinfo.utcoffset(parsed_dt) is None:
                    errors.append(f"Property '{ts_prop}' must include explicit timezone offset")
            except Exception:
                errors.append(f"Property '{ts_prop}' is not a valid ISO 8601 string: '{ts_val}'")

    # Horizon bars
    h_bars = data.get("horizon_bars")
    if isinstance(h_bars, bool) or not isinstance(h_bars, int) or h_bars < 1:
        errors.append(f"Property 'horizon_bars' must be an integer >= 1, got {h_bars}")

    # Enums
    valid_pred_types = {e.value for e in PredictionType}
    if data.get("prediction_type") not in valid_pred_types:
        errors.append(f"Property 'prediction_type' must be one of {sorted(valid_pred_types)}, got {data.get('prediction_type')}")

    valid_avail = {e.value for e in PredictionAvailability}
    if data.get("availability_status") not in valid_avail:
        errors.append(f"Property 'availability_status' must be one of {sorted(valid_avail)}, got {data.get('availability_status')}")

    # Numeric value
    p_val = data.get("prediction_value")
    if isinstance(p_val, bool) or not isinstance(p_val, (int, float)) or not math.isfinite(p_val):
        errors.append(f"Property 'prediction_value' must be a finite number, got {p_val}")

    # confidence_or_uncertainty
    conf = data.get("confidence_or_uncertainty")
    if conf is not None:
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf):
            errors.append(f"Property 'confidence_or_uncertainty' must be a finite number or null, got {conf}")
        elif not (0.0 <= conf <= 1.0):
            errors.append(f"Property 'confidence_or_uncertainty' must be in [0.0, 1.0], got {conf}")

    # prediction_distribution
    dist = data.get("prediction_distribution")
    if dist is not None and not isinstance(dist, Mapping):
        errors.append(f"Property 'prediction_distribution' must be an object or null, got {type(dist)}")

    # metadata
    meta = data.get("metadata")
    if meta is not None and not isinstance(meta, Mapping):
        errors.append(f"Property 'metadata' must be an object, got {type(meta)}")
    elif meta is not None and isinstance(meta, Mapping):
        try:
            validate_no_secrets(meta, "metadata")
        except ContractValidationError as err:
            errors.append(str(err))

    return errors

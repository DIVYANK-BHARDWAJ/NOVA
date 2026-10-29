"""NOVA MAX Quantitative Research & Strategy Engine Modules.

Phase 1 establishes typed data contracts, validation rules, and research configuration.
All cross-engineer interfaces are explicitly labeled: PROPOSED — VERSION 0.
"""

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    CandidateSignal,
    RegimeCategory,
    RegimeState,
    ResearchMetadata,
    ScoreSemantics,
    SignalDirection,
    StrategyEvaluation,
    TemporalValidationError,
    NumericalValidationError,
    TimestampSemantics,
    UnitIncompatibilityError,
)

__all__ = [
    "ContractValidationError",
    "DataQualityStatus",
    "EvaluationStatus",
    "MarketBar",
    "CandidateSignal",
    "RegimeCategory",
    "RegimeState",
    "ResearchMetadata",
    "ScoreSemantics",
    "SignalDirection",
    "StrategyEvaluation",
    "TemporalValidationError",
    "NumericalValidationError",
    "TimestampSemantics",
    "UnitIncompatibilityError",
]

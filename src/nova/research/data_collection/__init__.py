"""NOVA Research Data Collection Package (Person 2).

Provides:
- Stable Observation Identity & Delivery Identity (`identity.py`)
- Quality & Event Filtering (`quality.py`)
- Dataset Manifest & Cryptographic Lineage (`manifest.py`)
- Replay Dataset Recorder (`recorder.py`)
- Dataset Quality & Deduplication Auditor (`audit.py`)
- XGBoost Readiness & Sample Sufficiency Evaluator (`xgboost_readiness.py`)
"""

from nova.research.data_collection.audit import DatasetAuditor
from nova.research.data_collection.identity import DeliveryIdentity, ObservationIdentity
from nova.research.data_collection.manifest import DatasetManifest
from nova.research.data_collection.quality import (
    EventEligibility,
    EventQualityValidator,
    QualityValidationResult,
)
from nova.research.data_collection.recorder import ReplayDatasetRecorder
from nova.research.data_collection.xgboost_readiness import (
    DatasetSplitCounts,
    XGBoostReadinessEvaluator,
)

__all__ = [
    "DatasetAuditor",
    "DatasetManifest",
    "DatasetSplitCounts",
    "DeliveryIdentity",
    "EventEligibility",
    "EventQualityValidator",
    "ObservationIdentity",
    "QualityValidationResult",
    "ReplayDatasetRecorder",
    "XGBoostReadinessEvaluator",
]


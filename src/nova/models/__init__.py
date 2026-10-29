"""NOVA Quantitative Machine Learning Models Package (Person 2).

Provides:
- XGBoost / GBDT Forward Return Predictor (`xgboost_model.py`)
- Model Configuration (`config.py`)
- Dataset Assembly & Purged Embargo (`dataset.py`)
- Evaluation Metrics & Benchmarks (`metrics.py`)
- Training & Evaluation Orchestrator (`training.py`)
"""

from nova.models.config import DEFAULT_FEATURE_NAMES, XGBoostModelConfig
from nova.models.dataset import DatasetAssembler, ModelDataset
from nova.models.metrics import (
    compare_to_baselines,
    directional_accuracy,
    evaluate_regression_metrics,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from nova.models.training import ModelTrainer, ModelTrainingResult
from nova.models.xgboost_model import NovaGradientBoostedRegressor, XGBoostPredictionModel

__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "DatasetAssembler",
    "ModelDataset",
    "ModelTrainer",
    "ModelTrainingResult",
    "NovaGradientBoostedRegressor",
    "XGBoostModelConfig",
    "XGBoostPredictionModel",
    "compare_to_baselines",
    "directional_accuracy",
    "evaluate_regression_metrics",
    "mean_absolute_error",
    "r2_score",
    "root_mean_squared_error",
]


"""NOVA Quantitative Machine Learning — Model Training & Pipeline Orchestrator (Person 2).

Orchestrates:
1. Chronological dataset assembly and embargo purging via DatasetAssembler.
2. Model training with early stopping on validation split.
3. Out-of-sample evaluation on untouched test split.
4. Comparative benchmark evaluation (against Zero Return and Mean Return).
5. Versioned artifact persistence and cryptographic manifest generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from nova.models.config import XGBoostModelConfig
from nova.models.dataset import DatasetAssembler, ModelDataset
from nova.models.metrics import compare_to_baselines, evaluate_regression_metrics
from nova.models.xgboost_model import XGBoostPredictionModel
from nova.quant.contracts import MarketBar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelTrainingResult:
    """Complete results package from a model training and evaluation run."""
    model_id: str
    model_version: str
    artifact_path: str
    artifact_checksum: str
    manifest_path: str
    training_rows: int
    val_rows: int
    test_rows: int
    purged_embargo_rows: int
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    test_metrics: dict[str, Any]
    baseline_comparisons: dict[str, Any]
    training_summary: dict[str, Any]


class ModelTrainer:
    """Training orchestrator for Person 2 quantitative models."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        self.config = config or XGBoostModelConfig()
        self.assembler = DatasetAssembler(self.config)

    def train_and_evaluate(
        self,
        universe_bars: Mapping[str, Sequence[MarketBar]],
        artifacts_root: Path | str,
        artifact_subfolder: str = "gbdt_ohlcv_baseline_v1",
    ) -> ModelTrainingResult:
        """Execute end-to-end training, validation, out-of-sample testing, and artifact persistence."""
        logger.info("Assembling chronological dataset across %d instruments...", len(universe_bars))
        dataset = self.assembler.assemble_universe(universe_bars)

        logger.info(
            "Dataset assembled: %d train rows (%s to %s), %d val rows (%s to %s), %d test rows (%s to %s). Purged embargo: %d rows.",
            dataset.train_count,
            dataset.train_dates[0],
            dataset.train_dates[1],
            dataset.val_count,
            dataset.val_dates[0],
            dataset.val_dates[1],
            dataset.test_count,
            dataset.test_dates[0],
            dataset.test_dates[1],
            dataset.purged_embargo_count,
        )

        # Initialize and fit model
        model = XGBoostPredictionModel(self.config)
        train_summary = model.fit(dataset)
        logger.info("Model fitting complete via backend %s.", model.backend_type)

        # Out-of-sample evaluation on untouched test set
        y_test_pred = model.predict_raw(dataset.X_test)
        test_metrics = evaluate_regression_metrics(dataset.y_test, y_test_pred, dataset_split="test")
        baseline_comp = compare_to_baselines(dataset.y_test, y_test_pred)

        logger.info(
            "Test Set Evaluation: MAE=%.6f, RMSE=%.6f, Directional Accuracy=%.2f%% (baseline zero RMSE=%.6f).",
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["directional_accuracy_pct"],
            baseline_comp["zero_baseline"]["rmse"],
        )

        # Persist artifact
        out_dir = Path(artifacts_root) / "models" / artifact_subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / "model.json"
        save_info = model.save(model_file)

        # Write manifest
        manifest_data = {
            "model_id": self.config.model_id,
            "model_version": self.config.model_version,
            "backend": model.backend_type,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": f"forward_return_{self.config.horizon_bars}d",
            "features": list(self.config.feature_names),
            "sample_counts": {
                "raw_observations": dataset.total_raw_observations,
                "train_rows": dataset.train_count,
                "val_rows": dataset.val_count,
                "test_rows": dataset.test_count,
                "purged_embargo_rows": dataset.purged_embargo_count,
            },
            "chronological_partitions": {
                "train": list(dataset.train_dates),
                "val": list(dataset.val_dates),
                "test": list(dataset.test_dates),
            },
            "test_metrics": test_metrics,
            "baseline_comparisons": baseline_comp,
            "artifact_checksum": save_info["checksum"],
        }

        manifest_file = out_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return ModelTrainingResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            artifact_path=str(model_file),
            artifact_checksum=save_info["checksum"],
            manifest_path=str(manifest_file),
            training_rows=dataset.train_count,
            val_rows=dataset.val_count,
            test_rows=dataset.test_count,
            purged_embargo_rows=dataset.purged_embargo_count,
            train_dates=dataset.train_dates,
            val_dates=dataset.val_dates,
            test_dates=dataset.test_dates,
            test_metrics=test_metrics,
            baseline_comparisons=baseline_comp,
            training_summary=train_summary,
        )


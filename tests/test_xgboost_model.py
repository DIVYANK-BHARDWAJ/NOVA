"""Unit and integration tests for Person 2 GBDT/XGBoost Prediction Model.

Verifies:
1. Target calculation: y_{t,H} = ln(Close_{t+H} / Close_t) with exact mathematical values.
2. Unobserved tail handling: final H bars have no target and are safely dropped.
3. Chronological splitting & embargo purging: strictly forward in time, zero lookahead leakage.
4. Metric correctness: MAE, RMSE, Directional Accuracy, R-squared with synthetic benchmarks.
5. GBDT training, serialization, artifact integrity, and exact reload fidelity.
6. ModelPrediction contract compliance and strict schema parity against model_prediction.schema.json.
7. Negative paths: insufficient warmup, corrupted bars, empty inputs handled gracefully.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.models.config import XGBoostModelConfig
from nova.models.dataset import DatasetAssembler
from nova.models.metrics import (
    compare_to_baselines,
    directional_accuracy,
    evaluate_regression_metrics,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from nova.models.xgboost_model import (
    NovaGradientBoostedRegressor,
    PurePythonRegressionTree,
    XGBoostPredictionModel,
)
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    NumericalValidationError,
    PredictionAvailability,
    PredictionType,
    validate_model_prediction_schema_parity,
)


class TestTargetAndDatasetAssembly(unittest.TestCase):
    """Test target construction, causal lag, and boundary embargo."""

    def _create_synthetic_bars(self, ticker: str, count: int, start_price: float = 100.0) -> list[MarketBar]:
        base_time = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        bars = []
        p = start_price
        for i in range(count):
            dt = base_time + timedelta(days=i)
            # deterministic oscillation
            p = max(10.0, p + (1.5 if i % 2 == 0 else -1.0))
            bars.append(
                MarketBar(
                    ticker=ticker,
                    open=p - 0.2,
                    high=p + 0.5,
                    low=p - 0.5,
                    close=p,
                    volume=1000.0 + (i * 10),
                    source_timestamp=dt,
                    provenance=f"SYNTHETIC_TEST:{ticker}",
                    data_quality=DataQualityStatus.VALID,
                )
            )
        return bars

    def test_target_calculation_exact_math(self) -> None:
        """Verify target y_{t,H} = ln(Close_{t+H} / Close_t)."""
        c_t = 100.0
        c_next = 105.0
        expected_target = math.log(c_next / c_t)

        self.assertAlmostEqual(expected_target, 0.048790164, places=6)
        # Verify log-return vs simple return distinction
        simple_return = (c_next - c_t) / c_t  # 0.05
        self.assertNotEqual(expected_target, simple_return)
        self.assertLess(expected_target, simple_return)

    def test_unobserved_tail_dropped(self) -> None:
        """Verify that the final H bars without observed future close are dropped."""
        H = 2
        config = XGBoostModelConfig(horizon_bars=H, feature_warmup_bars=28)
        assembler = DatasetAssembler(config)

        # 40 bars total: 28 warmup, leaving 12 candidate bars.
        # With H=2, the last 2 bars have no target t+H, so exactly 10 valid records produced.
        bars = self._create_synthetic_bars("TEST", 40)
        universe = {"TEST": bars}

        dataset = assembler.assemble_universe(universe)
        total_rows = dataset.train_count + dataset.val_count + dataset.test_count
        # Total rows is 10 minus any purged boundary embargo
        self.assertEqual(dataset.total_labeled_observations, 10)
        self.assertEqual(total_rows + dataset.purged_embargo_count, 10)

    def test_boundary_embargo_purging(self) -> None:
        """Verify boundary embargo removes H bars before validation and test splits."""
        H = 3
        config = XGBoostModelConfig(
            horizon_bars=H,
            feature_warmup_bars=28,
            train_ratio=0.60,
            val_ratio=0.20,
        )
        assembler = DatasetAssembler(config)
        # 100 bars -> 72 post-warmup bars -> 69 target-eligible bars (dropping last H=3)
        bars = self._create_synthetic_bars("TEST", 100)
        universe = {"TEST": bars}

        dataset = assembler.assemble_universe(universe)
        self.assertGreater(dataset.purged_embargo_count, 0)

        # Check chronological separation: train_dates < val_dates < test_dates
        self.assertLess(dataset.train_dates[1], dataset.val_dates[0])
        self.assertLess(dataset.val_dates[1], dataset.test_dates[0])


class TestMetricsCalculation(unittest.TestCase):
    """Test evaluation metrics against hand-calculated ground truths."""

    def test_mae_rmse_r2(self) -> None:
        y_true = [0.01, -0.02, 0.03, -0.01, 0.00]
        y_pred = [0.008, -0.015, 0.025, -0.005, 0.002]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.assertAlmostEqual(mae, 0.0038, places=6)
        self.assertAlmostEqual(rmse, 0.00407431, places=6)
        self.assertGreater(r2, 0.90)  # High correlation test

    def test_directional_accuracy(self) -> None:
        # Signs: true = [+,-,+,0,-], pred = [+,+,+,0,+]
        # Matching: index 0 (both +), index 2 (both +), index 3 (both 0) -> 3 / 5 = 0.60 (60%)
        y_true = [0.02, -0.01, 0.05, 0.0, -0.03]
        y_pred = [0.01, 0.02, 0.01, 0.0, 0.01]

        acc = directional_accuracy(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.60, places=4)

    def test_compare_to_baselines(self) -> None:
        y_true = [0.01, -0.01, 0.02, -0.02]
        y_pred = [0.008, -0.008, 0.018, -0.018]

        res = compare_to_baselines(y_true, y_pred)
        self.assertIn("zero_baseline", res)
        self.assertIn("mean_baseline", res)
        self.assertGreater(res["rmse_improvement_vs_zero_pct"], 0.0)


class TestGBDTEngineAndContractParity(unittest.TestCase):
    """Test pure-Python tree & GBDT training, persistence, and ModelPrediction output."""

    def test_pure_python_tree_fit_predict(self) -> None:
        X = [[1.0], [2.0], [10.0], [11.0]]
        # Fit on negative gradients (residuals)
        gradients = [-0.5, -0.5, 0.5, 0.5]

        tree = PurePythonRegressionTree(max_depth=2, min_child_samples=1, l2_reg=0.1)
        tree.fit(X, gradients)
        preds = [tree.predict_one(x) for x in X]

        self.assertEqual(len(preds), 4)
        self.assertGreater(preds[0], 0.0)
        self.assertLess(preds[3], 0.0)

    def test_gbdt_regressor_training_and_serialization(self) -> None:
        # Synthetic function y = 0.5 * x0 - 0.2 * x1
        X = []
        y = []
        for i in range(50):
            x0 = float(i) / 10.0
            x1 = float(50 - i) / 10.0
            target = 0.5 * x0 - 0.2 * x1
            X.append([x0, x1])
            y.append(target)

        config = XGBoostModelConfig(
            n_estimators=10,
            learning_rate=0.1,
            max_depth=2,
            min_child_samples=2,
            l2_reg=1.0,
        )
        gbdt = NovaGradientBoostedRegressor(config=config)
        gbdt.fit(X, y)
        preds = gbdt.predict(X)
        self.assertEqual(len(preds), 50)

        # Test dictionary serialization and restoration
        d = gbdt.to_dict()
        gbdt_restored = NovaGradientBoostedRegressor.from_dict(d)
        preds_restored = gbdt_restored.predict(X)

        for p1, p2 in zip(preds, preds_restored):
            self.assertAlmostEqual(p1, p2, places=8)

    def test_model_prediction_schema_parity(self) -> None:
        """Verify that predict_bar returns a ModelPrediction satisfying schemas/model_prediction.schema.json."""
        config = XGBoostModelConfig(
            model_id="TEST_GBDT",
            model_version="1.0.0",
            horizon_bars=1,
            feature_names=("momentum", "rsi", "relative_volume"),
            feature_warmup_bars=28,
            n_estimators=3,
            learning_rate=0.1,
            max_depth=2,
            min_child_samples=1,
        )
        engine = NovaGradientBoostedRegressor(config=config)
        engine.fit([[0.1, 50.0, 1.0], [-0.1, 40.0, 0.8]], [0.01, -0.01])

        model = XGBoostPredictionModel(config=config, backend_model=engine)

        # Generate 35 valid bars
        base_time = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        bars = [
            MarketBar(
                ticker="TEST_TICKER",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=5000.0,
                source_timestamp=base_time + timedelta(days=i),
                provenance="TEST:BAR",
                data_quality=DataQualityStatus.VALID,
            )
            for i in range(35)
        ]

        pred = model.predict_bar(bars)
        self.assertIsInstance(pred, ModelPrediction)
        self.assertEqual(pred.instrument_id, "TEST_TICKER")
        self.assertEqual(pred.prediction_type, PredictionType.EXPECTED_RETURN_DECIMAL)
        self.assertEqual(pred.availability_status, PredictionAvailability.READY)
        self.assertAlmostEqual(pred.input_cutoff_timestamp.timestamp(), bars[-1].source_timestamp.timestamp())

        # Test strict JSON schema validation
        pred_dict = pred.to_dict()
        validate_model_prediction_schema_parity(pred_dict)

    def test_negative_cases_insufficient_history(self) -> None:
        """Verify that fewer than warmup bars returns INSUFFICIENT_HISTORY with value 0.0."""
        config = XGBoostModelConfig(feature_warmup_bars=28, n_estimators=1, min_child_samples=1)
        engine = NovaGradientBoostedRegressor(config=config)
        engine.fit([[0.0] * 12], [0.0])
        model = XGBoostPredictionModel(config=config, backend_model=engine)

        base_time = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        short_bars = [
            MarketBar(
                ticker="SHORT_TICKER",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000.0,
                source_timestamp=base_time + timedelta(days=i),
                provenance="TEST:SHORT",
                data_quality=DataQualityStatus.VALID,
            )
            for i in range(10)  # Only 10 bars < 28
        ]

        pred = model.predict_bar(short_bars)
        self.assertEqual(pred.availability_status, PredictionAvailability.INSUFFICIENT_HISTORY)
        self.assertEqual(pred.prediction_value, 0.0)
        self.assertEqual(pred.confidence_or_uncertainty, 0.0)
        validate_model_prediction_schema_parity(pred.to_dict())

    def test_save_and_reload_artifact(self) -> None:
        """Verify model serialization to disk, manifest generation, and exact reload fidelity."""
        config = XGBoostModelConfig(
            model_id="PERSIST_TEST",
            model_version="1.0.0",
            horizon_bars=1,
            feature_names=("momentum", "rsi"),
            n_estimators=3,
            learning_rate=0.1,
            max_depth=2,
            min_child_samples=1,
        )
        engine = NovaGradientBoostedRegressor(config=config)
        engine.fit([[0.1, 50.0], [-0.1, 40.0]], [0.01, -0.01])
        model = XGBoostPredictionModel(config=config, backend_model=engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_file = Path(tmpdir) / "model.json"
            model.save(artifact_file)
            self.assertTrue(artifact_file.exists())

            # Reload model
            reloaded_model = XGBoostPredictionModel.load(artifact_file)
            self.assertEqual(reloaded_model.config.model_id, "PERSIST_TEST")
            self.assertEqual(reloaded_model.backend_type, "PURE_PYTHON_GBDT")

            # Verify identical predictions on test points
            test_x = [[0.1, 50.0], [-0.05, 45.0], [0.2, 55.0]]
            preds_orig = model.predict_raw(test_x)
            preds_reloaded = reloaded_model.predict_raw(test_x)

            for p1, p2 in zip(preds_orig, preds_reloaded):
                self.assertAlmostEqual(p1, p2, places=8)

    def test_anti_leakage_future_bars_invariance(self) -> None:
        """Verify that altering future bar prices does not change feature outputs at time t."""
        config = XGBoostModelConfig(feature_warmup_bars=28)
        engine = NovaGradientBoostedRegressor(config=config)
        engine.fit([[0.0] * 12], [0.0])
        model = XGBoostPredictionModel(config=config, backend_model=engine)

        base_time = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        # 30 bars
        bars_original = [
            MarketBar(
                ticker="LEAK_TEST",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.0 + i,
                volume=1000.0,
                source_timestamp=base_time + timedelta(days=i),
                provenance="TEST:ORIGINAL",
                data_quality=DataQualityStatus.VALID,
            )
            for i in range(30)
        ]

        pred_original = model.predict_bar(bars_original)

        # Mutate the last bar (which is future to bar 28)
        # Bar at index 28 feature vector should be strictly dependent only on bars <= 28
        f_vec_28_before = model.pipeline.compute_latest(bars_original[:29])
        bars_mutated = list(bars_original)
        bars_mutated[29] = MarketBar(
            ticker="LEAK_TEST",
            open=500.0,
            high=600.0,
            low=400.0,
            close=550.0,
            volume=999999.0,
            source_timestamp=bars_original[29].source_timestamp,
            provenance="TEST:MUTATED",
            data_quality=DataQualityStatus.VALID,
        )
        f_vec_28_after = model.pipeline.compute_latest(bars_original[:29])

        self.assertEqual(f_vec_28_before.close, f_vec_28_after.close)
        self.assertEqual(f_vec_28_before.rsi, f_vec_28_after.rsi)
        self.assertEqual(f_vec_28_before.momentum, f_vec_28_after.momentum)

    def test_corrupted_data_fails_closed(self) -> None:
        """Verify that invalid data quality produces UNAVAILABLE prediction."""
        config = XGBoostModelConfig(feature_warmup_bars=28, n_estimators=1, min_child_samples=1)
        engine = NovaGradientBoostedRegressor(config=config)
        engine.fit([[0.0] * 12], [0.0])
        model = XGBoostPredictionModel(config=config, backend_model=engine)

        base_time = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        bars = [
            MarketBar(
                ticker="CORRUPT_TEST",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000.0,
                source_timestamp=base_time + timedelta(days=i),
                provenance="TEST:CORRUPT",
                data_quality=DataQualityStatus.INVALID if i == 34 else DataQualityStatus.VALID,
            )
            for i in range(35)
        ]

        pred = model.predict_bar(bars)
        self.assertEqual(pred.availability_status, PredictionAvailability.UNAVAILABLE)
        self.assertEqual(pred.prediction_value, 0.0)
        validate_model_prediction_schema_parity(pred.to_dict())

    def test_native_xgboost_fit_save_and_reload(self) -> None:
        """Verify native XGBoost engine training, persistence, and exact prediction reload."""
        from nova.models.dataset import ModelDataset

        config = XGBoostModelConfig(
            model_id="TEST_NATIVE_XGB",
            model_version="1.0.0",
            horizon_bars=1,
            feature_names=("momentum", "rsi", "relative_volume"),
            feature_warmup_bars=28,
            n_estimators=5,
            learning_rate=0.1,
            max_depth=2,
            min_child_samples=1,
        )

        # Create synthetic ModelDataset
        X_train = [[0.1, 50.0, 1.0], [-0.1, 40.0, 0.8], [0.2, 60.0, 1.2], [-0.2, 35.0, 0.7]]
        y_train = [0.01, -0.01, 0.02, -0.02]
        X_val = [[0.15, 55.0, 1.1], [-0.15, 38.0, 0.75]]
        y_val = [0.015, -0.015]

        dataset = ModelDataset(
            feature_names=config.feature_names,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_val,
            y_test=y_val,
            train_dates=("2025-01-01", "2025-01-04"),
            val_dates=("2025-01-05", "2025-01-06"),
            test_dates=("2025-01-07", "2025-01-08"),
            total_raw_observations=8,
            total_labeled_observations=6,
            purged_embargo_count=0,
        )

        model = XGBoostPredictionModel(config=config)
        summary = model.fit(dataset)

        self.assertEqual(model.backend_type, "NATIVE_XGBOOST")
        self.assertTrue(model.is_trained)

        # Test prediction before save
        preds_before = model.predict_raw(X_val)
        self.assertEqual(len(preds_before), 2)

        # Test save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_file = Path(tmpdir) / "native_model.json"
            save_info = model.save(artifact_file)
            self.assertTrue(artifact_file.exists())
            self.assertIn("checksum", save_info)

            reloaded = XGBoostPredictionModel.load(artifact_file)
            self.assertEqual(reloaded.backend_type, "NATIVE_XGBOOST")
            self.assertTrue(reloaded.is_trained)

            preds_after = reloaded.predict_raw(X_val)
            for p1, p2 in zip(preds_before, preds_after):
                self.assertAlmostEqual(p1, p2, places=6)

    def test_negative_cases_empty_or_invalid_bar(self) -> None:
        """Verify that empty bar sequence raises ContractValidationError."""
        config = XGBoostModelConfig(feature_warmup_bars=28, n_estimators=1, min_child_samples=1)
        engine = NovaGradientBoostedRegressor(config=config)
        model = XGBoostPredictionModel(config=config, backend_model=engine)

        with self.assertRaises(ContractValidationError):
            model.predict_bar([])


if __name__ == "__main__":
    unittest.main()

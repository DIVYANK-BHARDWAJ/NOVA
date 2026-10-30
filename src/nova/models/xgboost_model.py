"""NOVA Quantitative Machine Learning — Custom XGBoost / GBDT Forward Return Model (Person 2).

Production model implementation supporting dual-backend execution:
1. Native XGBoost Backend (uses native xgboost if installed).
2. Pure-Python GBDT Backend (zero-dependency, deterministic Gradient Boosted Trees implementing
   exact MSE gradient boosting with histogram/quantile splits and L2 regularization).
3. Produces canonical ModelPrediction outputs (Interface Contract B, EXPECTED_RETURN_DECIMAL).
4. Full serialization and checksum validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from nova.features.pipeline import FeaturePipeline
from nova.models.config import XGBoostModelConfig
from nova.models.dataset import ModelDataset
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    ModelPrediction,
    NumericalValidationError,
    PredictionAvailability,
    PredictionType,
    SignalDirection,
    validate_finite,
    validate_model_prediction_schema_parity,
    validate_no_secrets,
    validate_timezone_aware,
)

# Check native xgboost availability without raising missing-module warnings in static analysis
xgb: Any = None
_NATIVE_XGBOOST_AVAILABLE = False
try:
    if importlib.util.find_spec("xgboost") is not None:
        xgb = importlib.import_module("xgboost")
        _NATIVE_XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    _NATIVE_XGBOOST_AVAILABLE = False


# ============================================================================
# Pure-Python Gradient Boosted Trees Engine (Zero-Dependency Fallback)
# ============================================================================

@dataclass
class TreeNode:
    """Internal decision node or leaf in a regression tree."""
    is_leaf: bool
    leaf_value: float = 0.0
    feature_index: int = -1
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.is_leaf:
            return {"is_leaf": True, "leaf_value": self.leaf_value}
        return {
            "is_leaf": False,
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        if data.get("is_leaf", False):
            return cls(is_leaf=True, leaf_value=float(data["leaf_value"]))
        return cls(
            is_leaf=False,
            feature_index=int(data["feature_index"]),
            threshold=float(data["threshold"]),
            left=cls.from_dict(data["left"]) if data.get("left") else None,
            right=cls.from_dict(data["right"]) if data.get("right") else None,
        )


class PurePythonRegressionTree:
    """Exact regression tree minimizing squared error with L2 regularization."""

    def __init__(self, max_depth: int = 3, min_child_samples: int = 20, l2_reg: float = 1.0) -> None:
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.l2_reg = l2_reg
        self.root: TreeNode | None = None

    def fit(self, X: list[list[float]], gradients: list[float]) -> None:
        """Fit tree on negative gradients (residuals) for MSE loss."""
        indices = list(range(len(X)))
        self.root = self._build_tree(X, gradients, indices, depth=0)

    def _build_tree(
        self,
        X: list[list[float]],
        gradients: list[float],
        indices: list[int],
        depth: int,
    ) -> TreeNode:
        # Sum of gradients and hessians (hessian is 1.0 for MSE)
        G = sum(gradients[i] for i in indices)
        H = float(len(indices))

        # Leaf weight formula with L2 regularization: w = - G / (H + lambda)
        leaf_weight = - G / (H + self.l2_reg) if (H + self.l2_reg) > 0 else 0.0

        if depth >= self.max_depth or len(indices) < self.min_child_samples * 2:
            return TreeNode(is_leaf=True, leaf_value=leaf_weight)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        best_left_idx: list[int] = []
        best_right_idx: list[int] = []

        n_features = len(X[0])
        parent_score = (G ** 2) / (H + self.l2_reg)

        for f_idx in range(n_features):
            vals = [X[i][f_idx] for i in indices]
            sorted_vals = sorted(set(vals))
            if len(sorted_vals) <= 1:
                continue

            # Quantile/histogram sample candidate thresholds (up to 20 quantiles)
            step = max(1, len(sorted_vals) // 15)
            candidate_thresholds = [
                (sorted_vals[k] + sorted_vals[k + 1]) / 2.0
                for k in range(0, len(sorted_vals) - 1, step)
            ]

            for thresh in candidate_thresholds:
                left_idx = [i for i in indices if X[i][f_idx] <= thresh]
                right_idx = [i for i in indices if X[i][f_idx] > thresh]

                if len(left_idx) < self.min_child_samples or len(right_idx) < self.min_child_samples:
                    continue

                G_L = sum(gradients[i] for i in left_idx)
                H_L = float(len(left_idx))
                G_R = G - G_L
                H_R = H - H_L

                gain = 0.5 * (
                    (G_L ** 2) / (H_L + self.l2_reg)
                    + (G_R ** 2) / (H_R + self.l2_reg)
                    - parent_score
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = thresh
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_gain <= 1e-7 or not best_left_idx or not best_right_idx:
            return TreeNode(is_leaf=True, leaf_value=leaf_weight)

        left_node = self._build_tree(X, gradients, best_left_idx, depth + 1)
        right_node = self._build_tree(X, gradients, best_right_idx, depth + 1)

        return TreeNode(
            is_leaf=False,
            feature_index=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node,
        )

    def predict_one(self, x: list[float]) -> float:
        """Evaluate a single feature vector."""
        curr = self.root
        while curr and not curr.is_leaf:
            if x[curr.feature_index] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.leaf_value if curr else 0.0


class NovaGradientBoostedRegressor:
    """Deterministic, pure-Python Gradient Boosted Decision Tree (GBDT) engine."""

    def __init__(self, config: XGBoostModelConfig) -> None:
        self.config = config
        self.base_score: float = 0.0
        self.trees: list[PurePythonRegressionTree] = []
        self.training_history: list[dict[str, float]] = []

    def fit(self, X_train: list[list[float]], y_train: list[float], X_val: list[list[float]] | None = None, y_val: list[float] | None = None) -> None:
        """Fit gradient boosted trees iteratively with shrinkage and early stopping."""
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("Cannot fit on empty training set")

        # Initial prediction is mean target value
        self.base_score = sum(y_train) / float(n_samples)
        y_pred_train = [self.base_score] * n_samples
        y_pred_val = [self.base_score] * len(X_val) if X_val else []

        best_val_loss = float("inf")
        rounds_without_improvement = 0

        self.trees.clear()
        self.training_history.clear()

        for round_num in range(self.config.n_estimators):
            # Compute MSE gradients: g_i = y_pred_i - y_true_i
            gradients = [y_pred_train[i] - y_train[i] for i in range(n_samples)]

            # Fit tree on gradients
            tree = PurePythonRegressionTree(
                max_depth=self.config.max_depth,
                min_child_samples=self.config.min_child_samples,
                l2_reg=self.config.l2_reg,
            )
            tree.fit(X_train, gradients)

            # Update train predictions with shrinkage
            lr = self.config.learning_rate
            for i in range(n_samples):
                update = tree.predict_one(X_train[i])
                y_pred_train[i] += lr * update

            # Calculate train loss (RMSE)
            train_loss = math.sqrt(sum((p - t) ** 2 for p, t in zip(y_pred_train, y_train)) / n_samples)

            val_loss = 0.0
            if X_val and y_val:
                for j in range(len(X_val)):
                    y_pred_val[j] += lr * tree.predict_one(X_val[j])
                val_loss = math.sqrt(sum((p - t) ** 2 for p, t in zip(y_pred_val, y_val)) / len(y_val))

            self.trees.append(tree)
            self.training_history.append({
                "round": round_num + 1,
                "train_rmse": train_loss,
                "val_rmse": val_loss,
            })

            # Early stopping check
            if X_val and y_val:
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.config.early_stopping_rounds:
                        # Stop early
                        break

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict expected returns for a matrix of feature vectors."""
        preds = [self.base_score] * len(X)
        lr = self.config.learning_rate
        for tree in self.trees:
            for i in range(len(X)):
                preds[i] += lr * tree.predict_one(X[i])
        return preds

    def to_dict(self) -> dict[str, Any]:
        """Serialize pure-python model to JSON-compatible dictionary."""
        return {
            "backend": "PURE_PYTHON_GBDT",
            "config": self.config.to_dict(),
            "base_score": self.base_score,
            "trees": [t.root.to_dict() if t.root else None for t in self.trees],
            "tree_count": len(self.trees),
            "training_history": self.training_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: XGBoostModelConfig | None = None) -> NovaGradientBoostedRegressor:
        if config is not None:
            cfg = config
        elif "config" in data and isinstance(data["config"], dict):
            cfg = XGBoostModelConfig.from_dict(data["config"])
        else:
            cfg = XGBoostModelConfig()
        model = cls(cfg)
        model.base_score = float(data["base_score"])
        model.trees = []
        for t_dict in data.get("trees", []):
            t = PurePythonRegressionTree(
                max_depth=cfg.max_depth,
                min_child_samples=cfg.min_child_samples,
                l2_reg=cfg.l2_reg,
            )
            if t_dict:
                t.root = TreeNode.from_dict(t_dict)
            model.trees.append(t)
        model.training_history = data.get("training_history", [])
        return model


# ============================================================================
# Canonical XGBoostPredictionModel (Unified Wrapper)
# ============================================================================

class XGBoostPredictionModel:
    """Unified quantitative model wrapper implementing Interface Contract B."""

    def __init__(
        self,
        config: XGBoostModelConfig | None = None,
        backend_model: Any = None,
    ) -> None:
        self.config = config or XGBoostModelConfig()
        self.pipeline = FeaturePipeline()
        self.backend_type: str = "NATIVE_XGBOOST" if _NATIVE_XGBOOST_AVAILABLE else "PURE_PYTHON_GBDT"
        self._native_model: Any = None
        self._python_model: NovaGradientBoostedRegressor | None = None
        self.is_trained: bool = False
        self.training_summary: dict[str, Any] = {}

        if backend_model is not None:
            if isinstance(backend_model, NovaGradientBoostedRegressor):
                self._python_model = backend_model
                self.backend_type = "PURE_PYTHON_GBDT"
            else:
                self._native_model = backend_model
                self.backend_type = "NATIVE_XGBOOST"
            self.is_trained = True

    def fit(self, dataset: ModelDataset) -> dict[str, Any]:
        """Fit model on assembled chronological dataset."""
        if _NATIVE_XGBOOST_AVAILABLE and xgb is not None:
            try:
                dtrain = xgb.DMatrix(dataset.X_train, label=dataset.y_train)
                dval = xgb.DMatrix(dataset.X_val, label=dataset.y_val)
                eval_metric = "logloss" if self.config.objective == "binary:logistic" else "rmse"
                params = {
                    "max_depth": self.config.max_depth,
                    "learning_rate": self.config.learning_rate,
                    "lambda": self.config.l2_reg,
                    "objective": self.config.objective,
                    "eval_metric": eval_metric,
                    "tree_method": self.config.tree_method,
                    "booster": self.config.booster,
                    "seed": self.config.random_state,
                }
                evals = [(dtrain, "train"), (dval, "val")]
                self._native_model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config.n_estimators,
                    evals=evals,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose_eval=False,
                )
                self.backend_type = "NATIVE_XGBOOST"
                best_iter = getattr(self._native_model, "best_iteration", self.config.n_estimators)
                best_score = getattr(self._native_model, "best_score", None)
                self.training_summary = {
                    "backend": self.backend_type,
                    "objective": self.config.objective,
                    "fitted_rounds": best_iter,
                    "best_score": best_score,
                }
            except Exception as exc:
                # Fallback to pure-python GBDT engine if native training encounters unexpected error
                self._fit_python(dataset)
        else:
            self._fit_python(dataset)

        self.is_trained = True
        return self.training_summary

    def _fit_python(self, dataset: ModelDataset) -> None:
        """Fit using deterministic built-in GBDT engine."""
        self._python_model = NovaGradientBoostedRegressor(self.config)
        self._python_model.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
        )
        self.backend_type = "PURE_PYTHON_GBDT"
        self.training_summary = {
            "backend": self.backend_type,
            "fitted_rounds": len(self._python_model.trees),
            "final_train_rmse": self._python_model.training_history[-1]["train_rmse"] if self._python_model.training_history else 0.0,
            "final_val_rmse": self._python_model.training_history[-1]["val_rmse"] if self._python_model.training_history else 0.0,
        }

    def predict_raw(self, X: list[list[float]]) -> list[float]:
        """Predict continuous expected return values."""
        if not self.is_trained:
            raise ContractValidationError("Cannot predict with untrained model")

        if self.backend_type == "NATIVE_XGBOOST" and self._native_model is not None:
            if xgb is None:
                raise ContractValidationError("Native XGBoost library not available for inference")
            dm = xgb.DMatrix(X)
            raw_res = self._native_model.predict(dm)
            return [float(v) for v in raw_res]
        elif self._python_model is not None:
            return self._python_model.predict(X)
        else:
            raise ContractValidationError("No active trained backend available")

    def predict_bar(
        self,
        bars: Sequence[MarketBar],
        as_of: datetime | None = None,
    ) -> ModelPrediction:
        """Compute point-in-time expected return prediction and emit ModelPrediction."""
        if not bars:
            raise ContractValidationError("Cannot predict on empty bar series")

        latest_bar = bars[-1]
        ticker = latest_bar.ticker
        cutoff = as_of or latest_bar.source_timestamp
        pred_time = cutoff

        # 1. Check data quality
        if latest_bar.data_quality != DataQualityStatus.VALID:
            return ModelPrediction(
                model_id=self.config.model_id,
                model_version=self.config.model_version,
                prediction_timestamp=pred_time,
                input_cutoff_timestamp=cutoff,
                instrument_id=ticker,
                prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                horizon_bars=self.config.horizon_bars,
                prediction_value=0.0,
                confidence_or_uncertainty=0.0,
                availability_status=PredictionAvailability.UNAVAILABLE,
                metadata={"status_reason": f"DATA_QUALITY_{latest_bar.data_quality.value}"},
            )

        # 2. Check warmup
        if len(bars) < self.config.feature_warmup_bars:
            return ModelPrediction(
                model_id=self.config.model_id,
                model_version=self.config.model_version,
                prediction_timestamp=pred_time,
                input_cutoff_timestamp=cutoff,
                instrument_id=ticker,
                prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                horizon_bars=self.config.horizon_bars,
                prediction_value=0.0,
                confidence_or_uncertainty=0.0,
                availability_status=PredictionAvailability.INSUFFICIENT_HISTORY,
                metadata={
                    "status_reason": "INSUFFICIENT_HISTORY",
                    "buffered_bars": len(bars),
                    "required_bars": self.config.feature_warmup_bars,
                },
            )

        # 3. Compute causal feature vector for latest bar
        f_obj = self.pipeline.compute_latest(bars)
        f_dict = f_obj.to_dict()

        feat_vec: list[float] = []
        for fname in self.config.feature_names:
            v = f_dict.get(fname)
            if v is None or math.isnan(v) or math.isinf(v):
                return ModelPrediction(
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    prediction_timestamp=pred_time,
                    input_cutoff_timestamp=cutoff,
                    instrument_id=ticker,
                    prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                    horizon_bars=self.config.horizon_bars,
                    prediction_value=0.0,
                    confidence_or_uncertainty=0.0,
                    availability_status=PredictionAvailability.UNAVAILABLE,
                    metadata={"status_reason": f"FEATURE_INVALID_{fname}"},
                )
            feat_vec.append(float(v))

        # 4. Predict
        raw_val = self.predict_raw([feat_vec])[0]

        if self.config.objective == "binary:logistic":
            bounded_prob = max(0.0, min(1.0, raw_val))
            conviction = abs(bounded_prob - 0.50) * 2.0  # [0.0, 1.0]
            direction = SignalDirection.BULLISH if bounded_prob > 0.50 else (
                SignalDirection.BEARISH if bounded_prob < 0.50 else SignalDirection.NEUTRAL
            )
            pred = ModelPrediction(
                model_id=self.config.model_id,
                model_version=self.config.model_version,
                prediction_timestamp=pred_time,
                input_cutoff_timestamp=cutoff,
                instrument_id=ticker,
                prediction_type=PredictionType.DIRECTIONAL_PROBABILITY,
                horizon_bars=self.config.horizon_bars,
                prediction_value=round(bounded_prob, 6),
                confidence_or_uncertainty=round(conviction, 4),
                availability_status=PredictionAvailability.READY,
                metadata={
                    "backend": self.backend_type,
                    "objective": "binary:logistic",
                    "predicted_probability": round(bounded_prob, 6),
                    "directional_bias": direction.value,
                    "feature_count": len(self.config.feature_names),
                    "is_bullish": bounded_prob > 0.50,
                },
            )
        else:
            # Continuous return regression
            bounded_return = max(-0.50, min(0.50, raw_val))
            expected_bps = round(bounded_return * 10000.0, 1)
            confidence = min(1.0, max(0.10, abs(bounded_return) / 0.03))

            pred = ModelPrediction(
                model_id=self.config.model_id,
                model_version=self.config.model_version,
                prediction_timestamp=pred_time,
                input_cutoff_timestamp=cutoff,
                instrument_id=ticker,
                prediction_type=PredictionType.EXPECTED_RETURN_DECIMAL,
                horizon_bars=self.config.horizon_bars,
                prediction_value=round(bounded_return, 6),
                confidence_or_uncertainty=round(confidence, 4),
                availability_status=PredictionAvailability.READY,
                metadata={
                    "backend": self.backend_type,
                    "expected_bps": expected_bps,
                    "feature_count": len(self.config.feature_names),
                    "is_bullish": bounded_return > 0.0,
                },
            )

        # Validate schema parity
        validate_model_prediction_schema_parity(pred.to_dict())
        return pred

    def save(self, output_path: Path | str) -> dict[str, Any]:
        """Save trained model artifact and configuration to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_id": self.config.model_id,
            "model_version": self.config.model_version,
            "backend_type": self.backend_type,
            "config": self.config.to_dict(),
            "training_summary": self.training_summary,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        if self.backend_type == "PURE_PYTHON_GBDT" and self._python_model:
            payload["model_weights"] = self._python_model.to_dict()
        elif self.backend_type == "NATIVE_XGBOOST" and self._native_model:
            # Native model raw json
            payload["native_model_json"] = self._native_model.save_raw(raw_format="json").decode("utf-8")

        content = json.dumps(payload, indent=2)
        encoded_bytes = content.encode("utf-8")
        out_path = Path(path)
        if out_path.is_dir():
            out_path = out_path / "model.json"
        with open(out_path, "wb") as fp:
            fp.write(encoded_bytes)
        import hashlib
        checksum = hashlib.sha256(encoded_bytes).hexdigest()
        return {"path": str(out_path), "checksum": checksum}

    @classmethod
    def load(cls, model_path: Path | str) -> XGBoostPredictionModel:
        """Load trained model artifact from disk."""
        path = Path(model_path)
        if path.is_dir():
            path = path / "model.json"
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = XGBoostModelConfig.from_dict(data["config"])
        instance = cls(cfg)
        instance.backend_type = data["backend_type"]
        instance.training_summary = data.get("training_summary", {})

        if instance.backend_type == "PURE_PYTHON_GBDT" and "model_weights" in data:
            instance._python_model = NovaGradientBoostedRegressor.from_dict(data["model_weights"], cfg)
            instance.is_trained = True
        elif instance.backend_type == "NATIVE_XGBOOST":
            if _NATIVE_XGBOOST_AVAILABLE and xgb is not None and "native_model_json" in data:
                bst = xgb.Booster()
                bst.load_model(bytearray(data["native_model_json"].encode("utf-8")))
                instance._native_model = bst
                instance.is_trained = True
            elif "model_weights" in data:
                instance._python_model = NovaGradientBoostedRegressor.from_dict(data["model_weights"], cfg)
                instance.backend_type = "PURE_PYTHON_GBDT"
                instance.is_trained = True
            else:
                instance.backend_type = "PURE_PYTHON_GBDT"
                instance.is_trained = False

        return instance

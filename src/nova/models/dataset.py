"""NOVA Quantitative Machine Learning — Feature & Target Assembly (Person 2).

Assembles causal features and forward return targets without lookahead leakage:
1. Forward return target: y_{t, H} = ln(Close_{t+H} / Close_t) within-instrument only.
2. Unobserved target tail exclusion: last H bars of each stock dropped.
3. Feature warmup exclusion: first W bars dropped where indicators warm up.
4. Unified chronological train / val / test partitioning across all instruments.
5. Boundary Purging & Embargo: purges H bars at split transitions to prevent target overlap leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from nova.features.pipeline import BarFeatures, FeaturePipeline, FeaturePipelineConfig
from nova.models.config import XGBoostModelConfig
from nova.quant.contracts import ContractValidationError, MarketBar


@dataclass(frozen=True, slots=True)
class ModelDataset:
    """Immutable assembled dataset with chronological train, validation, and test splits."""
    feature_names: tuple[str, ...]
    X_train: list[list[float]]
    y_train: list[float]
    X_val: list[list[float]]
    y_val: list[float]
    X_test: list[list[float]]
    y_test: list[float]
    train_dates: tuple[str, str]  # (min_date, max_date)
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    total_raw_observations: int
    total_labeled_observations: int
    purged_embargo_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def train_count(self) -> int:
        return len(self.y_train)

    @property
    def val_count(self) -> int:
        return len(self.y_val)

    @property
    def test_count(self) -> int:
        return len(self.y_test)

    def to_classification(self) -> ModelDataset:
        """Return a copy of the dataset with binary direction targets: 1.0 if return > 0.0 else 0.0."""
        return ModelDataset(
            feature_names=self.feature_names,
            X_train=self.X_train,
            y_train=[1.0 if y > 0.0 else 0.0 for y in self.y_train],
            X_val=self.X_val,
            y_val=[1.0 if y > 0.0 else 0.0 for y in self.y_val],
            X_test=self.X_test,
            y_test=[1.0 if y > 0.0 else 0.0 for y in self.y_test],
            train_dates=self.train_dates,
            val_dates=self.val_dates,
            test_dates=self.test_dates,
            total_raw_observations=self.total_raw_observations,
            total_labeled_observations=self.total_labeled_observations,
            purged_embargo_count=self.purged_embargo_count,
            metadata={**self.metadata, "target_type": "binary_direction"},
        )



class DatasetAssembler:
    """Assembles causal features and forward targets from market bar histories."""

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        self.config = config or XGBoostModelConfig()
        self.pipeline = FeaturePipeline()

    def assemble_universe(
        self,
        universe_bars: Mapping[str, Sequence[MarketBar]],
    ) -> ModelDataset:
        """Process universe bars into chronological train, validation, and test sets with embargo."""
        h = self.config.horizon_bars
        warmup = self.config.feature_warmup_bars
        selected_features = list(self.config.feature_names)

        # 1. Process each instrument independently
        # Record format: (market_date, ticker, feature_vector, target)
        all_records: list[tuple[str, str, list[float], float]] = []
        total_raw = sum(len(bars) for bars in universe_bars.values())

        for ticker, raw_bars in sorted(universe_bars.items()):
            # Sort chronologically
            bars = sorted(list(raw_bars), key=lambda b: b.source_timestamp)
            n_bars = len(bars)
            if n_bars <= warmup + h:
                continue

            # Compute features chronologically using pipeline
            computed_features = self.pipeline.compute_history(bars)

            # Construct targets and extract feature vectors
            # An eligible observation index i satisfies: warmup <= i < n_bars - h
            for i in range(warmup, n_bars - h):
                curr_bar = bars[i]
                target_bar = bars[i + h]

                # Forward log-return target: ln(Close_{t+h} / Close_t)
                c_curr = curr_bar.close
                c_target = target_bar.close
                if c_curr <= 0.0 or c_target <= 0.0:
                    continue

                target_return = math.log(c_target / c_curr)

                # Extract features for bar i
                f_obj = computed_features[i]
                f_dict = f_obj.to_dict()

                # Extract ordered feature values, verifying finiteness
                feat_vec: list[float] = []
                is_valid = True
                for fname in selected_features:
                    val = f_dict.get(fname)
                    if val is None or math.isnan(val) or math.isinf(val):
                        is_valid = False
                        break
                    feat_vec.append(float(val))

                if is_valid:
                    date_str = curr_bar.source_timestamp.date().isoformat()
                    all_records.append((date_str, ticker, feat_vec, target_return))

        if not all_records:
            raise ContractValidationError("No valid feature/target records assembled from universe")

        # 2. Establish Global Chronological Timeline
        unique_dates = sorted(list({rec[0] for rec in all_records}))
        n_dates = len(unique_dates)
        if n_dates < 10:
            raise ContractValidationError(f"Insufficient unique dates for chronological partitioning: {n_dates}")

        # Split timeline chronologically
        n_train_dates = int(n_dates * self.config.train_ratio)
        n_val_dates = int(n_dates * self.config.val_ratio)
        n_test_dates = n_dates - n_train_dates - n_val_dates

        train_date_set = set(unique_dates[:n_train_dates])
        val_date_set = set(unique_dates[n_train_dates : n_train_dates + n_val_dates])
        test_date_set = set(unique_dates[n_train_dates + n_val_dates :])

        # 3. Purging and Embargo around split boundaries
        # Identify the boundary dates to purge (the last H dates of train, and last H dates of val)
        # Because a target at date t uses information up to date t+H, the last H dates of train
        # would overlap with the start of val.
        embargo_train_dates = set(unique_dates[n_train_dates - h : n_train_dates]) if n_train_dates >= h else set()
        embargo_val_dates = set(unique_dates[n_train_dates + n_val_dates - h : n_train_dates + n_val_dates]) if n_val_dates >= h else set()

        X_train: list[list[float]] = []
        y_train: list[float] = []
        X_val: list[list[float]] = []
        y_val: list[float] = []
        X_test: list[list[float]] = []
        y_test: list[float] = []

        purged_count = 0
        actual_train_dates: list[str] = []
        actual_val_dates: list[str] = []
        actual_test_dates: list[str] = []

        for d_str, ticker, f_vec, target in all_records:
            if d_str in train_date_set:
                if d_str in embargo_train_dates:
                    purged_count += 1
                else:
                    X_train.append(f_vec)
                    y_train.append(target)
                    actual_train_dates.append(d_str)
            elif d_str in val_date_set:
                if d_str in embargo_val_dates:
                    purged_count += 1
                else:
                    X_val.append(f_vec)
                    y_val.append(target)
                    actual_val_dates.append(d_str)
            elif d_str in test_date_set:
                X_test.append(f_vec)
                y_test.append(target)
                actual_test_dates.append(d_str)

        train_range = (min(actual_train_dates), max(actual_train_dates)) if actual_train_dates else ("N/A", "N/A")
        val_range = (min(actual_val_dates), max(actual_val_dates)) if actual_val_dates else ("N/A", "N/A")
        test_range = (min(actual_test_dates), max(actual_test_dates)) if actual_test_dates else ("N/A", "N/A")

        return ModelDataset(
            feature_names=tuple(selected_features),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_dates=train_range,
            val_dates=val_range,
            test_dates=test_range,
            total_raw_observations=total_raw,
            total_labeled_observations=len(all_records),
            purged_embargo_count=purged_count,
            metadata={
                "horizon_bars": h,
                "warmup_bars": warmup,
                "total_unique_dates": n_dates,
                "train_dates_count": len(train_date_set) - len(embargo_train_dates),
                "val_dates_count": len(val_date_set) - len(embargo_val_dates),
                "test_dates_count": len(test_date_set),
            },
        )


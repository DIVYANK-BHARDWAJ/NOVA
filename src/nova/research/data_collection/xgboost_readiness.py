"""NOVA Research Data Collection — XGBoost Readiness & Dataset Sufficiency Evaluator.

Assesses empirical supervised ML training feasibility without lookahead leakage:
1. Ingests chronological observations per instrument.
2. Calculates causal features (from nova.features) with warmup deduction.
3. Constructs forward-return labels within-instrument only:
   y_{t, h} = ln(Close_{t+h} / Close_t).
4. Enforces strict chronological train / validation / test partitioning (60% / 20% / 20%).
5. Applies boundary purging and embargo (H rows) to prevent forward-label overlap leakage.
6. Quantifies exact row counts and distinguishes mechanical training feasibility from statistical credibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from nova.features.pipeline import FeaturePipeline, FeaturePipelineConfig
from nova.quant.contracts import MarketBar, DataQualityStatus


@dataclass(frozen=True, slots=True)
class DatasetSplitCounts:
    """Exact sample counts across chronological evaluation partitions."""
    total_raw_bars: int
    warmup_deducted_bars: int
    valid_feature_rows: int
    unobserved_target_tail_bars: int
    valid_labeled_rows: int
    train_rows: int
    train_purged_embargo_rows: int
    train_usable_rows: int
    val_rows: int
    val_purged_embargo_rows: int
    val_usable_rows: int
    test_usable_rows: int
    total_model_usable_rows: int
    is_mechanically_trainable: bool
    is_statistically_credible: bool
    justification: str


class XGBoostReadinessEvaluator:
    """Evaluates whether an OHLCV dataset is sufficient for supervised ML modeling."""

    def __init__(
        self,
        feature_warmup_bars: int = 28,
        horizon_bars: int = 1,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
    ) -> None:
        self.warmup_bars = feature_warmup_bars
        self.horizon = horizon_bars
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def evaluate_instrument(self, ticker: str, bars: Sequence[MarketBar]) -> DatasetSplitCounts:
        """Evaluate supervised dataset sufficiency for a single instrument."""
        n_raw = len(bars)

        # 1. Warmup deduction
        if n_raw <= self.warmup_bars:
            return DatasetSplitCounts(
                total_raw_bars=n_raw,
                warmup_deducted_bars=n_raw,
                valid_feature_rows=0,
                unobserved_target_tail_bars=0,
                valid_labeled_rows=0,
                train_rows=0,
                train_purged_embargo_rows=0,
                train_usable_rows=0,
                val_rows=0,
                val_purged_embargo_rows=0,
                val_usable_rows=0,
                test_usable_rows=0,
                total_model_usable_rows=0,
                is_mechanically_trainable=False,
                is_statistically_credible=False,
                justification=f"Total bars ({n_raw}) <= warmup required ({self.warmup_bars})",
            )

        n_features = n_raw - self.warmup_bars

        # 2. Forward target tail deduction: last H rows have unobserved future endpoints
        if n_features <= self.horizon:
            return DatasetSplitCounts(
                total_raw_bars=n_raw,
                warmup_deducted_bars=self.warmup_bars,
                valid_feature_rows=n_features,
                unobserved_target_tail_bars=n_features,
                valid_labeled_rows=0,
                train_rows=0,
                train_purged_embargo_rows=0,
                train_usable_rows=0,
                val_rows=0,
                val_purged_embargo_rows=0,
                val_usable_rows=0,
                test_usable_rows=0,
                total_model_usable_rows=0,
                is_mechanically_trainable=False,
                is_statistically_credible=False,
                justification=f"Feature rows ({n_features}) <= target horizon ({self.horizon})",
            )

        n_labeled = n_features - self.horizon

        # 3. Chronological partitions
        n_train = int(n_labeled * self.train_ratio)
        n_val = int(n_labeled * self.val_ratio)
        n_test = n_labeled - n_train - n_val

        # 4. Boundary Purging & Embargo (H rows between train & val, and val & test)
        embargo = self.horizon
        train_usable = max(0, n_train - embargo)
        val_usable = max(0, n_val - embargo)
        test_usable = max(0, n_test)
        total_usable = train_usable + val_usable + test_usable

        # Minimum thresholds for statistical credibility:
        # At least 250 usable training bars (1 full trading year) and 50 validation/test bars
        is_trainable = (train_usable >= 20) and (val_usable >= 5) and (test_usable >= 5)
        is_credible = (train_usable >= 250) and (val_usable >= 50) and (test_usable >= 50)

        if is_credible:
            justification = f"Robust statistical sample: {train_usable} train, {val_usable} val, {test_usable} test bars."
        elif is_trainable:
            justification = f"Technically trainable ({train_usable} train rows), but small sample risks acute overfitting."
        else:
            justification = f"Insufficient sample size ({train_usable} train rows after embargo)."

        return DatasetSplitCounts(
            total_raw_bars=n_raw,
            warmup_deducted_bars=self.warmup_bars,
            valid_feature_rows=n_features,
            unobserved_target_tail_bars=self.horizon,
            valid_labeled_rows=n_labeled,
            train_rows=n_train,
            train_purged_embargo_rows=embargo,
            train_usable_rows=train_usable,
            val_rows=n_val,
            val_purged_embargo_rows=embargo,
            val_usable_rows=val_usable,
            test_usable_rows=test_usable,
            total_model_usable_rows=total_usable,
            is_mechanically_trainable=is_trainable,
            is_statistically_credible=is_credible,
            justification=justification,
        )

    def evaluate_universe(self, universe_bars: Mapping[str, Sequence[MarketBar]]) -> dict[str, Any]:
        """Aggregate evaluation across all instruments in the universe."""
        per_ticker = {}
        agg_raw = 0
        agg_train_usable = 0
        agg_val_usable = 0
        agg_test_usable = 0
        agg_total_usable = 0

        for t, bars in universe_bars.items():
            res = self.evaluate_instrument(t, bars)
            per_ticker[t] = res
            agg_raw += res.total_raw_bars
            agg_train_usable += res.train_usable_rows
            agg_val_usable += res.val_usable_rows
            agg_test_usable += res.test_usable_rows
            agg_total_usable += res.total_model_usable_rows

        is_cross_sectional_credible = agg_train_usable >= 2000

        return {
            "horizon_bars": self.horizon,
            "warmup_bars": self.warmup_bars,
            "aggregate_raw_bars": agg_raw,
            "aggregate_usable_train_rows": agg_train_usable,
            "aggregate_usable_val_rows": agg_val_usable,
            "aggregate_usable_test_rows": agg_test_usable,
            "aggregate_total_usable_rows": agg_total_usable,
            "is_cross_sectional_trainable": agg_train_usable >= 100,
            "is_cross_sectional_statistically_credible": is_cross_sectional_credible,
            "per_ticker_counts": {t: {
                "raw_bars": r.total_raw_bars,
                "usable_train": r.train_usable_rows,
                "usable_val": r.val_usable_rows,
                "usable_test": r.test_usable_rows,
                "total_usable": r.total_model_usable_rows,
                "is_trainable": r.is_mechanically_trainable,
                "is_credible": r.is_statistically_credible,
                "justification": r.justification,
            } for t, r in per_ticker.items()},
        }


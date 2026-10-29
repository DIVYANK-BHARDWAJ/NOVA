"""NOVA Features — Unified Causal Feature Pipeline.

Orchestrates multi-family quantitative feature extraction across:
- Trend & Crossover
- Mean Reversion & Oscillators
- Volatility & Downside Risk
- Volume & Liquidity Proxies
- Benchmark-Relative Context

Temporal Integrity Invariants:
1. Pure point-in-time calculation: bar t uses strictly information <= t.
2. Appending future bars never mutates past feature values.
3. Batch and incremental calculations yield bit-for-bit identical results.
4. Warmup states are tracked explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence

from nova.features.context import (
    calculate_benchmark_relative_return,
    calculate_relative_momentum,
)
from nova.features.mean_reversion import (
    calculate_mean_deviation,
    calculate_rolling_mean,
    calculate_rolling_std,
    calculate_rolling_vwap,
    calculate_rolling_z_score,
    calculate_rsi_series,
)
from nova.features.moving_averages import (
    calculate_ema_series,
    compute_ema_crossover_features,
    validate_window,
)
from nova.features.trend import (
    calculate_adx_series,
    calculate_crossover_recency,
    calculate_kaufman_efficiency_ratio,
    calculate_momentum_series,
    calculate_trend_slope,
)
from nova.features.volatility import (
    calculate_atr_series,
    calculate_downside_deviation,
    calculate_opening_gap,
    calculate_parkinson_volatility,
    calculate_realized_volatility,
    calculate_true_range,
)
from nova.features.volume import (
    calculate_on_balance_volume,
    calculate_price_volume_divergence,
    calculate_relative_volume,
    calculate_rolling_mean_volume,
    calculate_volume_z_score,
)
from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    MarketBar,
    NumericalValidationError,
    TemporalValidationError,
    validate_finite,
    validate_timezone_aware,
)


@dataclass(frozen=True, slots=True)
class FeaturePipelineConfig:
    """Configurable parameters and window lengths for the quantitative feature pipeline."""
    fast_ema_window: int = 12
    slow_ema_window: int = 26
    momentum_window: int = 5
    trend_slope_window: int = 10
    ker_window: int = 10
    adx_window: int = 14
    rolling_mean_window: int = 20
    rsi_window: int = 14
    vwap_window: int = 20
    atr_window: int = 14
    realized_vol_window: int = 20
    parkinson_vol_window: int = 20
    downside_vol_window: int = 20
    volume_mean_window: int = 20
    volume_divergence_window: int = 10
    benchmark_window: int = 10
    strict_live_mode: bool = False
    config_version: str = "PROPOSED_V0"

    def __post_init__(self) -> None:
        windows = [
            ("fast_ema_window", self.fast_ema_window),
            ("slow_ema_window", self.slow_ema_window),
            ("momentum_window", self.momentum_window),
            ("trend_slope_window", self.trend_slope_window),
            ("ker_window", self.ker_window),
            ("adx_window", self.adx_window),
            ("rolling_mean_window", self.rolling_mean_window),
            ("rsi_window", self.rsi_window),
            ("vwap_window", self.vwap_window),
            ("atr_window", self.atr_window),
            ("realized_vol_window", self.realized_vol_window),
            ("parkinson_vol_window", self.parkinson_vol_window),
            ("downside_vol_window", self.downside_vol_window),
            ("volume_mean_window", self.volume_mean_window),
            ("volume_divergence_window", self.volume_divergence_window),
            ("benchmark_window", self.benchmark_window),
        ]
        for name, w in windows:
            validate_window(w, name)

        if self.fast_ema_window >= self.slow_ema_window:
            raise ContractValidationError(
                f"fast_ema_window ({self.fast_ema_window}) must be < slow_ema_window ({self.slow_ema_window})"
            )

    @property
    def maximum_warmup_bars(self) -> int:
        """Maximum warmup bars required across all enabled feature extractors."""
        return max(
            self.slow_ema_window,
            2 * self.adx_window,
            self.rolling_mean_window,
            self.realized_vol_window,
            self.parkinson_vol_window,
            self.downside_vol_window,
            self.volume_mean_window,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeaturePipelineConfig:
        """Instantiate FeaturePipelineConfig from a dictionary with strict validation."""
        fields = {
            "fast_ema_window",
            "slow_ema_window",
            "momentum_window",
            "trend_slope_window",
            "ker_window",
            "adx_window",
            "rolling_mean_window",
            "rsi_window",
            "vwap_window",
            "atr_window",
            "realized_vol_window",
            "parkinson_vol_window",
            "downside_vol_window",
            "volume_mean_window",
            "volume_divergence_window",
            "benchmark_window",
            "strict_live_mode",
            "config_version",
        }
        kwargs = {k: v for k, v in data.items() if k in fields}
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, toml_path: str | Path, section: str = "features") -> FeaturePipelineConfig:
        """Load feature configuration from centralized TOML configuration file."""
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        if section not in cfg:
            raise ContractValidationError(f"Missing section '[{section}]' in TOML configuration {path}")
        return cls.from_dict(cfg[section])


@dataclass(frozen=True, slots=True)
class BarFeatures:
    """Unified point-in-time quantitative feature vector for a single candle bar."""
    ticker: str
    timestamp: datetime
    input_cutoff_timestamp: datetime
    close: float

    # Trend & Crossover
    fast_ema: float
    slow_ema: float
    ema_spread: float
    ema_normalized_spread: float
    crossover_event: int
    crossover_recency: int | None
    momentum: float | None
    trend_slope: float | None
    kaufman_efficiency_ratio: float | None
    adx: float | None
    plus_di: float | None
    minus_di: float | None

    # Mean Reversion
    rolling_mean: float | None
    rolling_std: float | None
    rolling_z_score: float | None
    mean_deviation: float | None
    rsi: float | None
    vwap: float | None
    vwap_deviation: float | None

    # Volatility
    true_range: float
    atr: float | None
    normalized_atr: float | None
    realized_volatility: float | None
    parkinson_volatility: float | None
    downside_deviation: float | None
    opening_gap: float | None

    # Volume & Liquidity
    volume_mean: float | None
    relative_volume: float | None
    volume_z_score: float | None
    on_balance_volume: float
    price_volume_divergence: float | None

    # Benchmark Context
    benchmark_relative_return: float | None
    benchmark_relative_momentum: float | None

    # State Metadata
    is_warmed_up: bool
    feature_config_version: str = "PROPOSED_V0"

    def to_dict(self) -> dict[str, Any]:
        """Convert features to a dictionary of numeric / string fields."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "input_cutoff_timestamp": self.input_cutoff_timestamp.isoformat(),
            "close": self.close,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "ema_spread": self.ema_spread,
            "ema_normalized_spread": self.ema_normalized_spread,
            "crossover_event": self.crossover_event,
            "crossover_recency": self.crossover_recency,
            "momentum": self.momentum,
            "trend_slope": self.trend_slope,
            "kaufman_efficiency_ratio": self.kaufman_efficiency_ratio,
            "adx": self.adx,
            "plus_di": self.plus_di,
            "minus_di": self.minus_di,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "rolling_z_score": self.rolling_z_score,
            "mean_deviation": self.mean_deviation,
            "rsi": self.rsi,
            "vwap": self.vwap,
            "vwap_deviation": self.vwap_deviation,
            "true_range": self.true_range,
            "atr": self.atr,
            "normalized_atr": self.normalized_atr,
            "realized_volatility": self.realized_volatility,
            "parkinson_volatility": self.parkinson_volatility,
            "downside_deviation": self.downside_deviation,
            "opening_gap": self.opening_gap,
            "volume_mean": self.volume_mean,
            "relative_volume": self.relative_volume,
            "volume_z_score": self.volume_z_score,
            "on_balance_volume": self.on_balance_volume,
            "price_volume_divergence": self.price_volume_divergence,
            "benchmark_relative_return": self.benchmark_relative_return,
            "benchmark_relative_momentum": self.benchmark_relative_momentum,
            "is_warmed_up": self.is_warmed_up,
            "feature_config_version": self.feature_config_version,
        }


class FeaturePipeline:
    """Orchestrates deterministic, point-in-time feature extraction for NOVA models."""

    def __init__(self, config: FeaturePipelineConfig | None = None) -> None:
        self.config = config or FeaturePipelineConfig()

    def _validate_bars(self, bars: Sequence[MarketBar]) -> None:
        """Strictly validate chronological order and consistency of market bar sequence."""
        if not bars:
            raise ContractValidationError("Cannot compute features on empty bar sequence")

        ticker = bars[0].ticker
        for i in range(len(bars)):
            b = bars[i]
            if b.ticker != ticker:
                raise ContractValidationError(
                    f"Inconsistent ticker in bar sequence at index {i}: expected '{ticker}', got '{b.ticker}'"
                )
            validate_timezone_aware(b.source_timestamp, f"bars[{i}].source_timestamp")

            if self.config.strict_live_mode:
                if b.data_quality == DataQualityStatus.SYNTHETIC_TEST_FIXTURE:
                    raise ContractValidationError(
                        f"CRITICAL: Synthetic test fixture detected in live processing at bar {i}! "
                        "Per Non-Negotiable Override, synthetic data is strictly forbidden in live mode."
                    )
                if b.data_quality == DataQualityStatus.UNVERIFIED_PROVENANCE:
                    raise ContractValidationError(
                        f"CRITICAL: Unverified stream provenance at bar {i}! "
                        "Per Non-Negotiable Override, only genuine authorized competition feeds are permitted in live mode."
                    )
                if b.data_quality == DataQualityStatus.INVALID:
                    raise ContractValidationError(
                        f"CRITICAL: Invalid data quality at bar {i}! "
                        "Corrupt or invalid competition bars cannot enter feature pipeline."
                    )

            if i > 0:
                prev_ts = bars[i - 1].source_timestamp
                curr_ts = b.source_timestamp
                if curr_ts <= prev_ts:
                    raise TemporalValidationError(
                        f"MarketBar sequence must be strictly chronological: bar {i} ({curr_ts}) <= bar {i-1} ({prev_ts})"
                    )

    def compute_history(
        self,
        bars: Sequence[MarketBar],
        benchmark_bars: Sequence[MarketBar] | None = None,
    ) -> list[BarFeatures]:
        """Compute full chronological feature vectors for a sequence of bars.

        Strictly causal: BarFeatures at index i depends strictly on bars 0..i.
        """
        self._validate_bars(bars)
        n = len(bars)

        # Align benchmark if provided
        bm_closes: list[float] | None = None
        if benchmark_bars is not None:
            self._validate_bars(benchmark_bars)
            if len(benchmark_bars) != n:
                raise ContractValidationError(
                    f"Benchmark bar count ({len(benchmark_bars)}) must match asset bar count ({n})"
                )
            for i in range(n):
                if bars[i].source_timestamp != benchmark_bars[i].source_timestamp:
                    raise TemporalValidationError(
                        f"Benchmark timestamp mismatch at index {i}: asset={bars[i].source_timestamp} vs bm={benchmark_bars[i].source_timestamp}"
                    )
            bm_closes = [b.close for b in benchmark_bars]

        opens = [b.open for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        timestamps = [b.source_timestamp for b in bars]

        # Extract features per family
        # Trend
        fast_emas = calculate_ema_series(closes, self.config.fast_ema_window)
        slow_emas = calculate_ema_series(closes, self.config.slow_ema_window)
        ema_features = compute_ema_crossover_features(
            closes,
            timestamps,
            fast_window=self.config.fast_ema_window,
            slow_window=self.config.slow_ema_window,
        )
        crossover_events = [f.crossover_event for f in ema_features]
        crossover_recencies = calculate_crossover_recency(crossover_events)
        momentums = calculate_momentum_series(closes, self.config.momentum_window)
        trend_slopes = calculate_trend_slope(closes, self.config.trend_slope_window)
        kers = calculate_kaufman_efficiency_ratio(closes, self.config.ker_window)
        plus_dis, minus_dis, adxs = calculate_adx_series(highs, lows, closes, self.config.adx_window)

        # Mean Reversion
        rolling_means = calculate_rolling_mean(closes, self.config.rolling_mean_window)
        rolling_stds = calculate_rolling_std(closes, self.config.rolling_mean_window, ddof=1)
        rolling_z_scores = calculate_rolling_z_score(closes, self.config.rolling_mean_window)
        mean_deviations = calculate_mean_deviation(closes, self.config.rolling_mean_window)
        rsis = calculate_rsi_series(closes, self.config.rsi_window)
        vwaps, vwap_devs = calculate_rolling_vwap(highs, lows, closes, volumes, self.config.vwap_window)

        # Volatility
        true_ranges = calculate_true_range(highs, lows, closes)
        atrs, natrs = calculate_atr_series(highs, lows, closes, self.config.atr_window)
        realized_vols = calculate_realized_volatility(closes, self.config.realized_vol_window)
        parkinson_vols = calculate_parkinson_volatility(highs, lows, self.config.parkinson_vol_window)
        downside_devs = calculate_downside_deviation(closes, self.config.downside_vol_window)
        opening_gaps = calculate_opening_gap(opens, closes)

        # Volume
        volume_means = calculate_rolling_mean_volume(volumes, self.config.volume_mean_window)
        rvols = calculate_relative_volume(volumes, self.config.volume_mean_window)
        volume_zs = calculate_volume_z_score(volumes, self.config.volume_mean_window)
        obvs = calculate_on_balance_volume(closes, volumes)
        pv_divs = calculate_price_volume_divergence(closes, volumes, self.config.volume_divergence_window)

        # Benchmark Context
        bm_returns = calculate_benchmark_relative_return(closes, bm_closes)
        bm_momentums = calculate_relative_momentum(closes, bm_closes, self.config.benchmark_window)

        max_warmup = self.config.maximum_warmup_bars

        features_list: list[BarFeatures] = []
        for i in range(n):
            b = bars[i]
            cutoff = b.available_timestamp if b.available_timestamp is not None else b.source_timestamp
            is_warmed_up = i >= max_warmup

            bf = BarFeatures(
                ticker=b.ticker,
                timestamp=b.source_timestamp,
                input_cutoff_timestamp=cutoff,
                close=b.close,
                fast_ema=fast_emas[i],
                slow_ema=slow_emas[i],
                ema_spread=ema_features[i].spread,
                ema_normalized_spread=ema_features[i].normalized_spread,
                crossover_event=crossover_events[i],
                crossover_recency=crossover_recencies[i],
                momentum=momentums[i],
                trend_slope=trend_slopes[i],
                kaufman_efficiency_ratio=kers[i],
                adx=adxs[i],
                plus_di=plus_dis[i],
                minus_di=minus_dis[i],
                rolling_mean=rolling_means[i],
                rolling_std=rolling_stds[i],
                rolling_z_score=rolling_z_scores[i],
                mean_deviation=mean_deviations[i],
                rsi=rsis[i],
                vwap=vwaps[i],
                vwap_deviation=vwap_devs[i],
                true_range=true_ranges[i],
                atr=atrs[i],
                normalized_atr=natrs[i],
                realized_volatility=realized_vols[i],
                parkinson_volatility=parkinson_vols[i],
                downside_deviation=downside_devs[i],
                opening_gap=opening_gaps[i],
                volume_mean=volume_means[i],
                relative_volume=rvols[i],
                volume_z_score=volume_zs[i],
                on_balance_volume=obvs[i],
                price_volume_divergence=pv_divs[i],
                benchmark_relative_return=bm_returns[i],
                benchmark_relative_momentum=bm_momentums[i],
                is_warmed_up=is_warmed_up,
                feature_config_version=self.config.config_version,
            )
            features_list.append(bf)

        return features_list

    def compute_latest(
        self,
        bars: Sequence[MarketBar],
        benchmark_bars: Sequence[MarketBar] | None = None,
    ) -> BarFeatures:
        """Compute and return the single latest point-in-time feature vector."""
        history = self.compute_history(bars, benchmark_bars=benchmark_bars)
        return history[-1]

